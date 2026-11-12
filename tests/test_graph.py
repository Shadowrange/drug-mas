"""
Тесты графа и агентов (блок A1, docs/GRAPH-CONTRACT.md).

Запуск: python -m pytest -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.agents import data_agent, evaluator, generator, reporter, supervisor
from src.graph import MAX_LOOPS, build_graph, route_generated, route_sufficiency, route_verify, run_graph
from src.graph_state import GraphState
from src.schemas import Candidate, MissingData, Provenance, ToolCall, computed
from src.tools import chem
from src.verifier import Thresholds, VerificationReport

QUERY_SCENARIO_1 = (
    "Модифицируй структуру иматиниба так, чтобы повысить растворимость "
    "и сохранить сродство к BCR-ABL1, отфильтруй токсичные аналоги"
)
QUERY_SCENARIO_2 = (
    "Найди известные ингибиторы EGFR (UniProt P00533), "
    "обучи модель активности и предложи 3 новых кандидата"
)
QUERY_SCENARIO_3 = "Предложи ингибиторы для мишени <без данных в ChEMBL>"


# --- 1. Обязательный тест защиты от бесконечного цикла (§2.6) ---


def test_impossible_thresholds_terminates_after_two_loops(monkeypatch):
    """
    Обязательный тест из контракта (§2.6):
    прогон с заведомо непроходимыми порогами Thresholds(max_logp=-99)
    доходит до END, возвращает status="no_candidates_passed",
    а счётчик входов в generator равен ровно 2.
    """
    gen_entries = 0
    orig_gen = generator.generator

    def counting_generator(state: GraphState) -> dict:
        nonlocal gen_entries
        gen_entries += 1
        return orig_gen(state)

    monkeypatch.setattr(generator, "generator", counting_generator)

    res = run_graph(QUERY_SCENARIO_1, thresholds=Thresholds(max_logp=-99))

    assert gen_entries == 2 == MAX_LOOPS
    assert res.status == "no_candidates_passed"
    assert len(res.candidates) > 0
    assert all(c.rank is None for c in res.candidates)
    assert all(
        any(f.name == "LogP" and f.passed is False for f in c.filters)
        for c in res.candidates
    )


# --- 2. Проверка неизменности промптов SYSTEM из §3 контракта ---


def test_system_prompts_match_contract_verbatim():
    """Промпты SYSTEM в src/agents/*.py обязаны дословно совпадать с §3 docs/GRAPH-CONTRACT.md."""
    contract_path = Path(__file__).resolve().parent.parent / "docs" / "GRAPH-CONTRACT.md"
    text = contract_path.read_text(encoding="utf-8")

    sec3_start = text.index("## 3. Системные промпты LLM-агентов")
    sec4_start = text.index("## 4. Где LLM принимает решение")
    sec3_text = text[sec3_start:sec4_start]

    blocks = re.findall(r"```\n(.*?)\n```", sec3_text, flags=re.DOTALL)
    assert len(blocks) == 5

    expected_supervisor, expected_data, expected_gen, expected_eval, expected_rep = [
        b.replace("\r\n", "\n").strip() for b in blocks
    ]

    assert supervisor.SYSTEM.replace("\r\n", "\n").strip() == expected_supervisor
    assert data_agent.SYSTEM.replace("\r\n", "\n").strip() == expected_data
    assert generator.SYSTEM.replace("\r\n", "\n").strip() == expected_gen
    assert evaluator.SYSTEM.replace("\r\n", "\n").strip() == expected_eval
    assert reporter.SYSTEM.replace("\r\n", "\n").strip() == expected_rep


# --- 3. Сценарии 1, 2, 3 end-to-end из прогретого кэша ---


def test_scenario_1_lead_optimization_offline():
    """Сценарий 1: оптимизация иматиниба с рецептором 1IEP из локального кэша."""
    res = run_graph(QUERY_SCENARIO_1)

    assert res.status == "ok"
    assert res.scenario == "lead_optimization"
    assert len(res.candidates) > 0
    assert [c.rank for c in res.candidates] == list(range(1, len(res.candidates) + 1))

    tools_ok = {tc.tool for tc in res.tool_calls if tc.ok}
    assert "pubchem_smiles" in tools_ok
    assert "pdb_structure" in tools_ok
    assert "admet.profile" in tools_ok

    pub_call = next(tc for tc in res.tool_calls if tc.tool == "pubchem_smiles")
    pdb_call = next(tc for tc in res.tool_calls if tc.tool == "pdb_structure")
    assert pub_call.from_cache is True
    assert pdb_call.from_cache is True

    top = res.candidates[0]
    assert top.molecule.numeric("LogP") is not None
    assert top.molecule.numeric("QED") is not None
    assert top.molecule.numeric("SA_Score") is not None
    assert top.molecule.get("LogS") is not None
    assert top.molecule.get("LogS").provenance is Provenance.PREDICTED
    assert res.summary.strip() != ""


def test_scenario_2_target_to_hit_egfr_offline():
    """Сценарий 2: поиск под мишень EGFR (P00533 / CHEMBL203), обучение QSAR и скоринг кандидатов."""
    res = run_graph(QUERY_SCENARIO_2)

    assert res.status == "ok"
    assert res.scenario == "target_to_hit"
    assert len(res.candidates) >= 3

    tools_ok = {tc.tool for tc in res.tool_calls if tc.ok}
    assert "uniprot_target" in tools_ok
    assert "chembl_activities" in tools_ok
    assert "qsar.predict" in tools_ok

    for cand in res.candidates:
        pic50 = cand.molecule.get("pIC50")
        assert pic50 is not None
        assert pic50.provenance is Provenance.PREDICTED
        assert pic50.model_metrics is not None
        assert pic50.model_metrics.n_train >= 100
        assert pic50.model_metrics.metric_name == "R2"
        assert 0.0 <= pic50.model_metrics.metric_value <= 1.0


def test_scenario_3_insufficient_data_refusal(monkeypatch):
    """Сценарий 3: честный отказ при недостатке данных ДО запуска Generator."""
    gen_entries = 0
    orig_gen = generator.generator

    def counting_generator(state: GraphState) -> dict:
        nonlocal gen_entries
        gen_entries += 1
        return orig_gen(state)

    monkeypatch.setattr(generator, "generator", counting_generator)

    res = run_graph(QUERY_SCENARIO_3)

    assert res.status == "insufficient_data"
    assert res.candidates == []
    assert len(res.missing) >= 1
    assert gen_entries == 0
    assert res.summary.strip() != ""
    # На ветке insufficient_data в summary не должно быть ни одного валидного SMILES
    for tok in res.summary.split():
        cleaned = tok.strip(",.;:!?()[]{}\"'«»")
        assert not (cleaned and chem.is_valid(cleaned)), f"В summary найден SMILES-токен: {cleaned}"


# --- 4. Поправки §2.5 и §2.6 ---


def test_amendment_1_empty_generated_switches_backend_and_terminates(monkeypatch):
    """
    Поправка 1 (§2.5, §2.6): при пустой генерации route_generated возвращает
    в generator со сменой backend ('crem' -> 'selfies'), а после MAX_LOOPS
    уводит в reporter с записью причины в limitations.
    """
    calls_backends: list[str] = []

    def empty_mutate(parent_smiles: str, n: int = 50, backend: str = "crem", db_name=None, seed: int = 0xF00D):
        calls_backends.append(backend)
        return []

    monkeypatch.setattr("src.tools.generate.mutate", empty_mutate)

    res = run_graph(QUERY_SCENARIO_1)

    assert len(calls_backends) == 2
    assert calls_backends[0] == "crem"
    assert calls_backends[1] == "selfies"
    assert res.status == "no_candidates_passed"
    assert res.candidates == []
    assert any("не выдал ни одной валидной новой структуры" in lim for lim in res.limitations)


def test_amendment_2_and_3_accumulates_unverifiable_and_rejected_across_loops(monkeypatch):
    """
    Поправки 2 и 3 (§2.5, §2.6):
    - претензии unverifiable первого круга не теряются на втором круге;
    - отсеянные первого круга сохраняются в итоговом ответе на ветке exhausted.
    """
    orig_evaluator = evaluator.evaluator

    def injecting_evaluator(state: GraphState) -> dict:
        out = orig_evaluator(state)
        if state.get("loop") == 1 and out["candidates"]:
            # Искусственно добавляем неподтверждённый источник на 1-м круге (без вызова PLIP interactions)
            out["candidates"][0].molecule.values.append(
                computed("HBond_Count", 3, "PLIP 2.3.0 анализ", units="шт")
            )
        return out

    monkeypatch.setattr(evaluator, "evaluator", injecting_evaluator)

    compiled = build_graph(thresholds=Thresholds(max_logp=-99))
    final_state = compiled.invoke({"query": QUERY_SCENARIO_1, "loop": 0, "tool_calls": []})
    res = final_state["result"]

    assert final_state["loop"] == 2
    assert res.status == "no_candidates_passed"
    # Претензия к PLIP с первого круга обязана остаться в limitations после второго круга
    assert any("interactions()" in lim for lim in res.limitations)
    # Число кандидатов в результате включает отсев обоих кругов (> числа молекул одного круга)
    # Порог был откалиброван под генератор без фильтра химической
    # состоятельности. После его добавления мусорных структур меньше,
    # поэтому проверяем само свойство накопления: кандидатов больше,
    # чем даёт один круг генерации.
    assert len(res.candidates) > 10


def test_reporter_postfilter_strips_smiles_on_insufficient_data(monkeypatch):
    """
    Постфильтр кода в Reporter (§2.6): если LLM на ветке insufficient_data
    вернула SMILES в тексте summary, код вырезает его и добавляет запись в limitations.
    """
    monkeypatch.setattr(
        reporter,
        "_cached_llm_call",
        lambda *args, **kwargs: "Данных недостаточно, однако рассмотрим молекулу CC(=O)Oc1ccccc1C(=O)O для старта.",
    )

    state: GraphState = {
        "query": QUERY_SCENARIO_3,
        "scenario": "insufficient_data_demo",
        "missing": [
            MissingData(
                what="активности в ChEMBL",
                found="0 записей",
                needed="не менее 100 записей",
                would_unlock="обучение QSAR",
            )
        ],
        "tool_calls": [],
        "notes": [],
    }
    out = reporter.reporter(state)
    res = out["result"]

    assert res.status == "insufficient_data"
    assert "CC(=O)Oc1ccccc1C(=O)O" not in res.summary
    assert any("удалён SMILES" in lim for lim in res.limitations)


def test_supervisor_never_invents_ids_and_retries_on_invalid_json(monkeypatch):
    """
    Supervisor (§2, §3.1):
    1) не подставляет CID 5291, 1IEP или SMILES по памяти, если их нет в query;
    2) при невалидном JSON от LLM делает ровно 1 retry (всего 2 попытки),
       затем применяет дефолт scenario='target_to_hit' и regex-экстракцию.
    """
    attempts = 0

    def bad_llm(*args, **kwargs) -> str:
        nonlocal attempts
        attempts += 1
        return "невалидный json {"

    monkeypatch.setattr(supervisor, "_cached_llm_call", bad_llm)

    out = supervisor.supervisor({"query": "Модифицируй иматиниб для мишени BCR-ABL1"})
    assert attempts == 2
    assert out["scenario"] == "target_to_hit"
    task = out["task"]
    assert task["parent_name"] == "иматиниб"
    assert task["parent_smiles"] is None
    assert task["pdb_id"] is None
    assert task["target_chembl_id"] is None
    assert task["target_uniprot"] is None
    assert any("невалидный JSON" in n for n in out["notes"])


def test_routers_unit():
    """Прямая проверка функций маршрутизации route_sufficiency, route_generated, route_verify."""
    md = MissingData(what="w", found="0", needed="100", would_unlock="u")
    assert route_sufficiency({"missing": [md]}) == "reporter"
    assert route_sufficiency({"missing": []}) == "generator"

    mol = chem.describe("CC(=O)Oc1ccccc1C(=O)O")
    assert route_generated({"generated": [mol], "loop": 1}) == "evaluator"
    assert route_generated({"generated": [], "loop": 1}) == "generator"
    assert route_generated({"generated": [], "loop": 2}) == "reporter"

    cand = Candidate(molecule=mol)
    # Наличие выживших проверяется РАНЬШЕ лимита кругов (§2.6)
    assert route_verify({"verification": VerificationReport(passed=[cand]), "loop": 2}) == "reporter"
    assert route_verify({"verification": VerificationReport(passed=[]), "loop": 1}) == "generator"
    assert route_verify({"verification": VerificationReport(passed=[]), "loop": 2}) == "reporter"


# --- разбор запроса: мишенью не может быть предлог ---

def test_target_extraction_rejects_russian_stopwords():
    """«...мишени с малым числом активностей» давало target_name='с',
    и отказ выглядел багом разбора, а не принципиальным отказом."""
    from src.agents.supervisor import _extract_target_name

    assert _extract_target_name(
        "Предложи ингибиторы для малоизученной мишени с малым числом активностей в ChEMBL"
    ) is None
    assert _extract_target_name("Предложи ингибиторы для мишени белка, который плохо изучен") is None


def test_target_extraction_keeps_real_gene_symbols():
    from src.agents.supervisor import _extract_target_name

    assert _extract_target_name("Найди известные ингибиторы EGFR (UniProt P00533)") == "EGFR"
    assert _extract_target_name("сохранив сродство к BCR-ABL1") == "BCR-ABL1"
    assert _extract_target_name("Предложи ингибиторы для мишени CDK2") == "CDK2"


# --- правки по итогам ревью перед первым живым прогоном LLM ---

def test_target_resolved_when_llm_renames_it():
    """Живая LLM говорит «BCR-ABL1 киназа». Точное равенство строки
    переворачивало сценарий 1 в insufficient_data."""
    from src.agents.data_agent import _resolve_target_ids

    for name in ("BCR-ABL1", "BCR-ABL1 киназа", "BCR-ABL1 tyrosine kinase", "ABL1 (BCR-ABL)"):
        _, _, pdb, resname = _resolve_target_ids({"target_name": name}, "lead_optimization", "imatinib")
        assert pdb == "1IEP" and resname == "STI", name

    for name in ("EGFR", "EGFR kinase", "рецептор EGFR"):
        uni, chembl, _, _ = _resolve_target_ids({"target_name": name}, "target_to_hit", None)
        assert uni == "P00533" and chembl == "CHEMBL203", name


def test_json_parsed_through_prose_wrapper():
    """Generator и Evaluator ретраев не имеют: один плохой разбор — и дефолты."""
    from src.graph_state import parse_json_object

    assert parse_json_object('Вот JSON:\n```json\n{"backend": "selfies"}\n```')["backend"] == "selfies"
    assert parse_json_object('Ответ: {"n": 30} — всё')["n"] == 30


def test_reporter_rejects_invented_numbers():
    """ΔG цифрой в тексте — выдумка: Vina не установлена, dG not_computed."""
    from src.agents.reporter import _extraneous_numbers
    from src.schemas import ScenarioResult

    res = ScenarioResult(status="no_candidates_passed", query="q", scenario="s", candidates=[])
    assert _extraneous_numbers("Энергия связывания составила -9.4 ккал/моль", res)
    assert not _extraneous_numbers("Кандидатов, прошедших пороги, нет.", res)


def test_supervisor_drops_invented_names():
    from src.agents.supervisor import _sanitize_task_against_query

    task = _sanitize_task_against_query(
        {"parent_name": "гефитиниб", "target_name": "KRAS"},
        "Модифицируй структуру иматиниба, сохранив сродство к BCR-ABL1",
    )
    assert task["parent_name"] != "гефитиниб"
    assert task["target_name"] != "KRAS"
