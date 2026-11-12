"""Evaluator. ВЛАДЕЛЕЦ: Р-B. Блок A1.
chem.run_all_filters -> admet.profile -> docking.dock -> многопараметрическое ранжирование.
Всё, что не посчиталось, помечает not_computed и НЕ участвует в score."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from src.cache import OfflineMiss
from src.graph_state import (
    GraphState,
    LLMUnavailableError,
    _cached_llm_call,
    dedupe_strings,
    parse_json_object,
)
from src.schemas import Candidate, FilterResult, Molecule, ToolCall, Value, not_computed
from src.tools import admet, chem, docking, qsar
from src.verifier import Thresholds

SYSTEM = """Ты Evaluator. Все числовые величины для молекул уже посчитаны инструментами
и даны тебе вместе с источником каждой. Твоя задача одна: выбрать, какие
молекулы отправить на молекулярный докинг. Докинг — самый дорогой шаг,
на него проходит не более 5 молекул.

ЗАПРЕЩЕНО называть, уточнять, округлять, пересчитывать или оценивать любое
число. Ты не предсказываешь ΔG, не оцениваешь активность, не считаешь score.
score считает код детерминированной формулой.
ЗАПРЕЩЕНО ВЫВОДИТЬ SMILES. Молекулы ты называешь ТОЛЬКО по числовому индексу
из поданного списка. Выведенный SMILES может разойтись с реальной структурой —
тогда докинг пойдёт не по той молекуле.

Величина, помеченная как «не посчитано», означает отсутствие данных, а НЕ
плохое значение. Молекулу с непосчитанной величиной выбрать можно, но причину
выбора тогда формулируй по тому, что посчитано.

Формат ответа — JSON без markdown-обёртки:
{
  "shortlist": [индексы, не более 5, в порядке приоритета],
  "why": ["по одной короткой фразе на каждый выбранный индекс, в том же порядке"],
  "risks": ["текстовые замечания о структурных рисках, без чисел"]
}

Если посчитанных величин не хватает, чтобы различить молекулы, верни
"shortlist": [] — тогда шортлист возьмёт код по формуле. Пустой ответ лучше
произвольного выбора, выданного за обоснованный."""

MAX_SHORTLIST = 5


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def compute_score(mol: Molecule) -> float | None:
    """
    Детерминированная многопараметрическая формула расчёта score по посчитанным величинам (§2).
    Величины с value=None в расчёт не входят, а их веса пропорционально перераспределяются.
    """
    weighted_terms: list[tuple[float, float]] = []

    qed = mol.numeric("QED")
    if qed is not None and math.isfinite(qed):
        weighted_terms.append((0.25, _clamp(float(qed), 0.0, 1.0)))

    sa = mol.numeric("SA_Score")
    if sa is not None and math.isfinite(sa):
        weighted_terms.append((0.20, _clamp((10.0 - float(sa)) / 9.0, 0.0, 1.0)))

    logp = mol.numeric("LogP")
    if logp is not None and math.isfinite(logp):
        weighted_terms.append((0.15, _clamp((5.0 - float(logp)) / 6.0, 0.0, 1.0)))

    logs = mol.numeric("LogS")
    if logs is not None and math.isfinite(logs):
        weighted_terms.append((0.15, _clamp((float(logs) + 8.0) / 8.0, 0.0, 1.0)))

    pic50 = mol.numeric("pIC50")
    if pic50 is not None and math.isfinite(pic50):
        weighted_terms.append((0.25, _clamp((float(pic50) - 4.0) / 6.0, 0.0, 1.0)))

    dg = mol.numeric("dG")
    if dg is not None and math.isfinite(dg):
        weighted_terms.append((0.25, _clamp((-4.0 - float(dg)) / 8.0, 0.0, 1.0)))

    brenk = mol.numeric("Brenk_Alerts")
    if brenk is not None and math.isfinite(brenk):
        weighted_terms.append((0.10, _clamp(1.0 - 0.5 * float(brenk), 0.0, 1.0)))

    herg = mol.numeric("hERG")
    if herg is not None and math.isfinite(herg):
        weighted_terms.append((0.10, _clamp(1.0 - float(herg), 0.0, 1.0)))

    if not weighted_terms:
        return None

    total_w = sum(w for w, _ in weighted_terms)
    if total_w <= 0:
        return None
    raw_score = sum(w * s for w, s in weighted_terms) / total_w
    return round(raw_score, 4)


def _upsert_value(mol: Molecule, val: Value) -> None:
    for i, existing in enumerate(mol.values):
        if existing.name == val.name:
            mol.values[i] = val
            return
    mol.values.append(val)


def _top_k_indices(mols: list[Molecule], k: int = MAX_SHORTLIST) -> list[int]:
    scored = [(idx, compute_score(m) or 0.0) for idx, m in enumerate(mols)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in scored[:k]]


def _build_evaluator_user_prompt(mols: list[Molecule], filters_per_mol: list[list[FilterResult]]) -> str:
    items: list[dict[str, Any]] = []
    for idx, (mol, flts) in enumerate(zip(mols, filters_per_mol)):
        items.append(
            {
                "index": idx,
                "values": [
                    {
                        "name": v.name,
                        "value": v.value if v.value is not None else "не посчитано",
                        "units": v.units,
                        "provenance": v.provenance.value,
                        "source": v.source,
                    }
                    for v in mol.values
                ],
                "filters": [
                    {"name": f.name, "passed": f.passed, "detail": f.detail}
                    for f in flts
                ],
            }
        )
    return json.dumps({"molecules": items, "max_shortlist": MAX_SHORTLIST}, ensure_ascii=False)


def _select_shortlist(
    mols: list[Molecule],
    filters_per_mol: list[list[FilterResult]],
    loop: int,
    notes: list[str],
) -> list[int]:
    """
    Выбирает до 5 индексов молекул для докинга (§2, §3.4).
    На первом круге (loop == 1) обращается к LLM; если LLM вернула пустой список
    или недоступна, а также всегда на втором круге (loop > 1), берёт top-k по score.
    """
    fallback = _top_k_indices(mols, k=MAX_SHORTLIST)
    if loop > 1 or not mols:
        return fallback

    user_prompt = _build_evaluator_user_prompt(mols, filters_per_mol)
    try:
        raw = _cached_llm_call("Evaluator", SYSTEM, user_prompt, json_mode=True, attempt=1)
        parsed = parse_json_object(raw)
        raw_shortlist = parsed.get("shortlist")
        # risks из ответа LLM сюда НЕ кладём: notes вливаются в
        # result.limitations, которые жюри читает как вывод системы.
        # Непроверенная проза модели там выглядит как заключение,
        # а канал notes заведён под деградацию, а не под мнения (§2.5).

        if isinstance(raw_shortlist, list) and raw_shortlist:
            valid_indices: list[int] = []
            for idx in raw_shortlist:
                if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(mols):
                    if idx not in valid_indices:
                        valid_indices.append(idx)
                if len(valid_indices) >= MAX_SHORTLIST:
                    break
            if valid_indices:
                return valid_indices
    except (LLMUnavailableError, OfflineMiss):
        notes.append("Evaluator: LLM недоступна, шортлист на докинг выбран детерминированно (top-5 по score)")
    except Exception:
        notes.append("Evaluator: LLM вернула невалидный JSON, шортлист на докинг выбран кодом (top-5 по score)")

    return fallback


def evaluator(state: GraphState) -> dict[str, Any]:
    """
    Узел Evaluator (§2, §3.4).
    Порядок: chem.run_all_filters -> admet.profile -> qsar.predict -> docking.dock (top-5) -> score.
    Candidate.rank здесь НЕ выставляется (его ставит build_result).
    """
    generated = state.get("generated") or []
    t = state.get("thresholds") or Thresholds()
    loop = int(state.get("loop") or 1)
    parent: Molecule | None = state.get("parent")
    parent_smiles = parent.smiles if parent is not None else None
    qsar_state = state.get("qsar")
    receptor = state.get("receptor")

    tool_calls: list[ToolCall] = []
    notes: list[str] = list(state.get("notes") or [])

    if not generated:
        return {
            "candidates": [],
            "tool_calls": tool_calls,
            "notes": dedupe_strings(notes),
        }

    mols: list[Molecule] = []
    for raw_mol in generated:
        m = raw_mol.model_copy(deep=True)
        if not m.values:
            described = chem.describe(m.smiles, name=m.name, source=m.source)
            if described is not None:
                m = described
        mols.append(m)

    # 1. Детерминированные фильтры RDKit (Lipinski, PAINS, SA_Score)
    t0 = time.time()
    filters_per_mol: list[list[FilterResult]] = [
        chem.run_all_filters(m.smiles, sa_threshold=t.max_sa) for m in mols
    ]
    tool_calls.append(
        ToolCall(
            agent="Evaluator",
            tool="chem.run_all_filters",
            args_summary=f"n={len(mols)}",
            ok=True,
            from_cache=False,
            duration_s=round(time.time() - t0, 3),
        )
    )

    # 2. ADMET-профиль
    t0 = time.time()
    admet_ok = True
    admet_err: str | None = None
    try:
        for m in mols:
            for v in admet.profile(m.smiles):
                _upsert_value(m, v)
    except Exception as exc:
        admet_ok = False
        admet_err = str(exc)
        for m in mols:
            _upsert_value(m, not_computed("LogS", f"Сбой ADMET: {exc}"))
    tool_calls.append(
        ToolCall(
            agent="Evaluator",
            tool="admet.profile",
            args_summary=f"n={len(mols)}",
            ok=admet_ok,
            from_cache=False,
            duration_s=round(time.time() - t0, 3),
            error=admet_err,
        )
    )

    # 3. QSAR-инференс (если модель была обучена в Data Agent)
    if qsar_state and qsar_state.get("model") is not None and qsar_state.get("metrics") is not None:
        t0 = time.time()
        q_model = qsar_state["model"]
        q_metrics = qsar_state["metrics"]
        any_pred_ok = False
        for m in mols:
            pred_v = qsar.predict(q_model, q_metrics, m.smiles)
            if pred_v.value is not None:
                any_pred_ok = True
            _upsert_value(m, pred_v)
        tool_calls.append(
            ToolCall(
                agent="Evaluator",
                tool="qsar.predict",
                args_summary=f"n={len(mols)}",
                ok=any_pred_ok,
                from_cache=False,
                duration_s=round(time.time() - t0, 3),
                error=None if any_pred_ok else "QSAR не вернул валидных предсказаний",
            )
        )
    else:
        for m in mols:
            if m.get("pIC50") is None:
                _upsert_value(m, not_computed("pIC50", "QSAR-модель в этом прогоне не обучалась"))

    # 4. Выбор шортлиста на докинг (k <= 5) и докинг при наличии структуры рецептора
    shortlist = _select_shortlist(mols, filters_per_mol, loop, notes)
    shortlist_set = set(shortlist[:MAX_SHORTLIST])

    if (
        receptor
        and isinstance(receptor, dict)
        and receptor.get("pdbqt_path")
        and receptor.get("center")
    ):
        rec_pdbqt = str(receptor["pdbqt_path"])
        rec_center = tuple(float(x) for x in receptor["center"])
        rec_box = tuple(float(x) for x in receptor.get("box", (22.0, 22.0, 22.0)))
        rec_id = str(receptor.get("pdb_id") or "receptor")

        for idx, m in enumerate(mols):
            if idx in shortlist_set:
                t0 = time.time()
                dg_val = docking.dock(
                    m.smiles,
                    rec_pdbqt,
                    center=(rec_center[0], rec_center[1], rec_center[2]),
                    box=(rec_box[0], rec_box[1], rec_box[2]),
                )
                dur = round(time.time() - t0, 3)
                _upsert_value(m, dg_val)
                tool_calls.append(
                    ToolCall(
                        agent="Evaluator",
                        tool="dock",
                        args_summary=f"cand#{idx} in {rec_id}",
                        ok=dg_val.value is not None,
                        from_cache=False,
                        duration_s=dur,
                        error=None if dg_val.value is not None else (dg_val.note or "dG не вычислен"),
                    )
                )
            else:
                if m.get("dG") is None:
                    _upsert_value(m, not_computed("dG", "не вошёл в шортлист докинга (топ-5)"))
    else:
        for m in mols:
            if m.get("dG") is None:
                _upsert_value(m, not_computed("dG", "экспериментальная структура рецептора для докинга не задана"))

    # 5. Расчёт итогового score кодом; rank остаётся None до build_result
    candidates: list[Candidate] = []
    for m, flts in zip(mols, filters_per_mol):
        candidates.append(
            Candidate(
                molecule=m,
                parent_smiles=parent_smiles,
                filters=list(flts),
                score=compute_score(m),
                rank=None,
            )
        )

    return {
        "candidates": candidates,
        "tool_calls": tool_calls,
        "notes": dedupe_strings(notes),
    }


run = evaluator
