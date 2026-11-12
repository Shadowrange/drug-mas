"""Supervisor. ВЛАДЕЛЕЦ: Р-B. Блок A1.

Классифицирует запрос -> lead_optimization | target_to_hit | insufficient_data_demo,
строит план вызовов, маршрутизирует, решает когда остановить цикл возврата.
Максимум 2 круга возврата к Generator — иначе демо на защите не уложится в минуту.
"""

from __future__ import annotations

import re
from typing import Any

from src.cache import OfflineMiss
from src.graph_state import (
    GraphState,
    LLMUnavailableError,
    _cached_llm_call,
    dedupe_strings,
    parse_json_object,
)
from src.tools import chem

SYSTEM = """Ты Supervisor мультиагентной системы для дизайна лекарственных молекул.
Ты получаешь запрос пользователя на естественном языке и возвращаешь ОДИН
объект JSON. Больше ты не делаешь ничего: инструменты не вызываешь, молекулы
не предлагаешь, результаты не оцениваешь.

ЗАПРЕЩЕНО НАЗЫВАТЬ ЛЮБЫЕ ЧИСЛА И ИДЕНТИФИКАТОРЫ, КОТОРЫХ НЕТ В ЗАПРОСЕ
ДОСЛОВНО. Это касается SMILES, ΔG, IC50, Ki, LogP, PubChem CID, ChEMBL ID,
UniProt AC, PDB ID. Если пользователь написал «иматиниб», ты возвращаешь
parent_name="иматиниб" и parent_smiles=null. Ты НЕ ПОДСТАВЛЯЕШЬ CID 5291,
структуру или PDB 1IEP по памяти, даже если уверен в них. Эти значения
приходят из вызова базы данных в Data Agent или не приходят вообще.
Подставленный по памяти идентификатор — это выдуманный результат, и
верификатор дисквалифицирует за него весь прогон.

Выбери ровно один сценарий:
- "lead_optimization" — в запросе есть конкретное соединение, которое надо
  модифицировать, улучшив заданные свойства.
- "target_to_hit" — названа мишень, соединения-отправной точки нет, кандидатов
  надо искать и ранжировать под мишень.
- "insufficient_data_demo" — пользователь прямо просит показать поведение
  системы при недостатке данных.
Сомневаешься между первыми двумя — выбирай "target_to_hit".

Решать, достаточно ли данных, НЕ ТВОЯ РАБОТА. Это считает код по числу
фактически найденных записей. Ты только раскладываешь запрос на поля.

Формат ответа — JSON без markdown-обёртки, без текста до и после:
{
  "scenario": "lead_optimization" | "target_to_hit" | "insufficient_data_demo",
  "task": {
    "parent_name":       строка или null,
    "parent_smiles":     строка или null — ТОЛЬКО если SMILES дословно в запросе,
    "target_name":       строка или null,
    "target_uniprot":    строка или null — ТОЛЬКО если дословно в запросе,
    "target_chembl_id":  строка или null — ТОЛЬКО если дословно в запросе,
    "pdb_id":            строка или null — ТОЛЬКО если дословно в запросе,
    "objectives":        список коротких строк, чего хочет пользователь,
    "constraints":       список коротких строк, что запрещено или обязано сохраниться,
    "plan":              список шагов словами, без чисел
  }
}

Любое поле, которого в запросе нет, — null или пустой список. Пустое поле
честнее угаданного: дальше по графу его заполнит инструмент."""

VALID_SCENARIOS = {"lead_optimization", "target_to_hit", "insufficient_data_demo"}

_KNOWN_COMPOUNDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bиматиниб\w*\b", re.IGNORECASE), "иматиниб"),
    (re.compile(r"\bimatinib\b", re.IGNORECASE), "imatinib"),
    (re.compile(r"\bаспирин\w*\b", re.IGNORECASE), "аспирин"),
    (re.compile(r"\baspirin\b", re.IGNORECASE), "aspirin"),
    (re.compile(r"\bгефитиниб\w*\b", re.IGNORECASE), "гефитиниб"),
    (re.compile(r"\bgefitinib\b", re.IGNORECASE), "gefitinib"),
    (re.compile(r"\bэрлотиниб\w*\b", re.IGNORECASE), "эрлотиниб"),
    (re.compile(r"\berlotinib\b", re.IGNORECASE), "erlotinib"),
    (re.compile(r"\bдазатиниб\w*\b", re.IGNORECASE), "дазатиниб"),
    (re.compile(r"\bdasatinib\b", re.IGNORECASE), "dasatinib"),
    (re.compile(r"\bсорафениб\w*\b", re.IGNORECASE), "сорафениб"),
    (re.compile(r"\bsorafenib\b", re.IGNORECASE), "sorafenib"),
]

_UNIPROT_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9](?:[A-Z][A-Z0-9]{2}[0-9])?)\b"
)
_CHEMBL_RE = re.compile(r"\b(CHEMBL\d+)\b", re.IGNORECASE)
_PDB_RE = re.compile(r"\b([1-9][A-Za-z0-9]{3})\b")


def _extract_literal_smiles(query: str) -> str | None:
    """Извлекает SMILES ТОЛЬКО если он дословно присутствует в тексте запроса."""
    for raw_tok in query.split():
        tok = raw_tok.strip(",.;:()[]{}\"'«»")
        if len(tok) < 4:
            continue
        if _UNIPROT_RE.fullmatch(tok) or _CHEMBL_RE.fullmatch(tok) or _PDB_RE.fullmatch(tok):
            continue
        if not re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$%/\\\.]+", tok):
            continue
        if not any(ch in tok for ch in ("=", "#", "1", "2", "3", "(", ")", "[", "]", "c")):
            continue
        if chem.is_valid(tok):
            return tok
    return None


def _extract_target_name(query: str) -> str | None:
    """Извлекает имя мишени дословно из текста запроса."""
    for known in ("BCR-ABL1", "BCR-ABL", "EGFR", "ABL1", "BRAF", "CDK2", "HER2", "ALK", "JAK2", "KRAS"):
        m = re.search(rf"\b({re.escape(known)})\b", query, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    m_angle = re.search(r"мишен[иь]\s*(<[^>]+>)", query, flags=re.IGNORECASE)
    if m_angle:
        return m_angle.group(1).strip()

    m_named = re.search(
        r"(?:для\s+мишени|к\s+мишени|мишени|мишень|target)\s+([A-Za-zА-Яа-яЁё0-9_\-]+)",
        query,
        flags=re.IGNORECASE,
    )
    if m_named:
        cand = m_named.group(1).strip(",.;:()[]{}\"'«»")
        if _looks_like_target(cand):
            return cand
    return None


# Слова, которые стоят после «мишени» в обычной русской фразе и мишенью
# не являются. Без этого «...для малоизученной мишени с малым числом
# активностей» даёт target_name='с', и в журнале появляется
# «Для мишени 'с' не найден ChEMBL ID» — отказ правильный, но выглядит
# как баг разбора, а не как принципиальный отказ.
_TARGET_STOPWORDS = {
    "с", "со", "и", "в", "во", "на", "по", "из", "от", "для", "без", "при",
    "к", "ко", "о", "об", "у", "за", "над", "под", "про", "же", "ли", "не",
    "белка", "белок", "мишени", "мишень", "которой", "которая", "которого",
    "малым", "малого", "низким", "этой", "этого", "такой", "любой", "новой",
}


def _looks_like_target(cand: str) -> bool:
    """
    Символы генов латинские, содержат заглавную букву или цифру и длиннее
    одного знака: EGFR, BCR-ABL1, CDK2. Русское слово в нижнем регистре
    мишенью не бывает.

    Пустой результат честнее угаданного: дальше по графу недостающее имя
    приведёт к отказу с понятной формулировкой, а не к поиску по предлогу.
    """
    if not cand or len(cand) < 2:
        return False
    if cand.lower() in _TARGET_STOPWORDS:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]*", cand):
        return False
    return any(ch.isupper() or ch.isdigit() for ch in cand)


def _extract_task_regex(query: str) -> dict[str, Any]:
    """
    Детерминированная regex-экстракция полей task строго по дословному тексту query.
    Никакие ID из памяти не подставляются (§3.1).
    """
    parent_name: str | None = None
    for pat, canonical_name in _KNOWN_COMPOUNDS:
        if pat.search(query):
            parent_name = canonical_name
            break

    parent_smiles = _extract_literal_smiles(query)

    m_uni = _UNIPROT_RE.search(query)
    target_uniprot = m_uni.group(1).upper() if m_uni else None

    m_chembl = _CHEMBL_RE.search(query)
    target_chembl_id = m_chembl.group(1).upper() if m_chembl else None

    pdb_id: str | None = None
    for m_pdb in _PDB_RE.finditer(query):
        cand_pdb = m_pdb.group(1).upper()
        if cand_pdb != target_uniprot:
            pdb_id = cand_pdb
            break

    target_name = _extract_target_name(query)

    q_low = query.lower()
    objectives: list[str] = []
    if "растворим" in q_low or "solubil" in q_low:
        objectives.append("повысить растворимость")
    if "активн" in q_low or "модель" in q_low or "qsar" in q_low:
        objectives.append("обучить модель активности и оценить кандидатов")
    if "ингибитор" in q_low or "кандидат" in q_low or "предложи" in q_low:
        objectives.append("предложить и ранжировать кандидатов")
    if not objectives:
        objectives.append("выполнить поиск и оценку молекулярных кандидатов")

    constraints: list[str] = []
    if "сродств" in q_low or "связыван" in q_low or "affinity" in q_low:
        constraints.append("сохранить сродство к мишени")
    if "токсич" in q_low or "фильтр" in q_low or "toxic" in q_low:
        constraints.append("отфильтровать токсичные аналоги")

    plan: list[str] = [
        "опросить внешние базы данных по исходному соединению и мишени",
        "проверить достаточность экспериментальных данных",
        "сгенерировать структурные аналоги и рассчитать физико-химические и ADMET-свойства",
        "выполнить верификацию порогов и происхождения каждого значения",
        "сформировать интерпретируемый отчёт",
    ]

    return {
        "parent_name": parent_name,
        "parent_smiles": parent_smiles,
        "target_name": target_name,
        "target_uniprot": target_uniprot,
        "target_chembl_id": target_chembl_id,
        "pdb_id": pdb_id,
        "objectives": objectives,
        "constraints": constraints,
        "plan": plan,
    }


def classify(query: str) -> str:
    """Классификация сценария по запросу: lead_optimization | target_to_hit | insufficient_data_demo."""
    q_low = (query or "").lower()
    if any(
        kw in q_low
        for kw in (
            "недостат",
            "insufficient_data",
            "без данных",
            "редкая",
            "несуществующ",
            "отказ при недостатке",
        )
    ):
        return "insufficient_data_demo"

    task = _extract_task_regex(query or "")
    if task.get("parent_name") or task.get("parent_smiles"):
        return "lead_optimization"

    if any(kw in q_low for kw in ("модифицир", "оптимизир", "улучш", "аналог")):
        return "lead_optimization"

    return "target_to_hit"


def _sanitize_task_against_query(raw_task: dict[str, Any], query: str) -> dict[str, Any]:
    """
    Гарантирует соблюдение запрета §3.1: идентификаторы (SMILES, UniProt, ChEMBL, PDB),
    которых нет дословно в query, обнуляются даже если LLM подставила их по памяти.
    """
    fallback = _extract_task_regex(query)
    q_low = query.lower()

    def _keep_if_literal(val: Any) -> str | None:
        if not val or not isinstance(val, str):
            return None
        s = val.strip()
        if not s:
            return None
        return s if s.lower() in q_low else None

    parent_name = raw_task.get("parent_name")
    if isinstance(parent_name, str):
        parent_name = parent_name.strip() or None
    else:
        parent_name = None
    if parent_name and parent_name.lower() not in q_low and fallback["parent_name"]:
        parent_name = fallback["parent_name"]
    elif parent_name is None:
        parent_name = fallback["parent_name"]

    target_name = raw_task.get("target_name")
    if isinstance(target_name, str):
        target_name = target_name.strip() or None
    else:
        target_name = None
    if target_name is None:
        target_name = fallback["target_name"]

    parent_smiles = _keep_if_literal(raw_task.get("parent_smiles"))
    if parent_smiles and not chem.is_valid(parent_smiles):
        parent_smiles = None

    target_uniprot = _keep_if_literal(raw_task.get("target_uniprot"))
    target_chembl_id = _keep_if_literal(raw_task.get("target_chembl_id"))
    pdb_id = _keep_if_literal(raw_task.get("pdb_id"))

    objectives = raw_task.get("objectives")
    if not isinstance(objectives, list) or not all(isinstance(x, str) for x in objectives):
        objectives = fallback["objectives"]

    constraints = raw_task.get("constraints")
    if not isinstance(constraints, list) or not all(isinstance(x, str) for x in constraints):
        constraints = fallback["constraints"]

    plan = raw_task.get("plan")
    if not isinstance(plan, list) or not all(isinstance(x, str) for x in plan):
        plan = fallback["plan"]

    return {
        "parent_name": parent_name,
        "parent_smiles": parent_smiles,
        "target_name": target_name,
        "target_uniprot": target_uniprot.upper() if target_uniprot else None,
        "target_chembl_id": target_chembl_id.upper() if target_chembl_id else None,
        "pdb_id": pdb_id.upper() if pdb_id else None,
        "objectives": objectives,
        "constraints": constraints,
        "plan": plan,
    }


def _parse_supervisor_payload(raw: str, query: str) -> tuple[str, dict[str, Any]]:
    data = parse_json_object(raw)
    scenario = data.get("scenario")
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Недопустимый scenario от LLM: {scenario!r}")
    raw_task = data.get("task")
    if not isinstance(raw_task, dict):
        raise ValueError("Поле 'task' в ответе Supervisor не является объектом")
    task = _sanitize_task_against_query(raw_task, query)
    return str(scenario), task


def supervisor(state: GraphState) -> dict[str, Any]:
    """
    Узел Supervisor (§2, §3.1).
    Делает 1 вызов LLM в JSON-режиме; при невалидном JSON — 1 retry, затем
    дефолт scenario="target_to_hit", task из regex-экстракции и запись в notes/limitations.
    """
    query = (state.get("query") or "").strip()
    notes = list(state.get("notes") or [])

    for attempt in (1, 2):
        try:
            raw = _cached_llm_call("Supervisor", SYSTEM, query, json_mode=True, attempt=attempt)
            scenario, task = _parse_supervisor_payload(raw, query)
            return {
                "scenario": scenario,
                "task": task,
                "tool_calls": [],
                "notes": dedupe_strings(notes),
            }
        except (LLMUnavailableError, OfflineMiss):
            # Без ключа LLM или при отсутствии записи в кэше не делаем бессмысленный второй сетевой вызов
            task = _extract_task_regex(query)
            scenario = classify(query)
            notes.append(
                "Supervisor: LLM недоступна в текущем окружении, применён детерминированный разбор запроса"
            )
            return {
                "scenario": scenario,
                "task": task,
                "tool_calls": [],
                "notes": dedupe_strings(notes),
            }
        except Exception:
            if attempt == 2:
                task = _extract_task_regex(query)
                notes.append(
                    "Supervisor: LLM вернула невалидный JSON после повтора, применён дефолт и regex-экстракция"
                )
                return {
                    "scenario": "target_to_hit",
                    "task": task,
                    "tool_calls": [],
                    "notes": dedupe_strings(notes),
                }

    task = _extract_task_regex(query)
    return {
        "scenario": "target_to_hit",
        "task": task,
        "tool_calls": [],
        "notes": dedupe_strings(notes),
    }


run = supervisor
