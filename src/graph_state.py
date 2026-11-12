"""
Объект состояния графа и общие типы оркестрации. ВЛАДЕЛЕЦ: Р-B. Блок A1.

Редьюсер `operator.add` задан только у `tool_calls`, остальные поля
перезаписываются последним писателем (см. docs/GRAPH-CONTRACT.md §1, §2.5).
Все вызовы к внешнему LLM проходят строго через `@cached("llm")` из `src/cache.py`.
"""

from __future__ import annotations

import json
import operator
import os
import re
import urllib.error
import urllib.request
from typing import Annotated, Any, TypedDict

from src.cache import cached
from src.schemas import Candidate, MissingData, Molecule, ScenarioResult, ToolCall
from src.tools import chem
from src.verifier import Thresholds, VerificationReport


class TaskSpec(TypedDict, total=False):
    parent_name: str | None
    parent_smiles: str | None
    target_name: str | None
    target_uniprot: str | None
    target_chembl_id: str | None
    pdb_id: str | None
    objectives: list[str]
    constraints: list[str]
    plan: list[str]


class GraphState(TypedDict, total=False):
    """Состояние графа LangGraph строго по docs/GRAPH-CONTRACT.md §1 и §2.5."""

    query: str
    scenario: str
    task: dict[str, Any]
    parent: Molecule | None
    activities: list[dict[str, Any]]
    n_activities: int
    has_structure: bool
    receptor: dict[str, Any] | None
    qsar: dict[str, Any] | None
    missing: list[MissingData]
    generated: list[Molecule]
    candidates: list[Candidate]
    verification: VerificationReport | None
    rejected_all: list[Candidate]
    feedback: list[str]
    loop: int
    tool_calls: Annotated[list[ToolCall], operator.add]
    summary: str
    result: ScenarioResult | None

    # Поля из §2.5 (поправки 1–3)
    unverifiable_all: list[str]
    passed_all: list[Candidate]
    notes: list[str]

    # Опциональные пороги верификатора (например, для теста с max_logp=-99)
    thresholds: Thresholds | None


class LLMUnavailableError(RuntimeError):
    """Ключ/сервис LLM недоступен и ответа нет в кэше. Громкое исключение без выдумки."""


def dedupe_by_smiles(candidates: list[Candidate]) -> list[Candidate]:
    """Дедупликация кандидатов по каноническому SMILES с сохранением порядка."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        canon = chem.canonical(c.molecule.smiles) or c.molecule.smiles
        if canon not in seen:
            seen.add(canon)
            out.append(c)
    return out


def dedupe_strings(items: list[str]) -> list[str]:
    """Дедупликация строковых списков (unverifiable_all, notes) с сохранением порядка."""
    return list(dict.fromkeys(items))


@cached("llm")
def _cached_llm_call(
    agent: str,
    system: str,
    user: str,
    json_mode: bool = True,
    attempt: int = 1,
) -> str:
    """
    Единая точка обращения к LLM через @cached("llm").
    При наличии ответа в cache/ возвращает его без обращения к сети и без ключа.
    При отсутствии в кэше и пустом LLM_API_KEY поднимает явный LLMUnavailableError.
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise LLMUnavailableError(
            f"[{agent}] LLM_API_KEY не задан в окружении, а ответ отсутствует в cache/."
        )

    model = os.getenv("LLM_MODEL", "").strip() or "gpt-4o-mini"
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "drug-mas/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_resp = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise LLMUnavailableError(f"[{agent}] HTTP {exc.code} от LLM API: {exc}") from exc
    except Exception as exc:
        raise LLMUnavailableError(f"[{agent}] Ошибка вызова LLM API: {type(exc).__name__}: {exc}") from exc

    try:
        data = json.loads(raw_resp)
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMUnavailableError(f"[{agent}] Некорректный формат ответа LLM API: {exc}") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMUnavailableError(f"[{agent}] Пустой ответ от LLM API")
    return content.strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    """Разбирает JSON-объект из ответа LLM (допуская случайную markdown-обёртку)."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # «Вот JSON:\n```json\n{...}```» — обёртка не в начале строки, и
        # срез префикса выше не срабатывает. Generator и Evaluator ретраев
        # не имеют: один плохой разбор, и они молча берут дефолты.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Ожидался JSON-объект (dict), получено: {type(parsed).__name__}")
    return parsed
