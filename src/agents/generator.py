"""Узел `Generator`: генерация структурных аналогов вокруг исходного соединения (`parent`).

Формирует параметры генерации с учётом целей запроса и обратной связи верификатора (`feedback`),
вызывает `tools.generate.mutate` и отфильтровывает дубликаты и ранее отсеянные молекулы.
"""

from __future__ import annotations

import json
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
from src.schemas import Molecule, ToolCall
from src.tools import chem, generate

SYSTEM = """Ты Generator. Ты задаёшь ПАРАМЕТРЫ генерации структур. Ты не придумываешь
молекулы.

ЗАПРЕЩЕНО ВЫВОДИТЬ SMILES, InChI, InChIKey, названия соединений или любое
описание конкретной структуры. Структуры порождает алгоритм — CReM или
STONED-SELFIES — из исходного соединения. SMILES, который ты напишешь сам,
будет либо невалиден, либо выдумкой, и в обоих случаях прогон испорчен.
ЗАПРЕЩЕНО называть любые числовые значения свойств: LogP, QED, ΔG, IC50,
растворимость. Их считают инструменты после генерации.

Тебе дано: исходное соединение — только как факт его наличия и его посчитанные
величины с источниками, objectives и constraints из task, и feedback
верификатора с прошлого круга, если круг не первый.

Feedback — это не пожелание, а список конкретных причин отсева с порогами.
Твои параметры обязаны отвечать именно на него. Формулировки вида «сделать
лучше» бесполезны: у тебя один круг на исправление.

Формат ответа — JSON без markdown-обёртки:
{
  "backend": "crem" | "selfies",
  "n": целое от 20 до 100,
  "strategy": "одна фраза: что меняем в скелете и почему это отвечает на feedback",
  "keep": ["фрагменты или фармакофорные элементы, которые нельзя терять, словами"],
  "avoid": ["чего избегать, словами"]
}

На первом круге backend = "crem". Если прошлый круг не дал ни одной валидной
новой структуры — "selfies": повтор с теми же параметрами даст тот же пустой
результат."""


def _build_generator_user_prompt(state: GraphState, new_loop: int) -> str:
    parent: Molecule | None = state.get("parent")
    task = state.get("task") or {}
    feedback = state.get("feedback") or []
    prev_empty = new_loop > 1 and not state.get("generated")

    parent_values = [
        {
            "name": v.name,
            "value": v.value,
            "units": v.units,
            "provenance": v.provenance.value,
            "source": v.source,
        }
        for v in (parent.values if parent else [])
    ]
    payload = {
        "loop": new_loop,
        "previous_round_empty": prev_empty,
        "parent_present": parent is not None,
        "parent_source": parent.source if parent else None,
        "parent_values": parent_values,
        "objectives": task.get("objectives") or [],
        "constraints": task.get("constraints") or [],
        "feedback": feedback,
    }
    return json.dumps(payload, ensure_ascii=False)


def _choose_params(state: GraphState, new_loop: int, notes: list[str]) -> tuple[str, int]:
    default_backend = "crem" if new_loop == 1 else "selfies"
    default_n = 25

    user_prompt = _build_generator_user_prompt(state, new_loop)
    try:
        raw = _cached_llm_call("Generator", SYSTEM, user_prompt, json_mode=True, attempt=1)
        parsed = parse_json_object(raw)
        backend = str(parsed.get("backend") or default_backend).strip().lower()
        if backend not in {"crem", "selfies"}:
            backend = default_backend
        raw_n = parsed.get("n", default_n)
        n = int(raw_n) if isinstance(raw_n, (int, float)) else default_n
        n = max(20, min(100, n))
    except (LLMUnavailableError, OfflineMiss):
        backend = default_backend
        n = default_n
        notes.append(
            f"Generator (круг {new_loop}): LLM недоступна, применены дефолтные параметры (backend={backend}, n={n})"
        )
    except Exception:
        backend = default_backend
        n = default_n
        notes.append(
            f"Generator (круг {new_loop}): LLM вернула невалидный JSON, применены дефолтные параметры (backend={backend}, n={n})"
        )

    # По §2.6: при повторном круге переключаем backend кодом (crem -> selfies),
    # чтобы исключить детерминированный повтор того же результата.
    if new_loop > 1:
        backend = "selfies"

    return backend, n


def generator(state: GraphState) -> dict[str, Any]:
    """
    Узел Generator (§2, §2.5 Поправка 1, §2.6, §3.3).
    Единственное место инкремента `loop += 1` во всём графе.
    """
    new_loop = int(state.get("loop", 0)) + 1
    notes: list[str] = list(state.get("notes") or [])
    tool_calls: list[ToolCall] = []

    parent: Molecule | None = state.get("parent")
    if parent is None or state.get("missing"):
        notes.append(f"генератор не выдал ни одной валидной новой структуры за {new_loop} кругов")
        return {
            "generated": [],
            "loop": new_loop,
            "tool_calls": tool_calls,
            "notes": dedupe_strings(notes),
        }

    backend, n = _choose_params(state, new_loop, notes)
    seed = 0xF00D + (new_loop - 1)

    t0 = time.time()
    try:
        raw_mols = generate.mutate(parent.smiles, n=n, backend=backend, seed=seed)
        dur = round(time.time() - t0, 3)
        tool_calls.append(
            ToolCall(
                agent="Generator",
                tool="generate.mutate",
                args_summary=f"backend={backend}, n={n}, loop={new_loop}",
                ok=True,
                from_cache=False,
                duration_s=dur,
            )
        )
    except Exception as exc:
        dur = round(time.time() - t0, 3)
        tool_calls.append(
            ToolCall(
                agent="Generator",
                tool="generate.mutate",
                args_summary=f"backend={backend}, n={n}, loop={new_loop}",
                ok=False,
                from_cache=False,
                duration_s=dur,
                error=str(exc),
            )
        )
        raw_mols = []

    # Исключаем уже отсеянные (rejected_all) и уже прошедшие (passed_all) структуры,
    # а также валидируем и дедуплицируем по каноническому SMILES (§2, §2.6).
    banned_smiles = [
        c.molecule.smiles
        for c in list(state.get("rejected_all") or []) + list(state.get("passed_all") or [])
    ]
    parent_canon = chem.canonical(parent.smiles)
    if parent_canon:
        banned_smiles.append(parent_canon)

    filtered_mols = generate.exclude_known(raw_mols, banned_smiles)

    seen_canon: set[str] = set()
    generated: list[Molecule] = []
    for m in filtered_mols:
        canon = chem.canonical(m.smiles)
        if canon is None or canon in seen_canon:
            continue
        seen_canon.add(canon)
        if not m.values:
            described = chem.describe(canon, name=m.name, source=m.source)
            if described is None:
                continue
            generated.append(described)
        else:
            generated.append(m.model_copy(deep=True))

    if not generated:
        notes.append(f"генератор не выдал ни одной валидной новой структуры за {new_loop} кругов")

    return {
        "generated": generated,
        "loop": new_loop,
        "tool_calls": tool_calls,
        "notes": dedupe_strings(notes),
    }


run = generator
