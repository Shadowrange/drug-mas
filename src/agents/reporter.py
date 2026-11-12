"""Узел `Reporter`: сборка финального `ScenarioResult` и формирование текстового резюме (`summary`).

Сначала детерминированно собирает валидированный объект `ScenarioResult` через `verifier.build_result`,
затем передаёт сериализованный объект в LLM для генерации связной интерпретации и проверяет
результирующий текст кодовыми постфильтрами на отсутствие посторонних чисел и SMILES (в режиме отказа).
"""

from __future__ import annotations

import re

from typing import Any

from src.cache import OfflineMiss
from src.graph_state import (
    GraphState,
    LLMUnavailableError,
    _cached_llm_call,
    dedupe_by_smiles,
    dedupe_strings,
)
from src.schemas import Provenance, ScenarioResult, Value
from src.tools import chem
from src.verifier import VerificationReport, build_result

SYSTEM = """Ты Reporter. Тебе дан ОДИН объект ScenarioResult, уже собранный и
провалидированный кодом. Кроме него у тебя нет никаких данных. Ты пишешь
связный текст на русском для поля summary. На фронте этот текст помечен
как ИНТЕРПРЕТАЦИЯ и показан отдельно от таблицы данных.

ГЛАВНЫЙ ЗАПРЕТ. Тебе запрещено называть любое число, которого нет в полях
поданного объекта. Разрешённые источники чисел — ровно эти поля и ничего
больше:
  candidates[].molecule.values[].value   вместе с units
  candidates[].filters[].threshold
  candidates[].score, candidates[].rank
  missing[].found, missing[].needed
  tool_calls[].duration_s
Запрещено: округлять, усреднять, переводить единицы, считать разности
и проценты, восстанавливать одну величину из другой, называть число
«примерно» или «порядка». Любая арифметика над этими числами — это новое
число, которого в объекте нет. Если нужного числа в объекте нет, скажи
словами, что оно не вычислялось.

КАЖДОЕ НАЗВАННОЕ ЧИСЛО СОПРОВОЖДАЙ КАТЕГОРИЕЙ ИСТОЧНИКА в скобках по полю
provenance: (БД), (расчёт), (прогноз), (гипотеза). Для прогноза добавляй
метрику модели из model_metrics — предсказание без метрики качества модели
не является аргументом.

Величина со value=null — это «не посчитано», а НЕ «плохое значение» и не
нуль. Фильтр с passed=null — «не проверялось», а не «провалено». Это разные
вещи, и путать их нельзя.

Поведение по полю status:
- "ok": сводка по кандидатам из candidates, в порядке rank. По каждому —
  что посчитано и чем подтверждено. Обязательно перечисли содержимое
  limitations: что не вычислялось и не участвовало в ранжировании.
- "no_candidates_passed": КАНДИДАТОВ, ПРОШЕДШИХ ПОРОГИ, НЕТ. Это результат,
  а не сбой. Скажи это первой фразой, затем по каждому кандидату — по какому
  фильтру и с каким порогом он отсеян, по полям filters. Не предлагай считать
  отсеянных пригодными и не советуй ослабить пороги.
- "insufficient_data": МОЛЕКУЛ НЕТ, И ПРЕДЛАГАТЬ ИХ ЗАПРЕЩЕНО. Ни одного
  SMILES, ни одного названия соединения, ни одного «можно было бы
  рассмотреть» в тексте быть не должно. Вместо этого по каждому элементу
  missing скажи: чего не хватило (what), сколько и где нашли (found),
  сколько нужно (needed), что это разблокировало бы (would_unlock).
  Закончи тем, что система может сделать без этих данных.

Пиши прозой, без списков-маркеров, 4–8 предложений. Не хвали результат
и не обещай шагов, которых система не делала."""

_PROV_LABEL: dict[Provenance, str] = {
    Provenance.DATABASE: "БД",
    Provenance.COMPUTED: "расчёт",
    Provenance.PREDICTED: "прогноз",
    Provenance.HYPOTHESIS: "гипотеза",
}


def _format_value_strict(v: Value) -> str:
    """Форматирует одно значение из ScenarioResult строго с категорией provenance и метриками."""
    if v.value is None:
        return f"{v.name}: не посчитано"
    cat = _PROV_LABEL.get(v.provenance, v.provenance.value)
    unit_str = f" {v.units}" if v.units else ""
    if v.provenance is Provenance.PREDICTED and v.model_metrics is not None:
        m = v.model_metrics
        return (
            f"{v.name}={v.value}{unit_str} ({cat}, {m.model_name}, "
            f"{m.metric_name}={m.metric_value}, n_train={m.n_train})"
        )
    return f"{v.name}={v.value}{unit_str} ({cat})"


def _build_deterministic_summary(result: ScenarioResult) -> str:
    """
    Детерминированная генерация текста интерпретации ТОЛЬКО по валидированному ScenarioResult
    с соблюдением всех правил §3.5 (без выдуманных чисел, без арифметики, с пометками источников).
    """
    if result.status == "insufficient_data":
        sentences: list[str] = [
            "Данных по запрошенной мишени недостаточно, поэтому молекулы-кандидаты системой не предлагаются."
        ]
        for m in result.missing:
            sentences.append(
                f"Отсутствует категория «{m.what}»: обнаружено {m.found}, требуется {m.needed}, "
                f"что разблокировало бы {m.would_unlock}."
            )
        sentences.append(
            "Без указанных экспериментальных данных система может выполнять только детерминированный "
            "расчёт физико-химических дескрипторов и проверку структурных фильтров для уже известных структур, "
            "без предсказания биологической активности и без молекулярного докинга."
        )
        return " ".join(sentences)

    if result.status == "no_candidates_passed":
        sentences = ["Кандидатов, прошедших пороги, нет."]
        for cand in result.candidates[:5]:
            failed_parts = [
                f"{f.name} ({f.detail}, порог {f.threshold})"
                for f in cand.failed
            ]
            reason_str = "; ".join(failed_parts) if failed_parts else "не пройдена проверка верификатора"
            sentences.append(f"Структура {cand.molecule.smiles} отсеяна по фильтрам: {reason_str}.")
        if result.limitations:
            sentences.append(f"Зафиксированные ограничения прогона: {'; '.join(result.limitations)}.")
        return " ".join(sentences)

    # status == "ok"
    sentences = []
    for cand in result.candidates[:3]:
        key_names = ("LogP", "QED", "SA_Score", "LogS", "pIC50", "dG")
        val_parts: list[str] = []
        for name in key_names:
            v = cand.molecule.get(name)
            if v is not None:
                val_parts.append(_format_value_strict(v))
        vals_joined = ", ".join(val_parts)
        sentences.append(
            f"Кандидат с рангом rank={cand.rank} и интегральной оценкой score={cand.score} (расчёт) "
            f"({cand.molecule.smiles}, источник структуры: {cand.molecule.source}) имеет показатели: {vals_joined}."
        )
    if result.limitations:
        sentences.append(f"Ограничения прогона: {'; '.join(result.limitations)}.")
    return " ".join(sentences)


def _postfilter_insufficient_data_smiles(summary: str) -> tuple[str, bool]:
    """
    Постфильтр кода (§2.6): на ветке insufficient_data проверяет каждый токен через
    `chem.is_valid(tok)` и вырезает любые валидные SMILES.
    """
    kept_tokens: list[str] = []
    removed = False
    for raw_tok in summary.split():
        cleaned = raw_tok.strip(",.;:!?()[]{}\"'«»")
        if cleaned and chem.is_valid(cleaned):
            removed = True
            continue
        kept_tokens.append(raw_tok)
    return " ".join(kept_tokens), removed


# Регулярное выражение для извлечения отдельно стоящих чисел (игнорирует цифры внутри
# идентификаторов и названий метрик вроде pIC50, CHEMBL203, CYP3A4, 1IEP).
_NUM_RE = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9_])-?\d+(?:[.,]\d+)?")


def _allowed_numbers(result: ScenarioResult) -> set[str]:
    """Числа, которые Reporter имеет право назвать: ровно поля объекта (§3.5)."""
    def norm(x: object) -> str:
        try:
            f = float(str(x).replace(",", "."))
        except (TypeError, ValueError):
            return ""
        return f"{f:g}"

    out: set[str] = set()
    for c in result.candidates:
        for v in c.molecule.values:
            if v.value is not None:
                out.add(norm(v.value))
            if v.model_metrics:
                out.add(norm(v.model_metrics.metric_value))
                out.add(norm(v.model_metrics.n_train))
        for f in c.filters:
            for tok in _NUM_RE.findall(f.threshold or ""):
                out.add(norm(tok))
            for tok in _NUM_RE.findall(f.detail or ""):
                out.add(norm(tok))
        out.add(norm(c.score))
        out.add(norm(c.rank))
    for md in result.missing:
        for field in (md.found, md.needed):
            for tok in _NUM_RE.findall(field or ""):
                out.add(norm(tok))
    out.add(norm(len(result.candidates)))
    out.discard("")
    return out


def _extraneous_numbers(text: str, result: ScenarioResult) -> list[str]:
    """Числа из текста, которых нет в объекте. Маленькие целые (счёт,
    перечисление, годы) не считаем: они не выдают себя за результат."""
    allowed = _allowed_numbers(result)
    bad: list[str] = []
    for tok in _NUM_RE.findall(text):
        try:
            f = float(tok.replace(",", "."))
        except ValueError:
            continue
        if f.is_integer() and 0 <= abs(f) <= 10:
            continue
        if 1900 <= f <= 2100 and f.is_integer():
            continue
        if f"{f:g}" not in allowed and tok not in bad:
            bad.append(tok)
    return bad


def reporter(state: GraphState) -> dict[str, Any]:
    """
    Узел Reporter (§2.5 Поправка 3, §2.6, §3.5).
    Строгий порядок:
      1. Собрать VerificationReport из накопленных passed_all, rejected_all, unverifiable_all
         и вызвать build_result(...) с пустым summary="".
      2. Передать в LLM ТОЛЬКО сериализованный ScenarioResult.
      3. Записать полученный текст в summary (с постфильтром SMILES для insufficient_data).
    """
    state_missing = list(state.get("missing") or [])
    notes = list(state.get("notes") or [])

    if state_missing:
        missing = state_missing
        rep = VerificationReport(
            passed=[],
            rejected=[],
            unverifiable=dedupe_strings(list(state.get("unverifiable_all") or [])),
            feedback=[],
        )
    else:
        missing = []
        passed = dedupe_by_smiles(list(state.get("passed_all") or []))
        rejected = dedupe_by_smiles(list(state.get("rejected_all") or []))
        unverifiable = dedupe_strings(list(state.get("unverifiable_all") or []))

        if not passed and not rejected and state.get("verification") is not None:
            v_rep = state["verification"]
            passed = dedupe_by_smiles(list(v_rep.passed))
            rejected = dedupe_by_smiles(list(v_rep.rejected))
            unverifiable = dedupe_strings(unverifiable + list(v_rep.unverifiable))

        if passed:
            passed_canons = {chem.canonical(c.molecule.smiles) or c.molecule.smiles for c in passed}
            rejected = [
                c
                for c in rejected
                if (chem.canonical(c.molecule.smiles) or c.molecule.smiles) not in passed_canons
            ]

        rep = VerificationReport(
            passed=passed,
            rejected=rejected,
            unverifiable=unverifiable,
            feedback=[],
        )

    # 1. Вызов build_result с пустым summary=""
    result = build_result(
        query=state.get("query") or "",
        scenario=state.get("scenario") or "target_to_hit",
        rep=rep,
        missing=missing,
        tool_calls=list(state.get("tool_calls") or []),
        summary="",
    )

    limitations = dedupe_strings(list(result.limitations) + notes)
    result = result.model_copy(update={"limitations": limitations})

    # 2. Передача в LLM только валидированного и сериализованного ScenarioResult.
    # Поля duration_s и from_cache нормализуются перед сериализацией, чтобы ключ
    # кэша @cached("llm") зависел только от содержательных результатов прогона.
    _stable = result.model_copy(update={
        "tool_calls": [
            tc.model_copy(update={"duration_s": None, "from_cache": False})
            for tc in result.tool_calls
        ]
    })
    serialized_result = _stable.model_dump_json(indent=2)
    try:
        summary_text = _cached_llm_call("Reporter", SYSTEM, serialized_result, json_mode=False, attempt=1)
    except (LLMUnavailableError, OfflineMiss):
        limitations = dedupe_strings(
            limitations
            + ["Reporter: LLM недоступна, интерпретация сформирована детерминированно по полям ScenarioResult"]
        )
        result = result.model_copy(update={"limitations": limitations})
        summary_text = _build_deterministic_summary(result)
    except Exception:
        limitations = dedupe_strings(
            limitations
            + ["Reporter: сбой генерации текста LLM, интерпретация сформирована детерминированно"]
        )
        result = result.model_copy(update={"limitations": limitations})
        summary_text = _build_deterministic_summary(result)

    # 3. Кодовый постфильтр чисел: проверяет, что в тексте LLM нет числовых значений,
    # отсутствующих в полях валидированного объекта ScenarioResult.
    extraneous = _extraneous_numbers(summary_text, result)
    if extraneous:
        limitations = dedupe_strings(
            limitations
            + [
                "Постфильтр Reporter: в тексте LLM найдены числа, которых нет в ScenarioResult "
                f"({', '.join(extraneous[:5])}); интерпретация заменена на детерминированную"
            ]
        )
        result = result.model_copy(update={"limitations": limitations})
        summary_text = _build_deterministic_summary(result)

    # 4. Постфильтр SMILES на ветке insufficient_data (§2.6)
    if result.status == "insufficient_data":
        summary_text, had_smiles = _postfilter_insufficient_data_smiles(summary_text)
        if had_smiles:
            limitations = dedupe_strings(
                limitations
                + ["Постфильтр Reporter: из текста интерпретации на ветке insufficient_data удалён SMILES"]
            )

    result = result.model_copy(update={"summary": summary_text, "limitations": limitations})

    return {
        "summary": result.summary,
        "result": result,
        "tool_calls": [],
    }


run = reporter
