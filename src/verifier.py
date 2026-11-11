"""
ВЕРИФИКАТОР. ВЛАДЕЛЕЦ: Р-B. НЕ LLM.

Главный аргумент команды на защите. Проверка «не выдумала ли система число»
не может жить в промпте: промпт — это просьба, а здесь нужен запрет.

Три исхода, которые нельзя путать:
  ok                   — кандидаты есть и прошли пороги
  no_candidates_passed — кандидаты были, все провалили пороги (это РЕЗУЛЬТАТ)
  insufficient_data    — данных не хватило, генерацию запускать нельзя (это ОТКАЗ)

Классическая ошибка: считать «значение не посчитано» за «порог не пройден».
Из-за неё система выдаёт уверенный рейтинг по пустым данным.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas import (
    Candidate,
    FilterResult,
    MissingData,
    Provenance,
    ScenarioResult,
    ToolCall,
)


@dataclass
class Thresholds:
    """Пороги. Менять только через пост [decision] на доску."""

    max_logp: float = 5.0
    min_qed: float = 0.30
    max_sa: float = 6.0
    max_dg: float = -6.0          # ккал/моль, чем отрицательнее тем лучше
    min_qsar_train: int = 100     # меньше — QSAR не обучаем
    max_tox_risk: float = 0.5


@dataclass
class VerificationReport:
    passed: list[Candidate] = field(default_factory=list)
    rejected: list[Candidate] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)   # уходит обратно в Generator
    unverifiable: list[str] = field(default_factory=list)


def check_provenance(candidates: list[Candidate]) -> list[str]:
    """
    Последний рубеж. Pydantic уже не пропустил значение без source,
    но проверяем ещё и осмысленность: PREDICTED без метрик модели,
    HYPOTHESIS, попавшую в числовые поля ранжирования.
    """
    problems: list[str] = []
    for c in candidates:
        for v in c.molecule.values:
            if v.provenance is Provenance.PREDICTED and v.model_metrics is None:
                problems.append(f"{c.molecule.smiles}: '{v.name}' — прогноз без метрик модели")
            if v.provenance is Provenance.HYPOTHESIS and v.name in {"dG", "pIC50", "QED", "LogP"}:
                problems.append(f"{c.molecule.smiles}: '{v.name}' помечено как гипотеза, но используется как число")
            if not v.source or len(v.source) < 3:
                problems.append(f"{c.molecule.smiles}: '{v.name}' без источника")
    return problems


def _threshold_checks(c: Candidate, t: Thresholds) -> list[FilterResult]:
    """
    passed=None означает «не посчитано», и это НЕ провал.
    Разница между None и False — разница между отказом и отрицательным ответом.
    """
    out: list[FilterResult] = []

    def check(name: str, val: float | None, ok: bool, detail: str, thr: str) -> None:
        out.append(FilterResult(name=name, passed=None if val is None else ok,
                                detail="не посчитано" if val is None else detail, threshold=thr))

    logp = c.molecule.numeric("LogP")
    check("LogP", logp, logp is not None and logp <= t.max_logp,
          f"LogP = {logp}", f"<= {t.max_logp}")

    qed = c.molecule.numeric("QED")
    check("QED", qed, qed is not None and qed >= t.min_qed,
          f"QED = {qed}", f">= {t.min_qed}")

    sa = c.molecule.numeric("SA_Score")
    check("SA_Score", sa, sa is not None and sa <= t.max_sa,
          f"SA = {sa}", f"<= {t.max_sa}")

    dg = c.molecule.numeric("dG")
    check("dG", dg, dg is not None and dg <= t.max_dg,
          f"ΔG = {dg} ккал/моль", f"<= {t.max_dg}")

    return out


def verify(candidates: list[Candidate], t: Thresholds | None = None) -> VerificationReport:
    t = t or Thresholds()
    rep = VerificationReport()

    for problem in check_provenance(candidates):
        rep.unverifiable.append(problem)

    for c in candidates:
        c.filters = (c.filters or []) + _threshold_checks(c, t)
        if c.failed:
            rep.rejected.append(c)
        else:
            rep.passed.append(c)

    # фидбек для цикла возврата к Generator'у — конкретный, не «сделай лучше»
    if rep.rejected:
        by_rule: dict[str, int] = {}
        for c in rep.rejected:
            for f in c.failed:
                by_rule[f.name] = by_rule.get(f.name, 0) + 1
        for rule, n in sorted(by_rule.items(), key=lambda x: -x[1]):
            if rule == "LogP":
                rep.feedback.append(f"{n} кандидатов с LogP выше {t.max_logp}: добавляй полярные заместители")
            elif rule == "SA_Score":
                rep.feedback.append(f"{n} кандидатов с SA выше {t.max_sa}: упрощай скелет, меньше новых стереоцентров")
            elif rule == "QED":
                rep.feedback.append(f"{n} кандидатов с QED ниже {t.min_qed}: держи MW в пределах 500")
            elif rule == "dG":
                rep.feedback.append(f"{n} кандидатов с ΔG хуже {t.max_dg}: сохраняй фарма­кофор исходного хита")

    return rep


def check_data_sufficiency(
    n_activities: int,
    has_structure: bool,
    t: Thresholds | None = None,
) -> list[MissingData]:
    """
    Сценарий 3. Вызывается ДО генерации. Если вернулся непустой список —
    ветка генерации не запускается вообще. Это и есть отказ от выдуманного ответа.
    """
    t = t or Thresholds()
    missing: list[MissingData] = []

    if n_activities < t.min_qsar_train:
        missing.append(MissingData(
            what="данные о биологической активности по мишени",
            found=f"{n_activities} записей в ChEMBL",
            needed=f"не менее {t.min_qsar_train} записей одного типа (IC50 или Ki)",
            would_unlock="обучение QSAR-модели и предсказание активности кандидатов",
        ))

    if not has_structure:
        missing.append(MissingData(
            what="экспериментальная 3D-структура мишени",
            found="в PDB подходящей записи не найдено",
            needed="структура с разрешением лучше 3.0 Å, желательно с ко-кристаллизованным лигандом",
            would_unlock="молекулярный докинг и расчёт энергии связывания",
        ))

    return missing


def build_result(
    query: str,
    scenario: str,
    rep: VerificationReport,
    missing: list[MissingData],
    tool_calls: list[ToolCall],
    summary: str = "",
) -> ScenarioResult:
    """Единственная точка сборки ответа. Reporter не собирает ScenarioResult сам."""
    if missing:
        status = "insufficient_data"
        candidates: list[Candidate] = []     # при отказе кандидатов быть не должно
    elif rep.passed:
        status = "ok"
        candidates = sorted(rep.passed, key=lambda c: c.score if c.score is not None else 0.0, reverse=True)
        for i, c in enumerate(candidates, 1):
            c.rank = i
    else:
        status = "no_candidates_passed"
        candidates = []

    limitations: list[str] = []
    if rep.unverifiable:
        limitations.extend(rep.unverifiable)
    unchecked = {f.name for c in (rep.passed + rep.rejected) for f in c.unchecked}
    if unchecked:
        limitations.append(f"не вычислялось и не учитывалось в ранжировании: {', '.join(sorted(unchecked))}")

    return ScenarioResult(
        status=status,
        query=query,
        scenario=scenario,
        candidates=candidates,
        missing=missing,
        tool_calls=tool_calls,
        summary=summary,
        limitations=limitations,
    )
