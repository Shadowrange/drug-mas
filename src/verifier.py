"""
ВЕРИФИКАТОР. ВЛАДЕЛЕЦ: Р-B. НЕ LLM.

Главный аргумент команды на защите. Проверка «не выдумала ли система число»
не может жить в промпте: промпт — это просьба, а здесь нужен запрет.

Три исхода, которые нельзя путать:
  ok                   — кандидаты есть и прошли пороги
  no_candidates_passed — кандидаты были, все провалили пороги (это РЕЗУЛЬТАТ,
                         и их надо ПОКАЗАТЬ с причинами отсева)
  insufficient_data    — данных не хватило, генерацию запускать нельзя (это ОТКАЗ)

Классическая ошибка: считать «значение не посчитано» за «порог не пройден».
Из-за неё система выдаёт уверенный рейтинг по пустым данным.

Ключевая мысль: схема гарантирует НАЛИЧИЕ источника, а не его ПРАВДИВОСТЬ.
Поэтому источник сверяется с журналом вызовов: значение с источником Vina
не принимается, если в этом прогоне не было успешного вызова докинга.
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

# Подстрока в source -> инструмент, который обязан был отработать в этом прогоне.
SOURCE_REQUIRES_TOOL: dict[str, str] = {
    "Vina": "dock",
    "PLIP": "interactions",
    "CatBoost": "qsar.predict",
    "RandomForest": "qsar.predict",
    "CHEMBL": "chembl_activities",
    "PubChem": "pubchem_smiles",
    "UniProt": "uniprot_target",
    "PDB": "pdb_structure",
    "ADMET": "admet.profile",
}

# Без этих величин кандидат не ранжируется: ранг по одному-двум числам —
# это и есть уверенный рейтинг по пустым данным.
REQUIRED_FOR_RANKING = ("LogP", "QED", "SA_Score")


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


def check_provenance(
    candidates: list[Candidate],
    tool_calls: list[ToolCall] | None = None,
    t: Thresholds | None = None,
) -> dict[str, list[str]]:
    """
    Возвращает {smiles: [проблемы]}. Непустой список = кандидата нельзя ранжировать.

    Проверяет четыре вещи:
      1. источник сверен с журналом успешных вызовов инструментов;
      2. прогноз сопровождается осмысленными метриками модели;
      3. гипотеза не используется как число;
      4. источник вообще заполнен.
    """
    t = t or Thresholds()
    succeeded = {c.tool for c in (tool_calls or []) if c.ok}
    problems: dict[str, list[str]] = {}

    def add(smiles: str, msg: str) -> None:
        problems.setdefault(smiles, []).append(msg)

    for c in candidates:
        smi = c.molecule.smiles
        for v in c.molecule.values:
            if not v.source or len(v.source) < 3:
                add(smi, f"'{v.name}' без источника")
                continue

            # 1. Подпись без вызова: строку "AutoDock Vina" может написать кто угодно.
            for marker, tool in SOURCE_REQUIRES_TOOL.items():
                if marker.lower() in v.source.lower() and v.value is not None:
                    if tool_calls is None:
                        add(smi, f"'{v.name}' ссылается на {marker}, но журнал вызовов не передан")
                    elif tool not in succeeded:
                        add(smi, f"'{v.name}' ссылается на {marker}, но успешного вызова {tool}() в этом прогоне не было")
                    break

            # 2. Прогноз с негодными метриками хуже гипотезы: выглядит проверенным.
            if v.provenance is Provenance.PREDICTED:
                m = v.model_metrics
                if m is None:
                    add(smi, f"'{v.name}' — прогноз без метрик модели")
                else:
                    if m.n_train < t.min_qsar_train:
                        add(smi, f"'{v.name}' — модель обучена на {m.n_train} точках при пороге {t.min_qsar_train}")
                    if m.metric_name in {"R2", "ROC-AUC"} and not (0.0 <= m.metric_value <= 1.0):
                        add(smi, f"'{v.name}' — {m.metric_name} = {m.metric_value} вне допустимого диапазона")

            # 3. Гипотеза в роли числа. Проверяем все имена, а не белый список.
            if v.provenance is Provenance.HYPOTHESIS and v.value is not None:
                add(smi, f"'{v.name}' помечено как гипотеза, но несёт значение")

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
    check("LogP", logp, logp is not None and logp <= t.max_logp, f"LogP = {logp}", f"<= {t.max_logp}")

    qed = c.molecule.numeric("QED")
    check("QED", qed, qed is not None and qed >= t.min_qed, f"QED = {qed}", f">= {t.min_qed}")

    sa = c.molecule.numeric("SA_Score")
    check("SA_Score", sa, sa is not None and sa <= t.max_sa, f"SA = {sa}", f"<= {t.max_sa}")

    dg = c.molecule.numeric("dG")
    check("dG", dg, dg is not None and dg <= t.max_dg, f"ΔG = {dg} ккал/моль", f"<= {t.max_dg}")

    return out


def verify(
    candidates: list[Candidate],
    tool_calls: list[ToolCall] | None = None,
    t: Thresholds | None = None,
) -> VerificationReport:
    t = t or Thresholds()
    rep = VerificationReport()
    problems = check_provenance(candidates, tool_calls, t)

    for c in candidates:
        smi = c.molecule.smiles
        c.filters = (c.filters or []) + _threshold_checks(c, t)

        # Претензия к происхождению ДИСКВАЛИФИЦИРУЕТ кандидата, а не уходит
        # в примечание мелким шрифтом под рейтингом, где он стоит первым.
        if smi in problems:
            rep.unverifiable.extend(f"{smi}: {p}" for p in problems[smi])
            c.filters.append(FilterResult(
                name="Происхождение данных", passed=False,
                detail="; ".join(problems[smi]), threshold="источник подтверждён журналом вызовов",
            ))
            rep.rejected.append(c)
            continue

        # Кандидат, у которого не посчитано НИЧЕГО, не «прошёл фильтры» —
        # его просто нечем оценивать. Без этой ветки он получает rank=1.
        missing_required = [n for n in REQUIRED_FOR_RANKING if c.molecule.numeric(n) is None]
        if missing_required:
            msg = f"не посчитано обязательное для ранжирования: {', '.join(missing_required)}"
            rep.unverifiable.append(f"{smi}: {msg}")
            c.filters.append(FilterResult(
                name="Полнота данных", passed=False, detail=msg,
                threshold=f"обязательны: {', '.join(REQUIRED_FOR_RANKING)}",
            ))
            rep.rejected.append(c)
            continue

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
        hints = {
            "LogP": f"добавляй полярные заместители (порог {t.max_logp})",
            "SA_Score": f"упрощай скелет, меньше новых стереоцентров (порог {t.max_sa})",
            "QED": f"держи MW в пределах 500 (порог {t.min_qed})",
            "dG": f"сохраняй фармакофор исходного хита (порог {t.max_dg})",
        }
        for rule, n in sorted(by_rule.items(), key=lambda x: -x[1]):
            if rule in hints:
                rep.feedback.append(f"{n} кандидатов отсеяно по {rule}: {hints[rule]}")

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
    limitations: list[str] = list(rep.unverifiable)

    if missing:
        status = "insufficient_data"
        candidates: list[Candidate] = []     # при отказе кандидатов быть не должно
    elif rep.passed:
        status = "ok"
        candidates = sorted(rep.passed, key=lambda c: c.score if c.score is not None else 0.0, reverse=True)
        for i, c in enumerate(candidates, 1):
            c.rank = i
    else:
        # Отсеянных ПОКАЗЫВАЕМ: «все кандидаты провалили пороги» — это результат
        # с причинами, а не пустой экран, неотличимый от отказа.
        status = "no_candidates_passed"
        candidates = rep.rejected
        for c in candidates:
            c.rank = None

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
