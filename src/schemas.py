"""
КОНТРАКТ ДАННЫХ ВСЕЙ СИСТЕМЫ. ВЛАДЕЛЕЦ: Р-B. ЗАМОРОЖЕН.

Это несущая конструкция требования №5 из ТЗ: система обязана прозрачно
различать реальные расчёты, предсказания и гипотезы.

Проверка живёт ЗДЕСЬ, в типах, а не в промпте агента. Агент физически
не может вернуть число без указания происхождения — Pydantic не даст.

Менять этот файл может только Р-B и только после поста [decision] на доску.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provenance(str, Enum):
    """Откуда взялось значение. Четыре категории, пятой не бывает."""

    DATABASE = "database"      # запись из ChEMBL / PubChem / PDB / UniProt
    COMPUTED = "computed"      # детерминированный расчёт: RDKit, AutoDock Vina
    PREDICTED = "predicted"    # прогноз ML-модели, обязаны быть метрики модели
    HYPOTHESIS = "hypothesis"  # рассуждение LLM, числом не подкреплено

    @property
    def badge(self) -> str:
        """Для отображения во фронте."""
        return {
            "database": "🗄 БД",
            "computed": "🧮 РАСЧЁТ",
            "predicted": "📈 ПРОГНОЗ",
            "hypothesis": "💭 ГИПОТЕЗА",
        }[self.value]


class ModelMetrics(BaseModel):
    """Качество ML-модели. Без этого предсказание не является аргументом."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    n_train: int
    metric_name: str              # "R2" | "ROC-AUC" | "RMSE"
    metric_value: float
    split: str = "random 80/20"


class Value(BaseModel):
    """
    Любое число или факт в системе. Нет provenance — нет значения.

    value=None означает «не посчитано». Это НЕ то же самое, что «не прошло
    порог», и верификатор обязан различать эти два случая — см. verifier.py.
    """

    model_config = ConfigDict(use_enum_values=False)

    name: str = Field(min_length=1)
    value: float | str | bool | None
    units: str | None = None
    provenance: Provenance
    source: str = Field(min_length=3, description="CHEMBL25 | RDKit 2026.09 Descriptors.MolLogP | AutoDock Vina 1.2.5")
    model_metrics: ModelMetrics | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Value:
        if self.provenance is Provenance.PREDICTED and self.model_metrics is None:
            raise ValueError(
                f"'{self.name}': provenance=PREDICTED требует model_metrics. "
                "Предсказание без метрик качества модели не является аргументом."
            )
        if self.provenance is Provenance.HYPOTHESIS and isinstance(self.value, (int, float)):
            raise ValueError(
                f"'{self.name}': гипотеза не может быть числом. "
                "Число приходит из инструмента или не приходит вообще."
            )
        return self

    def __str__(self) -> str:
        v = "—" if self.value is None else self.value
        u = f" {self.units}" if self.units and self.value is not None else ""
        return f"{self.name}: {v}{u} [{self.provenance.badge} · {self.source}]"


class Molecule(BaseModel):
    """Молекула. smiles валидируется RDKit'ом в tools/chem.py, не здесь."""

    smiles: str = Field(min_length=1)
    name: str | None = None
    inchikey: str | None = None
    source: str = Field(min_length=3, description="PubChem CID 5291 | сгенерировано CReM из <parent>")
    values: list[Value] = Field(default_factory=list)

    def get(self, name: str) -> Value | None:
        return next((v for v in self.values if v.name == name), None)

    def numeric(self, name: str) -> float | None:
        v = self.get(name)
        return v.value if v and isinstance(v.value, (int, float)) else None


class FilterResult(BaseModel):
    name: str
    passed: bool | None            # None = не проверялось, данных нет
    detail: str
    threshold: str | None = None


class Candidate(BaseModel):
    """Молекула-кандидат после прохождения Evaluator."""

    molecule: Molecule
    parent_smiles: str | None = None
    filters: list[FilterResult] = Field(default_factory=list)
    score: float | None = None
    rank: int | None = None

    @property
    def failed(self) -> list[FilterResult]:
        return [f for f in self.filters if f.passed is False]

    @property
    def unchecked(self) -> list[FilterResult]:
        return [f for f in self.filters if f.passed is None]


class MissingData(BaseModel):
    """Чего не хватило. Ядро Сценария 3 — честного отказа."""

    what: str                      # "активности по мишени в ChEMBL"
    found: str                     # "4 записи IC50"
    needed: str                    # "не менее 100 записей одного типа"
    would_unlock: str              # "обучение QSAR и предсказание активности"


class ToolCall(BaseModel):
    """Лог вызова инструмента. Показывается во фронте как цепочка рассуждений."""

    agent: str
    tool: str
    args_summary: str
    ok: bool
    from_cache: bool = False
    duration_s: float | None = None
    error: str | None = None


class ScenarioResult(BaseModel):
    """Итоговый ответ системы пользователю."""

    status: Literal["ok", "insufficient_data", "no_candidates_passed"]
    query: str
    scenario: str
    candidates: list[Candidate] = Field(default_factory=list)
    missing: list[MissingData] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    summary: str = Field(default="", description="Текст Reporter'а. Это ИНТЕРПРЕТАЦИЯ, не данные.")
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ScenarioResult:
        if self.status == "insufficient_data":
            if not self.missing:
                raise ValueError("status=insufficient_data требует непустой missing[]: надо сказать, чего именно не хватило.")
            if self.candidates:
                raise ValueError("status=insufficient_data с кандидатами — это и есть выдуманный результат. Запрещено.")
        if self.status == "ok" and not self.candidates:
            raise ValueError("status=ok без кандидатов. Используй no_candidates_passed.")
        return self


# --- хелперы, чтобы заполнение provenance было короче, чем его пропуск ---

def computed(name: str, value: Any, source: str, units: str | None = None, note: str | None = None) -> Value:
    return Value(name=name, value=value, units=units, provenance=Provenance.COMPUTED, source=source, note=note)


def from_db(name: str, value: Any, source: str, units: str | None = None, note: str | None = None) -> Value:
    return Value(name=name, value=value, units=units, provenance=Provenance.DATABASE, source=source, note=note)


def predicted(name: str, value: Any, source: str, metrics: ModelMetrics, units: str | None = None) -> Value:
    return Value(name=name, value=value, units=units, provenance=Provenance.PREDICTED, source=source, model_metrics=metrics)


def not_computed(name: str, reason: str, source: str = "не вычислялось") -> Value:
    """Явное «не посчитали». Лучше, чем отсутствие поля: видно, что пропуск осознанный."""
    return Value(name=name, value=None, provenance=Provenance.COMPUTED, source=source, note=reason)
