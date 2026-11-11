"""
QSAR по выгрузке ChEMBL. ВЛАДЕЛЕЦ: Р-A. Блок T4.

Метрики модели сохранять ОБЯЗАТЕЛЬНО: без них predicted-значение
не пройдёт валидацию в schemas.Value.
"""
from __future__ import annotations

from src.schemas import ModelMetrics, Value


def train(smiles: list[str], y: list[float]) -> tuple[object, ModelMetrics]:
    """TODO Р-A: ECFP4 (chem.morgan_fp) -> CatBoost/RandomForest, split 80/20, вернуть (модель, метрики)."""
    raise NotImplementedError


def predict(model: object, metrics: ModelMetrics, smiles: str) -> Value:
    """TODO Р-A: schemas.predicted('pIC50', value, source=..., metrics=metrics)."""
    raise NotImplementedError
