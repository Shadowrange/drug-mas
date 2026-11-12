"""Обучение и инференс регрессионных QSAR-моделей по экспериментальным выборкам ChEMBL.

Обучает `CatBoostRegressor` (с автоматическим переключением на `RandomForestRegressor` при
отсутствии пакета `catboost`) на битовых векторах фингерпринтов Morgan ECFP4 (2048 бит)
и сопровождает каждое предсказание обязательным объектом `ModelMetrics` (`R2`, `n_train`, `split`).
"""
from __future__ import annotations

import math
from statistics import median
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.schemas import ModelMetrics, Value, not_computed, predicted
from src.tools import chem

try:
    from catboost import CatBoostRegressor  # type: ignore

    _CATBOOST_OK = True
except ImportError:
    CatBoostRegressor = None
    _CATBOOST_OK = False

MIN_VALID_SAMPLES = 10


def ic50_nm_to_pic50(ic50_nm: float) -> float | None:
    """Перевод IC50 (нМ) в pIC50 = 9 - log10(IC50 [nM])."""
    if not isinstance(ic50_nm, (int, float)) or not math.isfinite(float(ic50_nm)) or ic50_nm <= 0:
        return None
    return round(9.0 - math.log10(float(ic50_nm)), 4)


def train(smiles: list[str], y: list[float]) -> tuple[object, ModelMetrics]:
    """
    Обучение регрессионной QSAR-модели на фингерпринтах ECFP4 (Morgan radius=2, 2048 бит).
    Разбиение 80/20, расчёт R² на отложенной выборке и обязательная упаковка в ModelMetrics.
    """
    if len(smiles) != len(y):
        raise ValueError(f"Размерности smiles ({len(smiles)}) и y ({len(y)}) не совпадают")

    fps: list[np.ndarray] = []
    targets: list[float] = []
    for s, val in zip(smiles, y):
        if val is None or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            continue
        fp = chem.morgan_fp(s, radius=2, n_bits=2048)
        if fp is None:
            continue
        fps.append(fp)
        targets.append(float(val))

    if len(fps) < MIN_VALID_SAMPLES:
        raise ValueError(
            f"Недостаточно валидных молекул для обучения QSAR: получено {len(fps)}, "
            f"требуется минимум {MIN_VALID_SAMPLES}"
        )

    X = np.vstack(fps)
    y_arr = np.asarray(targets, dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=0.2, random_state=42
    )

    if _CATBOOST_OK and CatBoostRegressor is not None:
        model: Any = CatBoostRegressor(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            loss_function="RMSE",
            random_seed=42,
            thread_count=2,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(X_train, y_train)
        model_name = "CatBoostRegressor (ECFP4 2048)"
    else:
        model = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        model_name = "RandomForestRegressor (ECFP4 2048)"

    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else 0.0
    if not math.isfinite(r2):
        r2 = 0.0

    metrics = ModelMetrics(
        model_name=model_name,
        n_train=int(len(X_train)),
        metric_name="R2",
        metric_value=round(r2, 3),
        split="random 80/20",
    )
    return model, metrics


def train_from_chembl(records: list[dict[str, Any]]) -> tuple[object, ModelMetrics]:
    """
    Вспомогательная функция для обучения QSAR напрямую по выгрузке databases.chembl_activities:
    переводит IC50 (нМ) в pIC50 и агрегирует дубликаты по каноническому SMILES медианой.
    """
    by_smiles: dict[str, list[float]] = {}
    for rec in records:
        raw_smi = rec.get("smiles")
        can_smi = chem.canonical(raw_smi) if isinstance(raw_smi, str) else None
        if not can_smi:
            continue

        if "pIC50" in rec and rec["pIC50"] is not None:
            try:
                pic50 = float(rec["pIC50"])
            except (TypeError, ValueError):
                continue
        else:
            try:
                raw_val = float(rec.get("value"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            pic50_opt = ic50_nm_to_pic50(raw_val)
            if pic50_opt is None:
                continue
            pic50 = pic50_opt

        if math.isfinite(pic50):
            by_smiles.setdefault(can_smi, []).append(pic50)

    clean_smiles = list(by_smiles.keys())
    clean_y = [float(median(vals)) for vals in by_smiles.values()]
    return train(clean_smiles, clean_y)


def predict(model: object, metrics: ModelMetrics, smiles: str) -> Value:
    """
    Инференс обученной QSAR-модели с обязательным прикреплением ModelMetrics.
    При невалидном SMILES возвращает schemas.not_computed('pIC50', ...).
    """
    if model is None or metrics is None:
        return not_computed("pIC50", "QSAR-модель или её метрики не переданы", source="QSAR")

    fp = chem.morgan_fp(smiles, radius=2, n_bits=2048)
    if fp is None:
        return not_computed("pIC50", "Невалидный SMILES для расчёта ECFP4", source=metrics.model_name)

    try:
        raw_pred = model.predict(fp.reshape(1, -1))
        val = float(np.asarray(raw_pred).ravel()[0])
    except Exception as exc:
        return not_computed("pIC50", f"Ошибка инференса QSAR: {exc}", source=metrics.model_name)

    source_str = (
        f"{metrics.model_name} (n_train={metrics.n_train}, "
        f"{metrics.metric_name}={metrics.metric_value:.3f}, split={metrics.split})"
    )
    return predicted("pIC50", round(val, 2), source=source_str, metrics=metrics)

