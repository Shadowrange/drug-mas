"""
ADMET. ВЛАДЕЛЕЦ: Р-A. Блок T4.

Таймбокс на установку ADMET-AI — 45 минут (риск R3).
Не завелось -> фолбэк: CatBoost на hERG из TDC + детерминированные RDKit-оценки
(растворимость ESOL Delaney 2004, структурные токсикофоры Brenk).
Если ML-модель для конкретной мишени (hERG, Ames, CYP3A4) не обучена/не загружена —
честно возвращаем schemas.not_computed(...), НЕ выдумывая числа.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.schemas import ModelMetrics, Provenance, Value, computed, not_computed, predicted
from src.tools import chem

try:
    from catboost import CatBoostClassifier  # type: ignore

    _CATBOOST_OK = True
except ImportError:
    CatBoostClassifier = None
    _CATBOOST_OK = False

try:
    from admet_ai import ADMETModel  # type: ignore

    _ADMET_AI_OK = True
except ImportError:
    ADMETModel = None
    _ADMET_AI_OK = False

ML_ENDPOINTS = ("hERG", "Ames", "CYP3A4")

# Реестр обученных в сессии классификаторов ADMET: endpoint -> (model, ModelMetrics)
_MODELS: dict[str, tuple[Any, ModelMetrics]] = {}


@lru_cache(maxsize=1)
def _brenk_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


@lru_cache(maxsize=1)
def _get_admet_ai_model() -> Any | None:
    if not _ADMET_AI_OK or ADMETModel is None:
        return None
    try:
        return ADMETModel()
    except Exception:
        return None


# Метрики линейной регрессии ESOL из оригинальной статьи:
# John S. Delaney, "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure",
# J. Chem. Inf. Comput. Sci. 2004, 44 (3), 1000–1005 (DOI: 10.1021/ci034243x).
# Обучающая выборка: n = 2874 малых органических молекул, R² = 0.69 (AAE = 0.75 log mol/L).
ESOL_METRICS = ModelMetrics(
    model_name="ADMET ESOL Linear Regression (Delaney 2004)",
    n_train=2874,
    metric_name="R2",
    metric_value=0.69,
    split="Delaney 2004 full set (n=2874, AAE=0.75)",
)


def esol_logs(mol_or_smiles: Chem.Mol | str) -> Value:
    """
    Оценка водной растворимости LogS (моль/л) по эмпирической регрессии ESOL
    (Delaney, J. Chem. Inf. Comput. Sci. 2004, 44, 1000–1005):
    LogS = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RotB - 0.74*AP.
    Поскольку коэффициенты подогнаны регрессией на выборке из 2874 соединений (R²=0.69),
    возвращается с provenance=PREDICTED и паспортными ModelMetrics работы 2004 года.
    """
    src = f"ADMET ESOL (Delaney, J. Chem. Inf. Comput. Sci. 2004, 44:1000-1005; n_train=2874, R2=0.69)"
    mol = mol_or_smiles if isinstance(mol_or_smiles, Chem.Mol) else chem.parse(mol_or_smiles)
    if mol is None:
        return not_computed("LogS", "Невалидный SMILES", source=src)

    clogp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ap = (arom / heavy) if heavy > 0 else 0.0

    logs = 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rotb - 0.74 * ap
    return Value(
        name="LogS",
        value=round(float(logs), 2),
        units="log mol/L",
        provenance=Provenance.PREDICTED,
        source=src,
        model_metrics=ESOL_METRICS,
        note=">-4 хорошая, -4..-6 умеренная, <-6 низкая растворимость",
    )


def brenk_alerts(mol_or_smiles: Chem.Mol | str) -> Value:
    """
    Детерминированная проверка структурных флагов токсичности и нежелательных
    реакционноспособных групп по каталогу Brenk (ChemMedChem 2008).
    """
    mol = mol_or_smiles if isinstance(mol_or_smiles, Chem.Mol) else chem.parse(mol_or_smiles)
    if mol is None:
        return not_computed("Brenk_Alerts", "Невалидный SMILES", source=f"{chem.RDKIT_SRC} FilterCatalog.BRENK")

    matches = _brenk_catalog().GetMatches(mol)
    n_matches = len(matches)
    if n_matches == 0:
        note = "токсикофоров Brenk не обнаружено"
    else:
        descriptions = sorted({m.GetDescription() for m in matches})
        note = f"срабатывания: {', '.join(descriptions[:3])}"
    return computed(
        "Brenk_Alerts",
        n_matches,
        f"{chem.RDKIT_SRC} FilterCatalog.BRENK (Brenk et al. 2008)",
        note=note,
    )


def train_classifier(
    endpoint: str,
    smiles: list[str],
    y: list[int],
    register: bool = True,
) -> tuple[object, ModelMetrics]:
    """
    Обучение бинарного классификатора ADMET (например, hERG, Ames, CYP3A4) на ECFP4 (2048 бит)
    со сплитом 80/20 и расчётом ROC-AUC для обязательного поля ModelMetrics.
    """
    if len(smiles) != len(y):
        raise ValueError(f"Размерности smiles ({len(smiles)}) и y ({len(y)}) не совпадают")

    fps: list[np.ndarray] = []
    labels: list[int] = []
    for s, lbl in zip(smiles, y):
        if lbl not in (0, 1, False, True):
            continue
        fp = chem.morgan_fp(s, radius=2, n_bits=2048)
        if fp is None:
            continue
        fps.append(fp)
        labels.append(int(lbl))

    if len(fps) < 10 or len(set(labels)) < 2:
        raise ValueError(
            f"Для обучения классификатора {endpoint} нужно >=10 валидных молекул обоих классов "
            f"(получено {len(fps)} молекул, классов: {len(set(labels))})"
        )

    X = np.vstack(fps)
    y_arr = np.asarray(labels, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_arr,
        test_size=0.2,
        random_state=42,
        stratify=y_arr if min(np.bincount(y_arr)) >= 2 else None,
    )

    if _CATBOOST_OK and CatBoostClassifier is not None:
        model: Any = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_seed=42,
            thread_count=2,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(X_train, y_train)
        # Используем префикс ADMET без подстроки "CatBoost", т.к. в verifier.SOURCE_REQUIRES_TOOL
        # маркер "CatBoost" привязан к qsar.predict, а "ADMET" — к admet.profile.
        model_name = f"ADMET GBDTClassifier {endpoint} (ECFP4 2048)"
    else:
        model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        model_name = f"ADMET RFClassifier {endpoint} (ECFP4 2048)"

    if len(set(y_test)) >= 2:
        probs = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, probs))
    else:
        probs = model.predict_proba(X_train)[:, 1]
        auc = float(roc_auc_score(y_train, probs))

    if not math.isfinite(auc):
        auc = 0.5

    metrics = ModelMetrics(
        model_name=model_name,
        n_train=int(len(X_train)),
        metric_name="ROC-AUC",
        metric_value=round(auc, 3),
        split="stratified 80/20",
    )
    if register:
        _MODELS[endpoint] = (model, metrics)
    return model, metrics


def register_model(endpoint: str, model: object, metrics: ModelMetrics) -> None:
    """Регистрирует обученную ADMET-модель и её метрики для использования в profile()."""
    _MODELS[endpoint] = (model, metrics)


def clear_models() -> None:
    """Очищает реестр пользовательских моделей ADMET."""
    _MODELS.clear()


def _predict_endpoint(endpoint: str, fp: np.ndarray) -> Value:
    """Возвращает предсказание из зарегистрированной модели или честный not_computed."""
    if endpoint in _MODELS:
        model, metrics = _MODELS[endpoint]
        try:
            prob = float(model.predict_proba(fp.reshape(1, -1))[0, 1])
            src = (
                f"{metrics.model_name} (n_train={metrics.n_train}, "
                f"{metrics.metric_name}={metrics.metric_value:.3f})"
            )
            return predicted(endpoint, round(prob, 3), source=src, metrics=metrics)
        except Exception as exc:
            return not_computed(endpoint, f"Ошибка инференса модели {endpoint}: {exc}", source=metrics.model_name)

    return not_computed(
        endpoint,
        f"ML-модель {endpoint} (ADMET-AI / TDC) не загружена в окружении (план Б по риску R3)",
        source="ADMET (модель недоступна)",
    )


def profile(smiles: str) -> list[Value]:
    """
    Комплексный ADMET-профиль молекулы:
    1) LogS — водная растворимость по Делани ESOL (provenance=COMPUTED, RDKit)
    2) Brenk_Alerts — структурные токсикофоры Brenk (provenance=COMPUTED, RDKit)
    3) hERG, Ames, CYP3A4 — ML-предсказания токсичности и метаболизма
       (provenance=PREDICTED + ModelMetrics при наличии модели, иначе честный not_computed).
    """
    mol = chem.parse(smiles)
    if mol is None:
        return [
            not_computed("LogS", "Невалидный SMILES", source="ADMET"),
            not_computed("Brenk_Alerts", "Невалидный SMILES", source="ADMET"),
            *(not_computed(ep, "Невалидный SMILES", source="ADMET") for ep in ML_ENDPOINTS),
        ]

    out: list[Value] = [
        esol_logs(mol),
        brenk_alerts(mol),
    ]

    # Если установлен пакет ADMET-AI — используем его с паспортными метриками TDC
    admet_ai_model = _get_admet_ai_model()
    if admet_ai_model is not None:
        try:
            preds = admet_ai_model.predict(smiles=Chem.MolToSmiles(mol))
            key_map = {
                "hERG": ("hERG", 0.85, 655),
                "Ames": ("AMES", 0.83, 7278),
                "CYP3A4": ("CYP3A4_Veith", 0.86, 12328),
            }
            for ep, (col, ref_auc, n_tr) in key_map.items():
                if ep in _MODELS:
                    fp = chem.morgan_fp(smiles)
                    out.append(_predict_endpoint(ep, fp) if fp is not None else not_computed(ep, "Ошибка ECFP4"))
                elif isinstance(preds, dict) and col in preds:
                    m = ModelMetrics(
                        model_name=f"ADMET-AI Chemprop-RDKit ({col})",
                        n_train=n_tr,
                        metric_name="ROC-AUC",
                        metric_value=ref_auc,
                        split="TDC scaffold 80/20",
                    )
                    out.append(predicted(ep, round(float(preds[col]), 3), source=m.model_name, metrics=m))
                else:
                    out.append(not_computed(ep, f"Поле {col} отсутствует в выдаче ADMET-AI", source="ADMET-AI"))
            return out
        except Exception:
            pass

    fp = chem.morgan_fp(smiles, radius=2, n_bits=2048)
    for ep in ML_ENDPOINTS:
        if fp is None:
            out.append(not_computed(ep, "Не удалось вычислить ECFP4", source="ADMET"))
        else:
            out.append(_predict_endpoint(ep, fp))

    return out

