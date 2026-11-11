"""
Тесты детерминированного слоя. ВЛАДЕЛЕЦ: Р-A.

Запуск:  python -m pytest -q
Эти тесты — страховка от того, что правка в chem.py тихо сломает Evaluator в 3 часа ночи.
"""

import pytest

from src.schemas import ModelMetrics, Provenance, Value, computed, predicted
from src.tools import chem
from src.verifier import Thresholds, check_data_sufficiency

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
BROKEN = "C1CC"


# --- схемы: запрет на выдуманные значения ---

def test_value_requires_source():
    with pytest.raises(Exception):
        Value(name="dG", value=-9.1, provenance=Provenance.COMPUTED, source="")


def test_predicted_requires_metrics():
    """Прогноз без метрик качества модели не является аргументом."""
    with pytest.raises(Exception):
        Value(name="pIC50", value=7.2, provenance=Provenance.PREDICTED, source="CatBoost")


def test_predicted_with_metrics_ok():
    m = ModelMetrics(model_name="CatBoost", n_train=842, metric_name="R2", metric_value=0.71)
    v = predicted("pIC50", 7.2, "CatBoost на ChEMBL CHEMBL203", m)
    assert v.model_metrics.metric_value == 0.71


def test_hypothesis_cannot_be_number():
    with pytest.raises(Exception):
        Value(name="dG", value=-9.1, provenance=Provenance.HYPOTHESIS, source="рассуждение LLM")


# --- химия ---

def test_broken_smiles_returns_none():
    """Невалидный SMILES даёт None, а не заглушку с нулями."""
    assert chem.describe(BROKEN) is None
    assert chem.parse(BROKEN) is None


def test_aspirin_descriptors():
    m = chem.describe(ASPIRIN, name="аспирин")
    assert m is not None
    assert 180 <= m.numeric("MW") <= 181
    assert m.numeric("HBD") == 1
    assert all(v.provenance is Provenance.COMPUTED for v in m.values)
    assert all(v.source for v in m.values)


def test_imatinib_descriptors():
    m = chem.describe(IMATINIB, name="иматиниб")
    assert m is not None
    assert 493 <= m.numeric("MW") <= 494
    assert m.numeric("LogP") > 4
    assert m.inchikey


def test_filters():
    assert chem.lipinski(ASPIRIN).passed is True
    assert chem.pains(ASPIRIN).passed is True
    assert chem.sa_ok(ASPIRIN).passed is True
    assert chem.lipinski(BROKEN).passed is None     # None = не проверялось, не «провалено»


def test_embed_3d():
    pdb = chem.embed_3d(ASPIRIN)
    # RDKit пишет лиганд как HETATM, не ATOM — это нормально для подготовки к докингу
    assert pdb and "HETATM" in pdb and "END" in pdb


def test_morgan_fp():
    assert chem.morgan_fp(IMATINIB).shape == (2048,)


# --- верификатор: отказ вместо выдумки ---

def test_insufficient_data_detected():
    missing = check_data_sufficiency(n_activities=4, has_structure=False, t=Thresholds())
    assert len(missing) == 2
    assert all(md.would_unlock for md in missing)


def test_sufficient_data_passes():
    assert check_data_sufficiency(n_activities=800, has_structure=True) == []
