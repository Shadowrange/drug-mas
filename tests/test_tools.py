"""
Тесты детерминированного слоя. ВЛАДЕЛЕЦ: Р-A.

Запуск:  python -m pytest -q
Эти тесты — страховка от того, что правка в chem.py тихо сломает Evaluator в 3 часа ночи.
"""

import pytest

from src.schemas import ModelMetrics, Molecule, Provenance, Value, computed, predicted
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


# --- дыры, найденные при ревью: каждая закрыта тестом ---

from src.schemas import Candidate, Provenance, ToolCall, Value, computed, predicted
from src.verifier import build_result, check_provenance, verify


def _cand(smiles=ASPIRIN, extra=None):
    m = chem.describe(smiles, name="test")
    if extra:
        m.values.extend(extra)
    return Candidate(molecule=m, filters=chem.run_all_filters(smiles), score=0.7)


def test_hypothesis_as_string_number_rejected():
    """Д4: '-9.1' строкой — та же выдумка, только мимо проверки типа."""
    with pytest.raises(Exception):
        Value(name="ΔG", value="-9.1", provenance=Provenance.HYPOTHESIS, source="рассуждение LLM")


def test_signed_without_call_is_disqualified():
    """Д1: подпись 'AutoDock Vina' без успешного вызова докинга не принимается."""
    c = _cand(extra=[computed("dG", -9.4, "AutoDock Vina 1.2.5", "ккал/моль")])
    rep = verify([c], tool_calls=[])          # докинга в прогоне не было
    assert rep.passed == [] and rep.rejected == [c]
    assert any("успешного вызова dock()" in p for p in rep.unverifiable)


def test_signed_with_call_is_accepted():
    c = _cand(extra=[computed("dG", -9.4, "AutoDock Vina 1.2.5", "ккал/моль")])
    rep = verify([c], tool_calls=[ToolCall(agent="Evaluator", tool="dock", args_summary="1IEP", ok=True)])
    assert rep.passed == [c]


def test_unverifiable_blocks_ranking():
    """Д2: претензия к происхождению дисквалифицирует, а не уходит в сноску."""
    c = _cand(extra=[computed("dG", -9.4, "AutoDock Vina 1.2.5", "ккал/моль")])
    rep = verify([c], tool_calls=[])
    r = build_result("q", "lead_optimization", rep, [], [])
    assert r.status == "no_candidates_passed"
    assert all(cc.rank is None for cc in r.candidates)


def test_candidate_without_numbers_not_ranked():
    """Д3: нечего считать — нечего ранжировать."""
    bare = Candidate(molecule=Molecule(smiles=ASPIRIN, source="тест"), filters=[], score=None)
    rep = verify([bare], tool_calls=[])
    assert rep.passed == []
    assert any("обязательное для ранжирования" in p for p in rep.unverifiable)


def test_qsar_on_tiny_sample_rejected():
    """Д5: R2=0.99 на 12 точках выглядит проверенным — и потому опаснее гипотезы."""
    m = ModelMetrics(model_name="CatBoost", n_train=12, metric_name="R2", metric_value=0.99)
    c = _cand(extra=[predicted("pIC50", 8.4, "CatBoost на CHEMBL203", m)])
    problems = check_provenance([c], tool_calls=[ToolCall(agent="E", tool="qsar.predict", args_summary="", ok=True)])
    assert any("12 точках" in p for p in problems[c.molecule.smiles])


def test_rejected_candidates_are_shown():
    """«Все провалили пороги» — это результат с причинами, а не пустой экран."""
    c = _cand("CCCCCCCCCCCCCCCCCCCCc1ccccc1")   # заведомо высокий LogP
    rep = verify([c], tool_calls=[])
    r = build_result("q", "lead_optimization", rep, [], [])
    assert r.status == "no_candidates_passed" and len(r.candidates) == 1
    assert r.candidates[0].failed
