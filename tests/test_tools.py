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


# --- докинг (блок D1 + план Б по R1) ---

SAMPLE_RECEPTOR_PDB = """\
ATOM      1  N   ILE A 313      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ILE A 313      11.200  10.000  10.000  1.00 20.00           C
ATOM      3  CB  ILE A 313      11.800  11.400  10.000  1.00 20.00           C
ATOM      4  CG1 ILE A 313      12.200  11.800  11.400  1.00 20.00           C
ATOM      5  OG1 THR A 315      14.000  12.000  12.000  1.00 18.00           O
HETATM   50  C1  STI A 901      15.000  12.000  12.000  1.00 15.00           C
HETATM   51  N1  STI A 901      17.000  14.000  16.000  1.00 15.00           N
HETATM   99  O   HOH A 999      50.000  50.000  50.000  1.00 30.00           O
END
"""

SAMPLE_POSE_PDB = """\
HETATM    1  O1  LIG A   1      16.800  12.000  12.000  1.00  0.00           O
HETATM    2  C1  LIG A   1      15.700  11.800  11.400  1.00  0.00           C
END
"""


def test_docking_prepare_receptor_and_box_center(tmp_path):
    from src.tools import docking

    pdb_file = tmp_path / "1iep_sample.pdb"
    pdb_file.write_text(SAMPLE_RECEPTOR_PDB, encoding="utf-8")

    center = docking.extract_box_center(str(pdb_file), ligand_resname="STI")
    assert center == (16.0, 13.0, 14.0)

    pdbqt_path = docking.prepare_receptor(str(pdb_file))
    assert pdbqt_path.endswith(".pdbqt")
    content = (tmp_path / "1iep_sample.pdbqt").read_text(encoding="utf-8")
    assert "ATOM" in content
    assert "HOH" not in content
    assert "STI" not in content


def test_docking_dock_honest_provenance():
    from src.tools import docking

    pdbqt_str = docking.prepare_receptor(SAMPLE_RECEPTOR_PDB)
    res_broken = docking.dock(BROKEN, pdbqt_str, center=(16.0, 13.0, 14.0))
    assert res_broken.name == "dG"
    assert res_broken.value is None

    # Без Meeko молекула с вращаемыми связями (иматиниб, RotB=7) НЕ должна конвертироваться
    # в жёсткий PDBQT с TORSDOF 0, иначе Vina вернёт энергию случайного конформера.
    if not docking._MEEKO_OK:
        assert docking._ligand_to_pdbqt(IMATINIB) == (None, None)
        res_ima = docking.dock(IMATINIB, pdbqt_str, center=(16.0, 13.0, 14.0))
        assert res_ima.value is None
        assert "вращаемых связей" in (res_ima.note or "")

    # Жёсткая молекула без вращаемых связей (бензол, RotB=0) корректно собирается с TORSDOF 0
    bz_pdbqt, bz_pdb = docking._ligand_to_pdbqt("c1ccccc1")
    assert bz_pdbqt is not None and "TORSDOF 0" in bz_pdbqt and bz_pdb is not None

    res_valid = docking.dock("c1ccccc1", pdbqt_str, center=(16.0, 13.0, 14.0))
    assert res_valid.name == "dG"
    assert res_valid.provenance is Provenance.COMPUTED
    assert len(res_valid.source) >= 3

    # Если Vina не установлена (план Б / R1), value=None не должен валить кандидата по порогу dG
    mol = chem.describe(ASPIRIN, name="аспирин")
    mol.values.append(res_valid)
    cand = Candidate(molecule=mol)
    calls = [ToolCall(agent="Evaluator", tool="dock", args_summary="1IEP", ok=res_valid.value is not None)]
    assert check_provenance([cand], tool_calls=calls) == {}
    rep = verify([cand], tool_calls=calls)
    if res_valid.value is None:
        assert len(rep.passed) == 1
        assert any(f.name == "dG" and f.passed is None for f in cand.unchecked)


def test_docking_interactions():
    from src.tools import docking

    contacts = docking.interactions(SAMPLE_POSE_PDB, SAMPLE_RECEPTOR_PDB)
    assert len(contacts) >= 2
    assert any("THR315" in c and "Водородная связь" in c for c in contacts)
    assert any("ILE313" in c and "Гидрофобный контакт" in c for c in contacts)
    assert docking.interactions("", SAMPLE_RECEPTOR_PDB) == []


# --- QSAR (блок T4) ---

def test_qsar_train_and_predict():
    from src.tools import qsar

    assert qsar.ic50_nm_to_pic50(10.0) == 8.0
    assert qsar.ic50_nm_to_pic50(1000.0) == 6.0
    assert qsar.ic50_nm_to_pic50(-5.0) is None

    # Набор реальных структур (ароматические ингибиторы vs малые алифатические молекулы)
    train_smiles = [
        IMATINIB,
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",  # гефитиниб
        "COCCOC1=C(OCCOC)C=C2C(NC3=CC=CC(C#C)=C3)=NC=NC2=C1",  # эрлотиниб
        "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",  # лапатиниб
        "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",  # афатиниб
        "COc1cc(Nc2nccc(Nc3ccc4c(C)n[nH]c4c3)n2)cc(OC)c1OC",
        "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",  # дазатиниб
        "Cc1ccc(C(=O)Nc2ccc(CN3CCN(C)CC3)c(C(F)(F)F)c2)cc1C#Cc1cnc2cccnn12",  # понатиниб
        "COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1",  # кабозантиниб
        "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",  # сорафениб
        ASPIRIN,
        "CC(=O)Nc1ccc(O)cc1",  # парацетамол
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ибупрофен
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # кофеин
        "CCO",
        "CCCO",
        "CCCCO",
        "CC(=O)O",
        "CCC(=O)O",
        "c1ccccc1",
    ]
    train_y = [
        8.2, 8.5, 8.4, 8.1, 8.6, 7.9, 8.8, 8.7, 8.0, 7.8,
        4.5, 4.3, 4.6, 4.2, 3.1, 3.2, 3.4, 3.0, 3.1, 3.5,
    ]

    model, metrics = qsar.train(train_smiles, train_y)
    assert metrics.metric_name == "R2"
    assert metrics.n_train == 16

    pred_val = qsar.predict(model, metrics, IMATINIB)
    assert pred_val.name == "pIC50"
    assert pred_val.provenance is Provenance.PREDICTED
    assert pred_val.model_metrics is not None
    assert isinstance(pred_val.value, float)

    mol = chem.describe(IMATINIB)
    mol.values.append(pred_val)
    assert check_provenance(
        [Candidate(molecule=mol)],
        tool_calls=[ToolCall(agent="Evaluator", tool="qsar.predict", args_summary="EGFR", ok=True)],
        t=Thresholds(min_qsar_train=10),
    ) == {}

    broken_pred = qsar.predict(model, metrics, BROKEN)
    assert broken_pred.value is None


# --- ADMET (блок T4 + риск R3) ---

def test_admet_profile_and_classifier():
    from src.tools import admet

    admet.clear_models()
    vals_asp = {v.name: v for v in admet.profile(ASPIRIN)}
    vals_ima = {v.name: v for v in admet.profile(IMATINIB)}

    # ESOL (Delaney 2004) — эмпирическая линейная регрессия (n_train=2874, R2=0.69), поэтому PREDICTED
    assert vals_asp["LogS"].provenance is Provenance.PREDICTED
    assert vals_ima["LogS"].provenance is Provenance.PREDICTED
    assert vals_asp["LogS"].model_metrics is not None
    assert vals_asp["LogS"].model_metrics.n_train == 2874
    assert vals_asp["LogS"].model_metrics.metric_name == "R2"
    assert vals_asp["LogS"].model_metrics.metric_value == 0.69
    # Аспирин существенно лучше растворим в воде, чем липофильный иматиниб
    assert vals_asp["LogS"].value > vals_ima["LogS"].value
    assert vals_ima["Brenk_Alerts"].provenance is Provenance.COMPUTED

    # Без загруженной ML-модели hERG возвращается как not_computed (value=None), а не выдумка
    assert vals_ima["hERG"].value is None

    # После обучения классификатора hERG возвращается PREDICTED с обязательными ModelMetrics
    smiles_cls = [
        IMATINIB,
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
        "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
        "Cc1ccc(C(=O)Nc2ccc(CN3CCN(C)CC3)c(C(F)(F)F)c2)cc1C#Cc1cnc2cccnn12",
        "COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1",
        ASPIRIN,
        "CC(=O)Nc1ccc(O)cc1",
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CCO",
        "CC(=O)O",
    ]
    labels_cls = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    _, metrics = admet.train_classifier("hERG", smiles_cls, labels_cls)
    assert metrics.metric_name == "ROC-AUC"

    vals_after = {v.name: v for v in admet.profile(IMATINIB)}
    herg_v = vals_after["hERG"]
    assert herg_v.provenance is Provenance.PREDICTED
    assert herg_v.model_metrics is not None
    assert 0.0 <= herg_v.value <= 1.0

    # Проверяем, что верификатор принимает профиль ADMET при наличии ToolCall(tool="admet.profile")
    mol = chem.describe(IMATINIB)
    mol.values.extend(vals_after.values())
    assert check_provenance(
        [Candidate(molecule=mol)],
        tool_calls=[ToolCall(agent="Evaluator", tool="admet.profile", args_summary="imatinib", ok=True)],
        t=Thresholds(min_qsar_train=5),
    ) == {}
    admet.clear_models()
