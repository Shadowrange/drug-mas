"""Детерминированный хемоинформатический слой на базе RDKit.

Все вычисляемые здесь дескрипторы и структурные проверки являются детерминированными
расчётами (`Provenance.COMPUTED`). Ни одна функция не обращается к внешним API и не вызывает LLM.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, RDConfig
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from src.schemas import FilterResult, Molecule, Value, computed, not_computed

RDLogger.DisableLog("rdApp.*")

RDKIT_SRC = f"RDKit {Chem.rdBase.rdkitVersion}"

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer  # type: ignore
    _SA_OK = True
except ImportError:  # pragma: no cover
    _SA_OK = False


@lru_cache(maxsize=1)
def _pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def parse(smiles: str) -> Chem.Mol | None:
    """Единственная точка входа для SMILES. Невалидное -> None, без исключений."""
    if not smiles or not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def canonical(smiles: str) -> str | None:
    mol = parse(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def is_valid(smiles: str) -> bool:
    return parse(smiles) is not None


def describe(smiles: str, name: str | None = None, source: str = "входные данные") -> Molecule | None:
    """
    Главная функция слоя: SMILES -> Molecule со всеми дескрипторами.
    Возвращает None, если SMILES не разбирается. Молчаливых заглушек нет.
    """
    mol = parse(smiles)
    if mol is None:
        return None

    values: list[Value] = [
        computed("MW", round(Descriptors.MolWt(mol), 2), f"{RDKIT_SRC} Descriptors.MolWt", "Da"),
        computed("LogP", round(Crippen.MolLogP(mol), 2), f"{RDKIT_SRC} Crippen.MolLogP"),
        computed("TPSA", round(Descriptors.TPSA(mol), 2), f"{RDKIT_SRC} Descriptors.TPSA", "Å²"),
        computed("HBD", Descriptors.NumHDonors(mol), f"{RDKIT_SRC} Descriptors.NumHDonors"),
        computed("HBA", Descriptors.NumHAcceptors(mol), f"{RDKIT_SRC} Descriptors.NumHAcceptors"),
        computed("RotB", Descriptors.NumRotatableBonds(mol), f"{RDKIT_SRC} Descriptors.NumRotatableBonds"),
        computed("HeavyAtoms", mol.GetNumHeavyAtoms(), f"{RDKIT_SRC} Mol.GetNumHeavyAtoms"),
        computed("Rings", Descriptors.RingCount(mol), f"{RDKIT_SRC} Descriptors.RingCount"),
        computed("QED", round(QED.qed(mol), 3), f"{RDKIT_SRC} QED.qed"),
        computed("FormalCharge", Chem.GetFormalCharge(mol), f"{RDKIT_SRC} Chem.GetFormalCharge"),
    ]

    if _SA_OK:
        values.append(
            computed("SA_Score", round(sascorer.calculateScore(mol), 2),
                     f"{RDKIT_SRC} Contrib SA_Score (Ertl & Schuffenhauer 2009)",
                     note="1 = легко синтезировать, 10 = тяжело")
        )
    else:
        values.append(not_computed("SA_Score", "sascorer из RDKit Contrib недоступен"))

    return Molecule(
        smiles=Chem.MolToSmiles(mol),
        name=name,
        inchikey=Chem.MolToInchiKey(mol) or None,
        source=source,
        values=values,
    )


def lipinski(mol_or_smiles: Molecule | str) -> FilterResult:
    """Правило пяти Липински. Допускается одно нарушение — так принято в скрининге."""
    m = mol_or_smiles if isinstance(mol_or_smiles, Molecule) else describe(mol_or_smiles)
    if m is None:
        return FilterResult(name="Lipinski", passed=None, detail="SMILES не разбирается")

    checks = {
        "MW <= 500": (m.numeric("MW"), lambda v: v <= 500),
        "LogP <= 5": (m.numeric("LogP"), lambda v: v <= 5),
        "HBD <= 5": (m.numeric("HBD"), lambda v: v <= 5),
        "HBA <= 10": (m.numeric("HBA"), lambda v: v <= 10),
    }
    if any(v is None for v, _ in checks.values()):
        return FilterResult(name="Lipinski", passed=None, detail="не все дескрипторы посчитаны")

    broken = [rule for rule, (v, ok) in checks.items() if not ok(v)]
    return FilterResult(
        name="Lipinski",
        passed=len(broken) <= 1,
        detail="нарушений нет" if not broken else f"нарушено: {', '.join(broken)}",
        threshold="допускается не более 1 нарушения",
    )


def pains(smiles: str) -> FilterResult:
    """PAINS — частые ложноположительные скаффолды. Срабатывание = красный флаг."""
    mol = parse(smiles)
    if mol is None:
        return FilterResult(name="PAINS", passed=None, detail="SMILES не разбирается")

    entry = _pains_catalog().GetFirstMatch(mol)
    return FilterResult(
        name="PAINS",
        passed=entry is None,
        detail="совпадений нет" if entry is None else f"совпадение: {entry.GetDescription()}",
        threshold="совпадений быть не должно",
    )


def sa_ok(smiles: str, threshold: float = 6.0) -> FilterResult:
    """Фильтр синтетической доступности `SA Score` (шкала 1–10, порог по умолчанию <= 6.0)."""
    m = describe(smiles)
    sa = m.numeric("SA_Score") if m else None
    if sa is None:
        return FilterResult(name="SA_Score", passed=None, detail="не посчитано", threshold=f"<= {threshold}")
    return FilterResult(
        name="SA_Score",
        passed=sa <= threshold,
        detail=f"SA = {sa}",
        threshold=f"<= {threshold}",
    )


def embed_3d(smiles: str, seed: int = 0xF00D) -> str | None:
    """
    3D-конформер (ETKDGv3 + MMFF94) -> PDB-блок. Нужен для подготовки лиганда к докингу.
    None, если встраивание не удалось — это нормальный исход, не ошибка.
    """
    mol = parse(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # геометрия без оптимизации всё ещё пригодна для докинга
    return Chem.MolToPDBBlock(mol)


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    """ECFP4 для QSAR. Возвращает numpy-массив или None."""
    import numpy as np
    from rdkit.Chem import rdFingerprintGenerator

    mol = parse(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(gen.GetFingerprint(mol), dtype=np.int8)


def run_all_filters(smiles: str, sa_threshold: float = 6.0) -> list[FilterResult]:
    """Полный детерминированный набор фильтров для Evaluator'а."""
    return [lipinski(smiles), pains(smiles), sa_ok(smiles, sa_threshold)]
