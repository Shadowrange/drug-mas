"""
Докинг. ВЛАДЕЛЕЦ: Р-A. Блок D1 — ГЛАВНЫЙ РИСК ПРОЕКТА (R1).

Жёсткий дедлайн спайка: 01:00 среды. Не заработало -> план Б из PLAN §10:
ΔG не считаем вообще и честно помечаем как not_computed. Выдумывать ΔG нельзя
ни при каких обстоятельствах — на этом строится вся защита.
"""
from __future__ import annotations

from src.schemas import Value


def prepare_receptor(pdb_path: str) -> str:
    """TODO Р-A: Meeko/OpenBabel -> PDBQT."""
    raise NotImplementedError


def dock(ligand_smiles: str, receptor_pdbqt: str, center: tuple[float, float, float],
         box: tuple[float, float, float] = (22.0, 22.0, 22.0), exhaustiveness: int = 8) -> Value:
    """TODO Р-A: chem.embed_3d -> PDBQT -> vina.Vina().dock() -> лучший ΔG.
    Вернуть computed('dG', value, 'AutoDock Vina 1.2.x', 'ккал/моль')."""
    raise NotImplementedError


def interactions(pose_pdb: str, receptor_pdb: str) -> list[str]:
    """TODO Р-A: PLIP -> список водородных связей и гидрофобных контактов.
    Это обоснование выбора молекулы для жюри."""
    raise NotImplementedError
