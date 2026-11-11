"""
Генерация и модификация структур. ВЛАДЕЛЕЦ: Р-B. Блок T3.

План А — CReM (нужна скачанная база фрагментов).
План Б — STONED-SELFIES (базы не требует). Решение по R2 принять до 23:00 среды.
"""
from __future__ import annotations

from src.schemas import Molecule


def mutate(parent_smiles: str, n: int = 50, backend: str = "crem") -> list[Molecule]:
    """TODO Р-B.
    crem: from crem.crem import mutate_mol, нужен db_name=replacements.db
    stoned: selfies, случайные мутации строки SELFIES -> всегда валидный SMILES
    На выходе ТОЛЬКО валидные уникальные SMILES, source='сгенерировано {backend} из {parent}'.
    """
    raise NotImplementedError
