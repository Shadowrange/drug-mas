"""
ADMET. ВЛАДЕЛЕЦ: Р-A. Блок T4.

Таймбокс на установку ADMET-AI — 45 минут (риск R3).
Не завелось -> фолбэк: CatBoost на hERG из TDC. Решение записать на доску.
"""
from __future__ import annotations

from src.schemas import Value


def profile(smiles: str) -> list[Value]:
    """TODO Р-A: hERG, Ames, растворимость, CYP. Все значения provenance=PREDICTED + ModelMetrics.
    Если модель недоступна — schemas.not_computed(...), НЕ выдумывать числа."""
    raise NotImplementedError
