"""Data Agent. ВЛАДЕЛЕЦ: Р-B. Блок A1.
Тянет SMILES, активности, структуру мишени, координаты бокса. Обучает QSAR при достаточной выборке.
ОБЯЗАТЕЛЬНО вызывает verifier.check_data_sufficiency ДО передачи в Generator."""
from __future__ import annotations
