"""Сборка LangGraph. ВЛАДЕЛЕЦ: Р-B. Блок A1.

    START -> supervisor -> data_agent -> [проверка достаточности]
                                      |-> insufficient -> reporter -> END
                                      |-> ok -> generator -> evaluator -> verifier
                                                    ^                  |
                                                    |---- feedback ----|  (макс. 2 круга)
                                                                       -> reporter -> END
"""
from __future__ import annotations

MAX_LOOPS = 2
