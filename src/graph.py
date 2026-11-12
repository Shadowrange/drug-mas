"""Сборка LangGraph. ВЛАДЕЛЕЦ: Р-B. Блок A1.

Схема с учётом поправок §2.5 и §2.6 docs/GRAPH-CONTRACT.md:

    START -> supervisor -> data_agent -> route_sufficiency
                                           |-- missing     --> reporter -> END
                                           |-- ok          --> generator
    generator -> route_generated
                     |-- empty & loop>=MAX --> reporter -> END
                     |-- empty & loop<MAX  --> generator
                     |-- ok                --> evaluator
    evaluator -> verifier_node -> route_verify
                                           |-- passed      --> reporter -> END
                                           |-- retry       --> generator
                                           |-- exhausted   --> reporter -> END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents import (
    data_agent as _data_agent_mod,
    evaluator as _evaluator_mod,
    generator as _generator_mod,
    reporter as _reporter_mod,
    supervisor as _supervisor_mod,
)
from src.agents.data_agent import data_agent
from src.agents.evaluator import evaluator
from src.agents.generator import generator
from src.agents.reporter import reporter
from src.agents.supervisor import supervisor
from src.graph_state import GraphState, dedupe_by_smiles, dedupe_strings
from src.schemas import ScenarioResult
from src.verifier import Thresholds, verify

MAX_LOOPS = 2

_ORIG_SUPERVISOR = supervisor
_ORIG_DATA_AGENT = data_agent
_ORIG_GENERATOR = generator
_ORIG_EVALUATOR = evaluator
_ORIG_REPORTER = reporter


def route_sufficiency(state: GraphState) -> str:
    """Роутер достаточности данных (§2): непустой missing -> reporter, иначе generator."""
    return "reporter" if state.get("missing") else "generator"


def route_generated(state: GraphState) -> str:
    """Роутер после генератора (§2.5 Поправка 1, §2.6)."""
    if state.get("generated"):
        return "evaluator"
    return "generator" if int(state.get("loop", 0)) < MAX_LOOPS else "reporter"


def route_verify(state: GraphState) -> str:
    """
    Роутер после верификатора (§2.6).
    Наличие выживших проверяется РАНЬШЕ лимита кругов.
    """
    rep = state.get("verification")
    if rep is not None and rep.passed:
        return "reporter"          # passed
    if int(state.get("loop", 0)) >= MAX_LOOPS:
        return "reporter"          # exhausted
    return "generator"             # retry


def verifier_node(state: GraphState) -> dict[str, Any]:
    """
    Узел верификатора (§2, §2.5 Поправка 2, §2.6).
    Не LLM — обёртка над verifier.verify с накоплением passed_all, rejected_all, unverifiable_all.
    """
    t = state.get("thresholds") or Thresholds()
    rep = verify(state.get("candidates") or [], state.get("tool_calls") or [], t=t)

    passed_all = dedupe_by_smiles(list(state.get("passed_all") or []) + rep.passed)
    rejected_all = dedupe_by_smiles(list(state.get("rejected_all") or []) + rep.rejected)
    unverifiable_all = dedupe_strings(list(state.get("unverifiable_all") or []) + rep.unverifiable)

    return {
        "verification": rep,
        "feedback": list(rep.feedback),
        "passed_all": passed_all,
        "rejected_all": rejected_all,
        "unverifiable_all": unverifiable_all,
        "tool_calls": [],
    }


_ORIG_VERIFIER_NODE = verifier_node


def build_graph(thresholds: Thresholds | None = None):
    """Собирает и компилирует StateGraph(GraphState) строго по §2.6."""

    def _inject_thresholds(state: GraphState) -> GraphState:
        if thresholds is not None and state.get("thresholds") is None:
            copied = dict(state)
            copied["thresholds"] = thresholds
            return copied  # type: ignore[return-value]
        return state

    def _run_supervisor(state: GraphState) -> dict[str, Any]:
        fn = supervisor if supervisor is not _ORIG_SUPERVISOR else _supervisor_mod.supervisor
        return fn(_inject_thresholds(state))

    def _run_data_agent(state: GraphState) -> dict[str, Any]:
        fn = data_agent if data_agent is not _ORIG_DATA_AGENT else _data_agent_mod.data_agent
        return fn(_inject_thresholds(state))

    def _run_generator(state: GraphState) -> dict[str, Any]:
        fn = generator if generator is not _ORIG_GENERATOR else _generator_mod.generator
        return fn(_inject_thresholds(state))

    def _run_evaluator(state: GraphState) -> dict[str, Any]:
        fn = evaluator if evaluator is not _ORIG_EVALUATOR else _evaluator_mod.evaluator
        return fn(_inject_thresholds(state))

    def _run_verifier(state: GraphState) -> dict[str, Any]:
        fn = verifier_node if verifier_node is not _ORIG_VERIFIER_NODE else _ORIG_VERIFIER_NODE
        return fn(_inject_thresholds(state))

    def _run_reporter(state: GraphState) -> dict[str, Any]:
        fn = reporter if reporter is not _ORIG_REPORTER else _reporter_mod.reporter
        return fn(_inject_thresholds(state))

    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", _run_supervisor)
    workflow.add_node("data_agent", _run_data_agent)
    workflow.add_node("generator", _run_generator)
    workflow.add_node("evaluator", _run_evaluator)
    workflow.add_node("verifier_node", _run_verifier)
    workflow.add_node("reporter", _run_reporter)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "data_agent")
    workflow.add_conditional_edges(
        "data_agent",
        route_sufficiency,
        {
            "reporter": "reporter",
            "generator": "generator",
        },
    )
    workflow.add_conditional_edges(
        "generator",
        route_generated,
        {
            "evaluator": "evaluator",
            "generator": "generator",
            "reporter": "reporter",
        },
    )
    workflow.add_edge("evaluator", "verifier_node")
    workflow.add_conditional_edges(
        "verifier_node",
        route_verify,
        {
            "reporter": "reporter",
            "generator": "generator",
        },
    )
    workflow.add_edge("reporter", END)

    return workflow.compile()


GRAPH = build_graph()


def run_graph(
    query: str,
    thresholds: Thresholds | None = None,
) -> ScenarioResult:
    """
    Публичная точка входа для запуска графа (в т.ч. из app.py и тестов).
    Возвращает валидированный ScenarioResult.
    """
    compiled = build_graph(thresholds=thresholds)
    init_state: GraphState = {
        "query": query,
        "loop": 0,
        "tool_calls": [],
        "rejected_all": [],
        "passed_all": [],
        "unverifiable_all": [],
        "feedback": [],
        "notes": [],
    }
    if thresholds is not None:
        init_state["thresholds"] = thresholds

    final_state = compiled.invoke(init_state)
    result = final_state.get("result")
    if result is None:
        raise RuntimeError("Граф завершился без формирования ScenarioResult")
    return result
