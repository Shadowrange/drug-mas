"""Streamlit. ВЛАДЕЛЕЦ: Р-C. Блоки F0/F1.

Четыре зоны, сверху вниз:
 1) чат с запросом на естественном языке
 2) лог вызовов инструментов (ToolCall) — видимая цепочка работы агентов
 3) таблица кандидатов, где У КАЖДОГО ЧИСЛА виден бейдж Provenance.badge
 4) 2D-структуры (RDKit) и 3D-поза докинга (stmol/py3Dmol, заглушка)

Зона 3 — самое важное на защите: требование №5 должно быть ВИДНО на экране.

TODO Р-C:
  - заменить _fixture() на run_graph() из src/graph.py (OFFLINE=1);
  - 3D-поза через stmol/py3Dmol;
  - проверить шрифты и цвета бейджей на проекторе до пятницы.
"""
from __future__ import annotations

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.schemas import (
    Candidate, FilterResult, ModelMetrics, Provenance, ScenarioResult,
    ToolCall, Value, from_db, not_computed, predicted,
)
from src.tools.chem import describe

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
GRAY, GREEN, RED = "#888", "#1a7f37", "#cf222e"


def _fixture() -> ScenarioResult:
    """Числа computed — реальный RDKit по аспирину, не вписаны руками.
    from_db, predicted, hypothesis — демонстрация бейджа: в source стоит
    FIXTURE, чтобы это нельзя было спутать с реальным прогоном.
    RotB-фильтр с порогом <= 2 — намеренный passed=False, чтобы на экране
    было видно красное состояние (в Thresholds такого правила нет)."""
    mol = describe(ASPIRIN, name="аспирин (фикстура)", source="PubChem CID 2244")
    if mol is None:
        raise RuntimeError("RDKit не разобрал SMILES фикстуры")

    mol.values.extend([
        from_db("CID", 2244, "PubChem CID 2244"),
        predicted(
            "LogS", -2.1, "FIXTURE CatBoost (демо бейджа, не прогон)",
            ModelMetrics(model_name="FIXTURE", n_train=842,
                         metric_name="R2", metric_value=0.71),
            units="log mol/L",
        ),
        not_computed("dG", "докинг в этом прогоне не запускался"),
        Value(
            name="Гипотеза: механизм", value=None,
            provenance=Provenance.HYPOTHESIS,
            source="рассуждение LLM (фикстура, не прогон)",
            note="иллюстрация 4-го бейджа; числа у гипотезы быть не может",
        ),
    ])

    sa = mol.numeric("SA_Score")
    rotb = mol.numeric("RotB")
    cand = Candidate(
        molecule=mol,
        filters=[
            FilterResult(name="Lipinski", passed=True,
                         detail="нарушений нет",
                         threshold="допускается не более 1 нарушения"),
            FilterResult(name="PAINS", passed=True,
                         detail="совпадений нет",
                         threshold="совпадений быть не должно"),
            FilterResult(name="SA_Score",
                         passed=sa is not None and sa <= 6.0,
                         detail=f"SA = {sa}", threshold="<= 6.0"),
            FilterResult(name="RotB (демо)", passed=rotb is not None and rotb <= 2,
                         detail=f"RotB = {rotb}",
                         threshold="<= 2 (только для показа passed=False)"),
            FilterResult(name="dG", passed=None, detail="не посчитано",
                         threshold="<= -6.0"),
        ],
    )

    return ScenarioResult(
        status="no_candidates_passed",
        query="фикстура: показать бейджи provenance до первого прогона",
        scenario="fixture",
        candidates=[cand],
        tool_calls=[
            ToolCall(agent="Data", tool="pubchem_smiles",
                     args_summary="aspirin", ok=True, from_cache=True,
                     duration_s=0.02),
            ToolCall(agent="Evaluator", tool="chem.describe",
                     args_summary="aspirin", ok=True, from_cache=False,
                     duration_s=0.11),
            ToolCall(agent="Evaluator", tool="dock",
                     args_summary="1IEP", ok=False, from_cache=False,
                     duration_s=0.0,
                     error="NotImplementedError: TODO Р-A, dG не считается"),
        ],
        limitations=[
            "докинг не реализован: dG помечен «не посчитано» "
            "и не участвует в ранжировании",
        ],
    )


def _render_values(cand: Candidate) -> None:
    rows: list[str] = []
    for v in cand.molecule.values:
        if v.value is None:
            val = f'<span style="color:{GRAY}">—</span>'
            note = (f'<br><span style="color:{GRAY};font-size:0.8em">'
                    f'{v.note or "не посчитано"}</span>')
        else:
            val = str(v.value) + (f" {v.units}" if v.units else "")
            note = ""
        rows.append(
            f"<tr><td>{v.name}</td>"
            f"<td>{val}{note}</td>"
            f"<td>{v.provenance.badge}</td>"
            f"<td><span style='font-size:0.8em;color:{GRAY}'>{v.source}</span></td>"
            f"</tr>"
        )
    st.markdown(
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr>"
        "<th align='left'>Показатель</th><th align='left'>Значение</th>"
        "<th align='left'>Происхождение</th><th align='left'>Источник</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>",
        unsafe_allow_html=True,
    )


def _render_filters(cand: Candidate) -> None:
    for f in cand.filters:
        if f.passed is True:
            icon, color = "✅", GREEN
        elif f.passed is False:
            icon, color = "❌", RED
        else:
            icon, color = "—", GRAY
        thr = f" · порог {f.threshold}" if f.threshold else ""
        st.markdown(
            f"<div style='color:{color}'>{icon} <b>{f.name}</b>: "
            f"{f.detail}{thr}</div>",
            unsafe_allow_html=True,
        )


def _render_tool_calls(calls: list[ToolCall]) -> None:
    if not calls:
        st.caption("Вызовов ещё не было.")
        return
    st.dataframe(
        [
            {
                "Агент": c.agent,
                "Инструмент": c.tool,
                "Аргументы": c.args_summary,
                "OK": "✅" if c.ok else "❌",
                "Источник": "🗄 из кэша" if c.from_cache else "🌐 сеть",
                "t, с": c.duration_s,
                "Ошибка": c.error or "",
            }
            for c in calls
        ],
        width="stretch",
        hide_index=True,
    )


st.set_page_config(page_title="drug-mas", layout="wide")
st.title("Мультиагентная система для дизайна лекарственных молекул")
st.caption("ХимБио-хакатон «Синтез» · кейс 1 · ПИШ ИТМО")

# --- (a) запрос на естественном языке ---
query = st.text_input(
    "Запрос на естественном языке",
    placeholder="Модифицируй иматиниб, повысив растворимость...",
)
if "result" not in st.session_state:
    st.session_state["result"] = _fixture()
if st.button("Запустить"):
    # TODO Р-C: st.session_state["result"] = run_graph(query) при OFFLINE=1
    st.session_state["result"] = _fixture()

result: ScenarioResult = st.session_state["result"]
if result.scenario == "fixture":
    st.info("Показана фикстура: graph.py ещё не подключён. "
            "Значения 🧮 РАСЧЁТ — реальный RDKit по аспирину; "
            "🗄 БД, 📈 ПРОГНОЗ и 💭 ГИПОТЕЗА помечены FIXTURE в source.")

# --- (b) лог вызовов инструментов ---
st.subheader("Журнал вызовов инструментов")
_render_tool_calls(result.tool_calls)

# --- (c) таблица кандидатов с бейджами provenance ---
st.subheader("Кандидаты")
if result.status == "insufficient_data":
    st.warning("Недостаточно данных — молекул нет. Чего не хватило:")
    for m in result.missing:
        st.markdown(
            f"- **{m.what}**: найдено {m.found}, нужно {m.needed}. "
            f"Разблокирует: {m.would_unlock}"
        )
elif not result.candidates:
    st.error("Кандидатов нет.")
else:
    if result.status == "no_candidates_passed":
        st.error("Все кандидаты провалили пороги — показываем с причинами отсева.")
    for cand in result.candidates:
        title = cand.molecule.name or cand.molecule.smiles
        st.markdown(f"#### {title} · rank={cand.rank}")
        st.caption(f"SMILES: `{cand.molecule.smiles}` · "
                   f"источник: {cand.molecule.source}")
        _render_values(cand)
        st.markdown("**Фильтры**")
        _render_filters(cand)

# --- (d) 2D-структура и заглушка 3D-вьюера ---
st.subheader("Структура")
smiles_list = [c.molecule.smiles for c in result.candidates]
if smiles_list:
    pick = st.selectbox("Кандидат", smiles_list)
    m = Chem.MolFromSmiles(pick)
    if m is None:
        st.error("RDKit не разобрал SMILES")
    else:
        st.image(Draw.MolToImage(m, size=(400, 300)))
else:
    st.caption("Показывать нечего: кандидатов нет.")
st.info("TODO Р-C: 3D-поза докинга через stmol/py3Dmol — заглушка.")

if result.limitations:
    st.subheader("Ограничения прогона")
    for lim in result.limitations:
        st.markdown(f"- {lim}")