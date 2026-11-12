"""Streamlit. ВЛАДЕЛЕЦ: Р-C. Блоки F0/F1.

Зоны экрана, сверху вниз:
 1) запрос на естественном языке и три кнопки-пресета
 2) журнал вызовов инструментов (ToolCall) — видимая цепочка работы агентов
 3) кандидаты: одна карточка = 2D-структура (RDKit) слева и таблица величин
    справа, где У КАЖДОГО ЧИСЛА видна плашка Provenance; ниже — фильтры
 4) ограничения прогона

Зона 3 — самое важное на защите: требование №5 должно быть ВИДНО на экране,
в том числе с последнего ряда. Поэтому происхождение — крупная цветная плашка,
а не значок в ячейке.

Ограничение интерфейса: 3D-поза докинга не визуализируется —
AutoDock Vina не входит в окружение прототипа, ΔG честно помечается
как «не вычислялось» и не участвует в ранжировании.

Вся вёрстка — HTML/CSS внутри этого файла: менять тему через
.streamlit/config.toml нельзя (зона Р-C — только app.py), а дефолтные
виджеты Streamlit не дают ни плашек, ни карточек.
"""
from __future__ import annotations

import base64
import html

import streamlit as st
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from src.schemas import (
    Candidate, FilterResult, MissingData, ModelMetrics, Provenance,
    ScenarioResult, ToolCall, Value, from_db, not_computed, predicted,
)
from src.tools.chem import describe

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

# ← перед защитой: False. Фикстура остаётся только как аварийный запас,
# если граф не поднимется на чужой машине.
USE_FIXTURE: bool = False


# --------------------------------------------------------------------------
# Оформление. Палитра проекта: фон #0F1010, текст #E6E6E6, лайм #C4F747,
# синий #7C99FD, сиреневый #E0A9FB, серый #8A8F8F. Других цветов нет:
# «не прошёл» и «сбой» выделены формой (светлая инверсная плашка), а не
# красным — красного в палитре нет, и отказ системы не должен выглядеть сбоем.
# Шрифты только системные: CDN на защите может не открыться.
# --------------------------------------------------------------------------
_CSS = """
<style>
:root{
  --dm-bg:#0F1010; --dm-surface:#151717; --dm-line:#272A2A;
  --dm-text:#E6E6E6; --dm-gray:#8A8F8F;
  --dm-lime:#C4F747; --dm-blue:#7C99FD; --dm-lilac:#E0A9FB;
  --dm-sans:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
  --dm-mono:"Cascadia Mono","Cascadia Code",Consolas,"SFMono-Regular",Menlo,"DejaVu Sans Mono",monospace;
}
.stApp{background:var(--dm-bg);color:var(--dm-text);font-family:var(--dm-sans);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMainBlockContainer"]{padding-top:1.75rem;padding-bottom:4rem;max-width:1680px;}
[data-testid="stElementContainer"]:has(style){display:none;}
.stApp [data-testid="stMarkdownContainer"]{color:var(--dm-text);font-family:var(--dm-sans);}

/* --- шапка --- */
.stApp h1.dm-title{font-family:var(--dm-sans);font-size:2.2rem;line-height:1.15;font-weight:600;letter-spacing:-.02em;margin:0;padding:0;color:var(--dm-text);}
.dm-sub{font-size:1rem;color:var(--dm-gray);margin:.4rem 0 .25rem;letter-spacing:.02em;}

/* --- заголовки секций --- */
.dm-sec{display:flex;align-items:center;gap:1rem;margin:1.1rem 0 0;
  font-size:.95rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--dm-gray);}
.dm-sec::after{content:"";flex:1;height:1px;background:var(--dm-line);}

/* --- ввод и кнопки --- */
.stApp [data-testid="stWidgetLabel"] p{font-size:.95rem;font-weight:600;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dm-gray);}
.stApp [data-testid="stTextInputRootElement"],
.stApp [data-testid="stTextInput"] div[data-baseweb="input"],
.stApp [data-testid="stTextInput"] div[data-baseweb="base-input"]{
  background:var(--dm-surface);border-color:var(--dm-line);border-radius:10px;}
.stApp [data-testid="stTextInputRootElement"]{min-height:3.2rem;align-items:center;}
.stApp [data-testid="stTextInputRootElement"]:focus-within,
.stApp [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within{border-color:var(--dm-lime);}
.stApp [data-testid="stTextInput"] input{color:var(--dm-text);-webkit-text-fill-color:var(--dm-text);
  caret-color:var(--dm-lime);font-family:var(--dm-sans);font-size:1.2rem;padding:.8rem 1rem;}
.stApp [data-testid="stTextInput"] input::placeholder{color:var(--dm-gray);-webkit-text-fill-color:var(--dm-gray);opacity:1;}
.stApp [data-testid="stButton"] button{
  background:var(--dm-surface);color:var(--dm-text);border:1px solid var(--dm-line);
  border-radius:10px;min-height:3.2rem;padding:.6rem 1rem;font-family:var(--dm-sans);}
.stApp [data-testid="stButton"] button p{font-size:1.05rem;line-height:1.25;font-weight:600;color:inherit;
  white-space:normal;overflow:visible;text-overflow:clip;}
.stApp [data-testid="stButton"] button div{white-space:normal;overflow:visible;}
.stApp [data-testid="stButton"] button:hover,
.stApp [data-testid="stButton"] button:focus-visible{
  border-color:var(--dm-lime);color:var(--dm-lime);background:var(--dm-surface);}
.stApp [data-testid="stButton"] button:active{background:var(--dm-line);color:var(--dm-lime);}
.stApp [data-testid="stButton"] button[kind="primary"]{
  background:transparent;border:2px solid var(--dm-lime);color:var(--dm-lime);}
.stApp [data-testid="stButton"] button[kind="primary"]:hover{background:var(--dm-lime);color:var(--dm-bg);}

/* --- плашки происхождения: главный элемент экрана --- */
.dm-b{display:inline-block;box-sizing:border-box;min-width:13.4em;text-align:center;white-space:nowrap;
  font-family:var(--dm-sans);font-size:16px;line-height:1;font-weight:700;letter-spacing:.07em;
  text-transform:uppercase;padding:.6em .9em;border-radius:8px;border:2px solid transparent;color:var(--dm-bg);}
.dm-b-database{background:var(--dm-blue);border-color:var(--dm-blue);}
.dm-b-computed{background:var(--dm-lime);border-color:var(--dm-lime);}
.dm-b-predicted{background:var(--dm-lilac);border-color:var(--dm-lilac);}
.dm-b-hypothesis{background:transparent;border-color:var(--dm-gray);color:var(--dm-text);}
.dm-b-none{background:transparent;border:2px dashed var(--dm-gray);color:var(--dm-gray);}
.dm-b-inline{min-width:0;font-size:14px;padding:.4em .7em;vertical-align:baseline;margin:0 .1em;}

/* --- спокойные сообщения вместо st.warning / st.error --- */
.dm-note{border:1px solid var(--dm-line);border-left:4px solid var(--dm-gray);border-radius:10px;
  background:var(--dm-surface);padding:1rem 1.25rem;font-size:1.1rem;line-height:1.6;color:var(--dm-text);margin:.5rem 0;}

/* --- журнал вызовов --- */
.dm-log{font-family:var(--dm-mono);font-size:14px;line-height:1.45;border:1px solid var(--dm-line);
  border-radius:10px;background:var(--dm-surface);padding:.35rem 0;margin-top:.5rem;}
.dm-log-row{display:grid;column-gap:1rem;align-items:baseline;padding:.22rem 1rem;
  grid-template-columns:1.6em 6.5em minmax(9em,13em) minmax(8em,16em) 6.5em 4em 4.2em minmax(0,1fr);
  border-left:3px solid transparent;color:var(--dm-gray);}
.dm-log-row>span{overflow-wrap:anywhere;}
.dm-log-head{font-family:var(--dm-sans);font-size:14px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;border-bottom:1px solid var(--dm-line);padding-bottom:.4rem;margin-bottom:.2rem;}
.dm-log-row .dm-log-tool{color:var(--dm-text);}
.dm-log-row .dm-log-t{text-align:right;}
.dm-log-fail{border-left-color:var(--dm-text);color:var(--dm-text);background:rgba(230,230,230,.045);}
.dm-st{display:inline-block;min-width:3.6em;text-align:center;font-weight:700;letter-spacing:.06em;
  border-radius:4px;padding:0 .35em;}
.dm-st-ok{color:var(--dm-gray);}
.dm-st-fail{background:var(--dm-text);color:var(--dm-bg);}

/* --- карточка кандидата --- */
.dm-card{border:1px solid var(--dm-line);border-radius:14px;background:var(--dm-surface);
  padding:1.5rem 1.75rem 1.6rem;margin:.5rem 0 1rem;}
.dm-card-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:.4rem 1rem;margin-bottom:.35rem;}
.dm-name{font-size:1.7rem;line-height:1.2;font-weight:600;letter-spacing:-.01em;color:var(--dm-text);}
.dm-name-smiles{font-family:var(--dm-mono);font-size:1.25rem;font-weight:400;letter-spacing:0;overflow-wrap:anywhere;}
.dm-rank{font-family:var(--dm-mono);font-size:1.05rem;color:var(--dm-bg);background:var(--dm-text);
  border-radius:6px;padding:.1em .6em;white-space:nowrap;}
.dm-meta{font-size:1rem;line-height:1.55;color:var(--dm-gray);margin:.15rem 0;}
.dm-mono{font-family:var(--dm-mono);color:var(--dm-text);overflow-wrap:anywhere;}
.dm-card-body{display:flex;flex-wrap:wrap;gap:1.75rem;align-items:flex-start;margin-top:1.1rem;}
.dm-struct{flex:0 0 34%;min-width:320px;max-width:560px;position:sticky;top:4.5rem;
  border:1px solid var(--dm-line);border-radius:12px;background:var(--dm-bg);padding:.5rem;box-sizing:border-box;}
.dm-struct img{display:block;width:100%;height:auto;}
.dm-struct-err{padding:3rem 1rem;text-align:center;color:var(--dm-gray);font-size:1.1rem;}
.dm-vals{flex:1 1 560px;min-width:0;}
.dm-vals table{width:100%;border-collapse:collapse;border:0!important;margin:0;}
.dm-vals thead,.dm-vals tbody,.dm-vals tr{border:0!important;background:transparent!important;}
.dm-vals th,.dm-vals td{border:0!important;border-bottom:1px solid var(--dm-line)!important;
  padding:.5rem 1rem .5rem 0!important;text-align:left;vertical-align:middle;background:transparent!important;}
.dm-vals th{font-size:14px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--dm-gray)!important;}
.dm-vals tr:last-child td{border-bottom:0!important;}
.dm-v-name{width:1%;font-size:1.2rem;font-weight:600;color:var(--dm-text);white-space:nowrap;}
.dm-v-val{font-family:var(--dm-mono);font-size:1.3rem;color:var(--dm-text);font-variant-numeric:tabular-nums;}
.dm-v-units{font-size:1rem;color:var(--dm-gray);}
.dm-v-empty{font-family:var(--dm-mono);font-size:1.3rem;color:var(--dm-gray);}
.dm-v-note{display:block;overflow-wrap:anywhere;font-size:14px;line-height:1.4;color:var(--dm-gray);margin-top:.15rem;}
.dm-v-src{width:27%;font-size:14px;line-height:1.4;color:var(--dm-gray);overflow-wrap:anywhere;}
.dm-v-badge{width:1%;white-space:nowrap;}

/* --- фильтры --- */
.dm-sublabel{font-size:14px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  color:var(--dm-gray);margin:1.4rem 0 .6rem;}
.dm-filters{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:.5rem 2rem;}
.dm-f{display:flex;align-items:baseline;gap:.8rem;font-size:1.05rem;line-height:1.45;color:var(--dm-text);}
.dm-f-st{flex:0 0 11.4em;box-sizing:border-box;text-align:center;font-size:14px;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;border-radius:6px;padding:.25em .4em;border:2px solid transparent;white-space:nowrap;}
.dm-f-pass{color:var(--dm-text);border-color:var(--dm-line);}
.dm-f-fail{background:var(--dm-text);border-color:var(--dm-text);color:var(--dm-bg);}
.dm-f-none{border:2px dashed var(--dm-gray);color:var(--dm-gray);}
.dm-f-thr{color:var(--dm-gray);}

/* --- экран отказа: результат, а не сбой --- */
.dm-refusal{border:1px solid var(--dm-line);border-radius:16px;background:var(--dm-surface);
  padding:1.9rem 2.25rem 2.25rem;margin:.5rem 0 1.5rem;}
.dm-refusal-kicker{font-size:1rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--dm-gray);}
.dm-refusal-title{font-size:3.2rem;line-height:1.1;font-weight:600;letter-spacing:-.025em;color:var(--dm-text);margin:.4rem 0 .4rem;}
.dm-refusal-sub{font-size:1.3rem;color:var(--dm-gray);margin-bottom:1.25rem;}
.dm-miss-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:1.25rem;}
.dm-miss{border:1px solid var(--dm-line);border-radius:12px;background:var(--dm-bg);padding:1.25rem 1.5rem;}
.dm-miss-pair{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin:1rem 0;
  padding:1rem 0;border-top:1px solid var(--dm-line);border-bottom:1px solid var(--dm-line);}
.dm-flabel{font-size:14px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--dm-gray);margin-bottom:.35rem;}
.dm-fval{font-size:1.3rem;line-height:1.35;color:var(--dm-text);}
.dm-fval-what{font-size:1.65rem;line-height:1.25;font-weight:600;letter-spacing:-.01em;}

/* --- ограничения --- */
.dm-lims{margin:.75rem 0 0;padding:0;list-style:none;}
.dm-lims li{font-size:1.15rem;line-height:1.5;color:var(--dm-text);padding:.6rem 0 .6rem 1.1rem;
  border-left:3px solid var(--dm-gray);margin:0 0 .6rem;}
</style>
"""


def _e(x: object) -> str:
    """Всё, что пришло из ScenarioResult, экранируется: SMILES и тексты
    ошибок содержат <, >, & и ломают разметку."""
    return html.escape(str(x))


def _md(block: str) -> None:
    st.markdown(block, unsafe_allow_html=True)


def _section(title: str) -> None:
    _md(f'<div class="dm-sec">{_e(title)}</div>')


def _badge(prov: Provenance, extra_cls: str = "") -> str:
    # Подпись берём из schemas.Provenance.badge (единый источник названий),
    # отрезая эмодзи: на проекторе он превращается в цветное пятно,
    # а категорию несёт цвет плашки.
    label = prov.badge.split(" ", 1)[-1]
    return f'<span class="dm-b dm-b-{prov.value} {extra_cls}">{_e(label)}</span>'


_BADGE_NONE = '<span class="dm-b dm-b-none">НЕ ВЫЧИСЛЯЛОСЬ</span>'


def _fixture() -> ScenarioResult:
    """computed — реальный RDKit по аспирину. from_db / predicted / hypothesis
    помечены FIXTURE в source: это демонстрация бейджа, не результат прогона.
    Ни одного выдуманного правила в FilterResult нет — пороги только из
    chem.sa_ok и verifier.Thresholds."""
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
                     error="AutoDock Vina недоступна: dG не вычисляется"),
        ],
        limitations=[
            "докинг не реализован: dG помечен «не посчитано» "
            "и не участвует в ранжировании",
        ],
    )


def _run_real(query: str) -> ScenarioResult:
    """Реальный прогон графа. Ошибку не глушим (AGENTS.md §6): если граф
    упал, на экране должен быть виден трейсбек, а не молча фикстура,
    выданная за результат."""
    from src.graph import run_graph

    return run_graph(query)


# --------------------------------------------------------------------------
# Журнал вызовов
# --------------------------------------------------------------------------

# Только эти четыре инструмента ходят во внешние базы. Для остальных
# флаг from_cache смысла не имеет: RDKit, генератор и докинг считают
# локально, и подпись «сеть» у них — такая же неверная разметка источника,
# против которой построен весь проект.
NETWORK_TOOLS = {"pubchem_smiles", "chembl_activities", "uniprot_target", "pdb_structure"}


def _call_origin(c: ToolCall) -> str:
    if c.tool not in NETWORK_TOOLS:
        return "локально"
    return "из кэша" if c.from_cache else "сеть"


def _render_tool_calls(calls: list[ToolCall]) -> None:
    if not calls:
        _md('<div class="dm-note">Вызовов ещё не было.</div>')
        return
    rows = [
        '<div class="dm-log-row dm-log-head"><span>#</span><span>Агент</span>'
        '<span>Инструмент</span><span>Аргументы</span><span>Источник</span>'
        '<span class="dm-log-t">t, с</span><span>OK</span><span>Ошибка</span></div>'
    ]
    for i, c in enumerate(calls, 1):
        dur = "—" if c.duration_s is None else f"{c.duration_s:.3f}"
        status = ('<span class="dm-st dm-st-ok">OK</span>' if c.ok
                  else '<span class="dm-st dm-st-fail">СБОЙ</span>')
        rows.append(
            f'<div class="dm-log-row{"" if c.ok else " dm-log-fail"}">'
            f"<span>{i}</span><span>{_e(c.agent)}</span>"
            f'<span class="dm-log-tool">{_e(c.tool)}</span>'
            f"<span>{_e(c.args_summary)}</span><span>{_call_origin(c)}</span>"
            f'<span class="dm-log-t">{dur}</span><span>{status}</span>'
            f"<span>{_e(c.error or '')}</span></div>"
        )
    _md(f'<div class="dm-log">{"".join(rows)}</div>')


# --------------------------------------------------------------------------
# Карточка кандидата
# --------------------------------------------------------------------------

def _structure_html(smiles: str) -> str:
    """2D-структура как SVG в data-URI: светлые линии на фоне проекта.
    PNG из Draw.MolToImage — белый прямоугольник, на тёмном экране он
    перетягивает взгляд с плашек на себя."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return '<div class="dm-struct"><div class="dm-struct-err">RDKit не разобрал SMILES</div></div>'
    drawer = rdMolDraw2D.MolDraw2DSVG(640, 480)
    opts = drawer.drawOptions()
    opts.setBackgroundColour((0.0, 0.0, 0.0, 0.0))
    # Один цвет для всех атомов: гетероатомы читаются по буквам, а палитра
    # CPK принесла бы на экран красный кислород.
    light = (0.902, 0.902, 0.902)
    opts.setAtomPalette({-1: light})
    opts.setSymbolColour(light)
    opts.setAnnotationColour(light)
    opts.bondLineWidth = 2.4
    opts.minFontSize = 20
    opts.padding = 0.06
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    b64 = base64.b64encode(drawer.GetDrawingText().encode("utf-8")).decode("ascii")
    return (f'<div class="dm-struct"><img alt="2D-структура" '
            f'src="data:image/svg+xml;base64,{b64}"></div>')


def _values_html(cand: Candidate) -> str:
    rows: list[str] = []
    for v in cand.molecule.values:
        if v.value is None:
            val = (f'<span class="dm-v-empty">—</span>'
                   f'<span class="dm-v-note">{_e(v.note or "не посчитано")}</span>')
            # У несуществующего значения не может быть категории происхождения.
            # not_computed() ставит COMPUTED по умолчанию, и строка с прочерком
            # получала бы бейдж «РАСЧЁТ» — то есть заявляла расчёт, которого не было.
            badge = _BADGE_NONE
        else:
            units = f' <span class="dm-v-units">{_e(v.units)}</span>' if v.units else ""
            val = f'<span class="dm-v-val">{_e(v.value)}</span>{units}'
            # Прогноз без видимой метрики качества модели — первый вопрос жюри.
            # Данные уже лежат в ScenarioResult, вопрос только в том, показать их
            # или заставить спрашивать.
            if v.model_metrics is not None:
                m = v.model_metrics
                val += (f'<span class="dm-v-note">{_e(m.metric_name)} = {_e(m.metric_value)}'
                        f' · n = {_e(m.n_train)} · {_e(m.split)}</span>')
            badge = _badge(v.provenance)
        rows.append(
            f'<tr><td class="dm-v-name">{_e(v.name)}</td><td>{val}</td>'
            f'<td class="dm-v-badge">{badge}</td>'
            f'<td class="dm-v-src">{_e(v.source)}</td></tr>'
        )
    return (
        '<div class="dm-vals"><table><thead><tr>'
        "<th>Показатель</th><th>Значение</th><th>Происхождение</th><th>Источник</th>"
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _filters_html(cand: Candidate) -> str:
    items: list[str] = []
    for f in cand.filters:
        # Три разных исхода, три разные формы: «не проверялось» (passed=None)
        # нельзя рисовать ни как успех, ни как провал — см. AGENTS.md.
        if f.passed is True:
            cls, label = "dm-f-pass", "✓ прошёл"
        elif f.passed is False:
            cls, label = "dm-f-fail", "✕ не прошёл"
        else:
            cls, label = "dm-f-none", "не проверялось"
        thr = f' <span class="dm-f-thr">· порог {_e(f.threshold)}</span>' if f.threshold else ""
        items.append(
            f'<div class="dm-f"><span class="dm-f-st {cls}">{label}</span>'
            f"<span><b>{_e(f.name)}</b>: {_e(f.detail)}{thr}</span></div>"
        )
    return f'<div class="dm-sublabel">Фильтры</div><div class="dm-filters">{"".join(items)}</div>'


def _render_candidate(cand: Candidate) -> None:
    mol = cand.molecule
    rank = "—" if cand.rank is None else cand.rank
    smiles_line = f'<div class="dm-meta">SMILES <span class="dm-mono">{_e(mol.smiles)}</span></div>'
    if mol.name:
        name = f'<span class="dm-name">{_e(mol.name)}</span>'
    else:
        # Имени нет — заголовком, как и раньше, служит сам SMILES. Отдельная
        # строка SMILES под ним была бы той же строкой второй раз.
        name = f'<span class="dm-name dm-name-smiles">{_e(mol.smiles)}</span>'
        smiles_line = ""
    _md(
        '<div class="dm-card">'
        f'<div class="dm-card-head">{name}<span class="dm-rank">rank {_e(rank)}</span></div>'
        f"{smiles_line}"
        f'<div class="dm-meta">источник: {_e(mol.source)}</div>'
        f'<div class="dm-card-body">{_structure_html(mol.smiles)}{_values_html(cand)}</div>'
        f"{_filters_html(cand)}"
        "</div>"
    )


# --------------------------------------------------------------------------
# Экран отказа
# --------------------------------------------------------------------------

def _field(label: str, value: str, cls: str = "") -> str:
    return (f'<div><div class="dm-flabel">{label}</div>'
            f'<div class="dm-fval {cls}">{_e(value)}</div></div>')


def _render_refusal(missing: list[MissingData]) -> None:
    """status=insufficient_data. Это не ошибка, а результат работы системы:
    ни жёлтого, ни красного, ни значка сбоя."""
    cards = "".join(
        '<div class="dm-miss">'
        + _field("Что искали", m.what, "dm-fval-what")
        + '<div class="dm-miss-pair">'
        + _field("Найдено", m.found)
        + _field("Нужно", m.needed)
        + "</div>"
        + _field("Что это разблокирует", m.would_unlock)
        + "</div>"
        for m in missing
    )
    _md(
        '<div class="dm-refusal">'
        '<div class="dm-refusal-kicker">Недостаточно данных</div>'
        '<div class="dm-refusal-title">Молекулы не предложены</div>'
        '<div class="dm-refusal-sub">Чего не хватило:</div>'
        f'<div class="dm-miss-grid">{cards}</div>'
        "</div>"
    )


# --------------------------------------------------------------------------
# Страница
# --------------------------------------------------------------------------

st.set_page_config(page_title="drug-mas", layout="wide")
_md(_CSS)
_md(
    '<h1 class="dm-title">Мультиагентная система для дизайна лекарственных молекул</h1>'
    '<div class="dm-sub">ХимБио-хакатон «Синтез» · кейс 1 · ПИШ ИТМО</div>'
)

# --- (a) запрос на естественном языке ---
if "result" not in st.session_state:
    st.session_state["result"] = _fixture() if USE_FIXTURE else None

# Поле и кнопка запуска в одной строке: каждая лишняя строка сверху
# выталкивает результат за нижний край проектора.
q_col, run_col = st.columns([4, 1], vertical_alignment="bottom")
query = q_col.text_input(
    "Запрос на естественном языке",
    placeholder="Модифицируй иматиниб, повысив растворимость...",
)
if run_col.button("Запустить свой запрос", type="primary", use_container_width=True):
    st.session_state["result"] = _fixture() if USE_FIXTURE else _run_real(query)

PRESETS = {
    "Сценарий 1 — оптимизация иматиниба":
        "Модифицируй структуру иматиниба так, чтобы повысить растворимость "
        "и сохранить сродство к BCR-ABL1, отфильтруй токсичные аналоги",
    "Сценарий 2 — кандидаты под EGFR":
        "Найди известные ингибиторы EGFR (UniProt P00533), обучи модель "
        "активности и предложи 3 новых кандидата",
    "Сценарий 3 — отказ при нехватке данных":
        "Предложи ингибиторы для малоизученной мишени с малым числом "
        "активностей в ChEMBL",
}

# Кнопки-пресеты, а не только свободный ввод: при OFFLINE=1 произвольный
# запрос даст OfflineMiss и трейсбек на проекторе.
cols = st.columns(len(PRESETS))
for col, (label, preset) in zip(cols, PRESETS.items()):
    if col.button(label, use_container_width=True):
        query = preset
        st.session_state["result"] = _fixture() if USE_FIXTURE else _run_real(preset)

result: ScenarioResult | None = st.session_state["result"]
if result is None:
    # До первого клика на экране должно быть приглашение, а не сообщение
    # об ошибке: открывать защиту кадром «граф не вернул результат» нельзя,
    # а инструкция «не забудь нажать» забывается ровно тогда, когда нельзя.
    _md(
        '<div class="dm-refusal">'
        '<div class="dm-refusal-kicker">Готово к запуску</div>'
        '<div class="dm-refusal-title">Выберите сценарий</div>'
        '<div class="dm-refusal-sub">Три подготовленных запроса выше, '
        'либо свой запрос на естественном языке. Прогон идёт из локального '
        'кэша: сеть не требуется.</div>'
        '<div class="dm-miss-grid">'
        + "".join(
            '<div class="dm-miss">'
            f'<div class="dm-flabel">{lbl}</div>'
            f'<div class="dm-fval">{txt}</div></div>'
            for lbl, txt in (
                ("Происхождение каждого числа",
                 "База данных · расчёт · прогноз модели · гипотеза"),
                ("При нехватке данных",
                 "Система не предлагает молекулы, а называет дефицит"),
            )
        )
        + "</div></div>"
    )
    st.stop()

if result.scenario == "fixture":
    _md(
        '<div class="dm-note">Показана фикстура: graph.py ещё не подключён. '
        f'{_badge(Provenance.COMPUTED, "dm-b-inline")} — реальный RDKit по аспирину; '
        f'{_badge(Provenance.DATABASE, "dm-b-inline")}, '
        f'{_badge(Provenance.PREDICTED, "dm-b-inline")} и '
        f'{_badge(Provenance.HYPOTHESIS, "dm-b-inline")} помечены FIXTURE в source.</div>'
    )

# --- (b) лог вызовов инструментов ---
_section("Журнал вызовов инструментов")
_render_tool_calls(result.tool_calls)

# --- (c) кандидаты: структура + величины с плашками provenance ---
_section("Кандидаты")
if result.status == "insufficient_data":
    _render_refusal(result.missing)
elif not result.candidates:
    _md('<div class="dm-note">Кандидатов нет.</div>')
else:
    if result.status == "no_candidates_passed":
        _md('<div class="dm-note">Все кандидаты провалили пороги — '
            "показываем с причинами отсева.</div>")
    for cand in result.candidates:
        _render_candidate(cand)

# --- (d) ограничения ---
if result.limitations:
    _section("Ограничения прогона")
    _md('<ul class="dm-lims">'
        + "".join(f"<li>{_e(lim)}</li>" for lim in result.limitations)
        + "</ul>")
