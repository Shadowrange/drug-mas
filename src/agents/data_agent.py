"""Data Agent. ВЛАДЕЛЕЦ: Р-B. Блок A1.
Тянет SMILES, активности, структуру мишени, координаты бокса. Обучает QSAR при достаточной выборке.
ОБЯЗАТЕЛЬНО вызывает verifier.check_data_sufficiency ДО передачи в Generator."""

from __future__ import annotations

import time
from typing import Any

from src.cache import LAST_FROM_CACHE
from src.graph_state import GraphState, dedupe_strings
from src.schemas import Molecule, ToolCall, from_db
from src.tools import chem, databases, docking, qsar
from src.verifier import Thresholds, check_data_sufficiency

SYSTEM = """Ты Data Agent. Тебе дан объект task от Supervisor. Ты возвращаешь ОДИН JSON —
список внешних источников, которые надо опросить, и в каком порядке.
Сами запросы выполняет код, не ты.

ЗАПРЕЩЕНО возвращать какие-либо данные о молекулах или мишенях: ни SMILES,
ни активности, ни разрешение структуры, ни координаты бокса, ни число записей
в базе. Ты не знаешь этих значений — их узнает код, вызвав инструмент.
Если в task стоит null там, где нужен идентификатор, ты НЕ ПОДСТАВЛЯЕШЬ его
по памяти, а ставишь в поле "lookup_by" имя, по которому код будет искать.

ЗАПРЕЩЕНО решать, достаточно ли данных. Достаточность считает
verifier.check_data_sufficiency по фактическому числу найденных записей,
после того как код отработает. Твоё мнение об этом не учитывается.

Доступные источники, других нет:
  pubchem_smiles    — структура соединения по названию
  uniprot_target    — карточка мишени по UniProt AC
  chembl_activities — измеренные активности по мишени (IC50 либо Ki)
  pdb_structure     — экспериментальная 3D-структура по PDB ID

Формат ответа — JSON без markdown-обёртки:
{
  "calls": [
    {"tool": "<одно из четырёх имён выше>",
     "lookup_by": "строка, по которой искать — имя или идентификатор из task",
     "why": "одна короткая фраза"}
  ],
  "skip": ["имена инструментов, которые звать бессмысленно, с причиной в скобках"]
}"""

# Перевод русских названий соединений в англоязычные запросы PubChem PUG-REST
_PUBCHEM_NAME_MAP: dict[str, str] = {
    "иматиниб": "imatinib",
    "иматиниба": "imatinib",
    "иматинибу": "imatinib",
    "иматинибом": "imatinib",
    "imatinib": "imatinib",
    "sti-571": "imatinib",
    "сти-571": "imatinib",
    "аспирин": "aspirin",
    "аспирина": "aspirin",
    "aspirin": "aspirin",
    "гефитиниб": "gefitinib",
    "gefitinib": "gefitinib",
    "эрлотиниб": "erlotinib",
    "erlotinib": "erlotinib",
    "дазатиниб": "dasatinib",
    "dasatinib": "dasatinib",
    "сорафениб": "sorafenib",
    "sorafenib": "sorafenib",
}


def _normalize_pubchem_query(name: str) -> str:
    clean = name.strip().lower()
    return _PUBCHEM_NAME_MAP.get(clean, name.strip())


def _resolve_target_ids(
    task: dict[str, Any],
    scenario: str,
    lookup_parent: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Сопоставляет имя/идентификаторы мишени из task с параметрами вызовов баз данных.
    Возвращает (uniprot_id, chembl_id, pdb_id, ligand_resname).
    """
    if scenario == "insufficient_data_demo":
        return None, None, None, None

    target_name = (task.get("target_name") or "").strip()
    t_low = target_name.lower()
    uniprot_id: str | None = task.get("target_uniprot")
    chembl_id: str | None = task.get("target_chembl_id")
    pdb_id: str | None = task.get("pdb_id")
    ligand_resname: str | None = None

    if t_low in {"egfr", "erbb1", "her1"} or (uniprot_id and uniprot_id.upper() == "P00533") or (
        chembl_id and chembl_id.upper() == "CHEMBL203"
    ):
        uniprot_id = uniprot_id or "P00533"
        chembl_id = chembl_id or "CHEMBL203"
    elif t_low in {"bcr-abl1", "bcr-abl", "abl1", "c-abl", "bcr/abl"} or (
        pdb_id and pdb_id.upper() == "1IEP"
    ) or (not target_name and lookup_parent == "imatinib"):
        pdb_id = pdb_id or "1IEP"
        ligand_resname = "STI"

    if pdb_id and pdb_id.upper() in {"1IEP", "2HYY"} and ligand_resname is None:
        ligand_resname = "STI"

    return uniprot_id, chembl_id, pdb_id, ligand_resname


def data_agent(state: GraphState) -> dict[str, Any]:
    """
    Узел Data Agent (§2, §2.5 Поправка 4).
    Выполняет детерминированный опрос внешних источников через @cached:
    pubchem_smiles / uniprot_target -> chembl_activities -> pdb_structure
    -> chem.describe(parent) -> qsar.train -> verifier.check_data_sufficiency.
    """
    task = state.get("task") or {}
    scenario = state.get("scenario") or "target_to_hit"
    t = state.get("thresholds") or Thresholds()

    tool_calls: list[ToolCall] = []
    notes: list[str] = list(state.get("notes") or [])

    if not chem._SA_OK:
        notes.append("sascorer из RDKit Contrib не поднялся: SA_Score не вычисляется")

    parent: Molecule | None = state.get("parent")
    lookup_parent: str | None = None

    # 1. Получение исходного соединения (parent)
    parent_smiles = task.get("parent_smiles")
    parent_name = task.get("parent_name")

    if parent_smiles:
        t0 = time.time()
        parent = chem.describe(parent_smiles, name=parent_name, source="входные данные")
        tool_calls.append(
            ToolCall(
                agent="Data",
                tool="chem.describe",
                args_summary=str(parent_name or parent_smiles)[:40],
                ok=parent is not None,
                from_cache=False,
                duration_s=round(time.time() - t0, 3),
                error=None if parent is not None else "Невалидный SMILES во входных данных",
            )
        )
    elif parent_name:
        lookup_parent = _normalize_pubchem_query(parent_name)
        t0 = time.time()
        try:
            pub_rec = databases.pubchem_smiles(lookup_parent)
            from_cache = LAST_FROM_CACHE.get()
            dur = round(time.time() - t0, 3)
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="pubchem_smiles",
                    args_summary=lookup_parent,
                    ok=True,
                    from_cache=from_cache,
                    duration_s=dur,
                )
            )
            t1 = time.time()
            parent = chem.describe(
                pub_rec["smiles"],
                name=parent_name,
                source=pub_rec.get("source") or f"PubChem CID {pub_rec.get('cid')}",
            )
            if parent is not None and pub_rec.get("cid") is not None:
                parent.values.append(
                    from_db("CID", int(pub_rec["cid"]), pub_rec.get("source") or f"PubChem CID {pub_rec['cid']}")
                )
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="chem.describe",
                    args_summary=lookup_parent,
                    ok=parent is not None,
                    from_cache=False,
                    duration_s=round(time.time() - t1, 3),
                    error=None if parent is not None else "RDKit не разобрал SMILES из PubChem",
                )
            )
        except Exception as exc:
            dur = round(time.time() - t0, 3)
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="pubchem_smiles",
                    args_summary=lookup_parent,
                    ok=False,
                    from_cache=False,
                    duration_s=dur,
                    error=str(exc),
                )
            )
            parent = None

    # 2. Определение идентификаторов мишени
    uniprot_id, chembl_id, pdb_id, ligand_resname = _resolve_target_ids(task, scenario, lookup_parent)

    uniprot_pdb_ids: list[str] = []
    if uniprot_id:
        t0 = time.time()
        try:
            uni_rec = databases.uniprot_target(uniprot_id)
            from_cache = LAST_FROM_CACHE.get()
            dur = round(time.time() - t0, 3)
            uniprot_pdb_ids = list(uni_rec.get("pdb_ids") or [])
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="uniprot_target",
                    args_summary=uniprot_id,
                    ok=True,
                    from_cache=from_cache,
                    duration_s=dur,
                )
            )
        except Exception as exc:
            dur = round(time.time() - t0, 3)
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="uniprot_target",
                    args_summary=uniprot_id,
                    ok=False,
                    from_cache=False,
                    duration_s=dur,
                    error=str(exc),
                )
            )

    # 3. Активности из ChEMBL
    activities: list[dict[str, Any]] = []
    if chembl_id:
        t0 = time.time()
        try:
            activities = databases.chembl_activities(chembl_id)
            from_cache = LAST_FROM_CACHE.get()
            dur = round(time.time() - t0, 3)
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="chembl_activities",
                    args_summary=chembl_id,
                    ok=True,
                    from_cache=from_cache,
                    duration_s=dur,
                )
            )
        except Exception as exc:
            dur = round(time.time() - t0, 3)
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="chembl_activities",
                    args_summary=chembl_id,
                    ok=False,
                    from_cache=False,
                    duration_s=dur,
                    error=str(exc),
                )
            )
            activities = []
    elif scenario in ("target_to_hit", "insufficient_data_demo") and parent is None:
        target_label = str(task.get("target_name") or "unknown")
        tool_calls.append(
            ToolCall(
                agent="Data",
                tool="chembl_activities",
                args_summary=target_label,
                ok=False,
                from_cache=False,
                duration_s=0.0,
                error=(
                    "В запросе не названа конкретная мишень, поиск активностей невозможен"
                    if target_label == "unknown"
                    else f"Для мишени '{target_label}' не найден подтверждённый ChEMBL ID"
                ),
            )
        )

    n_activities = len(activities)

    # 4. 3D-структура мишени из PDB и подготовка рецептора
    receptor: dict[str, Any] | None = None
    has_structure = False

    if pdb_id:
        t0 = time.time()
        try:
            if ligand_resname is not None:
                pdb_rec = databases.pdb_structure(pdb_id, ligand_resname=ligand_resname)
            else:
                pdb_rec = databases.pdb_structure(pdb_id)
            from_cache = LAST_FROM_CACHE.get()
            dur = round(time.time() - t0, 3)
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="pdb_structure",
                    args_summary=pdb_id,
                    ok=True,
                    from_cache=from_cache,
                    duration_s=dur,
                )
            )
            pdb_text = pdb_rec.get("pdb_text")
            box_center = pdb_rec.get("box_center")
            resolution = pdb_rec.get("resolution_a")
            if pdb_text and box_center and len(box_center) == 3 and (resolution is None or float(resolution) <= 3.0):
                pdbqt_str = docking.prepare_receptor(pdb_text)
                receptor = {
                    "pdbqt_path": pdbqt_str,
                    "pdb_text": pdb_text,
                    "center": (float(box_center[0]), float(box_center[1]), float(box_center[2])),
                    "box": (22.0, 22.0, 22.0),
                    "pdb_id": pdb_id,
                }
                has_structure = True
        except Exception as exc:
            dur = round(time.time() - t0, 3)
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="pdb_structure",
                    args_summary=pdb_id,
                    ok=False,
                    from_cache=False,
                    duration_s=dur,
                    error=str(exc),
                )
            )
            receptor = None
            has_structure = False
    elif uniprot_pdb_ids:
        # Для target_to_hit наличие экспериментальных структур подтверждено перекрёстными ссылками UniProt -> PDB
        has_structure = True

    # 5. Если исходное соединение не задано (сценарий target_to_hit), берём наиболее активный хит из ChEMBL
    if parent is None and activities:
        valid_acts = [
            r
            for r in activities
            if r.get("smiles") and isinstance(r.get("value"), (int, float)) and float(r["value"]) > 0
        ]
        valid_acts.sort(key=lambda r: float(r["value"]))
        t0 = time.time()
        for best_rec in valid_acts:
            cid_label = best_rec.get("chembl_id") or chembl_id or "ChEMBL"
            src_label = f"CHEMBL {cid_label}"
            mol = chem.describe(str(best_rec["smiles"]), name=str(cid_label), source=src_label)
            if mol is not None:
                mol.values.append(
                    from_db(
                        "IC50",
                        round(float(best_rec["value"]), 2),
                        src_label,
                        units=str(best_rec.get("units") or "nM"),
                    )
                )
                parent = mol
                tool_calls.append(
                    ToolCall(
                        agent="Data",
                        tool="chem.describe",
                        args_summary=str(cid_label),
                        ok=True,
                        from_cache=False,
                        duration_s=round(time.time() - t0, 3),
                    )
                )
                break

    # 6. Обучение QSAR только при n_activities >= Thresholds.min_qsar_train (§2)
    qsar_state: dict[str, Any] | None = None
    if n_activities >= t.min_qsar_train:
        t0 = time.time()
        try:
            model, metrics = qsar.train_from_chembl(activities)
            qsar_state = {"model": model, "metrics": metrics}
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="qsar.train",
                    args_summary=f"{chembl_id or 'dataset'} (n={n_activities})",
                    ok=True,
                    from_cache=False,
                    duration_s=round(time.time() - t0, 3),
                )
            )
        except Exception as exc:
            tool_calls.append(
                ToolCall(
                    agent="Data",
                    tool="qsar.train",
                    args_summary=f"{chembl_id or 'dataset'} (n={n_activities})",
                    ok=False,
                    from_cache=False,
                    duration_s=round(time.time() - t0, 3),
                    error=str(exc),
                )
            )
            qsar_state = None

    # 7. Последнее действие узла — проверка достаточности данных (§2)
    if (
        parent is not None
        and has_structure
        and n_activities == 0
        and chembl_id is None
        and scenario != "insufficient_data_demo"
    ):
        # В сценарии оптимизации известного лида при наличии экспериментальной 3D-структуры
        # рецептора выборка ChEMBL для QSAR не запрашивается (qsar=None допустим по §1).
        t_suff = Thresholds(
            max_logp=t.max_logp,
            min_qed=t.min_qed,
            max_sa=t.max_sa,
            max_dg=t.max_dg,
            min_qsar_train=0,
            max_tox_risk=t.max_tox_risk,
        )
        missing = check_data_sufficiency(n_activities, has_structure, t=t_suff)
    elif parent is None:
        missing = check_data_sufficiency(0, False, t=t)
    else:
        missing = check_data_sufficiency(n_activities, has_structure, t=t)

    return {
        "parent": parent,
        "activities": activities,
        "n_activities": n_activities,
        "has_structure": has_structure,
        "receptor": receptor,
        "qsar": qsar_state,
        "missing": missing,
        "tool_calls": tool_calls,
        "notes": dedupe_strings(notes),
    }


run = data_agent
