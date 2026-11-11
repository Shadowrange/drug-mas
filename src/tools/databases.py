"""
Клиенты внешних баз. ВЛАДЕЛЕЦ: Р-B. Блок T2.

⚠️ НЕ ПРОВЕРЕНО СЕТЕВЫМИ ВЫЗОВАМИ. Писалось в песочнице без доступа
к PubChem/ChEMBL/UniProt/PDB. Первый прогон у вас — это и есть тест;
при расхождении формата ответа правьте разбор, а не логику кэша.

ВСЁ через @cached — прямых запросов в сеть быть не должно, иначе
на защите с OFFLINE=1 демо встанет. Возвращаемые объекты обязаны быть
JSON-сериализуемыми: в cache.py убран default=str, и numpy/float32
уронят запись громко, а не превратятся молча в строку.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from src.cache import cached

USER_AGENT = "drug-mas/0.1 (ChemBio hackathon Sintez, ITMO; research use)"
TIMEOUT = 30


class ExternalSourceError(RuntimeError):
    """Внешний источник недоступен или ответил непонятным. Громко, без заглушки."""


def _get(url: str, accept: str = "application/json") -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise ExternalSourceError(f"HTTP {exc.code} от {url}") from exc
    except Exception as exc:
        raise ExternalSourceError(f"{type(exc).__name__} при запросе {url}: {exc}") from exc


# --------------------------------------------------------------------------- PubChem

@cached("pubchem_smiles")
def pubchem_smiles(name: str) -> dict:
    """
    Структура соединения по названию через PUG-REST.

    PubChem в разное время отдаёт свойство под именами CanonicalSMILES,
    IsomericSMILES, ConnectivitySMILES или просто SMILES — поэтому просим
    несколько и берём первое непустое, а не полагаемся на одно имя.
    """
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    props = "IsomericSMILES,CanonicalSMILES,SMILES,IUPACName,MolecularFormula"
    url = f"{base}/{urllib.parse.quote(name)}/property/{props}/JSON"

    try:
        payload = json.loads(_get(url))
    except ExternalSourceError:
        # Запасной путь: часть свойств может не поддерживаться — просим минимум.
        url = f"{base}/{urllib.parse.quote(name)}/property/SMILES,IUPACName/JSON"
        payload = json.loads(_get(url))

    try:
        rec = payload["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError) as exc:
        raise ExternalSourceError(f"PubChem: соединение '{name}' не найдено или ответ без Properties") from exc

    smiles = next(
        (rec[k] for k in ("IsomericSMILES", "CanonicalSMILES", "SMILES") if rec.get(k)),
        None,
    )
    if not smiles:
        raise ExternalSourceError(f"PubChem: в ответе по '{name}' нет ни одного поля со SMILES: {list(rec)}")

    return {
        "query": name,
        "smiles": smiles,
        "cid": rec.get("CID"),
        "iupac": rec.get("IUPACName"),
        "formula": rec.get("MolecularFormula"),
        "source": f"PubChem CID {rec.get('CID')}",
    }


# --------------------------------------------------------------------------- UniProt

@cached("uniprot_target")
def uniprot_target(uniprot_id: str) -> dict:
    """Карточка мишени: имя, ген, длина, перекрёстные ссылки на PDB."""
    url = f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(uniprot_id)}.json"
    data = json.loads(_get(url))

    name = (
        data.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )
    genes = [g.get("geneName", {}).get("value") for g in data.get("genes", [])]
    pdb_ids = [
        x.get("id") for x in data.get("uniProtKBCrossReferences", [])
        if x.get("database") == "PDB"
    ]

    return {
        "uniprot": uniprot_id,
        "name": name,
        "genes": [g for g in genes if g],
        "length": data.get("sequence", {}).get("length"),
        "organism": data.get("organism", {}).get("scientificName"),
        "pdb_ids": pdb_ids[:50],
        "source": f"UniProt {uniprot_id}",
    }


# --------------------------------------------------------------------------- ChEMBL

@cached("chembl_activities")
def chembl_activities(
    target_chembl_id: str,
    standard_type: str = "IC50",
    limit: int = 2000,
) -> list[dict]:
    """
    Измеренные активности по мишени.

    ⚠️ target_chembl_id проверяйте поиском по ChEMBL, а не по памяти:
    подставленный наугад правдоподобный ID даст пустую выборку, система
    уйдёт в отказ по нехватке данных, и вы полчаса будете искать причину
    не там. Это ровно тот случай, против которого написан AGENTS.md §2.

    Единицы приводятся к нМ и отфильтровываются записи без структуры или
    без значения: запись без числа — это не ноль, а отсутствие данных.
    Пустой список — валидный ответ; достаточность считает
    verifier.check_data_sufficiency, а не эта функция.
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError as exc:
        raise ExternalSourceError(
            "Не установлен chembl-webresource-client: pip install chembl-webresource-client"
        ) from exc

    try:
        qs = new_client.activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type=standard_type,
            standard_relation="=",
        ).only(
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "standard_units",
            "standard_type",
            "assay_chembl_id",
        )
    except Exception as exc:
        raise ExternalSourceError(f"ChEMBL: сбой запроса по {target_chembl_id}: {exc}") from exc

    out: list[dict] = []
    for rec in qs:
        if len(out) >= limit:
            break
        smi = rec.get("canonical_smiles")
        val = rec.get("standard_value")
        units = (rec.get("standard_units") or "").lower()
        if not smi or val in (None, ""):
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if units == "nm":
            v_nm = v
        elif units in ("um", "µm"):
            v_nm = v * 1e3
        elif units == "mm":
            v_nm = v * 1e6
        elif units == "m":
            v_nm = v * 1e9
        else:
            continue  # неизвестные единицы не пересчитываем наугад
        out.append({
            "smiles": smi,
            "value": v_nm,
            "units": "nM",
            "type": rec.get("standard_type") or standard_type,
            "chembl_id": rec.get("molecule_chembl_id"),
            "assay_chembl_id": rec.get("assay_chembl_id"),
        })
    return out


# --------------------------------------------------------------------------- PDB

@cached("pdb_structure")
def pdb_structure(pdb_id: str, ligand_resname: str | None = None) -> dict:
    """
    Экспериментальная структура + координаты центра бокса для докинга.

    Центр бокса берётся по ко-кристаллизованному лиганду: для 1IEP
    (BCR-ABL1 с иматинибом) это резидент STI. Это и заменяет нам поиск
    кармана с нуля — см. §2 плана, направление Б урезано осознанно.

    Если лиганд не найден — center=None, и докинг не запускается.
    Выдумывать координаты нельзя: неверный бокс даёт правдоподобный,
    но бессмысленный ΔG.
    """
    pdb_id = pdb_id.strip().upper()
    text = _get(f"https://files.rcsb.org/download/{pdb_id}.pdb", accept="text/plain")

    center = None
    reason = None
    try:
        from src.tools import docking
        center = docking.extract_box_center(text, ligand_resname=ligand_resname)
        if center is None:
            reason = (
                f"в {pdb_id} не найден ко-кристаллизованный лиганд"
                + (f" с resname {ligand_resname}" if ligand_resname else "")
            )
    except Exception as exc:
        reason = f"не удалось определить центр бокса: {type(exc).__name__}: {exc}"

    resolution = None
    for line in text.splitlines():
        if line.startswith("REMARK   2 RESOLUTION."):
            for tok in line.split():
                try:
                    resolution = float(tok)
                    break
                except ValueError:
                    continue
            break

    return {
        "pdb_id": pdb_id,
        "pdb_text": text,
        "resolution_a": resolution,
        "ligand_resname": ligand_resname,
        "box_center": list(center) if center else None,
        "box_center_missing_reason": reason,
        "source": f"PDB {pdb_id}",
    }
