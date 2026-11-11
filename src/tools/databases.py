"""
Клиенты внешних баз. ВЛАДЕЛЕЦ: Р-B. Блок T2.

ВСЁ через @cached — прямых запросов в сеть быть не должно, иначе
на защите с OFFLINE=1 демо встанет.
"""
from __future__ import annotations

from src.cache import cached
from src.schemas import Molecule, Value, from_db


@cached("pubchem_smiles")
def pubchem_smiles(name: str) -> dict:
    """TODO Р-B: PUG-REST /compound/name/{name}/property/CanonicalSMILES,IsomericSMILES/JSON
    Вернуть {"smiles":..., "cid":..., "iupac":...}. Для иматиниба CID 5291."""
    raise NotImplementedError


@cached("chembl_activities")
def chembl_activities(target_chembl_id: str, standard_type: str = "IC50") -> list[dict]:
    """TODO Р-B: chembl_webresource_client, activity.filter(...).
    Вернуть [{"smiles":..., "value": nM, "type":..., "chembl_id":...}].
    Для EGFR target_chembl_id = CHEMBL203."""
    raise NotImplementedError


@cached("uniprot_target")
def uniprot_target(uniprot_id: str) -> dict:
    """TODO Р-B: rest.uniprot.org/uniprotkb/{id}.json -> имя, ген, длина, перекрёстные ссылки на PDB."""
    raise NotImplementedError


@cached("pdb_structure")
def pdb_structure(pdb_id: str) -> dict:
    """TODO Р-B: files.rcsb.org/download/{id}.pdb + RCSB Data API.
    ВАЖНО: вернуть координаты центра ко-кристаллизованного лиганда — это бокс для докинга.
    Для BCR-ABL1 с иматинибом смотреть 1IEP / 2HYY."""
    raise NotImplementedError
