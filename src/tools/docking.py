"""
Докинг. ВЛАДЕЛЕЦ: Р-A. Блок D1 — ГЛАВНЫЙ РИСК ПРОЕКТА (R1).

Жёсткий дедлайн спайка: 01:00 среды. Не заработало -> план Б из PLAN §10:
ΔG не считаем вообще и честно помечаем как not_computed. Выдумывать ΔG нельзя
ни при каких обстоятельствах — на этом строится вся защита.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from src.schemas import Value, computed, not_computed
from src.tools import chem

try:
    import vina as _vina_mod  # type: ignore

    _VINA_PY_OK = True
    _VINA_VER = getattr(_vina_mod, "__version__", "1.2.x")
except ImportError:
    _vina_mod = None
    _VINA_PY_OK = False
    _VINA_VER = "1.2.x"

try:
    import meeko as _meeko_mod  # type: ignore

    _MEEKO_OK = True
except ImportError:
    _meeko_mod = None
    _MEEKO_OK = False

try:
    from plip.structure.preparation import PDBComplex as _PLIPComplex  # type: ignore

    _PLIP_OK = True
except ImportError:
    _PLIPComplex = None
    _PLIP_OK = False

_SOLVENT_AND_IONS = {
    "HOH", "WAT", "TIP", "SOL", "DOD",
    "SO4", "PO4", "NO3", "ACT", "GOL", "EDO", "PEG", "DMS", "MPD", "TRS", "MES",
    "NA", "CL", "MG", "ZN", "CA", "K", "MN", "FE", "CU", "CO", "NI", "BR", "IOD",
}

_HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}
_CHARGED_RES = {"ASP", "GLU", "LYS", "ARG", "HIS"}
_AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}


@dataclass(frozen=True)
class _PDBAtom:
    record: str
    serial: int
    atom_name: str
    res_name: str
    chain_id: str
    res_seq: int
    x: float
    y: float
    z: float
    element: str


def _read_pdb_text(path_or_block: str) -> str:
    """Принимает либо путь к PDB/PDBQT-файлу, либо сам текст блока."""
    if not path_or_block or not isinstance(path_or_block, str):
        return ""
    if "\n" not in path_or_block and len(path_or_block) < 512:
        p = Path(path_or_block)
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
        raise FileNotFoundError(f"Файл структуры не найден: {path_or_block}")
    return path_or_block


def _parse_pdb_atoms(pdb_text: str, records: tuple[str, ...] = ("ATOM", "HETATM")) -> list[_PDBAtom]:
    """Разбирает атомы из стандартных колонок PDB/PDBQT без внешних зависимостей."""
    atoms: list[_PDBAtom] = []
    for line in pdb_text.splitlines():
        if len(line) < 54:
            continue
        rec = line[:6].strip()
        if rec not in records:
            continue
        try:
            serial = int(line[6:11].strip() or "0")
        except ValueError:
            serial = 0
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain_id = line[21:22].strip() or "A"
        try:
            res_seq = int(line[22:26].strip() or "0")
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError:
            continue

        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            cleaned = re.sub(r"[^A-Za-z]", "", atom_name)
            elem = cleaned[:2].capitalize() if cleaned[:2].upper() in {"CL", "BR", "FE", "ZN", "MG", "CA", "NA"} else (cleaned[:1].upper() or "C")
        else:
            elem = elem.capitalize()
            if elem.upper() in {"HD", "HS"}:
                elem = "H"
            elif elem.upper() in {"OA", "OS"}:
                elem = "O"
            elif elem.upper() in {"NA", "NS"}:
                elem = "N"
            elif elem.upper() in {"SA"}:
                elem = "S"
            elif elem.upper() == "A":
                elem = "C"

        atoms.append(
            _PDBAtom(
                record=rec,
                serial=serial,
                atom_name=atom_name,
                res_name=res_name,
                chain_id=chain_id,
                res_seq=res_seq,
                x=x,
                y=y,
                z=z,
                element=elem,
            )
        )
    return atoms


def extract_box_center(pdb_path_or_block: str, ligand_resname: str | None = None) -> tuple[float, float, float] | None:
    """
    Вычисляет геометрический центр ко-кристаллизованного лиганда из записей HETATM
    (например, STI для иматиниба в 1IEP/2HYY), игнорируя молекулы воды и буферные ионы.
    """
    try:
        text = _read_pdb_text(pdb_path_or_block)
    except FileNotFoundError:
        return None
    het_atoms = [
        a for a in _parse_pdb_atoms(text, records=("HETATM",))
        if a.res_name.upper() not in _SOLVENT_AND_IONS and a.element != "H"
    ]
    if not het_atoms:
        return None

    if ligand_resname:
        target = ligand_resname.strip().upper()
        selected = [a for a in het_atoms if a.res_name.upper() == target]
    else:
        # Выбираем самый крупный гетероатомный остаток (сам лиганд в комплексе)
        by_res: dict[tuple[str, str, int], list[_PDBAtom]] = {}
        for a in het_atoms:
            key = (a.res_name, a.chain_id, a.res_seq)
            by_res.setdefault(key, []).append(a)
        selected = max(by_res.values(), key=len)

    if not selected:
        return None

    n = len(selected)
    cx = round(sum(a.x for a in selected) / n, 3)
    cy = round(sum(a.y for a in selected) / n, 3)
    cz = round(sum(a.z for a in selected) / n, 3)
    return (cx, cy, cz)


def _autodock_atom_type(element: str, is_aromatic: bool = False, is_hbond_acceptor: bool = False, is_polar_h: bool = False) -> str:
    """Сопоставление элемента с типом атома AutoDock4/Vina."""
    el = element.capitalize()
    if el == "H":
        return "HD" if is_polar_h else "H"
    if el == "C":
        return "A" if is_aromatic else "C"
    if el == "N":
        return "NA" if is_hbond_acceptor else "N"
    if el == "O":
        return "OA"
    if el == "S":
        return "SA" if is_hbond_acceptor else "S"
    if el in {"F", "Cl", "Br", "I", "P", "Mg", "Zn", "Ca", "Mn", "Fe"}:
        return el
    return el[:2]


def prepare_receptor(pdb_path: str) -> str:
    """
    Конвертация структуры рецептора (PDB) в формат PDBQT.
    Использует OpenBabel/Meeko при наличии в системе, иначе детерминированную
    очистку от воды/лигандов с расстановкой типов атомов AutoDock Vina.
    Возвращает путь к .pdbqt-файлу (если передан путь) или строку PDBQT.
    """
    text = _read_pdb_text(pdb_path)
    if not text.strip():
        raise ValueError("Пустое содержимое структуры рецептора")

    is_path = "\n" not in pdb_path and Path(pdb_path).is_file()

    # Если внешний obabel установлен и на входе файл — используем его
    obabel_bin = shutil.which("obabel")
    if is_path and obabel_bin:
        in_p = Path(pdb_path)
        out_p = in_p.with_suffix(".pdbqt")
        cmd = [obabel_bin, "-ipdb", str(in_p), "-opdbqt", "-O", str(out_p), "-xr"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and out_p.is_file() and out_p.stat().st_size > 0:
            return str(out_p)

    # Детерминированная подготовка PDBQT: оставляем только белковые ATOM, убираем воду
    pdbqt_lines: list[str] = []
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec != "ATOM" or len(line) < 54:
            continue
        res_name = line[17:20].strip().upper()
        if res_name in _SOLVENT_AND_IONS:
            continue
        atom_name = line[12:16].strip()
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            cleaned = re.sub(r"[^A-Za-z]", "", atom_name)
            elem = cleaned[:1].upper() if cleaned else "C"
        is_arom = res_name in _AROMATIC_RES and atom_name not in {"N", "CA", "C", "O", "CB"} and elem.upper() == "C"
        is_polar_h = elem.upper() == "H" and any( atom_name.startswith(p) for p in ("H", "HN", "HG", "HH", "HD", "HE", "HZ") )
        ad_type = _autodock_atom_type(elem, is_aromatic=is_arom, is_hbond_acceptor=elem.upper() in {"O", "N"}, is_polar_h=is_polar_h)
        base54 = line[:54]
        pdbqt_lines.append(f"{base54}  1.00  0.00    +0.000 {ad_type:<2s}")

    if not pdbqt_lines:
        raise ValueError("В структуре рецептора не найдено белковых записей ATOM")

    pdbqt_content = "\n".join(pdbqt_lines) + "\n"
    if is_path:
        out_p = Path(pdb_path).with_suffix(".pdbqt")
        try:
            out_p.write_text(pdbqt_content, encoding="utf-8")
            return str(out_p)
        except OSError:
            pass
    return pdbqt_content


def _ligand_to_pdbqt(smiles: str) -> tuple[str | None, str | None]:
    """
    Строит 3D-конформер лиганда (ETKDGv3 + MMFF94) и переводит в PDBQT.
    Возвращает (pdbqt_block, pdb_block) или (None, None) при неудаче.
    """
    pdb_block = chem.embed_3d(smiles)
    if not pdb_block:
        return None, None

    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
    if mol is None:
        return None, None

    if _MEEKO_OK and hasattr(_meeko_mod, "MoleculePreparation"):
        try:
            preparator = _meeko_mod.MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            if hasattr(_meeko_mod, "PDBQTWriterLegacy"):
                pdbqt_str, is_ok, _ = _meeko_mod.PDBQTWriterLegacy.write_string(mol_setups[0])
                if is_ok and pdbqt_str:
                    return pdbqt_str, pdb_block
            elif hasattr(preparator, "write_pdbqt_string"):
                return preparator.write_pdbqt_string(), pdb_block
        except Exception:
            pass

    # ЗАПАСНОЙ ПУТЬ БЕЗ MEEKO: записывается ОДИН ROOT и TORSDOF 0, то есть лиганд
    # докуется АБСОЛЮТНО ЖЁСТКИМ. Для молекулы с вращаемыми связями Vina вернёт
    # энергию одного случайного конформера, а не результат докинга — число будет
    # выглядеть законным (computed, "AutoDock Vina"), но будет неверным.
    # Молчаливое правдоподобное число — ровно то, против чего построен verifier.py,
    # поэтому здесь мы отказываемся считать, а не считаем плохо.
    n_rot = Descriptors.NumRotatableBonds(Chem.RemoveHs(mol))
    if n_rot > 0:
        return None, pdb_block

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    conf = mol.GetConformer()
    lines = ["REMARK  Name = ligand", "ROOT"]
    for idx, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(idx - 1)
        sym = atom.GetSymbol()
        try:
            q = float(atom.GetProp("_GasteigerCharge"))
            if not math.isfinite(q):
                q = 0.0
        except Exception:
            q = 0.0
        is_arom = atom.GetIsAromatic()
        is_polar_h = sym == "H" and any(nb.GetSymbol() in {"N", "O", "S"} for nb in atom.GetNeighbors())
        is_acc = sym in {"N", "O", "S"} and atom.GetTotalValence() <= 3
        ad_type = _autodock_atom_type(sym, is_aromatic=is_arom, is_hbond_acceptor=is_acc, is_polar_h=is_polar_h)
        atom_label = f"{sym}{idx}"[:4]
        lines.append(
            f"HETATM{idx:5d} {atom_label:<4s} LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {q:+6.3f} {ad_type:<2s}"
        )
    lines.append("ENDROOT")
    lines.append("TORSDOF 0")
    return "\n".join(lines) + "\n", pdb_block


def dock(
    ligand_smiles: str,
    receptor_pdbqt: str,
    center: tuple[float, float, float],
    box: tuple[float, float, float] = (22.0, 22.0, 22.0),
    exhaustiveness: int = 8,
) -> Value:
    """
    Докинг лиганда в активный центр рецептора (AutoDock Vina).
    Возвращает computed('dG', value, 'AutoDock Vina ...', 'ккал/моль').
    По плану Б (риск R1): если Vina недоступна или входные данные некорректны,
    возвращает честный not_computed('dG', reason=...) без выдумывания чисел.
    """
    if not chem.is_valid(ligand_smiles):
        return not_computed("dG", "Невалидный SMILES лиганда", source="AutoDock Vina")

    if not receptor_pdbqt or not isinstance(receptor_pdbqt, str) or not receptor_pdbqt.strip():
        return not_computed("dG", "Отсутствует структура рецептора (receptor_pdbqt)", source="AutoDock Vina")

    if (
        not center
        or len(center) != 3
        or not all(isinstance(c, (int, float)) and math.isfinite(float(c)) for c in center)
    ):
        return not_computed("dG", "Некорректные координаты центра бокса докинга", source="AutoDock Vina")

    ligand_pdbqt, _ = _ligand_to_pdbqt(ligand_smiles)
    if not ligand_pdbqt:
        return not_computed(
            "dG",
            "Не удалось сгенерировать 3D-конформер лиганда (RDKit ETKDGv3)",
            source="RDKit + AutoDock Vina",
        )

    vina_bin = shutil.which("vina")
    if not _VINA_PY_OK and not vina_bin:
        return not_computed(
            "dG",
            "AutoDock Vina не установлен в окружении (план Б по риску R1: ΔG не вычисляется без бинарника Vina)",
            source="AutoDock Vina (недоступен)",
        )

    # Обеспечиваем путь к файлу рецептора на диске для вызовов Vina
    tmp_rec_file = None
    try:
        if "\n" not in receptor_pdbqt and Path(receptor_pdbqt).is_file():
            rec_path = receptor_pdbqt
        else:
            rec_text = _read_pdb_text(receptor_pdbqt)
            tmp_rec_file = tempfile.NamedTemporaryFile("w", suffix=".pdbqt", delete=False, encoding="utf-8")
            tmp_rec_file.write(rec_text)
            tmp_rec_file.close()
            rec_path = tmp_rec_file.name

        if _VINA_PY_OK:
            v = _vina_mod.Vina(sf_name="vina", verbosity=0)
            v.set_receptor(rec_path)
            v.set_ligand_from_string(ligand_pdbqt)
            v.compute_vina_maps(center=[float(c) for c in center], box_size=[float(b) for b in box])
            v.dock(exhaustiveness=int(exhaustiveness), n_poses=5)
            energies = v.energies(n_poses=1)
            best_dg = round(float(energies[0][0]), 2)
            return computed("dG", best_dg, f"AutoDock Vina {_VINA_VER}", "ккал/моль")

        if vina_bin:
            with tempfile.NamedTemporaryFile("w", suffix=".pdbqt", delete=False, encoding="utf-8") as lig_f:
                lig_f.write(ligand_pdbqt)
                lig_path = lig_f.name
            out_path = lig_path + ".out.pdbqt"
            try:
                cmd = [
                    vina_bin,
                    "--receptor", rec_path,
                    "--ligand", lig_path,
                    "--center_x", str(center[0]),
                    "--center_y", str(center[1]),
                    "--center_z", str(center[2]),
                    "--size_x", str(box[0]),
                    "--size_y", str(box[1]),
                    "--size_z", str(box[2]),
                    "--exhaustiveness", str(exhaustiveness),
                    "--out", out_path,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
                if proc.returncode == 0 and Path(out_path).is_file():
                    out_txt = Path(out_path).read_text(encoding="utf-8", errors="replace")
                    m = re.search(r"REMARK VINA RESULT:\s+([-\d.]+)", out_txt)
                    if m:
                        return computed("dG", round(float(m.group(1)), 2), "AutoDock Vina CLI", "ккал/моль")
                return not_computed(
                    "dG",
                    f"Ошибка выполнения AutoDock Vina CLI: {(proc.stderr or proc.stdout).strip()[:160]}",
                    source="AutoDock Vina CLI",
                )
            finally:
                Path(lig_path).unlink(missing_ok=True)
                Path(out_path).unlink(missing_ok=True)

    except Exception as exc:
        return not_computed("dG", f"Сбой докинга: {type(exc).__name__}: {exc}", source="AutoDock Vina")
    finally:
        if tmp_rec_file is not None:
            Path(tmp_rec_file.name).unlink(missing_ok=True)

    return not_computed("dG", "AutoDock Vina недоступен", source="AutoDock Vina")


def interactions(pose_pdb: str, receptor_pdb: str) -> list[str]:
    """
    Анализ межмолекулярных контактов (водородные связи, ионные и гидрофобные взаимодействия)
    между 3D-позой лиганда и рецептором.
    Использует PLIP при наличии, иначе детерминированный геометрический расчёт по 3D-координатам.
    """
    try:
        pose_text = _read_pdb_text(pose_pdb)
        rec_text = _read_pdb_text(receptor_pdb)
    except FileNotFoundError:
        return []

    if not pose_text.strip() or not rec_text.strip():
        return []

    # 1. Если установлен PLIP — используем его профилирование
    if _PLIP_OK and _PLIPComplex is not None:
        try:
            merged_lines = [
                ln for ln in rec_text.splitlines() if ln.startswith("ATOM")
            ] + [
                ln if ln.startswith("HETATM") else "HETATM" + ln[6:]
                for ln in pose_text.splitlines()
                if ln.startswith(("ATOM", "HETATM"))
            ] + ["END"]
            with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, encoding="utf-8") as tmp:
                tmp.write("\n".join(merged_lines) + "\n")
                tmp_path = tmp.name
            try:
                mol_complex = _PLIPComplex()
                mol_complex.load_pdb(tmp_path)
                mol_complex.analyze()
                plip_contacts: list[str] = []
                for _, iset in mol_complex.interaction_sets.items():
                    for hb in list(getattr(iset, "hbonds_pdon", [])) + list(getattr(iset, "hbonds_ldon", [])):
                        plip_contacts.append(
                            f"Водородная связь: {hb.restype}{hb.resnr} ({hb.dist_d_a:.2f} Å) [PLIP]"
                        )
                    for hc in getattr(iset, "hydrophobic_contacts", []):
                        plip_contacts.append(
                            f"Гидрофобный контакт: {hc.restype}{hc.resnr} ({hc.distance:.2f} Å) [PLIP]"
                        )
                    for sb in getattr(iset, "saltbridge_lneg", []) + getattr(iset, "saltbridge_pneg", []):
                        plip_contacts.append(
                            f"Солевой мостик: {sb.restype}{sb.resnr} ({sb.distance:.2f} Å) [PLIP]"
                        )
                if plip_contacts:
                    return sorted(dict.fromkeys(plip_contacts))
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    # 2. Детерминированный 3D-геометрический анализ межатомных расстояний
    lig_atoms = [a for a in _parse_pdb_atoms(pose_text, records=("HETATM", "ATOM")) if a.element != "H"]
    rec_atoms = [
        a for a in _parse_pdb_atoms(rec_text, records=("ATOM",))
        if a.element != "H" and a.res_name.upper() not in _SOLVENT_AND_IONS
    ]
    if not lig_atoms or not rec_atoms:
        return []

    best_contacts: dict[tuple[str, str, int], tuple[float, str]] = {}

    for la in lig_atoms:
        for ra in rec_atoms:
            dx = la.x - ra.x
            if abs(dx) > 4.0:
                continue
            dy = la.y - ra.y
            if abs(dy) > 4.0:
                continue
            dz = la.z - ra.z
            if abs(dz) > 4.0:
                continue
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 1.5 or dist > 4.0:
                continue

            res_label = f"{ra.res_name.upper()}{ra.res_seq}"

            # Полярный контакт / водородная связь (N/O/S/F ... N/O/S, 1.8–3.5 Å)
            if la.element in {"N", "O", "F", "S"} and ra.element in {"N", "O", "S"} and dist <= 3.5:
                key = ("hbond", res_label, ra.res_seq)
                desc = f"Водородная связь: {res_label}:{ra.atom_name} ({dist:.2f} Å)"
                if key not in best_contacts or dist < best_contacts[key][0]:
                    best_contacts[key] = (dist, desc)

            # Ионный контакт / солевой мостик (заряженная боковая цепь 1.8–4.0 Å)
            elif (
                la.element in {"N", "O"}
                and ra.element in {"N", "O"}
                and ra.res_name.upper() in _CHARGED_RES
                and ra.atom_name not in {"N", "O"}
                and dist <= 4.0
            ):
                key = ("salt", res_label, ra.res_seq)
                desc = f"Ионный контакт: {res_label}:{ra.atom_name} ({dist:.2f} Å)"
                if key not in best_contacts or dist < best_contacts[key][0]:
                    best_contacts[key] = (dist, desc)

            # Гидрофобный контакт (C/Cl/Br/F/S ... C гидрофобного остатка, 2.0–4.0 Å)
            elif (
                la.element in {"C", "Cl", "Br", "I", "S"}
                and ra.element == "C"
                and ra.res_name.upper() in _HYDROPHOBIC_RES
                and ra.atom_name not in {"C", "CA"}
                and 2.0 <= dist <= 4.0
            ):
                key = ("hydrophobic", res_label, ra.res_seq)
                desc = f"Гидрофобный контакт: {res_label} ({dist:.2f} Å)"
                if key not in best_contacts or dist < best_contacts[key][0]:
                    best_contacts[key] = (dist, desc)

    return [desc for _, desc in sorted(best_contacts.values(), key=lambda item: item[0])]

