"""Генерация и локальная модификация молекулярных структур.

Поддерживает два алгоритмических бэкенда:
  - `crem`: мутации фрагментов по базе данных `CReM` (при наличии локальной базы фрагментов);
  - `selfies`: точечные мутации токенов `STONED-SELFIES` с детерминированным фильтром
    химической состоятельности `chemically_sane` (не требует внешней базы фрагментов).

Выбор и переключение бэкенда выполняются кодом с явной записью метода в поле `Molecule.source`.
"""

from __future__ import annotations

import random
from pathlib import Path

from rdkit import Chem

from src.schemas import Molecule
from src.tools import chem

try:
    import selfies as _sf
    _SELFIES_OK = True
except ImportError:  # pragma: no cover
    _sf = None
    _SELFIES_OK = False

try:
    from crem.crem import mutate_mol as _crem_mutate
    _CREM_OK = True
except ImportError:  # pragma: no cover
    _crem_mutate = None
    _CREM_OK = False


class GenerationUnavailable(RuntimeError):
    """Ни один алгоритмический бэкенд генерации структур не доступен в текущем окружении."""


def crem_db_available(db_name: str | None) -> bool:
    return bool(db_name) and Path(db_name).is_file()


# Элементы, встречающиеся в лекарственных молекулах. Всё остальное —
# артефакт мутаций SELFIES, а не кандидат.
DRUGLIKE_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}


def chemically_sane(
    smiles: str,
    parent_fp=None,
    min_similarity: float = 0.40,
    size_ratio: tuple[float, float] = (0.6, 1.5),
    parent_heavy: int | None = None,
) -> tuple[bool, str]:
    """Детерминированный фильтр химической состоятельности для сгенерированных молекул.

    Отбрасывает структуры с ненулевым формальным зарядом, неспаренными радикальными
    электронами, нелекарственными элементами, кумуленовыми фрагментами (`C=C=C`),
    аномальным числом тяжёлых атомов или близостью Танимото (ECFP4) ниже `min_similarity`.
    Возвращает кортеж `(passed, reason)`.
    """
    from rdkit import DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    mol = chem.parse(smiles)
    if mol is None:
        return False, "SMILES не разбирается"

    if Chem.GetFormalCharge(mol) != 0:
        return False, f"ненулевой формальный заряд ({Chem.GetFormalCharge(mol):+d})"

    if Descriptors.NumRadicalElectrons(mol) > 0:
        return False, "радикальные электроны"

    bad = {a.GetSymbol() for a in mol.GetAtoms()} - DRUGLIKE_ELEMENTS
    if bad:
        return False, f"нелекарственные элементы: {', '.join(sorted(bad))}"

    if parent_heavy:
        lo, hi = size_ratio
        n = mol.GetNumHeavyAtoms()
        if not (parent_heavy * lo <= n <= parent_heavy * hi):
            return False, f"размер вне диапазона: {n} тяжёлых атомов при исходных {parent_heavy}"

    # Кумулены (C=C=C) и прочие цепочки двойных связей подряд — характерный
    # артефакт мутаций SELFIES. Формально валидны, в лекарствах не встречаются.
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[*]=[*]=[*]")):
        return False, "кумулен (две двойные связи подряд)"

    if parent_fp is not None:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        sim = DataStructs.TanimotoSimilarity(parent_fp, gen.GetFingerprint(mol))
        if sim < min_similarity:
            return False, f"сходство с исходной молекулой {sim:.2f} < {min_similarity}"

    return True, ""


def _mutate_selfies(parent_smiles: str, n: int, seed: int) -> list[str]:
    """
    STONED-SELFIES: точечные мутации строки SELFIES. Любая строка SELFIES
    декодируется в валидную молекулу, поэтому база фрагментов не нужна.
    """
    if not _SELFIES_OK:
        return []

    rng = random.Random(seed)
    try:
        parent_sf = _sf.encoder(parent_smiles)
    except Exception:
        return []
    if not parent_sf:
        return []

    tokens = list(_sf.split_selfies(parent_sf))
    alphabet = sorted(_sf.get_semantic_robust_alphabet())
    if not tokens or not alphabet:
        return []

    out: list[str] = []
    # Запас попыток: часть мутаций даст дубликаты или исходную молекулу.
    for _ in range(n * 150):
        if len(out) >= n:
            break
        mutated = list(tokens)
        for _ in range(1):
            op = rng.choice(("replace", "insert", "delete"))
            pos = rng.randrange(len(mutated))
            if op == "replace":
                mutated[pos] = rng.choice(alphabet)
            elif op == "insert":
                mutated.insert(pos, rng.choice(alphabet))
            elif len(mutated) > 1:
                mutated.pop(pos)
        try:
            smi = _sf.decoder("".join(mutated))
        except Exception:
            continue
        if smi:
            out.append(smi)
    return out


def _mutate_crem(parent_smiles: str, n: int, db_name: str, seed: int) -> list[str]:
    mol = chem.parse(parent_smiles)
    if mol is None or not _CREM_OK:
        return []
    rng = random.Random(seed)
    try:
        gen = _crem_mutate(mol, db_name=db_name, max_size=8, return_mol=False)
        got = [s for _, s in zip(range(n * 5), gen)]
    except Exception as exc:
        raise GenerationUnavailable(
            f"CReM не смог отработать по базе {db_name}: {type(exc).__name__}: {exc}. "
            "Вероятная причина — база собрана под другой версией RDKit (риск R2). "
            "Переключайтесь на backend='selfies'."
        ) from exc
    rng.shuffle(got)
    return got


def mutate(
    parent_smiles: str,
    n: int = 50,
    backend: str = "crem",
    db_name: str | None = None,
    seed: int = 0xF00D,
    min_similarity: float = 0.40,
) -> list[Molecule]:
    """
    Порождает до n новых валидных уникальных структур вокруг исходной.

    Возвращает список Molecule с заполненным source. Исходная молекула
    и дубликаты по каноническому SMILES исключены — иначе «новым кандидатом»
    окажется то, с чего начали.

    Пустой список — нормальный исход, а не ошибка: он означает, что алгоритм
    не породил ни одной новой валидной структуры. Граф обязан это обработать
    (route_generated), а не показывать пустой экран.
    """
    parent_canon = chem.canonical(parent_smiles)
    if parent_canon is None:
        raise ValueError(f"Исходный SMILES не разбирается RDKit: {parent_smiles!r}")

    use = backend
    reason = ""
    if backend == "crem" and not (_CREM_OK and crem_db_available(db_name)):
        use = "selfies"
        reason = (
            "база фрагментов CReM недоступна" if _CREM_OK
            else "пакет crem не установлен"
        )

    if use == "crem":
        raw = _mutate_crem(parent_smiles, n, db_name or "", seed)
    elif use == "selfies":
        if not _SELFIES_OK:
            raise GenerationUnavailable(
                "Недоступны оба backend'а: crem без базы фрагментов и selfies не установлен. "
                "Генерация структур невозможна, выдумывать SMILES запрещено."
            )
        raw = _mutate_selfies(parent_smiles, n, seed)
    else:
        raise ValueError(f"Неизвестный backend: {backend!r}. Допустимо: 'crem' | 'selfies'.")

    src = f"сгенерировано {use} из {parent_canon}"
    if reason:
        src += f" (fallback: {reason})"

    from rdkit.Chem import rdFingerprintGenerator

    parent_mol = chem.parse(parent_canon)
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    parent_fp = _gen.GetFingerprint(parent_mol)
    parent_heavy = parent_mol.GetNumHeavyAtoms()

    seen: set[str] = {parent_canon}
    out: list[Molecule] = []
    rejected: dict[str, int] = {}
    for smi in raw:
        canon = chem.canonical(smi)
        if canon is None or canon in seen:
            continue
        seen.add(canon)

        ok, why = chemically_sane(
            canon,
            parent_fp=parent_fp,
            min_similarity=min_similarity,
            parent_heavy=parent_heavy,
        )
        if not ok:
            key = why.split(":")[0].split("(")[0].strip()
            rejected[key] = rejected.get(key, 0) + 1
            continue

        m = chem.describe(canon, source=src)
        if m is None:
            continue
        out.append(m)
        if len(out) >= n:
            break

    # Сохраняем статистику причин отсева структур за последний вызов mutate()
    mutate.last_rejected = dict(sorted(rejected.items(), key=lambda x: -x[1]))
    return out


mutate.last_rejected = {}


def exclude_known(candidates: list[Molecule], rejected_smiles: list[str]) -> list[Molecule]:
    """
    Убирает структуры, уже отвергнутые на прошлом круге, по каноническому виду.
    Без этого второй круг возврата повторит те же молекулы и verify() удвоит
    им фильтры (см. предупреждение в verifier.py).
    """
    banned = {c for c in (chem.canonical(s) for s in rejected_smiles) if c}
    return [m for m in candidates if chem.canonical(m.smiles) not in banned]
