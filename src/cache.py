"""
Кэш всех внешних вызовов. ВЛАДЕЛЕЦ: Р-B.

Зачем: на защите может не быть сети. OFFLINE=1 заставляет систему брать
только из кэша и падать с понятной ошибкой, если чего-то нет.

ПРАВИЛО: ни один вызов к ChEMBL / PubChem / PDB / UniProt / LLM
не делается в обход этого декоратора. Прямой requests.get ломает демо.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

CACHE_DIR = Path(os.getenv("CACHE_DIR", Path(__file__).resolve().parent.parent / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OFFLINE = os.getenv("OFFLINE", "0") == "1"


class OfflineMiss(RuntimeError):
    """OFFLINE=1, а в кэше этого нет. Значит, кэш не прогрет."""


def _key(namespace: str, args: tuple, kwargs: dict) -> str:
    raw = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str, ensure_ascii=False)
    return f"{namespace}__{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def cached(namespace: str) -> Callable:
    """
    @cached("chembl_activities")
    def fetch(target_id: str) -> dict: ...

    Кэш вечный: на хакатоне данные не протухают, а повторный вызов стоит времени.
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = CACHE_DIR / f"{_key(namespace, args, kwargs)}.json"

            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                wrapper.last_from_cache = True
                return payload["result"]

            if OFFLINE:
                raise OfflineMiss(
                    f"OFFLINE=1, но в кэше нет {namespace}({args}, {kwargs}). "
                    f"Прогрей кэш с OFFLINE=0 до защиты."
                )

            t0 = time.time()
            result = fn(*args, **kwargs)
            path.write_text(
                json.dumps(
                    {
                        "namespace": namespace,
                        "args": list(args),
                        "kwargs": kwargs,
                        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_s": round(time.time() - t0, 2),
                        "result": result,
                    },
                    ensure_ascii=False,
                    indent=1,
                    default=str,
                ),
                encoding="utf-8",
            )
            wrapper.last_from_cache = False
            return result

        wrapper.last_from_cache = False
        return wrapper

    return deco


def cache_stats() -> dict[str, int]:
    """Для проверки перед защитой: что вообще лежит в кэше."""
    stats: dict[str, int] = {}
    for f in CACHE_DIR.glob("*.json"):
        ns = f.stem.split("__")[0]
        stats[ns] = stats.get(ns, 0) + 1
    return dict(sorted(stats.items()))


if __name__ == "__main__":
    print(f"OFFLINE={OFFLINE}  CACHE_DIR={CACHE_DIR}")
    for k, v in cache_stats().items():
        print(f"  {k}: {v}")
