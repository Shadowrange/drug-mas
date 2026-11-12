"""Файловый кэш внешних вызовов (базы данных и LLM) и поддержка автономного режима `OFFLINE=1`.

Все обращения к ChEMBL, PubChem, PDB, UniProt и LLM обёрнуты декоратором `@cached`.
При `OFFLINE=1` данные считываются исключительно из локальной директории `cache/`,
что гарантирует детерминированную воспроизводимость без доступа к сети.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

# Контекстная переменная для потокобезопасной фиксации источника последнего вызова (кэш / сеть)
LAST_FROM_CACHE: contextvars.ContextVar[bool] = contextvars.ContextVar("last_from_cache", default=False)

CACHE_DIR = Path(os.getenv("CACHE_DIR", Path(__file__).resolve().parent.parent / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OFFLINE = os.getenv("OFFLINE", "0") == "1"


class OfflineMiss(RuntimeError):
    """Запрошенная запись отсутствует в локальном кэше при включённом режиме OFFLINE=1."""


def _key(namespace: str, args: tuple, kwargs: dict) -> str:
    raw = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str, ensure_ascii=False)
    return f"{namespace}__{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def cached(namespace: str) -> Callable:
    """Декоратор файлового кэширования JSON-сериализуемых результатов по пространству имён."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = CACHE_DIR / f"{_key(namespace, args, kwargs)}.json"

            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                LAST_FROM_CACHE.set(True)
                return payload["result"]

            if OFFLINE:
                raise OfflineMiss(
                    f"OFFLINE=1, но в кэше нет {namespace}({args}, {kwargs}). "
                    f"Прогрей кэш с OFFLINE=0 до защиты."
                )

            t0 = time.time()
            result = fn(*args, **kwargs)
            # Сериализация без default=str гарантирует сохранение строгих числовых типов
            # (например, исключает неявное превращение numpy-float в строку).
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
                ),
                encoding="utf-8",
            )
            LAST_FROM_CACHE.set(False)
            return result

        return wrapper

    return deco


def cache_stats() -> dict[str, int]:
    """Возвращает количество закэшированных записей в разбивке по пространствам имён."""
    stats: dict[str, int] = {}
    for f in CACHE_DIR.glob("*.json"):
        ns = f.stem.split("__")[0]
        stats[ns] = stats.get(ns, 0) + 1
    return dict(sorted(stats.items()))


if __name__ == "__main__":
    print(f"OFFLINE={OFFLINE}  CACHE_DIR={CACHE_DIR}")
    for k, v in cache_stats().items():
        print(f"  {k}: {v}")
