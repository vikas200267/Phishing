from __future__ import annotations
from threading import RLock
from .feedback import DB, init_db, verified_rows
from .features import normalize_url

_LOCK=RLock()
_VERIFIED:dict[str,int]={}
_LOADED=False


def reload_verified_feedback(limit: int=100000) -> int:
    global _VERIFIED, _LOADED
    init_db()
    latest={}
    # Rows are newest-first; first occurrence wins for exact normalized URL.
    for normalized,label in verified_rows(limit=limit):
        latest.setdefault(normalized,int(label))
    with _LOCK:
        _VERIFIED=latest
        _LOADED=True
    return len(latest)


def _ensure_loaded():
    global _LOADED
    with _LOCK:
        ready=_LOADED
    if not ready:
        reload_verified_feedback()


def verified_verdict(url: str) -> int | None:
    _ensure_loaded(); u=normalize_url(url)
    with _LOCK:
        value=_VERIFIED.get(u)
    return int(value) if value is not None else None


def verified_verdicts(urls: list[str]) -> dict[str,int]:
    _ensure_loaded(); normalized=[normalize_url(u) for u in urls]
    with _LOCK:
        return {u:int(_VERIFIED[u]) for u in set(normalized) if u in _VERIFIED}
