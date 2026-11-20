"""Bounded L1 + optional Redis L2 cache with per-entry expiration.

Cache failures are non-fatal: detection falls back to local inference. Cache keys
must include the model version for model-derived scores and must never cache a
verdict from one path under another path's namespace.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from threading import RLock
from typing import Any

try:
    import redis
except ImportError:  # Redis is optional for development and unit tests.
    redis = None


class TTLCache:
    def __init__(self, maxsize: int = 100_000, ttl: int = 300):
        self.maxsize = max(1, int(maxsize))
        self.ttl = max(1, int(ttl))
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str):
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if now >= expires_at:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: int | None = None):
        ttl_value = self.ttl if ttl is None else int(ttl)
        if ttl_value <= 0:
            return
        expires_at = time.monotonic() + ttl_value
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()

    def __len__(self):
        with self._lock:
            return len(self._data)


memory_cache = TTLCache(
    maxsize=int(os.getenv('CACHE_MAXSIZE', '100000')),
    ttl=int(os.getenv('CACHE_TTL_SECONDS', '300')),
)
redis_cache = None
_redis_last_error: str | None = None
if redis and os.getenv('REDIS_URL'):
    try:
        redis_cache = redis.Redis.from_url(
            os.environ['REDIS_URL'], decode_responses=True,
            socket_timeout=0.15, socket_connect_timeout=0.15,
            retry_on_timeout=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        _redis_last_error = type(exc).__name__
        redis_cache = None


def cache_key(prefix: str, value: str) -> str:
    return prefix + ':' + hashlib.sha256(value.encode('utf-8', 'strict')).hexdigest()


def get(key: str):
    value = memory_cache.get(key)
    if value is not None:
        return value
    if redis_cache:
        try:
            raw = redis_cache.get(key)
            if raw is not None:
                value = json.loads(raw)
                # Respect Redis' remaining TTL so L1 never outlives the L2 entry.
                try:
                    ttl = int(redis_cache.ttl(key))
                except Exception:
                    ttl = memory_cache.ttl
                memory_cache.set(key, value, ttl=max(1, ttl) if ttl >= 0 else memory_cache.ttl)
                return value
        except Exception as exc:
            global _redis_last_error
            _redis_last_error = type(exc).__name__
    return None


def set(key: str, value: Any, ttl: int | None = None):
    ttl_value = memory_cache.ttl if ttl is None else int(ttl)
    if ttl_value <= 0:
        return
    memory_cache.set(key, value, ttl=ttl_value)
    if redis_cache:
        try:
            redis_cache.setex(key, ttl_value, json.dumps(value, separators=(',', ':')))
        except Exception as exc:
            global _redis_last_error
            _redis_last_error = type(exc).__name__


def backend_status(check: bool = False) -> dict[str, Any]:
    """Expose cache mode without disclosing Redis host or credentials."""
    global _redis_last_error
    if redis_cache is None:
        return {'l1': 'memory', 'redis_configured': False, 'redis_status': 'not_configured', 'l1_entries': len(memory_cache)}
    status = 'unknown'
    if check:
        try:
            redis_cache.ping()
            status = 'connected'
            _redis_last_error = None
        except Exception as exc:
            status = 'unavailable'
            _redis_last_error = type(exc).__name__
    return {
        'l1': 'memory', 'redis_configured': True,
        'redis_status': status,
        'last_error_type': _redis_last_error,
        'l1_entries': len(memory_cache),
    }
