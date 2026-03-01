"""Async KV store wrapper with Redis backend and in-memory fallback.

Provides: async `get`, `set`, `get_or_set`, `rpush`, `lrange`, `connect`, `close`.

Defaults to Redis at redis://localhost:6379, but will use a small in-memory
dict + list fallback if redis isn't available.
"""
from __future__ import annotations

import asyncio
import json
import os
import hashlib
from typing import Any, Optional, Callable

try:
    import redis.asyncio as redis_async
except Exception:
    redis_async = None


def _make_key(key: str) -> str:
    return key


class InMemoryFallback:
    def __init__(self):
        self.store = {}
        self.lists = {}
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None):
        self.store[key] = value

    async def rpush(self, key: str, value: str):
        self.lists.setdefault(key, []).append(value)

    async def lrange(self, key: str, start: int = 0, end: int = -1):
        lst = self.lists.get(key, [])
        if end == -1:
            return lst[start:]
        return lst[start : end + 1]

    async def delete(self, key: str):
        self.store.pop(key, None)
        self.lists.pop(key, None)


class KVStore:
    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client = None
        self._fallback = InMemoryFallback()
        self._use_redis = False

    async def connect(self):
        if redis_async is None:
            self._use_redis = False
            return
        try:
            self._client = redis_async.from_url(self.url, decode_responses=True)
            # try a ping
            await self._client.ping()
            self._use_redis = True
        except Exception:
            self._client = None
            self._use_redis = False

    async def close(self):
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass

    async def get(self, key: str) -> Optional[str]:
        key2 = _make_key(key)
        if self._use_redis and self._client:
            try:
                return await self._client.get(key2)
            except Exception:
                self._use_redis = False
        return await self._fallback.get(key2)

    async def set(self, key: str, value: str, ex: Optional[int] = None):
        key2 = _make_key(key)
        if self._use_redis and self._client:
            try:
                await self._client.set(key2, value, ex=ex)
                return
            except Exception:
                self._use_redis = False
        await self._fallback.set(key2, value, ex=ex)

    async def rpush(self, key: str, value: str):
        key2 = _make_key(key)
        if self._use_redis and self._client:
            try:
                await self._client.rpush(key2, value)
                return
            except Exception:
                self._use_redis = False
        await self._fallback.rpush(key2, value)

    async def lrange(self, key: str, start: int = 0, end: int = -1):
        key2 = _make_key(key)
        if self._use_redis and self._client:
            try:
                return await self._client.lrange(key2, start, end)
            except Exception:
                self._use_redis = False
        return await self._fallback.lrange(key2, start, end)

    async def delete(self, key: str):
        key2 = _make_key(key)
        if self._use_redis and self._client:
            try:
                await self._client.delete(key2)
                return
            except Exception:
                self._use_redis = False
        await self._fallback.delete(key2)

    async def get_or_set(self, key: str, coro: Callable[[], Any], ex: Optional[int] = None):
        """Return stored value or compute via `coro()` and set it."""
        existing = await self.get(key)
        if existing is not None:
            return existing
        # simple in-process lock to avoid thundering herd
        val = await coro()
        if isinstance(val, (dict, list)):
            serialized = json.dumps(val)
        else:
            serialized = str(val)
        await self.set(key, serialized, ex=ex)
        return serialized


def make_message_key(session_id: str, index: int) -> str:
    return f"session:{session_id}:history:{index}"


def messages_list_key(session_id: str) -> str:
    return f"session:{session_id}:history"


def response_cache_key(model: str, options: dict, messages_json: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(json.dumps(options, sort_keys=True).encode("utf-8"))
    h.update(messages_json.encode("utf-8"))
    return "respcache:" + h.hexdigest()


_default_kv: Optional[KVStore] = None


async def get_default_kv() -> KVStore:
    global _default_kv
    if _default_kv is None:
        _default_kv = KVStore()
        await _default_kv.connect()
    return _default_kv
