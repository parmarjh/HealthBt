"""
Cache Layer
TTL-based in-memory cache for storing query -> response pairs.
Prevents redundant LLM + retrieval calls for repeated queries.
"""

import time
import hashlib
import json
from typing import Any, Optional


class QueryCache:
    """
    Simple TTL in-memory cache.
    Key: SHA-256 hash of the normalized query string.
    Value: Cached response dict with timestamp.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._store: dict[str, dict] = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_key(self, query: str) -> str:
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[Any]:
        key = self._make_key(query)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if time.time() - entry["ts"] > self.ttl:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return entry["value"]

    def set(self, query: str, value: Any) -> None:
        key = self._make_key(query)
        self._store[key] = {"value": value, "ts": time.time()}

    def invalidate(self, query: str) -> bool:
        key = self._make_key(query)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "total_entries": len(self._store),
            "hits":   self.hits,
            "misses": self.misses,
            "hit_rate": f"{(self.hits / total * 100):.1f}%" if total else "0%",
        }
