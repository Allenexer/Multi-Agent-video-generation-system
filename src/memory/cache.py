"""
ResponseCache — TTL-based cache for LLM responses.

Reduces API costs during development by caching identical requests.
Cache is in-memory only (cleared on restart).
"""
import json
import hashlib
import time
from threading import Lock


class ResponseCache:
    """
    Simple TTL cache for LLM API responses.

    Usage:
      cache = ResponseCache(ttl_seconds=300)  # 5 min TTL

      key = cache.make_key(model, messages)
      cached = cache.get(key)
      if cached:
          return cached

      response = call_llm(...)
      cache.put(key, response)
      return response
    """

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 500):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: dict[str, tuple[float, dict]] = {}
        self._lock = Lock()

    @staticmethod
    def make_key(model: str, messages: list) -> str:
        """Generate a cache key from model + messages."""
        raw = json.dumps({"model": model, "messages": messages},
                         ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        """Retrieve cached response, or None if expired/missing."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.time() - ts > self._ttl:
                del self._store[key]
                return None
            return value

    def put(self, key: str, value: dict):
        """Store a response in cache."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._store) >= self._max:
                oldest = min(self._store.keys(),
                            key=lambda k: self._store[k][0])
                del self._store[oldest]
            self._store[key] = (time.time(), value)

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)
