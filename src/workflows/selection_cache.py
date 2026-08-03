"""Bounded process-local cache for validated technology-selection results."""

import json
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
from threading import RLock
from time import monotonic

from src.config.settings import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SELECTION_STRICT_CACHE_ENABLED,
    SELECTION_STRICT_CACHE_MAX_ENTRIES,
    SELECTION_STRICT_CACHE_TTL_SECONDS,
)


class StrictTTLCache:
    """Thread-safe LRU cache with TTL and copy-on-read/write isolation."""

    def __init__(self, max_entries, ttl_seconds, clock=monotonic):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries = OrderedDict()
        self._lock = RLock()

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return deepcopy(value)

    def set(self, key, value, ttl_seconds=None):
        with self._lock:
            effective_ttl = (
                self._ttl_seconds if ttl_seconds is None else ttl_seconds
            )
            if effective_ttl <= 0:
                raise ValueError("ttl_seconds must be positive")
            self._entries[key] = (
                self._clock() + effective_ttl,
                deepcopy(value),
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self, namespace=None):
        with self._lock:
            if namespace is None:
                self._entries.clear()
                return
            for key in list(self._entries):
                if key[0] == namespace:
                    del self._entries[key]


_SELECTION_CACHE = StrictTTLCache(
    max_entries=SELECTION_STRICT_CACHE_MAX_ENTRIES,
    ttl_seconds=SELECTION_STRICT_CACHE_TTL_SECONDS,
)


def _selection_cache_key(namespace, chain, prompt, inputs):
    payload = {
        "namespace": namespace,
        "chain_identity": id(chain),
        "prompt": str(prompt),
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "base_url": LLM_BASE_URL,
        "inputs": inputs,
    }
    digest = sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return namespace, digest


def get_selection_cache(namespace, chain, prompt, inputs):
    """Return a copied result only for an exact runtime/input fingerprint."""
    if not SELECTION_STRICT_CACHE_ENABLED:
        return None
    key = _selection_cache_key(namespace, chain, prompt, inputs)
    return _SELECTION_CACHE.get(key)


def set_selection_cache(
    namespace,
    chain,
    prompt,
    inputs,
    value,
    ttl_seconds=None,
):
    """Store a validated result under an exact runtime/input fingerprint."""
    if not SELECTION_STRICT_CACHE_ENABLED:
        return
    key = _selection_cache_key(namespace, chain, prompt, inputs)
    _SELECTION_CACHE.set(key, value, ttl_seconds=ttl_seconds)


def clear_selection_cache(namespace=None):
    """Clear all cached results or one node namespace."""
    _SELECTION_CACHE.clear(namespace)
