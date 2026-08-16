from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import time
from collections import OrderedDict
from threading import RLock
from typing import Any


@dataclass
class PrefixCacheEntry:
    token_ids: tuple[int, ...]
    past_key_values: Any
    last_logits: Any
    last_access: float


@dataclass
class PrefixCacheHit:
    token_ids: tuple[int, ...]
    past_key_values: Any
    last_logits: Any

    @property
    def prefix_length(self) -> int:
        return len(self.token_ids)


@dataclass
class _PrefixCacheNode:
    children: dict[int, "_PrefixCacheNode"] = field(default_factory=dict)
    entry_key: tuple[int, ...] | None = None


class PrefixCache:
    def __init__(
        self,
        max_entries: int = 128,
        ttl_seconds: float | None = 300.0,
        clock=time.monotonic,
    ):
        self._root = _PrefixCacheNode()
        self._entries: OrderedDict[tuple[int, ...], PrefixCacheEntry] = OrderedDict()
        self._lock = RLock()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def _now(self) -> float:
        return self._clock()

    def _is_expired(self, entry: PrefixCacheEntry, now: float) -> bool:
        if self._ttl_seconds is None:
            return False

        return (now - entry.last_access) >= self._ttl_seconds

    def _purge_expired_locked(self):
        if self._ttl_seconds is None or not self._entries:
            return

        now = self._now()

        expired_keys = []
        for token_ids, entry in self._entries.items():
            if self._is_expired(entry, now):
                expired_keys.append(token_ids)
            else:
                break

        for token_ids in expired_keys:
            self._remove_entry_locked(token_ids)

    def _remove_entry_locked(self, token_ids: tuple[int, ...]):
        self._entries.pop(token_ids, None)

        node = self._root
        path: list[tuple[_PrefixCacheNode, int]] = []

        for token_id in token_ids:
            next_node = node.children.get(token_id)
            if next_node is None:
                return

            path.append((node, token_id))
            node = next_node

        if node.entry_key != token_ids:
            return

        node.entry_key = None

        for parent, token_id in reversed(path):
            child = parent.children[token_id]
            if child.entry_key is not None or child.children:
                break
            del parent.children[token_id]

    def find_longest_prefix(self, token_ids: list[int]) -> PrefixCacheHit | None:
        with self._lock:
            self._purge_expired_locked()

            node = self._root
            best_entry_key = node.entry_key
            best_entry = self._entries.get(best_entry_key) if best_entry_key else None

            for token_id in token_ids:
                next_node = node.children.get(token_id)
                if next_node is None:
                    break

                node = next_node

                if node.entry_key is not None:
                    entry = self._entries.get(node.entry_key)
                    if entry is not None:
                        best_entry = entry
                        best_entry_key = node.entry_key

            if best_entry is None or best_entry_key is None:
                return None

            if self._is_expired(best_entry, self._now()):
                self._remove_entry_locked(best_entry_key)
                return None

            best_entry.last_access = self._now()
            self._entries.move_to_end(best_entry_key)

            return PrefixCacheHit(
                token_ids=best_entry.token_ids,
                past_key_values=deepcopy(best_entry.past_key_values),
                last_logits=deepcopy(best_entry.last_logits),
            )

    def store(self, token_ids: list[int], past_key_values: Any, last_logits: Any):
        if not token_ids:
            return

        token_tuple = tuple(token_ids)

        with self._lock:
            self._purge_expired_locked()

            now = self._now()
            node = self._root

            for token_id in token_tuple:
                node = node.children.setdefault(token_id, _PrefixCacheNode())

            existing = self._entries.get(token_tuple)
            if existing is not None:
                existing.past_key_values = deepcopy(past_key_values)
                existing.last_logits = deepcopy(last_logits)
                existing.last_access = now
                self._entries.move_to_end(token_tuple)
                node.entry_key = token_tuple
                return

            self._entries[token_tuple] = PrefixCacheEntry(
                token_ids=token_tuple,
                past_key_values=deepcopy(past_key_values),
                last_logits=deepcopy(last_logits),
                last_access=now,
            )
            node.entry_key = token_tuple

            while self._max_entries is not None and len(self._entries) > self._max_entries:
                oldest_token_ids, _ = self._entries.popitem(last=False)
                self._remove_entry_locked(oldest_token_ids)

    def clear(self):
        with self._lock:
            self._root = _PrefixCacheNode()
            self._entries.clear()


prefix_cache = PrefixCache()
