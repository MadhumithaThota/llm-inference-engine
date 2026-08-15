from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from threading import RLock
from typing import Any


@dataclass
class PrefixCacheEntry:
    token_ids: tuple[int, ...]
    past_key_values: Any
    last_logits: Any


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
    entry: PrefixCacheEntry | None = None


class PrefixCache:
    def __init__(self):
        self._root = _PrefixCacheNode()
        self._lock = RLock()

    def find_longest_prefix(self, token_ids: list[int]) -> PrefixCacheHit | None:
        with self._lock:
            node = self._root
            best_entry = node.entry

            for token_id in token_ids:
                next_node = node.children.get(token_id)
                if next_node is None:
                    break

                node = next_node

                if node.entry is not None:
                    best_entry = node.entry

            if best_entry is None:
                return None

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
            node = self._root

            for token_id in token_tuple:
                node = node.children.setdefault(token_id, _PrefixCacheNode())

            node.entry = PrefixCacheEntry(
                token_ids=token_tuple,
                past_key_values=deepcopy(past_key_values),
                last_logits=deepcopy(last_logits),
            )

    def clear(self):
        with self._lock:
            self._root = _PrefixCacheNode()


prefix_cache = PrefixCache()
