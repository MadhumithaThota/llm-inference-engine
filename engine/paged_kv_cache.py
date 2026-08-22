from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PagedLayerCache:
    keys_pages: list[torch.Tensor] = field(default_factory=list)
    values_pages: list[torch.Tensor] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.keys_pages)


class PagedKVCache:
    """
    A learning-friendly paged view over Hugging Face KV cache tensors.

    This does not replace the model's cache object. It stores the same cache
    data in small pages so we can reason about paging, compaction, and batch
    selection without rewriting the full model runtime.
    """

    def __init__(self, page_size: int = 16):
        if page_size <= 0:
            raise ValueError("page_size must be greater than 0")

        self.page_size = page_size
        self._source_cache: Any | None = None
        self.layers: list[PagedLayerCache] = []

    def is_empty(self) -> bool:
        return self._source_cache is None

    def clear(self):
        self._source_cache = None
        self.layers = []

    def _split_into_pages(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        if tensor.ndim < 3:
            return [tensor.detach().clone()]

        pages: list[torch.Tensor] = []
        sequence_length = tensor.shape[2]

        for start in range(0, sequence_length, self.page_size):
            end = start + self.page_size
            pages.append(tensor[:, :, start:end, :].detach().clone())

        return pages

    def update(self, past_key_values: Any):
        self._source_cache = deepcopy(past_key_values)
        self.layers = []

        if past_key_values is None:
            return

        for layer in past_key_values.layers:
            self.layers.append(
                PagedLayerCache(
                    keys_pages=self._split_into_pages(layer.keys),
                    values_pages=self._split_into_pages(layer.values),
                )
            )

    def materialize(self):
        if self._source_cache is None:
            return None

        cache = deepcopy(self._source_cache)

        for cache_layer, paged_layer in zip(cache.layers, self.layers):
            if paged_layer.keys_pages:
                cache_layer.keys = torch.cat(paged_layer.keys_pages, dim=2)
            if paged_layer.values_pages:
                cache_layer.values = torch.cat(paged_layer.values_pages, dim=2)

        return cache

    def select_batch(self, indices: list[int]):
        if self._source_cache is None or not indices:
            return

        index_tensor = torch.tensor(
            indices,
            device=self._source_cache.layers[0].keys.device,
            dtype=torch.long,
        )

        for source_layer, paged_layer in zip(self._source_cache.layers, self.layers):
            source_layer.keys = source_layer.keys.index_select(0, index_tensor)
            source_layer.values = source_layer.values.index_select(0, index_tensor)

            paged_layer.keys_pages = [
                page.index_select(0, index_tensor) for page in paged_layer.keys_pages
            ]
            paged_layer.values_pages = [
                page.index_select(0, index_tensor) for page in paged_layer.values_pages
            ]


