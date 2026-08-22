from types import SimpleNamespace

import torch

from engine.paged_kv_cache import PagedKVCache


def _make_cache(batch_size=3, sequence_length=5, hidden_size=2):
    keys = torch.arange(batch_size * 2 * sequence_length * hidden_size, dtype=torch.float32).reshape(
        batch_size, 2, sequence_length, hidden_size
    )
    values = keys + 1000
    layer = SimpleNamespace(keys=keys, values=values)
    return SimpleNamespace(layers=[layer])


def test_paged_kv_cache_splits_cache_into_pages():
    cache = _make_cache(sequence_length=5)
    paged_cache = PagedKVCache(page_size=2)

    paged_cache.update(cache)

    assert paged_cache.layers[0].page_count == 3
    assert paged_cache.layers[0].keys_pages[0].shape[2] == 2
    assert paged_cache.layers[0].keys_pages[-1].shape[2] == 1


def test_paged_kv_cache_select_batch_compacts_each_page():
    cache = _make_cache(batch_size=3, sequence_length=4)
    paged_cache = PagedKVCache(page_size=2)

    paged_cache.update(cache)
    paged_cache.select_batch([0, 2])

    assert paged_cache.layers[0].keys_pages[0].shape[0] == 2
    assert paged_cache.layers[0].values_pages[0].shape[0] == 2


