from engine.prefix_cache import PrefixCache


def test_prefix_cache_returns_longest_matching_prefix():
    cache = PrefixCache()

    cache.store([1], {"prefix": 1}, [0.1])
    cache.store([1, 2, 3], {"prefix": 123}, [0.2])

    hit = cache.find_longest_prefix([1, 2, 3, 4])

    assert hit is not None
    assert hit.token_ids == (1, 2, 3)
    assert hit.past_key_values == {"prefix": 123}
    assert hit.past_key_values is not cache.find_longest_prefix([1, 2, 3]).past_key_values


def test_prefix_cache_returns_shorter_prefix_when_longer_match_missing():
    cache = PrefixCache()

    cache.store([9, 8], {"prefix": 98}, [0.3])

    hit = cache.find_longest_prefix([9, 8, 7, 6])

    assert hit is not None
    assert hit.token_ids == (9, 8)


def test_prefix_cache_miss_returns_none():
    cache = PrefixCache()

    cache.store([4, 5], {"prefix": 45}, [0.4])

    assert cache.find_longest_prefix([1, 2, 3]) is None


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_prefix_cache_evicts_least_recently_used_entry():
    clock = FakeClock()
    cache = PrefixCache(max_entries=2, ttl_seconds=None, clock=clock)

    cache.store([1], {"prefix": 1}, [0.1])
    clock.advance(1)
    cache.store([2], {"prefix": 2}, [0.2])
    clock.advance(1)
    cache.store([3], {"prefix": 3}, [0.3])

    assert cache.find_longest_prefix([1]) is None
    assert cache.find_longest_prefix([2]) is not None
    assert cache.find_longest_prefix([3]) is not None


def test_prefix_cache_expires_entries_by_ttl():
    clock = FakeClock()
    cache = PrefixCache(max_entries=10, ttl_seconds=5.0, clock=clock)

    cache.store([7, 8], {"prefix": 78}, [0.7])
    clock.advance(4.9)

    assert cache.find_longest_prefix([7, 8, 9]) is not None

    clock.advance(0.2)

    assert cache.find_longest_prefix([7, 8, 9]) is None
