import unittest


class StrictSelectionCacheTest(unittest.TestCase):
    def test_cache_is_lru_bounded_and_ttl_expiring(self):
        from src.workflows.selection_cache import StrictTTLCache

        now = [0.0]
        cache = StrictTTLCache(
            max_entries=2,
            ttl_seconds=5,
            clock=lambda: now[0],
        )

        cache.set("a", {"value": 1})
        cache.set("b", {"value": 2})
        self.assertEqual(cache.get("a"), {"value": 1})

        cache.set("c", {"value": 3})
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), {"value": 1})

        now[0] = 6.0
        self.assertIsNone(cache.get("a"))

    def test_cached_values_are_copied_before_reuse(self):
        from src.workflows.selection_cache import StrictTTLCache

        cache = StrictTTLCache(max_entries=1, ttl_seconds=5)
        cache.set("key", {"items": ["original"]})

        cached = cache.get("key")
        cached["items"].append("changed")

        self.assertEqual(cache.get("key"), {"items": ["original"]})


if __name__ == "__main__":
    unittest.main()
