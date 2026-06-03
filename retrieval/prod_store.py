"""Redis-backed product (parent) store.

Key schema:  prod:{parent_asin}  -> JSON blob (full merged_meta record)
Use SET+JSON (vs HSET) because all fields are written/read atomically and
nested types (categories list, details dict, images list) are preserved.
"""
import json
import os

import redis


KEY_PREFIX = "prod:"
DEFAULT_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class RedisProdStore:
    def __init__(self, url: str = DEFAULT_URL):
        self.client = redis.Redis.from_url(url, decode_responses=True)

    @staticmethod
    def _key(parent_asin: str) -> str:
        return f"{KEY_PREFIX}{parent_asin}"

    def upsert(self, prod: dict) -> None:
        """Write one product. `prod` must contain 'parent_asin'."""
        self.client.set(self._key(prod["parent_asin"]), json.dumps(prod))

    def upsert_many(self, prods) -> int:
        """Bulk write via pipeline. Returns number written."""
        pipe = self.client.pipeline(transaction=False)
        n = 0
        for p in prods:
            pipe.set(self._key(p["parent_asin"]), json.dumps(p))
            n += 1
        pipe.execute()
        return n

    def get(self, parent_asin: str) -> dict | None:
        raw = self.client.get(self._key(parent_asin))
        return json.loads(raw) if raw else None

    def mget(self, parent_asins: list[str]) -> list[dict | None]:
        """Batch fetch in one RTT. Returns list aligned with input order;
        missing keys come back as None."""
        keys = [self._key(a) for a in parent_asins]
        raws = self.client.mget(keys)
        return [json.loads(r) if r else None for r in raws]

    def count(self) -> int:
        """Number of prod:* keys currently stored."""
        return sum(1 for _ in self.client.scan_iter(match=f"{KEY_PREFIX}*"))
