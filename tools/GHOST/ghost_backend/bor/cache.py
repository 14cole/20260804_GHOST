"""Solve-local, thread-safe reuse of small contracted modal tiles."""
from collections import OrderedDict
import threading
from ghost_backend.execution.runtime import ScopedValue

_ACTIVE = ScopedValue('ghost_bor_tile_cache', default=None)


class TileCache:
    def __init__(self, budget_bytes):
        self.budget = int(budget_bytes)
        self.values = OrderedDict()
        self.lock = threading.Lock()
        self.bytes = self.peak = self.hits = self.misses = self.evictions = 0

    def get(self, key):
        with self.lock:
            value = self.values.get(key)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
                self.values.move_to_end(key)
            return value

    def put(self, key, value):
        if value.nbytes > self.budget:
            return
        with self.lock:
            old = self.values.pop(key, None)
            if old is not None:
                self.bytes -= old.nbytes
            # Bound Python/key overhead as well as numerical payload.
            while self.values and (self.bytes + value.nbytes > self.budget or len(self.values) >= 4096):
                self.bytes -= self.values.popitem(last=False)[1].nbytes
                self.evictions += 1
            self.values[key] = value
            self.bytes += value.nbytes
            self.peak = max(self.peak, self.bytes)

    def evidence(self):
        with self.lock:
            return dict(budget_bytes=self.budget, peak_payload_bytes=self.peak,
                        hits=self.hits, misses=self.misses, evictions=self.evictions,
                        entries=len(self.values))


def current_cache():
    return _ACTIVE.get()


def cache_scope(cache):
    return _ACTIVE.override(cache)
