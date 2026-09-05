"""Solve-local wall timings and sampled process memory, without Qt dependencies."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import threading
import time


_ACTIVE = ContextVar("ghost_solver_metrics", default=None)


class SolveMetrics:
    def __init__(self):
        self.seconds = {}
        self.calls = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sampler = None
        self._process = None
        self.baseline_rss = self.peak_rss = None
        try:
            import psutil
            self._process = psutil.Process()
        except (ImportError, OSError):
            pass

    def sample_memory(self):
        if self._process is not None:
            try:
                value = int(self._process.memory_info().rss)
                with self._lock:
                    if self.baseline_rss is None:
                        self.baseline_rss = value
                    self.peak_rss = max(self.peak_rss or 0, value)
            except Exception:
                # Memory instrumentation must never change a numerical result.
                pass

    def start(self):
        self.started = time.perf_counter()
        self.sample_memory()
        if self._process is not None:
            def poll():
                while not self._stop.wait(0.05):
                    self.sample_memory()
            self._sampler = threading.Thread(target=poll, daemon=True,
                                             name="GHOST memory sampler")
            self._sampler.start()

    def finish(self):
        self.sample_memory()
        self.elapsed = time.perf_counter() - self.started
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
                self.calls[name] = self.calls.get(name, 0) + 1

    def wrap(self, name, function):
        if function is None:
            return None
        @wraps(function)
        def measured(*args, **kwargs):
            with self.stage(name):
                return function(*args, **kwargs)
        return measured

    def report(self):
        return {
            "wall_seconds": self.elapsed,
            "stage_seconds": dict(self.seconds),
            "stage_calls": dict(self.calls),
            "stage_semantics": "inclusive elapsed; parallel/nested stages may overlap",
            "sampled_peak_process_rss_bytes": self.peak_rss,
            "initial_process_rss_bytes": self.baseline_rss,
            "memory_semantics": "50 ms process RSS samples, including other jobs; not an exclusive allocation peak",
        }


def active_metrics():
    return _ACTIVE.get()


def timed_stage(name):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            metrics = active_metrics()
            if metrics is None:
                return function(*args, **kwargs)
            with metrics.stage(name):
                return function(*args, **kwargs)
        return measured
    return decorate


def profiled_solve(function):
    @wraps(function)
    def measured(*args, **kwargs):
        # Nested polarization/certification/frequency calls share one record.
        if active_metrics() is not None:
            return function(*args, **kwargs)
        metrics = SolveMetrics()
        token = _ACTIVE.set(metrics)
        metrics.start()
        try:
            result = function(*args, **kwargs)
        finally:
            metrics.finish()
            _ACTIVE.reset(token)
        if isinstance(result, dict):
            container = result.get("metadata", result)
            container["runtime_profile"] = metrics.report()
        return result
    return measured
