"""Solve-local wall timings and sampled process memory, without Qt dependencies."""

from contextlib import contextmanager
from ghost_backend.execution.runtime import ScopedValue
from functools import wraps
import threading
import time


_ACTIVE = ScopedValue("ghost_solver_metrics", default=None)
_LISTENER = ScopedValue("ghost_progress_listener", default=None)


@contextmanager
def progress_listener(callback):
    """Receive stage, elapsed time, and sampled process memory updates."""
    with _LISTENER.override(callback):
        yield


def _stage_label(name):
    if 'assembly' in name or name.startswith('assemble'):
        return 'Assembly'
    if 'factor' in name:
        return 'Factorization'
    if 'solve' in name or 'rhs' in name:
        return 'Angle solving'
    if 'condition' in name or 'residual' in name:
        return 'Quality checks'
    return name.replace('_', ' ').capitalize()


class SolveMetrics:
    def __init__(self):
        self.seconds = {}
        self.calls = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sampler = None
        self._process = None
        self._listener = _LISTENER.get()
        self._stages = []
        self.phase = ''
        self._last_event = 0.0
        self.baseline_rss = self.peak_rss = None
        self.current_rss = None
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
                    self.current_rss = value
                    if self.baseline_rss is None:
                        self.baseline_rss = value
                    self.peak_rss = max(self.peak_rss or 0, value)
            except Exception:

                pass

    def start(self):
        self.started = time.perf_counter()
        self.sample_memory()
        if self._process is not None or self._listener is not None:
            def poll():
                while not self._stop.wait(0.05):
                    self.sample_memory()
                    self.publish()
            self._sampler = threading.Thread(target=poll, daemon=True,
                                             name="GHOST memory sampler")
            self._sampler.start()
        self.publish(force=True)

    def publish(self, force=False):
        if self._listener is None:
            return
        now = time.perf_counter()
        with self._lock:
            if not force and now - self._last_event < 0.5:
                return
            self._last_event = now
            name = self._stages[-1][1] if self._stages else 'preparing_geometry'
            event = dict(stage=_stage_label(name), stage_key=name, phase=self.phase,
                         elapsed_seconds=max(0.0, now-self.started),
                         process_rss_bytes=self.current_rss,
                         peak_process_rss_bytes=self.peak_rss)
        try:
            self._listener(event)
        except Exception:
            pass

    def finish(self):
        self.sample_memory()
        self.elapsed = time.perf_counter() - self.started
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        token = object()
        with self._lock:
            self._stages.append((token, name))
        self.publish()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
                self.calls[name] = self.calls.get(name, 0) + 1
                self._stages = [entry for entry in self._stages if entry[0] is not token]

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


def metrics_scope(metrics):
    """Carry an existing solve's thread-safe metrics into a worker thread."""
    return _ACTIVE.override(metrics)


@contextmanager
def solve_phase(name):
    """Label base, refined, or certification work in live metrics."""
    metrics = active_metrics()
    previous = metrics.phase if metrics is not None else ''
    if metrics is not None:
        metrics.phase = name
        metrics.publish(force=True)
    try:
        yield
    finally:
        if metrics is not None:
            metrics.phase = previous


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

        if active_metrics() is not None:
            return function(*args, **kwargs)
        metrics = SolveMetrics()
        with _ACTIVE.override(metrics):
            metrics.start()
            try:
                result = function(*args, **kwargs)
            finally:
                metrics.finish()
        if isinstance(result, dict):
            container = result.get("metadata", result)
            container["runtime_profile"] = metrics.report()
        return result
    return measured
