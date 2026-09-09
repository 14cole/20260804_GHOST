"""Opt-in CPU execution state, isolated per synchronous solve/thread.

No function replacement or process-wide environment changes are used. Tile
workers receive their kernel evaluators explicitly from the assembly caller.
"""
from collections import OrderedDict
from functools import wraps
import inspect
import pickle
from ghost_runtime import ScopedValue

EXPERIMENTAL_METHOD = "experimental_cpu"
BATCH_SIZE = 256
CACHE_BYTES = 64 * 1024 ** 2
TABLE_BYTES = 32 * 1024 ** 2
STREAMED_FORMULATIONS = frozenset(("te_robin", "robin", "single_dielectric", "multi_region"))
_STATE = ScopedValue("ghost_cpu_execution", default=None)


def current_state():
    state = _STATE.get()
    return state if state is not None and state.active else None


def requested_cpu():
    return _STATE.get() is not None


class CPUState:
    def __init__(self, abort_event=None, progress_callback=None):
        self.active = True
        self.reuse_operators = True
        self.abort_event = abort_event
        self.progress_callback = None
        self.batch_size = BATCH_SIZE
        self.cache = OrderedDict()
        self.tables = OrderedDict()
        self.table_bytes = 0
        self.table_events = []
        self.systems = []
        self.formulations = []
        self.cache_stats = dict(hits=0, stores=0, evictions=0, bytes=0,
                                peak_bytes=0, budget_bytes=CACHE_BYTES)

    def checkpoint(self, completed=None, total=None):
        if self.abort_event is not None and self.abort_event.is_set():
            raise InterruptedError("Solve canceled by user.")
        # Keep the canonical solve's step/percentage contract. Batch messages
        # report actual work without resetting its progress bar.
        if completed is not None and self.progress_callback is not None:
            try:
                self.progress_callback(completed, total)
            except Exception:
                pass
        if self.abort_event is not None and self.abort_event.is_set():
            raise InterruptedError("Solve canceled by user.")

    def select(self, resources):
        self.active = resources["formulation"] in STREAMED_FORMULATIONS
        # Robin channels request different operators (PEC S versus K') or
        # different element weights (IBC). These complete assembly requests
        # do not repeat, so retaining them only extends dense-array lifetimes.
        self.reuse_operators = resources["formulation"] not in ("te_robin", "robin")
        self.formulations.append(dict(formulation=resources["formulation"],
            streamed=self.active, reason="" if self.active else
            "This formulation uses the reference CPU implementation."))
        self.checkpoint()

    def report(self):
        return dict(version=1, precision="double", device="cpu",
                    batch_size=self.batch_size, cache=dict(self.cache_stats),
                    kernel_tables=list(self.table_events),
                    table_budget_bytes=TABLE_BYTES, systems=list(self.systems),
                    formulations=list(self.formulations))


def select_formulation(resources, progress_callback=None):
    state = _STATE.get()
    if state is not None:
        state.progress_callback = progress_callback
        state.select(resources)


def select_solver(reference):
    if current_state() is None:
        return reference
    import cpu_streaming
    return getattr(cpu_streaming, reference.__name__)


def experimental_monostatic(function):
    """Own one bounded cache across channels and certification mesh pairs."""
    signature = inspect.signature(function)

    @wraps(function)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        method = str(bound.arguments.get("solver_method", "direct")).strip().lower()
        if method != EXPERIMENTAL_METHOD:
            return function(*args, **kwargs)
        from refined_lu import requested_precision
        if requested_precision() != "double":
            raise ValueError("Experimental CPU requires double LU precision.")
        if _STATE.get() is not None:
            return function(*args, **kwargs)
        state = CPUState(bound.arguments.get("abort_event"), bound.arguments.get("progress_callback"))
        with _STATE.override(state):
            state.checkpoint()
            result = function(*args, **kwargs)
            state.checkpoint()
            metadata = result.setdefault("metadata", {})
            metadata["solver_method_requested"] = EXPERIMENTAL_METHOD
            metadata["solver_method"] = "dense_lu_experimental_cpu" if state.systems else "dense_lu"
            metadata["experimental_cpu"] = state.report()
            return result
    return call


def cached_operator(label):
    """Cache immutable operator references within one bounded solve scope."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def call(*args, **kwargs):
            state = current_state()
            if state is None or not state.reuse_operators:
                return function(*args, **kwargs)
            state.checkpoint()
            from cpu_kernels import mesh_key
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)
            mesh = params.pop("mesh")
            params["k0"] = complex(params["k0"])
            key = (label, mesh_key(mesh), pickle.dumps(params, protocol=4))
            stats = state.cache_stats
            if key in state.cache:
                value, size = state.cache.pop(key)
                state.cache[key] = (value, size)
                stats["hits"] += 1
                return value
            value = function(*args, **kwargs)
            arrays = [value] if label == "D" else [a for pair in value for a in pair]
            size = sum(a.nbytes for a in arrays if any(a.strides))
            if size <= CACHE_BYTES:
                for a in arrays:
                    a.flags.writeable = False
                while state.cache and stats["bytes"] + size > CACHE_BYTES:
                    _, (_, old) = state.cache.popitem(last=False)
                    stats["bytes"] -= old
                    stats["evictions"] += 1
                state.cache[key] = (value, size)
                stats["bytes"] += size
                stats["stores"] += 1
                stats["peak_bytes"] = max(stats["peak_bytes"], stats["bytes"])
            return value
        return call
    return decorate
