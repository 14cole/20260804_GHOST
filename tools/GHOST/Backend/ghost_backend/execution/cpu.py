"""Opt-in CPU execution state, isolated per synchronous solve/thread."""
from ghost_backend.execution.options import environment_value
from collections import OrderedDict
from functools import wraps
import inspect
import pickle
import os
from ghost_backend.execution.runtime import ScopedValue

EXPERIMENTAL_METHOD = "experimental_cpu"
BATCH_SIZE = 256
CACHE_BYTES = 64 * 1024 ** 2
TABLE_BYTES = 32 * 1024 ** 2
STREAMED_FORMULATIONS = frozenset(("te_robin", "robin", "single_dielectric", "multi_region",
                                  "sheet", "mixed_sheet_pec", "thin_dielectric_layer"))
_STATE = ScopedValue("ghost_cpu_execution", default=None)


def configured_batch_size():
    """Same explicit angle-batch limit for execution and resource planning."""
    value = str(environment_value('GHOST_CPU_ANGLE_BATCH_SIZE', str(BATCH_SIZE))).strip()
    try:
        size = int(value)
    except ValueError:
        raise ValueError('GHOST_CPU_ANGLE_BATCH_SIZE must be a positive integer.')
    if size < 1 or size > BATCH_SIZE:
        raise ValueError('GHOST_CPU_ANGLE_BATCH_SIZE must be between 1 and {}.'.format(BATCH_SIZE))
    return size


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
        self.batch_size = configured_batch_size()
        self.cache = OrderedDict()
        self.tables = OrderedDict()
        self.table_bytes = 0
        self.table_events = []
        self.systems = []
        self.formulations = []
        self.memory_estimates = []
        self.cache_stats = dict(hits=0, stores=0, evictions=0, bytes=0,
                                peak_bytes=0, budget_bytes=CACHE_BYTES)

    def checkpoint(self, completed=None, total=None):
        if self.abort_event is not None and self.abort_event.is_set():
            raise InterruptedError("Solve canceled by user.")


        if completed is not None and self.progress_callback is not None:
            try:
                self.progress_callback(completed, total)
            except Exception:
                pass
        if self.abort_event is not None and self.abort_event.is_set():
            raise InterruptedError("Solve canceled by user.")

    def select(self, resources):
        self.active = resources["formulation"] in STREAMED_FORMULATIONS


        self.reuse_operators = False
        self.formulations.append(dict(formulation=resources["formulation"],
            streamed=self.active, reason="" if self.active else
            "This formulation uses the reference CPU implementation."))
        self.checkpoint()

    def report(self):
        return dict(version=1, precision="double", device="cpu",
                    batch_size=self.batch_size, cache=dict(self.cache_stats),
                    kernel_tables=list(self.table_events),
                    table_budget_bytes=TABLE_BYTES, systems=list(self.systems),
                    formulations=list(self.formulations), memory_estimates=list(self.memory_estimates))


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
        from ghost_backend.linalg.refined_lu import requested_precision
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
            from ghost_backend.compressed.runtime import enabled
            if enabled() and state.systems:metadata['solver_method']='compressed_experimental_cpu'
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
            from ghost_backend.twod.assembly.session import current_session
            if (state is None or not state.reuse_operators or current_session() is not None
                    or kwargs.get('operator_outputs') is not None or kwargs.get('destination') is not None
                    or kwargs.get('prepared_geometry') is not None):
                return function(*args, **kwargs)
            state.checkpoint()
            from ghost_backend.twod.assembly.kernels import mesh_key
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)


            if (params.get('destination') is not None or params.get('operator_outputs') is not None
                    or params.get('prepared_geometry') is not None):
                return function(*args, **kwargs)
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
            size = sum(a.nbytes for a in arrays if any(a.strides) or hasattr(a, 'row_map'))
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
