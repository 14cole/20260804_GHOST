"""Validated, serializable execution settings scoped to one solve."""
from contextlib import contextmanager
from functools import wraps
import inspect
import math
import ntpath
import os
from pathlib import Path
import tempfile
import threading
import time

from ghost_backend.execution.runtime import ScopedValue

DEFAULTS = {
    'version': 1,
    'factorization': 'dense',
    'discretization': 'galerkin',
    'pulse_pec_cfie': True,
    'mesh_strategy': 'global',
    'basis_order': 1,
    'compressed_storage_mib': 2048,
    'ram_budget_gib': None,
    'temporary_directory': '',
    'assembly_threads': 1,
    'blas_threads': 1,
    'rhs_compression': 'auto',
    'angle_batch_size': 256,
    'assembly_tile': 0,
    'far_quadrature_order': 0,
    'far_grading': True,
    'fmm_tolerance': 1e-10,
    'fmm_solver_tolerance': 1e-9,
    'fmm_restart': 80,
    'fmm_max_iterations': 800,
    'fmm_recycle_vectors': 'auto',
    'fmm_pec_cfie': 'auto',
    'fmm_quadrature_order': 0,
}
EFFICIENT_DEFAULTS = dict(DEFAULTS, factorization='adaptive', mesh_strategy='adaptive', compressed_storage_mib=2048,
                          assembly_threads=4, blas_threads=2)
_ACTIVE = ScopedValue('ghost_execution_options', default=None)
_ASSEMBLY_ALLOCATION = ScopedValue('ghost_assembly_allocation', default=None)
_MEMORY_ALLOCATION = ScopedValue('ghost_memory_allocation', default=None)
_BLAS_LOCK = threading.RLock()
_PUBLIC_DEPTH = ScopedValue('ghost_public_execution_depth',default=0)
_AUTOMATIC_REQUEST = ScopedValue('ghost_automatic_backend_request', default=False)


def automatic_backend_requested():
    return _AUTOMATIC_REQUEST.get()

_ENV_FIELDS = {
    'GHOST_CPU_FACTORIZATION': 'factorization',
    'GHOST_COMPRESSED_STORAGE_MIB': 'compressed_storage_mib',
    'GHOST_CPU_RHS_COMPRESSION': 'rhs_compression',
    'GHOST_CPU_ANGLE_BATCH_SIZE': 'angle_batch_size',
    'GHOST_ASSEMBLY_THREADS': 'assembly_threads',
    'GHOST_ASSEMBLY_TILE': 'assembly_tile',
    'GHOST_FAR_QUAD_ORDER': 'far_quadrature_order',
    'GHOST_MAX_SOLVE_GB': 'ram_budget_gib',
}


def validate_options(value):
    """Return a complete independent JSON record; reject unsupported values."""
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('Execution settings must contain supported fields only.')
    result = dict(DEFAULTS)
    result.update(value)
    if type(result['version']) is not int or result['version'] != 1:
        raise ValueError('Unsupported execution settings version.')
    if result['factorization'] not in ('dense', 'hierarchical', 'auto', 'compressed', 'adaptive', 'fmm'):
        raise ValueError('Choose dense, hierarchical, auto, compressed, adaptive, or fmm factorization.')
    if result['discretization'] not in ('galerkin','pulse'):
        raise ValueError('Choose galerkin or pulse discretization.')
    if result['discretization']=='pulse':
        if result['basis_order']!=1:
            raise ValueError('Pulse uses constant panel densities; polynomial enrichment requires Galerkin.')
        if result['mesh_strategy']=='adaptive':result['mesh_strategy']='global'
    if type(result['pulse_pec_cfie']) is not bool:
        raise ValueError('Pulse PEC combined field must be true or false.')
    for key in ('fmm_tolerance','fmm_solver_tolerance'):
        number=result[key]
        if type(number) not in (int,float) or not math.isfinite(number) or not 1e-13<=number<=1e-5:
            raise ValueError('{} must be finite and between 1e-13 and 1e-5.'.format(key))
    if result['fmm_tolerance']>result['fmm_solver_tolerance']:
        raise ValueError('FMM kernel tolerance must not exceed solver tolerance.')
    if type(result['basis_order']) is not int or result['basis_order'] not in (1, 2, 3):
        raise ValueError('Boundary polynomial degree must be 1, 2 or 3.')
    if result['mesh_strategy'] not in ('global', 'local', 'adaptive'):
        raise ValueError('Mesh strategy must be global, local or adaptive.')
    if result['rhs_compression'] not in ('off', 'auto', 'on'):
        raise ValueError('RHS compression must be off, auto, or on.')
    for key, lower, upper in [('compressed_storage_mib', 16, 1048576),
                              ('blas_threads', 1, 1024), ('angle_batch_size', 1, 256),
                              ('assembly_tile', 0, 65536), ('far_quadrature_order', 0, 64),
                              ('fmm_restart',10,512),('fmm_max_iterations',1,100000)]:
        number = result[key]
        if type(number) is not int or not lower <= number <= upper:
            raise ValueError('{} must be an integer from {} to {}.'.format(key, lower, upper))
    recycle = result['fmm_recycle_vectors']
    if recycle != 'auto' and (type(recycle) is not int or not 0 <= recycle <= 64):
        raise ValueError('FMM recycle vectors must be auto or an integer from 0 to 64.')
    quadrature = result['fmm_quadrature_order']
    if type(quadrature) is not int or quadrature != 0 and not 6 <= quadrature <= 64:
        raise ValueError('FMM quadrature order must be 0 (automatic) or an integer from 6 to 64.')
    threads = result['assembly_threads']
    if threads != 'auto' and (type(threads) is not int or not 1 <= threads <= 1024):
        raise ValueError('Assembly threads must be auto or an integer from 1 to 1024.')
    ram = result['ram_budget_gib']
    if ram is not None and (type(ram) not in (int, float) or not math.isfinite(ram) or ram <= 0):
        raise ValueError('RAM budget must be positive GiB or null for available memory.')
    if type(result['far_grading']) is not bool:
        raise ValueError('Far grading must be true or false.')
    if type(result['fmm_pec_cfie']) is not bool and result['fmm_pec_cfie'] != 'auto':
        raise ValueError('FMM PEC combined-field selection must be auto, true or false.')
    directory = result['temporary_directory']
    if not isinstance(directory, str) or any(c in directory for c in '\r\n\x00'):
        raise ValueError('Temporary directory must be a path string.')
    if directory and not (os.path.isabs(directory) or ntpath.isabs(directory)):
        raise ValueError('Temporary directory must be absolute or empty for the system temporary directory.')
    return result


def efficient_defaults():
    """Return the large-sweep preset, bounded by the host's CPU count."""
    values = dict(EFFICIENT_DEFAULTS)
    cores = max(1, os.cpu_count() or 1)
    for key in ('assembly_threads', 'blas_threads'):
        values[key] = min(values[key], cores)
    return values


def geometry_preset(name):
    """Return 2D monostatic performance settings without changing mesh accuracy."""
    values = efficient_defaults()
    if name != 'adaptive': values['mesh_strategy'] = 'global'
    if name == 'small':
        values.update(factorization='dense', rhs_compression='off')
        method = 'direct'
    elif name == 'balanced':
        values.update(factorization='dense')
        method = 'experimental_cpu'
    elif name == 'adaptive':
        values.update(factorization='adaptive')
        method = 'auto'
    elif name == 'large':
        values.update(factorization='compressed',compressed_storage_mib=8192)
        method = 'experimental_cpu'
    else:
        raise ValueError('Choose small, balanced, large, or adaptive geometry settings.')
    return dict(solver_method=method, lu_precision='double', execution_options=values)


def from_environment(defaults=None):
    """Capture launch defaults once, for callers without an explicit profile."""
    values = validate_options(defaults if defaults is not None else {})
    for name, key in _ENV_FIELDS.items():
        raw = os.environ.get(name, '').strip()
        if not raw:
            continue
        if key == 'ram_budget_gib':
            values[key] = float(raw)
        elif key in ('factorization', 'rhs_compression') or (key == 'assembly_threads' and raw == 'auto'):
            values[key] = raw.lower()
        else:
            values[key] = int(raw)
    values['blas_threads'] = int(os.environ.get('OPENBLAS_NUM_THREADS', '') or
                                 os.environ.get('MKL_NUM_THREADS', '') or values['blas_threads'])
    values['far_grading'] = os.environ.get('GHOST_FAR_GRADED', '1') != '0'
    return validate_options(values)


def current_options():
    """Return a copy of the active settings, or None outside a configured run."""
    value = _ACTIVE.get()
    return dict(value) if value is not None else None


def option(name, fallback=None):
    active = _ACTIVE.get()
    return active[name] if active is not None else fallback


def effective_assembly_threads(fallback=1):
    """Resolve requested concurrency against the scheduler's CPU allocation."""
    active = _ACTIVE.get()
    if active is None:
        return fallback
    requested = active['assembly_threads']
    allocation = _ASSEMBLY_ALLOCATION.get()
    if requested == 'auto':
        return allocation or 1
    return min(requested, allocation) if allocation is not None else requested


def allocated_memory_budget():
    return _MEMORY_ALLOCATION.get()


def environment_value(name, default=''):
    """Read a captured solver setting, with launch-environment compatibility."""
    active = _ACTIVE.get()
    if active is not None:
        if name == 'GHOST_DENSE_BACKEND':
            return 'cpu'
        key = _ENV_FIELDS.get(name)
        if key is not None:
            value = active[key]
            return '' if value is None else str(value)
    return os.environ.get(name, default)


def validate_for_run(options, method='direct', precision='double', scattering='monostatic', kind='2d'):
    value = validate_options(options)
    mode = value['factorization']
    if (value['mesh_strategy']=='adaptive' or value['basis_order']>1) and (kind!='2d' or scattering!='monostatic'):
        raise ValueError('Adaptive polynomial meshing supports 2D monostatic runs only.')
    if value['discretization']=='pulse' and (kind!='2d' or scattering!='monostatic' or precision!='double'):
        raise ValueError('Pulse collocation currently supports double-precision 2D monostatic solves only.')
    if value['mesh_strategy'] == 'local' and (kind != '2d' or scattering != 'monostatic'):
        raise ValueError('Local material meshing supports 2D monostatic runs only.')
    if kind != '2d' and mode != 'dense':
        raise ValueError('Hierarchical and compressed selections apply to the 2D solver only.')
    if mode != 'dense' and (precision != 'double' or scattering != 'monostatic'):
        raise ValueError('Hierarchical and compressed runs require monostatic scattering and double precision.')
    if mode in ('compressed', 'adaptive', 'fmm') and method not in ('auto','experimental_cpu','fmm'):
        raise ValueError('Compressed assembly requires CPU streaming kernel evaluation.')
    if method == 'experimental_cpu' and (precision != 'double' or scattering != 'monostatic'):
        raise ValueError('CPU streaming requires monostatic scattering and double precision.')
    return value


def temporary_directory():
    """Return and check the configured local directory for compressed spooling."""
    directory = option('temporary_directory', '') or tempfile.gettempdir()
    path = Path(directory)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError('Temporary directory is not available on this host: {}'.format(directory))
    return str(path)


@contextmanager
def execution_scope(value, limit_blas=False, assembly_threads=None, memory_budget_gib=None):
    """Restore settings after completion or failure; optionally control native BLAS."""
    checked = validate_options(value)
    allocation = assembly_threads if assembly_threads is not None else _ASSEMBLY_ALLOCATION.get()
    memory=memory_budget_gib if memory_budget_gib is not None else _MEMORY_ALLOCATION.get()
    if memory is not None and (not math.isfinite(memory) or memory <= 0):
        raise ValueError('Allocated solve memory must be positive and finite.')
    inherited_memory=_MEMORY_ALLOCATION.get()
    if inherited_memory is not None:
        memory=min(memory,inherited_memory)
    with _ACTIVE.override(checked), _ASSEMBLY_ALLOCATION.override(allocation), _MEMORY_ALLOCATION.override(memory):
        if not limit_blas:
            yield checked
            return
        import numpy
        import scipy.linalg
        from ghost_backend.execution.thread_control import threadpool_limits
        with _BLAS_LOCK:
            with threadpool_limits(limits=checked['blas_threads'], user_api='blas'):
                yield checked


def configured_execution(function):
    """Accept execution_options at public solver entry points."""
    signature = inspect.signature(function)
    def invoke_configured(*args, **kwargs):
        execution_start=time.perf_counter()
        requested = kwargs.pop('execution_options', None)
        inherited = current_options()
        # Public convenience spelling; internally retain the existing CPU
        # kernel/field lifecycle, with an explicitly matrix-free factorization.
        supplied=signature.bind(*args,**kwargs)
        # Normalize at the public boundary before planners or nested solves use
        # sequence truth tests. Keep dimensionality and finite-value validation.
        import numpy as np
        for name in ('frequencies_ghz','elevations_deg','incidence_angles_deg','observation_angles_deg'):
            if name not in supplied.arguments or supplied.arguments[name] is None:
                continue
            raw=np.asarray(supplied.arguments[name])
            if raw.ndim != 1 or raw.dtype.kind not in 'iuf':
                raise ValueError('{} must be a one-dimensional real numeric sequence.'.format(name))
            supplied.arguments[name]=raw.tolist()
        args,kwargs=supplied.args,supplied.kwargs
        method_parameter=signature.parameters.get('solver_method')
        public_method=str(supplied.arguments.get('solver_method',method_parameter.default if method_parameter else '')).strip().lower()
        automatic=public_method=='auto' and 'monostatic' in function.__name__
        if automatic:
            from ghost_backend.linalg.refined_lu import requested_precision
            if requested is not None and not isinstance(requested,dict):
                raise ValueError('Execution settings must be an object.')
            if inherited is None:
                # Explicit old profiles and launch overrides retain their meaning.
                base=from_environment()
                if not os.environ.get('GHOST_CPU_FACTORIZATION','').strip():
                    base['factorization']='adaptive'
                if requested is None or 'mesh_strategy' not in requested:
                    base['mesh_strategy']='adaptive'
                requested=dict(base,**(requested or {}))
            supplied.arguments['solver_method']='experimental_cpu' if requested_precision()=='double' else 'direct'
            if requested_precision()!='double' and inherited is None and requested['factorization']=='adaptive':
                requested['factorization']='dense'
            if 'max_panels' in signature.parameters and 'max_panels' not in supplied.arguments:
                supplied.arguments['max_panels']=100_000
            args,kwargs=supplied.args,supplied.kwargs
        fmm_requested=str(supplied.arguments.get('solver_method','')).strip().lower()=='fmm'
        if fmm_requested:
            if requested is not None and not isinstance(requested,dict):
                raise ValueError('Execution settings must be an object.')
            if requested is not None and requested.get('factorization','fmm')!='fmm':
                raise ValueError('solver_method=fmm conflicts with the requested factorization.')
            requested=dict(requested or inherited or {},factorization='fmm')
            supplied.arguments['solver_method']='experimental_cpu'
            args,kwargs=supplied.args,supplied.kwargs
        if requested is not None and inherited is not None and validate_options(requested) != inherited:
            raise ValueError('A nested solve must use the active execution settings.')
        value = requested if requested is not None else inherited
        if value is None and os.environ.get('GHOST_CPU_FACTORIZATION', '').strip().lower() == 'adaptive':
            value = from_environment()
        if value is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        from ghost_backend.linalg.refined_lu import requested_precision
        value = validate_for_run(value, method=bound.arguments.get('solver_method', 'direct'),
                         precision=requested_precision(),
                         scattering='bistatic' if 'bistatic' in function.__name__ else 'monostatic')
        timing_key=None
        if _PUBLIC_DEPTH.get()==1 and 'monostatic' in function.__name__:
            from ghost_backend.execution.timing_history import request_key
            try:
                with execution_scope(value):
                    timing_key=request_key(bound.arguments,value,function.__name__)
            except (OSError,TypeError,ValueError):
                pass  # Optional timing evidence never replaces input validation.
        selection = None
        requested_value = value
        if value['factorization'] == 'adaptive':
            if 'cfie_alpha' in bound.arguments:
                from ghost_backend.twod.solver import _validate_disabled_2d_cfie_alpha
                _validate_disabled_2d_cfie_alpha(bound.arguments['cfie_alpha'])
            from ghost_backend.execution.selection import select_backend, current_batch_selection
            selection = current_batch_selection()
            batch_planned=selection is not None
            if selection is None:
                planning_start=time.perf_counter()
                selection = select_backend(bound.arguments, value, certified='certified' in function.__name__)
                selection['planning_seconds']=time.perf_counter()-planning_start
            if timing_key is not None:
                from ghost_backend.execution.timing_history import adjust
                selection=adjust(selection,timing_key,batch=batch_planned)
            value = dict(value, factorization=selection['selected'])
        if bound.arguments.get('solver_method') == 'auto' and value['factorization'] in ('compressed','fmm'):
            bound.arguments['solver_method']='experimental_cpu'
            args,kwargs=bound.args,bound.kwargs
        def invoke():
            nonlocal value,selection
            from ghost_backend.execution.errors import BackendNumericalError
            from numpy.linalg import LinAlgError
            failures=[]
            modes=[value['factorization']]+(selection.get('retry_order',[]) if selection else [])
            for index,mode in enumerate(modes):
                value=dict(value,factorization=mode)
                attempt_start=time.perf_counter()
                try:
                    with execution_scope(value):
                        result=function(*args,**kwargs)
                except (MemoryError,BackendNumericalError,LinAlgError) as exc:
                    if selection is None or index == len(modes)-1:
                        raise
                    failures.append(dict(backend=mode,error=type(exc).__name__,message=str(exc),
                                         wall_seconds=time.perf_counter()-attempt_start))
                else:
                    if failures:
                        selection=dict(selection,initial_selection=selection['selected'],selected=mode,
                                       failed_attempts=failures,reason='Selected after an admitted numerical/resource retry.')
                    return result
                # Exception tracebacks no longer own failed native workspaces.
                import gc
                gc.collect()
            raise RuntimeError('Automatic backend retry exhausted.')

        with _AUTOMATIC_REQUEST.override(selection is not None or _AUTOMATIC_REQUEST.get()), execution_scope(value, limit_blas=requested is not None and inherited is None):
            result = invoke()
            if isinstance(result, dict):
                metadata = result.setdefault('metadata', {})
                adaptation = metadata.get('adaptive_mesh', {})
                actual = dict(value, basis_order=metadata.get('polynomial_degree', value['basis_order']))
                per_frequency = metadata.get('frequency_metadata', [])
                if per_frequency:
                    actual = dict(value)
                    metadata['execution_options_scope'] = 'request; effective settings are in frequency_metadata'
                if adaptation.get('final_backend') and not per_frequency:
                    actual['factorization'] = adaptation['final_backend']
                    if selection is not None and selection['selected'] != actual['factorization']:
                        selection = dict(selection, initial_selection=selection['selected'],
                            selected=actual['factorization'],
                            reason='Backend reselected on the adaptive mesh; see adaptive_mesh steps for admission evidence.')
                metadata['execution_options'] = actual
                result['metadata'].setdefault('polynomial_degree', value['basis_order'])
                result['metadata']['execution_wall_seconds']=time.perf_counter()-execution_start
                if allocated_memory_budget() is not None:
                    result['metadata']['execution_memory_reservation_gib']=allocated_memory_budget()
                if value['factorization']=='fmm' and not result['metadata'].get('adaptive_mesh'):
                    result['metadata']['solver_method']=('pulse_fmm_gmres' if value['discretization']=='pulse' else 'galerkin_fmm_gmres')
                    if fmm_requested:result['metadata']['solver_method_requested']='fmm'
                if selection is not None:
                    if per_frequency and metadata.get('backend_selection'):
                        metadata['request_backend_selection'] = selection
                    else:
                        metadata['backend_selection'] = selection
                    result['metadata']['requested_execution_options'] = requested_value
                if automatic:
                    result['metadata']['solver_method_requested']='auto'
                result['metadata']['execution_threads'] = dict(
                    assembly=effective_assembly_threads(), blas=current_options()['blas_threads'])
                if timing_key is not None:
                    from ghost_backend.execution.timing_history import record
                    record(timing_key,actual['factorization'],result['metadata']['execution_wall_seconds'],result['metadata'])
                result['metadata']['discretization']=value['discretization']
            return result
    @wraps(function)
    def call(*args,**kwargs):
        with _PUBLIC_DEPTH.override(_PUBLIC_DEPTH.get()+1):
            return invoke_configured(*args,**kwargs)
    parameters = list(signature.parameters.values())
    parameters.append(inspect.Parameter('execution_options', inspect.Parameter.KEYWORD_ONLY, default=None))
    call.__signature__ = signature.replace(parameters=parameters)
    return call
