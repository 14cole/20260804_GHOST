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

from ghost_backend.execution.runtime import ScopedValue

DEFAULTS = {
    'version': 1,
    'factorization': 'dense',
    'mesh_strategy': 'global',
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
}
EFFICIENT_DEFAULTS = dict(DEFAULTS, factorization='compressed', compressed_storage_mib=8192,
                          assembly_threads=4, blas_threads=2)
_ACTIVE = ScopedValue('ghost_execution_options', default=None)
_ASSEMBLY_ALLOCATION = ScopedValue('ghost_assembly_allocation', default=None)
_BLAS_LOCK = threading.RLock()
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
    if result['factorization'] not in ('dense', 'hierarchical', 'auto', 'compressed', 'adaptive'):
        raise ValueError('Choose dense, hierarchical, auto, compressed, or adaptive factorization.')
    if result['mesh_strategy'] not in ('global', 'local'):
        raise ValueError('Mesh strategy must be global or local.')
    if result['rhs_compression'] not in ('off', 'auto', 'on'):
        raise ValueError('RHS compression must be off, auto, or on.')
    for key, lower, upper in [('compressed_storage_mib', 16, 1048576),
                              ('blas_threads', 1, 1024), ('angle_batch_size', 1, 256),
                              ('assembly_tile', 0, 65536), ('far_quadrature_order', 0, 64)]:
        number = result[key]
        if type(number) is not int or not lower <= number <= upper:
            raise ValueError('{} must be an integer from {} to {}.'.format(key, lower, upper))
    threads = result['assembly_threads']
    if threads != 'auto' and (type(threads) is not int or not 1 <= threads <= 1024):
        raise ValueError('Assembly threads must be auto or an integer from 1 to 1024.')
    ram = result['ram_budget_gib']
    if ram is not None and (type(ram) not in (int, float) or not math.isfinite(ram) or ram <= 0):
        raise ValueError('RAM budget must be positive GiB or null for available memory.')
    if type(result['far_grading']) is not bool:
        raise ValueError('Far grading must be true or false.')
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
    if name == 'small':
        values.update(factorization='dense', rhs_compression='off')
        method = 'direct'
    elif name == 'balanced':
        values.update(factorization='dense')
        method = 'experimental_cpu'
    elif name == 'adaptive':
        values.update(factorization='adaptive')
        method = 'experimental_cpu'
    elif name == 'large':
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
    if value['mesh_strategy'] == 'local' and (kind != '2d' or scattering != 'monostatic'):
        raise ValueError('Local material meshing supports 2D monostatic runs only.')
    if kind != '2d' and mode != 'dense':
        raise ValueError('Hierarchical and compressed selections apply to the 2D solver only.')
    if mode != 'dense' and (precision != 'double' or scattering != 'monostatic'):
        raise ValueError('Hierarchical and compressed runs require monostatic scattering and double precision.')
    if mode in ('compressed', 'adaptive') and method != 'experimental_cpu':
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
def execution_scope(value, limit_blas=False, assembly_threads=None):
    """Restore settings after completion or failure; optionally control native BLAS."""
    checked = validate_options(value)
    allocation = assembly_threads if assembly_threads is not None else _ASSEMBLY_ALLOCATION.get()
    with _ACTIVE.override(checked), _ASSEMBLY_ALLOCATION.override(allocation):
        if not limit_blas:
            yield checked
            return
        import numpy
        import scipy.linalg
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            raise RuntimeError('Install the project dependencies: threadpoolctl is required for saved BLAS thread limits.')
        with _BLAS_LOCK:
            with threadpool_limits(limits=checked['blas_threads'], user_api='blas'):
                yield checked


def configured_execution(function):
    """Accept execution_options at public solver entry points."""
    signature = inspect.signature(function)
    @wraps(function)
    def call(*args, **kwargs):
        requested = kwargs.pop('execution_options', None)
        inherited = current_options()
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
        selection = None
        requested_value = value
        if value['factorization'] == 'adaptive':
            from ghost_backend.execution.selection import select_backend
            selection = select_backend(bound.arguments, value, certified='certified' in function.__name__)
            value = dict(value, factorization=selection['selected'])
        with execution_scope(value, limit_blas=requested is not None and inherited is None):
            result = function(*args, **kwargs)
            if isinstance(result, dict):
                result.setdefault('metadata', {})['execution_options'] = current_options()
                if selection is not None:
                    result['metadata']['backend_selection'] = selection
                    result['metadata']['requested_execution_options'] = requested_value
                result['metadata']['execution_threads'] = dict(
                    assembly=effective_assembly_threads(), blas=current_options()['blas_threads'])
            return result
    parameters = list(signature.parameters.values())
    parameters.append(inspect.Parameter('execution_options', inspect.Parameter.KEYWORD_ONLY, default=None))
    call.__signature__ = signature.replace(parameters=parameters)
    return call
