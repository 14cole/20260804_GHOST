"""BOR execution controls, independent of the 2D execution profile."""
from functools import wraps
import inspect
from ghost_backend.execution.runtime import ScopedValue


DEFAULTS = dict(version=1, angle_batch_size=64, rhs_compression='auto',
                factorization='dense', compressed_storage_mib=2048,
                compression_tile=32, tile_cache_mib=16)
_ACTIVE = ScopedValue('ghost_bor_options', default=None)
_ABORT = ScopedValue('ghost_bor_abort', default=None)


def validate_options(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('BOR options must contain supported fields only.')
    result = dict(DEFAULTS)
    result.update(value)
    for name, lower, upper in (('version', 1, 1), ('angle_batch_size', 1, 256),
                               ('compressed_storage_mib', 16, 1048576),
                               ('compression_tile', 8, 128)):
        number = result[name]
        if type(number) is not int or not lower <= number <= upper:
            raise ValueError('BOR {} must be an integer in {}..{}.'.format(name, lower, upper))
    if type(result['tile_cache_mib']) is not int or not 0 <= result['tile_cache_mib'] <= 4096:
        raise ValueError('BOR tile_cache_mib must be an integer in 0..4096.')
    if result['rhs_compression'] not in ('off', 'auto', 'on'):
        raise ValueError('BOR rhs_compression must be off, auto, or on.')
    if result['factorization'] not in ('dense', 'compressed'):
        raise ValueError('BOR factorization must be dense or compressed.')
    return result


def current_options():
    return dict(_ACTIVE.get() or DEFAULTS)


def compressed_requested():
    return current_options()['factorization'] == 'compressed'


def option_scope(value):
    return _ACTIVE.override(value)


def current_checkpoint():
    return _ABORT.get()


def configured(function):
    """Accept bor_options= on public APIs, restoring nested calls on failure."""
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        supplied = kwargs.pop('bor_options', None)
        options = current_options() if supplied is None else validate_options(supplied)
        bound = signature.bind_partial(*args, **kwargs)
        checkpoint = bound.arguments.get('check_abort', current_checkpoint())
        if options['factorization'] == 'compressed':
            precision = str(bound.arguments.get('table_precision', 'auto')).strip().lower()
            if precision not in ('auto', 'double', 'single'):
                raise ValueError('BOR table_precision must be auto, single, or double.')
            if str(bound.arguments.get('assembly', 'auto')).strip().lower() not in ('auto', 'tables', 'streaming'):
                raise ValueError('BOR assembly must be auto, tables, or streaming.')
            if precision == 'single':
                raise ValueError('Compressed BOR assembly requires double precision.')
            for name, value in (('assembly', 'tables'), ('table_precision', 'double')):
                if name in signature.parameters:
                    bound.arguments[name] = value
            args, kwargs = bound.args, bound.kwargs
        from ghost_backend.bor.cache import TileCache, current_cache, cache_scope
        cache = current_cache() if options['factorization'] == 'compressed' else None
        if (cache is None or supplied is not None) and options['factorization'] == 'compressed':
            cache = TileCache(options['tile_cache_mib'] * 1024**2)
        with _ACTIVE.override(options):
            with _ABORT.override(checkpoint):
                with cache_scope(cache):
                    result = function(*args, **kwargs)
        if isinstance(result, dict):
            result['bor_execution_options'] = dict(options)
            if cache is not None:
                result['bor_tile_cache'] = cache.evidence()
                if isinstance(result.get('metadata'), dict):
                    result['metadata']['bor_tile_cache'] = cache.evidence()
            if isinstance(result.get('metadata'), dict):
                result['metadata']['bor_execution_options'] = dict(options)
            if options['factorization'] == 'compressed' and 'assembly' in result:
                result['assembly'] = 'compressed'
        return result
    parameters = list(signature.parameters.values())
    position = next((i for i, parameter in enumerate(parameters)
                     if parameter.kind == inspect.Parameter.VAR_KEYWORD), len(parameters))
    parameters.insert(position, inspect.Parameter('bor_options', inspect.Parameter.KEYWORD_ONLY, default=None))
    wrapped.__signature__ = signature.replace(parameters=parameters)
    return wrapped


def bounded_rhs_count(count, polarizations=2):
    return min(int(count), polarizations * current_options()['angle_batch_size'])
