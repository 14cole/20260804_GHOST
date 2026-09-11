"""Capture execution settings for batch planning, manifests, and workers."""
from functools import wraps
import json
import time
from pathlib import Path

from ghost_backend.execution.options import (
    current_options, efficient_defaults, execution_scope, from_environment, validate_for_run, validate_options,
)

RESOURCE_KEYS = {'ASSEMBLY_THREADS': 'assembly_threads',
                 'BLAS_THREADS_PER_WORKER': 'blas_threads', 'MAX_SOLVE_GB': 'ram_budget_gib'}


def driver_options(namespace):
    raw = namespace.get('EXECUTION_OPTIONS')
    if raw is None:
        defaults = efficient_defaults() if (namespace.get('SOLVER_METHOD') == 'experimental_cpu'
            and namespace.get('LU_PRECISION', 'double') == 'double') else None
        raw = from_environment(defaults)
        for key, field in RESOURCE_KEYS.items():
            if key in namespace and not (key == 'MAX_SOLVE_GB' and namespace[key] is None):
                raw[field] = namespace[key]
    return validate_for_run(raw, namespace.get('SOLVER_METHOD', 'direct'),
                            namespace.get('LU_PRECISION', 'double'))


def reconcile_settings(settings):
    """Validate explicit profile/driver resource values and fill matching keys."""
    checked = dict(settings)
    if checked.get('EXECUTION_OPTIONS') is None:
        return checked
    profile = validate_for_run(checked['EXECUTION_OPTIONS'], checked.get('SOLVER_METHOD', 'direct'),
                               checked.get('LU_PRECISION', 'double'))
    for key, field in RESOURCE_KEYS.items():
        if key in checked and checked[key] != profile[field]:
            raise ValueError('{} conflicts with EXECUTION_OPTIONS.{}'.format(key, field))
        checked[key] = profile[field]
    checked['EXECUTION_OPTIONS'] = profile
    return checked


def driver_execution(function):
    """Scope driver planning and worker management to a captured profile."""
    @wraps(function)
    def call(*args, **kwargs):
        namespace = function.__globals__
        profile = None
        if function.__name__ == 'worker':
            directory = args[0] if args else kwargs['run_dir_str']
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
            config = manifest['solver_config']
            if config.get('execution_options') is not None:
                profile = validate_for_run(config['execution_options'], config.get('solver_method', 'direct'),
                                           config.get('lu_precision', 'double'))
        if profile is None:
            profile = driver_options(namespace)
        previous = {key: namespace[key] for key in RESOURCE_KEYS}
        try:
            for key, field in RESOURCE_KEYS.items():
                namespace[key] = profile[field]
            with execution_scope(profile):
                return function(*args, **kwargs)
        finally:
            namespace.update(previous)
    return call


def unit_execution(function):
    """Apply manifest settings inside a fresh or reused pool worker."""
    @wraps(function)
    def call(unit, context, destination):
        profile = context.get('execution_options') or current_options() or driver_options(function.__globals__)
        from ghost_backend.execution.metrics import progress_listener
        last = [0.0]
        def progress(event):
            now = time.monotonic()
            if now - last[0] < 10.0:
                return
            last[0] = now
            rss = event.get('process_rss_bytes')
            memory = '; RSS {:.2f} GiB'.format(rss / 1024**3) if rss is not None else ''
            print('  {} | {}{} | {:.1f}s{}'.format(unit.get('name', unit.get('geometry', 'solve')),
                event['phase'] + ': ' if event['phase'] else '', event['stage'],
                event['elapsed_seconds'], memory), flush=True)
        with execution_scope(profile, limit_blas=True,
                             assembly_threads=context.get('execution_assembly_threads')):
            with progress_listener(progress):
                return function(unit, context, destination)
    return call
