"""One entry point for batch solve presets and deliberate execution overrides."""
from ghost_backend.execution.options import geometry_preset, validate_for_run, validate_options

PRESETS = ('auto', 'automatic', 'small', 'balanced', 'large', 'custom')
LEGACY_KEYS = ('SOLVER_METHOD', 'LU_PRECISION', 'EXECUTION_OPTIONS',
               'ASSEMBLY_THREADS', 'BLAS_THREADS_PER_WORKER', 'MAX_SOLVE_GB')
RESOURCE_KEYS = {'ASSEMBLY_THREADS': 'assembly_threads',
                 'BLAS_THREADS_PER_WORKER': 'blas_threads', 'MAX_SOLVE_GB': 'ram_budget_gib'}


def validate_overrides(value):
    if not isinstance(value, dict):
        raise ValueError('ADVANCED_OVERRIDES must be an execution-settings object.')
    validate_options(value)
    return dict(value)


def resolve_preset(settings):
    """Resolve a named preset; keep old explicit recipes authoritative.

    New presets and legacy settings cannot silently disagree. Old JSON without
    a preset opts into custom mode, preserving its kernel/profile semantics.
    """
    result = dict(settings)
    name = result.get('SOLVE_PRESET')
    overrides = validate_overrides(result.get('ADVANCED_OVERRIDES', {}))
    if name is not None and name not in PRESETS:
        raise ValueError('SOLVE_PRESET must be auto, small, balanced, large, or custom.')
    if name in (None, 'custom'):
        if overrides:
            raise ValueError('ADVANCED_OVERRIDES requires a named SOLVE_PRESET.')
        if name is None and any(key in result for key in LEGACY_KEYS):
            result['SOLVE_PRESET'] = 'custom'
        return result
    recipe = geometry_preset('adaptive' if name in ('auto', 'automatic') else name)
    profile = dict(recipe['execution_options'], **overrides)
    for key, field in RESOURCE_KEYS.items():
        if key in result:
            if field in overrides and result[key] != profile[field]:
                raise ValueError('{} conflicts with ADVANCED_OVERRIDES.{}'.format(key, field))
            profile[field] = result[key]
    profile = validate_for_run(profile, recipe['solver_method'], recipe['lu_precision'])
    for key, expected in (('SOLVER_METHOD', recipe['solver_method']),
                          ('LU_PRECISION', recipe['lu_precision']), ('EXECUTION_OPTIONS', profile)):
        if key in result and result[key] != expected:
            raise ValueError('{} conflicts with SOLVE_PRESET; use custom for a legacy profile.'.format(key))
        result[key] = expected
    result.update({key: profile[field] for key, field in RESOURCE_KEYS.items()})
    return result
