"""Serializable, Qt-free per-layer manufacturing inputs."""
from __future__ import annotations

import math

PARAMETERS = {'thickness': 'Thickness', 'eps_real': 'ε′', 'eps_imag': 'ε″',
              'mu_real': 'μ′', 'mu_imag': 'μ″', 'sheet_resistance': 'Sheet resistance'}
DISTRIBUTIONS = ('Uniform', 'Truncated normal (±3σ)')
DEFAULT_SPEC = dict(bound=0., units='%', distribution=DISTRIBUTIONS[0], group='', loading=1.)
SETUP_DEFAULTS = dict(f_start='8', f_stop='12', f_step='0.1', a_start='0', a_stop='30',
                      a_step='10', polarization='Both', target='-10', points='21',
                      mode='Sensitivity only', samples='1024', seed='12345')
MODES = ('Sensitivity only', 'Sensitivity + statistical trials')


def layer_parameters(is_sheet):
    return ('sheet_resistance',) if is_sheet else tuple(k for k in PARAMETERS if k != 'sheet_resistance')


def validate_tolerances(raw, is_sheet):
    if not isinstance(raw, dict):
        raise ValueError('Layer tolerances must be an object.')
    result = {}
    for key, value in raw.items():
        if key not in layer_parameters(is_sheet) or not isinstance(value, dict):
            raise ValueError(f'Unsupported layer tolerance: {key}.')
        spec = {**DEFAULT_SPEC, **value}
        try:
            spec['bound'], spec['loading'] = float(spec['bound']), float(spec['loading'])
        except (TypeError, ValueError):
            raise ValueError(f'{PARAMETERS[key]}: bound and loading must be numeric.') from None
        if not math.isfinite(spec['bound']) or spec['bound'] < 0:
            raise ValueError(f'{PARAMETERS[key]}: bound must be finite and nonnegative.')
        if not math.isfinite(spec['loading']) or not -1 <= spec['loading'] <= 1:
            raise ValueError('Group loading must be between −1 and 1.')
        if spec['units'] not in ('%', 'Absolute') or spec['distribution'] not in DISTRIBUTIONS:
            raise ValueError(f'{PARAMETERS[key]}: invalid units or distribution.')
        spec['group'] = str(spec['group']).strip()
        if len(spec['group']) > 80:
            raise ValueError('Group names must have at most 80 characters.')
        result[key] = {k: spec[k] for k in DEFAULT_SPEC}
    return result


def validate_setup(raw):
    if not isinstance(raw, dict):
        raise ValueError('Sensitivity setup must be an object.')
    state = {k: str(raw.get(k, v)) for k, v in SETUP_DEFAULTS.items()}
    try:
        nums = {k: float(state[k]) for k in ('f_start', 'f_stop', 'f_step', 'a_start', 'a_stop', 'a_step', 'target')}
        integers = {k: int(state[k]) for k in ('points', 'samples', 'seed')}
    except (ValueError, TypeError):
        raise ValueError('Sensitivity setup requires numeric sweep/target values and integer counts/seed.') from None
    if not all(math.isfinite(v) for v in nums.values()):
        raise ValueError('Sensitivity setup must use finite values.')
    if not 0 < nums['f_start'] <= nums['f_stop'] or nums['f_step'] <= 0:
        raise ValueError('Frequency start/step must be positive; stop must be at least start.')
    if not 0 <= nums['a_start'] <= nums['a_stop'] < 90 or nums['a_step'] <= 0:
        raise ValueError('Angles must satisfy 0 ≤ start ≤ stop < 90°, with a positive step.')
    if not -200 <= nums['target'] <= 0:
        raise ValueError('Reflection requirement must be between −200 and 0 dB.')
    if state['polarization'] not in ('TE', 'TM', 'Both') or state['mode'] not in MODES:
        raise ValueError('Invalid sensitivity mode or polarization.')
    if not 5 <= integers['points'] <= 101 or integers['points'] % 2 == 0:
        raise ValueError('Sensitivity points must be an odd integer from 5 to 101.')
    n = integers['samples']
    if not 16 <= n <= 65536 or n & (n - 1):
        raise ValueError('Statistical samples must be a power of two from 16 to 65536.')
    if not 0 <= integers['seed'] < 2**32:
        raise ValueError('Seed must be an integer from 0 to 4294967295.')
    # Reject accidental huge grids before make_sweep allocates Python lists.
    nf = (nums['f_stop'] - nums['f_start']) / nums['f_step'] + 1
    na = (nums['a_stop'] - nums['a_start']) / nums['a_step'] + 1
    if nf > 20001 or na > 1001 or nf * na * (2 if state['polarization'] == 'Both' else 1) > 2_000_000:
        raise ValueError('Sensitivity grid is too large; use at most 20,001 frequencies, 1,001 angles and 2 million response points per trial.')
    return state
