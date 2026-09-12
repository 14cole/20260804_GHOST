"""Per-layer sensitivity and sampled yield with bounded working storage.

One response grid is evaluated at a time. Material interpolation is reused;
statistical trials retain scalar margins and aggregate failure counts only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from .compute import (INCH_TO_M, prepare_layer_properties_many, compute_angle_metrics_many,
                      validate_sweep_coverage, make_frequency_sweep, make_sweep)
from .io import MATERIAL_SINGULAR_TOL
from .tolerance_config import PARAMETERS, layer_parameters, validate_setup, validate_tolerances


class StopToleranceAnalysis(Exception):
    """No partial trial or incomplete statistical estimate is published."""


@dataclass(frozen=True)
class Parameter:
    layer: int
    key: str
    bound: float
    units: str
    distribution: str
    group: str
    loading: float

    @property
    def label(self):
        return f'Layer {self.layer + 1} · {PARAMETERS[self.key]}'


def parameters_from_layers(layers):
    params = []
    for i, layer in enumerate(layers):
        specs = validate_tolerances(layer.tolerances, layer.is_sheet)
        # JSON key ordering must not change which Sobol coordinate drives an input.
        for key in layer_parameters(layer.is_sheet):
            spec = specs.get(key)
            if spec and spec['bound'] > 0:
                params.append(Parameter(i, key, **spec))
    if not params:
        raise ValueError('Set a nonzero tolerance bound for at least one layer parameter.')
    if len(params) > 128:
        raise ValueError('Use at most 128 active tolerance parameters per study.')
    return params


def study_grid(setup):
    s = validate_setup(setup)
    return (make_frequency_sweep(float(s['f_start']), float(s['f_stop']), float(s['f_step'])),
            make_sweep(float(s['a_start']), float(s['a_stop']), float(s['a_step'])),
            ['te', 'tm'] if s['polarization'] == 'Both' else [s['polarization'].lower()])


def study_workload(setup, parameter_count, layer_count):
    s = validate_setup(setup)
    freqs, angles, pols = study_grid(s)
    nstat = int(s['samples']) if s['mode'] != 'Sensitivity only' else 0
    trials = 1 + parameter_count * (int(s['points']) - 1) + nstat
    points = len(freqs) * len(angles) * len(pols)
    # Conservative array allowance, not a whole-process or OS RAM prediction.
    array_bytes = 40 * points + 256 * len(freqs) * max(1, layer_count) + nstat * 8 + 128 * (parameter_count + 1) * 8
    return dict(evaluations=trials, response_points=trials * points, grid_points=points,
                array_mib=array_bytes / 2**20)


def sampled_deviations(parameters, samples, seed, batch_size=64):
    """Stream the entire scrambled Sobol sequence without skipping/thinning.

    The completed sequence has a power-of-two length. Gaussian one-factor
    groups give latent correlation loading_i * loading_j within each group.
    """
    from scipy.special import ndtr, ndtri
    from scipy.stats import qmc

    if samples < 16 or samples & (samples - 1) or samples > 65536:
        raise ValueError('Statistical samples must be a power of two from 16 to 65536.')
    if batch_size < 1 or batch_size & (batch_size - 1):
        raise ValueError('Sampling batch size must be a positive power of two.')
    groups = sorted({p.group for p in parameters if p.group})
    group_indices = {g: len(parameters) + i for i, g in enumerate(groups)}
    sampler = qmc.Sobol(d=len(parameters) + len(groups), scramble=True, seed=seed)
    lower, upper = ndtr(-3.), ndtr(3.)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        u = sampler.random(count)
        z = ndtri(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
        for j, p in enumerate(parameters):
            if p.group:
                v = p.loading * z[:, group_indices[p.group]] + np.sqrt(1 - p.loading**2) * z[:, j]
                u[:, j] = ndtr(v)
            if p.distribution == 'Uniform':
                u[:, j] = 2 * u[:, j] - 1
            else:
                u[:, j] = ndtri(lower + u[:, j] * (upper - lower)) / 3
        yield u[:, :len(parameters)]


class PreparedStudy:
    def __init__(self, layers, frequencies, angles, polarizations, parameters,
                 stop_requested=lambda: False, compute_metrics=compute_angle_metrics_many):
        self.layers = layers
        self.freqs, self.angles, self.pols = frequencies, angles, polarizations
        self.params, self.stop_requested, self.compute_metrics = parameters, stop_requested, compute_metrics
        for layer in layers:
            if not layer.is_sheet:
                validate_sweep_coverage(frequencies, layer.table_0deg, 'Sensitivity material')
                if layer.anisotropic:
                    validate_sweep_coverage(frequencies, layer.table_90deg, 'Sensitivity directional material')
        self.properties = [None if p is None else tuple(np.asarray(v, dtype=complex) for v in p)
                           for p in prepare_layer_properties_many(frequencies, layers)]
        self.widths = []
        for p in parameters:
            layer = layers[p.layer]
            if p.key == 'thickness':
                nominal = layer.thickness_m / INCH_TO_M
            elif p.key == 'sheet_resistance':
                nominal = layer.sheet_resistance
            else:
                material = self.properties[p.layer][0 if p.key.startswith('eps') else 1]
                nominal = material.real if p.key.endswith('real') else material.imag
            width = p.bound * np.abs(nominal) / 100 if p.units == '%' else p.bound
            self.widths.append(width)
            if not np.all(np.isfinite(width)):
                raise ValueError(f'{p.label}: tolerance produces non-finite values.')
            if not np.any(np.asarray(width) > 0):
                raise ValueError(f'{p.label}: percentage of a zero component has no effect. Set a physical nominal value or disable this tolerance.')
            if p.key in ('thickness', 'sheet_resistance') and np.any(nominal - width <= 0):
                raise ValueError(f'{p.label}: tolerance must keep thickness/resistance strictly positive.')
            if p.key.endswith('imag') and np.any(nominal + width > 0):
                raise ValueError(f'{p.label}: tolerance would introduce gain; signed imaginary components must stay ≤ 0.')
        # Validate the full rectangular material support, including simultaneous
        # real/imaginary deviations: neither epsilon nor mu may approach zero.
        for i, props in enumerate(self.properties):
            if props is None:
                continue
            for name, values in zip(('eps', 'mu'), props):
                wr = wi = 0.
                for p, width in zip(parameters, self.widths):
                    if p.layer == i and p.key == name + '_real': wr = width
                    if p.layer == i and p.key == name + '_imag': wi = width
                closest = np.hypot(np.maximum(0, np.abs(values.real) - wr), np.maximum(0, np.abs(values.imag) - wi))
                if np.any(closest <= MATERIAL_SINGULAR_TOL):
                    raise ValueError(f'Layer {i+1}: {name} tolerance support includes a singular medium; reduce its bounds.')

    def response(self, deviations):
        layers = [replace(layer) for layer in self.layers]
        props = list(self.properties)
        changed = {}
        for p, width, delta in zip(self.params, self.widths, deviations):
            if delta == 0:
                continue
            shift = width * delta
            if p.key == 'thickness':
                layers[p.layer].thickness_m += shift * INCH_TO_M
            elif p.key == 'sheet_resistance':
                layers[p.layer].sheet_resistance += shift
            else:
                if p.layer not in changed:
                    changed[p.layer] = [v.copy() for v in props[p.layer]]
                component = changed[p.layer][0 if p.key.startswith('eps') else 1]
                if p.key.endswith('real'): component.real += shift
                else: component.imag += shift
        for i, values in changed.items():
            props[i] = values
        response = np.empty((len(self.pols), len(self.angles), len(self.freqs)))
        for pi, pol in enumerate(self.pols):
            for ai, angle in enumerate(self.angles):
                if self.stop_requested():
                    raise StopToleranceAnalysis()
                response[pi, ai] = self.compute_metrics(self.freqs, angle, layers, pol,
                                                       prepared_properties=props, return_arrays=True)['metal_loss_db']
        if not np.all(np.isfinite(response)):
            raise ValueError('A tolerance trial produced non-finite reflection.')
        return response


def crossing_brackets(offsets, margins):
    """First outward pass/miss bracket in each direction; no monotonicity claim."""
    center = len(offsets) // 2
    if margins[center] < 0:
        return {'negative': None, 'positive': None, 'nominal_pass': False}
    result = {'nominal_pass': True}
    for direction, indices in [('negative', range(center - 1, -1, -1)), ('positive', range(center + 1, len(offsets)))]:
        last = 0.
        result[direction] = None
        for i in indices:
            if margins[i] < 0:
                result[direction] = [last, float(offsets[i])]
                break
            last = float(offsets[i])
    return result


def run_tolerance_study(layers, layer_configs, setup, *, stop_requested=lambda: False,
                        progress=lambda *_: None, compute_metrics=compute_angle_metrics_many):
    setup = validate_setup(setup)
    params = parameters_from_layers(layer_configs)
    freqs, angles, pols = study_grid(setup)
    workload = study_workload(setup, len(params), len(layers))
    engine = PreparedStudy(layers, freqs, angles, pols, params, stop_requested, compute_metrics)
    zero = np.zeros(len(params))
    target = float(setup['target'])
    nominal = engine.response(zero)
    nominal_margin = target - float(nominal.max())
    done = 1
    progress(done, workload['evaluations'], 'Nominal')
    offsets = np.linspace(-1, 1, int(setup['points']))
    sensitivities = []
    for j, p in enumerate(params):
        margins = np.empty(len(offsets))
        for i, offset in enumerate(offsets):
            if offset == 0:
                margins[i] = nominal_margin
                continue
            values = zero.copy()
            values[j] = offset
            margins[i] = target - float(engine.response(values).max())
            done += 1
            progress(done, workload['evaluations'], p.label)
        sensitivities.append(dict(label=p.label, parameter=vars(p), margins_db=margins.tolist(),
                                  margin_loss_db=nominal_margin - float(margins.min()),
                                  crossings=crossing_brackets(offsets, margins)))
    nstat = int(setup['samples']) if setup['mode'] != 'Sensitivity only' else 0
    counts = np.zeros_like(nominal, dtype=np.uint32) if nstat else None
    trial_margins = np.empty(nstat)
    convergence = []
    completed = passing = 0
    if nstat:
        for batch in sampled_deviations(params, nstat, int(setup['seed'])):
            for values in batch:
                response = engine.response(values)
                margin = target - float(response.max())
                counts += response > target
                trial_margins[completed] = margin
                completed += 1
                passing += int(margin >= 0)
                done += 1
                if completed & (completed - 1) == 0:
                    convergence.append([completed, 100 * passing / completed])
                progress(done, workload['evaluations'], 'Statistical trials')
                del response
    if stop_requested():
        raise StopToleranceAnalysis()
    nominal_worst = np.unravel_index(int(nominal.argmax()), nominal.shape)
    source_hash = hashlib.sha256()
    for layer, props in zip(layers, engine.properties):
        source_hash.update(repr((layer.thickness_m, layer.sheet_resistance, layer.anisotropic, layer.polarization_deg)).encode())
        if props:
            for values in props:
                source_hash.update(np.asarray(values, dtype='<c16').tobytes())
    import scipy
    implementation = hashlib.sha256()
    for name in ('tolerance_analysis.py', 'tolerance_config.py', 'compute.py'):
        implementation.update(Path(__file__).with_name(name).read_bytes())
    return dict(setup=copy.deepcopy(setup), frequencies_ghz=freqs, angles_deg=angles, polarizations=pols,
                target_db=target, nominal_reflection_db=nominal, nominal_margin_db=nominal_margin,
                nominal_worst=dict(polarization=pols[nominal_worst[0]], angle_deg=angles[nominal_worst[1]], frequency_ghz=freqs[nominal_worst[2]]),
                offsets=offsets.tolist(), sensitivities=sensitivities, trial_margins_db=trial_margins,
                failure_counts=counts, trials=nstat, passing=passing,
                pass_fraction_pct=100 * passing / nstat if nstat else None, convergence=convergence,
                evaluated=done, workload=workload, effective_material_sha256=source_hash.hexdigest(),
                versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                              method='freddy-tolerance-v1', implementation_sha256=implementation.hexdigest()),
                sampling=dict(method='scrambled Sobol', seed=int(setup['seed']), trials=nstat,
                              distribution_convention='Uniform within ±bound; truncated normal with underlying sigma=bound/3 and hard limits ±bound.',
                              correlation_model='Shared Gaussian factor per named group: z_i = loading_i*z_group + sqrt(1-loading_i**2)*z_independent. Within-group latent correlation is loading_i*loading_j; transformed Pearson correlations can differ. Blank groups are independent.'),
                interpretation='Pass fraction is a model estimate under the captured distributions/group correlations and sampled frequencies/angles. It is not a manufacturing guarantee. Sensitivity is one parameter at a time; crossing brackets do not guarantee continuous or joint tolerance acceptance.')


def export_tolerance_report(path, result):
    from .io import _atomic_text_file
    def encode(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        raise TypeError(type(obj).__name__)
    with _atomic_text_file(path) as stream:
        json.dump(result, stream, default=encode, allow_nan=False, indent=2)
        stream.write('\n')
