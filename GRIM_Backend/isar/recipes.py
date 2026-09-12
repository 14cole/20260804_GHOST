"""Versioned, non-executable ISAR recipes with physical source selectors."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import numpy as np

RECIPE_SCHEMA = 'grim.isar-recipe.v1'
MAX_RECIPE_BYTES = 2 * 1024**2


def recipe_from_params(params):
    from GRIM_Backend.plotting.modes.isar_mode import _angle_values_to_degrees
    dataset = params['dataset']
    indices = sorted({int(i) for band in params['bands'] for i in band})
    target = params.get('az_target_deg')
    return {'schema': RECIPE_SCHEMA,
        'azimuth_degrees': _angle_values_to_degrees(dataset, 'azimuth', dataset.azimuths[indices]).tolist(),
        'frequency_hz': np.asarray(params['freq_hz'], dtype=float).tolist(),
        'elevation_degrees': float(params['elevation_deg']),
        'polarization': str(dataset.polarizations[params['pol_idx']]),
        'options': {'window': str(params['window_name']), 'reconstruction': str(params['recon']),
            'length_unit': str(params['unit_name']), 'aperture_center_degrees': params.get('az_center_deg'),
            'azimuth_target_degrees': None if target is None else np.asarray(target, dtype=float).tolist(),
            'l1_strength': float(params['l1_strength']), 'l1_iterations': int(params['l1_iters']),
            'flip_x': bool(params['flip_x']), 'flip_y': bool(params['flip_y']),
            'aperture_mode': str(params.get('aperture_mode', 'auto')),
            'scene_half_extent_m': params.get('scene_half_extent_m'),
            'composite_side': int(params.get('composite_side', 1024)),
            'native_diagnostics': bool(params.get('native_diagnostics', True))}}


def save_recipe(path, recipe):
    text = json.dumps(recipe, indent=2, allow_nan=False)
    if len(text.encode('utf-8')) > MAX_RECIPE_BYTES:
        raise ValueError('ISAR recipe exceeds its size limit')
    destination = Path(path).expanduser().resolve()
    fd, temp = tempfile.mkstemp(prefix='.isar-recipe-', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return destination


def load_recipe(path):
    with Path(path).open('rb') as stream:
        content = stream.read(MAX_RECIPE_BYTES + 1)
    if len(content) > MAX_RECIPE_BYTES:
        raise ValueError('ISAR recipe exceeds its size limit')
    recipe = json.loads(content)
    if not isinstance(recipe, dict) or recipe.get('schema') != RECIPE_SCHEMA:
        raise ValueError('Unsupported ISAR recipe schema')
    return recipe


def _matching_indices(axis, values, *, atol):
    axis, values = np.asarray(axis, dtype=float), np.asarray(values, dtype=float)
    if axis.ndim != 1 or not axis.size or not np.all(np.isfinite(axis)):
        raise ValueError('Active dataset has an invalid physical axis')
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError('Recipe selectors must be finite one-dimensional arrays')
    if len(np.unique(values)) != len(values):
        raise ValueError('Recipe selectors contain duplicates')
    order = np.argsort(axis)
    sorted_axis = axis[order]
    loc = np.clip(np.searchsorted(sorted_axis, values), 0, len(axis) - 1)
    before = np.maximum(0, loc - 1)
    loc = np.where(abs(sorted_axis[before] - values) < abs(sorted_axis[loc] - values), before, loc)
    if not np.allclose(sorted_axis[loc], values, rtol=1e-12, atol=atol):
        raise ValueError('The active dataset lacks physical samples required by this recipe')
    ids = order[loc].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError('Recipe selectors map ambiguously to source samples')
    return ids


def recipe_arguments(dataset, recipe):
    from GRIM_Backend.plotting.modes.isar_mode import (
        _angle_values_to_degrees, _unit_to_hz_scale, _window_name, _length_unit, _sparse_parameters,
    )
    from .quality import aperture_mode, scene_extents
    if not isinstance(recipe, dict) or recipe.get('schema') != RECIPE_SCHEMA:
        raise ValueError('Unsupported ISAR recipe schema')
    for key in ('azimuth_degrees', 'frequency_hz'):
        if np.asarray(recipe.get(key)).ndim != 1 or len(recipe[key]) < 2:
            raise ValueError('Recipe needs at least two angular and frequency samples')
    options = dict(recipe.get('options', {}))
    allowed = {'window', 'reconstruction', 'length_unit', 'aperture_center_degrees', 'azimuth_target_degrees',
               'l1_strength', 'l1_iterations', 'flip_x', 'flip_y', 'aperture_mode', 'scene_half_extent_m',
               'composite_side', 'native_diagnostics'}
    if set(options) - allowed:
        raise ValueError('Recipe contains unsupported formation options')
    options['window'] = _window_name(options.get('window', 'Hanning'))
    options['length_unit'] = _length_unit(options.get('length_unit', 'm'))[0]
    options['l1_strength'], options['l1_iterations'] = _sparse_parameters(
        options.get('l1_strength', .05), options.get('l1_iterations', 300))
    if options.get('reconstruction', 'fast') not in {'fft', 'fast', 'accurate', 'sparse', 'auto'}:
        raise ValueError('Unknown recipe reconstruction')
    aperture_mode(options.get('aperture_mode', 'auto'))
    scene_extents(options.get('scene_half_extent_m'))
    side = options.get('composite_side', 1024)
    if isinstance(side, bool) or not isinstance(side, int) or not 32 <= side <= 4096:
        raise ValueError('Invalid recipe composite grid size')
    for key in ('flip_x', 'flip_y', 'native_diagnostics'):
        if key in options and not isinstance(options[key], bool):
            raise ValueError(f'Recipe {key} must be Boolean')
    center = options.get('aperture_center_degrees')
    if center is not None and not np.isfinite(float(center)):
        raise ValueError('Recipe aperture center must be finite')
    target = options.get('azimuth_target_degrees')
    if target is not None:
        target = np.asarray(target, dtype=float)
        if target.ndim != 1 or target.size < 2 or not np.all(np.isfinite(target)) or np.any(np.diff(target) <= 0):
            raise ValueError('Recipe interpolation grid must be finite and increasing')
    polarizations = list(map(str, dataset.polarizations))
    if recipe.get('polarization') not in polarizations:
        raise ValueError('Active dataset lacks the recipe polarization')
    return {**options,
        'azimuth_indices': _matching_indices(_angle_values_to_degrees(dataset, 'azimuth', dataset.azimuths), recipe['azimuth_degrees'], atol=1e-8),
        'frequency_indices': _matching_indices(np.asarray(dataset.frequencies) * _unit_to_hz_scale(dataset.units.get('frequency', '')), recipe['frequency_hz'], atol=1e-3),
        'elevation_index': _matching_indices(_angle_values_to_degrees(dataset, 'elevation', dataset.elevations), [recipe['elevation_degrees']], atol=1e-8)[0],
        'polarization_index': polarizations.index(recipe['polarization'])}
