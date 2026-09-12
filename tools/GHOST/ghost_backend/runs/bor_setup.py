"""Portable BOR physics settings, with explicit aspect-angle semantics."""
import math
from ghost_backend.bor.options import validate_options


def validate_bor_setup(value):
    fields = {'schema', 'version', 'frequencies_ghz', 'aspects_deg', 'units',
              'mesh_certification', 'accuracy', 'cfie_alpha', 'bor_options'}
    if (not isinstance(value, dict) or set(value) - {'radar_grid'} != fields or
            value.get('schema') != 'grim.bor-run-setup' or
            type(value.get('version')) is not int or value['version'] != 1):
        raise ValueError('Choose a supported GRIM BOR run setup (version 1).')
    result = dict(value)
    for key in ('frequencies_ghz', 'aspects_deg'):
        samples = value[key]
        if not isinstance(samples, list) or not 0 < len(samples) <= 100000:
            raise ValueError('{}: supply 1 to 100,000 samples.'.format(key))
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in samples):
            raise ValueError('{}: every sample must be finite and numeric.'.format(key))
        if len(samples) != len(set(samples)):
            raise ValueError('{}: duplicate samples are not supported.'.format(key))
        if key == 'frequencies_ghz' and any(x <= 0 for x in samples):
            raise ValueError('Frequencies must be positive GHz values.')
        if key == 'aspects_deg' and any(x < 0 or x > 180 for x in samples):
            raise ValueError('BOR aspects must lie in [0, 180] degrees from +z.')
        result[key] = list(samples)
    if value['units'] not in ('inches', 'meters'):
        raise ValueError('BOR geometry units must be inches or meters.')
    if type(value['mesh_certification']) is not bool:
        raise ValueError('mesh_certification must be true or false.')
    if value['accuracy'] not in ('standard', 'tight'):
        raise ValueError('BOR accuracy must be standard or tight.')
    alpha = value['cfie_alpha']
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('BOR CFIE alpha must lie strictly between 0 and 1.')
    result['bor_options'] = validate_options(value['bor_options'])
    grid = value.get('radar_grid')
    if grid is not None:
        import numpy as np
        from ghost_backend.assembly.fields import radar_grid_aspects, validate_radar_grid
        keys = {'azimuths_deg', 'elevations_deg', 'axis_az_deg', 'axis_el_deg', 'roll_deg'}
        if not isinstance(grid, dict) or set(grid) != keys:
            raise ValueError('Invalid BOR radar-grid fields.')
        for key in ('azimuths_deg', 'elevations_deg'):
            if (not isinstance(grid[key], list) or not 0 < len(grid[key]) <= 100000 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in grid[key])):
                raise ValueError('BOR radar axes must be finite numeric lists.')
        for key in ('axis_az_deg', 'axis_el_deg', 'roll_deg'):
            if type(grid[key]) not in (int, float) or not math.isfinite(grid[key]):
                raise ValueError('BOR body attitude must be finite and numeric.')
        validate_radar_grid(grid['azimuths_deg'], grid['elevations_deg'])
        if len(grid['azimuths_deg'])*len(grid['elevations_deg']) > 100000:
            raise ValueError('Saved BOR radar grids support at most 100,000 looks.')
        aspects = radar_grid_aspects(grid['azimuths_deg'], grid['elevations_deg'], grid['axis_az_deg'], grid['axis_el_deg'])
        if len(aspects) != len(result['aspects_deg']) or not np.allclose(aspects, result['aspects_deg'], rtol=0., atol=1e-10):
            raise ValueError('BOR body aspects do not match the saved radar grid.')
        grid = dict(grid, azimuths_deg=list(grid['azimuths_deg']), elevations_deg=list(grid['elevations_deg']))
    result['radar_grid'] = grid
    return result


def driver_settings(value):
    value = validate_bor_setup(value)
    grid = value['radar_grid'] or dict(azimuths_deg=[0.],
        elevations_deg=[90.-x for x in value['aspects_deg']], axis_az_deg=0., axis_el_deg=90., roll_deg=0.)
    return dict(FREQUENCIES_GHZ=value['frequencies_ghz'], AZIMUTHS_DEG=grid['azimuths_deg'],
                ELEVATIONS_DEG=grid['elevations_deg'], BODY_AXIS_AZ_DEG=grid['axis_az_deg'],
                BODY_AXIS_EL_DEG=grid['axis_el_deg'], BODY_ROLL_DEG=grid['roll_deg'],
                GEOMETRY_UNITS=value['units'], MESH_CERTIFICATION=value['mesh_certification'],
                ACCURACY_TARGET=value['accuracy'], CFIE_ALPHA=value['cfie_alpha'],
                BOR_EXECUTION_OPTIONS=value['bor_options'])


def resource_summary(snapshot, base_dir, value, checkpoint=None):
    import os
    from ghost_backend.bor.dispatch import estimate_bor_resources
    from ghost_backend.runs.quality import accuracy_target_policy
    from ghost_backend.runs.setup import geometry_dimensions
    value = validate_bor_setup(value)
    policy = accuracy_target_policy(value['accuracy'])
    peak, elements = 0., 0
    for frequency in value['frequencies_ghz']:
        if checkpoint is not None:
            checkpoint()
        estimate = estimate_bor_resources(snapshot, frequency, value['aspects_deg'],
            geometry_units=value['units'], material_base_dir=base_dir,
            workers=max(1, (os.cpu_count() or 2)-1),
            mesh_certification=value['mesh_certification'], fine_factor=policy['fine_factor'],
            bor_options=value['bor_options'])
        peak = max(peak, estimate['estimated_peak_gb'])
        elements = max(elements, estimate['mesh_elements'])
    options = value['bor_options']
    return ('BOR geometry and material checks passed. ' + geometry_dimensions(snapshot, value['units']) + '\n'
        + '{} frequencies; {} aspects from +z; VV + HH. '.format(len(value['frequencies_ghz']), len(value['aspects_deg']))
        + ('Base/fine mesh comparison' if value['mesh_certification'] else 'Survey; no mesh certificate')
        + '; {} accuracy; CFIE alpha {:g}.\n'.format(value['accuracy'], value['cfie_alpha'])
        + '{} factorization; {} aspects per batch; {} incident basis reuse.\n'.format(
            options['factorization'], options['angle_batch_size'], options['rhs_compression'])
        + 'Up to {} mesh elements; estimated peak {:.2f} GB with desktop mode workers. '.format(elements, peak)
        + 'This is an allocation forecast, not measured RAM. Numerical quality and convergence are checked during the solve.')
