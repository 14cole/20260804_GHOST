"""Bounded polynomial mesh candidates on the user's original primitives."""
import math
import numpy as np

MIN_AUTOMATIC_REFERENCE_PANELS = 1024


def point_key(point):
    return tuple(int(round(float(v) / 1e-9)) for v in point)


def primitive_key(segment, a, b):
    ends = sorted((point_key(a), point_key(b)))
    return '{}:{},{}:{},{}'.format(segment, *ends[0], *ends[1])


def protected_vertices(snapshot, scale):
    """Protect geometric corners, open ends and junctions, including imports."""
    incident = {}
    for segment in snapshot.get('segments', []):
        for pair in segment.get('point_pairs', []):
            a = np.array([float(pair['x1']), float(pair['y1'])]) * scale
            b = np.array([float(pair['x2']), float(pair['y2'])]) * scale
            direction = b-a
            length = np.linalg.norm(direction)
            if length <= 0: continue
            incident.setdefault(point_key(a), []).append(direction/length)
            incident.setdefault(point_key(b), []).append(-direction/length)
    cosine = -math.cos(math.radians(15.))
    return {key for key, directions in incident.items()
            if len(directions) != 2 or np.dot(*directions) > cosine}


def panel_parameters(snapshot, segment, a, b, base_count, fixed_count, refinement, protected):
    """Coarsen wavelength-based P1 reference counts, preserving explicit counts."""
    key = primitive_key(segment, a, b)
    coarsening = float(snapshot.get('_2d_hp_coarsening', 1.))
    multiplier = float(snapshot.get('_2d_hp_refinements', {}).get(key, 1.))
    if not math.isfinite(coarsening) or not 1 <= coarsening <= 8:
        raise ValueError('Invalid adaptive mesh coarsening factor.')
    if not math.isfinite(multiplier) or not 1 <= multiplier <= 64:
        raise ValueError('Invalid adaptive primitive refinement factor.')
    count = max(1, int(math.ceil(base_count * multiplier / (1 if fixed_count else coarsening))))
    if refinement > 1:
        count = max(count + 1, int(math.ceil(count * refinement)))
    t = np.linspace(0., 1., count + 1)
    left, right = point_key(a) in protected, point_key(b) in protected
    if left and right: t = .5 * (1 - np.cos(np.pi*t))
    elif left: t = t**1.7
    elif right: t = 1 - (1-t)**1.7
    return key, a[None, :] + t[:, None] * (b-a)[None, :]


def eligible_snapshot(snapshot, materials, frequencies=None, scale=1., mesh_reference=None):
    from ghost_backend.twod.formulations.thin_layer import ThinLayerDefinition
    from ghost_backend.twod.geometry import ImpedanceTaper
    if any(isinstance(value, ThinLayerDefinition) for value in materials.impedance_models.values()):
        return False, 'Thin-layer asymptotic formulation retains its qualified linear discretization.'
    if any(isinstance(value, ImpedanceTaper) and value.kind != 'constant'
           for value in materials.impedance_models.values()):
        return False, 'Spatially varying impedance retains h refinement to verify material sampling.'
    properties = [list(segment.get('properties', [])) for segment in snapshot.get('segments', [])]
    if all(len(row) > 1 and float(row[1] or 0) > 0 for row in properties):
        return False, 'Explicit fixed panel counts retain their requested discretization.'
    if frequencies is not None:
        # Count reference panels without allocating them. Small measured cases
        # cannot repay polynomial near-quadrature overhead through smaller LU.
        # The same threshold is used in initial planning and actual execution.
        from ghost_backend.twod.geometry import (_panel_count_from_n, _parse_int,
            _mesh_wavelength_for_snapshot, _conservative_mesh_wavelength_for_frequencies)
        counts = []
        for frequency in ([mesh_reference] if mesh_reference else frequencies):
            wavelength = (_conservative_mesh_wavelength_for_frequencies(snapshot, materials,
                          set(frequencies) | {mesh_reference})[0] if mesh_reference else
                          _mesh_wavelength_for_snapshot(snapshot, materials, frequency)[0])
            count = 0
            for segment in snapshot.get('segments', []):
                n = _parse_int((list(segment.get('properties', []))+[0,0])[1], 0)
                for pair in segment.get('point_pairs', []):
                    length = math.hypot(float(pair['x2'])-float(pair['x1']),
                                        float(pair['y2'])-float(pair['y1'])) * scale
                    count += _panel_count_from_n(n, length, wavelength)
            counts.append(count)
        if max(counts, default=0) < MIN_AUTOMATIC_REFERENCE_PANELS:
            return False, 'Small reference mesh retains linear basis to avoid polynomial quadrature overhead.'
    return True, ''


def candidate_meshes(snapshot, materials, factor, adaptive, frequencies=None, scale=1., mesh_reference=None):
    """The initial solve/check meshes shared by desktop and HPC forecasting.

    Later error-driven candidates are admitted again on the executing node;
    the worker cannot expand its scheduler memory reservation.
    """
    import copy
    from ghost_backend.runs.quality import scale_snapshot_panel_density
    if factor > 1 and adaptive and eligible_snapshot(snapshot, materials, frequencies, scale, mesh_reference)[0]:
        candidate = copy.deepcopy(snapshot)
        candidate['_2d_hp_coarsening'] = 4.
        candidate['_2d_hp_refinements'] = {}
        return [('base', candidate, 2), ('fine', candidate, 3)]
    records = [('base', snapshot, 1)]
    if factor > 1:
        fine = scale_snapshot_panel_density(snapshot, factor)
        fine['_2d_certification_refinement_factor'] = factor
        fine['_2d_certification_base_segment_n'] = [
            (list(segment.get('properties', [])) + [0, 0])[1] for segment in snapshot['segments']]
        records.append(('fine', fine, 1))
    return records
