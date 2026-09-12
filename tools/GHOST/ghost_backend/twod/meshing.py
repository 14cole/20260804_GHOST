"""Optional local material sizing with conservative feature protection.

The global policy remains the default. The local policy is a candidate mesh,
not an error estimator; production acceptance still uses complex-field mesh
convergence, with a global-mesh retry if the candidate fails certification.
"""
import math
from ghost_backend.execution.options import option
from ghost_backend.geometry.guidance import geometry_refinement_candidates


def segment_wavelengths(snapshot, materials, frequencies, scale, global_wavelength):
    if option('mesh_strategy', 'global') != 'local':
        return None
    from ghost_backend.twod.geometry import _mesh_wavelength_for_snapshot
    from ghost_backend.twod.constants import C0
    segments = snapshot.get('segments', [])
    frequencies = [float(f) for f in frequencies]
    wavelengths, boxes = [], []
    for segment in segments:
        local = dict(snapshot, segments=[segment])
        wavelength = min(_mesh_wavelength_for_snapshot(local, materials, f)[0] for f in frequencies)
        # Bound the departure from the existing mesh even on distant surfaces.
        wavelengths.append(min(wavelength, 4 * global_wavelength))
        points = [(float(p[x]) * scale, float(p[y]) * scale)
                  for p in segment.get('point_pairs', []) for x, y in (('x1', 'y1'), ('x2', 'y2'))]
        boxes.append((min(p[0] for p in points), max(p[0] for p in points),
                      min(p[1] for p in points), max(p[1] for p in points)))
    # Corners, open ends, junctions and nearby boundaries retain global sizing.
    protected = geometry_refinement_candidates(segments, point_tolerance=1e-9/scale)
    influence = C0 / (max(frequencies) * 1e9)
    from ghost_backend.geometry.spatial import overlapping_pairs
    expanded = [(a-influence/2, b+influence/2, c-influence/2, d+influence/2)
                for a,b,c,d in boxes]
    close = set(protected)
    for i,j in overlapping_pairs(expanded):
        a,b = boxes[i], boxes[j]
        dx = max(a[0]-b[1], b[0]-a[1], 0.)
        dy = max(a[2]-b[3], b[2]-a[3], 0.)
        if math.hypot(dx,dy) <= influence:
            close.update((i,j))
    for i in close:
        wavelengths[i] = global_wavelength
    return wavelengths
