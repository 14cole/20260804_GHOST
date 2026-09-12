"""Comparison of compatible numerical ISAR artifacts on a physical grid."""
from __future__ import annotations

import numpy as np


def hydrate_band(band, manifest, index=0):
    """Attach saved diagnostics to numerical arrays without copying image buffers."""
    info = manifest['bands'][index]
    out = dict(info)
    out.update(band)
    out['az_values'] = np.asarray(info.get('azimuth_values_degrees', []), dtype=float)
    out['composite'] = int(info.get('composite_looks', 0))
    out.update(info.get('sparse', {}))
    out['isar_contract_undeclared_fields'] = manifest.get('formation', {}).get('isar_contract_undeclared_fields', [])
    return out


def comparison_issues(first_manifest, first_band, second_manifest, second_band):
    """Require compatible physics before interpreting a level difference."""
    a, b = first_band.get('image_contract', {}), second_band.get('image_contract', {})
    issues = []
    if not a or not b:
        return ['An artifact lacks an image/frame contract. Re-form it with the current engine before quantitative comparison.']
    for key in ('image_plane', 'phase_convention', 'amplitude_normalization', 'image_kind', 'phase_center'):
        if not a.get(key) or not b.get(key) or a.get(key) != b.get(key):
            issues.append(f'Different image {key.replace("_", " ")}')
    for key in ('cross_range_basis', 'range_basis'):
        av, bv = np.asarray(a.get(key, [])), np.asarray(b.get(key, []))
        if (av.shape != (3,) or bv.shape != (3,) or not np.allclose(av, bv, atol=1e-10, rtol=0)
                or not np.isclose(np.linalg.norm(av), 1., atol=1e-10, rtol=0)):
            issues.append('Different image coordinate frames')
    fa, fb = first_manifest.get('formation', {}), second_manifest.get('formation', {})
    for key in ('window', 'reconstruction'):
        if fa.get(key) != fb.get(key):
            issues.append(f'Different {key}; use the same image recipe before quantitative comparison')
    if first_band.get('resolved_reconstruction') != second_band.get('resolved_reconstruction'):
        issues.append('Different resolved reconstruction methods')
    sa, sb = first_manifest.get('source', {}), second_manifest.get('source', {})
    for key in ('selected_polarization', 'selected_elevation_degrees'):
        if sa.get(key) != sb.get(key):
            issues.append(f'Different {key.replace("selected_", "").replace("_", " ")}')
    frequencies_a = np.asarray(sa.get('selected_frequency_values_hz', []))
    frequencies_b = np.asarray(sb.get('selected_frequency_values_hz', []))
    if not len(frequencies_a) or frequencies_a.shape != frequencies_b.shape or not np.allclose(frequencies_a, frequencies_b, rtol=1e-12, atol=1e-3):
        issues.append('Different acquired frequency samples')
    from GRIM_Backend.plotting.modes.common import convert_axis_values
    acquired_angles = [convert_axis_values(source.get('selected_azimuth_values_native', []),
        'azimuth', source.get('units', {}).get('azimuth', 'deg'), 'deg') for source in (sa, sb)]
    if (not len(acquired_angles[0]) or acquired_angles[0].shape != acquired_angles[1].shape
            or not np.allclose(acquired_angles[0], acquired_angles[1], rtol=0, atol=1e-8)):
        issues.append('Different acquired angular samples; use the same acquisition selection')
    aza, azb = np.asarray(first_band.get('az_values', [])), np.asarray(second_band.get('az_values', []))
    if not aza.size or aza.shape != azb.shape or not np.allclose(aza, azb, rtol=0, atol=1e-8):
        issues.append('Different acquired angular apertures')
    for key in ('phase_reference', 'polarization_basis', 'amplitude_convention',
                'complex_field_domain', 'phase_center_xyz_m'):
        def value(source):
            return source.get('metadata', {}).get(key, source.get('units', {}).get(key))
        if value(sa) != value(sb):
            issues.append(f'Different declared {key.replace("_", " ")}')
    for key in ('rcs_linear_quantity', 'rcs_log_unit'):
        if sa.get('units', {}).get(key) != sb.get('units', {}).get(key):
            issues.append(f'Different source {key.replace("_", " ")}')
    return list(dict.fromkeys(issues))


def _ordered(band):
    contract = band['image_contract']
    scale = float(contract['distance_scale_per_metre'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid artifact length scale')
    x, y = np.asarray(band['x_range']) / scale, np.asarray(band['y_range']) / scale
    mag = np.asarray(band['magnitude'])
    # A live result can retain the full complex array and a reduced display
    # magnitude. Keep the complex view and take abs only on bounded samples.
    if mag.shape != (len(x), len(y)):
        mag = np.asarray(band.get('complex_image'))
    if mag.shape != (len(x), len(y)) or min(len(x), len(y)) < 2:
        raise ValueError('Image dimensions do not match its physical axes')
    if x[0] > x[-1]:
        x, mag = x[::-1], mag[::-1]
    if y[0] > y[-1]:
        y, mag = y[::-1], mag[:, ::-1]
    return x, y, mag


def compare_images(first_manifest, first_band, second_manifest, second_band, *, allow_resample=False,
                   maximum_cells=4_000_000):
    """Return A−B intensity level in dB. No calibrated per-pixel RCS claim."""
    issues = comparison_issues(first_manifest, first_band, second_manifest, second_band)
    if issues:
        raise ValueError('; '.join(issues))
    xa, ya, ma = _ordered(first_band)
    xb, yb, mb = _ordered(second_band)
    xids = np.flatnonzero((xa >= xb[0] - 1e-10) & (xa <= xb[-1] + 1e-10))
    yids = np.flatnonzero((ya >= yb[0] - 1e-10) & (ya <= yb[-1] + 1e-10))
    if min(len(xids), len(yids)) < 2:
        raise ValueError('Images have fewer than two overlapping centers on an axis')
    if len(xids) * len(yids) > maximum_cells:
        raise ValueError('Comparison exceeds its cell budget; form a smaller scene first')
    x, y = xa[xids], ya[yids]
    same = xa.shape == xb.shape and ya.shape == yb.shape and np.allclose(xa, xb, rtol=0, atol=1e-10) and np.allclose(ya, yb, rtol=0, atol=1e-10)
    if not same and not allow_resample:
        raise ValueError('Image grids differ. Enable physical-grid resampling to compare their overlap.')
    a = np.asarray(abs(ma[np.ix_(xids, yids)]), dtype=np.float32)**2
    if same:
        b = np.asarray(abs(mb[np.ix_(xids, yids)]), dtype=np.float32)**2
    else:
        # Bilinear interpolation of linear intensity; never interpolate dB.
        # Allocate only the bounded overlap and block scratch, not the full B image.
        b = np.empty((len(x), len(y)), dtype=np.float32)
        jx = np.clip(np.searchsorted(xb, x), 1, len(xb) - 1)
        jy = np.clip(np.searchsorted(yb, y), 1, len(yb) - 1)
        tx = ((x - xb[jx - 1]) / (xb[jx] - xb[jx - 1])).astype(np.float32)
        ty = ((y - yb[jy - 1]) / (yb[jy] - yb[jy - 1])).astype(np.float32)
        np.clip(tx, 0, 1, out=tx)
        np.clip(ty, 0, 1, out=ty)
        for start in range(0, len(x), 64):
            stop = min(start + 64, len(x))
            i = jx[start:stop, None]
            j = jy[None, :]
            wx, wy = tx[start:stop, None], ty[None, :]
            b[start:stop] = ((abs(mb[i-1, j-1])**2 * (1-wx) + abs(mb[i, j-1])**2 * wx) * (1-wy)
                + (abs(mb[i-1, j])**2 * (1-wx) + abs(mb[i, j])**2 * wx) * wy)
    # Keep the same display floor used by both formation front ends.
    ad = 10 * np.log10(np.maximum(a, np.float32(1e-12)))
    bd = 10 * np.log10(np.maximum(b, np.float32(1e-12)))
    delta = ad - bd
    return {'x_range': x, 'y_range': y, 'a_db': ad, 'b_db': bd, 'delta_db': delta,
            'resampled': not same, 'resampling': 'linear intensity on A overlap grid' if not same else 'none',
            'units': 'image intensity dB', 'floor_db': -120.,
            'statistics': roi_statistics(delta),
            'quality_note': 'Interpret differences with each image coverage and PSF. Calibration provenance remains in the source manifests.'}


def roi_statistics(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        raise ValueError('ROI contains no finite image samples')
    data = finite.astype(float)
    return {'pixels': int(data.size), 'mean_delta_db': float(np.mean(data)),
            'rms_delta_db': float(np.sqrt(np.mean(data**2))),
            'minimum_delta_db': float(data.min()), 'maximum_delta_db': float(data.max())}
