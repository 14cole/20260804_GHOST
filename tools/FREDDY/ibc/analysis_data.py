"""Run-owned plot data and metrics, independent of editable GUI controls."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from .compute import ETA0, build_uncertainty_scales, is_nominal_scale
from .inverse_performance import band_performance


@dataclass
class SweepResult:
    frequencies: list[float]
    values: list[float]
    axis_label: str
    context: str
    # Each grid is frequency x selected thickness/angle. Polarizations are explicit.
    metrics: dict
    polarization: str = 'TE'
    backing: str = 'pec'
    lower: dict = field(default_factory=dict)
    upper: dict = field(default_factory=dict)
    impedance: object = None
    impedance_bounds: dict = field(default_factory=dict)
    files: list = field(default_factory=list)
    file_hashes: list = field(default_factory=list)
    summary: str = ''
    layers: list = field(default_factory=list)


def grid_metrics(out):
    import numpy as np
    return {key: np.asarray(value, dtype=float).copy() for key, value in out.items()
            if key not in ('freq_ghz', 'angle_deg', 'thickness_in')}


def reflection_from_impedance(impedance):
    import numpy as np
    z = np.asarray(impedance, dtype=complex)
    if not np.isfinite(z).all():
        raise ValueError('Impedance contains a nonfinite value.')
    return 20 * np.log10(np.maximum(np.abs((z - ETA0) / (z + ETA0)), 1e-15))


def impedance_result(frequencies, capture, metrics, backing, context):
    import numpy as np
    nominal = np.asarray(capture['nominal'], dtype=complex)
    cases = np.asarray(capture['cases'], dtype=complex)
    reflection = reflection_from_impedance(cases)
    key = 'metal_loss_db' if backing == 'pec' else 'air_loss_db'
    lower = {key: reflection.min(axis=0)[:, None]}
    upper = {key: reflection.max(axis=0)[:, None]}
    return SweepResult(list(frequencies), [0.], 'Normal incidence', context,
                       {'TE': {k: np.asarray(v)[:, None] for k, v in metrics.items()}},
                       backing=backing, lower={'TE': lower}, upper={'TE': upper},
                       impedance=nominal[:, None], impedance_bounds={
                           'real_min': cases.real.min(axis=0), 'real_max': cases.real.max(axis=0),
                           'imag_min': cases.imag.min(axis=0), 'imag_max': cases.imag.max(axis=0)})


def extra_polarization(compute_grid, uncertainty):
    """Compute a separately labeled polarization and its systematic envelopes."""
    import numpy as np
    nominal = grid_metrics(compute_grid(1., 1., 1.))
    lower, upper = {}, {}
    if uncertainty.enabled:
        lower = {k: v.copy() for k, v in nominal.items()}
        upper = {k: v.copy() for k, v in nominal.items()}
        for scales in build_uncertainty_scales(uncertainty):
            if is_nominal_scale(*scales):
                continue
            changed = grid_metrics(compute_grid(*scales))
            for key, values in changed.items():
                if 'phase' in key:
                    values = nominal[key] + (values - nominal[key] + 180) % 360 - 180
                lower[key] = np.minimum(lower[key], values)
                upper[key] = np.maximum(upper[key], values)
    return nominal, lower, upper


def band_metrics(frequencies, grid, target, low=None, high=None):
    """Use the exact selected sample region. No silent out-of-range clipping."""
    import numpy as np
    f = np.asarray(frequencies, dtype=float)
    data = np.asarray(grid, dtype=float)
    low = f[0] if low is None else float(low)
    high = f[-1] if high is None else float(high)
    if low > high or low < f[0] or high > f[-1]:
        raise ValueError('Choose a band within the run’s frequency range; start must not exceed stop.')
    # Insert boundary values by interpolation; never extrapolate or bridge a
    # missing frequency without making the interpolation assumption explicit.
    selected = sorted(set([low, high] + [float(v) for v in f if low < v < high]))
    curves = [np.interp(selected, f, data[:, j]).tolist() for j in range(data.shape[1])]
    return [band_performance(selected, curve, target, continuous=True) for curve in curves]


def passing_sample_ranges(values, performance, threshold):
    """Group consecutive passing samples; do not infer between-sample validity."""
    ranges = []
    start = None
    for index, metric in enumerate(performance):
        if metric.worst_db <= threshold:
            if start is None:
                start = index
        elif start is not None:
            ranges.append((values[start], values[index - 1]))
            start = None
    if start is not None:
        ranges.append((values[start], values[len(performance) - 1]))
    return ranges


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def expected_ibc_digest(frequencies, impedance):
    """Hash the exact writer output before publication, without a second copy."""
    import os
    from .io import _write_impedance_rows
    digest = hashlib.sha256()
    class HashStream:
        def write(self, text):
            # The export writer opens a native text stream (newline=None).
            digest.update(text.replace('\n', os.linesep).encode('utf-8'))
    rows = ((f, z.real, z.imag) for f, z in zip(frequencies, impedance))
    _write_impedance_rows(HashStream(), rows)
    return digest.hexdigest()


def verified_batch_file(result, index):
    if result.backing != 'pec' or not 0 <= index < len(result.files):
        raise ValueError('Select an exported PEC-backed IBC file.')
    path = Path(result.files[index]).resolve()
    if file_digest(path) != result.file_hashes[index]:
        raise ValueError('This exported file changed after the run. Re-export the batch before using it for GHOST.')
    return path


def inverse_case_grid(samples, angles, scales):
    """Recover explicit axes only from the labels captured alongside the run."""
    import numpy as np
    values = np.asarray(samples, dtype=float)
    if not angles or not scales or values.ndim != 2 or values.shape[1] != len(angles) * len(scales):
        raise ValueError('Labeled angle/tolerance data are unavailable. Run the search again.')
    if not np.isfinite(values).all():
        raise ValueError('Candidate response samples are incomplete.')
    return values.reshape(values.shape[0], len(scales), len(angles))
