"""Power and phase comparison for duplicate imported samples."""
from __future__ import annotations

import numpy as np


def _cst_samples_equivalent(left_power, left_phase, right_power, right_phase):
    """Return True when two seam/duplicate rows encode the same field."""

    if not np.isclose(
        float(left_power), float(right_power), rtol=1.0e-8, atol=1.0e-12
    ):
        return False


    if float(left_power) == 0.0 and float(right_power) == 0.0:
        return True
    left_phase = float(left_phase)
    right_phase = float(right_phase)
    if np.isnan(left_phase) and np.isnan(right_phase):
        return True
    if not (np.isfinite(left_phase) and np.isfinite(right_phase)):
        return False
    phase_error = np.angle(np.exp(1j * (left_phase - right_phase)))
    return abs(float(phase_error)) <= 1.0e-8
