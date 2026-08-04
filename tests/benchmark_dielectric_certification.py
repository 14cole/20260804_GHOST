#!/usr/bin/env python3
"""Benchmark the certified dense path on the coated-rectangle fixture.

The optional reference module should be the pre-change ``rcs_solver.py``. The
benchmark uses the survey entry point intentionally: it exercises the same
mandatory condition telemetry and multi-angle factorization as each half of a
production base/fine certification pair, without conflating the comparison
with mesh-refinement error.

Usage:
    python tests/benchmark_dielectric_certification.py [reference_rcs_solver.py]
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
GEOMETRY = REPO / "tests" / "fixtures" / "coated_rectangle_24x6_eps50.geo"
sys.path.insert(0, str(BACKEND))


def load_reference(path):
    spec = importlib.util.spec_from_file_location("rcs_solver_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rcs_solver_reference"] = module
    spec.loader.exec_module(module)
    return module


def snapshot():
    from geometry_io import build_geometry_snapshot, parse_geometry

    title, segments, ibcs, dielectrics = parse_geometry(GEOMETRY.read_text())
    value = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    value["source_path"] = str(GEOMETRY)
    return value


def solve(module):
    started = time.perf_counter()
    result = module.solve_monostatic_rcs_2d_survey(
        geometry_snapshot=snapshot(),
        frequencies_ghz=[0.25],
        elevations_deg=list(np.linspace(0.0, 180.0, 181)),
        polarization="TM",
        geometry_units="inches",
        material_base_dir=str(GEOMETRY.parent),
        max_panels=50_000,
    )
    return result, time.perf_counter() - started


def amplitudes(result):
    return np.asarray([
        complex(sample["rcs_amp_real"], sample["rcs_amp_imag"])
        for sample in result["samples"]
    ])


def main():
    import rcs_solver as current

    reference_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    reference = (
        load_reference(reference_path)
        if reference_path is not None and reference_path.is_file()
        else None
    )

    current_result, current_seconds = solve(current)
    metadata = current_result["metadata"]
    print(
        f"current: {current_seconds:.3f} s, panels={metadata['panel_count']}, "
        f"condition={metadata['condition_est_max']:.6g}, "
        f"method={metadata['condition_est_method']}, "
        f"residual={metadata['residual_norm_max']:.3e}"
    )
    if reference is None:
        return 0

    reference_result, reference_seconds = solve(reference)
    current_amp = amplitudes(current_result)
    reference_amp = amplitudes(reference_result)
    scale = float(np.max(np.abs(reference_amp))) or 1.0
    amplitude_error = float(np.max(np.abs(current_amp - reference_amp))) / scale
    print(
        f"reference: {reference_seconds:.3f} s; "
        f"speedup={reference_seconds / max(current_seconds, 1e-12):.2f}x; "
        f"relative amplitude error={amplitude_error:.3e}"
    )
    return 1 if amplitude_error > 1.0e-9 else 0


if __name__ == "__main__":
    sys.exit(main())
