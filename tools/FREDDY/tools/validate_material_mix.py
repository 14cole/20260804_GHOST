#!/usr/bin/env python3
"""Validate FREDDY material mixing against the NIST BaM/PDMS dataset."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable


FREDDY_ROOT = Path(__file__).resolve().parents[1]
if str(FREDDY_ROOT) not in sys.path:
    sys.path.insert(0, str(FREDDY_ROOT))

from ibc.compute import (  # noqa: E402
    MIX_RULE_MG,
    MixComponent,
    combine_mix,
    interp_components_on_grid,
    property_match_error,
    property_match_error_curve,
)
from ibc.io import read_material_table  # noqa: E402


VOLUME_FRACTION_30WT = 0.0726
VOLUME_FRACTION_60WT = 0.215
FIT_RMS_LIMIT_PERCENT = 0.1
INVERSE_FRACTION_TOLERANCE = 2e-4
MEASURED_MEDIAN_LIMIT_PERCENT = 2.5
MEASURED_P95_LIMIT_PERCENT = 5.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("A percentile requires at least one value.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _golden_section_minimize(
    objective: Callable[[float], float], lower: float, upper: float, iterations: int = 80
) -> tuple[float, float]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    f_left = objective(left)
    f_right = objective(right)
    for _ in range(iterations):
        if f_left <= f_right:
            upper, right, f_right = right, left, f_left
            left = upper - ratio * (upper - lower)
            f_left = objective(left)
        else:
            lower, left, f_left = left, right, f_right
            right = lower + ratio * (upper - lower)
            f_right = objective(right)
    volume = (lower + upper) / 2.0
    return volume, objective(volume)


def _prepared_components(pack_dir: Path, grid_ghz: list[float]):
    host = read_material_table(pack_dir / "pdms_fit.csv")
    particle = read_material_table(pack_dir / "bam_particle_fit.csv")
    eps_cols, mu_cols = interp_components_on_grid(
        [MixComponent(host, 1.0), MixComponent(particle, 1.0)], grid_ghz
    )
    return eps_cols, mu_cols


def forward_error(
    pack_dir: Path, target_filename: str, inclusion_volume_fraction: float
) -> float:
    target = read_material_table(pack_dir / target_filename)
    eps_cols, mu_cols = _prepared_components(pack_dir, target.freq_ghz)
    mixed = combine_mix(
        target.freq_ghz,
        eps_cols,
        mu_cols,
        [1.0 - inclusion_volume_fraction, inclusion_volume_fraction],
        MIX_RULE_MG,
    )
    return property_match_error(
        mixed.eps_r, mixed.mu_r, target.eps_r, target.mu_r
    )


def recover_inclusion_fraction(
    pack_dir: Path, target_filename: str
) -> tuple[float, float]:
    target = read_material_table(pack_dir / target_filename)
    eps_cols, mu_cols = _prepared_components(pack_dir, target.freq_ghz)

    def objective(volume: float) -> float:
        mixed = combine_mix(
            target.freq_ghz,
            eps_cols,
            mu_cols,
            [1.0 - volume, volume],
            MIX_RULE_MG,
        )
        return property_match_error(
            mixed.eps_r, mixed.mu_r, target.eps_r, target.mu_r
        )

    # A coarse global pass avoids assuming strict convexity, then a local
    # golden-section search gives a deterministic, SciPy-independent result.
    grid = [0.001 + index * (0.349 / 349.0) for index in range(350)]
    scores = [objective(volume) for volume in grid]
    best_index = min(range(len(grid)), key=scores.__getitem__)
    lower = grid[max(0, best_index - 1)]
    upper = grid[min(len(grid) - 1, best_index + 1)]
    return _golden_section_minimize(objective, lower, upper)


def measured_error_summary(
    pack_dir: Path, target_filename: str, inclusion_volume_fraction: float
) -> dict[str, float | int]:
    target = read_material_table(pack_dir / target_filename)
    eps_cols, mu_cols = _prepared_components(pack_dir, target.freq_ghz)
    mixed = combine_mix(
        target.freq_ghz,
        eps_cols,
        mu_cols,
        [1.0 - inclusion_volume_fraction, inclusion_volume_fraction],
        MIX_RULE_MG,
    )
    curve = property_match_error_curve(
        mixed.eps_r, mixed.mu_r, target.eps_r, target.mu_r
    )
    return {
        "rows": len(curve),
        "median_percent": statistics.median(curve),
        "p95_percent": _percentile(curve, 95.0),
        "rms_percent": math.sqrt(sum(value * value for value in curve) / len(curve)),
    }


def validate(pack_dir: Path) -> dict[str, object]:
    fit_specs = {
        "30wt": ("bam_pdms_30wt_fit.csv", VOLUME_FRACTION_30WT),
        "60wt": ("bam_pdms_60wt_fit.csv", VOLUME_FRACTION_60WT),
    }
    measured_specs = {
        "30wt_sample_1": (
            "bam_pdms_30wt_sample_1_passive.csv",
            VOLUME_FRACTION_30WT,
        ),
        "30wt_sample_2": (
            "bam_pdms_30wt_sample_2_passive.csv",
            VOLUME_FRACTION_30WT,
        ),
        "30wt_sample_3": (
            "bam_pdms_30wt_sample_3_passive.csv",
            VOLUME_FRACTION_30WT,
        ),
        "60wt_measured": (
            "bam_pdms_60wt_measured_passive.csv",
            VOLUME_FRACTION_60WT,
        ),
    }

    forward: dict[str, object] = {}
    inverse: dict[str, object] = {}
    failures: list[str] = []
    for label, (filename, expected_volume) in fit_specs.items():
        rms = forward_error(pack_dir, filename, expected_volume)
        recovered, recovered_rms = recover_inclusion_fraction(pack_dir, filename)
        forward[label] = {
            "volume_fraction": expected_volume,
            "rms_percent": rms,
        }
        inverse[label] = {
            "expected_volume_fraction": expected_volume,
            "recovered_volume_fraction": recovered,
            "absolute_fraction_error": abs(recovered - expected_volume),
            "rms_percent_at_recovered_fraction": recovered_rms,
        }
        if rms >= FIT_RMS_LIMIT_PERCENT:
            failures.append(
                f"{label} fit forward RMS {rms:.6g}% >= {FIT_RMS_LIMIT_PERCENT}%"
            )
        if abs(recovered - expected_volume) >= INVERSE_FRACTION_TOLERANCE:
            failures.append(
                f"{label} inverse fraction error {abs(recovered - expected_volume):.6g} "
                f">= {INVERSE_FRACTION_TOLERANCE}"
            )

    measured: dict[str, object] = {}
    for label, (filename, expected_volume) in measured_specs.items():
        summary = measured_error_summary(pack_dir, filename, expected_volume)
        measured[label] = summary
        median = float(summary["median_percent"])
        p95 = float(summary["p95_percent"])
        if median >= MEASURED_MEDIAN_LIMIT_PERCENT:
            failures.append(
                f"{label} measured median {median:.6g}% >= "
                f"{MEASURED_MEDIAN_LIMIT_PERCENT}%"
            )
        if p95 >= MEASURED_P95_LIMIT_PERCENT:
            failures.append(
                f"{label} measured p95 {p95:.6g}% >= {MEASURED_P95_LIMIT_PERCENT}%"
            )

    return {
        "forward_fit_validation": forward,
        "inverse_fit_validation": inverse,
        "measured_validation": measured,
        "limits": {
            "fit_rms_percent": FIT_RMS_LIMIT_PERCENT,
            "inverse_fraction_absolute": INVERSE_FRACTION_TOLERANCE,
            "measured_median_percent": MEASURED_MEDIAN_LIMIT_PERCENT,
            "measured_p95_percent": MEASURED_P95_LIMIT_PERCENT,
        },
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    default_pack = FREDDY_ROOT / "materials" / "validation" / "nist_bam_pdms" / "freddy"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-dir", type=Path, default=default_pack, help="Converted FREDDY CSV directory."
    )
    args = parser.parse_args()
    report = validate(args.pack_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
