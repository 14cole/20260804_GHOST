#!/usr/bin/env python3
"""Solve every geometries/FRD and geometries/OPN coupon on this machine.

Outputs are written directly to results/FRD and results/OPN. Either input
folder may be empty. Existing results are reused only when their embedded
input/solver fingerprint matches the current request.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Backend"))

from step1_monostatic import (  # noqa: E402
    _pin_blas,
    discover_jobs,
    prepare_jobs,
    solve_job,
    validate_config,
)

# -- USER SETTINGS ----------------------------------------------------------
FREQUENCIES_GHZ = [3.0, 6.0]
ANGLES_DEG = np.arange(0.0, 180.1, 5.0)
POLARIZATIONS = ["TM", "TE"]
GEOMETRY_UNITS = "meters"
WORKERS = None                 # None = all but one CPU core
SOLVER_METHOD = "auto"
MAX_PANELS = 50_000
FORCE = False
# ---------------------------------------------------------------------------


def main() -> 'None':
    try:
        frequencies, angles, polarizations = validate_config(
            FREQUENCIES_GHZ, ANGLES_DEG, POLARIZATIONS
        )
        jobs = discover_jobs(HERE, frequencies, polarizations)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not jobs:
        raise SystemExit(
            "No .geo files found. Put coupons in geometries/FRD or "
            "geometries/OPN; either folder may otherwise be empty."
        )
    jobs = prepare_jobs(
        jobs,
        angles_deg=angles,
        geometry_units=GEOMETRY_UNITS,
        solver_method=SOLVER_METHOD,
        max_panels=MAX_PANELS,
        runner_path=__file__,
    )
    cpu_count = os.cpu_count() or 1
    workers = max(1, min(int(WORKERS or max(1, cpu_count - 1)), len(jobs)))
    kwargs = {
        "angles_deg": angles,
        "geometry_units": GEOMETRY_UNITS,
        "solver_method": SOLVER_METHOD,
        "max_panels": MAX_PANELS,
        "force": FORCE,
    }
    print(
        f"Step 1 local: {len(jobs)} solve unit(s), {workers} worker(s); "
        "outputs -> results/{FRD,OPN}"
    )
    _pin_blas(1)
    failures = 0
    # ProcessPoolExecutor gained initializer/initargs only in Python 3.7.
    # The environment is pinned before workers spawn, so 3.6 workers inherit it.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(solve_job, job, **kwargs): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                status, path = future.result()
                print(f"[{index}/{len(jobs)}] {status:7s} {Path(path).name}")
            except Exception as exc:
                failures += 1
                print(
                    f"[{index}/{len(jobs)}] FAILED  {job['role']}/"
                    f"{Path(job['output']).name}: {exc}"
                )
    if failures:
        raise SystemExit(f"{failures} solve unit(s) failed.")


if __name__ == "__main__":
    main()
