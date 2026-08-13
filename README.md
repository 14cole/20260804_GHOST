# 20260804_GHOST

2-D boundary-integral / MoM RCS solver plus the SLURM drivers that run it as a
sweep, tuned for throughput on 64–96 core nodes with 375–750 GB across up to 50
nodes.

Start with **[HPC.md](HPC.md)** — how to size a run, what the tuning knobs do,
and what the solver and scheduler changes actually bought.

For `.geo` boundary types, material flags, inline/CSV dielectric and IBC
definitions, impedance tapers, winding, and phasor signs, see
**[GEOMETRY_INPUT_CHEATSHEET.md](GEOMETRY_INPUT_CHEATSHEET.md)**.

## Layout

```
Backend/
  rcs_solver.py               2-D BIE/MoM solver (Robin/MFIE, sheet, dielectric, multi-region)
  hpc_scheduler.py            shared sweep scheduling: cost model, claims, memory admission
  run_hpc_monostatic.py       manifest-tracked 2-D SLURM sweep
  run_hpc_bor_monostatic.py   body-of-revolution SLURM sweep
  run_local_monostatic.py     the same 2-D sweep on one machine, no SLURM
  run_local_bor.py            the same BoR sweep on one machine, no SLURM
  step1_monostatic.py         shared implementation for the numbered 2-D runners
  ...                         geometry/material I/O, quality gates, provenance, grim export
0_calibrate_shadowing/        shadowing bias calibration
1a_solve_2d_local/            single-machine coupon solve
1b_solve_2d_hpc/              coupon library on SLURM
1c_build_deltas/              delta assembly from solved coupons
tests/                        equivalence, scheduling, and benchmark scripts
```

## Quick start

```bash
# One machine
python 1a_solve_2d_local/run_monostatic_local.py

# SLURM: edit the CONFIG block at the top, then
python 1b_solve_2d_hpc/run_monostatic_hpc.py

# Check the scheduler and a real two-task sweep end to end
python tests/test_hpc_scheduling.py
python tests/test_local_drivers.py
```

Every driver has a local twin that runs the identical sweep without SLURM
(`Backend/run_local_monostatic.py`, `Backend/run_local_bor.py`,
`1a_solve_2d_local/run_monostatic_local.py`). They share the scheduler with the
cluster path, so units are cost-ordered and admitted against a memory budget
rather than filling every core, and `results/` holds one `.grim` per unit and
nothing else. See [HPC.md](HPC.md#running-without-slurm).

## Numerical validation

Performance-only changes are checked against a pristine copy of the
pre-optimization module — operator by operator and end to end on published RCS:

```bash
python tests/test_assembly_equivalence.py /path/to/pristine/rcs_solver.py
python tests/test_solver_equivalence.py   /path/to/pristine/rcs_solver.py
```

The current solver also contains two intentional correctness changes: adaptive
quadrature for separated but nearly touching panels, and element-weighted
Galerkin assembly for spatially varying sheet/Robin impedances. Focused tests
cover those corrections, their constant-coefficient limits, vectorized
far-field equivalence, PEC/IBC/homogeneous/coated cylinder references in both
polarizations, tapered-impedance invariance, and bistatic reciprocity:

```bash
python tests/test_rcs_physics_regression.py
```

Away from the corrected close-gap and varying-impedance cases, agreement with
the pristine solver remains at floating-point reassociation level. The optional
`GHOST_FAR_QUAD_ORDER` setting can trade far-pair quadrature accuracy for speed;
it is off by default and records itself in a solve's warnings. See
[HPC.md](HPC.md#optional-far-pair-quadrature-order).
