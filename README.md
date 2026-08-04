# 20260804_GHOST

2-D boundary-integral / MoM RCS solver plus the SLURM drivers that run it as a
sweep, tuned for throughput on 64–96 core nodes with 375–750 GB across up to 50
nodes.

Start with **[HPC.md](HPC.md)** — how to size a run, what the tuning knobs do,
and what the solver and scheduler changes actually bought.

## Layout

```
Backend/
  rcs_solver.py               2-D BIE/MoM solver (Robin/MFIE, sheet, dielectric, multi-region)
  hpc_scheduler.py            shared sweep scheduling: cost model, claims, memory admission
  run_hpc_monostatic.py       manifest-tracked 2-D SLURM sweep
  run_hpc_bor_monostatic.py   body-of-revolution SLURM sweep
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
```

## Numerical equivalence

The solver optimizations change how the element-pair quadrature is staged, not
what it computes. Both entry points are checked against a pristine copy of the
pre-optimization module — operator by operator and end to end on published RCS:

```bash
python tests/test_assembly_equivalence.py /path/to/pristine/rcs_solver.py
python tests/test_solver_equivalence.py   /path/to/pristine/rcs_solver.py
```

Agreement is at floating-point reassociation level (~1e-16 relative on the
operators, ~1e-13 dB on published RCS). The one setting that *does* change
values, `GHOST_FAR_QUAD_ORDER`, is off by default and records itself in a
solve's warnings; see [HPC.md](HPC.md#optional-far-pair-quadrature-order).
