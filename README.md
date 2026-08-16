# 20260804_GHOST

2-D boundary-integral / MoM RCS solver plus the SLURM drivers that run it as a
sweep, tuned for throughput on 64–96 core nodes with 375–750 GB across up to 50
nodes.

Start with **[HPC.md](HPC.md)** — how to size a run, what the tuning knobs do,
and what the solver and scheduler changes actually bought.

For `.geo` boundary types, material flags, inline/CSV dielectric and IBC
definitions, impedance tapers, winding, and phasor signs, see
**[GEOMETRY_INPUT_CHEATSHEET.md](GEOMETRY_INPUT_CHEATSHEET.md)**.

## Open the desktop GUI

Install the GUI dependencies once:

```bash
python -m pip install numpy scipy matplotlib PySide6
```

Then start the combined Geometry and Solver interface from the repository root:

```bash
python Backend/ghost_gui.py
```

You can instead double-click `Launch_GHOST_GUI.bat` on Windows or
`Launch_GHOST_GUI.command` on macOS. The launchers check Python and the required
imports before opening the GUI and print the installation command if anything
is missing. In the GUI, load or construct a boundary on the **Geometry** tab,
validate it, then switch to **Solver** and click **Use Geometry Tab**.

## Layout

```
Backend/
  rcs_solver.py               2-D BIE/MoM solver (Robin/MFIE, sheet, dielectric, multi-region)
  hpc_scheduler.py            shared sweep scheduling: cost model, claims, memory admission
  run_hpc_monostatic.py       manifest-tracked 2-D SLURM sweep
  run_hpc_bor_monostatic.py   body-of-revolution SLURM sweep
  run_local_monostatic.py     the same 2-D sweep on one machine, no SLURM
  run_local_bor.py            the same BoR sweep on one machine, no SLURM
  build_bor_stream_kernel.py  build the optional native BoR streaming sampler
  ...                         geometry/material I/O, quality gates, provenance, grim export
0_calibrate_shadowing/        shadowing bias calibration
1c_build_deltas/              delta assembly from solved coupons
tests/                        equivalence, scheduling, and benchmark scripts
```

## Quick start

```bash
# One machine: edit the CONFIG block, then
python Backend/run_local_monostatic.py
python Backend/run_local_bor.py

# SLURM: edit the CONFIG block, then
python Backend/run_hpc_monostatic.py
python Backend/run_hpc_bor_monostatic.py

# Check the scheduler and a real two-task sweep end to end
python tests/test_hpc_scheduling.py
python tests/test_local_drivers.py
```

There is one general-purpose driver family per solver: the `run_hpc_*` driver
for SLURM and its `run_local_*` twin for one machine. The same result files work
in GRIM and downstream feature/delta tools whether or not mesh certification
was selected. They share the scheduler, so units are cost-ordered and admitted
against a memory budget rather than filling every core. See
[HPC.md](HPC.md#running-without-slurm).

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
python tests/test_bor_physics_regression.py
```

The BoR suite checks PEC, passive IBC, lossy dielectric, and lossy coated-PEC
spheres against independent Mie-series references. It also checks table versus
streaming assembly, per-frequency mesh/resource planning, survey provenance,
and VV/HH co-solve scheduling. See [BOR_CONVENTIONS.md](BOR_CONVENTIONS.md) for
the geometry, polarization, RCS, phasor, material-loss, and certification
conventions.

Large conductor BoR jobs use an optional native sampling kernel. The NumPy
fallback is physically equivalent but normally 2–8 times slower during
streaming assembly. Build the kernel on each target platform before submitting
a run (the native artifact is part of solver provenance) with:

```bash
python Backend/build_bor_stream_kernel.py
```

Away from the corrected close-gap and varying-impedance cases, agreement with
the pristine solver remains at floating-point reassociation level. The optional
`GHOST_FAR_QUAD_ORDER` setting can trade far-pair quadrature accuracy for speed;
it is off by default and records itself in a solve's warnings. See
[HPC.md](HPC.md#optional-far-pair-quadrature-order).
