# 20260804_GHOST

2-D boundary-integral / MoM RCS solver plus the SLURM drivers that run it as a
sweep, tuned for throughput on 64–96 core nodes with 375–750 GB across up to 50
nodes.

Start with **[HPC.md](HPC.md)** — how to size a run, what the tuning knobs do,
and what the solver and scheduler changes actually bought.

For `.geo` boundary types, material flags, inline/CSV dielectric and IBC
definitions, impedance tapers, winding, and phasor signs, see
**[GEOMETRY_INPUT_CHEATSHEET.md](GEOMETRY_INPUT_CHEATSHEET.md)**.

For direct full-wave validation of coherently placed BoR doors, seams, and
cavities, see **[FEATURE_VALIDATION_GUIDE.md](FEATURE_VALIDATION_GUIDE.md)**.

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
  place_features.py           coherent feature placement on BoR or external 3-D output
  import_3d_reference.py      import an attested external complex truth field
  validate_feature_reconstruction.py  compare truth to reconstructed fields
  build_bor_stream_kernel.py  build the optional native BoR streaming sampler
  ...                         geometry/material I/O, quality gates, provenance, grim export
1c_build_deltas/              strict CEM OPN-FRD delta entry point
CEM_Tools/                    headless/GUI GRIM join, conversion, and coherent subtraction
tests/fixtures/geometries/    solver-only regression geometry inputs
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

# After a BoR or imported 3-D run: select feature datasets and placement CSVs
python Backend/place_features.py

# Build placement-ready 2-D deltas from the newest complete FRD/OPN run
python 1c_build_deltas/subtract_datasets.py

# Optional CEM dataset GUI for inspection and general dataset utilities
python3 CEM_Tools/run_gui.py

# Check the scheduler and a real two-task sweep end to end
python tests/test_hpc_scheduling.py
python tests/test_local_drivers.py
```

`1c_build_deltas/subtract_datasets.py` is the sole production path from solved
2-D coupons to a placement-ready line-feature delta. It performs canonical
`OPN - FRD` subtraction on preserved float64 complex amplitudes, joins the
frequency files, and validates the embedded VV/HH pair. Do not concatenate
the solver outputs first, subtract dB values, or use GRIM overlap as a
preprocessing step. The general CEM concatenation operations remain available
for unrelated dataset-library work, but are not part of this workflow.

There is one general-purpose driver family per solver: the `run_hpc_*` driver
for SLURM and its `run_local_*` twin for one machine. The same result files work
in GRIM and downstream feature/delta tools whether or not mesh certification
was selected. They share the scheduler, so units are cost-ordered and admitted
against a memory budget rather than filling every core. See
[HPC.md](HPC.md#running-without-slurm).

The 2-D launchers expose geometry, frequency, cut angles, units, and quality
controls, but no polarization, solver-method, or CFIE selection. By default,
every `(geometry, frequency)` unit runs the certified dense-LU path for both
physical channels; setting `MESH_CERTIFICATION = False` selects the clearly
marked base-mesh survey path. Either path writes one
`<FREQ:.3f>GHz_<geometry_stem>.grim` containing canonical `VV` (`TE`) and
`HH` (`TM`).

The BoR launchers take the requested frequencies, radar azimuths, and radar
elevations directly. Each geometry produces one user-facing
`results/<geometry>.grim` containing the monostatic VV/HH/VH grid and the exact
body model needed for later coherent placement. Visible
`results/by_frequency/` VV/HH files appear as each frequency finishes and also
serve as restart state; the combined monostatic dataset is published when the
geometry is complete.

`Backend/place_features.py` also accepts an attested external monostatic GRIM
plus a platform `.facet` or STL surface. Set `SURFACE_MESH` and its units; the
mesh and placement coordinates must share the external solve's global origin
and the GHOST CAD frame (`+y` nose, `+x` right, `+z` up). Set `SHADOW=True` to
add platform ray blockage, or `False` to retain only each feature's local
outward-facing test. The supported `.facet` form is indexed ASCII:

```text
n_vertices n_facets
vertex_id x y z
...
facet_id vertex_id_1 vertex_id_2 vertex_id_3 [vertex_id_4]
...
```

Triangles and quads are accepted; winding must point outward, or set
`FLIP_SURFACE_NORMALS=True`. Shadowing is a geometric-optics visibility mask:
it does not model diffraction, creeping waves, or new body-feature multiple
scattering.

The launcher entry is the declaration of each file's role, so GUI-derived
power/phase datasets do not need `combine_role`, `phase_reference`,
`complex_field_domain`, or `rcs_domain` bookkeeping tags. The numerical data
must still identify their dimensional normalization (`sigma_2d` for line
deltas and `sigma_3d` for body and compact data), preserve finite power
and phase, and contain the channels and angular support actually used. Stale
descriptive metadata copied by a GUI is superseded by the launcher entry.

Point features use `POINT_FEATURE_DATASETS` plus one
`POINT_FEATURE_LOCATIONS_CSV`; line-expanded features use
`LINE_FEATURE_DATASETS` plus one `LINE_FEATURE_LOCATIONS_CSV`. See
[FEATURE_VALIDATION_GUIDE.md](FEATURE_VALIDATION_GUIDE.md#one-point-placement-csv)
and [the line CSV section](FEATURE_VALIDATION_GUIDE.md#one-line-placement-csv).
The point CSV contains a dataset selector, position, normal, and roll vector.
The line CSV contains a dataset selector, ordered endpoints, and endpoint
outward normals. Both workflows accept only the canonical OPN-FRD
(`featured - clean`) complex delta and do not offer a sign-reversal option.
Relative settings paths are resolved from the repository root, so the launcher
works the same way regardless of the terminal's current directory.

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
