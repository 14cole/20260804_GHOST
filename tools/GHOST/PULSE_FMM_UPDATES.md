# Pulse basis and FMM efficiency

## Select a discretization

In **Advanced Settings > Boundary discretization**, select **Pulse / midpoint
collocation**. Dense, compressed and FMM backends support this option. Automatic
chooses a backend within the selected basis. Galerkin remains the default and
retains its existing adaptive polynomial meshing.

```python
execution_options = {
    'discretization': 'pulse',
    'factorization': 'adaptive',
    'assembly_threads': 4,
    'blas_threads': 2,
    'ram_budget_gib': 8,
}
```

Pass these settings to `solve_monostatic_rcs_2d_certified`, or save them in a
desktop run profile. Local/HPC workers preserve the selection. Pulse uses a
constant density per straight panel and midpoint testing. It supports
double-precision 2-D monostatic PEC/impedance bodies, bulk dielectric interfaces,
layered/coated bodies and mixed body configurations. Thin sheets, thin-layer
approximations, bistatic/BOR Pulse and Pulse boundary-density export are not
implemented.

Pulse uses global panel refinement for certification. Selecting Pulse with the
GUI's adaptive meshing default normalizes the mesh strategy to `global`;
polynomial enrichment requires Galerkin. This prevents an unchanged Pulse
density space from being certified by increasing a Galerkin polynomial degree.
The existing Galerkin polynomial and spatial coarse-preconditioner paths remain
available in this integrated version.

## Implemented optimizations

- Closed pure-PEC TM uses the qualified combined-field coupling `-1j*k` under
  GHOST's inward-normal convention. Charge and dipole terms share an FMM call,
  with matching adjoint, near corrections and physical field projection.
- `fmm_pec_cfie='auto'` controls Galerkin FMM. Explicit saved booleans retain their
  meaning. `pulse_pec_cfie=True` controls Pulse consistently across backends.
- `fmm_recycle_vectors='auto'` selects ordinary GMRES for the qualified PEC
  equations and LGMRES augmentation for other formulations. A bounded incident
  basis is retained across angle batches. Every reconstructed solution is
  checked against the original, unscaled equation.
- Automatic FMM quadrature uses six points for electrically short linear/Pulse
  panels in the qualified tolerance range. It retains eight or more outside
  that range and for polynomial Galerkin bases. `fmm_quadrature_order=8` requests
  the former minimum. Electrical panel length can require a larger rule.
- Point-space work is bounded to eight native densities. Numeric near buffers,
  shared Galerkin sparse transposes and omitted unused channels reduce temporary
  storage. Memory admission uses geometric near counts, with separate native,
  sparse/ILU, coarse, sweep and Krylov allowances. The RAM budget is admission
  control, not an operating-system cap.
- Pulse assembly uses analytic singularity subtraction, checked near integrals,
  cached equal-length self integrals with length correction, bounded parallel
  tiles and real-Bessel fast paths. Guarded 2/3/4-point far rules reduce dense
  and compressed integration work; `far_grading=False` disables that reduction.
- Pulse FMM uses full source quadrature and distinct midpoint targets. Its
  adjoint exchanges source/testing maps explicitly. Native plans and sparse
  near primitives are reused across polarizations. Panel far-field moments use
  exact sinc integration.

The singular terms follow the [NIST Hankel expansions](https://dlmf.nist.gov/10.8);
the native operations follow the [FMM2D mathematical convention](https://fmm2d.readthedocs.io/en/latest/math.html).
Quadrature qualification is empirical, not a rigorous global error bound.
Default kernel/linear tolerances remain `1e-10` / `1e-9`.

## Measurements and accuracy

The following measurements were recorded on the standalone linear/Pulse source
before integration with the existing polynomial/coarse improvements. They are
single fresh-process runs on Windows, Ryzen 7 9800X3D, Python 3.12.14, NumPy 2.3.5
and SciPy 1.18.1. They are evidence for these workloads, not guaranteed timings
for the integrated Automatic policy or another solver. RAM is peak Windows
process working set; imports are excluded from elapsed time. Condition
estimation and mesh certification were disabled equally for timing.

For a 6 m x 2.5 m PEC rectangle at 1 GHz, 8,192 panels, one angle, both
polarizations and four threads:

| Method | Time | Peak RAM |
| --- | ---: | ---: |
| Linear Galerkin dense | 34.08 s | 2,181.8 MiB |
| Updated linear Galerkin FMM | 25.76 s | 257.7 MiB |
| Pulse dense | 26.96 s | 2,182.1 MiB |
| Pulse FMM | 19.90 s | 243.7 MiB |

Pulse FMM took 41.6% less time and 88.8% less RAM than dense Galerkin on this
mesh. Its complex field differed from same-mesh dense Galerkin by at most
0.00987%; that is representation agreement rather than independent physical
accuracy. At 2,048 panels / three angles, Pulse dense took 2.12 s versus
13.80 s for Pulse FMM. FMM is therefore not a universal speed default.

The linear Galerkin FMM 256-panel/181-angle benchmark improved from 33.12 s to
8.00 s. Solved incident columns fell from 76 TE + 73 TM to 29 + 28, with original
equation residual checks retained.

Independent circle series also show that Pulse and Galerkin have different
errors at equal panel count. Among tested sizes 64, 128, 256, 512 and 1,024:

| Physical target | Linear Galerkin | Pulse |
| --- | --- | --- |
| PEC TM, ka=10, 0.01% complex-field error | 1,024 panels; 1.296 s; 194.6 MiB | 1,024 panels; 0.638 s; 122.1 MiB |
| PEC TE, 1 GHz, 0.1% complex-field error | 128 panels; 0.090 s; 92.6 MiB | 512 panels; 0.170 s; 95.4 MiB |
| Dielectric TM, 1 GHz, 1% scattering-width error | 64 panels; 0.172 s; 89.3 MiB | 1,024 panels; 3.882 s; 222.3 MiB |

These use the same inscribed polygons, radius 0.06 m, three angles and one
thread. The dielectric has relative permittivity `3-0.1j` and permeability 1.
The table shows the first passing mesh on the tested grid; complex-field and
scattering-width errors are distinct metrics. Extra Pulse refinement can erase
its assembly advantage, especially for the tested dielectric. Keep physical
mesh checks enabled. Subdividing straight panels does not certify fidelity to
an underlying curved object.

## Verification

The standalone implementation passed 227 tests plus 90 subtests and eight
additional quadrature checks. The integrated merge additionally exercises
polynomial forward/adjoint CFIE operations, existing coarse correction,
Pulse panel-refinement settings and the release source inventory. Run:

```bash
python tools/GHOST/scripts/check_headless.py
python verify_project.py --mode quick
python tools/GHOST/ghost_backend/tests/test_hpc_scheduling.py
python tools/GHOST/ghost_backend/tests/test_local_drivers.py
```

The native source/binary interfaces are unchanged. FMM needs the existing
platform library; Automatic excludes it when unavailable. Persistent native
translation caching, broader multilevel preconditioning, additional material
couplings and larger capacity/contrast studies remain further development.
