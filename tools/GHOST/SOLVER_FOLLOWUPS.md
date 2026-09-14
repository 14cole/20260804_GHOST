# Automatic solver follow-ups and airfoil qualification

This update accelerates the existing Galerkin equations and improves automatic
execution planning for workstation and HPC runs. Numerical backend selection is
automatic in the normal GUI and driver workflows. It remains a prediction, not
a guarantee of the globally fastest solve.

## Implemented

- An optional portable C99 evaluator computes the validated lossy-material
  polynomial tables with normalized-coordinate Horner evaluation. Every table
  is checked against the special functions at construction, at the existing
  `2e-13` tolerance. Unavailable or unqualified native evaluation uses SciPy.
  This benefits arbitrary imported geometry; it does not fit or smooth contours.
- Closed PEC FMM equations can supplement their equilibrated near-field ILU
  with a 16-vector spatial coarse correction after an expensive first
  illumination, when at least eight
  illuminations remain. Useful recycled directions survive the change; products
  involving the old preconditioner are recomputed. The following illumination
  must show enough iteration reduction to justify setup across the remaining
  work, or the original preconditioner and recycled products are restored.
  Ill-conditioned coarse systems and failed trial solves are rejected. Both the
  adjoint and the original-equation residual remain checked. Coarse setup time,
  operator columns and storage are reported separately from iteration counts.
  Material interfaces, impedance sheets, open contours and combined-field
  equations retain the existing ILU until the new correction is qualified for
  them. This restriction follows measured regressions, not a user-facing switch.
- FMM reuses its immutable point-array identity and compares existing density
  arrays directly instead of copying them into dictionary byte keys.
- FMM memory admission now itemizes sparse near-matrix copies, ILU fill/setup,
  Krylov/recycled vectors, coarse work, material-specific quadrature points and
  native scratch. These are conservative allowances, not an operating-system
  allocation cap. The solver and HPC scheduler use the same forecast.
- Dense memory admission credits the exact owned matrix retained for the second
  polarization. The geometry, materials, frequency, formulation, quadrature,
  shape and ownership must match. Absolute configured and scheduler budgets
  still cap the total forecast. This fixes a false memory rejection encountered
  during the 5 GHz airfoil refinement test after about ten minutes of work.
- Automatic can rank backends from repeated successful timings of an identical
  request on the same host. At least two timings of each of two backends are
  needed. Changes to source/native artifacts, materials, geometry, angles,
  precision, numerical settings or CPU settings invalidate the evidence.
  Unmeasured candidates retain a scaled work prior. A worker can reconsider a
  batch choice only within its assigned RAM reservation. History expires after
  14 days; missing, corrupt or unwritable history leaves the prior available.
- Public solve inputs accept one-dimensional NumPy arrays for frequencies and
  angles. Malformed dimensions retain explicit errors.
- Headless qualification uses a fresh temporary directory. Installed-wheel
  testing finds shared dependency runtimes without executing editable import
  hooks. The release text gate permits reviewed platform-dispatch identifiers
  only in four exact, hash-checked source revisions; other prohibited terms and
  changed revisions remain rejected.
- Linux CI builds both optional native components. The new benchmark command
  runs fresh processes with time limits and records fields, timing, RAM,
  numerical evidence, source/runtime identity and Automatic decisions. Timeout
  cancellation stops owned descendant processes and produces a nonzero exit status.

## Airfoil conditions and measurements

The supplied `airfoil.geo` is interpreted in inches, matching the existing
airfoil study configuration. Its SHA-256 is
`b167517986a9d549b024b16f37c692e0ece3382bdcaaf35fe1a036a3c5a82d70`.
It contains 350 input primitives in 19 segments, PEC and two lossy dielectric
materials, and four material junction nodes. Geometry, material values, panel
rules and numerical tolerances were preserved.

Measurements use Windows, Python 3.12, NumPy 2.5.2, SciPy 1.18.1, two assembly
threads and two BLAS threads on the Ryzen 7 9800X3D workstation. Both VV and HH
are solved at 19 angles from 0 to 360 degrees. The following initial isolated
pairs use the same global mesh and Automatic selected dense in both revisions.
They exclude mesh certification; these are measured cases, not promised gains.
Condition estimation is disabled in these raw timing comparisons.

| Frequency | Panels | Before | After | Time saved | Maximum normalized complex-field difference |
|---|---:|---:|---:|---:|---:|
| 1 GHz | 1,938 | 13.95 s | 9.42 s | 32.5% | 1.31e-12 |
| 5 GHz | 9,412 | 246.85 s | 210.54 s | 14.7% | 5.35e-12 |

The 5 GHz peak sampled process RAM stayed essentially unchanged at 6.52 GiB.
The principal airfoil savings are assembly time, not reduced matrix storage.
A final independent 1 GHz pair measured 13.87 s versus 9.37 s (32.4% less time).

The 1 GHz certified run compares 1,938 and 3,028 panels. Both polarizations pass
the existing mesh, residual and condition gates. The worst normalized complex
field change is 0.00984%; the worst RCS change is 0.00238 dB. This is convergence
evidence for the supplied piecewise-linear model at the tested angles. It does
not establish material-model accuracy or replace an independent airfoil solution.

The 5 GHz certified run also passes, comparing 9,412 and 14,242 panels and
publishing the refined solution. The worst normalized complex-field change is
0.00628%, the worst RCS change is 0.00661 dB, and the largest estimated condition
number is 44,133. Peak sampled process RAM was 14.56 GiB. Completion took
754.8 s including refinement and condition estimation, with other verification
work active during part of the run; that duration is not a speedup benchmark.
The retained refined matrix received 6.96 GiB of verified admission credit,
while the 24 GiB configured cap remained enforced.

The airfoil retains its material-junction diagnostics. The trace/flux relations
are diagnostic rather than separate constraints on the indirect SLP unknowns;
regional potentials provide the coupling. Independent multi-material junction
validation remains necessary before claiming general physical accuracy.

At 20 GHz the forecast has 37,382 base panels and 56,108 refined panels, with
56,702 and 85,098 unknowns per polarization. The refined dense allowance is
216.1 GiB. For the nine-angle forecast, the same policy selects FMM under a
simulated 24 GiB admission budget and dense under 512 GiB. This is a planner
check, not a remotely executed or numerically certified 20 GHz airfoil solve.

## Arbitrary-geometry comparison

The following fresh-process comparisons force FMM to exercise its implementation
on seven 192-panel fixtures at 3 GHz, nine angles, both polarizations and one
assembly/BLAS thread. These are single timing pairs; changes near 1% should be
treated as noise. Automatic selects dense for all seven small problems.

| Geometry/material case | Before | After | Elapsed-time change |
|---|---:|---:|---:|
| PEC rectangle | 7.69 s | 7.27 s | 5.4% less |
| PEC reentrant corner | 8.94 s | 8.04 s | 10.0% less |
| PEC acute corner | 9.78 s | 9.70 s | 0.8% less |
| PEC close gap | 11.17 s | 10.00 s | 10.5% less |
| Dielectric | 84.22 s | 84.06 s | 0.2% less |
| Mixed PEC/dielectric | 30.63 s | 30.85 s | 0.7% more |
| Impedance sheet | 37.64 s | 37.80 s | 0.4% more |

Every case passes the existing numerical quality gate. The maximum complex-field
difference from the prior solver, normalized by the peak reference field for
each polarization, is 1.42e-10. The original unrestricted coarse trial slowed
mixed-material and sheet cases by 6-8%, even with rejection of unhelpful trials.
Automatic coarse correction is therefore restricted to closed PEC equations;
the material rows were rerun after that restriction and are effectively unchanged.
The reentrant case was also rerun to confirm its benefit remained. Other PEC
rows exercise the same retained numerical path. Short PEC sweeps can still pay
for an unsuccessful trial; a universal FMM speedup is not claimed.

## Verification

The integrated acceptance work passed the GRIM suite (1,036 unittest cases),
GHOST suite (741 cases) and FREDDY suite (197 cases), with one expected skip in
each. The final changed-solver headless suite, local-driver integration,
HPC-scheduling integration, ASCII-transfer check, startup diagnostics, source
inventories, offline wheelhouse tests and release text checks also passed.
The full run's local-driver step initially detected source edits made during
verification; it correctly rejected that mixed source state. Repeating the
step after freezing the backend passed. Later memory and FMM changes were
covered by the focused regressions and the final headless rerun.

An independently built GHOST wheel contains the new C source and Python
modules, excludes platform binaries, and loads its SciPy fallback from the
wheel payload. Its sampled relative kernel error was 4.78e-15. The benchmark
command also completed an actual Automatic airfoil solve. Linux CI is
configured, but a Linux run is not claimed here.

## Reproduction and deployment

Build the optional table evaluator with a C99 compiler (`CC` can name its full
path). Build FMM with a complete GNU Fortran installation (`FC`):

```bash
python tools/GHOST/ghost_backend/twod/assembly/native/build.py
python tools/GHOST/ghost_backend/twod/fmm/native/build.py
python tools/GHOST/scripts/check_headless.py
python tools/GHOST/scripts/benchmark_solver.py airfoil.geo --frequencies 1 5 --angles 19 --output airfoil-benchmark.json
python tools/GHOST/scripts/benchmark_solver.py airfoil.geo --frequencies 1 --certified --output airfoil-certified.json
```

The benchmark defaults to Automatic and two fresh-process repetitions. Explicit
`--modes` are for backend comparison experiments. `--forecast-only` builds the
resource plan without solving. Raw timings exclude condition estimation unless
`--condition-estimate` is supplied; `--certified` always includes it and mesh
refinement. Windows DLLs must be rebuilt for Linux workers. The Windows C build
now supplies the compiler's runtime search path and hides helper windows; the
reported missing `libgcc_s_seh-1.dll` compiler error was resolved and the new
table library builds and loads successfully.
No remote cluster run or queue-delay measurement was available in this update.
HPC parallelism distributes sweep units among workers. A single linear solve
must still fit its execution node; this update does not add a distributed-memory
factorization or automatically choose between local execution and a remote queue.

## Further qualification

The current coarse correction is bounded; it is not a full multilevel or
oscillation-aware preconditioner. The timing history applies to identical
requests, not an extrapolated machine model. Local material meshing remains a
candidate with the existing convergence gate. Adaptive higher-order elements,
general corner-aware Nystrom/RCIP, further material couplings and broader
combined-field coverage need independent qualification before Automatic can
use them. No unsupported formulation or relaxed tolerance was enabled to
manufacture a speed gain.

Useful research directions remain corner-aware Nystrom/RCIP
([PEC](https://arxiv.org/abs/1211.2467),
[dielectric transmission](https://arxiv.org/abs/1711.09796)) and high-frequency
enriched approximation spaces for eligible geometries
([dielectric polygons](https://arxiv.org/abs/1704.07745),
[screens and apertures](https://arxiv.org/abs/1912.09916)). These methods address
different approximation and integration costs; none is a universal replacement
for Galerkin across all supported inputs.

The strongest next general-purpose development target is adaptive higher-order
Galerkin with corner grading and a verified error estimator: reducing unknowns
can reduce assembly, factorization, FMM and storage together. This is an
engineering recommendation, not a measured result of this update. Nystrom/RCIP
is a separate candidate for qualified PEC and dielectric corner problems; the
cited papers do not qualify every IBC, junction and multi-material combination
accepted by this application. High-frequency enriched spaces are promising for
eligible polygons and screens. The cited dielectric-polygon method still uses
Galerkin; the screen paper uses oversampled, stabilized least-squares collocation
because a naive square collocation system can be unstable. A simple global swap
from Galerkin to collocation would therefore be unjustified.

The preserved FMM source notices still disagree with the upstream root license.
That external licensing question and a real Linux/HPC qualification remain
outstanding. The source text-gate repair does not resolve that discrepancy.
