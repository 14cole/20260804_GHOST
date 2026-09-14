# Automatic polynomial mesh adaptation

New certified 2-D monostatic runs use an adaptive polynomial Galerkin strategy.
The GUI, local driver and HPC driver share this default. Users provide geometry,
materials, frequencies, angles and accuracy settings. Dense, compressed and FMM
execution remain automatic choices on the execution node.

The [measurement report](ADAPTIVE_MESH_BENCHMARKS.md) contains paired airfoil
timings, memory use, field comparisons and 10/20 GHz resource forecasts.

## Implemented strategy

The initial candidate uses material-local wavelengths and approximately one
quarter of the reference linear panel density. Quadratic basis functions describe
the boundary density. A second solve uses cubic basis functions on the same
panels. Every original straight primitive, corner, gap and material boundary is
retained. Endpoints, junctions and corners receive graded subdivisions. This
does not fit a curve through imported vertices or simplify the physical model.

Models with fewer than 1,024 estimated linear reference panels retain the
reference method automatically. Paired small-model measurements found that
polynomial quadrature could outweigh factorization savings. This conservative
size rule is shared by the desktop and HPC planners; it is not a universal
hardware-independent crossover guarantee.
Sweeps evaluate the crossover separately for each frequency, so one request
can correctly use a linear mesh at lower frequencies and a cubic mesh above it.

Both polarizations must pass the existing original-equation residual and
condition checks. The solver compares complex far fields at every requested
frequency and angle. Adaptive convergence limits are the stricter of the user's
settings and these ceilings:

| Metric | RMS | Maximum |
|---|---:|---:|
| Normalized complex-field change | 0.1% | 0.2% |
| RCS change | 0.05 dB | 0.25 dB |
| Phase change above the configured field floor | 0.5 degrees | 2 degrees |

A failed comparison triggers global h refinement with additional subdivisions
on primitives marked by the highest Legendre coefficient of the solved density.
The indicator combines requested illuminations and material-interface densities.
Two such refinement rounds are allowed. If they fail, or polynomial quadrature
cannot meet its tolerance, the solver checks the established linear reference
mesh pair using the requested reference policy. A failed reference check produces
an error rather than an accepted unconverged result.

Polynomial degree is uniform within each candidate (quadratic then cubic);
subdivision density varies by primitive. This is a bounded h/p strategy, not an
implementation of arbitrary element-wise polynomial degrees. Modal marking and
mesh comparisons provide convergence evidence, not a rigorous physical error
bound. Unrequested angles, geometric modelling error and constitutive-model
accuracy require separate qualification.

## Operator and resource changes

Dense, compressed and FMM operators support degree 1, 2 and 3. Excitation,
far-field projection, mass matrices, Maue hypersingular terms, interface routing
and storage estimates use the actual basis width. Polynomial self terms subtract
the logarithmic singularity analytically. Touching panels use a Duffy transform
with stable relative coordinates and no arbitrary nonzero-distance cutoff.
Near quadrature retains its 1e-9 relative convergence check.

The workstation planner and HPC scheduler forecast the same initial polynomial
meshes. Each later candidate is admitted again against the executing worker's
RAM reservation, with compatible backend retries. The reservation never expands
implicitly. Result metadata records degrees, panel and node counts, timings,
backend decisions, refinement steps, acceptance policy and fallback reasons.
Mixed-backend attempts and reference fallbacks do not train single-backend
timing history.
An unwritable optional timing cache is ignored after one file-creation attempt;
Windows permission failures cannot trap a completed solve in a retry loop.

Explicit positive panel-count profiles remain on the linear reference route.
Spatially varying impedance also retains h refinement so material sampling is
checked. The thin-layer asymptotic formulation retains its qualified linear
basis. Existing saved explicit settings remain authoritative. Raw solves do not
perform adaptive certification; keep certification enabled to use this strategy.
Body-of-revolution and bistatic algorithms retain their existing discretizations.

HPC execution still distributes sweep work among nodes/workers. A single solve
must fit its node. This change does not add MPI distributed matrix storage or
automatically submit local jobs to a remote queue.

## Qualification and reproduction

The polynomial tests check exact polynomial reproduction and mass matrices,
an independent one-dimensional self-integral, translation/scaling of touching
panels, and dense/compressed/FMM agreement for PEC, dielectric, mixed and sheet
cases. Adaptive tests exercise refinement, reference fallback, fixed-count
profiles and actual execution inside a fresh HPC worker.

For a paired measurement, run these sequentially on an otherwise idle machine:

```bash
python tools/GHOST/scripts/benchmark_solver.py airfoil.geo --frequencies 1 5 --certified --angles 19 --threads 2 --ram-gib 24 --storage-mib 256 --mesh-strategy global --output reference.json
python tools/GHOST/scripts/benchmark_solver.py airfoil.geo --frequencies 1 5 --certified --angles 19 --threads 2 --ram-gib 24 --storage-mib 256 --mesh-strategy adaptive --output adaptive.json
```

The benchmark launches fresh workers, records source/runtime identity and refuses
results if solver source changes during a solve. Measured savings are specific
to each case; small problems can cost more because higher-order quadrature and
accuracy checks have overhead.

## Numerical context

Higher polynomial order and local mesh refinement are established Galerkin BEM
techniques. The frequency-explicit analysis of
[Loehndorf and Melenk](https://epubs.siam.org/doi/10.1137/100786034)
concerns particular combined-field equations and analytic boundaries. Its
assumptions do not establish accuracy guarantees for this implementation's
arbitrary material junctions. Rigorous residual-based adaptivity such as
[Bespalov, Betcke, Haberl and Praetorius](https://arxiv.org/abs/1807.11802)
is a further direction; the modal indicator used here is not that estimator.
