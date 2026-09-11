# GHOST compressed CPU qualification

September 10, 2026. This pass promotes and qualifies the paired, disk-spooled
geometry-to-compressed prototype. Measurements below use the public certified
solver, including condition estimation, both polarizations, all requested angles,
and the base/fine mesh comparison. The earlier prototype's core timings excluded
some of these operations and are not interchangeable with these measurements.

## Status

The path is qualified for the tested numerical workloads, including the original
10 GHz airfoil, all 14 material families, and the runtime/operational cases below.
All 74 saved public runs passed their requested gates. There are 57 saved-run
comparisons against independent dense references, all below the 1e-10
channel-peak-relative limit. The
qualification script checks that every required result is present; its
`qualification_complete` result is true.

For the subsequent GUI readiness fixes and final-source smoke test, see
[READINESS.md](READINESS.md). That follow-up corrects the compressed GUI's panel
allowance and boundary-density diagnostics; the measurements below retain
their recorded source revisions.
The subsequent [memory-estimation update](MEMORY_ESTIMATION.md) replaces the
capacity-based RAM forecasts; the saved solve measurements below are unchanged.

The compressed path is now implemented in ordinary Backend modules and selected
explicitly with `GHOST_CPU_FACTORIZATION=compressed`. It has no global dense
fallback. Dense LU remains the default. See
[selection and behavior](../../tools/GHOST/COMPRESSED_CPU.md).

## Completed airfoil evidence

| Workload | Method | Wall time | Sampled peak RSS | Mesh certificate |
|---|---|---:|---:|---|
| 1 GHz, TE/TM, 361 angles | Compressed, initial qualification | 69.52 s | 313 MiB | Passed |
| 2 GHz, TE/TM, 361 angles | Dense | 130.86 s | 2603 MiB | Passed |
| 2 GHz, TE/TM, 361 angles | Compressed | 178.88 s | 577 MiB | Passed |
| 10 GHz, TE/TM, 361 angles | Compressed | 3214.40 s (53 min 34 s) | 3380 MiB (3.30 GiB) | Passed |

At 2 GHz the compressed certificate uses **77.8% less sampled peak RAM** and takes
**36.7% longer**. Fine-mesh complex fields differ from dense by at most
**4.47e-13**, normalized to each reference channel's peak. The original geometry,
material-driven meshing, quadrature and quality thresholds are preserved. The
base system has 5742 unknowns per polarization; the fine system has 8696.

The original **10 GHz, 0–360 degrees by 1 degree** case completed with 28,406
base unknowns and 42,686 fine unknowns per polarization. It used four inverse
factorizations, with no tighter rebuild or GMRES fallback. Fine-mesh condition
estimate: 112,167; final physical-angle relative residual: 1.02e-11. The maximum
base/fine complex-field difference was 1.38e-5 of the channel peak, maximum dB
difference 0.01089 dB and maximum phase difference 0.03801 degrees. All normal
quality gates passed. The input snapshot was checked against the current
`airfoil.geo` and matches exactly.

The incident basis required 110/110 solved columns on base TE/TM and 112/155
on fine TE/TM, including repairs. This totals 487 columns versus 1444 physical
illumination columns across the four solves. The base TM reconstruction needed
one extra physical solve and fine TM needed 43. Those rejected reconstructions
were repaired and checked; the report does not treat a small QR RHS error as
sufficient evidence of a correct physical solution. Historical event maxima
also include intermediate basis solves and rejected reconstruction candidates;
the final channel residuals and quality gates describe the published result.

At 10 GHz, compressed assembly took 1781.30 s and inverse construction 1250.78 s;
these are distinct stages, approximately 55% and 39% of elapsed time. The operator
kernel stage (1068.66 s) is nested inside assembly. The 119.18 s RHS-compression
stage includes 65.00 s of linear solves. Do not add nested stages twice.
Fine TE/TM operator payloads were 1722/1698 MB, and inverse payloads 527/478 MB.
The largest TM spool held 1695 MB, removed after loading. The test allowed
8192 MiB of retained payload; that allowance was not preallocated.

No dense 10 GHz reference was run. Its earlier approximately 87 GiB prediction,
and the later 56.41 GiB dense fine-mesh plan, are estimates rather than measured
peaks. The 2 GHz table provides the measured, equal-workload dense comparison.

The 1 GHz run preceded the final one-pass condition equilibration optimization.
Raw result files include Backend source hashes. `validated-results.json` lists
which files changed since each saved run, so earlier measurements are not
silently attributed to a later revision.
The 10 GHz run preceded final failure cleanup, mixed-precision rejection and
adaptive query-cap guards. These preserve its successful arithmetic: its
largest inverse query used 682,976 entries, so its original 32-row chunks remain
unchanged under the new cap. Final source regression and material tests cover
the guards; a full solve above 65,536 unknowns is not claimed.

## Changes made in this pass

1. **Direct native material assembly.** Added compact coefficient queries for
   Robin PEC/IBC, single dielectric S/K/K-prime/W blocks, sheets and transmitting
   thin layers. Compact hypersingular W assembly evaluates only element pairs
   supporting the requested rows/columns, preserving near, touching, endpoint
   and quadrature terms. Thin-layer products apply sparse mass LU to bounded
   complete column strips; no dense mass inverse or global S/K/W is retained.
2. **Shared polarization work with bounded lifetimes.** Regional TE/TM routes
   share compatible kernel traversals. Native material systems share primitive
   queries only until both constitutive laws consume them. TM compressed tiles
   spool to disk during TE, then load after TE operator/inverse release.
3. **Accurate operator plus smaller inverse.** Operator tiles undergo full
   coefficient QR checks at 1e-14. The inverse retains leaf LU and solved
   low-rank/Woodbury data, dropping redundant raw operators and U factors. It
   starts at 1e-6, refines to 3e-15, uses bounded GMRES repair when needed, and
   can retry at 2e-10 after releasing the coarse tree. There is no global dense
   allocation on rejection.
4. **Complete public sweep and condition integration.** Arbitrary angle grids
   use existing bounded batches and incident-basis reuse. Every reconstructed
   physical RHS, including repairs, is checked with original-coefficient error.
   Transpose/adjoint checks use column error sums. Condition equilibration now
   takes one tile pass using row maxima accumulated during assembly. The outer
   solve reuses the inverse's computed residual instead of another matvec.
5. **Operational checks.** Added combined storage accounting, resource admission,
   cancellation checkpoints, owned temporary files with per-record SHA-256
   checks, strict cleanup/rejection tests, actual execution metadata and storage
   budget provenance. Legacy SciPy warning and GMRES API differences are handled.
   Failed loads release partial tiles and remove their file immediately, even
   while an exception traceback retains the operator. File removal is attempted
   even if closing/flushing raises. Mixed precision rejects before public
   assembly. Inverse validation chunks shrink when needed to remain inside the
   16 MiB query cap on larger systems.

## Material qualification

Fourteen families have passed both polarizations and all 361 angles: PEC, IBC,
PEC/IBC, lossless dielectric, lossy dielectric, magnetic dielectric, coated,
layered, mixed dielectric/PEC, sheet, mixed sheet/PEC, thin dielectric, magnetic
thin layer and transparent thin layer. All fourteen also passed the production
base/fine mesh certificate on the final source.

Base-mesh fields were checked against independent dense results saved before
this pass. The maximum channel-peak-relative complex field difference was
6.05e-12, below the qualification threshold of 1e-10.
Fresh dense fine-mesh certificates also match all fourteen final compressed
certificates. Their maximum difference is **5.78e-11**, for magnetic HH, within
the same 1e-10 limit. These comparisons include full complex fields at every
requested angle. Detailed times and RAM are in
[the material comparison table](material-comparison.md).

Sharing native TE/TM primitives reduced measured base solve times from 4.44 to
2.56 s for lossless dielectric, 5.21 to 2.94 s for lossy dielectric, 5.26 to
2.87 s for magnetic dielectric and 3.98 to 2.33 s for magnetic thin layers.
These are single-run comparisons of this pass's initially separate native
assembly against its paired implementation, not comparisons against dense LU.
Both use 361 angles and condition checks. The first-order transmitting thin-layer
model retains `approximation_error_certified=False`; its mesh certificate does
not establish bulk-material model accuracy.

## Bounds and certification meaning

Regional pruning uses a lower separation bound for entire nodal supports and
absolute material/shape weights. For attenuating k with positive real part,
the Hankel-to-K connection and K integral give Gaussian envelopes for H0/H1.
The implementation inflates envelopes and clamps the decay exponent upward to
avoid understating underflow tails. Real, growing or unsupported domains are
not pruned. The supporting identities are
[DLMF 10.27.8](https://dlmf.nist.gov/10.27.E8) and
[DLMF 10.32.9](https://dlmf.nist.gov/10.32.E9).

The sum of pruning and measured tile error is included in each physical
backward-error bound. Condition estimation includes normal/adjoint refined
inverse applications. The normal 1e-12 backward-error, residual, condition and
mesh-convergence gates remain required. These numerical checks are not a formal
interval-arithmetic proof or certification of every geometry/material choice.
The input piecewise-linear geometry approximation remains separately uncertified.

The mesh policy is unchanged: fine factor 1.5, complex RMS/max limits 0.02/0.05,
RMS/max dB limits 1/3, phase RMS/max limits 5/15 degrees, with the existing phase
floor. Tested results are published from the fine mesh.

## Regression and compatibility evidence

| Runtime | Regression/contracts | Native coefficient checks | Compressed headless/export/resume |
|---|---:|---:|---:|
| Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1 | 159 passed | 3 passed | 3 passed |
| Python 3.6, NumPy 1.19.5, SciPy 1.5.4 | 16 compressed contracts passed | 3 passed | 3 passed |
| Python 3.6, NumPy 1.14.3, SciPy 1.0.0 | 16 compressed contracts passed | 3 passed | 3 passed |

This is 209 test executions, including repeated coverage on the two legacy
stacks. Native checks exercise rectangular W with masks and different quadrature
orders, all native material coefficient blocks, paired regional routes, and
pruned-versus-unpruned coefficient bounds. Contract tests cover normal,
transpose and adjoint actions, conditions, 1027-angle streaming, multiple
frequencies, bistatic comparisons, GMRES repair, budgets, singular rejection,
precision rejection, corruption, truncation, cancellation, cleanup and provenance.
Headless tests execute configured local/HPC workers, `.grim` export, portable
bundle verification and resume locally. No remote cluster performance is claimed.

The added extreme-loss pruning fixture initially hit the existing minimum-mesh
safety floor. Its explicit panel resolution was increased; the safety check was
preserved, and the corrected fixture passed on all three stacks. The initial
rejection log is retained. This did not require a solver equation change.

## Further opportunities

The current method still visits all final coefficient tiles. It saves retained
RAM but does not remove quadratic geometry assembly work. A nested H2/FMM or
directional high-frequency operator could avoid much of that work; it would
require a new multi-material near/far implementation and independent accuracy
qualification. This is the largest remaining architectural opportunity, not a
certified drop-in optimization from this pass.

The 10 GHz inverse construction reconstructed about 52.2 billion coefficient
entries from stored tiles across the four factors, approximately ten full-system
passes in aggregate. It did not repeat geometry assembly, but those repeated
queries explain why inverse construction still costs 20 min 51 s. Reusing query
index plans or checking low-rank blocks directly from tile factors is another
target for a future measured optimization; neither improvement is claimed here.

A final inner-tile threading experiment compared the same six representative
2 GHz paired queries, twice per setting. Median totals were 2.050 s with the
default and 1.994/2.201/2.119 s with 128/256/512 inner tiles. Coefficients agreed
within 6.1e-17 relative to the reference peak. The 2.7% best apparent gain was
smaller than the observed default timing spread, so it was not promoted and no
end-to-end speedup is claimed. The tested default remains unchanged. Raw evidence
is in `tile-probe-results.json`.

For a concrete follow-on candidate, FMM2D exposes complex-wavenumber Helmholtz
charges, dipoles, gradients and multiple density vectors on shared geometry.
These are useful primitives for a future operator, rather than a complete
replacement for this Galerkin solver. Its Hankel convention, self exclusion,
near quadrature, jumps and all regional material weights would need explicit
adaptation and tests. This assessment is an implementation inference from its
[official mathematical interfaces](https://fmm2d.readthedocs.io/en/latest/math.html).
The high-frequency directional alternative is supported by
[Engquist and Ying's 2D Helmholtz algorithm](https://web.stanford.edu/~lexing/Helm2d.pdf);
its published results are not speed or accuracy measurements of GHOST.

Higher-order or pulse-basis formulations, graded material meshes and physical
thin-layer replacements could reduce unknown counts, but change approximation
or discretization decisions. They should not silently replace the user's
linear-basis, material-resolved geometry. Dense LU or the existing hierarchy can
still be faster when their RAM fits; this path targets memory-limited cases.

## Reproduction and evidence

From the repository root:

```powershell
.venv/Scripts/python.exe experiments/ghost_certification_20260910/public_probe.py --frequency 2 --certified
.venv/Scripts/python.exe experiments/ghost_certification_20260910/public_probe.py --frequency 2 --certified --factor dense
$env:GHOST_COMPRESSED_STORAGE_MIB = '8192'
.venv/Scripts/python.exe experiments/ghost_certification_20260910/public_probe.py --frequency 10 --certified
.venv/Scripts/python.exe experiments/ghost_certification_20260910/public_probe.py --materials --certified
.venv/Scripts/python.exe experiments/ghost_certification_20260910/validate_results.py
```

The probe sets two BLAS threads and four assembly threads. Benchmarks run
sequentially in fresh processes on the same Windows host. RSS is sampled every
50 ms; brief peaks can be missed and this is not a hard memory bound. Nested
stage timings must not be added together. The payload cap is not a process-RAM
cap; disk, RHS, geometry and construction workspace requirements are documented
in the selection guide.

`finish_validation.ps1` reproduces the final regression, native, legacy and
headless checks followed by the material certificates. `source-checks.json`
records Python 3.6 grammar checks for 21 changed Backend modules. New compressed
modules have no trailing-whitespace issues, and the repository's normal
`git diff --check` passes. Existing unrelated user changes were preserved.

Raw public results and logs live in this folder. `validated-results.json`
contains compact numerical checks, `final-source-hashes.json` identifies source,
and `certification.diff` compares Backend changes against the snapshot taken at
the start of this request. The untracked local baseline and earlier dense
reference fixtures must be present to reproduce their saved comparisons.
