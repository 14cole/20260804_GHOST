# Solver and Assembly implementation record

Requested scope: the seven recommendations from the September 2026 review.
The subsequent request explicitly defers IBC on dielectric interfaces.
Existing UI refactoring and user datasets must be preserved.

1. Explicit material models and a transmitting thin dielectric layer, with
   finite-layer numerical references and documented approximation limits.
2. BoR free-sheet support and reactive-IBC robustness. IBC on dielectric
   interfaces/backings is deferred by the user.
3. Reliable native BoR acceleration and measured solve-stage telemetry.
4. Accuracy-driven workflow and targeted mesh refinement.
5. Assembly per-feature complex interference inspection and bounded caching.
6. Feature-family validation and applicability evidence for corners,
   terminations, curvature, and nearby features. No unvalidated response data
   may be presented as full-wave truth or mutual-coupling support.
7. A validated opt-in path to reduce large-system computational cost, retaining
   dense LU as the reference.

## Implementation status

| Recommendation | Delivered | Limits / remaining evidence |
| --- | --- | --- |
| 1. Materials | Friendly boundary names, material guide, thin-layer editor and 2D mono/bistatic solver. Normal-polarization terms retained. | Uniform isotropic thin layer in air, all-layer geometry; no BoR thin dielectric or dielectric-surface IBC. Approximation error is not certified by mesh convergence. |
| 2. BoR materials | Transmitting electric impedance sheets, joined sheet/PEC meridians, and uniform reactive-IBC CFIE. | One connected meridian. Sheet plus opaque IBC and nonuniform reactive CFIE are rejected. Independent sphere references cover the new pure-sheet and uniform-IBC routes; joined sheet/PEC does not yet have its own independent reference. |
| 3. Performance evidence | Portable Windows native kernel build and fresh-process load check; wall time, stage timings, sampled process RSS and backend recorded. | RSS is process-wide and sampled; inclusive timings can overlap. |
| 4. Accuracy workflow | Standard/Tight mesh targets, accuracy/performance report, corner/junction/open-end selection and selected-segment refinement. | Guidance is geometric, not an a posteriori error estimator or automatic adaptive mesher. |
| 5. Assembly inspection | Per-feature complex contributions, phase, cross terms, removal effect, gain/phase previews, phasor plot and bounded cache. Raw and power/phase fields share Assembly normalization and aliases; independent HV/VH are retained. | Fixed geometry and illumination; no new mutual-coupling model. Exact stored radar samples and VV/HH plus a VH/HV body channel are required. |
| 6. Feature families | Thirteen study definitions, three-level reference-convergence checks, complex clean/total/delta comparisons, artifact hashes and GUI/CLI checker. | **0/13 new family cases validated. Independent 3D full-wave reference artifacts are unavailable.** Templates and manufactured test fixtures are not physics evidence. |
| 7. Computation savings | Opt-in 2D CPU single-precision LU factors with double-precision residual refinement and automatic double-LU fallback. Double remains default. | Matrix assembly still uses double precision and quadratic memory. No FMM, compressed-matrix or higher-order-basis solver added. |

The implementation for all seven areas is present. Recommendation 6's physical
applicability evidence remains pending external reference data. IBC on dielectric
interfaces/backings remains deferred as requested.

## September 5 audit corrections

- Thin-layer assembly now joins geometric nodes across named segments and
  consistently orients each connected component before curvature, incident
  loads and signed operators. The 96-element, 100 mm strip reproduction at
  1 GHz changed from a 6.48% field discrepancy between one and 24 groups to
  less than 3e-13 relative complex difference. Its Tight-certified results
  now agree within 1e-12. Tests also cover reversed segments, curved loops,
  both polarizations, bistatic fields, genuine branching rejection and the
  independent finite-annulus reference. These are discretization corrections;
  they do not certify the thin-layer approximation for arbitrary geometry.
- Inspector samples use the same field decoder, dimensional-unit checks and
  channel aliases as Assembly. Raw arrays and reconstructed power/phase fields
  match actual Assembly output, including distinct measured VH and HV. Reading
  numerical field gaps is bounded to 256 KiB per chunk and remains cancellable.
- The Python package now includes the placement editor. An automated test
  builds and installs a wheel offline, disables editable import hooks, and
  opens, duplicates and undoes point placements using that installed artifact.
- FREDDY caches complete scores across initial search and local refinement.
  The audited two-design case now performs two physical score evaluations,
  reusing twelve requests, with unchanged candidates and plot samples. Cache
  scope is one run, and the report distinguishes requests from new evaluations.
  Coarse discrete refinement remains a limitation of the simplex method.
- Metadata-policy tests now verify successful arithmetic/exports, source
  declarations and advisories. Synthetic-workspace diagnostic tests isolate
  previously imported backend modules; deliberate stale-origin failures are
  still detected. The application retains advisory convention metadata.

Verification: the complete GRIM suite ran 1,022 tests and the complete FREDDY
suite ran 84 tests, each with one skip and no failures. An additional focused
run of 102 thin-layer, inspector, Assembly workflow, production-contract,
equivalence and SENTRi tests passed. Actual Windows startup diagnostics report
READY and load the native BoR library. GUI verification used offscreen Qt.

The broader 73-test solver/material selection passed 72 tests on its first run;
one HPC test correctly stopped after a source file was edited during its run.
All five HPC integration tests then passed against unchanged sources. The
workflow checks also passed after replacing two diagnostic degree symbols
with ASCII text; the GHOST source-format check is clean. Detailed logs and
before/after reproduction results are in the sibling
`self-audit-2026-09-05/` directory under `fix-*` filenames.

No independent reference datasets were added for the 13 feature-family cases
or the joined sheet/PEC configuration. No large-platform, cluster or CUDA
performance claim follows from these fixes.

## September 4 reference measurements

- At 1 GHz, an 80-segment circular midsurface of radius 80 mm, thickness 1 mm,
  epsilon 3 - j0.02 and mu 1 was compared with an explicit annulus (160 segments).
  Both used 13 monostatic aspects, both polarizations and condition estimates.
  Thin-layer runtime was 0.36 s versus 4.92 s for bulk geometry in one local run
  (about 13.5x). Maximum complex-amplitude error versus the independent finite
  annular Bessel solution was 0.089% for the thin layer; maximum power error was
  0.0078 dB. These are case-specific measurements, not a speed or accuracy guarantee.
- A 100 mm open strip, thickness 0.5 mm, epsilon 3 - j0.02 at 1 GHz agreed within
  0.067 dB with a refined finite-thickness 2D bulk strip at 12, 48 and 86 degrees.
  This is a numerical finite-bulk comparison, not an analytic open-strip reference.
- A 512 by 512 TE thin-layer matrix with four RHSs used 2 MiB for mixed LU factors
  versus 4 MiB for double factors; the original 4 MiB matrix remains necessary.
  Relative solution error was 7e-14 with one correction. Factor/RHS timings are
  recorded by `tools/GHOST/tests/benchmark_refined_lu.py`; they are not end-to-end
  performance measurements.
- New BoR material tests compare transmitting sheets to independent radial
  boundary matching and uniform reactive IBC to Mie theory, including ka=4.4934.
  A separate 12-case, 60-element sphere probe spanning ka=1.5, 2.7437, 4.4934
  and four impedances stayed within 0.033 dB.
- Regression runs: 105 2D/Assembly/material/memory tests passed; 78 BoR/audit/
  backend tests passed; 223 GUI tests ran with one skip and no failures.
  Focused material, numerical, validation and UI-state tests also passed.
  Those focused runs did not establish a clean full-suite baseline; the full
  GRIM metadata-policy and diagnostic test failures were corrected and verified
  in the September 5 audit work above.
- Final GUI regression: 134 tests passed after the inspector's palette, sizing
  and stale-plan guards. Neutral Dark and Light screenshots were inspected at
  1366 by 768. The example strip also passed the Tight certified mesh target
  with mixed LU in both polarizations: maximum normalized complex change 0.182%,
  maximum residual 2.85e-14. This remains a numerical, not model-error, certificate.

See [SOLVER_UPDATES.md](SOLVER_UPDATES.md) for the controls, example and scope.
Raw measurement scripts, JSON and test logs are in the workspace's sibling
`critique-review/` directory; no user datasets were overwritten.
