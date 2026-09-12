**GRIM / GHOST / FREDDY project audit — 12 September 2026**

The application starts successfully, all seven top-level tabs load, and the principal calculation and export workflows exercised here work. I found **three reproducible application defects, one failing acceptance-test assertion, and two documentation errors**. The current full acceptance gate is therefore failing. There is also an explicitly documented gap in independent physical validation for 13 feature-family cases.

The most consequential application issues are an RF agreement score that changes substantially solely because both inputs become weaker, and an impedance output path that can overwrite its own material input after a generic replacement confirmation. I recommend addressing those before using this revision for decisions based on weak-pattern agreement or distributing it to users.

**Scope and evidence**

I audited the active **grim-integrated** working tree, including the host, bundled GHOST and FREDDY implementations, test suites, packaging/verification code, documentation, and interfaces between the tools. The parent directory contains older experiments and source snapshots; those were not treated as additional live applications.

The underlying commit is **2c53ead5834ab9514eaf407a8a543a40c4ac7ed9**, with substantial existing uncommitted changes. This report describes that working tree, not just the commit. A [source hash manifest](source-snapshot.json) records 514 source/configuration/documentation files, and [source status](source-status.txt) records the working-tree state. I made no production-code changes. This repository copy contains the audit scripts, logs, source hashes and screenshots. The scripts regenerate synthetic inputs and output files when run.

The environment was Windows with CPython 3.12.14, NumPy 2.5.2, PySide6 6.11.2, Matplotlib 3.11.1 and SciPy 1.18.1. Required startup checks passed, the native BoR acceleration library loaded, and PowerPoint automation was available. See [startup diagnostics](01-/output.log).

I instantiated the real Qt application offscreen, navigated its tabs and subordinate pages, inspected rendered screenshots, and exercised the actual UI handlers and background workers with controlled inputs. This is programmatic GUI testing; it is not a claim of manual mouse-and-keyboard testing on every display configuration. PowerPoint export was additionally run through desktop PowerPoint, reopened and rendered for visual inspection.

**Confirmed findings, in priority order**

**1. High — RF agreement mis-scores weak but closely matching patterns.**

Location: [_lin_concordance](SOURCE_CONTEXT.md#rf-agreement), particularly the numerical threshold at line 87.

Reproduction uses 30 samples with x = 0…29, A = 1 + sin²(x/5), and B = 1.01 × A. With the original powers, the reported RF agreement is **99.6646766** and linear concordance is **0.9990077**. Multiplying both power arrays by 10⁻⁸ produces an agreement score of **54.7634398** and linear concordance of **0**. The same defect occurs at factors 10⁻¹⁰ and 10⁻¹². Scaling by 10⁻⁴ remains correct.

The relative error, pattern shape and sampling are unchanged. The absolute floor introduced by max(mean(A²) + mean(B²), 1.0) treats low-power variation as a degenerate constant case. This can sharply depress agreement for weak-scattering patterns and invalidate comparisons across absolute power levels.

Recommended correction: normalize the inputs before the concordance calculation, or use a numerically stable relative degeneracy criterion without this absolute unit-sized floor. Preserve the documented behavior for equal and unequal constant traces, and explicitly test zero, weak, ordinary and large powers.

Evidence: [independent regression and positive control](test_audit_regressions.py#L27), [failure output](audit-regressions.log). Identical weak traces still score 100; the defect requires a small nonzero difference. This is not a blanket failure of all weak inputs.

**2. High — FREDDY can replace an active material source with impedance output.**

Locations: [_confirm_output_replacements](SOURCE_CONTEXT.md#output-replacement) and [_compute_impedance](SOURCE_CONTEXT.md#impedance-worker).

Reproduction: load a five-column material CSV as a layer, enter that same CSV path as the impedance output, and accept the generic “Replace Existing Impedance Output?” prompt. Run a 9–11 GHz sweep on a 0.12-inch layer with ε = 4 − 0.4j and μ = 1. The worker reads the material and then successfully overwrites it with the three-column impedance result.

The source header changes from **frequency_hz,eps_real,eps_imag,mu_real,mu_imag** to **frequency_hz,resistance_ohm,reactance_ohm**. The original material data are lost, and the file is no longer valid as that layer’s material input.

There is an overwrite confirmation; this is not a silent write. The defect is that an active input is treated as an ordinary replaceable output without identifying the conflict. The converter already has stronger source/destination protection, so protection is inconsistent across FREDDY workflows.

Recommended correction: reject output paths that refer to active material inputs, including normalized aliases and existing-file identity matches. Apply the check to primary and anisotropic material inputs, planned sidecar/batch outputs, and material-mix inputs where relevant. Continue allowing explicitly confirmed replacement of ordinary result files.

Evidence: [real-worker regression](test_audit_regressions.py#L72) and [before/after data in the failure log](audit-regressions.log#L10). Only disposable audit fixtures were overwritten.

**3. Medium — mapped table import rejects partially missing phase.**

Locations: [GRIM table mapping](SOURCE_CONTEXT.md#phase-mapping) and [shared unit conversion](SOURCE_CONTEXT.md#unit-conversion).

Reproduction: import a CSV with frequency_GHz, azimuth_deg, elevation_deg, magnitude and phase_deg columns. Supply one row with phase 90 and a second row with an empty phase cell. Choose “RCS power (m²)” and keep Phase mapped.

The actual loader routes the file into the mapping dialog, but submitting it fails with **Line 3, phase: invalid number ''.** No dataset is imported. The backend supports this data correctly: the known phase can remain π/2 radians and the missing value can remain NaN. Unchecking Phase permits a power-only import but loses the phase that was measured.

The dialog sends optional phase through mandatory finite-number unit conversion before the backend can preserve its missing value.

Recommended correction: allow blank values specifically for optional phase during conversion, preserving them as unknown. Keep strict checks for required coordinates/power and malformed nonblank phase values. Test mixed known/missing phase through the real dialog, not only the grid constructor.

Evidence: [loader-to-dialog regression and direct-backend control](test_audit_regressions.py#L46), [failure log](audit-regressions.log#L26).

**4. Medium — a spelling-sensitive cancellation assertion fails the full acceptance gate.**

Location: [GHOST cancellation test](../../tools/GHOST/ghost_backend/tests/test_gui_entrypoint.py#L234).

The test passes **“Solve cancelled by user.”** into the cancellation handler and then asserts that the displayed message contains **“canceled”**. The handler preserves the supplied message, so this test fails consistently.

The state assertions before that line pass. I also separately checked both spellings and an empty-message fallback: all three clear running state and pending context, reset progress, restore Run/Cancel controls, preserve the previous result, and show no critical dialog. See [independent cancellation evidence](material-results.json).

This is a test defect, not evidence that solver cancellation is broken. Nevertheless, it blocks the shared development/release acceptance gate. Correct the fixture/assertion to test cancellation semantics, or compare the intended exact message consistently; retain all cleanup assertions.

Evidence: [complete GHOST suite output](06-/output.log#L722). The full gate must be rerun after a correction; it should not be bypassed because the failure is small.

**5. Low — the documented GHOST test command uses the old directory layout.**

Location: [root README development checks](../../README.md#L382).

The instructions change into tools/GHOST and run unittest discovery against tests. The tests now live in tools/GHOST/ghost_backend/tests. Executing the documented command fails with **ImportError: Start directory is not importable: 'tests'**.

Recommended correction: prefer the repository-root verifier below. If retaining individual commands, change into ghost_backend and update the subsequent relative path to FREDDY as well.

    .venv\Scripts\python.exe verify_project.py --mode full

The verifier’s own suite inventory already uses the correct location. [Reproduction log](documented-ghost-test-command.log).

**6. Low — FREDDY’s shared material-format documentation link is broken.**

Location: [FREDDY README](../../tools/FREDDY/README.md#L160).

The “shared file format” link targets ../../MATERIAL_CSV_FORMAT.md, which does not exist. The intended document is [tools/GHOST/MATERIAL_CSV_FORMAT.md](../../tools/GHOST/MATERIAL_CSV_FORMAT.md), reachable from FREDDY’s README as ../GHOST/MATERIAL_CSV_FORMAT.md.

A scan of 45 Markdown files found this missing local inline-link target. This check does not validate external URLs or every path written inside a code block. [Link and syntax results](source-and-doc-checks.json).

**What worked across the tabs and workflows**

“Additional probe” below means a separate audit exercise of real computation or UI behavior. “Existing suite” means the repository’s relevant tests ran as part of the full suite; it does not imply a second independently authored end-to-end scenario.

| Area | Investigation and observed result |
|---|---|
| Startup and top-level navigation | Plotting, ISAR, FREDDY, GHOST, Assembly, PPT and Python all loaded; no embedded-tool initialization errors or uncaught Qt callbacks in the navigation inventory. |
| Plotting and native data loading | Actual background loading of two synthetic GRIM grids passed. Rectangular/polar azimuth, frequency, elevation, waterfall, RF Compare and Delta Map rendered. Known levels were 0 dB and 6.0206 dB; the selected subtraction produced −6.0206 dB. RF Compare has finding 1. |
| Dataset editing/operations | Existing GUI/backend suites passed cases for selection-dependent actions, axis sorting and sample alignment, invalid/duplicate edits, join/overlap policies, crop/regrid unit conversion, wrap intervals, decimation, statistics, output and recorder behavior. The mapped-import exception is finding 3. |
| ISAR formation | Actual background processing formed a finite 256×256 image from a controlled phase ramp. The range peak matched the expected sign/location for that input. Settings controls were within the settled 1280×720 window. |
| ISAR artifacts and comparison | Complex image save/reopen round-trip passed. Self-comparison gave zero difference; doubling complex field gave 6.020599 dB over bright pixels. Profiles and ROI controls were exercised. |
| FREDDY Impedance | Real worker produced the three-column export and in-app results for a lossy constant-property layer. Source/output collision is finding 2. |
| FREDDY IBC Batch | Two requested thicknesses produced separate .1-inch and .2-inch CSV outputs. |
| FREDDY Off Angle / Thickness | Real sweeps generated results and CSV outputs. Off-angle exercise used TE/TM calculations; thickness exercise covered two thickness values. |
| FREDDY Inverse Design | A two-combination exhaustive search produced two candidates and a completed checkpoint with next = total = 2. Resume was disabled for the completed search. Existing suites additionally cover bounded search, checkpoints and inverse requirements. |
| FREDDY Material Mix | Equal-volume linear mixing of ε = 2 − 0.1j and 6 − 0.1j yielded ε = 4 − 0.1j at 9, 10 and 11 GHz. Exported material reopened with matching values. Other effective-medium models remain subject to their physical assumptions. |
| FREDDY Sensitivity & Yield | Five sensitivity points plus 16 statistical trials completed, with 21 evaluations recorded. All five result views rendered. This small reproducible run verifies execution; it is not a statistical convergence study. |
| FREDDY Material Explorer | Two valid materials loaded; a duplicate was recognized and a malformed file was rejected while retaining the valid files. In-range interpolation matched expected values; out-of-range queries returned no sample. Source files remained unchanged. |
| FREDDY About & Guide / project behavior | Guide page inspected. Existing tests passed project portability, unsaved-close decisions, background-job close blocking, schema/unit and source-validation cases. Documentation issue 6 remains. |
| GHOST Geometry / Solver | Loaded an authored square geometry, solved a small real 2-D problem, exported GRIM and received it through the host loader. Cancellation cleanup passed separately; existing acceptance assertion fails as described in finding 4. |
| GHOST numerical variants | Existing tests passed analytic cylinder/sphere comparisons, dielectric/coated/sheet cases, phase and energy checks, near-singular quadrature, compression equivalence, mixed-precision fallback and memory/cancellation gates. One thin-coating sphere comparison reported a maximum 0.0263 dB error for its specific case. |
| Assembly | Body, point/line authoring, review, 3-D, response and interference pages were inspected. A separate body-only service run preserved the input power and phase exactly. Existing point/line and curved/faceted-body tests passed. Independent full-wave coverage remains limited as explained below. |
| PPT | Real preview passed. Actual PowerPoint export created a one-slide 16:9 deck; ZIP integrity passed; desktop PowerPoint reopened it and rendered the six expected plots with correct synthetic signal levels. |
| Python | The generated 87-line workflow script ran in a separate Python process with exit code 0. A layout warning was emitted; it did not prevent completion. |
| HPC/local drivers | Standalone scheduler integration and local-driver integration passed, including restart/attestation and result collection checks. These were local tests; no real cluster job was submitted. |
| Packaging and source integrity | Startup/source inventories and offline wheelhouse tests passed. Existing wheel-install/release-builder tests ran. All 442 inventoried Python files parsed successfully. No new distributable release was built. |

Selected evidence: [workflow results](workflow-results.json), [extended results](extended-results.json), [material and cancellation results](material-results.json), [UI inventory](ui-inventory.json), [PowerPoint render](screens/powerpoint-export-1.png).

**Test validity and exact results**

I ran all 11 entries from the project’s full development/acceptance inventory. The audit runner continued after a failing suite so that later checks were still completed.

| Existing unittest suite | Tests reported | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| UTF-8 cleaner | 10 | 9 | 0 | 1 |
| Offline wheelhouse | 13 | 13 | 0 | 0 |
| GRIM | 1,032 | 1,031 | 0 | 1 |
| GHOST | 713 | 711 | 1 | 1 |
| GHOST CEM tools | 27 | 27 | 0 | 0 |
| FREDDY | 197 | 196 | 0 | 1 |
| **Total** | **1,992** | **1,987** | **1** | **4** |

Startup, source inventory, standalone HPC scheduling, local-driver integration and ASCII-transfer checks are additional command results, not extra unittest counts. Ten of the 11 suite commands exited successfully. GHOST took approximately 17 minutes 40 seconds; it completed rather than timing out. [Full command/results inventory](acceptance-results.json).

The four skips concern directory symlink creation, final-file symlink resolution, POSIX directory fsync and case-distinct files on a case-insensitive filesystem. These are unverified platform-specific branches, not passes.

The independent workflow set contains **25 passing probes**. Separately, the four expected-behavior regression tests have **one passing positive control and three failing test methods**. Unittest reports five failures because the RF-scale method has three failing subtests. These failures are deliberately retained to demonstrate the application defects; they are not marked expected failures or rewritten to accept current behavior.

There is useful numerical substance in the existing tests: closed-form resistive-sheet and quarter-wave checks, independent cylinder/sphere references, phase/energy checks, malformed input rejection, cancellation atomicity and dense-versus-compressed comparisons. The suite also includes mocked UI dispatch and source-structure assertions; those verify wiring/contracts, not physical accuracy on their own.

The new probes add independent expectations: analytical dB ratios, common-scale invariance, a controlled complex phase ramp, identity assembly, equal-volume mixing, missing-data preservation, and source-file preservation. Computations were not replaced with predetermined successful results. File/confirmation dialogs were controlled for reproducibility and audit-owned outputs. Worker timeouts and unexpected Qt callbacks fail the probes; audit runners return nonzero on failures.

The gaps behind the new defects are concrete: RF tests checked identical curves and gain/local errors but missed very weak power scales; the importer tested missing phase at the backend without exercising mixed blank phase through unit conversion; overwrite tests did not distinguish active inputs from replaceable results. There is no measured line/branch coverage percentage in this report.

**Completeness and usability improvements**

The largest outstanding scientific-validation item is already disclosed in the repository: **0 of 13 new corner, termination, curvature and feature-pair cases have external full-wave validation artifacts**. The templates contain no field results. Existing analytic, BoR and manufactured checks support their tested cases; they do not establish a broad real-world feature/coupling accuracy envelope.

The prescribed next step is independent clean/featured reference data at three mesh levels, matching complex-field conventions, followed by the existing convergence and reconstruction checks. Preserve the current distinction between “awaiting reference artifacts” and “validated.” See [feature-family study requirements](../../tools/GHOST/ghost_backend/validation/feature_family_studies/README.md#L3).

At 1280×720, Material Mix’s default presentation is crowded: nested horizontal/vertical scrolling hides portions of explanatory text, and the lower charts can be compressed enough for labels and panels to overlap. A separate, adequately sized results area and fewer nested scrolling regions would improve usability. This is a layout observation in the tested Qt rendering environment, not a claim that the controls cannot be reached at every window size. [Screenshot](screens/workflow-material-mix.png).

Assembly’s initial 3-D view could benefit from fewer ticks or more spacing at small sizes. Inverse-design action buttons require scrolling in the compact layout. Validate these layouts with native 100%, 125%, 150% and 200% display scaling, keyboard traversal and screen-reader use before making accessibility claims.

A suspected ISAR header-clipping issue during initial rendering did not persist after layout settled; the independent bounds check passed. I have not counted it as a product defect.

Release readiness is conditional. Verification/packaging tests passed their exercised cases, but the current acceptance failure and existing dirty source state remain barriers to a verified release. A real clean release build/install, Linux/cluster execution, unavailable platform-specific file semantics, production-size datasets and long-run memory/performance behavior were not established by this audit.

**Reproducing and acting on the findings**

Run these from the repository root using its existing interpreter:

    .\.venv\Scripts\python.exe .\experiments\project_audit_20260912\run_acceptance.py
    .\.venv\Scripts\python.exe .\experiments\project_audit_20260912\test_audit_regressions.py
    .\.venv\Scripts\python.exe .\experiments\project_audit_20260912\probe_workflows.py
    .\.venv\Scripts\python.exe .\experiments\project_audit_20260912\probe_extended.py
    .\.venv\Scripts\python.exe .\experiments\project_audit_20260912\probe_materials.py

The first two currently return nonzero for the reasons documented above. The PowerPoint probe additionally requires desktop PowerPoint automation access. These scripts write only their audit outputs and disposable fixtures; reruns replace previous audit results.

Recommended order: protect active material sources and correct low-power concordance; preserve partial phase in the importer; repair the cancellation assertion and documentation; rerun the complete acceptance inventory and the independent regressions; then address compact-layout usability and obtain the missing external validation data. This audit leaves the production implementation unchanged so each correction can be reviewed separately.
