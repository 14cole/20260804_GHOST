# Solver geometry presets and Advanced Settings

The GHOST main form now contains solver, solver units, geometry preset,
frequency and azimuth selections, scattering mode, geometry/setup checking,
and output controls. Run, cancel, export, and progress remain accessible while
the settings form scrolls. Advanced Settings starts collapsed.

Advanced Settings contains the geometry-file override, discretization details,
accuracy target, precision, mesh certification, quality thresholds, CFIE alpha,
kernel evaluation, factorization, CPU/RAM/disk/batch controls, saved setups,
performance reports, and boundary densities.

Frequency and angular controls display the selected list or sweep input only.
The 2D angle labels read Azimuth; BoR labels read Aspect. Loading a saved 2D
setup from BoR also refreshes these labels and any bistatic observation input.
The underlying angle values and exported coordinate conventions are unchanged.

## Preset settings

| Preset | Kernels | Factorization | Sweep basis reuse |
| --- | --- | --- | --- |
| Small Geometry (No RAM Optimization) | Reference | Dense LU | Off |
| Large Geometry (RAM Optimization) | CPU streaming | Compressed assembly | Automatic |
| Balanced | CPU streaming | Dense LU | Automatic |

Large Geometry remains the new-run default. All presets select double precision,
up to four assembly threads and two BLAS threads, and 256 angles per batch.
Large Geometry uses an 8192 MiB compressed payload allowance. Small Geometry
disables optional compression while retaining allocation checks and bounded
workspaces. Balanced does not automatically choose compressed assembly.

Selecting a preset leaves the geometry, frequency/angle selections, accuracy,
mesh certification, quality thresholds, RAM admission budget, and temporary
directory unchanged. Other performance settings reset to the chosen preset.
Edited performance settings display a non-selectable Custom placeholder when
they no longer match a preset; the dropdown contains only the three presets.

Presets apply to 2D monostatic solves and are disabled during active work or in
BoR/bistatic modes. The same selector and advanced controls are used by GRIM
Runs. Runs also places cluster allocation and BoR body orientation under
Advanced Settings. Its connection, geometry list, bundle, and submission
workflow remain available in their existing sections.

Saved setups and Runs preferences store actual numerical/resource settings.
Preset matching derives from those values without replacing them or changing
the saved-setup schema. Explicit launch-environment overrides still apply.

## Validation

| Check | Result |
| --- | --- |
| Full GRIM suite | 1057 tests run; 1056 passed, 1 skipped |
| Targeted GHOST GUI, execution, package, and headless driver suites | 111 passed |
| Python 3.6 with NumPy 1.14 / SciPy 1.0 execution and preset checks | 15 passed |
| Source and wheel inventory | Passed |
| GHOST Python source encoding | 211 files, all ASCII |
| Manual offscreen layout inspection | Collapsed, expanded, compact sweep, and Runs forms checked |

Physics comparisons exercised all three presets on PEC and mixed-material
fixtures with mesh certification, both polarizations, and 361 incident angles.
Relative maximum complex-field differences against Small Geometry were below
1e-10. GUI checks cover atomic preset transitions, custom settings, saved setup
exchange, preferences, exact RAM-budget retention, invalid temporary paths,
busy controls, BoR/bistatic transitions, and 0-360 degree sweeps in 1 degree steps.

Logs: `grim-tests.log`, `ghost-tests.log`, and `python36-tests.log`.
Screenshots: `ghost-main.png`, `ghost-advanced.png`,
`ghost-sweep-compact.png`, and `runs-main.png`.
`capture_ui.py` reproduces the screenshots.

The user-facing settings guide is `tools/GHOST/RUN_PROFILES.md`.
