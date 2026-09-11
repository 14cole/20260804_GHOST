# GRIM backend organization

`GRIM_Revised_2` is now `GRIM_Backend`. Its root contains only four runnable
entry points: `run_gui.py`, `run_diagnostics.py`, `run_headless.py`, and
`run_image_imprinter.py`. The nested `grim_backend` package was flattened into
categories. The module and folder guide is `GRIM_Backend/docs/BACKEND.md`.

## Locations

- `datasets/`: grid model, dataset operations, metadata, allocation checks, audits.
- `io/`: loaders, format adapters, batch exports, SS inspection; reference readers in `io/reference/`.
- `ui/`: application, dataset controls, dialogs, widgets, themes, and `assets/`.
- `plotting/` and `isar/`: plotting modes and ISAR processing/artifacts.
- `assembly/`: assembly models, editors, recipes, workflows, and workspaces.
- `integrations/`: GHOST and FREDDY discovery and embedding.
- `execution/` and `runs/`: diagnostics, background loading, remote jobs, Runs workspace.
- `reports/`: PowerPoint reporting, image imprinting, report workflows, and `templates/`.
- `scripting/`: public Python helpers, CLI, plot scripting, and action recording.
- `docs/`, `examples/`, `tests/`: documentation, runnable examples, and automated tests.

## Application wiring

Windows launchers, installed command entry points, source inventories, release
packaging, internal imports, GHOST viewer/Data Tools integration, generated
Python scripts, asset paths, templates, and documentation use the new locations.
The local editable installation was refreshed without installing dependencies.

Python scripts should use `GRIM_Backend.datasets.api`,
`GRIM_Backend.scripting.api`, and `GRIM_Backend.scripting.workspace`, or import
the specific service they need. External scripts using the former flat module
names need those imports updated. Existing Windows launchers keep their names.
`GRIM_BACKEND_PATH` is the preferred GHOST/Data Tools source override; the prior
environment variable names are accepted as fallbacks.

Diagnostics now finds the BoR native library in `bor/native/`. Namespace checks
in the Data Tools bridge validate actual package directories, including editable
installations, and reject mixing GRIM source trees.

Generated caches, obsolete package metadata, and the empty nested package
initializer were removed. Existing format reference sources were preserved in
`io/reference/`. All 146 mapped source/document/resource moves have existing
destinations. Solver numerical implementation was not changed.

## Verification

- Full GRIM suite: 1,057 tests, 1,056 passed and one skipped (73.5 s).
- GHOST regression coverage: 662 distinct cases, 661 passed and one skipped across
  the broad run, the isolated coated-sphere run, and focused rechecks. The broad
  run covered 661 cases in 795.9 s; its three stale fixture/path failures were
  corrected and passed in a four-case focused rerun (4.6 s). Numerical cases passed.
- GHOST Data Tools: 27 tests passed, including source-only imports and namespace checks.
- Development verification: startup and inventory checks passed; 52 smoke tests passed,
  including installed-wheel imports/editor behavior and release checks.
- Updated scripting/recorder and headless boundaries: 18 tests passed.
- All four run scripts and four examples import from an unrelated working directory
  with editable-install hooks disabled.
- Root layout, all moved destinations, documentation links, compilation, and
  Git whitespace checks passed.
- Local diagnostics reports READY with native BoR acceleration available.

Validation logs are stored beside this report. No commit or push was performed.

## Native error dialog during testing

The verbose test harness used `faulthandler.dump_traceback_later(60, repeat=True)`
to diagnose a long numerical check. Its process exited with `0xC0000005` while
printing a timed traceback. The user saw the Windows access-violation dialog.

`check_traceback.py --timed` reproduces the access violation using only Python
standard-library threading and faulthandler, with no application, solver, NumPy,
or native GHOST library imported. The runtime is Python 3.12.14, built August 25,
2026. Without timed dumps, the same thread-lifecycle stress completes normally.
The timed tracing hook was confined to the diagnostic command and is no longer
used. The application and normal verification runner do not enable it.

Fresh actual GRIM startup, the Qt event loop, embedded GHOST/FREDDY, and shutdown
passed. The interrupted broad run also revealed stale standalone GHOST test
bootstrap paths; those paths and moved GRIM test-fixture imports were corrected.
The 40 focused 2-D/Assembly/metadata/SENTRi interoperability checks passed.

The affected coated-PEC sphere numerical test passed independently in 161.3 s
without timed traceback dumps.

Final verification completed with no unresolved test failures. The GHOST fixture
fixes create the nested geometry test folder, put the package's parent on the
worker import path, and check the SLURM script against that same package parent.
The old open GRIM process must be restarted through `Launch_GRIM_GUI.bat` after
saving work, so it uses the renamed package.
