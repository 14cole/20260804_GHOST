# GRIM backend organization

Implemented a `GRIM_Revised_2/grim_backend` package with dataset operations,
file I/O, Python recording/plotting, and a command-line entry point.

The [backend guide](../../GRIM_Revised_2/grim_backend/README.md) maps tasks to
files and entry points.

## Implementation

- Split the 8,338-line dataset module into grid storage/access, axes,
  coordinates, arithmetic, calibration, combination, metadata, memory, and
  audit modules. `datasets/grid.py` defines the public `RcsGrid` class and
  inherits the operation and format methods.
- Replaced `grim_legacy_io.py` with format-named OUT, PTM, and Xpatch modules.
  PTM and Xpatch each contain their binary parser and grid adapter together.
- Grouped native GRIM, CSV, CST, SENTRi, Pioneer, OUT, PTM, and Xpatch handling
  under `io/`. File/folder dispatch is in `io/loaders.py`; batch save/export
  helpers are in `io/batch.py`.
- Moved data transformations out of `grim_python.py` into
  `datasets/transforms.py`. Python recording and plot rendering have separate
  files under `scripting/`.
- Kept `grim_dataset`, `grim_headless`, and `grim_python` as public import
  modules for saved scripts. Updated application callers and examples to use
  the package paths. Both the existing headless invocation and installed
  command remain supported.
- Removed source comments from the reorganized backend and rewrote narrative
  docstrings to describe behavior, inputs, outputs, validation, and units.
  Serialized metadata identifiers, including PTM convention identifiers, retain
  their values.
- Updated wheel packaging, the release inventory, startup diagnostics, and the
  CEM Tools bridge. The backend guide is included in the wheel.
- Added the 22 GHOST backend dependencies missing from the existing release
  inventory. The baseline suite identified this omission before the GRIM move.

## Validation

| Check | Result |
| --- | --- |
| Full GRIM suite | 1,051 passed, 1 skipped; 68.289 seconds |
| CEM Tools suite | 25 passed |
| GHOST interoperability/integration checks | 50 passed |
| Installed wheel | Every backend module imported from the isolated installation; public APIs, native save/load, and GUI editor exercised |
| Backend without Qt | Every backend module imported with Qt imports rejected |
| Startup diagnostics | Exit code 0 |
| Source and wheel inventories | Complete |
| Syntax, unresolved global references, source comments, whitespace | Passed |
| `git diff --check` | Passed |

The skipped GRIM test requires Windows file-symlink privileges unavailable in
this environment. The final source-only caller formatting changed import line
wrapping after the full suite; imports and inventories were checked afterward.

AST comparison accounted for all 335 moved functions. After excluding imports
and docstrings and normalizing the combined parser references, 334 function
bodies matched. The remaining function, `load_ss`, changes only two diagnostic
strings to point to `python -m grim_backend.io.xpatch`. The recorder's source
directory lookup was adjusted for its package location. Numerical algorithms,
array transformations, and file schemas retain their behavior.

Evidence is in `final-grim-tests.log`, `cem-tests.log`,
`ghost-integration-tests.log`, `startup-diagnostics.log`,
`function-comparison.json`, and `final-source-hashes.json`.

Restart GRIM to load the reorganized modules.
