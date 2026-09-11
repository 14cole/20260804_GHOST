# GHOST root cleanup

The package root contains exactly these files:

- `run_gui.py`
- `run_local_monostatic.py`
- `run_local_bor.py`
- `run_hpc_monostatic.py`
- `run_hpc_bor_monostatic.py`

Native C, its build script, and the platform library moved to `bor/native`.
The dataclasses fallback, license, and path helpers moved to `execution`.
Assembly commands moved to `assembly`; reference import to `io`; validation
commands to `validation`; environment checking to `hpc`.

The unused checked-LU adapter was deleted. CPU calls now invoke the same four
implementation functions directly, eliminating the re-export module and its
dispatch indirection. Nine redundant import wrappers were removed after their
active callers migrated to package imports. The HPC bundle command executes
`hpc/bundle.py` directly. Python namespace-package discovery eliminates the
empty root initializer while preserving Python 3.6 support. Discovery rejects
namespace paths that include another backend directory.

The exact file mapping is in `categorized-moves.json`. Guides, source/release
inventories, the Windows launcher, native lookup, and GRIM integration use the
new paths. Existing datasets, geometries, results, and experiment sources were
preserved. Obsolete root bytecode was removed.

## Validation

- Full GRIM: 1057 tests, 1056 passed and one skip.
- GHOST regression group: 284 tests, one skip and one stale bundle-command
  assertion. After correcting the assertion, all 32 tests in that module passed.
  The other tests in the original group passed, including CPU solver equivalence,
  matrix-pipeline checks, material cases, and command/import coverage.
- Python 3.6 / NumPy 1.14 / SciPy 1.0: 29 tests, 28 passed and one desktop skip.
- Data Tools: 25 passed after correcting its test import path.
- Standalone local drivers: 30 passed; standalone HPC scheduling: zero failures.
- Native sampler rebuilt into an experiment output folder and load-checked with
  OpenMP enabled; the installed sampler loads from `bor/native` as `native_c`.
- Source/release inventories, ASCII source checks, and whitespace checks passed.
- All local GHOST Markdown links resolve. Of the 48 preexisting numerical source
  modules, 46 retain identical parsed syntax; changes to `twod/solver.py` remove
  the forwarding calls, and `bor/streaming.py` changes native artifact lookup.

Evidence logs use the `categories-` prefix. Workers ran locally without SLURM
submission. No commit or push was performed.
