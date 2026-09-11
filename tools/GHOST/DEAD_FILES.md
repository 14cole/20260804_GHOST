# GHOST source cleanup

The cleanup checked active application imports, dynamic dispatch, command
invocations, tests, native-library lookup, and release inventories.

## Removed

- `cpu_checked_lu.py`: unused adapter around the active dense factorization.
- `cpu_streaming.py`: re-exported the same four functions supplied to CPU
  dispatch. Calls now use those functions directly; CPU execution state and
  resource management remain in `execution/cpu.py`.
- Root import wrappers `bor_dispatch.py`, `bor_solver.py`, `feature_sum.py`,
  `feature_workflow.py`, `geometry_io.py`, `grim_compat.py`, `grim_io.py`,
  `rcs_solver.py`, and `workflow_provenance.py`: callers use their categorized
  implementation modules listed in [IMPORTS.md](IMPORTS.md).
- `hpc_bundle.py`: its command runs directly from `hpc/bundle.py`.
- The empty root `__init__.py`: package discovery uses a namespace package.

## Categorized support

The native BoR source, build script, and platform library are active and live
in `bor/native/`. The Python 3.6 dataclasses fallback, its license, and path
helpers live in `execution/`. Assembly commands live in `assembly/`, reference
import in `io/`, validation commands in `validation/`, and environment checks
in `hpc/`. Only five `run_*.py` files remain at the package root.

No additional dead source was confirmed by this cleanup. Existing geometry,
dataset, result, and experiment files were preserved. Generated `__pycache__`
folders contain disposable bytecode; Python recreates them as needed.
