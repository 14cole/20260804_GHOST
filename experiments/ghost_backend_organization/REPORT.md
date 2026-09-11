# GHOST backend organization and removal audit

## Result

Moved 68 implementation modules into the `ghost_backend` package. The Backend
root now contains 25 Python files, down from 82. The package contains 84 Python
files, including package initializers and the backend-root helper.

- [Source guide](../../tools/GHOST/Backend/README.md)
- [Import relocation map](../../tools/GHOST/Backend/IMPORTS.md)
- [File-removal findings](../../tools/GHOST/Backend/DEAD_FILES.md)

The package groups 2-D solving, BoR solving, formulations, matrix assembly,
linear algebra, compressed storage, geometry, feature assembly, data I/O,
execution, run setup, HPC management, analytic validation, and desktop UI.
Public solver/workflow imports and command filenames retain their root entry
points. Editable driver settings and adjacent configuration files retain their
locations. Comments and docstrings were cleaned of development narratives;
functional configuration and interface documentation remains.

## Integration behavior

GRIM loads package implementations and checks the origin of both public entry
modules and package modules. CEM Tools resolves filename/pairing operations from
the package and rejects a process mixing backend directories. Release source
inventories include nested Python modules and the backend guides.

Native BoR lookup, configured HPC subprocesses, copied drivers, and the GRIM
viewer bridge resolve paths from the Backend root. On this host the native
sampler reports `native_c`.

Source fingerprints enumerate package files recursively under distinct relative
paths. Two files called `solver.py` in different packages therefore contribute
separate records. Editing a nested implementation invalidates recorded source
identity. Existing outputs remain readable; new runs use the new source
fingerprint, and resume checks reject mismatched source trees.

## Numerical equivalence

The AST comparison against pre-move source snapshots found 1,459 unchanged
function bodies after excluding imports and docstrings. Seven bodies differ:
three native-library/fallback path helpers, the GRIM path helper, two source
inventory functions, and the HPC stage subprocess path helper. No relocated
numerical function body changed. See [comparison](function-comparison.json).

This is an organization change; no new solve-time or RAM improvement is claimed.

## Validation

| Check | Result |
| --- | --- |
| Full GHOST unittest discovery | 639 tests: 637 passed, 1 skipped, 1 source-integrity failure caused by editing backend docstrings during a worker test |
| Affected HPC solver-options group, after edits finished | All 5 passed, including the BoR worker that detected the concurrent edits |
| Additional package relocation checks | 2 passed: copied standalone Backend with Qt blocked and unused LU adapter removed from the copy; nested source fingerprint invalidation |
| Full GRIM suite | 1,052 tests, OK with 1 skip |
| Release builder after adding guide inventory | 31 passed |
| CEM operations and UI progress | 25 passed |
| FREDDY suite | 151 exercised; two test-fixture keys were corrected, then all 18 tests in the affected portability/readiness groups passed; 1 optional skip |
| Standalone local driver integration | 30 passed, 0 failed |
| Standalone HPC scheduling integration | 0 failures; submission, workers, balancing, restart, and output attestations exercised |
| Python 3.6 / NumPy 1.14 / SciPy 1.0 | 49 solver/runtime tests plus 2 package deployment tests passed |
| GUI import check, source/release inventories, ASCII checks, diff whitespace | Passed |

The GHOST source-integrity failure reported exactly the package files being
edited and refused the mixed-source result. Its successful rerun is recorded
in [hpc-options-final.log](hpc-options-final.log). The full discovery log is
[full-ghost-tests.log](full-ghost-tests.log). No live SLURM cluster or new 10 GHz
airfoil benchmark was run for this refactor.

## Removal findings

`cpu_checked_lu.py` has no active application or test importers. It is a small
CheckedLU adapter over DenseFactor. The copied-backend test imports the headless
package without it. Removal also requires deleting its diagnostics inventory
entry and checking separately maintained scripts that may import it.

`cpu_streaming.py` is still imported by CPU dispatch. Its four re-exports could
be consolidated with that dispatch, but the file is not dead today. Python
bytecode caches can be regenerated. Native artifacts, analytic references,
command entry points, and the Python 3.6 backport remain required.

No removal candidate was deleted from the working backend. Saved experiment
snapshots retain their recorded source; use the import map before rerunning
scripts that refer to former internal module names. Restart GRIM/GHOST before
testing the reorganized package in an existing GUI session.
