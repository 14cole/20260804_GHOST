# Audit: `codex/grim-integrated-tools` (commit `3e6c0a1`)

Read-only audit of the `codex/grim-integrated-tools` branch as of
2026-09-07. No file on that branch was modified; this report is the only
addition and lives on a separate branch.

## Scope and method

- Branch head audited: `3e6c0a1` (`Default length controls and displays to
  inches while preserving saved units`). The branch shares no history with
  `main` (`main` is a single empty initial commit), so the whole tree was in
  scope, with emphasis on the three most recent integration/refactor commits
  `22e270e`, `d42af2c`, and `3e6c0a1`.
- Environment: Linux, CPython 3.11.15, numpy 2.4.6, scipy 1.17.1,
  matplotlib 3.11.1, PySide6 6.11.2, `QT_QPA_PLATFORM=offscreen`. The
  project's supported release target is 64-bit Windows CPython 3.12 with the
  lock in `requirements/constraints-windows-py312.txt`, so some results below
  are environment-dependent and are labelled as such.
- Checks run: `verify_project.py --mode quick` and `--mode full`, every
  acceptance suite individually, `pyflakes` over all Python sources,
  duplicate-definition and encoding scans, bisection of one regression across
  branch history, and a manual read of the integration modules
  (`ghost_integration.py`, `freddy_integration.py`, `grim_diagnostics.py`),
  the new GHOST driver helper modules, the FREDDY search/checkpoint modules,
  and the split GRIM dataset/assembly modules.

## Summary

The integration architecture is sound and the module split in `22e270e` and
`d42af2c` preserved public APIs: every facade symbol external callers import
still resolves, all new modules cold-import in isolation, and the inventory
gate passes. The branch is not release-ready as it stands, for one concrete
reason and several softer ones:

1. **The GHOST CEM tools suite fails on every commit since 637ba79
   (2026-09-04).** Its GRIM bridge loads `grim_dataset.py` by file path
   without putting `GRIM_Revised_2` on `sys.path`, and `grim_dataset.py` has
   used flat sibling imports since that commit. The release gate runs this
   suite, so `build_release.py` would refuse to build this head on Windows
   too. Three tests error; all pass with `GRIM_Revised_2` on `PYTHONPATH`.
2. **`verify_project.py --mode quick` cannot pass off Windows.** Two GRIM
   tests hard-code Windows behaviour, one asserts exact font metrics, and one
   asserts an exact float32 dB value. None of these indicate a product
   defect, but they make the documented "development check" red on the
   macOS machine these commits were authored on.
3. **The refactor's "consolidate shared driver helpers" goal is partly
   done.** `driver_io.py` and `driver_config.py` exist, but the four driver
   scripts still carry divergent private copies of six helpers, and
   `driver_config.validate_settings` accepts values the drivers later reject.
4. **A FREDDY data-loss path**: loading an inverse-search checkpoint and then
   pressing "Analyze all combinations" overwrites that checkpoint file with
   an empty one before any scoring starts.

Details, with file and line references, follow.

## Test results

All suites were run from a clean checkout of `3e6c0a1`.

| Suite | Result | Notes |
| --- | --- | --- |
| Startup diagnostics | pass | After installing `libegl1`; without it PySide6 fails to import. |
| Source inventory checks | pass | `verify_project.py --inventory-only`. |
| UTF-8 cleaner tests | pass | |
| Offline wheelhouse tests | pass | |
| GRIM tests (`GRIM_Revised_2`) | 1041 / 1045 | 4 failures, all environment-coupled; see findings M4, M5, L1. |
| GHOST unit tests (`tools/GHOST/tests`) | GHOST_UNIT_PLACEHOLDER | |
| GHOST CEM tools tests | 13 / 16 | 3 errors, real regression; see finding H1. |
| GHOST HPC scheduling integration | pass | |
| GHOST local-driver integration | pass | |
| GHOST ASCII-transfer check | pass | Covers `tools/GHOST` only; see finding M6. |
| FREDDY tests | 140 / 140 | |

`verify_project.py` stops at the first failing suite, so on this head
`--mode full` never reaches the GHOST or FREDDY suites; each was run
directly.

`pyflakes` over the whole tree: 382 unused imports (almost all deliberate
compatibility re-exports in `rcs_solver.py`, `grim_cut_dataset_mixin.py`,
`feature_assembly_panel.py`, and the FREDDY UI modules), 24 unused locals,
6 undefined names, 5 placeholder-less f-strings, 3 duplicate imports. The
undefined names are discussed under L3 and L4.

## Findings

Severity reflects impact on the branch's stated goals (a copy-ready,
verifiable Windows release with GHOST and FREDDY embedded in GRIM), not
code-style preference.

### High

**H1. GHOST CEM tools cannot load GRIM datasets; the acceptance suite has
been red since 2026-09-04.**
`tools/GHOST/CEM_Tools/cem_tools/grim_bridge.py:33-49` loads
`GRIM_Revised_2/grim_dataset.py` with `spec_from_file_location` under a
private module name but never adds `GRIM_Revised_2` to `sys.path`. Since
`637ba79`, `grim_dataset.py:17` does `from grim_metadata import ...`, and
since `22e270e`, `grim_dataset.py:1` does `from grim_format_io import
RcsGridFormatMixin`. Both are flat sibling imports that fail under the
bridge. Bisection: the CEM suite passes at `55251f2`, fails at `637ba79`
with `ModuleNotFoundError: grim_metadata`, and fails at `3e6c0a1` with
`ModuleNotFoundError: grim_format_io`. It passes on `3e6c0a1` with
`PYTHONPATH=GRIM_Revised_2`. The GHOST backend's own bridge
(`tools/GHOST/Backend/grim_compat.py:90-95`) does insert the directory and
is unaffected. Secondary defect: after the failed `exec_module`, the
half-initialised module stays in `sys.modules`, so the second and third
tests fail with `AttributeError: no attribute 'RcsGrid'` rather than the
root cause. Affected commands: `cem_tools` conversion and GRIM CSV export
(`operations.convert_files`, `grim_bridge.load_dataset`). The release gate
(`verify_project.acceptance_suites`, "GHOST CEM tools tests") will refuse
this head.

### Medium

**M1. FREDDY checkpoint overwrite on "Analyze all combinations" after
"Load checkpoint".** `tools/FREDDY/ibc/inverse_workflow.py:205` sets the
recovery path to the loaded `.fsearch`. A subsequent fresh analysis passes
`checkpoint=None` (`tools/FREDDY/ibc/ui.py:4214`), and
`tools/FREDDY/ibc/design_search.py:268` calls `publish_checkpoint(force=True)`
with `next_index=0` before scoring, atomically replacing the loaded file
with an empty checkpoint. No confirmation is shown. Related: after loading a
checkpoint, "Choose / save recovery file" raises "Run the search before
applying or saving a candidate" (`inverse_workflow.py:174-176, 390-391`), so
the user cannot redirect the path first.

**M2. `driver_config.validate_settings` accepts values the drivers reject.**
`tools/GHOST/Backend/driver_config.py:77` allows `CFIE_ALPHA` in the closed
interval `[0, 1]`; `run_local_bor.py:490` and `run_hpc_bor_monostatic.py:765`
require the open interval. `driver_config.py:79` allows any positive
`MEMORY_SAFETY` and `CLAIM_STALE_SECONDS`; `run_local_monostatic.py:447`,
`run_hpc_monostatic.py:684, 690` require `>= 1` and `>= 60`.
`driver_config.py:59-63` checks frequencies only for exact duplicates; the
drivers also require distinctness at 0.001 GHz. Result: `hpc_common.
configure_driver` stages a driver that later dies with `sys.exit`, contrary
to the fail-before-staging contract exercised by
`tools/GHOST/tests/test_driver_config.py:40`.

**M3. Driver helper consolidation is incomplete and copies have drifted.**
Commit `22e270e` says "consolidate shared driver helpers", and ARCHITECTURE
lists `driver_io.py` as the shared home. Still duplicated across
`run_local_bor.py`, `run_local_monostatic.py`, `run_hpc_monostatic.py`, and
`run_hpc_bor_monostatic.py` (AST body hashes differ in every case marked
"divergent"):

| Helper | Copies | State |
| --- | --- | --- |
| `_verify_run_provenance` | 4 | all divergent; HPC copies add an empty-hash "legacy runs" check the local ones lack |
| `_validate_config` | 3 + inline in BoR HPC `submit()` | 2-D pair identical except HPC-only fields |
| `_solve_and_export` | 4 | 2-D pair identical except a `history=` string |
| `_discover_geometries` | 4 | local BoR hard-codes `*.geo` and lacks `GEOMETRY_EXTS`; HPC BoR honours it |
| `_load_snapshot` | 3 | `run_hpc_monostatic.py:411-431` is a verbatim copy of `driver_io.load_geometry_snapshot` |
| atomic JSON writer | 5 implementations | three durability levels (with/without fsync, with/without directory fsync); HPC `manifest.json`/`schedule.json` still written non-atomically at `run_hpc_monostatic.py:785, 797` and `run_hpc_bor_monostatic.py:928` |

Also: `copy_configuration` is imported by both local drivers
(`run_local_bor.py:120`, `run_local_monostatic.py:131`) but only the HPC
drivers call it, so a local run directory never retains the configuration
that produced it, only its hash.

**M4. `verify_project.py --mode quick` is red on any non-Windows host.**
`GRIM_Revised_2/test_release_builder.py:548-565`
(`test_dependency_gate_rejects_missing_locked_optional_package`) does not
mock `sys.platform`, unlike its siblings at lines 523-539, so on Linux/macOS
it hits the Windows-only guard at `build_release.py:786` first and fails
with the wrong message. README describes `--mode quick` as the development
check, and the last twelve commits were authored on macOS.

**M5. `test_wheel_installation` depends on the host's setuptools.**
`GRIM_Revised_2/test_wheel_installation.py:29` runs `pip wheel
--no-build-isolation --no-index`, so it uses whatever setuptools is on the
host. With Debian/Ubuntu's patched setuptools 68 it fails with
`AttributeError: install_layout`; in a fresh venv with setuptools 84 it
passes. Environmental, but worth noting because the README says the quick
mode "checks the installed wheel".

**M6. The ASCII-source contract is enforced for GHOST only, while GRIM and
FREDDY contain exactly the characters it guards against.**
`tools/GHOST/tests/test_source_is_ascii.py` scans `tools/GHOST` only.
42 of 124 GRIM `.py` files and 15 of 40 FREDDY `.py` files contain
non-ASCII characters: box-drawing comment banners, em dashes, `×`, `…`,
`–`, and `−` in both comments and user-visible strings (for example
`assembly_tree.py:1041-1078`, `assembly_placement_editor.py:34`,
`assembly_response_comparison.py:195`). The docstring of the GHOST test and
the README's `clean_utf8.py` section describe the real failure mode: a
text-mode copy through a Windows tool turns these into undecodable bytes and
the module becomes unimportable. The release gate's strict-UTF-8 check
catches damage after the fact but not the exposure. Either extend the ASCII
check to the whole tree or document the decision to exempt GRIM/FREDDY.

**M7. Blank unit metadata is accepted everywhere except CSV export.**
`GRIM_Revised_2/grim_dataset.py:1808-1810` (`RcsGrid._canonical_unit`) and
`GRIM_Revised_2/dataset_dialogs.py:17-45` map empty or `None` units to the
default (`deg`/`GHz`), and native loading keeps blank unit strings. The
parallel helpers in `GRIM_Revised_2/grim_csv_schema.py:103-123` raise on
blank input. A grid with `units={"azimuth": ""}` loads, audits, and opens
every dialog, but flat CSV export (`dataset_jobs.py:321`,
`dataset_publication.py:270`) fails with `unsupported angular unit ''`.

**M8. `_available_memory_bytes` duplicated with different failure modes.**
`GRIM_Revised_2/dataset_jobs.py:21-60` catches only `ImportError`,
`AttributeError`, `OSError` from psutil; `grim_csv_schema.py:186-218`
catches `Exception`. A psutil `RuntimeError` therefore aborts the whole GUI
load batch (`dataset_jobs.py:147, 284-288`) and derived-grid operations
(`grim_cut_dataset_mixin.py:222-226`) while headless preflight degrades
gracefully. The sysconf fallbacks also differ: the schema copy can return a
negative product. Same duplication pattern applies to
`_canonical_frequency_unit`/`_canonical_angle_unit` (M7) and to
`_normalize_slurm_state` in `hpc_remote.py:228` and `runs_workspace.py:80`
(currently identical, no shared owner).

### Low

**L1. Exact-float and font-metric assertions in GRIM tests.**
`GRIM_Revised_2/test_python_recorder.py:331` asserts `min(display) ==
-120.0`; the float32 pipeline shared by GUI and headless
(`plot_modes/isar_render.py:326-328`, `grim_python.py:2064-2067`) yields
`-120.00000762939453` on numpy 2.4.6 (one float32 ulp). GUI and headless
still agree, so the "matches GUI" property holds; only the assertion is
brittle. `GRIM_Revised_2/test_compact_layout.py:72` asserts the window is
exactly 1200 px wide at the smallest size; offscreen Linux fonts give
1203 px. Both may pass on the locked Windows stack, but neither is robust.

**L2. Material-mix search cannot be cancelled.**
`tools/FREDDY/ibc/design_search.py:372-375` (`run_mix_search`) has no stop
or progress callback, and `ui.py` disables every button during the task. A
large `max_evals` with refinement leaves no exit other than killing the
process. `run_mix_search` also has no test.

**L3. Latent `NameError` in BoR near-kernel convergence.**
`tools/GHOST/Backend/bor_kernels.py:724-726` references `error` and
`scale` inside the `while True` loop before they are first assigned at
lines 738-740. On the first iteration the guarding condition
(`cf == c and tf == t`) cannot be true because `cf >= c + 16` unless `c`
is already at `NEAR_ANGULAR_MAX_ORDER`, which the check at line 705
excludes, so the path is unreachable today. Any change to those bounds
would turn the intended `ValueError` into a `NameError`.

**L4. Undefined `Any` in string annotations.**
`tools/GHOST/Backend/bor_solver.py:2469, 4477` use `'Dict[str, Any]'`
without importing `Any` (`bor_solver.py:35`). Harmless at runtime because
annotations are strings, but `typing.get_type_hints` on those functions
would raise; `test_module_boundaries.py:49-51` already uses
`get_type_hints` on other modules.

**L5. Silent exception swallowing in new FREDDY code.**
`design_search.py:551` (`except Exception: refined.append(candidate)`)
hides SLSQP post-processing failures while still reporting refinement
counts; `project_state.py:295-306` drops malformed mix components on
project load without notice; `mix_analysis.py:105` omits a mixing rule
from the comparison table on any error.

**L6. Inch spinbox precision rewrites SI recipe values.**
`feature_assembly_panel.py:1544-1549` and `dataset_dialogs.py:655-659`
round to 12 decimals of inches before multiplying by 0.0254, so a loaded
recipe re-saved without edits changes `skin_tol_m` from `0.001` to
`0.000999999999996` (`feature_assembly_recipe.py:139`, via `_pull_values`
at `feature_assembly_panel.py:3579`). Physically negligible; affects
provenance and round-trip fidelity.

**L7. Headless ISAR default unit changed under existing scripts.**
`grim_python.py:2039` now defaults `length_unit` to `"in"` while
`plot_modes/isar_mode.py:2819` (`form_isar`) still defaults to `"m"`.
Hand-written scripts that omit the option silently switch units. Recorded
scripts always emit the unit and are unaffected.

**L8. File permissions of published artifacts.**
`tools/FREDDY/ibc/search_checkpoint.py:70` and
`tools/GHOST/Backend/driver_config.py:122` create files via
`tempfile.mkstemp` (mode 0600) and `os.replace` them into place, so
`.fsearch` recovery files and `.config.json` are unreadable by
collaborators on shared POSIX folders, unlike other outputs. Not relevant
on the Windows target.

**L9. Minor pyflakes items.** `tools/FREDDY/ibc/design_search.py:44-47`
duplicate imports; f-strings with no placeholders at
`grim_compat.py:312`, `run_hpc_bor_monostatic.py:945, 997`,
`run_hpc_monostatic.py:918`, `geometry_tab.py:1614`; 24 unused locals,
of which `rcs_operators.py:399-402` (four `*_major` arrays computed and
discarded) and `rcs_solver.py:5076` (`preflight` computed and discarded)
are worth a look for dead computation.

**L10. Test coverage gaps in the new modules.** No dedicated tests for
`driver_io.py`, `feature_preparation.py`, `feature_library_contracts.py`,
`rcs_constants.py`, `rcs_special.py` (the series fallbacks and
`_raise_if_untrusted_math_backends` are untested), FREDDY `mix_analysis.py`,
`ui_controls.py`, `ui_dialogs.py`, `ui_options.py`, or `run_mix_search`.
Each is exercised only transitively.

## Suggested order of work

1. Fix H1 in `grim_bridge._rcs_grid_class`: insert `grim_project_path()`
   into `sys.path` before `exec_module` (as `grim_compat.rcsgrid_class`
   does) and pop the private module on failure. Re-run the CEM suite.
2. Fix M1 by refusing a fresh analysis when the recovery path names an
   existing checkpoint with `next_index > 0`, or by prompting.
3. Align `driver_config.validate_settings` with the drivers (M2), then
   finish the helper consolidation (M3) so the validator and the drivers
   share one source of truth.
4. Make the quick gate platform-neutral (M4, L1) or state in README that
   it is Windows-only like the release gate.
5. Decide the ASCII policy for GRIM and FREDDY (M6).
6. Give `_canonical_*_unit` and `_available_memory_bytes` one owner
   (M7, M8).

## Things that look right

- Integration boundaries are enforced, not just documented.
  `ghost_integration.load_ghost_module` validates that every already-loaded
  flat GHOST module comes from the selected backend and refuses a mixed
  process; `freddy_integration.load_freddy_package` loads `ibc` under a
  private package name and rolls back `sys.modules` on a failed import. The
  `ibc` package uses only relative imports, so the private namespace works
  (the only absolute `ibc` imports are in the standalone launchers outside
  the package).
- The module split preserved compatibility. An AST cross-check of every
  facade import, aliased attribute, and `mock.patch` target across tests,
  examples, `grim_headless.py`, `grim_python.py`, `ppt_*.py`, and
  `build_release.py` found no missing symbol. The `RcsGrid` MRO provides
  every classmethod callers use, and all 20 new or facade modules
  cold-import in isolation. The 21 moved GHOST constants match their
  pre-refactor values, and the moved numerical bodies are verbatim.
- `verify_project.py --inventory-only` cross-checks the startup inventory in
  `grim_diagnostics`, the `pyproject.toml` `py-modules` list, and the static
  local import closure, and passes on this head.
- Checkpoint and publication paths that claim atomicity are atomic:
  `search_checkpoint.py` (mkstemp, fsync, `os.replace`, unlink on failure),
  `dataset_publication.py` (stage-all-then-publish, reverse-order restore),
  `driver_io.publish_submission_journal` (fsync plus directory fsync).
  Checkpoint size and SHA-256 checks cannot be bypassed by editing the zip
  central directory; identity, engine hash, and totals are re-verified on
  resume.
- FREDDY background workers never read widgets or emit from non-GUI
  threads; the GUI keeps publication and selection.
- `hpc_remote.py` accepts no passwords, quotes every remote token with
  `shlex.quote`, and validates configuration before any subprocess call.
- Tests leave the working tree clean and run products (`rcs_runs/`,
  `results/`, `claims/`) are ignored. No `TODO`/`FIXME` markers, no large
  binaries, no secrets in tracked files.
