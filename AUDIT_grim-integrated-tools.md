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
  are explicitly environment-dependent and are labelled as such.
- Checks run: `verify_project.py --mode quick` and `--mode full`, every
  acceptance suite individually (GRIM, GHOST, GHOST CEM tools, GHOST HPC and
  local-driver integration, GHOST ASCII check, FREDDY, wheelhouse, UTF-8
  cleaner), `pyflakes` over all Python sources, duplicate-definition and
  encoding scans, and a manual read of the integration modules
  (`ghost_integration.py`, `freddy_integration.py`, `grim_diagnostics.py`),
  the new driver helper modules, the FREDDY search/checkpoint modules, and the
  split GRIM dataset/assembly modules.

## Summary

RESULTS_SUMMARY_PLACEHOLDER

## Test results

TEST_RESULTS_PLACEHOLDER

## Findings

FINDINGS_PLACEHOLDER

## Things that look right

- Integration boundaries are enforced, not just documented.
  `ghost_integration.load_ghost_module` validates that every already-loaded
  flat GHOST module comes from the selected backend and refuses a mixed
  process; `freddy_integration.load_freddy_package` loads `ibc` under a
  private package name and rolls back `sys.modules` on a failed import. The
  `ibc` package uses only relative imports, so the private namespace works.
- `verify_project.py --inventory-only` cross-checks the startup inventory in
  `grim_diagnostics`, the `pyproject.toml` `py-modules` list, and the static
  local import closure. It passes on this head, so the wheel and the checkout
  ship the same runtime contract.
- `hpc_remote.py` accepts no passwords, quotes every remote token with
  `shlex.quote`, and validates configuration before any subprocess call.
- Tests leave the working tree clean (`git status` is empty after the full
  run) and run products (`rcs_runs/`, `results/`, `claims/`) are ignored.
- No `TODO`/`FIXME` markers, no large binaries, no secrets in tracked files.
