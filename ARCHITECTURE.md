# Application responsibilities

GRIM composes the dataset, plotting, solver, material, and report workspaces.
The GUI shell wires them together; presentation components do not perform file
operations or numerical work.

| Module | Responsibility |
| --- | --- |
| `GRIM_Backend/ui/app.py` | Create the application window and tabs, wire user actions, coordinate the active plot context, and manage application preferences. |
| `GRIM_Backend/ui/dataset_sidebar.py` | Own the dataset table, action layout, parameter selectors, and drag/drop presentation. Emit export intentions for the shell to connect. |
| `GRIM_Backend/ui/widgets.py` | Reusable presentation widgets, searchable settings popup, collapsible sections, and initial window sizing. |
| `GRIM_Backend/ui/palette.py` | Palette names, descriptions, semantic color tokens, and preference normalization. |
| `GRIM_Backend/ui/theme.py` | Convert palette tokens into Qt stylesheets and branch indicators. |
| `GRIM_Backend/ui/dataset_actions.py` | Coordinate dataset operations, background jobs, saves, undo, and publication into the catalog. |
| `GRIM_Backend/datasets/api.py` | `RcsGrid` data representation, numerical operations, and existing dataset APIs. |
| `GRIM_Backend/datasets/audit.py` | Non-mutating dataset health diagnostics, with bounded array scans and the existing `RcsGrid.audit()` API. |
| `GRIM_Backend/io/native.py`, `cst.py`, `sentri.py`, `out.py`, `pioneer.py`, `ptm.py`, `xpatch.py` | Native archive loading and format-specific adapters inherited by `RcsGrid`. Preserve classmethod dispatch, allocation checks, metadata, and existing reader/writer signatures. |
| `GRIM_Backend/ui/dataset_dialogs.py` | Dataset operation dialogs and input-unit presentation. |
| `GRIM_Backend/ui/table_import.py`, `io/mapped_table.py` | Adapt the shared column editor and normalized rows into RCS grids using the existing CSV validation and memory admission. |
| `tools/FREDDY/ibc/table_conversion.py`, `converter_dialog.py` | Qt-free streaming text parsing, explicit unit conversion, atomic table export, and the shared column editor. GRIM loads the authoritative editor through its private FREDDY package. |
| `GRIM_Backend/execution/dataset_jobs.py` | Background workers, loader memory admission, and bounded parallel loading. |
| `GRIM_Backend/io/batch.py` | Atomic GRIM/CSV staging, rollback, and compression policy. |
| `GRIM_Backend/assembly/values.py` | Shared Qt-free form values and loaded recipe records. |
| `GRIM_Backend/assembly/model.py` | Qt-free Assembly form state, workload estimates, preflight, and backend adaptation. |
| `GRIM_Backend/assembly/recipe.py` | Portable recipe serialization and atomic recipe publication. |
| `GRIM_Backend/assembly/panel.py` | Qt controls and worker lifecycle, with compatibility exports for the form/recipe APIs. |
| `GRIM_Backend/plotting/modes/isar_render.py` | ISAR GUI selection capture and image presentation; numerical formation and caches remain in `isar_mode.py`. |
| `GRIM_Backend/datasets/metadata.py` | Inspect scalar metadata evidence and normalize convention declarations without Qt or dataset-object dependencies. |
| `GRIM_Backend/plotting/actions.py` and `GRIM_Backend/plotting/modes/` | Plot orchestration and mode-specific rendering. |
| `GRIM_Backend/integrations/ghost.py` and `GRIM_Backend/integrations/freddy.py` | Discover and embed the authoritative tools and relay their artifacts/signals. |
| `tools/GHOST/ghost_backend/ui/solver.py` | Solver form, execution controls, and progress. Its form scrolls separately from its action footer. |
| `tools/GHOST/ghost_backend/twod/solver.py` | 2-D solve orchestration, formulations, quality gates, resource admission, and compatibility imports for existing callers. |
| `tools/GHOST/ghost_backend/twod/constants.py`, `special.py` | Shared physical/default constants and trusted Bessel/Hankel backends. |
| `tools/GHOST/ghost_backend/twod/geometry.py` | Material tables, geometry validation, and panel/linear-mesh construction. |
| `tools/GHOST/ghost_backend/twod/operators.py` | Quadrature, boundary operators, field evaluation, and operator tuning state. |
| `tools/GHOST/ghost_backend/runs/inputs.py` | Shared geometry-input verification, cache-aware snapshot loading, and durable submission journals. Each local driver supplies its own cache. |
| `tools/GHOST/ghost_backend/runs/config.py` | Typed JSON setting validation, unchanged driver staging, desktop 2-D recipe adaptation, and configuration provenance. |
| `tools/GHOST/ghost_backend/assembly/preparation.py` | Capture source/output identities and prepare surface, line, and point placements in explicit stage records. |
| `tools/GHOST/ghost_backend/assembly/contracts.py` | Bind feature manifests, applicability limits, and component identities to an Assembly plan. |
| `tools/GHOST/ghost_backend/geometry/materials.py` | Material explanations and thin-layer input dialog. Numerical material semantics remain in the backend. |
| `tools/GHOST/ghost_backend/twod/formulations/thin_layer.py` | Thin-layer validity checks, jump operators and field evaluation. Reuses 2D quadrature and linear-solve primitives. |
| `tools/GHOST/ghost_backend/linalg/refined_lu.py` | Opt-in factor/refinement policy, double residual checks and fallback signaling. No Qt dependencies. |
| `tools/GHOST/ghost_backend/execution/metrics.py` | Scoped timing and sampled memory collection shared with worker callbacks. |
| `tools/GHOST/ghost_backend/geometry/guidance.py` | Pure snapshot-based refinement suggestions and density transformations. |
| `tools/GHOST/ghost_backend/runs/quality.py` | Accuracy policies, evidence interpretation and report summary. |
| `tools/GHOST/ghost_backend/assembly/inspector.py` | Source-verified complex contribution evaluation, bounded sample cache and interference algebra. |
| `GRIM_Backend/assembly/interference.py` | Inspector presentation and worker lifecycle; delegates numerical evaluation to the backend service. |
| `tools/GHOST/ghost_backend/validation/feature_family.py` | Reference-study definitions, convergence/reconstruction checks and evidence reports. Never generates purported full-wave truth. |
| `tools/FREDDY/ibc/design_search.py` | Qt-free inverse-stack and bounded material-recipe searches over captured request data; callbacks provide cancellation, progress, and numerical adapters. |
| `tools/FREDDY/ibc/mix_analysis.py` | Qt-free material recipe curves, target comparisons, and stack-performance evaluation. |
| `tools/FREDDY/ibc/search_checkpoint.py` | Atomic recovery archives with completed scores, search/source identity, size checks, and content checksums. |
| `tools/FREDDY/ibc/ui_controls.py`, `ui_dialogs.py`, `ui_options.py` | Shared bindings, layer/material editors, and stable form option values. |
| `tools/FREDDY/ibc/project_state.py` | Project capture/restoration between controls and portable dictionaries; path and file semantics remain in `ibc/io.py`. |
| `GRIM_Backend/examples/_folder_common.py` | Shared folder discovery and axis-limit validation for the separate editable sweep examples. |
| `tools/FREDDY/ibc/ghost_coating.py` | Planar reflection assessment of the scalar PEC-backed IBC approximation and frequency interpolation; no file writes, GUI dependencies or finite-body accuracy claims. |

## Dependency rules

- A presentation component may construct widgets and emit user intentions; it
  must not write datasets, launch a solver, or calculate fields.
- The shell connects intentions to operation handlers. `DatasetSidebar` owns
  its widgets; the shell retains aliases for the existing controllers for workspace controllers. `DatasetTable`, `build_qss`, and the shared
  widgets remain importable from `GRIM_Backend.ui.app` for existing callers.
- Put palette additions in `GRIM_Backend.ui.palette`, and stylesheet behavior in
  `GRIM_Backend.ui.theme`. Views consume those definitions rather than creating new
  palette registries.
- Metadata inspection reports `missing`, `consistent`, `conflicting`, or
  `malformed`, retaining declarations and their source containers. The legacy
  scalar adapter preserves the existing advisory/strict eligibility policy.
  Plot warnings can distinguish bad declarations from absent metadata without
  changing numerical eligibility or guessing field transformations.
- Keep GHOST and FREDDY numerical implementations within their tool trees.
  GRIM integration classes own embedding and handoff behavior.

## Module boundaries and compatibility

The established `RcsGrid` methods, dataset controller imports, Assembly form
imports, ISAR entrypoints, and GHOST driver paths remain available. The
controller/facade modules explicitly import moved symbols rather than retaining
second implementations. Tests that inject failures into an implementation patch
the module that now owns that implementation.

The GHOST dependency direction is constants -> special functions -> geometry ->
operators -> solve orchestration. Numerical functions/classes were moved with
their decorators and formulas intact. Operator tuning uses the existing
`rcs_solver.set_*` functions, whose implementation now owns state in
`rcs_operators`; direct inspection or patching of private operator state belongs
in that module. Both local drivers retain independent snapshot caches.

Format adapters import model policies at call time so the data model can inherit
the adapters without a module-initialization cycle. Assembly recipes share
form record types directly and resolve model policy helpers when called. Headless model, search,
and numerical modules must remain importable without a Qt installation. FREDDY
search services receive captured inputs rather than reading widgets while a job
runs; the GUI retains publication, selection, and background-job ownership.

FREDDY's layout construction has separate builders for Impedance, IBC Batch,
Off Angle, Thickness, Inverse Design, and Material Mix. Shared form factories
construct entries, output rows, and uncertainty controls. Search recovery is
optional and uses a captured destination; workers never read the path widget.

Assembly stage services resolve compatibility policy helpers from
`feature_workflow` at call time. That module remains the public orchestration
facade, including existing injection points. Stage records carry explicit
source identities and prepared placement geometry between operations.

The obsolete private helpers identified in the audit have been removed.
Public sidecar/artifact-manifest functions in `workflow_provenance` remain for
external script compatibility; current integrated drivers use embedded
attestations. Retiring those public APIs requires an external-consumer review.

## Saving and sizing

Save operates on selected catalog rows; Save All operates on every row. There
is no Save Dirty action. Internal unsaved-change tracking still supports close
prompts and protects derived results. Saving continues to use the existing
background jobs and atomic publishing path.

Window sizing uses Qt logical pixels and `QScreen.availableGeometry`, which
accounts for desktop taskbars. Embedded tall forms scroll instead of forcing
the whole main window to their minimum height. The sidebar has resizable
dataset/parameter sections and scrollbars when needed.

## Verification

`test_compact_layout.py` exercises the actual embedded workspaces at compact
sizes, sidebar visibility, solver actions while scrolling, export signals, and
palette propagation. `test_grim_metadata.py` covers evidence preservation and
the compatibility adapter. Existing shell, dataset, plot, and integration suites
cover the unchanged controller interfaces.

New eagerly imported modules must also appear in the `pyproject.toml`
package discovery and `GRIM_Backend.execution.diagnostics.GRIM_STARTUP_FILES`, so installed and
copy-ready distributions include the same runtime contract.

`verify_project.py --mode quick` checks the declared inventories against local
imports, including deferred imports, and runs packaging/startup smoke checks.
`--mode full` uses the same suite inventory as the mandatory release gate,
including standalone integration scripts. Inventory analysis does not execute
application code; dynamic imports and non-Python assets remain explicit entries.
