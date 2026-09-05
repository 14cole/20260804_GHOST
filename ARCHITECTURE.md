# Application responsibilities

GRIM composes the dataset, plotting, solver, material, and report workspaces.
The GUI shell wires them together; presentation components do not perform file
operations or numerical work.

| Module | Responsibility |
| --- | --- |
| `GRIM_Revised_2/grim_cut_gui.py` | Create the application window and tabs, wire user actions, coordinate the active plot context, and manage application preferences. |
| `GRIM_Revised_2/dataset_sidebar.py` | Own the dataset table, action layout, parameter selectors, and drag/drop presentation. Emit export intentions for the shell to connect. |
| `GRIM_Revised_2/grim_widgets.py` | Reusable presentation widgets, searchable settings popup, collapsible sections, and initial window sizing. |
| `GRIM_Revised_2/grim_palette.py` | Palette names, descriptions, semantic color tokens, and preference normalization. |
| `GRIM_Revised_2/grim_theme.py` | Convert palette tokens into Qt stylesheets and branch indicators. |
| `GRIM_Revised_2/grim_cut_dataset_mixin.py` | Coordinate dataset operations, background jobs, saves, undo, and publication into the catalog. |
| `GRIM_Revised_2/grim_dataset.py` | `RcsGrid` data representation, numerical operations, and existing dataset APIs. |
| `GRIM_Revised_2/grim_metadata.py` | Inspect scalar metadata evidence and normalize convention declarations without Qt or dataset-object dependencies. |
| `GRIM_Revised_2/grim_cut_plot_mixin.py` and `plot_modes/` | Plot orchestration and mode-specific rendering. |
| `GRIM_Revised_2/ghost_integration.py` and `freddy_integration.py` | Discover and embed the authoritative tools and relay their artifacts/signals. |
| `tools/GHOST/Backend/solver_tab.py` | Solver form, execution controls, and progress. Its form scrolls separately from its action footer. |
| `tools/GHOST/Backend/material_models.py` | Material explanations and thin-layer input dialog. Numerical material semantics remain in the backend. |
| `tools/GHOST/Backend/thin_sheet.py` | Thin-layer validity checks, jump operators and field evaluation. Reuses 2D quadrature and linear-solve primitives. |
| `tools/GHOST/Backend/refined_lu.py` | Opt-in factor/refinement policy, double residual checks and fallback signaling. No Qt dependencies. |
| `tools/GHOST/Backend/solver_metrics.py` | Scoped timing and sampled memory collection shared with worker callbacks. |
| `tools/GHOST/Backend/mesh_guidance.py` | Pure snapshot-based refinement suggestions and density transformations. |
| `tools/GHOST/Backend/solver_quality.py` | Accuracy policies, evidence interpretation and report summary. |
| `tools/GHOST/Backend/assembly_inspector.py` | Source-verified complex contribution evaluation, bounded sample cache and interference algebra. |
| `GRIM_Revised_2/assembly_interference.py` | Inspector presentation and worker lifecycle; delegates numerical evaluation to the backend service. |
| `tools/GHOST/Backend/feature_family_validation.py` | Reference-study definitions, convergence/reconstruction checks and evidence reports. Never generates purported full-wave truth. |
| `tools/FREDDY/ibc/ghost_coating.py` | Planar reflection assessment of the scalar PEC-backed IBC approximation and frequency interpolation; no file writes, GUI dependencies or finite-body accuracy claims. |

## Dependency rules

- A presentation component may construct widgets and emit user intentions; it
  must not write datasets, launch a solver, or calculate fields.
- The shell connects intentions to operation handlers. `DatasetSidebar` owns
  its widgets; the shell retains aliases for the existing controllers while
  they are migrated incrementally. `DatasetTable`, `build_qss`, and the shared
  widgets remain importable from `grim_cut_gui` for existing callers.
- Put palette additions in `grim_palette`, and stylesheet behavior in
  `grim_theme`. Views consume those definitions rather than creating new
  palette registries.
- Metadata inspection reports `missing`, `consistent`, `conflicting`, or
  `malformed`, retaining declarations and their source containers. The legacy
  scalar adapter preserves the existing advisory/strict eligibility policy.
  Plot warnings can distinguish bad declarations from absent metadata without
  changing numerical eligibility or guessing field transformations.
- Keep GHOST and FREDDY numerical implementations within their tool trees.
  GRIM integration classes own embedding and handoff behavior.

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
`py-modules` list and `grim_diagnostics.GRIM_STARTUP_FILES`, so installed and
copy-ready distributions include the same runtime contract.
