# GHOST solver and feature workflows

GHOST is bundled inside the GRIM distribution. This folder is a complete
solver project so its backend, tests, geometry studies, CEM utilities, and
launchers retain their established relative paths.

The recommended desktop workflow is the top-level GRIM application. Its
**GHOST** tab embeds the same `Backend/ghost_gui.py` workspace and the same
2-D/BoR numerical implementation found here; no solver is duplicated.

## Standalone GHOST

Run commands from this folder:

```powershell
py Backend\ghost_gui.py
```

On Windows, `Launch_GHOST_GUI.bat` first changes to this folder and then opens
the same workspace.

## Local and HPC drivers

Edit the configuration block in the relevant driver, then run:

```powershell
py Backend\run_local_monostatic.py
py Backend\run_local_bor.py
py Backend\run_hpc_monostatic.py
py Backend\run_hpc_bor_monostatic.py
```

The 2-D production path co-solves VV/TE and HH/TM and writes them into one
GRIM artifact per geometry/frequency. The BoR path produces a combined
feature-ready body artifact after its per-frequency restart units complete.

See:

- [HPC.md](HPC.md) for local/cluster operation and resource controls.
- [GEOMETRY_INPUT_CHEATSHEET.md](GEOMETRY_INPUT_CHEATSHEET.md) for `.geo`
  boundaries, regions, materials, winding, and units.
- [BOR_CONVENTIONS.md](BOR_CONVENTIONS.md) for BoR geometry, polarization,
  phasor, loss, and RCS conventions.
- [FEATURE_VALIDATION_GUIDE.md](FEATURE_VALIDATION_GUIDE.md) for point and
  line-feature dataset and placement requirements.
- [geometry_tests/non_bor_feature_validation/README.md](geometry_tests/non_bor_feature_validation/README.md)
  for the independent four-artifact clean/featured validation ladder and
  manifest-driven complex-field gates.
- [geometry_tests/non_bor_line_reconstruction/README.md](geometry_tests/non_bor_line_reconstruction/README.md)
  for the checked-in finite-plate, door-outline, and folded-panel line tests.
- [geometry_tests/non_bor_curved_feature_placement/README.md](geometry_tests/non_bor_curved_feature_placement/README.md)
  for the triaxial-ellipsoid point/line regression and shared-facet normal-tie
  controls.

## Feature assembly service

The GRIM Assembly form and automation wrapper both call
`Backend/feature_workflow.py`. `Backend/place_features.py` remains a thin
settings-based wrapper for unattended work. It defaults to the Production
profile: strict clean-body metadata, validated content-bound feature manifests,
and declared host material/coating IDs. Legacy compatibility must be selected
explicitly. A warning-bearing plan prints its sealed SHA-256 and requires that
exact digest as the acknowledgment before execution, so a changed plan cannot
reuse an old waiver. The GUI does not reimplement placement, phase, expansion,
or shadowing physics.

Create and check a reviewed feature-response sidecar with:

```powershell
py Backend\create_feature_manifest.py create --help
py Backend\create_feature_manifest.py check --help
py Backend\create_feature_manifest.py create-surface-binding --help
py Backend\create_feature_manifest.py check-surface-binding --help
```

For `validated` libraries this is now an evidence-binding and integrity tool:
it consumes the full-wave validator report, re-hashes all four case artifacts,
and proves that the assembled prediction used the exact response. Team review
is still required because software cannot establish external-solver
independence or mesh convergence. See
[FEATURE_VALIDATION_GUIDE.md](FEATURE_VALIDATION_GUIDE.md) for the exact
manifest fields, headless settings, reduced-order limitations, and required
independent full-wave evidence.

Use `1c_build_deltas/subtract_datasets.py` for canonical OPN-FRD 2-D deltas.
General CEM joins and coherent subtraction are under `CEM_Tools`.

## Tests

From this folder:

```powershell
py -m unittest discover -s tests -p "test*.py" -v
```
