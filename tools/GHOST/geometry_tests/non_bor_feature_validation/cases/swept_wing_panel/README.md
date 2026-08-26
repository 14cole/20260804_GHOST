# External full-wave case: swept wing panel

This case targets the most failure-prone line-placement behaviors before a
vehicle model: a long swept hinge, a deliberate facet-normal break, path
segmentation, and an anisotropic compact feature on an oblique panel.  The
definition contains no solver output and is not validation evidence by itself.

Generate exactly these four artifacts in this directory:

```text
cases/swept_wing_panel/
    clean_truth.grim
    clean_prediction.grim
    featured_truth.grim
    featured_prediction.grim
```

The clean and explicitly featured fields must come from an independent,
mesh-converged 3-D Maxwell solver.  The prediction must use the clean Assembly
body plus feature-centered complex line deltas and installed-minus-clean local
point deltas.  Every artifact must retain the grid, global origin, time sign,
outgoing-wave sign, polarization basis, and normalization in `case_spec.json`.

Copy `feature_case.template.json` to `feature_case.json`, fill the four files,
and run from `tools/GHOST`:

```text
python Backend/validate_feature_reconstruction.py --manifest geometry_tests/non_bor_feature_validation/cases/swept_wing_panel/feature_case.json --report geometry_tests/non_bor_feature_validation/cases/swept_wing_panel/report.json
```

The isolated complex delta is the primary placement metric.  Also run the
specified segment-splitting, path-reversal, translation, and roll controls;
whole-body agreement alone can hide a materially incorrect feature beneath a
large clean-wing return.
