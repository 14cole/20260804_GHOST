# Feature-family reference studies

`study.template.json` defines 13 cases: corner angles 30/60/90 degrees,
termination lengths 1/3/6 wavelengths, curved-seam radii 1/3/10 wavelengths,
and pair separations 0.1/0.25/0.5/1 wavelength. Geometry/material requirements
and radar conventions are recorded in the file. It contains no field results.

Copy the template to your study folder. Specify the independent reference solver
and document its exact geometry, settings and radar grid. For each case supply:

- Clean and featured full-wave references at three distinct mesh levels;
  the finest files are `clean_truth.grim` and `featured_truth.grim`.
- `clean_prediction.grim` and `featured_prediction.grim` from the Assembly
  workflow, including the normal provenance and feature-response metadata.

All files must preserve complex physical F in compatible conventions,
sigma = 4*pi*abs(F)^2, and matching VV/HH/VH grids. Filenames are relative to
the study JSON. The checker cannot independently attest the declared solver's
identity: the reference producer must establish that provenance.

Use the Assembly inspector's study button, or from the repository root:

```powershell
.venv/Scripts/python.exe tools/GHOST/Backend/feature_family_validation.py --study path/to/study.json --report path/to/new-report.json
```

The report path must be new. A nonzero exit means not all cases passed.
Missing artifacts remain `awaiting_reference_artifacts`; they are not passes.
The report checks two successive reference refinements (complex normalized RMS
2%, magnitude p95 0.5 dB, phase RMS 3 degrees, coherence 0.999), then uses the
existing clean/featured/isolated-delta reconstruction gates. It binds results
to source hashes and lists only tested parameter values as coverage.

Current status: **0/13 new cases validated; external full-wave artifacts are
needed**. Existing analytic, BoR and manufactured tests support their own cases,
not a broad corner/termination/curvature/coupling envelope. Failing close-pair
cases may require a jointly characterized feature or full-wave coupling.
