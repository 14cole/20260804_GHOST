# Gap feature isolation geometries

This folder is intentionally separate from the solver implementation and its
general regression tests. It contains controlled FRD/OPN geometry pairs for a
0.5-inch-wide by 1-inch-deep PEC-backed groove.

## Coupon shape

The illuminated surface is flat and the remote surface is a symmetric,
D-backed rounded contour. A cubic Bezier curve creates outward-rounded
shoulders and a curved lower closure, avoiding the four artificial corners and
long parallel back surface of the earlier rectangular coupon. The curve is
written as 48 straight `.geo` primitives by default.

The generator sizes that contour until every generated backing chord is at
least the requested number of wavelengths from the groove walls and floor.
FRD and OPN use the exact same remote contour and matching primitive endpoints
on their common top surface.

## Clearance study

The checked-in default study uses target minimum clearances of 3.5, 4.25, and
5 wavelengths at every integer frequency from 1 through 15 GHz. These are
convergence cases, not three interchangeable responses. Compare the complex
`OPN - FRD` result between clearances and retain the smallest clearance that
meets the study's error requirement.

Regenerate the defaults from the repository root with:

```text
python geometry_tests/gap_feature_isolation/generate_gap_geometry_study.py
```

Validate the checked-in topology, winding, FRD/OPN contour matching, and
minimum clearances with:

```text
python geometry_tests/gap_feature_isolation/validate_gap_geometry_study.py
```

Use different clearance values with:

```text
python geometry_tests/gap_feature_isolation/generate_gap_geometry_study.py \
    --clearance-lambda 3.75 4.5 5.25
```

The output organization is frequency first and then solver role:

```text
generated/06.000GHz/FRD/C4p250lambda/*.geo
generated/06.000GHz/OPN/C4p250lambda/*.geo
```

`run_local_monostatic.py` recursively discovers all clearance cases beneath
the configured role folders. To test every generated clearance at 6 GHz, use:

```text
FRD_DIR = "geometry_tests/gap_feature_isolation/generated/06.000GHz/FRD"
OPN_DIR = "geometry_tests/gap_feature_isolation/generated/06.000GHz/OPN"
FREQUENCIES_GHZ = [6.0]
```

Run each frequency folder only at its design frequency. The local driver forms
a Cartesian product of discovered geometries and configured frequencies; it
does not infer frequency from directory names.
