# Gap feature isolation geometries

This folder is intentionally separate from the solver implementation and its
general regression tests. It contains the controlled FRD/OPN geometry pairs
used to study a 0.5-inch-wide by 1-inch-deep PEC-backed groove.

Each frequency folder under `generated/` contains one matched pair:

- `FRD`: the flush reference body.
- `OPN`: the same body with the groove present.

The outer coupon provides four free-space wavelengths of lateral clearance
from each aperture edge and four wavelengths below the groove floor at the
folder's design frequency. Run a pair only at that design frequency, then form
the complex field difference `OPN - FRD`.

Regenerate the complete 1--15 GHz set from the repository root with:

```text
python geometry_tests/gap_feature_isolation/generate_gap_geometry_study.py
```

The rectangular coupons are controlled size-isolation specimens, not proof
that the finite outer boundary is negligible. Compare multiple clearances or
remote-boundary shapes before treating the difference as a transferable gap
response.
