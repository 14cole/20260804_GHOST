# Ring-gap feature reconstruction fixture

This fixture is the always-run, all-GHOST placement regression in
`tests/test_feature_reconstruction_physics.py`. It checks whether a clean PEC
body plus a placed 2-D gap delta reproduces the complex far field of a BoR body
with the same circumferential groove modeled explicitly.

The 1 GHz body is a 0.08 m-radius, 0.30 m-long cylinder. The centered groove is
0.02 m wide and 0.01 m deep. The fixed clean and grooved generatrices contain
30 and 38 elements, respectively. The matched 2-D coupons extend from -0.16 m
to +0.16 m across the feature and close 0.10 m below the clean surface. Their
featured-minus-clean coefficient is produced by the certified co-polarized
2-D solver at 0:15:180 degrees. A 256-segment, decreasing-angle ring and radial
endpoint normals place the coefficient at the cylinder radius.

No fitted amplitude, phase, range, or coordinate correction is applied. The
test deliberately invokes `feature_sum.sum_features` without phase overrides,
so it exercises the public TM/TE convention mapping used by production.

## Fixed-mesh refinement evidence

The committed fixed meshes were compared with independently refined profiles:

| Profile | Fixed edge counts | Refined edge counts |
| --- | --- | --- |
| Clean | 4, 10, 2, 10, 4 | 6, 15, 3, 15, 6 |
| Grooved | 4, 10, 3, 4, 3, 10, 4 | 6, 15, 5, 6, 5, 15, 6 |

At aspects 60, 75, 90, 105, and 120 degrees, the normalized complex change in
the direct grooved-minus-clean delta under this refinement was 0.004018 for VV
and 0.006329 for HH; channel coherence exceeded 0.999989. The complete grooved
body changed by 0.006488 for VV and 0.005491 for HH. This study uses the same
1 GHz frequency, CFIE formulation, eight-mode cap, double-precision table
assembly, and one worker as the regression.

The refinement is documented instead of repeated in every test run: the fixed
pair takes roughly 19 seconds, while repeating the refined pair adds roughly
43 seconds. The test pins both element counts so a geometry change cannot
silently continue to claim this evidence.

The reduced-order reconstruction is an approximation, not an identity. On the
documented fixture its combined feature-delta normalized complex RMS is about
0.394 and its coherence about 0.951; the complete-body normalized complex RMS
is about 0.023. Whole-body agreement alone is therefore not accepted as proof
that placement is correct: both whole-field and isolated-delta gates run.
