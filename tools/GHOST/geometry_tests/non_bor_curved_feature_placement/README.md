# Curved non-BoR feature-placement regression

`test_triaxial_ellipsoid_features.py` builds a closed triangulated triaxial
ellipsoid with three unequal semi-axes. It is a genuinely curved, faceted 3-D
platform, not a body of revolution.

The public feature-assembly workflow receives:

- a clean external platform GRIM from an independent faceted physical-optics
  fixture;
- an inch-valued compact-feature CSV with a stable placement ID, an oblique
  face normal, and an arbitrary roll reference;
- an inch-valued four-segment line CSV following mesh edges across multiple
  facets, with smooth analytic ellipsoid endpoint normals;
- one reciprocal anisotropic compact delta and one line-seal delta.

The test compares the isolated assembled feature field and the complete
clean-plus-featured field against independent Cartesian Jones-frame and
closed-form segmented-line oracles. It also proves that the clean artifact is
unchanged, verifies coherent power, and requires wrong location, inward
normal, parallel roll, and wrong line-normal controls to fail.

## Mesh creases

A shared mesh edge has more than one valid incident face normal. Placement now
uses open within-segment normal samples and a supplied-normal hint to resolve
true closest-face ties. A focused 35-degree folded-facet regression covers a
line crossing the crease, a line lying exactly on it from either incident
side, and a point feature located on the shared edge. Triangle storage order
must not decide whether these valid placements pass.

## Scope

This is an exact regression for GHOST's reduced-order placement, translation,
frame rotation, and coherent combination contracts. It does not model or
certify body-feature mutual coupling, multiple scattering, or the
transferability of the legacy line TM/TE phase mapping to a new physical seal.
Those claims require paired clean and featured full-wave results, compared as
the four-artifact cases described in `FEATURE_VALIDATION_GUIDE.md`.

High-value external follow-ons are a rounded rectangular enclosure, a swept
wing/fuselage panel, and a vehicle door/body section. Each adds compound
curvature, concavity/shadowing, or realistic closed seam loops without falling
back to axisymmetry.
