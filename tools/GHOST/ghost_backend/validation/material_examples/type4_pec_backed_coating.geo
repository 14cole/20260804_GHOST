Title: TYPE 4 explicit coating over PEC

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.

Segment: coating_outer 3
properties: 3 0 0 1 0
-0.05 -0.05 -0.05 0.05
-0.05 0.05 0.05 0.05
0.05 0.05 0.05 -0.05
0.05 -0.05 -0.05 -0.05

Segment: coating_inner 4
properties: 4 0 0 1 0
-0.04 -0.04 -0.04 0.04
-0.04 0.04 0.04 0.04
0.04 0.04 0.04 -0.04
0.04 -0.04 -0.04 -0.04

IBCS_Resistances:


Dielectrics:
1 2.8 -0.06 1.0 0.0
