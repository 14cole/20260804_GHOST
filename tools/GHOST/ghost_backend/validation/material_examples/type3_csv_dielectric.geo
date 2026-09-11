Title: TYPE 3 frequency dependent dielectric

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.

Segment: dielectric_body 3
properties: 3 0 0 50 0
-0.05 -0.05 -0.05 0.05
-0.05 0.05 0.05 0.05
0.05 0.05 0.05 -0.05
0.05 -0.05 -0.05 -0.05

IBCS_Resistances:


Dielectrics:
50 radome_material.csv
