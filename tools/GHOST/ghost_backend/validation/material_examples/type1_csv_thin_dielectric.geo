Title: TYPE 1 thin layer with frequency dependent dielectric

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.

Segment: thin_strip 1
properties: 1 80 10 0 0
-0.05 0 0.05 0

IBCS_Resistances:
10 thin_dielectric 0.0005 50

Dielectrics:
50 radome_material.csv
