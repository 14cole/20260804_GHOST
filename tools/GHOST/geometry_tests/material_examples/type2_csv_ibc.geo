Title: TYPE 2 frequency dependent impedance

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.

Segment: ibc_body 2
properties: 2 0 40 0 0
-0.05 -0.05 -0.05 0.05
-0.05 0.05 0.05 0.05
0.05 0.05 0.05 -0.05
0.05 -0.05 -0.05 -0.05

IBCS_Resistances:
40 surface_impedance.csv

Dielectrics:
