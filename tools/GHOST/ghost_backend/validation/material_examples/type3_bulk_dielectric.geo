Title: TYPE 3 bulk dielectric square

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.

Segment: dielectric_square 3
properties: 3 0 0 1 0
-0.05 -0.05 -0.05 0.05
-0.05 0.05 0.05 0.05
0.05 0.05 0.05 -0.05
0.05 -0.05 -0.05 -0.05

IBCS_Resistances:


Dielectrics:
1 4.0 -0.2 1.0 0.0
