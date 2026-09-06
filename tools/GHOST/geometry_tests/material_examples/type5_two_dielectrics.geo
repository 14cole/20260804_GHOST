Title: TYPE 5 dielectric core in dielectric shell

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.
# Both contours clockwise: TYPE 5 normal points from core material 2 into shell material 1.

Segment: outer_air_interface 3
properties: 3 0 0 1 0
-0.05 -0.05 -0.05 0.05
-0.05 0.05 0.05 0.05
0.05 0.05 0.05 -0.05
0.05 -0.05 -0.05 -0.05

Segment: core_interface 5
properties: 5 0 0 1 2
-0.025 -0.025 -0.025 0.025
-0.025 0.025 0.025 0.025
0.025 0.025 0.025 -0.025
0.025 -0.025 -0.025 -0.025

IBCS_Resistances:


Dielectrics:
1 2.2 -0.01 1.0 0.0
2 4.4 -0.10 1.0 0.0
