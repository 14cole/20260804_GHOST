Title: TYPE 1 thin dielectric strip

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.
# TYPE 1 is the midsurface; stored thickness is meters regardless of coordinate units.

Segment: thin_strip 1
properties: 1 80 10 0 0
-0.05 0 0.05 0

IBCS_Resistances:
10 thin_dielectric 0.0005 2

Dielectrics:
2 3.0 -0.02 1.0 0.0
