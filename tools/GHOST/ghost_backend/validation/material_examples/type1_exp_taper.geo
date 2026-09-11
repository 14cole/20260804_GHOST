Title: TYPE 1 exp impedance taper

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.

Segment: tapered_card 1
properties: 1 0 20 0 0
-0.05 0 0 0
0 0 0.05 0

IBCS_Resistances:
20 exp 5 1 160 32

Dielectrics:
