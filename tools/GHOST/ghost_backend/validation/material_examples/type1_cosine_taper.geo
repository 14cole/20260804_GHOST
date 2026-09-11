Title: TYPE 1 cosine impedance taper

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.

Segment: tapered_card 1
properties: 1 0 20 0 0
-0.05 0 0 0
0 0 0.05 0

IBCS_Resistances:
20 cosine 0 0 376.73 0

Dielectrics:
