Title: TYPE 1 free impedance sheet

# Coordinate units: meters. Start with 2D at 1 GHz.
# Illustrative input example; use mesh convergence for production results.

Segment: impedance_card 1
properties: 1 0 10 0 0
-0.05 0 0.05 0

IBCS_Resistances:
10 constant 75 -20 0 0

Dielectrics:
