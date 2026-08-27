# Wedge-to-Conic physical convention

## Configurations

The source acquisition is a vertical-axis turntable with the article pitched
relative to the pylon. With turntable angle `phi` and body-y wedge pitch `tau`,
the body-to-world attitude is

```text
R_source = Rz(phi) Ry(tau).
```

The desired normal-range configuration tilts the article and pylon together,
then rotates the assembly. Its data are represented in ordinary conic
longitude/elevation coordinates with the spherical vertical/horizontal basis.

For a fixed world line of sight along `+x`, the source look direction expressed
in body coordinates is

```text
r_body = (cos(tau) cos(phi), -sin(phi), sin(tau) cos(phi)).
```

Therefore

```text
longitude = atan2(r_y, r_x)
latitude  = asin(r_z).
```

This direction mapping was checked independently against the complete attitude
matrix `R_source.T @ (1,0,0)`.

## Polarization transformation

The range V/H basis in the source body frame is obtained by transforming fixed
world vertical and horizontal vectors:

```text
V_wedge = R_source.T (0,0,1)
H_wedge = R_source.T (0,1,0).
```

The output uses the standard conic spherical basis `(V_conic, H_conic)`. If
`C[i,a] = dot(wedge_basis[i], conic_basis[a])`, the monostatic Jones matrix is
transformed coherently as

```text
S_conic = C.T S_wedge C.
```

The implementation interpolates the complex source Jones matrix in measured
`(phi,tau)` coordinates, then applies this basis transformation at every output
query. It never interpolates dB values or wrapped phase.

## Data requirements and limitations

- A complete turntable revolution and at least two measured wedge tilts are
  required.
- A single fixed wedge tilt is a curved, pinched path on the body sphere. It
  cannot determine a constant-elevation normal azimuth cut.
- Finite phase is required for every finite sample.
- VV and HH plus at least one of VH/HV are required. For monostatic reciprocal
  data, one measured cross-pol supplies the other. Co-pol-only conversion is
  permitted only when the user explicitly assumes missing cross-pol is zero.
- The wedge acquisition cannot cover every normal conic look. In particular,
  nonzero-elevation side aspects require wedge pitch approaching +/-90 degrees.
  Unsupported output cells remain `NaN`; there is no extrapolation.
- The dialog requires the user to attest that its axes follow the physical
  convention above. Datasets explicitly tagged `wedge_turntable` satisfy the
  same convention programmatically.

The independent release regressions are in `test_wedge_conversion.py` and use
an object-fixed complex dyadic oracle rather than replaying the production
formulas.
