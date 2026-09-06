# Loadable material examples

These inputs accompany the [geometry cheat sheet](../../GEOMETRY_INPUT_CHEATSHEET.md).
They cover TYPE 1-5, including free impedance sheets, the new thin dielectric
layer, PEC, opaque IBC, bulk dielectric and magnetic material, explicit coatings,
two-material regions, spatial tapers, CSV tables, and legacy tables.

1. Load one `.geo` in GHOST Geometry.
2. Set geometry units to **meters** and select the **2D** solver.
3. Start at **1 GHz**, with a few angles such as 12, 48 and 86 degrees.
4. For production results, request mesh convergence and inspect the accuracy
   report. The thin-layer approximation also needs a physical applicability check.

Each file is a complete geometry. Keep the four material sidecars in this folder
when copying CSV or legacy examples. All tables span 0.8-1.2 GHz; CSV frequencies
are in Hz and `mat.*` frequencies are in GHz. The two formats describe the same
illustrative values. CSV headers may be omitted while retaining column order.

| Example files | Purpose |
|---|---|
| [type1_impedance_sheet.geo](type1_impedance_sheet.geo) | Free constant-impedance sheet |
| [type1_thin_dielectric.geo](type1_thin_dielectric.geo) | 0.5 mm transmitting layer; line is its midsurface |
| [type2_pec.geo](type2_pec.geo) | Ideal conductor with no material definition |
| [type2_ibc.geo](type2_ibc.geo) | Opaque conductor with complex surface impedance |
| [type3_bulk_dielectric.geo](type3_bulk_dielectric.geo) | Nonmagnetic dielectric in air |
| [type3_magnetic_dielectric.geo](type3_magnetic_dielectric.geo) | Isotropic complex epsilon and mu |
| [type4_pec_backed_coating.geo](type4_pec_backed_coating.geo) | Explicit coating with PEC core |
| [type4_ibc_backed_coating.geo](type4_ibc_backed_coating.geo) | Explicit coating with impedance core |
| [type5_two_dielectrics.geo](type5_two_dielectrics.geo) | Complete dielectric core, shell, and exterior air |
| [type1_linear_taper.geo](type1_linear_taper.geo), [type1_cosine_taper.geo](type1_cosine_taper.geo), [type1_exp_taper.geo](type1_exp_taper.geo) | Each supported spatial impedance taper |
| [type2_csv_ibc.geo](type2_csv_ibc.geo), [type3_csv_dielectric.geo](type3_csv_dielectric.geo) | Explicit nominal material CSV references |
| [type1_csv_thin_dielectric.geo](type1_csv_thin_dielectric.geo) | Fixed thickness with frequency-dependent epsilon and mu |
| [type2_legacy_ibc.geo](type2_legacy_ibc.geo), [type3_legacy_dielectric.geo](type3_legacy_dielectric.geo) | Older `mat.<flag>` references |

Most squares are 100 mm across. The explicit coating examples have an 80 mm
core; the two-dielectric example has a 50 mm core. All closed contours are
clockwise. The TYPE 5 core normal points into shell material 1; the core is
material 2. The thin strip is 100 mm long with thickness stored as 0.0005 meters.

For a FREDDY stack collapsed to a TYPE 2 boundary, use the separate
[PEC-backed coating workflow](../pec_backed_ibc/README.md). Its 30 mil examples
use **inches**, not the meters used here.

These are instructional fixtures with invented material values, not material
recommendations or certified scattering benchmarks. They are 2D cross sections;
BoR requires its own appropriate generating curve. Syntax, material loading,
round-trip serialization, and small two-polarization solves were checked on
September 5, 2026: all 17 files loaded and solved at 1 GHz for incidence and
observation angles 12, 48 and 86 degrees, yielding nine finite samples per
polarization. Production accuracy requires separate convergence.
