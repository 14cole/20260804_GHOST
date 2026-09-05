# PEC-backed FREDDY stack as a GHOST IBC

This workflow already uses the scalar conductor IBC supported by **both 2D
and BoR**. It does not use the TYPE 1 freestanding thin dielectric sheet.
FREDDY collapses the planar stack plus PEC backing into its front-face Z(f).
GHOST replaces the coating and conductor interior with that boundary law.

## Example and procedure

The `example/` folder contains **illustrative**, frequency-independent
epsilon 4 - j0.1 and mu 1. It is not a measured user material. Total coating
thickness is 0.03 inches = 30 mil = 0.762 mm. The two example geometries are a
2-inch-radius PEC core with a coating, represented by a single outer boundary
at 2.03 inches. The CSV covers 1-18 GHz in 0.1 GHz steps.

1. In FREDDY, add your measured material layers in order from air toward PEC.
   Enter each physical thickness; use 0.03 inches if that is the total thickness
   of a single layer. Select **Impedance**, frequency range 1-18 GHz, and
   **PEC** backing. Material input tables must cover the full requested band.
2. Use **Check GHOST coating approximation**. It compares the scalar normal-
   incidence IBC with the full planar stack for TE and TM at 0, 15, 30, 45,
   60, 75 and 85 degrees. It also checks normal-incidence reflection at CSV
   interpolation midpoints. It neither changes the export nor certifies RCS.
3. Compute the nominal IBC CSV. Use **Export and attach to current GHOST
   geometry**, after loading/saving the receiving `.geo` file. The handoff
   registers the material; segment assignment is a separate visible action.
4. In GHOST draw the **outer air/coating envelope**, select its TYPE 2 segments
   and the attached surface-material row, then click **Apply IBC to selected
   TYPE 2 segments**. Both region flags stay zero. Save the geometry.
5. Run 2D or BoR, using the corresponding geometry file and coordinate units.
   The examples use **inches**. Start at 1 GHz to inspect the setup before a
   larger sweep. Production mesh/mode checks remain necessary.

For a surface-material flag of 10, the `.geo` entries are:

```text
properties: 2 0 10 0 0

IBCS_Resistances:
10 example_coating_30mil.csv

Dielectrics:
```

Keep the nominal CSV beside the geometry. Its columns are frequency in Hz,
resistance in ohms and reactance in ohms. GHOST interpolates real/imaginary Z
linearly and refuses out-of-range frequencies. Refine frequency sampling near
resonances; a midpoint check cannot exclude narrower missed resonances.

## Reference surface and accuracy

The exported impedance is evaluated at the **front of the complete stack**.
Do not place it at the original metal surface without a justified reference-
plane transformation. Do not add a second coincident PEC contour or retain the
collapsed bulk coating interfaces. The assignment button changes material
flags only; it does not offset coordinates or repair a coating envelope.

For BoR, geometry is an axisymmetric generating curve; coating properties
must preserve axisymmetry. A general three-dimensional ten-foot body is not a
BoR model unless it has the required rotational symmetry.

The exported IBC retains the frequency response of the planar stack at normal
incidence. It does not retain independent angle-dependent TE/TM impedances,
tensor material orientation, curvature corrections or finite-edge physics.
The new check refuses anisotropic layers because a scalar boundary cannot
represent their general response. It is not enough that a coating is thin
compared with the body dimensions.

For the illustrative 30 mil material over 1-18 GHz, the planar check gives a
worst TM reflection phase difference of about 11.5 degrees at 60-degree
incidence and 16.8 degrees at 75 degrees. TE differences are much smaller in
this example. These are local reflection errors, not finite-body RCS errors;
power-only agreement can hide phase error important to Assembly interference.

This approach avoids explicit coating interfaces and their closely spaced
interaction integrals. It does **not** eliminate the dense matrix cost of the
large outer surface. Ten feet is about 10 wavelengths at 1 GHz and 183 at
18 GHz. Use GHOST's resource preview before a large run. The low-order sheet
model's electrical-thickness gate does not apply to FREDDY's full planar
stack cascade, but a scalar Z(f) still needs its own approximation assessment.

## Verification

This update's focused suites passed: 21 FREDDY tests, 17 GHOST stack/handoff
tests and 12 integration/layout tests. The new controls and report were
visually checked in the integrated application at 1366 by 768.

- `tests/test_pec_stack_ibc.py` sends actual FREDDY CSV output through both
  numerical solvers at 1, 9.5 and 18 GHz and compares against independent
  impedance-cylinder/sphere series. This validates the IBC handoff and solver
  law on small bodies, not equality with the original bulk coating.
- `tools/FREDDY/tests/test_ghost_coating.py` independently checks oblique
  short-circuited slab formulas, broadside equivalence and interpolation error.
- In one local 1 GHz 2D benchmark (80 mm PEC radius, 30 mil example coating,
  80 segments per surface, 13 aspects, both polarizations, condition estimates),
  the scalar IBC took 0.12 s versus 7.04 s for explicit coating geometry.
  Worst power error against the coated-cylinder analytic solution was 0.034 dB
  for the IBC and 0.010 dB for explicit bulk. This is one small-body run, not a
  certified runtime or error bound for a ten-foot body at 18 GHz. The script
  and raw measurements are in `critique-review/probe_pec_stack.py/.json` in the
  parent workspace.

Regenerate illustrative artifacts into a **new** folder from the repo root:

```powershell
.venv/Scripts/python.exe tools/GHOST/geometry_tests/pec_backed_ibc/generate_examples.py path/to/new-example-folder
```
