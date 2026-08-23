# GRIM application

GRIM is the host application for this distribution. Its desktop tabs are
**Plotting | ISAR | Assembly | GHOST**. The future FREDDY integration will be
another host tab; FREDDY's implementation belongs under `tools/FREDDY`, not in
the GRIM plotting code.

Run GRIM from the repository root after an editable installation:

```powershell
py -m pip install -e .
grim
```

## Assembly

Assembly has one canonical tree and one 3-D preview. Point and line feature
CSV files are selected in the form; each discovered `dataset_id` receives one
explicit OPN-FRD `.grim` mapping. The preview uses the vehicle CAD frame:
`+x` right, `+y` nose, and `+z` up.

The tree's **Show** controls and global **Show All** affect preview artists
only. They do not include or exclude a feature from the electromagnetic
assembly. A body mesh used for shadowing is likewise kept at full physics
resolution even if its display is decimated.

Point datasets require compatible 3-D delta channels (VV, HH, and reciprocal
cross-polarization where used). Line datasets require the TE and TM 2-D delta
responses consumed by line expansion. Shadowing is geometric blockage; it
does not add diffraction, creeping waves, or body-feature multiple scattering.

## GHOST integration

The GHOST tab loads the authoritative backend from
`tools/GHOST/Backend`. `GHOST_BACKEND_PATH` remains an optional development
override. Solver exports are routed into GRIM's existing dataset loader.

## Headless interface

```powershell
grim-headless a.grim b.grim --operation coherent-add -o sum.grim
grim-headless --folder results --pattern "*.grim" --operation join -o joined.grim
```

Coherent operations require compatible axes, units, phase reference,
polarizations, and dimensional RCS quantity. A 2-D `sigma_2d` field cannot be
coherently added directly to a 3-D `sigma_3d` body; it must first go through
the line-expansion placement workflow.

## Tests

From the repository root:

```powershell
py -m unittest discover -s GRIM_Revised_2 -p "test*.py" -v
```
