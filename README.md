# GRIM integrated RCS workbench

This branch is the single-folder distribution of GRIM and GHOST. GRIM is the
main desktop application. Its tabs are **Plotting | ISAR | Assembly | GHOST**;
the GHOST tab embeds the same 2-D and body-of-revolution solver used by the
standalone GHOST application. FREDDY has a reserved tool location so it can be
integrated as another GRIM tab without creating another checkout.

## Folder layout

```text
GRIM/
  GRIM_Revised_2/       GRIM application, plotting, ISAR, and Assembly
  tools/
    GHOST/              GHOST solver, workflows, tests, and documentation
    FREDDY/             reserved integration point for FREDDY
  pyproject.toml
  README.md
```

Keep this tree together when copying it to another machine. In particular,
do not copy only `GRIM_Revised_2`; the bundled GHOST tab discovers its one
authoritative solver backend under `tools/GHOST/Backend`.

## Install and run

From the top-level folder:

```powershell
py -m pip install -e .
grim
```

On Windows, `Launch_GRIM_GUI.bat` provides the same source-checkout launch and
prints the installation command if the selected Python is missing a dependency.

For solver-only work, the standalone GHOST window remains available:

```powershell
cd tools\GHOST
py Backend\ghost_gui.py
```

The GHOST launchers in `tools/GHOST` also set their own working directory, so
they can be double-clicked without moving files.

## Primary workflows

- Use **GHOST** to create and solve 2-D or axisymmetric BoR geometries. Solver
  exports are loaded directly into GRIM.
- Use **Assembly** to map point or line-feature dataset IDs to OPN-FRD GRIM
  files, validate placement CSVs, preview the body and feature locations in
  3-D, and assemble the output.
- Use **Plotting** and **ISAR** to inspect and process compatible RCS datasets.

Preview visibility controls affect only the 3-D display. They never enable or
disable a feature in the electromagnetic assembly.

## Component documentation

- GRIM usage and data conventions: `GRIM_Revised_2/README.md`
- GHOST solver and workflow guide: `tools/GHOST/README.md`
- HPC/local solver operation: `tools/GHOST/HPC.md`
- Geometry input format: `tools/GHOST/GEOMETRY_INPUT_CHEATSHEET.md`
- Point and line-feature validation: `tools/GHOST/FEATURE_VALIDATION_GUIDE.md`

## Development checks

```powershell
py -m unittest discover -s GRIM_Revised_2 -p "test*.py" -v
cd tools\GHOST
py -m unittest discover -s tests -p "test*.py" -v
```

The GRIM host, standalone GHOST GUI, command-line/HPC drivers, and Assembly
feature service all call the same solver and feature-physics implementation.
There is no duplicate embedded solver.
