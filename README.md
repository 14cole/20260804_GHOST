# GRIM integrated RCS workbench

This branch is the single-folder distribution of GRIM, GHOST, and FREDDY.
GRIM is the main desktop application. Its tabs are **Plotting | ISAR |
Assembly | GHOST | FREDDY**. GHOST supplies the 2-D and body-of-revolution RCS
solvers; FREDDY supplies planar material-stack, impedance, reflection,
transmission, absorption, and material-mixing analysis.

## Folder layout

```text
GRIM/
  GRIM_Revised_2/       GRIM application, plotting, ISAR, and Assembly
  tools/
    GHOST/              GHOST solver, workflows, tests, and documentation
    FREDDY/             FREDDY planar material and impedance tool
  pyproject.toml
  README.md
```

Keep this tree together when copying it to another machine. Do not copy only
`GRIM_Revised_2`; the embedded tabs discover their authoritative tools under
`tools/GHOST` and `tools/FREDDY`.

## Install and run

From the top-level folder:

```powershell
py -m pip install -e .
grim
```

The integrated desktop distribution is intentionally run from this complete
source tree (or an editable install), because the GHOST and FREDDY tabs include
their tools, validation data, launchers, and workflows from `tools/`. A wheel
containing only the GRIM Python modules is not the single-folder distribution.

On Windows, `Launch_GRIM_GUI.bat` provides the same source-checkout launch and
prints the installation command if the selected Python is missing a dependency.

Standalone tool windows remain available. On Windows, use
`tools\GHOST\Launch_GHOST_GUI.bat` or
`tools\FREDDY\Launch_FREDDY_GUI.bat`. The equivalent direct commands are:

```powershell
cd tools\GHOST
py Backend\ghost_gui.py

cd ..\FREDDY
py impedance_gui.py
```

On macOS, each tool folder contains a matching `.command` launcher.

## Primary workflows

- Use **FREDDY** to analyze infinite planar material stacks or design a mixed
  material. Save either a nominal three-column IBC impedance CSV or a
  five-column material CSV for use by GHOST.
- Use **GHOST** to create and solve 2-D or axisymmetric BoR geometries. Solver
  `.grim` exports are loaded directly into GRIM.
- Use **Assembly → Place Features** to load the same strict placement CSV used
  by local/HPC feature workflows, map each dataset ID to an OPN-FRD GRIM, and
  preview an STL/facet or embedded BoR body with point/line locations before
  assembling. Exact headers, examples, and blank-template buttons are built in.
- Use **Plotting** and **ISAR** to inspect and process compatible RCS datasets.

FREDDY does not calculate finite-object RCS or produce `.grim` datasets.
Therefore its CSV outputs are not sent to GRIM's RCS dataset loader. Save an
IBC or material CSV beside the applicable GHOST `.geo` file, then select it
from the GHOST Geometry tab.

GRIM embeds the authoritative FREDDY implementation from `tools/FREDDY`.
`FREDDY_ROOT_PATH` is an optional development override for a different FREDDY
root. GRIM prevents the application from closing while a FREDDY background
calculation is running; those calculations are not cancellable mid-run.

Preview visibility controls affect only the Assembly 3-D display. They never
enable or disable a feature in the electromagnetic assembly.

## Component documentation

- GRIM usage and data conventions: `GRIM_Revised_2/README.md`
- GHOST solver and workflow guide: `tools/GHOST/README.md`
- HPC/local solver operation: `tools/GHOST/HPC.md`
- Geometry input format: `tools/GHOST/GEOMETRY_INPUT_CHEATSHEET.md`
- Point and line-feature validation: `tools/GHOST/FEATURE_VALIDATION_GUIDE.md`
- FREDDY scope, formats, and validation: `tools/FREDDY/README.md`

## Development checks

```powershell
py -m unittest discover -s GRIM_Revised_2 -p "test*.py" -v

cd tools\GHOST
py -m unittest discover -s tests -p "test*.py" -v

cd ..\FREDDY
py -m unittest discover -s tests -p "test*.py" -v
```

The GRIM host and standalone windows call the same authoritative GHOST and
FREDDY implementations. Embedding a tool does not create a second numerical
implementation.
