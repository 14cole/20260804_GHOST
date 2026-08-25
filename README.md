# GRIM integrated RCS workbench

This branch is the single-folder distribution of GRIM, GHOST, and FREDDY.
GRIM is the main desktop application. Its tabs are **Plotting | ISAR | PPT |
Assembly | GHOST | FREDDY | Runs | Python**. GHOST supplies the 2-D and body-of-revolution RCS
solvers; FREDDY supplies planar material-stack, impedance, reflection,
transmission, absorption, and material-mixing analysis. PPT builds uniform,
previewed PowerPoint reports from loaded RCS datasets.

## Folder layout

```text
GRIM/
  GRIM_Revised_2/       GRIM application, plotting, ISAR, PPT, and Assembly
  tools/
    GHOST/              GHOST solver, workflows, tests, and documentation
    FREDDY/             FREDDY planar material and impedance tool
  clean_utf8.py         In-place recovery for text-mode copy damage
  pyproject.toml
  README.md
```

Keep this tree together when copying it to another machine. Do not copy only
`GRIM_Revised_2`; the embedded tabs discover their authoritative tools under
`tools/GHOST` and `tools/FREDDY`.

## Build a copy-ready release

On Windows, double-click `Build_GRIM_Release.bat`. From any operating system,
run `python -m build_release` at the top level (`python3 -m build_release` when
Python 3 uses that command name). The standard-library-only builder reads the
version from `pyproject.toml` and creates these items under `dist/`:

```text
GRIM-<version>/
GRIM-<version>.zip
GRIM-<version>-SHA256SUMS.txt
```

The ZIP keeps the complete GRIM, GHOST, and FREDDY source layout, assets, and
launchers, while omitting Git data, virtual environments, Python/test caches,
and temporary files. `SHA256SUMS.txt` is also inside the release folder and
ZIP. The external manifest verifies both the ZIP and every extracted payload
file. Existing release artifacts are never overwritten, and an incomplete
source tree fails before output is created.

Copy only the ZIP (and preferably its adjacent checksum manifest) to the other
machine, extract it, then follow **Install and run** below. Git is not needed on
the destination machine.

## Repair a damaged manual copy

If a text-mode copy changed source or data bytes and Python reports
`'utf-8' codec can't decode byte ...`, copy only `clean_utf8.py` into the
top-level folder on the destination machine. Preview the affected files first:

```powershell
# Windows PowerShell
py -3 .\clean_utf8.py .
```

```bash
# Linux
python3 ./clean_utf8.py .
```

Then repeat the applicable command with `--apply`. The cleaner strictly
converts recognized Windows-1252 and UTF-16 text to UTF-8, preserves line
endings and executable permissions, verifies each replacement, and writes a
recovery ZIP beside the scanned folder before changing anything. It leaves
already-valid UTF-8 byte-for-byte unchanged and skips symlinks, environments,
caches, build output, and binary or mixed formats such as `.grim`, `.ptm`,
`.ss`, `.pio`, `.cmplx_di`, and `.stl`. Use `--include-extension EXT` only for
an additional format that is known to be text.

If the copied folder contains `SHA256SUMS.txt`, repaired files will no longer
match that original release manifest; the cleaner reports this after applying
changes. Future transfers should use the release ZIP, `scp`, `rsync`, or
WinSCP binary mode so bytes are not re-encoded in transit.

## Install and run

From the top-level folder:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m grim_cut_gui
```

All Windows and macOS launchers prefer this one repository-root `.venv`, then
an active `VIRTUAL_ENV`, then a system Python. This keeps the integrated and
standalone windows on the same dependency set. On macOS, use
`Launch_GRIM_GUI.command` (and run `chmod +x Launch_GRIM_GUI.command` once if a
manual file copy did not preserve executable permission).

To export `.pptx` files on Windows, install the optional PowerPoint bridge and
have desktop Microsoft PowerPoint available:

```powershell
py -m pip install -e ".[powerpoint]"
```

The PPT slide preview works without PowerPoint; only final `.pptx` generation
uses the desktop application.

The integrated desktop distribution is intentionally run from this complete
source tree (or an editable install), because the GHOST and FREDDY tabs include
their tools, validation data, launchers, and workflows from `tools/`. A wheel
containing only the GRIM Python modules is not the single-folder distribution.

On Windows, `Launch_GRIM_GUI.bat` provides the same source-checkout launch and
prints the installation command if none of the preferred Python interpreters
has the required dependencies.

After copying the folder to a machine, run `Launch_GRIM_Diagnostics.bat` for a
read-only installation check. After an editable install, the equivalent text
commands are `grim-diagnose` or `py -3 -m grim_diagnostics`. The report checks
the authoritative GRIM, GHOST, and FREDDY paths and the required GUI/solver
dependencies. It labels PowerPoint export and platform-native GHOST
acceleration as optional. It starts neither a solver nor PowerPoint, changes no
files, and returns a nonzero exit code only for a required startup blocker.

Standalone tool windows remain available. On Windows, use
`tools\GHOST\Launch_GHOST_GUI.bat` or
`tools\FREDDY\Launch_FREDDY_GUI.bat`. The equivalent direct commands are:

```powershell
cd tools\GHOST
py Backend\ghost_gui.py

cd ..\FREDDY
py impedance_gui.py
```

On macOS, each tool folder contains a matching standalone `.command` launcher.

## Primary workflows

- Use **FREDDY** to analyze infinite planar material stacks or design a mixed
  material. After exporting a nominal PEC-backed three-column IBC CSV or a
  nominal five-column material CSV, choose **Export and attach to current GHOST
  geometry**. The active GHOST geometry must already be loaded or saved as a
  `.geo`; GRIM validates and copies the CSV beside it. Press **Save Geometry**
  in GHOST to persist the new reference. Off-angle, thickness, uncertainty,
  and other analysis CSVs are deliberately excluded from this handoff.
- Use **GHOST** to create and solve 2-D or axisymmetric BoR geometries. Solver
  `.grim` exports are loaded directly into GRIM.
- Use **Runs** to export a hash-verified portable HPC request or submit it
  directly from Windows to a headless Linux SLURM machine over OpenSSH or a
  saved PuTTY session. Direct mode builds the request from the visible form,
  uploads it, creates Linux-native provenance, records the returned job IDs,
  and can refresh status, show the submission log, cancel, or download results.
  An interrupted submission is recovered from its deterministic remote
  `stage_result.json` instead of being blindly resubmitted. A configurable
  remote Python path supports cluster virtual environments, and downloads are
  terminal-state-only and collision-safe.
  GRIM stores no SSH password or private-key contents. The exported bundle and
  its README remain the fallback for VPN, MFA, or site-policy restrictions.
- Use **Assembly → Place Features** to load the same strict placement CSV used
  by local/HPC feature workflows, map each dataset ID to an OPN-FRD GRIM, and
  preview an STL/facet or embedded BoR body with point/line locations before
  assembling. Exact headers, examples, and blank-template buttons are built in.
  The display-only viewer supports meter/inch/foot axes, solid/wireframe body
  styles, opacity, and bounded/adaptive facet detail for responsive rotation.
  It also draws normalized, scene-scaled orientation arrows: magenta for point
  and line-endpoint normals and lavender for the solver-effective projected
  point roll/local `+x` direction. These arrows are visual QA only.
- Use **Plotting** and **ISAR** to inspect and process compatible RCS datasets.
- Use **Python** to copy or save the readable headless script assembled from
  successful dataset operations and supported rectangular/polar azimuth,
  frequency, and elevation-sweep plot creation/export actions. PBP, Hold
  overlays, and other plot modes are identified in comments instead of being represented as
  falsely equivalent runnable code. Navigation, selection gestures, zoom/pan,
  and the PPT, Assembly, GHOST, FREDDY, and Runs workflows are not recorded.
- Use **PPT** to check loaded datasets independently of the Plotting selection,
  choose rectangular/polar azimuth plots or a frequency sweep, and review the
  actual 16:9 slide layout before export. Optional templates must also be blank
  widescreen 16:9 decks. Azimuth reports always use six fixed
  positions per slide (3 columns × 2 rows); frequency sweeps use one fixed
  full-slide plot. A shared vertical scale and fixed image rectangles prevent
  plots from shifting when slides change or when reports come from different
  analysts. The first six common frequencies are selected initially; one report
  is capped at 60 frequencies (10 azimuth slides) to keep the preview responsive.

The separately shipped **PowerPoint Image Imprinter** is a legacy/manual deck
formatting helper, not another report generator. It copies the position, size,
and optional crop of pictures selected in an already-open desktop PowerPoint
deck. New GRIM dataset reports should use the integrated **PPT** tab; keep the
imprinter only for aligning images in an existing custom presentation.

FREDDY does not calculate finite-object RCS or produce `.grim` datasets.
Therefore its CSV outputs are not sent to GRIM's RCS dataset loader. The
explicit nominal-artifact handoff above attaches only validated GHOST material
inputs; it never treats a FREDDY analysis CSV as an RCS dataset.

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
py -W error -m unittest -v test_clean_utf8.py

py -m unittest discover -s GRIM_Revised_2 -p "test*.py" -v

cd tools\GHOST
py -m unittest discover -s tests -p "test*.py" -v

cd ..\FREDDY
py -m unittest discover -s tests -p "test*.py" -v
```

The GRIM host and standalone windows call the same authoritative GHOST and
FREDDY implementations. Embedding a tool does not create a second numerical
implementation.
