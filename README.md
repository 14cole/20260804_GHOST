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

On 64-bit x86 Windows with CPython 3.12, double-click
`Build_GRIM_Release.bat` or run `python -m build_release` at the top level.
The release gate deliberately refuses another operating system or architecture
because it must exercise the actual supported Windows stack. Build from an exactly clean, committed Git
checkout whose `HEAD` has tag `v<version>` or `<version>` matching
`pyproject.toml`. The builder packages only Git-tracked files, runs the
complete release acceptance gate, and creates these items under `dist/`:

```text
GRIM-<version>/
GRIM-<version>.zip
GRIM-<version>-SHA256SUMS.txt
```

The gate verifies every package in the exact Windows Python dependency lock is
installed at the reviewed version, strict UTF-8,
prohibited release terms, startup diagnostics, the GRIM/GHOST/FREDDY unit
tests, the separate GHOST CEM-tools suite, the standalone GHOST HPC-scheduling
and local-driver integration tests, the GHOST ASCII-transfer check,
and the selected native-acceleration policy. Missing native acceleration is a
recorded warning by default; pass `--native-policy require` for a
performance-ready build that must contain the matching native binaries.

The ZIP keeps the complete GRIM, GHOST, and FREDDY tracked source layout,
assets, and launchers while omitting Git data, untracked files, virtual
environments, caches, and temporary files. `BUILD-INFO.json` records the source
commit and release tag, deterministic source-tree digest, dependency-lock
digest, gate status, target runtime, and build ID. `SHA256SUMS.txt` is also inside the release folder
and ZIP. The external manifest verifies both the ZIP and every extracted
payload file. Each artifact is exposed atomically, with the external manifest
published last as the completion marker. Existing release artifacts are never
overwritten, and an incomplete or dirty source tree fails before output is
created.

For a reviewed source export that is not inside any Git worktree, provide a
newline-delimited allowlist with `--source-inventory PATH`. Runtime modules and
acceptance tests present in that export are mandatory inventory entries. There
is no CLI option to bypass the acceptance gate.

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
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install `
  -r requirements\windows-py312.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation `
  -c requirements\constraints-windows-py312.txt -e .
.venv\Scripts\python.exe -m grim_cut_gui
```

For repeatable offline team installation, prepare the locked wheelhouse once
and install with `--no-index` as described in `requirements/README.md`. Release
creation itself never downloads or resolves packages from the network.

All Windows launchers prefer this one repository-root `.venv`, then an active
`VIRTUAL_ENV`, then a system Python. This keeps the integrated and standalone
windows on the same dependency set.

To export `.pptx` files on Windows, install the optional PowerPoint bridge and
have desktop Microsoft PowerPoint available:

```powershell
.venv\Scripts\python.exe -m pip install `
  -c requirements\constraints-windows-py312.txt -e ".[powerpoint]"
```

The PPT slide preview works without PowerPoint; only final `.pptx` generation
uses the desktop application. Export closes only GRIM's temporary report
presentation, never issues PowerPoint's application-wide **Quit** command, and
leaves presentations already open in PowerPoint running.

GRIM ships an editable temporary template at
`GRIM_Revised_2/templates/GRIM_Report_Template.pptx`. When present, the PPT tab
selects it automatically and applies **GRIM Azimuth 3x2** or **GRIM Frequency
Sweep** from **GRIM Report Master** to the corresponding report slides. Layout
selectors are editable and accept `Master :: Layout` to disambiguate repeated
names in a team template. Clearing the template path restores fresh-deck
export.

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

## Primary workflows

- Use **FREDDY** to analyze infinite planar material stacks or design a mixed
  material. Its read-only **Material Explorer** compares permittivity,
  permeability, coverage, and loss tangent across five-column material CSVs
  without changing the active stack. After exporting a nominal PEC-backed
  three-column IBC CSV or a
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
  its README remain the fallback for VPN, MFA, or site-policy restrictions. If
  Plink reports that it cannot answer an interactive prompt in batch mode, run
  the same saved session once from interactive Plink to identify the prompt;
  configure a verified host key plus Pageant/key authentication, or reuse an
  authenticated PuTTY session with SSH connection sharing enabled. The detailed
  commands are in `GRIM_Revised_2/README.md`.
- Use **Assembly → Place Features** to load the same strict placement CSV used
  by local/HPC feature workflows, map each dataset ID to an OPN-FRD GRIM, and
  preview an STL/facet or embedded BoR body with point/line locations before
  assembling. Exact headers, examples, and blank-template buttons are built in.
  One point CSV may contain every point family and one line CSV may contain all
  ordered line chains; both files use the shared coordinate-unit selection.
  Here OPN-FRD always means the coherent installed/featured response minus the
  clean-skin response—reversing that subtraction reverses the feature delta.
  The display-only viewer supports meter/inch/foot axes, solid/wireframe body
  styles, opacity, and bounded/adaptive facet detail for responsive rotation.
  It also draws normalized, scene-scaled orientation arrows: magenta for point
  and line-endpoint normals and lavender for the solver-effective projected
  point roll/local `+x` direction. These arrows are visual QA only.
  After either placement CSV is parsed, **Spatial Feature Configuration → Use**
  exposes the clean body, feature families, response IDs, and individual point
  or line instances as a hierarchy. A family or instance that is unchecked is
  omitted consistently from preview, physical validation, response loading,
  and assembly, making clean/featured and feature-on/feature-off trade studies
  possible without deleting CSV rows. The hierarchy can be searched by
  instance, response ID, or response filename; its exact selection can be
  copied for a trade-study record. With every feature unchecked, **Preview
  geometry** shows the clean body alone, while validation/build continue to
  require an enabled feature. **Preview Layers → Show** remains a display-only
  control and never changes the calculated response.
  A validated preview is reused by **Assemble & Save** while every path, option,
  and source-file fingerprint remains unchanged. Existing outputs require
  confirmation, and the clean body or a mapped response can never be selected
  as the output target. Named portable `.assembly.json` recipes retain exact
  mappings, tolerances, and feature membership for trade studies and warn when
  referenced inputs change. The linked placement-QA table, signed line-frame
  arrows, mesh-topology report, and optional strict feature-library manifests
  make assumptions visible before a build. Long builds report progress and can
  be cooperatively cancelled without publishing a partial artifact or replacing
  a prior output.
- Use **Plotting** and **ISAR** to inspect and process compatible RCS datasets.
- Use **Python** to copy or save the readable headless script assembled from
  successful dataset operations and supported rectangular/polar azimuth,
  frequency, and elevation-sweep plot creation/export actions. PBP, Hold
  overlays, and other plot modes are identified in comments instead of being represented as
  falsely equivalent runnable code. Navigation, selection gestures, zoom/pan,
  and the PPT, Assembly, GHOST, FREDDY, and Runs workflows are not recorded.
- Use **PPT** to check loaded datasets independently of the Plotting selection,
  choose rectangular/polar azimuth plots or a frequency sweep, and review the
  actual 16:9 slide layout before export. **VV and HH** produces separate
  co-polar plots in one report. Frequency traces can use an exact azimuth or a
  finite-sample percentile across an inclusive, optionally seam-wrapped azimuth
  band. The percentile is sample-weighted across common stored directions;
  periodic endpoint aliases count once. Optional templates must be widescreen
  16:9 decks. GRIM's bundled temporary template includes representative seed
  slides and named azimuth/frequency custom layouts; export inherits the
  master/layout and removes the positioning-guide seeds. Only master/layout
  styling persists. Template master graphics appear only in the exported deck,
  and a named layout's title placeholder keeps its master formatting; plot and
  legend images remain at GRIM's documented fixed coordinates.
  Azimuth reports always use six fixed
  positions per slide (3 columns × 2 rows); frequency sweeps use one fixed
  full-slide plot. A master dataset legend can span the header beneath the
  aligned title box, with per-plot and no-legend alternatives. Optional fixed
  horizontal and vertical minimum/maximum/step controls apply the same axes to
  every plot without resampling data. Footer and page furniture come from the
  slide master. Fixed image rectangles prevent plots from
  shifting when slides change or when reports come from different analysts. The
  first six common frequencies are selected initially; one report is capped at
  60 frequencies (10 azimuth slides) to keep the preview responsive.

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
- Non-BoR clean/featured validation ladder:
  `tools/GHOST/geometry_tests/non_bor_feature_validation/README.md`
- Curved non-BoR and shared-facet placement regression:
  `tools/GHOST/geometry_tests/non_bor_curved_feature_placement/README.md`
- FREDDY scope, formats, and validation: `tools/FREDDY/README.md`

## Development checks

```powershell
py -W error -m unittest -v test_clean_utf8.py

cd requirements
py -m unittest discover -s . -p "test*.py" -v
cd ..

py -m unittest discover -s GRIM_Revised_2 -p "test*.py" -v

cd tools\GHOST
py -m unittest discover -s tests -p "test*.py" -v

cd ..\FREDDY
py -m unittest discover -s tests -p "test*.py" -v
```

The GRIM host and standalone windows call the same authoritative GHOST and
FREDDY implementations. Embedding a tool does not create a second numerical
implementation.
