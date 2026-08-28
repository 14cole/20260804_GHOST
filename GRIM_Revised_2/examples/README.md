# GRIM Python examples

These scripts are runnable examples for common headless dataset workflows. They
use GRIM's production loaders, join logic, plot-data builders, and headless PNG
renderer. Run them from a source checkout as shown below, or from any directory
after installing the project with `python -m pip install -e .`.

All folder examples recognize the formats supported by `grim_headless`:
`.grim`, `.csv`, `.cst_data`, `.txt`, `.out`, `.pio`, `.cmplx_di`, `.ptm`, and
`.ss`. Use `--pattern "*.grim"` (or another glob) when a folder contains a mix
of files that should not all participate.

Each script has complete command-line help:

```powershell
python GRIM_Revised_2/examples/join_folder.py --help
python GRIM_Revised_2/examples/plot_folder_azimuth_sweeps.py --help
python GRIM_Revised_2/examples/plot_folder_frequency_sweeps.py --help
python GRIM_Revised_2/examples/query_dataset.py --help
```

## Join every dataset in a folder

```powershell
python GRIM_Revised_2/examples/join_folder.py `
  "./data/cuts" "./joined.grim" --pattern "*.grim"
```

`join_folder.py` sorts paths deterministically, excludes its own prior output,
and uses strict overlap checking by default. Complementary data and equivalent
duplicate samples join normally; conflicting finite samples stop the operation.
Only use `--overlap first` or `--overlap last` when sorted-path priority is an
intentional data decision. Add `--recursive`, `--workers 4`, or
`--max-output-gib 8` when appropriate. Existing output is preserved unless
`--overwrite` is explicit.

Join combines complementary coordinates into one grid. It is different from
Stitch, which deliberately resolves conflicting overlaps using a selected
policy.

## Cartesian azimuth sweeps from a folder

Create one overlaid PNG for every frequency common to the input datasets:

```powershell
python GRIM_Revised_2/examples/plot_folder_azimuth_sweeps.py `
  "./data/trade-study" --pattern "*.grim"
```

Choose exact fixed coordinates and a polarization:

```powershell
python GRIM_Revised_2/examples/plot_folder_azimuth_sweeps.py `
  "./data/trade-study" --frequency 8 --frequency 10 `
  --elevation 0 --polarization VV --output-dir "./azimuth-plots"
```

The plot is Cartesian (`azimuth_rect`), not polar. Repeat `--polarization` for
separate polarization plots. Use `--quantity phase` for phase. The selected
frequency and elevation values are in the first dataset's stored units;
`--angle-unit` and `--frequency-unit` select display units only. Fixed-axis
matching is exact within `--tol`; the script never interpolates.

## Frequency sweeps, with an optional azimuth percentile band

Plot an exact azimuth cut:

```powershell
python GRIM_Revised_2/examples/plot_folder_frequency_sweeps.py `
  "./data/trade-study" --azimuth 0 --elevation 0 --polarization VV
```

Plot P90 across an inclusive azimuth band at every frequency:

```powershell
python GRIM_Revised_2/examples/plot_folder_frequency_sweeps.py `
  "./data/trade-study" --azimuth-band -10 10 --percentile 90 `
  --elevation 0 --polarization VV --output-dir "./frequency-plots"
```

Omitting `--percentile` with a band uses P50. A descending band, such as
`170 -170` on a signed degree axis, crosses the periodic seam. The percentile
is calculated independently at each frequency in the displayed logarithmic RCS
domain, using only common stored azimuth samples. Periodic endpoint aliases are
counted once. Ordinary percentiles are not valid for wrapped phase, so phase
frequency sweeps require an exact `--azimuth`.

As with the azimuth script, coordinate selections use the first dataset's
stored units and there is no hidden interpolation or extrapolation.

## Query one sample by coordinate value

```powershell
python GRIM_Revised_2/examples/query_dataset.py "./result.grim" `
  --azimuth 45 --elevation 0 --frequency 10 --polarization VV
```

The query can be expressed in units different from the stored axes:

```powershell
python GRIM_Revised_2/examples/query_dataset.py "./result.grim" `
  --azimuth 1.5707963268 --elevation 0 --angle-unit rad `
  --frequency 9500 --frequency-unit MHz --polarization HH
```

The returned JSON shows the matched coordinates and the four indices. GRIM's
array order is always:

```text
[azimuth, elevation, frequency, polarization]
```

It also reports stored linear power, field magnitude, the dataset's normal dB
display value/unit, phase with its declared wrapping interval, and complex real
and imaginary parts when phase is known. Exact-with-tolerance matching is the
default. `--nearest` is deliberately explicit because silently snapping a
requested coordinate can select the wrong physical cut.

Use `--output sample.json` for a JSON artifact. An existing JSON file is kept
unless `--overwrite` is also supplied.

The essential direct access pattern is:

```python
indices = (azimuth_index, elevation_index, frequency_index, polarization_index)
power = dataset.rcs_power[indices]
phase_radians = dataset.rcs_phase[indices]
display_db = dataset.linear_to_default_db(
    power,
    frequency_value=dataset.frequencies[frequency_index],
)
```

Reading power and phase separately is important: a power-only dataset may have
valid power with unknown (`NaN`) phase, so reconstructing a complex value alone
would hide that usable magnitude sample.
