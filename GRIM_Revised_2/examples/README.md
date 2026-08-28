# GRIM Python examples

These are edit-and-run examples for common headless dataset workflows. There
are no command-line options. Open the desired script, change the clearly marked
`EDIT THESE SETTINGS` block near the top, save it, and run the file:

```powershell
python GRIM_Revised_2/examples/join_folder.py
python GRIM_Revised_2/examples/plot_folder_azimuth_sweeps.py
python GRIM_Revised_2/examples/plot_folder_frequency_sweeps.py
python GRIM_Revised_2/examples/query_dataset.py
```

The functions below each settings block remain importable when a larger Python
workflow needs to call them directly. Normal example use does not require
editing those functions.

All folder examples recognize the formats supported by `grim_headless`:
`.grim`, `.csv`, `.cst_data`, `.txt`, `.out`, `.pio`, `.cmplx_di`, `.ptm`, and
`.ss`. Set `FILE_PATTERN` or `INPUT_PATTERN` to `"*.grim"` (or another glob)
when only one file type should participate.

## Join every dataset in a folder

Edit [join_folder.py](join_folder.py):

```python
INPUT_FOLDER = Path(r"C:\data\cuts")
OUTPUT_FILE = Path(r"C:\data\joined.grim")
FILE_PATTERN = "*.grim"
SEARCH_SUBFOLDERS = False
PARALLEL_LOADERS = 1
OVERLAP_POLICY = "error"
COORDINATE_TOLERANCE = 1.0e-6
MAX_OUTPUT_GIB = None
OVERWRITE_OUTPUT = False
```

The script sorts paths deterministically, excludes its own prior output, and
uses strict overlap checking by default. Complementary data and equivalent
duplicate samples join normally; conflicting finite samples stop the operation.
Only set `OVERLAP_POLICY` to `"first"` or `"last"` when sorted-path priority is
an intentional data decision. Existing output is preserved unless
`OVERWRITE_OUTPUT` is `True`.

Join combines complementary coordinates into one grid. It is different from
Stitch, which deliberately resolves conflicting overlaps using a selected
policy.

## Cartesian azimuth sweeps from a folder

Edit [plot_folder_azimuth_sweeps.py](plot_folder_azimuth_sweeps.py):

```python
INPUT_FOLDER = Path(r"C:\data\trade_study")
INPUT_PATTERN = "*.grim"
OUTPUT_FOLDER = Path(r"C:\data\azimuth_plots")

# None means every common frequency. Values use the stored frequency unit.
FREQUENCIES = (8.0, 10.0)
ELEVATION = 0.0
POLARIZATIONS = ("VV",)

QUANTITY = "magnitude"
ANGLE_DISPLAY_UNIT = "deg"
FREQUENCY_DISPLAY_UNIT = "GHz"
```

The output is Cartesian (`azimuth_rect`), not polar. Multiple frequencies and
polarizations create separate PNGs, with all compatible folder datasets
overlaid in each plot. Set `QUANTITY = "phase"` for phase. `FREQUENCIES` and
`ELEVATION` use the first dataset's stored/native units; display-unit settings
only change axes and labels. Fixed-axis matching is exact within
`AXIS_MATCH_TOLERANCE`; the script never interpolates.

Leave `OUTPUT_FOLDER = None` to create `grim_azimuth_plots` under the input
folder. Existing PNGs receive a numeric suffix unless `OVERWRITE_EXISTING` is
`True`.

## Frequency sweeps with an optional azimuth percentile band

For an exact azimuth cut, edit
[plot_folder_frequency_sweeps.py](plot_folder_frequency_sweeps.py) like this:

```python
AZIMUTH = 0.0
AZIMUTH_BAND = None
AZIMUTH_PERCENTILE = None
ELEVATION = 0.0
POLARIZATIONS = ("VV",)
```

For P90 across an inclusive azimuth band at every frequency:

```python
AZIMUTH = None
AZIMUTH_BAND = (-10.0, 10.0)
AZIMUTH_PERCENTILE = 90.0
ELEVATION = 0.0
POLARIZATIONS = ("VV",)
```

When a band is active, `AZIMUTH_PERCENTILE = None` uses P50. A descending band,
such as `(170.0, -170.0)` on a signed degree axis, crosses the periodic seam.
The percentile is calculated independently at each frequency in the displayed
logarithmic RCS domain, using only common stored azimuth samples. Periodic
endpoint aliases are counted once. Ordinary percentiles are not meaningful for
wrapped phase, so phase frequency sweeps require an exact `AZIMUTH`.

Coordinate selections use the first dataset's stored units. There is no hidden
interpolation or extrapolation.

## Query one sample by coordinate value

Edit [query_dataset.py](query_dataset.py):

```python
DATASET_PATH = Path(r"C:\data\result.grim")

QUERY_AZIMUTH = 45.0
QUERY_ELEVATION = 0.0
QUERY_FREQUENCY = 10.0
QUERY_POLARIZATION = "VV"
QUERY_ANGLE_UNIT = "deg"
QUERY_FREQUENCY_UNIT = "GHz"

NEAREST_MATCH = False
ANGLE_TOLERANCE = None
FREQUENCY_TOLERANCE = None
JSON_OUTPUT_PATH = None
OVERWRITE_JSON = False
```

Query units may differ from the stored axes; the script converts them before
searching. The printed JSON shows the matched coordinates and all four indices.
GRIM's array order is always:

```text
[azimuth, elevation, frequency, polarization]
```

The result also includes stored linear power, field magnitude, the dataset's
normal dB display value/unit, phase with its declared wrapping interval, and
complex real and imaginary parts when phase is known. Exact-with-tolerance
matching is the default. Set `NEAREST_MATCH = True` only when deliberately
snapping to the closest stored coordinate.

Set `JSON_OUTPUT_PATH` to a `Path` to save the printed result. An existing JSON
file is preserved unless `OVERWRITE_JSON` is `True`.

The essential direct-access pattern is:

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
