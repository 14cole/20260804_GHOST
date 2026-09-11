# GHOST and FREDDY material/IBC file format

Material and surface-impedance files use `.csv`, comma-separated numeric values,
and frequency in **Hz**. Both tools require the headers below. Sweep controls
and plots display GHz; conversion happens only at the file boundary.

## Bulk material

```csv
frequency_hz,eps_real,eps_imag,mu_real,mu_imag
1000000000,3.2,-0.15,1,0
2000000000,3.1,-0.12,1,0
```

Epsilon and mu are relative complex properties. Under the `e^(+j omega t)`
convention, passive loss uses negative imaginary values; zero is lossless.
Use this schema for FREDDY layer inputs, material-mix exports, and GHOST bulk
material inputs. A FREDDY material export can be reloaded in either tool.

## Nominal IBC / surface impedance

```csv
frequency_hz,resistance_ohm,reactance_ohm
1000000000,120,15
2000000000,125,-10
```

Surface impedance is `resistance_ohm + j*reactance_ohm`. Resistance must be
nonnegative; reactance may have either sign. FREDDY's Impedance and IBC Batch
exports use this schema and can be assigned directly to a GHOST IBC boundary.

FREDDY uncertainty reports add bound columns in a separate `_uncertainty.csv`.
Off-angle and thickness exports are also CSV in Hz, with their own required
headers. These analysis reports are not nominal material or IBC inputs.

## Parsing and export rules

- Use the exact lowercase header names and column order shown above.
- Frequency must be finite, positive, and unique. `1e9` and `1000000000`
  both mean 1 GHz. Input rows may be unordered; readers sort by frequency.
- Every data row must have exactly the expected number of finite numeric
  fields. Scientific notation and spaces around individual cells are accepted.
- UTF-8 with or without a BOM is accepted, including Windows CRLF line endings.
- Blank lines and full-line `#` comments are allowed. Inline comments are not.
- Real and imaginary components interpolate linearly. Extrapolation is rejected.
- Exports always include a header and retain 17 significant digits for nominal
  material and impedance data. Closely spaced frequencies remain distinct.
- Space/tab-separated tables, headerless files, GHz headers, and implicit
  `mat.<flag>` tables are not supported. Frequency units are never guessed.

In GHOST, keep the CSV beside the `.geo` and reference its filename explicitly:

```text
IBCS_Resistances:
1 coating.csv

Dielectrics:
2 substrate.csv
```

The `.geo` format still describes geometry and inline constant/spatial models;
these CSV rules describe frequency-dependent material data. GHOST's tabular
2-D dBke result CSV also writes its frequency column as `frequency_hz`.
