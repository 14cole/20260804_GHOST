# NIST BaM/PDMS validation pack

This pack converts a public NIST broadband material dataset into the exact CSV
schema read by FREDDY and uses it to validate both forward and inverse
Maxwell-Garnett material mixing.

## Source and license

- Dataset: *Broadband Electromagnetic Properties of Engineered Flexible
  Absorber Materials*, version 1.0.0
- Dataset DOI: [10.18434/mds2-2911](https://doi.org/10.18434/mds2-2911)
- Associated article: [10.1002/admt.202300887](https://doi.org/10.1002/admt.202300887)
- License: [NIST Public Data/Open License](https://www.nist.gov/open/license)
- Accessed: 2026-08-14

The files under `source/` are unmodified. Their SHA-256 hashes are recorded in
`manifest.json` and enforced by the converter and test suite.

The article reports that nominal 30 wt% and 60 wt% BaM/PDMS samples correspond
to BaM volume fractions 0.0726 and 0.215, respectively. It reports a fitted BaM
relative permittivity of `16.65 + 0i`. The frequency-dependent BaM permeability
in this pack is analytically recovered from the NIST 30 wt% Maxwell-Garnett fit;
the separate 60 wt% fit then provides a cross-loading validation.

## FREDDY conversion

Every generated file under `freddy/` has exactly:

```text
frequency_hz,eps_real,eps_imag,mu_real,mu_imag
```

Frequency remains in Hz. NIST's Figure 7b and Figure 8b columns are positive
loss magnitudes. FREDDY uses `exp(+j*omega*t)`, so the converter writes:

```text
eps_imag = -source_permittivity_loss
mu_imag  = -source_permeability_loss
```

The fit tables are passive over the full common 100 MHz--110 GHz grid. Some
raw measured magnetic-loss values cross below zero because of measurement
noise. Those points would become active gain after the required sign
conversion. The `*_passive.csv` files therefore omit and count those rows;
the converter never clamps a loss to zero or changes a measured sign.

Generated material files:

- `pdms_fit.csv`: fitted host permittivity with relative permeability 1.
- `bam_particle_fit.csv`: reported particle permittivity and permeability
  recovered from the 30 wt% fit.
- `bam_pdms_30wt_fit.csv`: NIST fit at BaM volume fraction 0.0726.
- `bam_pdms_60wt_fit.csv`: NIST fit at BaM volume fraction 0.215.
- `bam_pdms_30wt_sample_[1-3]_passive.csv`: passive measured points from each
  nominal 30 wt% sample.
- `bam_pdms_60wt_measured_passive.csv`: passive measured points from the
  nominal 60 wt% sample.

`manifest.json` records exact row and omission counts as well as every
transformation.

## Reproduce and validate

Run from the repository root:

```text
py -3 tools/FREDDY/tools/convert_nist_bam_pdms.py
py -3 tools/FREDDY/tools/validate_material_mix.py
```

On macOS or Linux, replace `py -3` with `python3`.

The validator loads the generated files through FREDDY's production material
parser and evaluates them with FREDDY's production Maxwell-Garnett function.
It checks:

- forward prediction of both fitted composite curves;
- inverse recovery of both published volume fractions;
- agreement with four measured datasets using median and 95th-percentile
  complex epsilon/mu relative error.

The checked-in regression limits are deliberately looser than the current
results: fitted-curve RMS error must be below 0.1%, recovered volume fraction
must be within 0.0002 absolute, measured median error below 2.5%, and measured
95th-percentile error below 5%. The fit mismatch is nonzero because the
published decimal tables are rounded.
