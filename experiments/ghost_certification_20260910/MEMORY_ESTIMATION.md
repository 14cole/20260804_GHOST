# Compressed memory-estimation update

The airfoil's 10 GHz, 361-angle scheduling reservation falls from **16.01 GiB
to 7.11 GiB**, approximately **56% lower**. A single angle now reserves
**5.69 GiB** instead of 16.01 GiB. These are corrected forecasts, not additional
reductions in the solver's actual memory allocation.

| Airfoil, 10 GHz, both channels, base/fine certificate | Previous | Revised |
|---|---:|---:|
| Solver admission estimate, 361 angles | 11.41 GiB | 6.94 GiB |
| Scheduler reservation, 361 angles | 16.01 GiB | 7.11 GiB |
| Scheduler reservation, one angle | 16.01 GiB | 5.69 GiB |
| Enforced numeric payload allowance | 8 GiB | 8 GiB |

The prior 10 GHz run's sampled process peak was 3.30 GiB. The revised reservation
remains above that measurement because inverse ranks and repair workspaces are
not known before factorization. The model exposes those allowances explicitly.

## What changed

- The storage limit no longer substitutes for expected resident payload.
  Large systems sample production coefficient tiles across spatial-separation
  bands, using the same coefficient checks and QR storage decision as assembly.
  Planning visits at most four tiles per band, with no global matrix or
  quadratic list of candidate tiles. Sampling variability is reported and an
  allowance is added. It is an estimate, not a rigorous error interval.
- Systems with at most 8192 unknowns use a cheap dense-storage ceiling. They
  do not assemble coefficients just to estimate memory. This avoids substantial
  planning overhead on cases that already complete quickly.
- Inverse storage follows the actual balanced tree, rank limit of 256, and
  maximum dense leaf size of 512. Construction accounts for sequential ACA,
  child construction and U-to-E replacement. It no longer reserves arbitrary
  unused space up to the user's payload limit.
- Assembly, factorization and angle solving are separate phases. The forecast
  takes their maximum. Angle work uses the actual batch width, capped at 256;
  one angle does not reserve 256 illumination columns. The phase model counts
  geometry, Python tile/spool bookkeeping, bounded kernel work, and explicit
  kernel-tile overrides.
- Partner numeric data stored on disk is reported separately from resident
  RAM. Its Python bookkeeping still counts toward RAM. The reported disk value
  is an allowance for a possible paired partner; its exact payload depends on
  that partner's material equations and whether pairing is applicable.
- Compressed scheduler safety applies to uncertain sampled operator storage,
  rather than multiplying structural inverse ceilings and all workspaces again.
  The scheduler floor is a minimum total process reservation, not an additional
  copy of interpreter overhead. Both base and fine estimates are checked, and
  the larger reservation wins. Dense scheduling retains its prior behavior.
- Per-solve caches hold only small scalar forecasts, with hashed material/mesh
  keys. Sampled coefficient arrays and oracles are released. The existing
  payload checks, precision restrictions and numerical gates remain enforced.

For the revised 361-angle reservation, approximately 2.08 GiB covers the
sampled operator with its allowance, 2.73 GiB covers the structural inverse
ceiling, 1.99 GiB covers solve/repair workspace, and 0.31 GiB covers process,
geometry and tile bookkeeping. The old whole-budget-plus-35%-plus-0.6-GiB
calculation has been removed from compressed planning.

## Measurements and validation

Final forecasts were compared against saved process-RSS measurements for all
14 material fixtures, the 2 GHz airfoil, and the 10 GHz airfoil. All 16 scheduling
reservations cover those measured peaks. Both polarized operator allowances
cover the corresponding retained-operator measurements. Small material plans
complete in approximately 0.01–0.07 seconds and retain the scheduler's 0.6 GiB
minimum. The 2 GHz reservation is approximately 1.24 GiB versus 0.56 GiB sampled
in the earlier run.

The complete 10 GHz base/fine, VV/HH plan took approximately **29 seconds**,
including geometry work. It sampled 28 tiles per fine operator out of 16,384.
The initial six-sample-per-band experiment was reduced to four after measuring
planning overhead; the final allowances still cover the saved measurements.

A fresh 2 GHz, 361-angle, both-polarization certificate passed in **182.76 s**,
including **3.68 s** in memory-estimation calls, with **0.532 GiB** sampled peak
RSS. Its complex fields differ from the independent dense reference by at most
4.47e-13 of the reference channel peak. The earlier compressed run took 178.88 s;
these individual timings do not establish a speedup. The last additions after
this fresh run only account for explicit tile overrides and tile-object metadata;
they do not change successful coefficient sampling or solver arithmetic.

Validation also includes:

- 169 regression tests passed during implementation.
- 42 final desktop tests passed, including 11 estimation tests, GUI integration,
  configured local/HPC workers, export and resume.
- 29 final tests passed on each Python 3.6 runtime: NumPy 1.19.5/SciPy 1.5.4 and
  NumPy 1.14.3/SciPy 1.0.0.
- The standalone scheduler suite completed with zero failures, including
  resource planning, CPU/memory dispatch, restart, work distribution and output
  attestation.
- Changed Backend files parse with Python 3.6 grammar; `git diff --check` passes.

The full 10 GHz solve was not repeated for this estimation-only change. Its
previous numerical qualification remains scoped to the saved source revision.
Compressed storage can vary on unsampled tiles, so the new forecast is not a
hard process-RAM limit or a promise that every geometry will match its estimate.

## Evidence and reproduction

`check_memory_forecasts.py` recreates the material/airfoil comparisons. Its
output is `memory-forecast-validation.json`; it includes final Backend hashes.
`memory-airfoil-reservations.json` records the one-angle and 361-angle forecasts.
`memory-airfoil2-comparison.json` records the fresh certified solve comparison.
Raw new solve output is `airfoil-f2.0-compressed-certified-a361-r3.json`.

Tests are in `tools/GHOST/tests/test_compressed_memory.py`. Logs in this folder:
`memory-regression-tests.log`, `memory-final-integration-tests.log`,
`memory-python36-tests.log`, `memory-oldest-tests.log`,
`memory-scheduler-tests.log`, `memory-forecast-validation.log`, and
`memory-airfoil2-cert.log`. `memory-final-source-hashes.json` identifies changed
Backend files relative to the main numerical qualification.

Activation settings remain in [COMPRESSED_CPU.md](../../tools/GHOST/COMPRESSED_CPU.md).
