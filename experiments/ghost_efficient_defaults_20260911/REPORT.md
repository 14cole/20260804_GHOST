# Efficient defaults for large 2D sweeps

New GHOST/GRIM Runs monostatic jobs and 2D local/HPC drivers use:

| Setting | Default |
| --- | --- |
| Kernel method | CPU streaming |
| Factorization | Compressed assembly |
| Precision | Double |
| Compressed numeric storage allowance | 8192 MiB |
| Assembly threads | 4, reduced on smaller hosts and capped by worker allocation |
| BLAS threads | 2, reduced on smaller hosts |
| Illumination basis reuse | Automatic |
| Angles per batch | 256 |
| RAM admission budget | Available memory |
| Temporary disk directory | Execution host's system temporary directory |
| Accuracy / mesh certification | Standard / enabled |

The defaults target large RAM-constrained sweeps such as the supplied airfoil.
They are not a universal minimum of both time and RAM. The earlier 2 GHz
airfoil certificate took 130.86 seconds / 2603 MiB with dense LU and
178.88 seconds / 577 MiB with compression: 77.8% less sampled RAM, 36.7% more
time. The earlier 10 GHz compressed certificate completed in 3214.40 seconds
at 3380 MiB sampled peak RSS. The numeric allowance is not preallocated RAM.
See the [qualification report](../ghost_certification_20260910/REPORT.md).

The batch size remains 256 because the earlier 256/128/64-angle comparison
took 41.95/42.42/43.11 seconds at 1244/1195/1181 MiB. Smaller batches increased
the solved basis-column count. See the [final investigation](../ghost_final_20260910/REPORT.md).
No new timing claim or full 10 GHz run is made for this defaults change.

## Implementation and compatibility

`execution.options.efficient_defaults()` provides the complete new-run preset.
GUI startup and compatible batch driver defaults use it. The widget's
**Use efficient defaults** button applies it to an existing setup.
Saved profiles and versioned schema defaults retain their explicit meanings.
Python callers select the preset with `solver_method='experimental_cpu'` and
`execution_options=efficient_defaults()`.

Runs preferences restore method, precision, and resources together. A saved
dense/mixed setup can still be loaded after a compressed setup. A driver config
that explicitly chooses mixed precision without a method selects the compatible
reference method. Bistatic and BoR retain their supported solver choices.

## Verification

- Preset, GUI save/load/reset, scoped resources, and driver config: 53 tests passed.
- HPC/headless, environment handoff, memory safety, and GUI entry point: 95 tests passed.
- Additional default PEC/mixed certificates and saved/bistatic setup regressions: 19 tests passed.
- GRIM workflow, GUI integration, and release-readiness checks: 117 tests passed.
- Python 3.6 / NumPy 1.14 / SciPy 1.0: 29 tests passed, including certified preset comparisons.
- Local driver integration: 30 checks passed.
- HPC scheduling, restart, outputs, and attestations: zero failures.
- Source inventories and ASCII transfer checks passed.

The two certified preset cases match dense complex fields at every one of 361
azimuths to better than 1e-10 of the reference channel peak. Logs are saved
beside this report.
