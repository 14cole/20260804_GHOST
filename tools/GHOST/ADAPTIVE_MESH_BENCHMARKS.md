# Adaptive polynomial solver measurements

These comparisons use the integrated solver sources with either the linear
reference strategy or Automatic polynomial adaptation. All reported solves
include mesh certification, condition estimation, both polarizations and 19
angles from 0 to 360 degrees. Each timing is a fresh worker process with two
assembly threads, two BLAS threads, a 24 GiB RAM budget and a 256 MiB compressed
payload budget. No other solver tests ran during the timed comparisons.

The 1 GHz airfoil row uses the median of two runs per strategy; the other rows
are single pairs. RAM is the maximum sampled process RSS, including the Python
runtime, sampled every 50 ms. Small variations are not meaningful speedups.
Automatic selected dense execution for these measured cases.

| Case | Reference -> adaptive seconds | Time saved | Reference -> adaptive GiB | RAM saved | Unknowns per polarization | Maximum normalized complex-field difference |
|---|---:|---:|---:|---:|---:|---:|
| reentrant-large | 5.73 -> 5.34 | 6.8% | 0.238 -> 0.132 | 44.2% | 1,726 -> 876 | 0.0698% |
| gap-large | 7.99 -> 7.01 | 12.2% | 0.237 -> 0.170 | 28.3% | 2,056 -> 1,044 | 0.0631% |
| dielectric-large | 21.71 -> 9.54 | 56.1% | 0.501 -> 0.187 | 62.6% | 3,548 -> 1,788 | 0.0127% |
| mixed-large | 20.24 -> 10.05 | 50.3% | 0.732 -> 0.266 | 63.6% | 4,438 -> 2,238 | 0.0382% |
| airfoil 1 GHz | 26.59 -> 19.42 | 27.0% | 0.783 -> 0.381 | 51.3% | 4,622 -> 2,944 | 0.0061% |
| airfoil 5 GHz | 754.51 -> 148.54 | 80.3% | 14.211 -> 4.011 | 71.8% | 21,618 -> 11,284 | 0.0044% |

The four larger polygon/material fixtures use a 5x geometric scaling of the
repository's reentrant, narrow-gap, dielectric and mixed fixtures at 3 GHz and
40 reference panels per wavelength. All original vertices, gaps and material
definitions are preserved. Normalized field differences are relative to the
peak reference amplitude in each polarization; they are not absolute physical
error estimates. Both the reference and accepted adaptive solutions pass their
respective convergence and original-equation quality checks.

Seven smaller fixtures and the larger rectangle, acute-corner and sheet cases
retain the reference basis under Automatic. Their results agree to floating
point/solver tolerance and timings remain approximately unchanged. Calibration
showed that applying polynomial quadrature below the selected size threshold
could increase runtime, which is why Automatic retains the reference route.

The supplied airfoil is interpreted in inches. Its geometry SHA-256 is
`b167517986a9d549b024b16f37c692e0ece3382bdcaaf35fe1a036a3c5a82d70`.
The tested runtime is Python 3.12.14, NumPy 2.5.2,
SciPy 1.18.1 on Windows. The benchmark solver source/runtime fingerprint is
`4020674053815b8eb592fb88fc3fdfb1309a780f80e440a999db06d467aea21d`.

The final batch-planning correction evaluates the size crossover per frequency;
optional timing-cache writes also gained a bounded permission-failure path.
These changes followed the timings and do not change the single-frequency
discretizations or numerical kernels measured here. Qualification also
covers sweeps that use different polynomial degrees at different frequencies.
An additional certified 1 GHz solve after the final corrections reproduced
the measured adaptive fields exactly. Its timing is excluded because other
verification work was running. That validation run's source/runtime fingerprint was
`66ac00fb2ed6b67bcb8e1d33c03e6b8228514f6f49c51de205c59456f5200c1d`.

Reproduce the polygon comparisons from the integrated project root:

```bash
python tools/GHOST/scripts/benchmark_adaptive_geometries.py --output adaptive-polygon-benchmarks
```

## Larger airfoil forecasts

These are planning results for the same 19-angle certified request under a
24 GiB budget. They are not completed 10 or 20 GHz solves. The adaptive counts
describe the initial cubic candidate; further refinement can increase them.
Dense memory is a forecast including factorization/workspace allowances,
not the actual memory consumption of compressed or FMM execution.

| GHz | Reference -> cubic candidate unknowns | Reference -> candidate dense peak GiB | Automatic backend: reference -> candidate |
|---|---:|---:|---|
| 10 | 42,686 -> 21,994 | 54.60 -> 14.65 | compressed -> dense |
| 20 | 85,098 -> 42,850 | 216.26 -> 55.02 | compressed -> compressed |

The full implementation, convergence limits, automatic exceptions and
reproduction commands are in [the adaptive mesh guide](ADAPTIVE_MESH.md).
HPC workers use the same planning and reservation checks. These measurements
were made on the workstation; no remote cluster or distributed-memory solve
is claimed.

## Integrated verification

The actual integrated project passed the GRIM suite (1,036 cases), accelerated
GHOST qualification (203 cases plus 90 subtests), additional GHOST regressions
(94 cases plus 87 subtests), and FREDDY suite (197 cases). GRIM and FREDDY each
have one expected skip. Startup diagnostics, source inventories, local-driver
execution, HPC scheduling and ASCII-transfer checks also pass. The final
permission-cache and environment-profile fixes passed a further focused set of
23 cases plus five subtests. The polynomial reproduction command was exercised
successfully, and the final integrated airfoil validation reproduced the
measured 1 GHz adaptive fields exactly.
