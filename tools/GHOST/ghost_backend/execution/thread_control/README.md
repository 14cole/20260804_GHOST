# Bundled BLAS thread control

GHOST imports `threadpool_limits` and `threadpool_info` from this package.
No separate threadpoolctl installation is required. An installed copy is not
used by GHOST, so source copies and HPC workers use the reviewed version.

| Python | Implementation | License |
| --- | --- | --- |
| 3.9 and newer | `_threadpoolctl.py`, threadpoolctl 3.6.0 | `LICENSE-3.6.0.txt` |
| 3.6 through 3.8 | `_threadpoolctl_py36.py`, threadpoolctl 2.2.0 | `LICENSE-2.2.0.txt` |

The implementation files and BSD 3-Clause licenses are copied without content
changes from the official distributions. Their bytes were checked against the
installed distributions' RECORD hashes. The source filenames are changed to
keep imports private to GHOST.

Upstream: https://github.com/joblib/threadpoolctl

| File | SHA-256 |
| --- | --- |
| `_threadpoolctl.py` | `12fb9526b6a74d2e686b7ec148dc165c3587999b14fa86984aa180e10802400b` |
| `_threadpoolctl_py36.py` | `a3bd8629e7300409b201ccd42d5ec4812ce1965f04b135169ac68e2de3260ebf` |
| Each license | `81ac619075248b06e53660b652d10e485f4675f5d0ae0f97ea22370da1f7e23b` |

Copy this entire directory with the backend. NumPy and SciPy remain external
dependencies. Thread limits apply to their supported native BLAS libraries;
they do not replace those libraries or the solver's assembly-thread controls.
