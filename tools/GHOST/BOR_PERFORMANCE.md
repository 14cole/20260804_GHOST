# BOR performance controls

BOR now processes aspect angles in bounded batches, retains one double-precision
LU per signed mode, and can reuse an incrementally discovered incident basis.
The defaults are **dense LU**, **64 aspects per batch**, and **automatic basis
reuse**. They apply to PEC/IBC, electric sheets, dielectric, coated, partial,
and layered/junction formulations. Both polarizations remain co-solved.

In GHOST, select **BoR**, then open **Advanced Settings → BOR execution and
resources**. The same controls are available in GRIM's embedded GHOST tab.
Geometry presets remain specific to 2D.

**Save run setup / Load run setup** supports separate 2D and BOR `.run.json`
recipes in GHOST. BOR recipes preserve frequencies, units, mesh certification,
accuracy, CFIE alpha, and every BOR execution option. GHOST saves aspects from
the body's +z axis. Local/HPC driver configurations accept the recipe as
`run_setup`, mapping body aspects to azimuth 0, elevation `90 - aspect`, a +z
body axis, and zero roll; those drivers export radar coordinates.

Existing recipes containing a radar grid and body attitude remain loadable
in GHOST. They use the derived body aspects and plot in body coordinates;
the load summary describes this conversion. Geometry and output paths are
configured separately. **Check geometry and run setup** also works for BOR
and reports an allocation forecast for the chosen execution options.

## Python and local/HPC drivers

The public `ghost_backend.bor.solver` solve functions and
`ghost_backend.bor.dispatch` solve/resource-preview functions accept `bor_options`:

```python
result = solve_bor(
    points, 1.0e9, aspects_deg,
    formulation="cfie",
    bor_options={
        "angle_batch_size": 64,
        "rhs_compression": "auto",
        "factorization": "dense",
    },
)
```

For `run_local_bor.py` and `run_hpc_bor_monostatic.py`, put the options in the
driver JSON's `settings.BOR_EXECUTION_OPTIONS`. For example:

```json
{
  "schema": "ghost.driver-config",
  "version": 1,
  "driver": "bor",
  "settings": {
    "BOR_EXECUTION_OPTIONS": {
      "angle_batch_size": 64,
      "rhs_compression": "auto",
      "factorization": "compressed",
      "compressed_storage_mib": 2048,
      "compression_tile": 32
    }
  }
}
```

Options are validated, copied into manifests and worker contexts, used by RAM
planning, and recorded with solver results. BOR controls are independent of
2D execution profiles and their environment variables. Thread and streaming
budgets retain the existing BOR driver controls.

| Option | Values and meaning |
| --- | --- |
| `angle_batch_size` | Integer 1–256 aspects; each aspect supplies VV and HH columns. |
| `rhs_compression` | `off`, `auto`, or `on`. Automatic mode skips small systems. Every reconstructed physical excitation is checked; failed reconstructions receive direct solves. |
| `factorization` | `dense` (default) or `compressed` (experimental). |
| `compressed_storage_mib` | Integer ≥16; combined retained numeric operator/inverse allowance across concurrently active modes. Near caches, geometry, Python objects, FFT, construction, and RHS workspaces require additional RAM. |
| `compression_tile` | Integer 8–128; default 32. Controls retained operator tile dimensions. |
| `tile_cache_mib` | Integer 0–4096; default 16. Shared contracted-coefficient cache for one compressed solve; 0 disables it. Additional to the compressed operator/inverse cap. |

Smaller batches reduce angle-workspace RAM and can increase runtime. Basis reuse
is scoped to one mode's factor and bounded to at most 256 columns. It does not
reuse factors across modes, frequencies, materials, or certification meshes.

## Experimental compressed assembly

The compressed route queries bounded nodal tiles using the existing modal
angular rules and near-pair quadrature. It combines those tiles using the same
equation recipes as dense assembly, including material weights, source-side IBC
terms, magnetic-current rotations, and sparse pole/junction relations. Spatial
grouping follows the generating-curve coordinates, including all field families.
Small canonical nodal tiles allow repeated field-family and constraint queries
to reuse their contracted coefficients. A solve-local, thread-safe LRU bounds
the cache by payload bytes and 4,096 entries. Mode workers share this allowance;
the RAM forecast charges it once. Results record cache hits, misses, evictions,
and peak payload in `bor_tile_cache`. Explicit nested requests receive their own
cache, and completed top-level requests release it.
It avoids retaining full far-field nodal blocks and assembling a global dense
modal matrix. Hierarchical factors may retain dense local leaf blocks.

Off-diagonal tiles use checked QR compression at `1e-14`; unprofitable tiles
stay dense locally. The shared compressed inverse refines against the retained
operator and includes measured coefficient-compression error in the original
coefficient backward-error bound. The BOR `1e-12` backward-error and `1e12`
unscaled 1-norm condition-estimate limits remain in force. Normal and adjoint
inverse actions participate in condition estimation. Modal-tail and mesh
certification checks remain enabled. The condition estimator is recorded as
`compressed_original_1norm_onenormest`; dense solves retain LAPACK GECON.

Compression uses double precision; an explicit single-precision table request
is rejected. Selecting compressed factorization replaces table/streaming far
assembly with tile queries. Failure to meet the numerical or storage limits
rejects the solve without a global dense fallback.

This route can trade substantial runtime for reduced matrix storage: angular
FFTs are still repeated for different uncached tiles and modes, whereas the existing streaming
path shares angular work across mode blocks. Near contractions are still retained
across modes. Compression is therefore an explicit experimental selection, and
does not establish that a previously untested large geometry will fit or finish
faster. Dense LU remains appropriate when its memory requirement fits.

## Validation and measurements

`ghost_backend/tests/test_bor_execution.py` checks angle ordering, one-factor
reuse, physical-RHS reconstruction, sparse constraints, compressed coefficient
queries, normal/transpose/adjoint solves, memory caps and cancellation.
The existing BOR physics and material suites remain the physical references.

Recipe and coefficient-cache regression coverage is maintained in
`ghost_backend/tests/test_bor_run_setup.py` and `test_bor_tile_cache.py`.
Results for a particular fixture do not establish general speed or memory guarantees.
