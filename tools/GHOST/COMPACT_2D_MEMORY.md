# Compact storage for 2D multi-region solves

The reference Dense LU and experimental CPU streaming methods now share a
direct-scatter multi-region assembler. This changes storage and temporary-array
lifetimes, preserving the linear Galerkin equations, materials, mesh density,
polarizations, requested angles, and accuracy gates.

- Regional S/K-prime operators have independent row plans and scatter into
  the final system without separate operator payloads. Equal-wavenumber media share kernel requests
  while retaining the union of all required observation rows.
- Weighted Robin single-layer operators retain their element weights inside
  the observation integral. Unused observation contributions are skipped.
- The mass matrix uses sparse storage. Final-system blocks are assembled in
  64-row portions, avoiding full-interface arithmetic temporaries.
- Operator storage and assembly temporaries are released before factorization.
  Monostatic calls no longer return an unused full exterior-density array;
  bistatic calls project each bounded batch directly; density export skips
  the unused far-field calculation.
- Matrix norms and finite-value checks use bounded workspaces. Mixed LU uses
  an owned Fortran-order single-precision buffer in place, avoids conjugating
  the entire matrix for adjoint checks, and releases failed factors before a
  double-precision fallback.
- Desktop and HPC memory estimates use the same compact operator plan and
  take the maximum of assembly and solve phases. The estimate includes map,
  sparse-mass, block, near-pair, tile, RHS, and check-workspace allowances.

## Azimuth sweeps

The matrix remains independent of incident azimuth at a fixed frequency,
polarization, and mesh. Both CPU methods factor once and construct, solve,
check, and project bounded angle batches, retaining every requested angle.
This also covers sheet, thin-layer, mixed-precision CPU, and bistatic solves.

CPU streaming defaults to 256 columns per batch. To use smaller batches, set
`GHOST_CPU_ANGLE_BATCH_SIZE` in the process environment before launching the
GUI or driver. Accepted values are integers from 1 through 256. For example,
in PowerShell:

```powershell
$env:GHOST_CPU_ANGLE_BATCH_SIZE = '32'
```

This setting applies to both **Dense LU** on CPU and **CPU streaming
(experimental)**. The planner reads the same limit. Result metadata includes
`dense_factorization_count`, `dense_rhs_batch_count`, `dense_rhs_column_count`,
and `dense_max_rhs_columns`. The existing optional GPU path uses full RHS storage.
System matrix storage still grows quadratically with mesh size. Automatic
algebraic RHS compression can further reduce the number of columns actually
solved; all requested physical RHS and fields are retained and checked.
See [the latest measurements and hierarchical option](COMPUTATION_RAM_UPDATES.md).

## Airfoil sizing example

The supplied audit airfoil at 10 GHz, inches, and its existing `N=-20` density
produces 18,726 panels, 18,732 nodes, and 28,406 unknowns. No coarsening was used.
With default assembly settings, double precision, and 0-360 degrees inclusive:

| Run | Previous estimate | Updated estimate |
|---|---:|---:|
| Reference, one angle | 86.79 GiB | 24.12 GiB |
| Reference, 361 angles | 87.40 GiB | 25.41 GiB |
| CPU streaming, 361 angles | 87.75 GiB | 25.51 GiB |
| Reference, 1.5x certification fine mesh, 361 angles | 196.86 GiB | 56.32 GiB |

The fine mesh contains 28,138 panels. It also requires an adequate configured
panel limit; these sizing calculations allowed up to 100,000 panels. The
default low-level limit of 20,000 panels is unchanged. Certification still
requires the fine solve and its comparison to succeed.

These are allocation estimates, not measurements of the full 10 GHz solve.
Scheduler margins and interpreter/runtime overhead can increase the required
machine capacity. Windows committed memory and resident RAM are distinct,
especially for the previous zero-padded operator arrays.

At 0.5 GHz, a measured reference/compact comparison used the same 1,090-panel
airfoil mesh and all 361 angles in independent processes. Additional peak RSS
fell from 355.1 MiB to 175.4 MiB (50.6%), and additional private committed memory
fell from 522.0 MiB to 204.9 MiB (60.7%). Complex amplitudes were identical.
This smaller measurement is not a full-size peak-memory or speed guarantee.

## Validation

`ghost_backend/tests/test_compact_multi_region.py` checks compact/full coefficient equality,
weighted and equal-wavenumber requests, the absence of global dense operator
and mass allocations, operator lifetime at factorization, phase budgeting,
configurable angle batches with one factorization, bounded checks, and mixed
precision adjoint/fallback behavior.

The change was also checked against saved pre-change multi-region matrices,
RHS vectors and fields, the existing 2D physics/accuracy and solver suites,
HPC scheduling and local-driver checks, and the Python 3.6.8 / NumPy 1.14.3 /
SciPy 1.0.0 compatibility runtime. It introduces no FMM backend, basis change,
material approximation, automatic mesh coarsening, or angle interpolation.
