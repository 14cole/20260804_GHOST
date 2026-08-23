# Material validation data

This directory contains source-backed validation packs for FREDDY's material
mixing tools. A pack keeps the unmodified public source tables separate from
the generated five-column FREDDY files and includes a deterministic converter,
checksums, provenance, sign/unit transformations, and solver-level regression
tests.

Current pack:

- `nist_bam_pdms`: broadband barium-hexaferrite/PDMS measurements and fitted
  effective properties from NIST, 100 MHz to 110 GHz.

The University of Leeds PDielec MgO/PTFE dataset
([DOI 10.5518/21](https://doi.org/10.5518/21)) was also evaluated. Its Figure
03 table was not adopted as a passing regression because applying FREDDY's
standard spherical Maxwell-Garnett expression to the supplied Figure 02 MgO
properties and the stated 10 vol% loading does not reproduce the supplied
composite table. Retaining that dataset as a benchmark would therefore test an
unresolved source/model discrepancy rather than FREDDY's implementation.

Validation data establish numerical and format behavior for the stated model.
They do not make an effective-medium law universal: particle shape, dispersion,
agglomeration, anisotropy, percolation, and manufacturing variability remain
physical limitations.
