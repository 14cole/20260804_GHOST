FMM2D numerical sources are vendored, unmodified, from Flatiron Institute's
repository at commit `550dae5b77b1e006c8ffae37fc832f8c2b536871`.
Source: https://github.com/flatironinstitute/fmm2d

The upstream root LICENSE is Apache-2.0. Some individual source files retain
older GPL-2.0-or-later headers. Those headers and the root license are preserved.
Resolve that upstream inconsistency before distributing a proprietary build;
this local research integration does not assert a resolved relicensing status.

The supplied Windows x86-64 DLL was built with GNU Fortran/GCC 16.2.0 (MSYS2
UCRT64 Rev3), portable x86-64 instructions, and static GNU/OpenMP runtimes.
GNU compiler runtime terms include the GCC Runtime Library Exception 3.1:
https://www.gnu.org/licenses/gcc-exception-3.1.html
The compiler and source hashes, flags, and DLL digest are in build-info.json.
The adjacent build.py rebuilds the unmodified upstream sources and GHOST's
persistent-plan adapter, plan.f90, with a complete GNU Fortran installation.
It requires no downloads and writes local build provenance. The adapter calls
the upstream tree and evaluation routines. Windows and Linux builds are the
deployment targets of this integrated project.
Only the Windows x86-64 binary has been exercised in this workspace.
