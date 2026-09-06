"""Shared physical constants and default policies for the 2-D solver."""



C0 = 299_792_458.0
ETA0 = 376.730313668
EPS = 1e-12
MATERIAL_SINGULAR_TOL = EPS
RCS_DB_FLOOR_LINEAR = EPS
VIRTUAL_SHEET_REGION_START = 900_000
EULER_GAMMA = 0.5772156649015329
CFIE_ALPHA_DEFAULT = 0.0
MAX_PANELS_DEFAULT = 20_000
DEFAULT_PANELS_PER_WAVELENGTH = 20
# Explicit positive N remains a user-controlled mesh, but it may not request
# a discretization so coarse that the boundary phase is plainly unresolved.
# This is only a gross safety floor; production accuracy still requires the
# separate base/fine complex-field mesh-convergence certification.
MIN_EXPLICIT_PANELS_PER_WAVELENGTH = 4
# Monostatic 2D RCS normalization controls.
#
# For the physical asymptotic convention,
#   G(r) = (j/4) H_0^(2)(k r),
# and for a far-field amplitude A defined such that
#   u_s(r,phi) ~ sqrt(1 / (8*pi*k*r)) * exp(-j(kr-pi/4)) * A(phi),
# the 2D scattering width per unit length is
#   sigma_2d(phi) = |A(phi)|^2 / (4 k).
#
# Historical solver/projector outputs store the bare layer-potential integral
# B, for which A = +j*B.  That global unit-modulus factor leaves scattering
# width unchanged, but it matters to consumers of complex phase.  Keep B for
# feature-delta compatibility and label the convention explicitly everywhere.
#
# Use physical 2D scattering-width normalization by default.
#
RCS_NORM_NUMERATOR = 0.25
RCS_NORM_MODE_DEFAULT = "physical"
RCS_NORM_MODE_PHYSICAL = "physical"
RCS_AMPLITUDE_CONVENTION = "A_physical_asymptotic = +j * B_stored"
RCS_AMPLITUDE_VERSION = 2  # Absolute dielectric / TE-sheet / multi-region phase.
DENSE_LINEAR_BACKWARD_ERROR_MAX = 1.0e-12
DENSE_GPU_BACKEND_ENV = "GHOST_DENSE_BACKEND"
DENSE_GPU_MIN_N_ENV = "GHOST_DENSE_GPU_MIN_N"
DENSE_GPU_MIN_N_DEFAULT = 768
DENSE_GPU_PROBE_TIMEOUT_S = 20.0

