"""
Closed-form 2D scattering-width series for circular cylinders.

All formulas use the e^{+jwt} (engineering) time convention to match the
solver.  Outgoing radial waves are H_n^(2)(kR); passive lossy media have
complex permittivity with non-positive imaginary part (eps = eps' - j eps'',
eps'' >= 0).
Results are returned as sigma_2D (meters), the 2D scattering width per unit
length, which is what the solver returns as "rcs_linear".

Conventions:
- 2D geometries are treated as ELEVATION cuts -- the cylinder axis (z) is
  HORIZONTAL (out-of-plane = cross-range), so the radar aliases flip
  relative to the textbook "z is vertical" convention.
- TM = E_z polarization (electric field along cylinder axis) = Dirichlet-like
  on PEC.  E is horizontal -> matches the solver's "TM" / "HH" label.
- TE = H_z polarization (magnetic field along cylinder axis) = Neumann-like
  on PEC.  E is in-plane with vertical component -> matches "TE" / "VV".

References:
- Balanis, "Advanced Engineering Electromagnetics," 2nd ed., Ch. 11.
- Harrington, "Time-Harmonic Electromagnetic Fields," Ch. 5.
- Bohren & Huffman, "Absorption and Scattering of Light by Small Particles,"
  Ch. 8 (infinite-cylinder normal incidence).

All series return monostatic backscatter sigma_2D in meters.
"""

import numpy as np
from scipy import special as sp


C0 = 299_792_458.0
ETA0 = 376.730313668


def _hankel2(n, z):
    """Hankel function of the second kind, order n, argument z."""
    return sp.hankel2(n, z)


def _jn(n, z):
    return sp.jn(n, z)


def _jn_prime(n, z):
    """Derivative of Bessel J_n w.r.t. argument.  Array-safe."""
    return sp.jvp(n, z, 1)


def _hankel2_prime(n, z):
    """Derivative of Hankel H_n^{(2)} w.r.t. argument.  Array-safe."""
    return sp.h2vp(n, z, 1)


def _nmax_for_ka(ka, pad=10):
    """Safe series truncation. Wiscombe's rule + pad."""
    ka_abs = abs(complex(ka))
    n = int(np.ceil(ka_abs + 4.05 * ka_abs**(1.0 / 3.0) + 2.0))
    return max(10, n + pad)


def sigma_pec_cylinder(radius_m, freq_hz, polarization):
    """
    Monostatic 2D backscatter sigma_2D for a PEC circular cylinder.

    Parameters
    ----------
    radius_m : float
        Cylinder radius in meters.
    freq_hz : float
        Frequency in Hz.
    polarization : str
        'TM' (E_z axial, Dirichlet) or 'TE' (H_z axial, Neumann).

    Returns
    -------
    sigma_2D in meters.
    """
    pol = polarization.upper()
    k = 2.0 * np.pi * freq_hz / C0
    ka = k * radius_m
    N = _nmax_for_ka(ka)

    if pol == 'TM':
        # E_z polarization.  BC: E_z = 0 on PEC, i.e., total field vanishes.
        # Scattered-field expansion coefficient:
        #   a_n = -J_n(ka) / H_n^{(2)}(ka)
        # Monostatic scattered field amplitude at phi = pi (backscatter):
        #   f(phi=pi) = sum_{n=-inf}^{inf} a_n e^{jn pi}
        n_arr = np.arange(-N, N + 1)
        a_n = -_jn(n_arr, ka) / _hankel2(n_arr, ka)
        # Backscatter factor: e^{jn pi} = (-1)^n
        amp = np.sum(a_n * (-1.0)**n_arr)
    elif pol == 'TE':
        # H_z polarization.  BC: dH_z/drho = 0 on PEC.
        #   a_n = -J_n'(ka) / H_n^{(2)}'(ka)
        n_arr = np.arange(-N, N + 1)
        a_n = -_jn_prime(n_arr, ka) / _hankel2_prime(n_arr, ka)
        amp = np.sum(a_n * (-1.0)**n_arr)
    else:
        raise ValueError(f"Unknown polarization {polarization}")

    # 2D scattering width (RCS per unit length):
    #   sigma_2D = (4 / k) * |sum a_n e^{j n phi}|^2
    sigma = (4.0 / k) * abs(amp)**2
    return float(sigma)


def sigma_impedance_cylinder(radius_m, zs_ohm, freq_hz, polarization):
    """Monostatic 2-D width of a Leontovich impedance cylinder in air.

    ``zs_ohm`` uses the solver convention ``Zs = R + jX`` with
    ``exp(+j*omega*t)``.  With the radial normal directed from the cylinder
    into air, the scalar Robin coefficient is

      TM: beta = -j*k*eta0/Zs
      TE: beta = -j*k*Zs/eta0

    and each cylindrical harmonic satisfies
    ``d(u)/dr + beta*u = 0`` at ``r = radius_m``.  The PEC limits reduce to
    the Dirichlet TM and Neumann TE coefficients used by
    :func:`sigma_pec_cylinder`.
    """

    pol = polarization.upper()
    z_s = complex(zs_ohm)
    if abs(z_s) == 0.0:
        return sigma_pec_cylinder(radius_m, freq_hz, pol)
    k = 2.0 * np.pi * freq_hz / C0
    ka = k * radius_m
    N = _nmax_for_ka(ka)
    n_arr = np.arange(-N, N + 1)
    if pol == 'TM':
        beta = -1j * k * ETA0 / z_s
    elif pol == 'TE':
        beta = -1j * k * z_s / ETA0
    else:
        raise ValueError(f"Unknown polarization {polarization}")

    Jn = _jn(n_arr, ka)
    Jnp = np.asarray([_jn_prime(int(n), ka) for n in n_arr])
    Hn = _hankel2(n_arr, ka)
    Hnp = np.asarray([_hankel2_prime(int(n), ka) for n in n_arr])
    a_n = -(k * Jnp + beta * Jn) / (k * Hnp + beta * Hn)
    amp = np.sum(a_n * (-1.0)**n_arr)
    return float((4.0 / k) * abs(amp)**2)


def sigma_dielectric_cylinder(radius_m, eps_r, mu_r, freq_hz, polarization):
    """
    Monostatic 2D backscatter sigma_2D for a homogeneous dielectric cylinder
    in free space.

    Uses the standard infinite-cylinder series (Bohren & Huffman Ch. 8,
    normal incidence) adapted to 2D scattering width and e^{+jwt} convention.

    Parameters
    ----------
    radius_m : float
        Cylinder radius in meters.
    eps_r, mu_r : complex
        Relative permittivity / permeability.  For lossy media under the
        e^{+jwt} convention used here, eps_r has NEGATIVE imaginary part
        (eps_r = eps' - j*eps'', with eps'' >= 0 for passive attenuation).
    freq_hz : float
        Frequency in Hz.
    polarization : str
        'TM' (E_z axial) or 'TE' (H_z axial).
    """
    pol = polarization.upper()
    k0 = 2.0 * np.pi * freq_hz / C0
    n_rel = np.sqrt(complex(eps_r) * complex(mu_r))
    k1 = k0 * n_rel  # wavenumber inside dielectric
    x = k0 * radius_m
    mx = k1 * radius_m
    N = _nmax_for_ka(max(abs(x), abs(mx)))

    # Match traces of the scalar field u (E_z for TM, H_z for TE) and
    # a polarization-dependent normal-derivative operator across rho = a.
    #
    # TM (E_z): continuity of E_z and (1/mu) dE_z/drho
    # TE (H_z): continuity of H_z and (1/eps) dH_z/drho
    #
    # Inside:     u_n^in  = c_n J_n(k1 rho)
    # Outside:    u_n^out = J_n(k0 rho) + a_n H_n^(2)(k0 rho)
    #
    # Matching at rho = a gives a 2x2 system for (a_n, c_n).
    #
    # Writing the "transverse impedance" factor
    #   TM: zeta_i = mu_i, so RHS uses (k_i / mu_i)
    #   TE: zeta_i = eps_i, so RHS uses (k_i / eps_i)

    if pol == 'TM':
        zeta_out = 1.0  # mu0 = 1 (relative)
        zeta_in = complex(mu_r)
    elif pol == 'TE':
        zeta_out = 1.0  # eps0 = 1 (relative)
        zeta_in = complex(eps_r)
    else:
        raise ValueError(f"Unknown polarization {polarization}")

    n_arr = np.arange(-N, N + 1)
    # Scattered-field coefficients a_n from the boundary match.
    # System:
    #   J_n(x) + a_n H_n(x)           = c_n J_n(mx)
    #   (k0/zeta_out) [J_n'(x) + a_n H_n'(x)] = (k1/zeta_in) c_n J_n'(mx)
    # Solve for a_n:
    #   a_n = -[ (k1/zeta_in) J_n'(mx) J_n(x) - (k0/zeta_out) J_n(mx) J_n'(x) ]
    #         / [ (k1/zeta_in) J_n'(mx) H_n(x) - (k0/zeta_out) J_n(mx) H_n'(x) ]
    Jn_x  = _jn(n_arr, x)
    Jnp_x = np.array([_jn_prime(int(n), x) for n in n_arr])
    Hn_x  = _hankel2(n_arr, x)
    Hnp_x = np.array([_hankel2_prime(int(n), x) for n in n_arr])
    Jn_mx = _jn(n_arr, mx)
    Jnp_mx = np.array([_jn_prime(int(n), mx) for n in n_arr])

    p = k1 / zeta_in
    q = k0 / zeta_out

    num = -(p * Jnp_mx * Jn_x - q * Jn_mx * Jnp_x)
    den = (p * Jnp_mx * Hn_x - q * Jn_mx * Hnp_x)
    a_n = num / den

    amp = np.sum(a_n * (-1.0)**n_arr)
    sigma = (4.0 / k0) * abs(amp)**2
    return float(sigma)


def sigma_coated_pec_cylinder(a_inner_m, a_outer_m, eps_r, mu_r,
                              freq_hz, polarization):
    """
    Monostatic 2D backscatter sigma_2D for a PEC cylinder of radius
    a_inner coated with a homogeneous dielectric out to a_outer.

    Three-medium problem:
      region 0 (rho > a_outer): air (eps=mu=1)
      region 1 (a_inner < rho < a_outer): dielectric (eps_r, mu_r)
      region 2 (rho < a_inner): PEC

    Field expansions per mode n:
      Outside:    J_n(k0 rho) + a_n H_n^(2)(k0 rho)
      Coating:    b_n J_n(k1 rho) + d_n Y_n(k1 rho)
                  [or equivalently H_n^(1) and H_n^(2); we use Jn, Yn]
      Inside PEC: vanishes (boundary condition)

    Boundary conditions:
      at rho = a_inner:
         TM: coating field = 0        [E_z = 0 on PEC]
         TE: d(coating)/drho = 0      [dH_z/drho = 0 on PEC]
      at rho = a_outer:
         coating trace = outside trace
         (k/zeta)*coating_normal_deriv = (k/zeta)*outside_normal_deriv
    """
    pol = polarization.upper()
    k0 = 2.0 * np.pi * freq_hz / C0
    n_rel = np.sqrt(complex(eps_r) * complex(mu_r))
    k1 = k0 * n_rel

    x_in = k1 * a_inner_m       # argument at PEC surface (inside coating)
    x_out_in = k1 * a_outer_m   # argument at outer surface (inside coating)
    x_out = k0 * a_outer_m      # argument at outer surface (outside / air)

    N = _nmax_for_ka(max(abs(x_in), abs(x_out_in), abs(x_out)))

    if pol == 'TM':
        zeta_out = 1.0; zeta_in = complex(mu_r)
    elif pol == 'TE':
        zeta_out = 1.0; zeta_in = complex(eps_r)
    else:
        raise ValueError(polarization)

    p = k1 / zeta_in
    q = k0 / zeta_out

    n_arr = np.arange(-N, N + 1)
    a_n = np.zeros(n_arr.shape, dtype=complex)

    for idx, n in enumerate(n_arr):
        n = int(n)
        # Coating basis: J_n(k1 rho) and Y_n(k1 rho).
        # Use sp.jv / sp.yv which accept complex arguments.
        Jn_in   = sp.jv(n, x_in)
        Yn_in   = sp.yv(n, x_in)
        Jnp_in  = sp.jvp(n, x_in, 1)
        Ynp_in  = sp.yvp(n, x_in, 1)
        Jn_oi   = sp.jv(n, x_out_in)
        Yn_oi   = sp.yv(n, x_out_in)
        Jnp_oi  = sp.jvp(n, x_out_in, 1)
        Ynp_oi  = sp.yvp(n, x_out_in, 1)
        Jn_o    = sp.jv(n, x_out)
        Jnp_o   = sp.jvp(n, x_out, 1)
        Hn_o    = sp.hankel2(n, x_out)
        Hnp_o   = sp.h2vp(n, x_out, 1)

        # PEC BC at rho = a_inner.
        # TM:  b Jn_in + d Yn_in = 0  =>  d = -b Jn_in/Yn_in
        # TE:  b Jnp_in + d Ynp_in = 0 => d = -b Jnp_in/Ynp_in
        if pol == 'TM':
            alpha = -Jn_in / Yn_in
        else:
            alpha = -Jnp_in / Ynp_in

        # Combined coating basis: phi_n(rho) = J_n(k1 rho) + alpha Y_n(k1 rho).
        # Its value and radial derivative at rho = a_outer:
        phi_o   = Jn_oi  + alpha * Yn_oi
        phip_o  = Jnp_oi + alpha * Ynp_oi

        # Outer matching: two equations in (a_n, b_n):
        #   Jn_o + a_n Hn_o = b_n phi_o
        #   q*(Jnp_o + a_n Hnp_o) = p * b_n * phip_o
        # Solve for a_n:
        num = -(p * phip_o * Jn_o - q * phi_o * Jnp_o)
        den = (p * phip_o * Hn_o - q * phi_o * Hnp_o)
        a_n[idx] = num / den

    amp = np.sum(a_n * (-1.0)**n_arr)
    sigma = (4.0 / k0) * abs(amp)**2
    return float(sigma)


def sigma_two_layer_dielectric_cylinder(
    a_core_m,
    a_outer_m,
    eps_shell,
    mu_shell,
    eps_core,
    mu_core,
    freq_hz,
    polarization,
):
    """Monostatic width of a concentric two-dielectric cylinder in air.

    Region 0 is air, region 1 is the annular shell, and region 2 is the
    homogeneous core.  Each cylindrical harmonic is obtained from an
    independent four-unknown boundary match, making this a useful reference
    for TYPE 3 + TYPE 5 multi-region geometries.
    """

    if not (0.0 < float(a_core_m) < float(a_outer_m)):
        raise ValueError("Require 0 < a_core_m < a_outer_m.")

    pol = polarization.upper()
    k0 = 2.0 * np.pi * freq_hz / C0
    n_shell = np.sqrt(complex(eps_shell) * complex(mu_shell))
    n_core = np.sqrt(complex(eps_core) * complex(mu_core))
    k_shell = k0 * n_shell
    k_core = k0 * n_core

    if pol == "TM":
        zeta0 = 1.0 + 0.0j
        zeta_shell = complex(mu_shell)
        zeta_core = complex(mu_core)
    elif pol == "TE":
        zeta0 = 1.0 + 0.0j
        zeta_shell = complex(eps_shell)
        zeta_core = complex(eps_core)
    else:
        raise ValueError(f"Unknown polarization {polarization}")

    q0 = k0 / zeta0
    q_shell = k_shell / zeta_shell
    q_core = k_core / zeta_core
    x0_outer = k0 * a_outer_m
    xs_outer = k_shell * a_outer_m
    xs_core = k_shell * a_core_m
    xc_core = k_core * a_core_m
    nmax = _nmax_for_ka(max(
        abs(x0_outer), abs(xs_outer), abs(xs_core), abs(xc_core)
    ))
    n_arr = np.arange(-nmax, nmax + 1)
    a_n = np.zeros(n_arr.shape, dtype=complex)

    for idx, n_raw in enumerate(n_arr):
        n = int(n_raw)
        j0 = sp.jv(n, x0_outer)
        j0p = sp.jvp(n, x0_outer, 1)
        h0 = sp.hankel2(n, x0_outer)
        h0p = sp.h2vp(n, x0_outer, 1)

        js_o = sp.jv(n, xs_outer)
        js_op = sp.jvp(n, xs_outer, 1)
        ys_o = sp.yv(n, xs_outer)
        ys_op = sp.yvp(n, xs_outer, 1)
        js_i = sp.jv(n, xs_core)
        js_ip = sp.jvp(n, xs_core, 1)
        ys_i = sp.yv(n, xs_core)
        ys_ip = sp.yvp(n, xs_core, 1)
        jc = sp.jv(n, xc_core)
        jcp = sp.jvp(n, xc_core, 1)

        # Unknowns are [a_n, b_n, c_n, d_n] for exterior H2,
        # shell J/Y, and regular core J coefficients respectively.
        matrix = np.asarray([
            [h0, -js_o, -ys_o, 0.0],
            [q0 * h0p, -q_shell * js_op, -q_shell * ys_op, 0.0],
            [0.0, js_i, ys_i, -jc],
            [0.0, q_shell * js_ip, q_shell * ys_ip, -q_core * jcp],
        ], dtype=complex)
        rhs = np.asarray([-j0, -q0 * j0p, 0.0, 0.0], dtype=complex)
        a_n[idx] = np.linalg.solve(matrix, rhs)[0]

    amp = np.sum(a_n * (-1.0) ** n_arr)
    return float((4.0 / k0) * abs(amp) ** 2)


def sigma_coated_impedance_cylinder(
    a_inner_m,
    a_outer_m,
    eps_r,
    mu_r,
    zs_ohm,
    freq_hz,
    polarization,
):
    """Monostatic width of a dielectric-coated impedance cylinder.

    The annular dielectric is bounded by air at ``a_outer_m`` and by a
    Leontovich surface at ``a_inner_m``.  This independently exercises the
    solver's TYPE 4 dielectric/IBC boundary rather than only its PEC limit.
    """

    if not (0.0 < float(a_inner_m) < float(a_outer_m)):
        raise ValueError("Require 0 < a_inner_m < a_outer_m.")
    z_s = complex(zs_ohm)
    if abs(z_s) == 0.0:
        return sigma_coated_pec_cylinder(
            a_inner_m, a_outer_m, eps_r, mu_r, freq_hz, polarization
        )

    pol = polarization.upper()
    k0 = 2.0 * np.pi * freq_hz / C0
    n_rel = np.sqrt(complex(eps_r) * complex(mu_r))
    k1 = k0 * n_rel
    eta1 = ETA0 * complex(mu_r) / n_rel

    if pol == "TM":
        zeta_out = 1.0 + 0.0j
        zeta_in = complex(mu_r)
        beta = -1j * k1 * eta1 / z_s
    elif pol == "TE":
        zeta_out = 1.0 + 0.0j
        zeta_in = complex(eps_r)
        beta = -1j * k1 * z_s / eta1
    else:
        raise ValueError(f"Unknown polarization {polarization}")

    p = k1 / zeta_in
    q = k0 / zeta_out
    x_inner = k1 * a_inner_m
    x_outer_inner = k1 * a_outer_m
    x_outer = k0 * a_outer_m
    nmax = _nmax_for_ka(max(abs(x_inner), abs(x_outer_inner), abs(x_outer)))
    n_arr = np.arange(-nmax, nmax + 1)
    a_n = np.zeros(n_arr.shape, dtype=complex)

    for idx, n_raw in enumerate(n_arr):
        n = int(n_raw)
        j_i = sp.jv(n, x_inner)
        y_i = sp.yv(n, x_inner)
        jp_i = sp.jvp(n, x_inner, 1)
        yp_i = sp.yvp(n, x_inner, 1)
        j_oi = sp.jv(n, x_outer_inner)
        y_oi = sp.yv(n, x_outer_inner)
        jp_oi = sp.jvp(n, x_outer_inner, 1)
        yp_oi = sp.yvp(n, x_outer_inner, 1)
        j_o = sp.jv(n, x_outer)
        jp_o = sp.jvp(n, x_outer, 1)
        h_o = sp.hankel2(n, x_outer)
        hp_o = sp.h2vp(n, x_outer, 1)

        alpha = -(k1 * jp_i + beta * j_i) / (k1 * yp_i + beta * y_i)
        phi_o = j_oi + alpha * y_oi
        phip_o = jp_oi + alpha * yp_oi
        numerator = -(p * phip_o * j_o - q * phi_o * jp_o)
        denominator = p * phip_o * h_o - q * phi_o * hp_o
        a_n[idx] = numerator / denominator

    amp = np.sum(a_n * (-1.0) ** n_arr)
    return float((4.0 / k0) * abs(amp) ** 2)
