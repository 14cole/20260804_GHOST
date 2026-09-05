"""
BoR-MoM solver: PEC EFIE/CFIE + IBC (phases 1-2), PMCHWT dielectrics and
coated PEC (phase 3).

Mixed-potential EFIE, Galerkin-tested per azimuthal mode m (see
BOR_CONVENTIONS.md):

  Z I = V,   Z = j k eta0 * 2pi * [ vector-potential - (1/k^2) scalar ] terms

with triangle bases T_i(t) for both J_t and J_phi along the generatrix and
modal kernels G_m / Gc_m / Gs_m from bor_kernels.  Surface divergences use
(1/rho) d(rho T)/dt and (jm/rho) T, so only G itself is ever needed (no
kernel gradients).

Per-mode blocks (p = observation point, q = source point):

  Z^tt  =  C [ II rho rho' T T (t_rho t_rho' Gc + t_z t_z' G)
               - (1/k^2) II (rho T)' (rho' T)' G ]
  Z^tf  =  C [ II rho rho' T T t_rho(p) Gs      - (jm/k^2) II (rho T)' T G ]
  Z^ft  =  C [ -II rho rho' T T t_rho(q) Gs     + (jm/k^2) II T (rho' T)' G ]
  Z^ff  =  C [ II rho rho' T T Gc               - (m^2/k^2) II T T G ]
  C = j k eta0 2pi

Axis conditions at rho = 0 endpoints: end bases vanish except for |m| = 1.
For those modes the meridional and azimuthal coefficients are constrained to
the regular Cartesian-vector combination J_phi = j*m*sign(t_rho)*J_t.

Excitation: plane wave from direction (sin th, 0, cos th), phase
e^{+jk d.r} (matches the 2D solver), theta-pol (VV) / phi-pol (HH).
"""

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from solver_metrics import active_metrics, profiled_solve, timed_stage
from scipy import special as sp
from scipy.linalg import get_lapack_funcs
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

from bor_kernels import (
    C0, ETA0, FFT_BUILD_BUDGET, N_XI_SAFETY_CAP, Generatrix, cached_leggauss,
    gauss_on_generatrix,
    modal_kernels_fft, modal_kernels_near, kernels_for_mode,
    mfie_kernels_fft, mfie_kernels_near, mfie_for_mode,
    ibc_kernels_fft, ibc_kernels_near, n_xi_for_pairs,
)
from rcs_solver import _memory_gate_message, _solve_memory_limit_gb

# ``||A x - b||_2 / ||b||_2`` remains useful telemetry, but it is not a
# scale-invariant acceptance test: cancellation in ``A x`` can make it much
# larger than the actual backward error of an otherwise stable LU solve.  Keep
# the historical value as the point at which iterative refinement is tried and
# as an advisory diagnostic.  Release is governed by the normwise backward
# error below.
BOR_LINEAR_RESIDUAL_MAX = 1.0e-8
BOR_LINEAR_BACKWARD_ERROR_MAX = 1.0e-12
BOR_CONDITION_EST_MAX = 1.0e12

# Conservative complex128-equivalent storage held by one concurrently solved
# azimuthal mode.  The matrix allowance covers the assembled/full and reduced
# systems, the retained solve matrix, LU storage/copies, dense projection or
# condition-norm scratch, and one additional BLAS/LAPACK work copy.  The RHS
# allowance covers B, X, residual/refinement candidates, and the bounded
# full-solution reconstruction used by batch far-field evaluation.  These are
# deliberately upper bounds: the gate must reject a solve before the kernel
# tables are prepared rather than discover the peak through paging or an OOM.
BOR_DENSE_MATRIX_EQUIVALENTS = 8.0
BOR_DENSE_RHS_EQUIVALENTS = 12.0
BOR_TABLE_BUILD_PEAK_FACTOR = 3.5
BOR_PEAK_SAFETY_FACTOR = 1.20
BOR_PEAK_FIXED_MARGIN_GB = 0.5
_COMPLEX128_BYTES = np.dtype(np.complex128).itemsize


def _reduce_constrained_operator(matrix, transform):
    """Apply sparse pole/junction relations without dense cubic products."""
    q = csr_matrix(transform)
    return q.conj().T @ (q.T @ matrix.T).T


def estimate_bor_dense_peak_gb(
    n_dofs: 'int',
    n_rhs: 'int',
    workers: 'int' = 1,
    mode_tasks: 'Optional[int]' = None,
) -> 'float':
    """Conservative peak GB for the concurrent dense BoR linear systems.

    ``mode_tasks`` is the number of independent absolute-mode tasks that can
    actually be scheduled (normally ``m_max + 1``).  Capping the requested
    worker count by it accounts for real concurrency without charging for
    idle executor threads.
    """

    try:
        dofs = int(n_dofs)
        rhs_count = int(n_rhs)
        worker_count = max(1, int(workers))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "BoR memory estimates require integer DOFs, RHS count, and workers."
        ) from exc
    if dofs != n_dofs or dofs <= 0:
        raise ValueError("BoR dense-system DOFs must be a positive integer.")
    if rhs_count != n_rhs or rhs_count <= 0:
        raise ValueError("BoR dense-system RHS count must be a positive integer.")
    if mode_tasks is None:
        active_workers = worker_count
    else:
        try:
            task_count = int(mode_tasks)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "BoR mode-task count must be a positive integer."
            ) from exc
        if task_count != mode_tasks or task_count <= 0:
            raise ValueError("BoR mode-task count must be a positive integer.")
        active_workers = min(worker_count, task_count)

    matrix_bytes = (
        BOR_DENSE_MATRIX_EQUIVALENTS
        * dofs
        * dofs
        * _COMPLEX128_BYTES
    )
    rhs_bytes = (
        BOR_DENSE_RHS_EQUIVALENTS
        * dofs
        * rhs_count
        * _COMPLEX128_BYTES
    )
    # Pivot arrays, per-RHS norms, modal increments, and allocator alignment
    # are lower order, but include an explicit 5% margin rather than silently
    # treating them as free.
    per_worker_bytes = 1.05 * (matrix_bytes + rhs_bytes)
    return active_workers * per_worker_bytes / 1.0e9


def estimate_bor_total_peak_gb(
    assembly_peak_gb: 'float',
    dense_peak_gb: 'float',
) -> 'float':
    """Return the scheduler/runtime reservation for one BoR solve.

    ``assembly_peak_gb`` is already a *peak*, not persistent storage plus a
    second workspace allocation.  This distinction prevents double counting
    the retained tables when a caller has a more detailed build-workspace
    estimate, while still letting the direct all-mode table path pass its
    documented 3.5x construction peak.
    """

    assembly_peak = float(assembly_peak_gb)
    dense_peak = float(dense_peak_gb)
    if (
        not math.isfinite(assembly_peak)
        or assembly_peak < 0.0
        or not math.isfinite(dense_peak)
        or dense_peak < 0.0
    ):
        raise ValueError(
            "BoR peak-memory components must be finite and non-negative."
        )
    raw_peak = assembly_peak + dense_peak
    return max(
        BOR_PEAK_FIXED_MARGIN_GB,
        BOR_PEAK_FIXED_MARGIN_GB + BOR_PEAK_SAFETY_FACTOR * raw_peak,
    )


def _guard_bor_dense_memory(
    n_dofs: 'int',
    n_rhs: 'int',
    workers: 'int',
    mode_tasks: 'int',
    assembly_peak_gb: 'float' = 0.0,
    context: 'str' = "The BoR solve",
) -> 'float':
    """Gate a BoR solve before operator preparation and return required GB."""

    assembly_peak = float(assembly_peak_gb)
    if not math.isfinite(assembly_peak) or assembly_peak < 0.0:
        raise ValueError(
            "BoR assembly peak memory must be finite and non-negative."
        )
    dense = estimate_bor_dense_peak_gb(
        n_dofs,
        n_rhs,
        workers=workers,
        mode_tasks=mode_tasks,
    )
    required = estimate_bor_total_peak_gb(assembly_peak, dense)
    memory_limit_gb = _solve_memory_limit_gb()
    if required > memory_limit_gb:
        active_workers = min(max(1, int(workers)), int(mode_tasks))
        raise MemoryError(
            _memory_gate_message(
                required,
                memory_limit_gb,
                context,
                (
                    f"Planned dense system: {int(n_dofs)} complex128 DOFs, "
                    f"{int(n_rhs)} simultaneous RHS columns, "
                    f"{active_workers} concurrent mode worker"
                    f"{'s' if active_workers != 1 else ''}; estimated dense "
                    f"peak {dense:.2f} GB plus {assembly_peak:.2f} GB for "
                    "operator/table preparation; the total includes the "
                    f"{BOR_PEAK_SAFETY_FACTOR:.2f}x safety factor and "
                    f"{BOR_PEAK_FIXED_MARGIN_GB:.2f} GB fixed margin."
                ),
                (
                    "Reduce the mesh, aspect count, or worker count; for the "
                    "direct PEC/IBC solver, streaming or a lower stream "
                    "budget can also reduce resident assembly memory."
                ),
            )
        )
    return required


# -----------------------------------------------------------------------------
# Refined near-pair Galerkin integration (log singularity along diagonal /
# shared corner) via quadtree grading toward the singular set.
# -----------------------------------------------------------------------------

def _graded_cells(kind: 'str', depth: 'int' = 4) -> 'List[Tuple[float, float, float, float]]':
    """Cells (s0, s1, sp0, sp1) covering [0,1]^2 refined toward the singular
    set: kind = 'diag' (s == s'), 'corner00', 'corner01', 'corner10', 'corner11'
    where cornerAB means singular at s = A, s' = B."""

    cells = []

    def touches(kind, s0, s1, p0, p1):
        if kind == "diag":
            return not (s1 <= p0 or p1 <= s0)
        a = 0.0 if kind[6] == "0" else 1.0
        b = 0.0 if kind[7] == "0" else 1.0
        return (s0 <= a <= s1) and (p0 <= b <= p1)

    def recurse(s0, s1, p0, p1, d):
        if not touches(kind, s0, s1, p0, p1) or d >= depth:
            cells.append((s0, s1, p0, p1))
            return
        sm, pm = 0.5 * (s0 + s1), 0.5 * (p0 + p1)
        recurse(s0, sm, p0, pm, d + 1)
        recurse(s0, sm, pm, p1, d + 1)
        recurse(sm, s1, p0, pm, d + 1)
        recurse(sm, s1, pm, p1, d + 1)

    recurse(0.0, 1.0, 0.0, 1.0, 0)
    return cells


_CELL_CACHE: 'Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]' = {}


def _cell_points(kind: 'str', gorder: 'int' = 4,
                 depth: 'int' = 4) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """(s_points, sp_points, weights) for the graded cell set of `kind`."""

    depth = max(0, int(depth))
    key = f"{kind}:{gorder}:{depth}"
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    xg, wg = cached_leggauss(gorder)
    u = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    # Source axis uses order+1 so test/source nodes NEVER coincide exactly:
    # a coincident pair (R = 0) underflows the 1/R^3 MFIE kernel into NaN
    # and adds ln(eps)-noise to the EFIE log cells.
    xq, wq_ = cached_leggauss(gorder + 1)
    uq = 0.5 * (xq + 1.0)
    wq = 0.5 * wq_
    S, SP, W = [], [], []
    for (s0, s1, p0, p1) in _graded_cells(kind, depth=depth):
        hs, hp = s1 - s0, p1 - p0
        ss = s0 + u * hs
        pp = p0 + uq * hp
        SS, PP = np.meshgrid(ss, pp, indexing="ij")
        WW = np.outer(w * hs, wq * hp)
        S.append(SS.ravel()); SP.append(PP.ravel()); W.append(WW.ravel())
    out = (np.concatenate(S), np.concatenate(SP), np.concatenate(W))
    _CELL_CACHE[key] = out
    return out


def _regular_cell_points(gorder: 'int' = 12) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Tensor Gauss points for close but nonsingular element pairs."""

    key = f"regular:{int(gorder)}"
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    xg, wg = cached_leggauss(int(gorder))
    u = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    s, sp = np.meshgrid(u, u, indexing="ij")
    weights = np.outer(w, w)
    out = (s.ravel(), sp.ravel(), weights.ravel())
    _CELL_CACHE[key] = out
    return out


# -----------------------------------------------------------------------------
# Point-level geometry helpers
# -----------------------------------------------------------------------------

def _points_on_element(gen: 'Generatrix', e: 'int', s: 'np.ndarray'):
    n0, n1 = gen.elem_n0[e], gen.elem_n1[e]
    r0, r1 = gen.nodes[n0], gen.nodes[n1]
    rho = r0[0] + s * (r1[0] - r0[0])
    z = r0[1] + s * (r1[1] - r0[1])
    L = gen.lengths[e]
    T0, T1 = 1.0 - s, s
    drho = r1[0] - r0[0]
    dRT0 = (drho * (1.0 - s) - rho) / L
    dRT1 = (drho * s + rho) / L
    return rho, z, gen.trho[e], gen.tz[e], T0, T1, dRT0, dRT1, L


def _pair_blocks(m: 'int', k: 'float',
                 rho_p, tr_p, tz_p, T_p, D_p, w_p,
                 rho_q, tr_q, tz_q, T_q, D_q, w_q,
                 G, Gc, Gs):
    """
    The four per-mode Galerkin blocks for a set of weighted point pairs.

    T_p/D_p: [n_bases_p, n_pts] shape and (rho T)' matrices for the test side
    (likewise source side).  G/Gc/Gs: [n_pts_p, n_pts_q] kernels.  Weights
    include dt measures.  Returns (ztt, ztf, zft, zff) WITHOUT the C factor.
    """

    wp = w_p; wq = w_q
    rrw = (rho_p * wp)[:, None] * (rho_q * wq)[None, :]
    K_tt_vec = rrw * ((tr_p[:, None] * tr_q[None, :]) * Gc + (tz_p[:, None] * tz_q[None, :]) * G)
    K_sc = (wp[:, None] * wq[None, :]) * G
    K_tf_vec = rrw * (tr_p[:, None] * Gs)
    K_ft_vec = -rrw * (tr_q[None, :] * Gs)
    K_ff_vec = rrw * Gc

    ztt = T_p @ K_tt_vec @ T_q.T - (1.0 / k ** 2) * (D_p @ K_sc @ D_q.T)
    ztf = T_p @ K_tf_vec @ T_q.T - (1j * m / k ** 2) * (D_p @ K_sc @ T_q.T)
    zft = T_p @ K_ft_vec @ T_q.T + (1j * m / k ** 2) * (T_p @ K_sc @ D_q.T)
    zff = T_p @ K_ff_vec @ T_q.T - (m ** 2 / k ** 2) * (T_p @ K_sc @ T_q.T)
    return ztt, ztf, zft, zff


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------

def _causal_medium(eps_r: 'complex', mu_r: 'complex') -> 'Tuple[complex, complex]':
    """(m, eta_r) for a homogeneous medium: refractive index with Im(m) <= 0
    (causal decay, same branch as mie_sphere/_causal_index) and the
    relative impedance eta_r = mu_r / m, which guarantees k*eta = w mu mu0
    and k/eta = w eps eps0 exactly for whichever branch m took.

    For a passive double-negative medium the causal root has Re(m) < 0 and
    Im(m) < 0.  Do not subsequently force Re(m) positive: that selects the
    exponentially growing root.  In the exactly lossless case both roots
    have zero imaginary part, so choose the one with non-negative real wave
    impedance (forward power flow)."""

    eps_r = complex(eps_r)
    mu_r = complex(mu_r)
    if not (
        math.isfinite(eps_r.real)
        and math.isfinite(eps_r.imag)
        and math.isfinite(mu_r.real)
        and math.isfinite(mu_r.imag)
    ):
        raise ValueError("BoR medium epsilon and mu must be finite.")
    singular_tol = 1.0e-15
    if abs(eps_r) <= singular_tol:
        raise ValueError(
            "BoR PMCHWT does not support singular/near-ENZ epsilon."
        )
    if abs(mu_r) <= singular_tol:
        raise ValueError(
            "BoR PMCHWT does not support singular/near-MNZ mu."
        )
    eps_tol = 64.0 * np.finfo(float).eps * max(1.0, abs(eps_r))
    mu_tol = 64.0 * np.finfo(float).eps * max(1.0, abs(mu_r))
    if eps_r.imag > eps_tol or mu_r.imag > mu_tol:
        raise ValueError(
            "BoR PMCHWT supports passive media only. Under the "
            "exp(+j*omega*t) convention, Im(epsilon) and Im(mu) must be <= 0."
        )

    m = np.sqrt(eps_r * mu_r)
    branch_tol = 64.0 * np.finfo(float).eps * max(1.0, abs(m))
    if m.imag > branch_tol:
        m = -m
    elif abs(m.imag) <= branch_tol:
        eta_try = complex(mu_r) / m
        if eta_try.real < 0.0:
            m = -m
    return m, complex(mu_r) / m


def _validate_bor_surface_impedance(values, context: 'str') -> 'np.ndarray':
    """Return finite passive Leontovich impedances for exp(+j omega t)."""

    array = np.asarray(values, dtype=np.complex128)
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{context} contains a non-finite surface impedance.")
    tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(array))
    if np.any(array.real < -tolerance):
        bad = complex(array.flat[int(np.flatnonzero(array.real < -tolerance)[0])])
        raise ValueError(
            f"{context} contains active negative resistance {bad.real:g} ohm; "
            "passive BoR IBCs require Re(Zs) >= 0."
        )
    return array


EFFECTIVELY_REACTIVE_IBC_RATIO = 1.0e-3


def _effectively_reactive_surface_impedance(values) -> 'bool':
    """Whether a nonzero IBC lacks enough resistance for closed EFIE safety."""

    array = np.asarray(values, dtype=np.complex128)
    if array.size == 0:
        return False
    peak_magnitude = float(np.max(np.abs(array)))
    if peak_magnitude == 0.0:
        return False
    peak_resistance = float(np.max(np.abs(np.real(array))))
    return (
        peak_resistance
        < EFFECTIVELY_REACTIVE_IBC_RATIO * peak_magnitude
    )


class BorPecSolver:
    """Single-surface BoR operator factory + PEC/IBC solver.

    With medium=(eps_r, mu_r) the EFIE (T) and rotated-PV (P) operators are
    assembled in that homogeneous medium (complex k, medium eta) -- the
    building blocks of the phase-3 PMCHWT systems.  Excitation and far-field
    methods always refer to the EXTERIOR (air) and are only meaningful on an
    instance with medium=None."""

    def __init__(self, points, freq_hz: 'float', gauss_order: 'int' = 4,
                 near_depth: 'int' = 4, medium=None, single_tables: 'bool' = False):
        # single_tables stores the FAR kernel tables (the memory bound at
        # scale) as complex64; all near/self quadrature and the assembled
        # mode systems stay double.  Validated at <= 0.005 dB vs double.
        freq_hz = float(freq_hz)
        if (not math.isfinite(freq_hz)) or freq_hz <= 0.0:
            raise ValueError("BoR frequency must be a positive finite value.")
        self._table_dtype = np.complex64 if single_tables else np.complex128
        self.gen = Generatrix(np.asarray(points, dtype=float))
        k0 = 2.0 * math.pi * freq_hz / C0
        if medium is None:
            self.k = k0
            self.eta = ETA0
        else:
            m_idx, eta_r = _causal_medium(*medium)
            self.k = k0 * m_idx
            self.eta = ETA0 * eta_r
        self.freq_hz = freq_hz
        self.g = gauss_on_generatrix(self.gen, gauss_order)
        self.gauss_order = gauss_order
        self.near_depth = int(near_depth)
        if self.near_depth < 0:
            raise ValueError("near_depth must be a non-negative integer.")
        # A one-panel near stencil leaves the second neighbours of a smooth
        # refined curve in the FFT table.  Their gap scales like h, which can
        # force an arbitrarily large azimuth grid even though the interaction
        # is local and is better integrated directly.  Keep two neighbours on
        # each side in the near stencil. Geometrically sharp pairs beyond that
        # stencil are added below using the FFT resolution threshold.
        self.near_span = 2
        self._configure_near_pair_routing()
        self.Nn = self.gen.n_nodes
        self._build_point_matrices()
        # Compute the TRUE closest topologically-far same-surface pair before
        # assembly.  This both rejects non-manifold near-touches and lets the
        # FFT sampler fail closed when a re-entrant fold would exceed its
        # resolution cap.  The cached result is shared by table/streaming and
        # EFIE/MFIE/IBC paths.
        far_gap = self._far_gap()
        if far_gap > 0.0:
            try:
                n_xi_for_pairs(
                    self.k,
                    float(np.max(self.gen.nodes[:, 0])),
                    0,
                    far_gap,
                    bracket=False,
                )
            except ValueError as exc:
                pair = getattr(self, "_far_gap_pair", None)
                pair_note = (
                    f" for nonadjacent elements {pair[0]} and {pair[1]}"
                    if pair is not None else ""
                )
                raise ValueError(
                    "BoR same-surface far-quadrature preflight failed"
                    f"{pair_note}: {exc}"
                ) from exc
        self._G_table = None
        self._stream = None
        self._near_cache: 'Dict[int, Dict[Tuple[int, int], Tuple]]' = {}
        self._near_contractions: 'Dict[Tuple[str, int], Dict[str, np.ndarray]]' = {}
        self._mass_cache = None
        self._weighted_mass_cache = None
        self._basis_mask_cache: 'Dict[int, np.ndarray]' = {}
        self._basis_transform_cache: 'Dict[int, np.ndarray]' = {}
        self._angular_local = threading.local()

    def enable_streaming(self, m_max: 'int', efie: 'bool' = True,
                         mfie: 'bool' = False,
                         ibc_zs_pt: 'Optional[np.ndarray]' = None,
                         pmchwt: 'bool' = False,
                         single_blocks: 'bool' = False,
                         tile_budget_gb: 'float' = 1.0,
                         workers: 'int' = 1,
                         mode_block: 'Optional[int]' = None) -> 'None':
        """Phase-7b: build per-mode nodal far blocks instead of the
        [P, P, modes] Gauss-point tables (see bor_streaming).  Must be
        called before any assemble_* call; the near/self machinery is
        unaffected.  With IBC, the source Z_s is baked into the blocks, so
        assemble_ibc_extra must be called with the same zs_pt.  PMCHWT uses
        the same rotated-PV blocks with an exact unit source weight."""

        from bor_streaming import StreamingFarBlocks
        self._stream = StreamingFarBlocks(
            self, m_max, efie=efie, mfie=mfie, ibc_zs_pt=ibc_zs_pt,
            pmchwt=pmchwt,
            dtype=np.complex64 if single_blocks else np.complex128,
            tile_budget_gb=tile_budget_gb, workers=workers,
            mode_block=mode_block)

    # -- base-point kernel table (far pairs only; near pairs zeroed) --
    def _build_point_matrices(self):
        g = self.g
        P = len(g.rho)
        self.P = P
        # The dense nodal-to-Gauss matrices are useful for the legacy table
        # contractions, but cost O(Nn*P) storage each.  The streaming path,
        # near contractions, excitation, and far field all use the exact
        # two-node element scatter/evaluate operations below and never need
        # these matrices.  Build them lazily so a large streaming solve does
        # not retain two unnecessary dense arrays.
        self._B_T = None
        self._B_D = None
        # element adjacency classes for pair routing
        self.elem_of_pt = g.elem

    def _ensure_dense_point_matrices(self) -> 'None':
        if self._B_T is not None:
            return
        g = self.g
        T = np.zeros((self.Nn, self.P))
        D = np.zeros((self.Nn, self.P))
        p = np.arange(self.P)
        e = g.elem.astype(int, copy=False)
        T[e, p] = g.T0
        D[e, p] = g.dRT0
        T[e + 1, p] = g.T1
        D[e + 1, p] = g.dRT1
        self._B_T, self._B_D = T, D

    @property
    def B_T(self) -> 'np.ndarray':
        self._ensure_dense_point_matrices()
        return self._B_T

    @property
    def B_D(self) -> 'np.ndarray':
        self._ensure_dense_point_matrices()
        return self._B_D

    def _test_accumulate(self, point_values: 'np.ndarray') -> 'np.ndarray':
        """Apply the triangle test basis without the dense ``B_T`` matrix.

        Every Gauss point belongs to exactly one element and therefore touches
        only its two endpoint nodes.  Explicit local scatter is O(P), whereas
        the mathematically equivalent dense multiply is O(Nn*P).
        """

        values = np.asarray(point_values)
        if values.shape[0] != self.P:
            raise ValueError("BoR point-value array has the wrong leading size.")
        trailing = values.shape[1:]
        scale_shape = (self.P,) + (1,) * len(trailing)
        out = np.zeros((self.Nn,) + trailing,
                       dtype=np.result_type(values.dtype, np.float64))
        elem = self.g.elem.astype(int, copy=False)
        np.add.at(out, elem, self.g.T0.reshape(scale_shape) * values)
        np.add.at(out, elem + 1, self.g.T1.reshape(scale_shape) * values)
        return out

    def _basis_evaluate(self, nodal_values: 'np.ndarray') -> 'np.ndarray':
        """Evaluate nodal triangle coefficients at Gauss points in O(P)."""

        values = np.asarray(nodal_values)
        if values.shape[0] != self.Nn:
            raise ValueError("BoR nodal array has the wrong leading size.")
        trailing = values.shape[1:]
        scale_shape = (self.P,) + (1,) * len(trailing)
        elem = self.g.elem.astype(int, copy=False)
        return (
            self.g.T0.reshape(scale_shape) * values[elem]
            + self.g.T1.reshape(scale_shape) * values[elem + 1]
        )

    def _configure_near_pair_routing(self) -> 'None':
        """Route geometrically sharp pairs to direct modal integration.

        The topological stencil handles self/shared-node pairs.  In addition,
        any disjoint segment pair whose meridian gap would by itself require
        more than the bounded azimuthal FFT grid is treated by the existing
        regular-cell near path.  This prevents smooth electrically large
        bodies from acquiring an artificial radius ceiling while retaining
        the cap for genuinely excessive global oscillation/mode bandwidth.
        """

        gen = self.gen
        ne = gen.n_elems
        sources = [set(range(max(0, e - self.near_span),
                             min(ne, e + self.near_span + 1)))
                   for e in range(ne)]
        rho_max = float(np.max(gen.nodes[:, 0]))
        # n_xi_for_pairs uses eight samples across the closest far-pair peak.
        # A small margin keeps power-of-two rounding on the safe side.
        direct_gap = (
            1.001 * 8.0 * 2.0 * math.pi * rho_max / N_XI_SAFETY_CAP
        )
        scale = max(
            float(np.ptp(gen.nodes[:, 0])),
            float(np.ptp(gen.nodes[:, 1])),
            1.0e-15,
        )
        touch_tol = max(1.0e-14, 1.0e-10 * scale)
        mids = 0.5 * (gen.nodes[gen.elem_n0] + gen.nodes[gen.elem_n1])
        half = 0.5 * gen.lengths
        if ne and direct_gap > 0.0:
            tree = cKDTree(mids)
            max_half = float(np.max(half))
            for e in range(ne):
                radius = float(half[e]) + max_half + direct_gap
                for candidate in tree.query_ball_point(mids[e], radius):
                    f = int(candidate)
                    if f <= e + self.near_span:
                        continue
                    lower = (
                        float(np.linalg.norm(mids[e] - mids[f]))
                        - float(half[e]) - float(half[f])
                    )
                    if lower > direct_gap:
                        continue
                    gap = _segment_distance(
                        gen.nodes[gen.elem_n0[e]], gen.nodes[gen.elem_n1[e]],
                        gen.nodes[gen.elem_n0[f]], gen.nodes[gen.elem_n1[f]],
                    )
                    if gap <= touch_tol:
                        raise ValueError(
                            "Nonadjacent BoR generatrix elements "
                            f"{e} and {f} touch or overlap (gap {gap:.3g} m). "
                            "This creates a non-manifold surface and is not "
                            "supported."
                        )
                    if gap <= direct_gap:
                        sources[e].add(f)
                        sources[f].add(e)

        self._near_sources_by_element = tuple(
            tuple(sorted(values)) for values in sources
        )
        self._near_pair_count = sum(map(len, self._near_sources_by_element))
        self._far_gap_pair = None
        self._far_gap_cache = (
            0.0 if self._near_pair_count == ne * ne else None
        )

    def _near_gauss_mask(self) -> 'np.ndarray':
        """Boolean point-pair mask corresponding to direct element pairs."""

        ne = self.gen.n_elems
        element_mask = np.zeros((ne, ne), dtype=bool)
        for e, sources in enumerate(self._near_sources_by_element):
            element_mask[e, list(sources)] = True
        elem = self.g.elem.astype(int, copy=False)
        return element_mask[elem[:, None], elem[None, :]]

    def _far_gap(self) -> 'float':
        """Conservative minimum distance among FFT-routed element pairs.

        Same/nearby and geometrically sharp pairs use direct modal integration.
        The routing threshold is only a lower bound for every pair left in
        the FFT table.  The FFT grid must be sized from the true closest
        remaining pair: using the threshold itself needlessly drives every
        non-empty far table to the azimuthal safety cap.

        A midpoint bounding-sphere tree finds every pair that could improve
        the initial local-neighbour bound, then exact segment distance makes
        the result conservative.  The result is used identically by table
        and streaming assembly.
        """

        if getattr(self, "_far_gap_cache", None) is not None:
            return self._far_gap_cache
        gen = self.gen
        ne = gen.n_elems
        if ne <= self.near_span:
            self._far_gap_pair = None
            self._far_gap_cache = 0.0
            return self._far_gap_cache

        d = math.inf
        best_pair = None
        # Supply the tree search with a finite exact bound.  Additional
        # geometrically-near routing can consume every pair at the original
        # ``near_span + 1`` offset, so find the first topological offset that
        # still contains an FFT-routed pair.
        for offset in range(1, ne):
            found_at_offset = False
            for e in range(ne - offset):
                f = e + offset
                if f in self._near_sources_by_element[e]:
                    continue
                found_at_offset = True
                de = _segment_distance(
                    gen.nodes[gen.elem_n0[e]], gen.nodes[gen.elem_n1[e]],
                    gen.nodes[gen.elem_n0[f]], gen.nodes[gen.elem_n1[f]],
                )
                if de < d:
                    d = de
                    best_pair = (e, f)
            if found_at_offset:
                break

        if best_pair is None:
            self._far_gap_pair = None
            self._far_gap_cache = 0.0
            return self._far_gap_cache

        mids = 0.5 * (
            gen.nodes[gen.elem_n0] + gen.nodes[gen.elem_n1]
        )
        half = 0.5 * gen.lengths
        max_half = float(np.max(half))
        tree = cKDTree(mids)
        for e in range(ne):
            # If two segment bounding spheres are farther apart than the
            # current exact best distance, that pair cannot improve it.
            radius = float(half[e]) + max_half + d
            for f in tree.query_ball_point(mids[e], radius):
                f = int(f)
                if f <= e or f in self._near_sources_by_element[e]:
                    continue
                lower = (
                    float(np.linalg.norm(mids[e] - mids[f]))
                    - float(half[e]) - float(half[f])
                )
                if lower >= d:
                    continue
                de = _segment_distance(
                    gen.nodes[gen.elem_n0[e]], gen.nodes[gen.elem_n1[e]],
                    gen.nodes[gen.elem_n0[f]], gen.nodes[gen.elem_n1[f]],
                )
                if de < d:
                    d = de
                    best_pair = (e, f)

        scale = max(
            float(np.ptp(gen.nodes[:, 0])),
            float(np.ptp(gen.nodes[:, 1])),
            1.0e-15,
        )
        touch_tol = max(1.0e-14, 1.0e-10 * scale)
        if d <= touch_tol:
            raise ValueError(
                "Nonadjacent BoR generatrix elements "
                f"{best_pair[0]} and {best_pair[1]} touch or overlap "
                f"(gap {d:.3g} m). This creates a non-manifold surface and "
                "is not supported."
            )
        self._far_gap_pair = best_pair
        self._far_gap_cache = 0.0 if not math.isfinite(d) else float(d)
        return self._far_gap_cache

    def _kernel_tables(self, m_max: 'int'):
        """G_m table [P, P, m_max+2] at base Gauss points; local-neighbour
        element point-pairs zeroed (their Galerkin blocks are added by the
        refined near-pair path)."""

        if self._G_table is not None and self._G_table.shape[-1] >= m_max + 2:
            return self._G_table
        g = self.g
        # The kernel builder accepts broadcastable coordinate views; avoid
        # materializing four redundant P-by-P coordinate matrices.
        RP = g.rho[:, None]
        RQ = g.rho[None, :]
        ZP = g.z[:, None]
        ZQ = g.z[None, :]
        near_mask = self._near_gauss_mask()
        n_xi = n_xi_for_pairs(self.k, float(np.max(g.rho)), m_max,
                              self._far_gap(), bracket=False)
        G = modal_kernels_fft(RP, ZP, RQ, ZQ, self.k, m_max, n_xi=n_xi)
        # far table must not contain near-singular FFT garbage
        # (also excludes exact-coincidence pairs)
        near_flat = np.flatnonzero(near_mask.ravel())
        Gf = G.reshape(-1, G.shape[-1])
        Gf[near_flat, :] = 0.0
        self._G_table = Gf.reshape(self.P, self.P, -1).astype(
            self._table_dtype, copy=False)
        self._m_max_table = m_max
        return self._G_table

    # -- near element pairs: refined kernels cached per (e, f) --
    def _near_pair_data(self, e: 'int', f: 'int', m_max: 'int'):
        key = (e, f)
        cache = self._near_cache.setdefault(m_max, {})
        if key in cache:
            return cache[key]
        if e == f:
            s, sp, w = _cell_points("diag", depth=self.near_depth)
        elif abs(e - f) == 1:
            # shared node: e,f adjacent (f = e+1 or e-1)
            kind = "corner10" if f == e + 1 else "corner01"
            s, sp, w = _cell_points(kind, depth=self.near_depth)
        else:
            # Second neighbours are disjoint and nonsingular but sufficiently
            # close that direct modal integration is more reliable than a
            # global FFT sized from their O(h) gap.
            s, sp, w = _regular_cell_points()
        rho_p, z_p, tr_p, tz_p, T0p, T1p, D0p, D1p, Lp = _points_on_element(self.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, D0q, D1q, Lq = _points_on_element(self.gen, f, sp)
        Gm = modal_kernels_near(rho_p, z_p, rho_q, z_q, self.k, m_max)
        data = (s, sp, w * Lp * Lq,
                rho_p, tr_p, tz_p, np.vstack([T0p, T1p]), np.vstack([D0p, D1p]),
                rho_q, tr_q, tz_q, np.vstack([T0q, T1q]), np.vstack([D0q, D1q]),
                Gm)
        cache[key] = data
        return data

    # -- full node-based Z for mode m --
    def assemble_mode(self, m: 'int', m_max: 'int') -> 'np.ndarray':
        k = self.k
        g = self.g
        if self._stream is not None:
            ztt, ztf, zft, zff = self._stream.efie_blocks(m)
        else:
            Gtab = self._kernel_tables(m_max)
            G, Gc, Gs = kernels_for_mode(Gtab, m)
            ztt, ztf, zft, zff = _pair_blocks(
                m, k,
                g.rho, g.trho, g.tz, self.B_T, self.B_D, g.w,
                g.rho, g.trho, g.tz, self.B_T, self.B_D, g.w,
                G, Gc, Gs,
            )

        # Near element pairs (self + two topological neighbours) use the same
        # refined kernels as before.  ``prepare_operators`` precontracts every
        # signed mode into compact band-entry arrays, eliminating thousands of
        # tiny 2x2 einsums during the mode sweep.  Keep the direct fallback for
        # callers that assemble an operator without first preparing it.
        prepared = self._prepared_near("efie", m_max)
        if prepared is not None:
            midx = m + m_max
            rc = (prepared["rows"], prepared["cols"])
            for uv, tgt in enumerate((ztt, ztf, zft, zff)):
                np.add.at(tgt, rc, prepared["values"][uv, midx])
        else:
            ne = self.gen.n_elems
            for e in range(ne):
                for f in self._near_sources_by_element[e]:
                    (s, sp, w, rho_p, tr_p, tz_p, Tp, Dp,
                     rho_q, tr_q, tz_q, Tq, Dq, Gm) = self._near_pair_data(e, f, m_max)
                    Gn, Gcn, Gsn = kernels_for_mode(Gm, m)
                    rr = rho_p * rho_q * w
                    ktt = rr * ((tr_p * tr_q) * Gcn + (tz_p * tz_q) * Gn)
                    ksc = w * Gn
                    ktf = rr * (tr_p * Gsn)
                    kft = -rr * (tr_q * Gsn)
                    kff = rr * Gcn
                    btt = np.einsum("ip,p,jp->ij", Tp, ktt, Tq) - (1.0 / k ** 2) * np.einsum("ip,p,jp->ij", Dp, ksc, Dq)
                    btf = np.einsum("ip,p,jp->ij", Tp, ktf, Tq) - (1j * m / k ** 2) * np.einsum("ip,p,jp->ij", Dp, ksc, Tq)
                    bft = np.einsum("ip,p,jp->ij", Tp, kft, Tq) + (1j * m / k ** 2) * np.einsum("ip,p,jp->ij", Tp, ksc, Dq)
                    bff = np.einsum("ip,p,jp->ij", Tp, kff, Tq) - (m ** 2 / k ** 2) * np.einsum("ip,p,jp->ij", Tp, ksc, Tq)
                    rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
                    ztt[np.ix_(rows, cols)] += btt
                    ztf[np.ix_(rows, cols)] += btf
                    zft[np.ix_(rows, cols)] += bft
                    zff[np.ix_(rows, cols)] += bff

        C = 1j * k * self.eta * 2.0 * np.pi
        Nn = self.Nn
        Z = np.empty((2 * Nn, 2 * Nn), dtype=np.complex128)
        Z[:Nn, :Nn] = C * ztt
        Z[:Nn, Nn:] = C * ztf
        Z[Nn:, :Nn] = C * zft
        Z[Nn:, Nn:] = C * zff
        return Z

    # -- MFIE machinery (Phase 2) --
    def _mfie_tables(self, m_max: 'int'):
        """Four modal MFIE kernel tables [P, P, 2*m_max+1] at base Gauss
        points; near element-pair entries zeroed (refined path adds them)."""

        if getattr(self, "_K_tables", None) is not None:
            return self._K_tables
        g = self.g
        P = self.P
        args = (g.rho[:, None], g.z[:, None],
                g.trho[:, None], g.tz[:, None],
                g.rho[None, :], g.z[None, :],
                g.trho[None, :], g.tz[None, :])
        n_xi = n_xi_for_pairs(self.k, float(np.max(g.rho)), m_max,
                              self._far_gap(), bracket=True)
        K = mfie_kernels_fft(*args, self.k, m_max, n_xi=n_xi)
        near_flat = np.flatnonzero(self._near_gauss_mask().ravel())
        K = list(K)
        for i in range(4):
            Kf = K[i].reshape(-1, K[i].shape[-1])
            Kf[near_flat, :] = 0.0
            K[i] = Kf.reshape(P, P, -1).astype(self._table_dtype, copy=False)
        self._K_tables = tuple(K)
        return self._K_tables

    def _near_mfie_data(self, e: 'int', f: 'int', m_max: 'int'):
        cache = self._near_cache.setdefault(("mfie", m_max), {})
        if (e, f) in cache:
            return cache[(e, f)]
        if e == f:
            s, sp, w = _cell_points("diag", depth=self.near_depth)
        elif abs(e - f) == 1:
            kind = "corner10" if f == e + 1 else "corner01"
            s, sp, w = _cell_points(kind, depth=self.near_depth)
        else:
            s, sp, w = _regular_cell_points()
        rho_p, z_p, tr_p, tz_p, T0p, T1p, _, _, Lp = _points_on_element(self.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, _, _, Lq = _points_on_element(self.gen, f, sp)
        tr_pa = np.full_like(rho_p, tr_p); tz_pa = np.full_like(rho_p, tz_p)
        tr_qa = np.full_like(rho_q, tr_q); tz_qa = np.full_like(rho_q, tz_q)
        Kn = mfie_kernels_near(rho_p, z_p, tr_pa, tz_pa, rho_q, z_q, tr_qa, tz_qa,
                               self.k, m_max)
        data = (w * Lp * Lq, rho_p, np.vstack([T0p, T1p]),
                rho_q, np.vstack([T0q, T1q]), Kn)
        cache[(e, f)] = data
        return data

    def mass_blocks(self, weight=None) -> 'np.ndarray':
        """2pi * Int w(t) rho T_i T_j dt  (node-based [Nn, Nn]); weight is a
        per-Gauss-point array (default 1) -- used for the MFIE J/2 term and
        the IBC Z_s term (with weight = Z_s at the Gauss points)."""

        if weight is None and self._mass_cache is not None:
            return self._mass_cache

        g = self.g
        if weight is None:
            wgt = 1.0
            signature = None
        else:
            wgt = np.asarray(weight)
            if wgt.shape != (self.P,):
                raise ValueError("BoR mass weight must have one value per Gauss point.")
            signature = (wgt.dtype.str, wgt.shape, wgt.tobytes())
            cached = self._weighted_mass_cache
            if cached is not None and cached[0] == signature:
                return cached[1]

        K = g.w * g.rho * wgt
        elem = g.elem.astype(int, copy=False)
        M = np.zeros((self.Nn, self.Nn),
                     dtype=np.result_type(K, np.float64))
        factor = 2.0 * np.pi
        np.add.at(M, (elem, elem), factor * K * g.T0 * g.T0)
        np.add.at(M, (elem, elem + 1), factor * K * g.T0 * g.T1)
        np.add.at(M, (elem + 1, elem), factor * K * g.T1 * g.T0)
        np.add.at(M, (elem + 1, elem + 1), factor * K * g.T1 * g.T1)
        if weight is None:
            self._mass_cache = M
        else:
            self._weighted_mass_cache = (signature, M)
        return M

    def assemble_mfie_mode(self, m: 'int', m_max: 'int') -> 'np.ndarray':
        """Z_MFIE = (1/2) M - K  (node-based [2Nn, 2Nn]), where K is the
        Galerkin contraction of the modal MFIE brackets."""

        g = self.g
        if self._stream is not None and self._stream.K is not None:
            blocks = list(self._stream.bracket_blocks("mfie", m))
        else:
            Kt = self._mfie_tables(m_max)
            blocks = []
            wrho = g.w * g.rho
            for uv in range(4):
                Km = mfie_for_mode(Kt[uv], m, m_max)
                blocks.append(2.0 * np.pi * (self.B_T * wrho[None, :]) @ Km @ (self.B_T * wrho[None, :]).T)
        ktt, ktf, kft, kff = blocks

        prepared = self._prepared_near("mfie", m_max)
        if prepared is not None:
            midx = m + m_max
            rc = (prepared["rows"], prepared["cols"])
            for uv, tgt in enumerate((ktt, ktf, kft, kff)):
                np.add.at(tgt, rc, prepared["values"][uv, midx])
        else:
            ne = self.gen.n_elems
            for e in range(ne):
                for f in self._near_sources_by_element[e]:
                    w, rho_p, Tp, rho_q, Tq, Kn = self._near_mfie_data(e, f, m_max)
                    rr = rho_p * rho_q * w
                    rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
                    for uv, tgt in enumerate((ktt, ktf, kft, kff)):
                        Km = mfie_for_mode(Kn[uv], m, m_max)
                        blk = 2.0 * np.pi * np.einsum("ip,p,jp->ij", Tp, rr * Km, Tq)
                        tgt[np.ix_(rows, cols)] += blk

        Nn = self.Nn
        M = self.mass_blocks()
        Z = np.zeros((2 * Nn, 2 * Nn), dtype=np.complex128)
        Z[:Nn, :Nn] = 0.5 * M - ktt
        Z[:Nn, Nn:] = -ktf
        Z[Nn:, :Nn] = -kft
        Z[Nn:, Nn:] = 0.5 * M - kff
        return Z

    def _angular_data(self, m: 'int', theta_deg: 'float'):
        """Shared cylindrical-wave data for one mode and look direction.

        EFIE, MFIE, VV, HH, and monostatic far-field evaluation all reuse the
        same Bessel functions and axial phase.  A thread-local single-entry
        cache avoids recomputation without retaining a mesh-sized table for
        every requested angle.
        """

        key = (int(m), float(theta_deg))
        entry = getattr(self._angular_local, "entry", None)
        if entry is not None and entry[0] == key:
            return entry[1]
        th = math.radians(theta_deg)
        st, ct = math.sin(th), math.cos(th)
        u = self.k * self.g.rho * st
        phase = np.exp(1j * self.k * ct * self.g.z)
        jm = lambda n: (1j) ** n * sp.jv(n, u)
        Jm = jm(m)
        Jm_m1 = jm(m - 1)
        Jm_p1 = jm(m + 1)
        Ic = math.pi * (Jm_m1 + Jm_p1)
        Is = (math.pi / 1j) * (Jm_m1 - Jm_p1)
        I1 = 2.0 * math.pi * Jm
        data = (st, ct, phase, Ic, Is, I1)
        self._angular_local.entry = (key, data)
        return data

    def rhs_mfie_mode(self, m: 'int', theta_inc_deg: 'float', pol: 'str') -> 'np.ndarray':
        """<W, n_hat x H_inc> for the plane wave (see phase-2 derivation)."""

        g = self.g
        st, ct, P, Ic, Is, I1 = self._angular_data(m, theta_inc_deg)
        if pol.upper() in ("VV", "THETA", "TM"):
            # H_inc = -(1/eta0) y_hat e^{jk d.r}
            et = Ic / ETA0
            ef = -(g.trho * Is) / ETA0
        else:
            # H_inc = +(1/eta0) e_theta e^{jk d.r}
            et = (ct * Is) / ETA0
            ef = (ct * g.trho * Ic - st * g.tz * I1) / ETA0
        vt = self._test_accumulate(g.w * g.rho * P * et)
        vf = self._test_accumulate(g.w * g.rho * P * ef)
        return np.concatenate([vt, vf])

    # -- IBC operator (Phase 2): 0.5 Z_s mass + magnetic-current K' term --
    def _ibc_tables(self, m_max: 'int'):
        if getattr(self, "_KI_tables", None) is not None:
            return self._KI_tables
        g = self.g
        n_xi = n_xi_for_pairs(self.k, float(np.max(g.rho)), m_max,
                              self._far_gap(), bracket=True)
        K = ibc_kernels_fft(g.rho, g.z, g.trho, g.tz, g.rho, g.z, g.trho, g.tz,
                            self.k, m_max, n_xi=n_xi)
        near_flat = np.flatnonzero(self._near_gauss_mask().ravel())
        K = list(K)
        for i in range(4):
            Kf = K[i].reshape(-1, K[i].shape[-1])
            Kf[near_flat, :] = 0.0
            K[i] = Kf.reshape(self.P, self.P, -1).astype(self._table_dtype,
                                                         copy=False)
        self._KI_tables = tuple(K)
        return self._KI_tables

    def _near_ibc_data(self, e: 'int', f: 'int', m_max: 'int'):
        cache = self._near_cache.setdefault(("ibc", m_max), {})
        if (e, f) in cache:
            return cache[(e, f)]
        if e == f:
            s, sp, w = _cell_points("diag", depth=self.near_depth)
        elif abs(e - f) == 1:
            kind = "corner10" if f == e + 1 else "corner01"
            s, sp, w = _cell_points(kind, depth=self.near_depth)
        else:
            s, sp, w = _regular_cell_points()
        rho_p, z_p, tr_p, tz_p, T0p, T1p, _, _, Lp = _points_on_element(self.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, _, _, Lq = _points_on_element(self.gen, f, sp)
        Kn = ibc_kernels_near(rho_p, z_p, np.full_like(rho_p, tr_p), np.full_like(rho_p, tz_p),
                              rho_q, z_q, np.full_like(rho_q, tr_q), np.full_like(rho_q, tz_q),
                              self.k, m_max)
        data = (w * Lp * Lq, rho_p, np.vstack([T0p, T1p]),
                rho_q, np.vstack([T0q, T1q]), Kn)
        cache[(e, f)] = data
        return data

    def _rot_pv_blocks(self, m: 'int', m_max: 'int', src_wpt=None, src_welem=None):
        """Galerkin contraction of the rotated-PV brackets
        B_uv = p(R) W_u . [Rvec x (n_hat_q x f_v)] with an optional source
        weight (Z_s for the IBC path; unit for PMCHWT).  Returns the four
        node-based blocks (Btt, Btf, Bft, Bff)."""

        g = self.g
        stream_unit_source = bool(
            getattr(self._stream, "rot_pv_unit_source", False)
        )
        use_stream = (
            self._stream is not None
            and self._stream.B is not None
            and (
                (src_wpt is None and stream_unit_source)
                or (src_wpt is not None and not stream_unit_source)
            )
        )
        if use_stream:
            # Streaming blocks have either the solve's Z_s or the PMCHWT unit
            # source weight baked into the source side.
            blocks = list(self._stream.bracket_blocks("ibc", m))
        else:
            Kt = self._ibc_tables(m_max)
            wrho_p = g.w * g.rho
            wrho_q = g.w * g.rho if src_wpt is None else g.w * g.rho * src_wpt
            blocks = []
            for uv in range(4):
                Km = mfie_for_mode(Kt[uv], m, m_max)
                blocks.append(2.0 * np.pi * (self.B_T * wrho_p[None, :]) @ Km @ (self.B_T * wrho_q[None, :]).T)

        prepared = self._prepared_near("ibc", m_max)
        if prepared is not None:
            midx = m + m_max
            rc = (prepared["rows"], prepared["cols"])
            source_weight = (
                1.0 if src_welem is None
                else np.asarray(src_welem)[prepared["source_elems"]]
            )
            for uv, tgt in enumerate(blocks):
                np.add.at(
                    tgt, rc,
                    prepared["values"][uv, midx] * source_weight,
                )
        else:
            ne = self.gen.n_elems
            for e in range(ne):
                for f in self._near_sources_by_element[e]:
                    welem = 1.0 if src_welem is None else src_welem[f]
                    if abs(welem) == 0.0:
                        continue
                    w, rho_p, Tp, rho_q, Tq, Kn = self._near_ibc_data(e, f, m_max)
                    rr = rho_p * rho_q * w * welem
                    rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
                    for uv, tgt in enumerate(blocks):
                        Km = mfie_for_mode(Kn[uv], m, m_max)
                        blk = 2.0 * np.pi * np.einsum("ip,p,jp->ij", Tp, rr * Km, Tq)
                        tgt[np.ix_(rows, cols)] += blk
        return tuple(blocks)

    def assemble_ibc_extra(self, m: 'int', m_max: 'int', zs_pt: 'np.ndarray',
                           zs_elem: 'np.ndarray') -> 'np.ndarray':
        """
        IBC-EFIE additions from eliminating M = -Z_s n_hat x J:

            Z_extra = (1/2) <W, Z_s J> + <W, PV curl Int G M>

        The K' term applies Z_s at the SOURCE point.
        """

        ktt, ktf, kft, kff = self._rot_pv_blocks(m, m_max, zs_pt, zs_elem)
        Nn = self.Nn
        Mz = self.mass_blocks(weight=zs_pt)
        Z = np.zeros((2 * Nn, 2 * Nn), dtype=np.complex128)
        Z[:Nn, :Nn] = 0.5 * Mz + ktt
        Z[:Nn, Nn:] = ktf
        Z[Nn:, :Nn] = kft
        Z[Nn:, Nn:] = 0.5 * Mz + kff
        return Z

    def assemble_pmchwt_P(self, m: 'int', m_max: 'int') -> 'np.ndarray':
        """
        The PMCHWT rotated-PV operator P (node-based [2Nn, 2Nn]) acting on a
        magnetic current expanded in the SAME (t, phi) triangle bases:

            (P M)_tested = <W, PV Int p(R) Rvec x M dS'>
                         = <W, E_PV(M)> in this medium (E_s(M) = -curl Int G M)

        The IBC brackets B_uv are built for the rotated source n_hat_q x f_v
        (n x t = phi, n x phi = -t), so columns remap:  P[:, Mt] = -B[:, f_phi],
        P[:, Mphi] = +B[:, f_t].  The same bilinear form gives the H-side
        operator: <W, H_PV(J)> = -(P J).
        """

        Btt, Btf, Bft, Bff = self._rot_pv_blocks(m, m_max)
        Nn = self.Nn
        P = np.empty((2 * Nn, 2 * Nn), dtype=np.complex128)
        P[:Nn, :Nn] = -Btf
        P[:Nn, Nn:] = Btt
        P[Nn:, :Nn] = -Bff
        P[Nn:, Nn:] = Bft
        return P

    def _prepared_near(self, kind, m_max):
        """Direct operator callers get the same bounded, checked integration."""
        key = (kind, int(m_max))
        if key not in self._near_contractions:
            pairs = [(e, f) for e, sources in enumerate(self._near_sources_by_element)
                     for f in sources]
            self._prepare_near_contractions(kind, pairs, m_max)
        return self._near_contractions[key]

    def _prepare_near_contractions(self, kind, pairs, m_max):
        """Prepare compact mode blocks with bounded point-level scratch."""
        key = (kind, int(m_max))
        if key in self._near_contractions:
            return
        count = 4 * len(pairs)
        rows, cols, sources = (np.empty(count, dtype=np.intp) for _ in range(3))
        values = np.empty((4, 2 * m_max + 1, count), complex)
        for pi, (e, f) in enumerate(pairs):
            sl = slice(4 * pi, 4 * pi + 4)
            rows[sl] = (e, e, e + 1, e + 1)
            cols[sl] = (f, f + 1, f, f + 1)
            sources[sl] = f
            if abs(e - f) <= 1:
                cell = ('diag' if e == f else
                        ('corner10' if f == e + 1 else 'corner01'))
                blocks = _contract_near_points(self.gen, e, self.gen, f,
                    self.k, m_max, (kind,), _cell_points(cell, depth=self.near_depth))
            else:
                blocks, order, error = _converged_disjoint_blocks(
                    self.gen, e, self.gen, f, self.k, m_max, (kind,))
                self.near_quadrature_order_max = max(
                    getattr(self, 'near_quadrature_order_max', 0), order)
                self.near_quadrature_error_max = max(
                    getattr(self, 'near_quadrature_error_max', 0.), error)
            values[:, :, sl] = blocks[kind].reshape(4, 2 * m_max + 1, 4)
        self._near_contractions[key] = dict(rows=rows, cols=cols,
            source_elems=sources, values=values)
        self._near_cache.pop(m_max if kind == 'efie' else (kind, m_max), None)

    # -- cache warm-up (thread safety for parallel mode assembly) --
    def prepare_operators(self, m_max: 'int', efie: 'bool' = True,
                          mfie: 'bool' = False, ibc: 'bool' = False,
                          workers: 'int' = 1) -> 'None':
        """Build every kernel table and near-pair cache this solver will need
        up front, so parallel per-mode assembly only READS shared state.
        Near integration is sequential and bounded; workers parallelize mode
        assembly after these immutable compact blocks have been prepared."""

        ne = self.gen.n_elems
        pairs = [
            (e, f)
            for e, sources in enumerate(self._near_sources_by_element)
            for f in sources
        ]
        streaming = self._stream is not None   # far blocks already built
        if not streaming and (efie or mfie or ibc):
            # Table contractions use the dense basis matrices.  Realize them
            # once on the preparing thread, before parallel mode assembly,
            # rather than allowing multiple worker threads to race the lazy
            # initializer.
            self._ensure_dense_point_matrices()
        if efie:
            if not streaming:
                self._kernel_tables(m_max)
        if mfie:
            if not streaming:
                self._mfie_tables(m_max)
        if ibc:
            if not (streaming and self._stream.B is not None):
                self._ibc_tables(m_max)
        if efie:
            self._prepare_near_contractions("efie", pairs, m_max)
        if mfie:
            self._prepare_near_contractions("mfie", pairs, m_max)
        if ibc:
            self._prepare_near_contractions("ibc", pairs, m_max)

    # -- active-basis mask per mode --
    def basis_mask(self, m: 'int') -> 'np.ndarray':
        category = 0 if m == 0 else (1 if abs(m) == 1 else 2)
        cached = self._basis_mask_cache.get(category)
        if cached is not None:
            return cached
        Nn = self.Nn
        t_act = np.ones(Nn, dtype=bool)
        f_act = np.ones(Nn, dtype=bool)
        for end in (0, Nn - 1):
            if self.gen.node_on_axis(end):
                t_act[end] = (abs(m) == 1)
                f_act[end] = False
            else:
                t_act[end] = False   # open edge: J_t vanishes
                f_act[end] = True
        mask = np.concatenate([t_act, f_act])
        mask.setflags(write=False)
        self._basis_mask_cache[category] = mask
        return mask

    def basis_transform(self, m: 'int') -> 'np.ndarray':
        """Map regular reduced modal coefficients to full nodal components.

        At a smooth axis pole, a finite Cartesian tangential vector in the
        ``exp(j*m*phi)`` harmonic has ``J_phi = j*m*J_rho``.  The generatrix
        coefficient is ``J_t`` and its radial direction reverses at the two
        ends, hence the adjacent-element ``sign(t_rho)`` factor.
        """

        key = int(m) if abs(int(m)) == 1 else (0 if int(m) == 0 else 2)
        cached = self._basis_transform_cache.get(key)
        if cached is not None:
            return cached
        mask = self.basis_mask(m)
        active_rows = np.flatnonzero(mask)
        Q = np.zeros((2 * self.Nn, active_rows.size), dtype=np.complex128)
        Q[active_rows, np.arange(active_rows.size)] = 1.0
        if abs(int(m)) == 1:
            reduced_column = np.full(2 * self.Nn, -1, dtype=int)
            reduced_column[active_rows] = np.arange(active_rows.size)
            for end, element in ((0, 0), (self.Nn - 1, self.gen.n_elems - 1)):
                if not self.gen.node_on_axis(end):
                    continue
                column = int(reduced_column[end])
                if column < 0:
                    continue
                radial_sign = 1.0 if self.gen.trho[element] >= 0.0 else -1.0
                Q[self.Nn + end, column] = 1j * int(m) * radial_sign
        Q.setflags(write=False)
        self._basis_transform_cache[key] = Q
        return Q

    # -- excitation --
    def rhs_mode(self, m: 'int', theta_inc_deg: 'float', pol: 'str') -> 'np.ndarray':
        g = self.g
        st, ct, P, Ic, Is, I1 = self._angular_data(m, theta_inc_deg)
        if pol.upper() in ("VV", "THETA", "TM"):
            et = ct * g.trho * Ic - st * g.tz * I1
            ef = -ct * Is
        else:
            et = g.trho * Is
            ef = Ic
        vt = self._test_accumulate(g.w * g.rho * P * et)
        vf = self._test_accumulate(g.w * g.rho * P * ef)
        return np.concatenate([vt, vf])

    def rhs_vv_hh_batch(self, m: 'int', thetas_deg,
                        efie_scale: 'complex' = 1.0,
                        mfie_scale: 'complex' = 0.0,
                        angle_chunk: 'int' = 64) -> 'np.ndarray':
        """Return interleaved VV/HH RHS columns for all requested aspects.

        The result order is ``theta0-VV, theta0-HH, theta1-VV, ...`` to match
        ``_mode_sweep``.  Bessel/phase arrays are bounded by ``angle_chunk``;
        both polarizations and the EFIE/MFIE pieces share each evaluation.
        """

        thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
        count = len(thetas)
        out = np.zeros((2 * self.Nn, 2 * count), dtype=np.complex128)
        g = self.g
        base_weight = g.w * g.rho
        chunk_size = max(1, int(angle_chunk))
        for i0 in range(0, count, chunk_size):
            i1 = min(i0 + chunk_size, count)
            th = np.radians(thetas[i0:i1])
            st = np.sin(th)[:, None]
            ct = np.cos(th)[:, None]
            u = self.k * st * g.rho[None, :]
            phase = np.exp(1j * self.k * ct * g.z[None, :])
            jm = lambda n: (1j) ** n * sp.jv(n, u)
            Jm = jm(m)
            Jm_m1 = jm(m - 1)
            Jm_p1 = jm(m + 1)
            Ic = math.pi * (Jm_m1 + Jm_p1)
            Is = (math.pi / 1j) * (Jm_m1 - Jm_p1)
            I1 = 2.0 * math.pi * Jm
            common = phase * base_weight[None, :]

            columns = np.arange(2 * i0, 2 * i1)
            vv = columns[0::2]
            hh = columns[1::2]
            if efie_scale != 0.0:
                et_vv = ct * g.trho[None, :] * Ic - st * g.tz[None, :] * I1
                ef_vv = -ct * Is
                et_hh = g.trho[None, :] * Is
                ef_hh = Ic
                out[:self.Nn, vv] += efie_scale * self._test_accumulate(
                    (common * et_vv).T
                )
                out[self.Nn:, vv] += efie_scale * self._test_accumulate(
                    (common * ef_vv).T
                )
                out[:self.Nn, hh] += efie_scale * self._test_accumulate(
                    (common * et_hh).T
                )
                out[self.Nn:, hh] += efie_scale * self._test_accumulate(
                    (common * ef_hh).T
                )
            if mfie_scale != 0.0:
                scale = mfie_scale / ETA0
                et_vv = Ic
                ef_vv = -g.trho[None, :] * Is
                et_hh = ct * Is
                ef_hh = (
                    ct * g.trho[None, :] * Ic
                    - st * g.tz[None, :] * I1
                )
                out[:self.Nn, vv] += scale * self._test_accumulate(
                    (common * et_vv).T
                )
                out[self.Nn:, vv] += scale * self._test_accumulate(
                    (common * ef_vv).T
                )
                out[:self.Nn, hh] += scale * self._test_accumulate(
                    (common * et_hh).T
                )
                out[self.Nn:, hh] += scale * self._test_accumulate(
                    (common * ef_hh).T
                )
        return out

    def farfield_vv_hh_batch(self, m: 'int', solutions: 'np.ndarray',
                             thetas_deg, zs_pt: 'Optional[np.ndarray]' = None,
                             msolutions: 'Optional[np.ndarray]' = None,
                             angle_chunk: 'int' = 64) -> 'np.ndarray':
        """Monostatic VV/HH modal contributions for interleaved solutions."""

        thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
        solutions = np.asarray(solutions)
        if solutions.shape != (2 * self.Nn, 2 * len(thetas)):
            raise ValueError("BoR batch solution matrix has incompatible dimensions.")
        if msolutions is not None:
            msolutions = np.asarray(msolutions)
            if msolutions.shape != solutions.shape:
                raise ValueError(
                    "BoR batch magnetic-current matrix has incompatible dimensions."
                )
        out = np.zeros((2, len(thetas)), dtype=np.complex128)
        g = self.g
        k = self.k
        base_weight = g.w * g.rho
        pref_j = -1j * k * ETA0 / (4.0 * math.pi)
        pref_m = 1j * k / (4.0 * math.pi)
        chunk_size = max(1, int(angle_chunk))
        for i0 in range(0, len(thetas), chunk_size):
            i1 = min(i0 + chunk_size, len(thetas))
            th = np.radians(thetas[i0:i1])
            st = np.sin(th)[:, None]
            ct = np.cos(th)[:, None]
            u = k * st * g.rho[None, :]
            phase = np.exp(1j * k * ct * g.z[None, :])
            jm = lambda n: (1j) ** n * sp.jv(n, u)
            Jm = jm(m)
            Jm_m1 = jm(m - 1)
            Jm_p1 = jm(m + 1)
            Icos = math.pi * (Jm_m1 + Jm_p1)
            Isin = (math.pi / 1j) * (Jm_p1 - Jm_m1)
            I1 = 2.0 * math.pi * Jm
            common = (phase * base_weight[None, :]).T
            theta_t = (
                ct * g.trho[None, :] * Icos
                - st * g.tz[None, :] * I1
            ).T
            theta_f = (-ct * Isin).T
            phi_t = (g.trho[None, :] * Isin).T
            phi_f = Icos.T

            columns = np.arange(2 * i0, 2 * i1)
            Jt = self._basis_evaluate(solutions[:self.Nn, columns])
            Jf = self._basis_evaluate(solutions[self.Nn:, columns])
            Jt_vv, Jt_hh = Jt[:, 0::2], Jt[:, 1::2]
            Jf_vv, Jf_hh = Jf[:, 0::2], Jf[:, 1::2]
            vv_theta = np.sum(
                common * (Jt_vv * theta_t + Jf_vv * theta_f), axis=0
            )
            hh_phi = np.sum(
                common * (Jt_hh * phi_t + Jf_hh * phi_f), axis=0
            )
            out[0, i0:i1] = pref_j * vv_theta
            out[1, i0:i1] = pref_j * hh_phi

            if zs_pt is not None:
                zs = np.asarray(zs_pt)[:, None]
                Mt_vv, Mf_vv = zs * Jf_vv, -zs * Jt_vv
                Mt_hh, Mf_hh = zs * Jf_hh, -zs * Jt_hh
            elif msolutions is not None:
                Mt = self._basis_evaluate(msolutions[:self.Nn, columns])
                Mf = self._basis_evaluate(msolutions[self.Nn:, columns])
                Mt_vv, Mt_hh = Mt[:, 0::2], Mt[:, 1::2]
                Mf_vv, Mf_hh = Mf[:, 0::2], Mf[:, 1::2]
            else:
                Mt_vv = None
            if Mt_vv is not None:
                vv_phi_m = np.sum(
                    common * (Mt_vv * phi_t + Mf_vv * phi_f), axis=0
                )
                hh_theta_m = np.sum(
                    common * (Mt_hh * theta_t + Mf_hh * theta_f), axis=0
                )
                out[0, i0:i1] -= pref_m * vv_phi_m
                out[1, i0:i1] += pref_m * hh_theta_m
        return out

    def rhs_h_mode(self, m: 'int', theta_inc_deg: 'float', pol: 'str') -> 'np.ndarray':
        """UNROTATED <W, H_inc> (PMCHWT H-row; the MFIE rhs is the rotated
        <W, n_hat x H_inc>).  The incident pair is (E, H):
            VV: E = e_theta P,  H = -(1/eta0) y_hat P
            HH: E = y_hat P,    H = +(1/eta0) e_theta P
        so <W, H_inc> reuses rhs_mode with the polarizations swapped."""

        if pol.upper() in ("VV", "THETA", "TM"):
            return -self.rhs_mode(m, theta_inc_deg, "HH") / ETA0
        return self.rhs_mode(m, theta_inc_deg, "VV") / ETA0

    # -- far field for one mode's solution --
    def farfield_mode(self, m: 'int', sol: 'np.ndarray', theta_s_deg: 'float',
                      zs_pt: 'Optional[np.ndarray]' = None,
                      msol: 'Optional[np.ndarray]' = None) -> 'Tuple[complex, complex]':
        """
        Modal far-field (F_theta, F_phi).  For IBC surfaces the eliminated
        magnetic current M = -Z_s n_hat x J still RADIATES:
            M_t = Z_s J_phi,  M_phi = -Z_s J_t
            F_theta^M = -(jk/4pi) Int M . phi_hat_s e^{jk r_hat . r'}
            F_phi^M   = +(jk/4pi) Int M . theta_hat_s e^{...}
        (Weston's Z_s = eta0 null is exactly the J/M far-field cancellation --
        omitting this term leaves the operator right but the RCS wrong.)
        """

        g = self.g
        k = self.k
        Nn = self.Nn
        st, ct, P, Icos, Is_rhs, I1 = self._angular_data(m, theta_s_deg)
        # The RHS convention stores J_{m-1} - J_{m+1}; the radiation
        # projection uses the opposite sine ordering.
        Isin = -Is_rhs
        Jt = self._basis_evaluate(sol[:Nn])
        Jf = self._basis_evaluate(sol[Nn:])
        common = g.w * g.rho * P

        def proj_theta(Xt, Xf):
            return np.sum(common * (Xt * (ct * g.trho * Icos - st * g.tz * I1) + Xf * (-ct * Isin)))

        def proj_phi(Xt, Xf):
            return np.sum(common * (Xt * g.trho * Isin + Xf * Icos))

        pref_j = -1j * k * ETA0 / (4.0 * math.pi)
        f_theta = pref_j * proj_theta(Jt, Jf)
        f_phi = pref_j * proj_phi(Jt, Jf)
        Mt = Mf = None
        if zs_pt is not None:
            Mt = zs_pt * Jf
            Mf = -zs_pt * Jt
        elif msol is not None:
            Mt = self._basis_evaluate(msol[:Nn])
            Mf = self._basis_evaluate(msol[Nn:])
        if Mt is not None:
            pref_m = 1j * k / (4.0 * math.pi)
            f_theta += -pref_m * proj_phi(Mt, Mf)
            f_phi += pref_m * proj_theta(Mt, Mf)
        return f_theta, f_phi


def _mode_sweep(n_dofs: 'int', thetas, pols, m_max: 'int', mode_tol: 'float',
                assemble: 'Callable', rhs: 'Callable', farfield: 'Callable',
                prepare: 'Optional[Callable]' = None, workers: 'int' = 1,
                progress: 'Optional[Callable]' = None,
                check_abort: 'Optional[Callable]' = None,
                monitor_cond: 'bool' = False,
                rhs_batch: 'Optional[Callable]' = None,
                farfield_batch: 'Optional[Callable]' = None,
                min_mode_before_tail: 'int' = 0,
                assembly_peak_gb: 'float' = 0.0,
                memory_context: 'str' = "The BoR solve",
                signed_mode_symmetry: 'bool' = False):
    """
    Shared adaptive azimuthal-mode loop for every BoR formulation.

    assemble(m) -> (A_masked, mask); rhs(m, theta, pol) -> V (full, unmasked);
    farfield(m, full_sol, theta, pol) -> complex modal far-field contribution.

    mask=None means the closures own the reduction: rhs returns the REDUCED
    right-hand side and farfield receives the REDUCED solution vector (used
    by the junction solver, whose constraint reduction A_red = Q^T A Q is
    not expressible as a boolean mask).

    Each mode's system is factored ONCE: every (theta, pol) is a stacked RHS
    column of a single np.linalg.solve -- an aspect sweep at fixed frequency
    costs one assembly + one LU per mode.  Modes are independent, so waves of
    `workers` modes run on threads (BLAS releases the GIL); call prepare
    first so kernel/near caches are read-only during the parallel section.
    Accumulation and the 2-quiet-modes truncation test remain in strict mode
    order, so results are identical to the serial loop.  Every polarization
    and look must satisfy the tail tolerance independently; a strong return
    cannot provide the normalization for an unrelated weak channel.  Tail convergence is
    not eligible before ``min_mode_before_tail``.  Production callers set
    that floor from the incident-wave azimuthal bandwidth ``k*rho*sin(theta)``
    so two accidentally quiet low modes cannot terminate an electrically
    large solve before its physically expected modal content is reached.
    """

    metrics = active_metrics()
    if metrics is not None:
        # Capturing this collector also instruments worker threads; a new
        # thread does not inherit the caller's ContextVar automatically.
        assemble = metrics.wrap("modal_assembly", assemble)
        prepare = metrics.wrap("operators", prepare)
        rhs = metrics.wrap("excitation", rhs)
        rhs_batch = metrics.wrap("excitation", rhs_batch)
        farfield = metrics.wrap("far_field", farfield)
        farfield_batch = metrics.wrap("far_field", farfield_batch)
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    pols = list(pols)
    mode_tol = float(mode_tol)
    if not math.isfinite(mode_tol) or mode_tol <= 0.0:
        raise ValueError("mode_tol must be a positive finite value.")
    F = np.zeros((len(pols), len(thetas)), dtype=np.complex128)
    workers = max(1, int(workers))
    _guard_bor_dense_memory(
        n_dofs,
        len(thetas) * len(pols),
        workers,
        max(1, int(m_max) + 1),
        assembly_peak_gb=assembly_peak_gb,
        context=memory_context,
    )
    if prepare is not None:
        prepare(m_max)
    min_mode_before_tail = max(0, int(min_mode_before_tail))

    def linear_error_metrics(A, X, B):
        """Return residual matrix, RHS-relative residual, and backward error.

        The last quantity is the standard normwise infinity-norm backward
        error, evaluated independently for every RHS column:

            ||r||_inf / (||A||_inf ||x||_inf + ||b||_inf).

        Unlike ``||r|| / ||b||``, it remains meaningful when a scaled or
        ill-conditioned system produces a large solution whose matrix-vector
        product contains substantial cancellation.
        """

        residual_matrix = A @ X - B
        residual_norms = np.asarray(
            np.linalg.norm(residual_matrix, axis=0), dtype=float
        )
        rhs_norms = np.asarray(np.linalg.norm(B, axis=0), dtype=float)
        rhs_denominators = np.where(rhs_norms > 0.0, rhs_norms, 1.0)
        relative_residual = float(
            np.max(residual_norms / rhs_denominators)
        )

        matrix_inf_norm = float(np.linalg.norm(A, ord=np.inf))
        solution_inf_norms = np.asarray(
            np.linalg.norm(X, ord=np.inf, axis=0), dtype=float
        )
        rhs_inf_norms = np.asarray(
            np.linalg.norm(B, ord=np.inf, axis=0), dtype=float
        )
        residual_inf_norms = np.asarray(
            np.linalg.norm(residual_matrix, ord=np.inf, axis=0), dtype=float
        )
        backward_denominators = (
            matrix_inf_norm * solution_inf_norms + rhs_inf_norms
        )
        backward_by_column = np.full(
            backward_denominators.shape, math.inf, dtype=float
        )
        np.divide(
            residual_inf_norms,
            backward_denominators,
            out=backward_by_column,
            where=backward_denominators > 0.0,
        )
        backward_by_column[
            (backward_denominators <= 0.0) & (residual_inf_norms == 0.0)
        ] = 0.0
        backward_error = float(np.max(backward_by_column))
        return (
            residual_matrix,
            relative_residual,
            backward_error,
            residual_norms,
            rhs_norms,
        )

    def solve_am(am: 'int'):
        dF = np.zeros_like(F)
        res = 0.0
        backward = 0.0
        refinement_count = 0
        cond = 0.0

        def reduce_rhs(full_rhs, reduction):
            if reduction is None:
                return full_rhs
            reduction_array = np.asarray(reduction)
            if reduction_array.ndim == 1:
                return full_rhs[reduction_array]
            return reduction_array.conj().T @ full_rhs

        def expand_solution(reduced_solution, reduction):
            if reduction is None:
                return reduced_solution
            reduction_array = np.asarray(reduction)
            if reduction_array.ndim == 1:
                full_solution = np.zeros(
                    n_dofs, dtype=np.complex128
                )
                full_solution[reduction_array] = reduced_solution
                return full_solution
            return reduction_array @ reduced_solution

        signed_modes = (
            [0]
            if am == 0
            else ([am] if signed_mode_symmetry else [am, -am])
        )
        for m in signed_modes:
            A, mask = assemble(m)
            mode_condition = math.nan
            if not np.all(np.isfinite(A)):
                raise RuntimeError(
                    f"BoR mode m={m} produced a non-finite system matrix; "
                    "no field is returned."
                )
            if rhs_batch is not None:
                full_B = np.asarray(rhs_batch(m, thetas, pols))
                expected = (n_dofs, len(thetas) * len(pols))
                if full_B.shape != expected:
                    raise RuntimeError(
                        f"BoR mode m={m} batch excitation has shape "
                        f"{full_B.shape}, expected {expected}."
                    )
                B = reduce_rhs(full_B, mask)
            else:
                cols = [
                    reduce_rhs(rhs(m, th, pol), mask)
                    for th in thetas for pol in pols
                ]
                B = np.stack(cols, axis=1)
            if not np.all(np.isfinite(B)):
                raise RuntimeError(
                    f"BoR mode m={m} produced a non-finite excitation; "
                    "no field is returned."
                )
            if monitor_cond:
                # Factor once, solve every RHS, then reuse that LU for LAPACK's
                # inexpensive reciprocal 1-norm condition estimate. An SVD
                # based np.linalg.cond per mode is prohibitively expensive for
                # production meshes.
                getrf, getrs, gecon = get_lapack_funcs(
                    ("getrf", "getrs", "gecon"), (A,)
                )
                if metrics is not None:
                    getrf = metrics.wrap("factorization", getrf)
                    getrs = metrics.wrap("rhs_solve", getrs)
                    gecon = metrics.wrap("condition_estimate", gecon)
                lu, piv, factor_info = getrf(
                    np.asarray(A, dtype=np.complex128).copy()
                )
                if factor_info != 0:
                    raise RuntimeError(
                        f"BoR mode m={m} LU factorization failed "
                        f"(LAPACK info={factor_info}); no field is returned."
                    )
                X, solve_info = getrs(lu, piv, B)
                if solve_info != 0:
                    raise RuntimeError(
                        f"BoR mode m={m} LU solve failed "
                        f"(LAPACK info={solve_info}); no field is returned."
                    )
                matrix_one_norm = float(np.linalg.norm(A, ord=1))
                if (
                    not math.isfinite(matrix_one_norm)
                    or matrix_one_norm <= 0.0
                ):
                    raise RuntimeError(
                        f"BoR mode m={m} has an invalid matrix 1-norm; "
                        "no field is returned."
                    )
                reciprocal_condition, cond_info = gecon(
                    lu, matrix_one_norm
                )
                if (
                    cond_info != 0
                    or not math.isfinite(float(reciprocal_condition))
                    or float(reciprocal_condition) <= 0.0
                ):
                    raise RuntimeError(
                        f"BoR mode m={m} condition estimation failed "
                        f"(LAPACK info={cond_info}); no field is returned."
                    )
                mode_condition = 1.0 / float(reciprocal_condition)
                if (
                    not math.isfinite(mode_condition)
                    or mode_condition > BOR_CONDITION_EST_MAX
                ):
                    raise RuntimeError(
                        f"BoR mode m={m} estimated 1-norm condition "
                        f"{mode_condition:.6g} exceeds the release limit "
                        f"{BOR_CONDITION_EST_MAX:.6g}; no field is returned."
                    )
                cond = max(cond, mode_condition)
            else:
                solve_linear = (
                    metrics.wrap("factor_and_rhs_solve", np.linalg.solve)
                    if metrics is not None else np.linalg.solve
                )
                X = solve_linear(A, B)
            if not np.all(np.isfinite(X)):
                raise RuntimeError(
                    f"BoR mode m={m} produced a non-finite linear-system "
                    "solution; no field is returned."
                )
            (
                residual_matrix,
                residual,
                backward_error,
                residual_norms,
                rhs_norms,
            ) = linear_error_metrics(A, X, B)

            # A cheap correction solve reuses the existing LU.  It is only
            # attempted when the historical RHS-relative diagnostic is high
            # (or the true backward-error gate is high), so ordinary modes pay
            # no extra triangular solve.  Same-precision refinement is bounded
            # and accepted only when at least one error metric improves.
            for _attempt in range(2):
                if (
                    residual <= BOR_LINEAR_RESIDUAL_MAX
                    and backward_error <= BOR_LINEAR_BACKWARD_ERROR_MAX
                ):
                    break
                if monitor_cond:
                    correction, correction_info = getrs(
                        lu, piv, -residual_matrix
                    )
                    if correction_info != 0:
                        break
                else:
                    correction = np.linalg.solve(A, -residual_matrix)
                if not np.all(np.isfinite(correction)):
                    break
                candidate = X + correction
                candidate_metrics = linear_error_metrics(A, candidate, B)
                candidate_residual = candidate_metrics[1]
                candidate_backward = candidate_metrics[2]
                if (
                    candidate_residual >= residual
                    and candidate_backward >= backward_error
                ):
                    break
                X = candidate
                (
                    residual_matrix,
                    residual,
                    backward_error,
                    residual_norms,
                    rhs_norms,
                ) = candidate_metrics
                refinement_count += 1

            if (
                not np.all(np.isfinite(rhs_norms))
                or not np.all(np.isfinite(residual_norms))
                or not math.isfinite(backward_error)
            ):
                raise RuntimeError(
                    f"BoR mode m={m} produced a non-finite linear-system "
                    "residual; no field is returned."
                )
            if backward_error > BOR_LINEAR_BACKWARD_ERROR_MAX:
                raise RuntimeError(
                    f"BoR mode m={m} normwise linear backward error "
                    f"{backward_error:.6g} exceeds the release limit "
                    f"{BOR_LINEAR_BACKWARD_ERROR_MAX:.6g} "
                    f"(RHS-relative residual {residual:.6g}, estimated "
                    f"condition {mode_condition:.6g}); no field is returned."
                )
            res = max(res, residual)
            backward = max(backward, backward_error)
            if farfield_batch is not None:
                # Reconstruct masked full solutions only for a bounded angle
                # chunk.  Holding a second full [unknown, all-RHS] matrix here
                # would double peak memory on very dense aspect sweeps.
                angle_chunk = 64
                for i0 in range(0, len(thetas), angle_chunk):
                    i1 = min(i0 + angle_chunk, len(thetas))
                    c0 = i0 * len(pols)
                    c1 = i1 * len(pols)
                    if mask is None:
                        full_X = X[:, c0:c1]
                    else:
                        reduction_array = np.asarray(mask)
                        if reduction_array.ndim == 1:
                            full_X = np.zeros(
                                (n_dofs, c1 - c0), dtype=np.complex128
                            )
                            full_X[reduction_array] = X[:, c0:c1]
                        else:
                            full_X = reduction_array @ X[:, c0:c1]
                    contributions = np.asarray(
                        farfield_batch(
                            m, full_X, thetas[i0:i1], pols
                        )
                    )
                    expected = (len(pols), i1 - i0)
                    if contributions.shape != expected:
                        raise RuntimeError(
                            f"BoR mode m={m} batch far field has shape "
                            f"{contributions.shape}, expected {expected}."
                        )
                    if not np.all(np.isfinite(contributions)):
                        raise RuntimeError(
                            f"BoR mode m={m} produced a non-finite batch "
                            "far-field contribution; no field is returned."
                        )
                    dF[:, i0:i1] += contributions
            else:
                ci = 0
                for it, th in enumerate(thetas):
                    for ip, pol in enumerate(pols):
                        if mask is None:
                            sol = X[:, ci]
                        else:
                            sol = expand_solution(X[:, ci], mask)
                        ci += 1
                        contribution = complex(farfield(m, sol, th, pol))
                        if not (
                            math.isfinite(contribution.real)
                            and math.isfinite(contribution.imag)
                        ):
                            raise RuntimeError(
                                f"BoR mode m={m} produced a non-finite far-field "
                                f"contribution at aspect {float(th):g} deg, "
                                f"polarization {pol}; no field is returned."
                            )
                        dF[ip, it] += contribution
        if signed_mode_symmetry and am > 0:
            dF *= 2.0
        return dF, res, backward, refinement_count, cond

    modes_used = 0
    quiet = 0
    am = 0
    max_res = 0.0
    max_backward_error = 0.0
    refinement_steps = 0
    last_relative_increment = math.inf
    last_absolute_increment = math.inf
    last_absolute_floor = 0.0
    worst_tail_index = (0, 0)
    conds: 'List[float]' = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while am <= m_max and quiet < 2:
            if check_abort is not None:
                check_abort()
            wave = list(range(am, min(am + workers, m_max + 1)))
            for w_am, (dF, res, backward, refined, cond) in zip(
                wave, ex.map(solve_am, wave)
            ):
                F += dF
                if not np.all(np.isfinite(F)):
                    raise RuntimeError(
                        f"BoR modal accumulation became non-finite at "
                        f"|m|={w_am}; no field is returned."
                    )
                max_res = max(max_res, res)
                max_backward_error = max(max_backward_error, backward)
                refinement_steps += int(refined)
                if monitor_cond:
                    conds.append(cond)
                modes_used = w_am
                field_abs = np.abs(F)
                increment_abs = np.abs(dF)
                global_scale = max(float(np.max(field_abs)), 1e-300)
                # A pointwise relative test is essential for weak channels and
                # looks.  The tiny peak-referenced floor lets exact/deep nulls
                # terminate without allowing an O(mode_tol) weak-field tail to
                # hide under a strong unrelated return.
                last_absolute_floor = (
                    float(mode_tol) * float(mode_tol) * global_scale
                )
                effective_scale = (
                    field_abs + float(mode_tol) * global_scale
                )
                relative_by_sample = np.divide(
                    increment_abs,
                    effective_scale,
                    out=np.zeros_like(increment_abs, dtype=float),
                    where=effective_scale > 0.0,
                )
                worst_flat = int(np.argmax(relative_by_sample))
                worst_tail_index = tuple(
                    int(value)
                    for value in np.unravel_index(
                        worst_flat, relative_by_sample.shape
                    )
                )
                last_relative_increment = float(
                    relative_by_sample[worst_tail_index]
                )
                last_absolute_increment = float(
                    increment_abs[worst_tail_index]
                )
                if (
                    w_am >= min_mode_before_tail
                    and last_relative_increment < mode_tol
                ):
                    quiet += 1
                    if quiet >= 2:
                        break
                else:
                    quiet = 0
            am = wave[-1] + 1
            if progress is not None:
                progress(modes_used, m_max)
    stats = {
        "linear_residual": max_res,
        "linear_backward_error": max_backward_error,
        "linear_refinement_steps": int(refinement_steps),
        "mode_converged": bool(quiet >= 2),
        "mode_cap": int(m_max),
        "mode_tail_start": int(min_mode_before_tail),
        "mode_quiet_count": int(quiet),
        "mode_last_relative_increment": float(last_relative_increment),
        "mode_last_absolute_increment": float(last_absolute_increment),
        "mode_tail_absolute_floor": float(last_absolute_floor),
        "mode_worst_polarization": (
            str(pols[worst_tail_index[0]]) if pols else None
        ),
        "mode_worst_theta_deg": (
            float(thetas[worst_tail_index[1]]) if thetas.size else None
        ),
        "signed_mode_symmetry_used": bool(signed_mode_symmetry),
        "linear_residual_limit": float(BOR_LINEAR_RESIDUAL_MAX),
        "linear_residual_limit_kind": "iterative_refinement_advisory",
        "linear_backward_error_limit": float(
            BOR_LINEAR_BACKWARD_ERROR_MAX
        ),
        "condition_est_computed": bool(monitor_cond),
        "condition_est_method": (
            "lapack_gecon_1norm" if monitor_cond else None
        ),
        "condition_est_limit": float(BOR_CONDITION_EST_MAX),
    }
    if monitor_cond and conds:
        stats["max_cond"] = max(conds)
        stats["median_cond"] = float(np.median(conds))
    return F, modes_used, stats


def _mode_cap_warning(stats: 'Dict', mode_tol: 'float') -> 'Optional[str]':
    """Return a standard warning when adaptive modal truncation hit its cap."""

    if bool(stats.get("mode_converged", False)):
        return None
    cap = int(stats.get("mode_cap", -1))
    tail_start = int(stats.get("mode_tail_start", 0))
    tail = float(stats.get("mode_last_relative_increment", math.inf))
    if cap < tail_start:
        return (
            f"Azimuthal mode cap m={cap} is below the incident-wave physical "
            f"bandwidth estimate m={tail_start}. Increase n_modes; adaptive "
            "tail convergence was intentionally not evaluated."
        )
    return (
        "Azimuthal mode truncation did not reach two consecutive increments "
        f"below mode_tol={float(mode_tol):.3g} at or above the physical "
        f"tail start m={tail_start} before the cap m={cap} "
        f"(last relative increment {tail:.3g}). Increase n_modes or verify "
        "mode convergence before trusting this result."
    )

def _require_mode_convergence(stats: 'Dict', mode_tol: 'float') -> 'None':
    """Fail before publishing a field whose azimuthal series is unconverged."""

    message = _mode_cap_warning(stats, mode_tol)
    if message:
        raise RuntimeError(
            message
            + " No RCS/amplitude result is returned for an unconverged "
              "modal truncation."
        )


def _segments_intersect_2d(
    a: 'np.ndarray',
    b: 'np.ndarray',
    c: 'np.ndarray',
    d: 'np.ndarray',
    tol: 'float',
) -> 'bool':
    """Inclusive segment intersection with a length-scaled cross tolerance."""

    def cross(p, q, r) -> 'float':
        return float((q[0] - p[0]) * (r[1] - p[1])
                     - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p, q, r, cross_tol) -> 'bool':
        return (
            abs(cross(p, q, r)) <= cross_tol
            and min(p[0], r[0]) - tol <= q[0] <= max(p[0], r[0]) + tol
            and min(p[1], r[1]) - tol <= q[1] <= max(p[1], r[1]) + tol
        )

    length_scale = max(
        float(np.linalg.norm(b - a)),
        float(np.linalg.norm(d - c)),
        tol,
    )
    cross_tol = tol * length_scale
    o1 = cross(a, b, c)
    o2 = cross(a, b, d)
    o3 = cross(c, d, a)
    o4 = cross(c, d, b)
    if (
        ((o1 > cross_tol and o2 < -cross_tol)
         or (o1 < -cross_tol and o2 > cross_tol))
        and ((o3 > cross_tol and o4 < -cross_tol)
             or (o3 < -cross_tol and o4 > cross_tol))
    ):
        return True
    return (
        on_segment(a, c, b, cross_tol)
        or on_segment(a, d, b, cross_tol)
        or on_segment(c, a, d, cross_tol)
        or on_segment(c, b, d, cross_tol)
    )


def _validate_solve_bor_generatrix(points, formulation: 'str') -> 'np.ndarray':
    """Validate the direct single-surface ``solve_bor`` geometry.

    EFIE may represent an open shell.  CFIE/MFIE require the supported
    closed-body topology: both ends on the rotation axis and traversal from
    the +z pole to the -z pole.  Every path rejects intersections because a
    self-crossing meridian generates an overlapping/non-manifold surface.
    """

    raw = np.asarray(points)
    if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.shape[0] < 2:
        raise ValueError(
            "BoR generatrix must be a finite (N, 2) array of (rho, z) "
            "nodes with N >= 2."
        )
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("BoR generatrix coordinates must be real.")
    try:
        pts = np.asarray(np.real(raw), dtype=float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("BoR generatrix coordinates must be real numbers.") from exc
    if not np.all(np.isfinite(pts)):
        raise ValueError("BoR generatrix coordinates must all be finite.")

    diag = max(
        float(np.ptp(pts[:, 0])),
        float(np.ptp(pts[:, 1])),
        1.0e-15,
    )
    geom_tol = max(1.0e-14, 1.0e-10 * diag)
    axis_tol = 1.0e-12 * max(1.0, float(np.max(np.abs(pts[:, 0]))))
    if np.any(pts[:, 0] < -axis_tol):
        bad = float(np.min(pts[:, 0]))
        raise ValueError(
            f"BoR generatrix rho coordinates must be >= 0; found {bad:.6g}."
        )
    pts[np.abs(pts[:, 0]) <= axis_tol, 0] = 0.0

    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if np.any(lengths <= geom_tol):
        idx = int(np.argmin(lengths))
        raise ValueError(
            f"BoR generatrix element {idx} has zero or near-zero length "
            f"({lengths[idx]:.3g})."
        )
    if float(np.max(pts[:, 0])) <= axis_tol:
        raise ValueError(
            "BoR generatrix lies entirely on the rotation axis and sweeps "
            "zero surface area."
        )

    # Any meeting of nonadjacent segments is a self-crossing, pinch, or
    # overlap.  A bounding-sphere tree keeps this check near O(N log N) for
    # production meshes instead of testing every pair.  Adjacent pairs share
    # their common endpoint by construction.
    n_elem = len(pts) - 1
    seg_a = pts[:-1]
    seg_b = pts[1:]
    mids = 0.5 * (seg_a + seg_b)
    half = 0.5 * lengths
    max_half = float(np.max(half))
    tree = cKDTree(mids)
    for i in range(n_elem):
        radius = float(half[i]) + max_half + geom_tol
        for j in tree.query_ball_point(mids[i], radius):
            j = int(j)
            if j <= i + 1:
                continue
            lower = (
                float(np.linalg.norm(mids[i] - mids[j]))
                - float(half[i]) - float(half[j])
            )
            if lower > geom_tol:
                continue
            if _segments_intersect_2d(
                pts[i], pts[i + 1], pts[j], pts[j + 1], geom_tol
            ):
                raise ValueError(
                    "BoR generatrix self-intersection/overlap between "
                    f"elements {i} and {j} is not supported."
                )

    start_on_axis = pts[0, 0] == 0.0
    end_on_axis = pts[-1, 0] == 0.0
    closed_body = start_on_axis and end_on_axis
    if formulation in {"cfie", "mfie"} and not closed_body:
        raise ValueError(
            f"{formulation.upper()} requires a closed BoR whose two "
            "generatrix endpoints lie on the rotation axis; use EFIE for "
            "an intentionally open shell."
        )
    if closed_body and not (pts[0, 1] > pts[-1, 1] + geom_tol):
        raise ValueError(
            "A closed BoR generatrix must be traversed from its +z axis "
            "endpoint to its -z axis endpoint so the left-of-travel normal "
            "faces the exterior."
        )
    return pts


def estimate_bor_table_gb(n_elems: 'int', m_max: 'int', formulation: 'str' = "cfie",
                          has_ibc: 'bool' = False, gauss_order: 'int' = 4,
                          single_tables: 'bool' = False) -> 'float':
    """Persistent far-table memory (GB) -- the scale bound of the current
    all-modes-at-once FFT assembly (see the phase-7 streaming notes)."""

    P = float(gauss_order * n_elems)
    per = 8.0 if single_tables else 16.0
    total = P * P * (m_max + 2) * per
    if formulation in ("cfie", "mfie"):
        total += 4.0 * P * P * (2 * m_max + 1) * per
    if has_ibc:
        total += 4.0 * P * P * (2 * m_max + 1) * per
    return total / 1e9


def estimate_bor_cross_table_gb(
    n_test_elements: 'int', n_source_elements: 'int', m_max: 'int',
    test_gauss_order: 'int' = 4, source_gauss_order: 'int' = 4,
    single_tables: 'bool' = False,
) -> 'float':
    """Persistent far tables for one directed rectangular T/P mapping."""

    test_points = float(int(test_gauss_order) * int(n_test_elements))
    source_points = float(int(source_gauss_order) * int(n_source_elements))
    modes = int(m_max)
    if test_points <= 0.0 or source_points <= 0.0 or modes < 0:
        raise ValueError("Cross-table estimate dimensions are invalid.")
    item_bytes = 8.0 if single_tables else 16.0
    modal_values = (modes + 2) + 4.0 * (2 * modes + 1)
    return test_points * source_points * modal_values * item_bytes / 1.0e9


def estimate_bor_operator_storage_gb(
    m_max: 'int',
    solver_requirements,
    cross_operators=(),
    constraint_dofs: 'int' = 0,
    streaming: 'bool' = False,
) -> 'float':
    """Estimate retained operator auxiliaries and their build workspace.

    ``solver_requirements`` contains ``(solver, efie, mfie, ibc)`` tuples.
    Repeated solver instances are merged so their dense basis matrices and
    tables are counted once.  Cross-surface operators are counted independently.
    ``constraint_dofs`` adds the dense junction projection matrices retained by
    the partial/multiregion paths.  With ``streaming=True``, far tables and
    their FFT workspace are excluded because the combined streaming planner
    accounts for those blocks and sampling tiles separately; exact same- and
    cross-surface near caches plus junction projections remain included.
    """

    try:
        modes = int(m_max)
        constraint_size = int(constraint_dofs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "BoR operator estimates require integer mode and constraint sizes."
        ) from exc
    if modes != m_max or modes < 0:
        raise ValueError("BoR operator mode cap must be a non-negative integer.")
    if constraint_size != constraint_dofs or constraint_size < 0:
        raise ValueError("BoR constraint DOFs must be a non-negative integer.")

    merged = {}
    for requirement in solver_requirements:
        if len(requirement) != 4:
            raise ValueError(
                "Each BoR solver requirement must be (solver, efie, mfie, ibc)."
            )
        solver, efie, mfie, ibc = requirement
        key = id(solver)
        if key in merged:
            previous = merged[key]
            merged[key] = (
                solver,
                previous[1] or bool(efie),
                previous[2] or bool(mfie),
                previous[3] or bool(ibc),
            )
        else:
            merged[key] = (solver, bool(efie), bool(mfie), bool(ibc))

    retained = 0.0
    build_workspace = 0.0
    signed_modes = 2 * modes + 1
    index_bytes = np.dtype(np.intp).itemsize
    for solver, efie, mfie, ibc in merged.values():
        point_count = int(solver.P)
        node_count = int(solver.Nn)
        elem_count = int(solver.gen.n_elems)
        table_bytes = np.dtype(solver._table_dtype).itemsize
        if not streaming and (efie or mfie or ibc):
            # The non-streaming contractions realize two real Nn-by-P basis
            # matrices before any far table is built.
            retained += 2.0 * node_count * point_count * np.dtype(float).itemsize
        if not streaming and efie:
            retained += point_count ** 2 * (modes + 2) * table_bytes
        if not streaming and mfie:
            retained += 4.0 * point_count ** 2 * signed_modes * table_bytes
        if not streaming and ibc:
            retained += 4.0 * point_count ** 2 * signed_modes * table_bytes

        pair_count = int(solver._near_pair_count)
        enabled_kinds = int(efie) + int(mfie) + int(ibc)
        # Each prepared kind retains four block families, every signed mode,
        # and four nodal entries per directed near pair, plus three index maps.
        retained += (
            enabled_kinds
            * 4.0
            * signed_modes
            * (4 * pair_count)
            * _COMPLEX128_BYTES
        )
        retained += enabled_kinds * 3.0 * (4 * pair_count) * index_bytes

        if enabled_kinds:
            # One serial point tile is integrated and released immediately.
            # Includes angular refinement scratch, contraction intermediates,
            # and the maximum default meridian rule's coordinate arrays.
            build_workspace = max(build_workspace, 128.0e6)

        if not streaming and (efie or mfie or ibc):
            rho_max = float(np.max(solver.gen.nodes[:, 0]))
            far_gap = float(solver._far_gap())
            if efie:
                n_xi = n_xi_for_pairs(
                    solver.k, rho_max, modes, far_gap, bracket=False
                )
                build_workspace = max(
                    build_workspace,
                    min(
                        float(FFT_BUILD_BUDGET),
                        point_count ** 2 * n_xi * 4.0 * _COMPLEX128_BYTES,
                    ),
                )
            if mfie or ibc:
                n_xi = n_xi_for_pairs(
                    solver.k, rho_max, modes, far_gap, bracket=True
                )
                bracket_arrays = 18.0 if ibc else 16.0
                build_workspace = max(
                    build_workspace,
                    min(
                        float(FFT_BUILD_BUDGET),
                        point_count ** 2
                        * n_xi
                        * bracket_arrays
                        * _COMPLEX128_BYTES,
                    ),
                )

    seen_crosses = set()
    for cross in cross_operators:
        if id(cross) in seen_crosses:
            continue
        seen_crosses.add(id(cross))
        point_count_p = int(cross.sp.P)
        point_count_q = int(cross.sq.P)
        pair_points = point_count_p * point_count_q
        cross_table_bytes = np.dtype(cross.sp._table_dtype).itemsize
        if not streaming:
            retained += pair_points * (modes + 2) * cross_table_bytes
            retained += 4.0 * pair_points * signed_modes * cross_table_bytes

        retained += len(cross.near_pairs) * 2 * 4 * signed_modes * 4 * _COMPLEX128_BYTES
        if cross.near_pairs:
            build_workspace = max(build_workspace, 128.0e6,
                64.0e6 + 768 * int(cross.near_max_order)**2)

        if not streaming:
            rho_max = max(
                float(np.max(cross.sp.gen.nodes[:, 0])),
                float(np.max(cross.sq.gen.nodes[:, 0])),
            )
            n_xi_g = n_xi_for_pairs(
                cross.k, rho_max, modes, float(cross._far_gap), bracket=False
            )
            n_xi_b = n_xi_for_pairs(
                cross.k, rho_max, modes, float(cross._far_gap), bracket=True
            )
            build_workspace = max(
                build_workspace,
                min(
                    float(FFT_BUILD_BUDGET),
                    pair_points * n_xi_g * 4.0 * _COMPLEX128_BYTES,
                ),
                min(
                    float(FFT_BUILD_BUDGET),
                    pair_points * n_xi_b * 18.0 * _COMPLEX128_BYTES,
                ),
            )

    if constraint_size:
        # build_Q distinguishes m=0, m=-1, m=+1, and all |m|>=2 modes.
        # Both signed first-order transforms are retained because the regular
        # axis relation contains j*m and therefore changes sign.
        category_count = 1 if modes == 0 else (3 if modes == 1 else 4)
        retained += (
            category_count
            * constraint_size
            * constraint_size
            * _COMPLEX128_BYTES
        )

    # Account for container metadata, alignment, and the small real-valued
    # geometry arrays.  Build scratch is bounded inside the kernel routines
    # and occurs on top of the tables retained earlier in preparation.
    return (1.10 * retained + build_workspace) / 1.0e9


def _plan_multisurface_assembly(
    m_max: 'int',
    solver_requirements,
    cross_operators,
    constraint_dofs: 'int',
    assembly: 'str',
    table_precision: 'str',
    stream_budget_gb: 'float',
    workers: 'int',
    extra_retained_gb: 'float' = 0.0,
) -> 'Dict[str, Any]':
    """Plan table or generalized streaming assembly for a junction system.

    The streaming budget applies to retained far blocks.  Always-resident
    same-surface contractions, cross-surface near/junction caches, cached
    projection matrices, and caller-owned dense maps are added explicitly to
    the runtime peak before the memory gate is evaluated.
    """

    from bor_streaming import (
        BOR_STREAM_TILE_BUDGET_GB,
        estimate_rectangular_streaming_gb,
        plan_combined_streaming_mode_block,
    )

    requirements = tuple(solver_requirements)
    crosses = tuple(dict.fromkeys(cross_operators))
    asm = str(assembly).strip().lower()
    precision = str(table_precision).strip().lower()
    budget = float(stream_budget_gb)
    extra = float(extra_retained_gb)
    if asm not in {"auto", "tables", "streaming"}:
        raise ValueError("assembly must be 'auto', 'tables', or 'streaming'.")
    if precision not in {"auto", "single", "double"}:
        raise ValueError(
            "table_precision must be 'auto', 'single', or 'double'."
        )
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")
    if not math.isfinite(extra) or extra < 0.0:
        raise ValueError("Extra retained operator storage must be non-negative.")

    merged = {}
    for solver, efie, mfie, ibc in requirements:
        key = id(solver)
        previous = merged.get(key)
        if previous is None:
            merged[key] = [solver, bool(efie), bool(mfie), bool(ibc)]
        else:
            previous[1] = previous[1] or bool(efie)
            previous[2] = previous[2] or bool(mfie)
            previous[3] = previous[3] or bool(ibc)
    merged_requirements = tuple(tuple(value) for value in merged.values())

    table_double_peak = estimate_bor_operator_storage_gb(
        m_max,
        merged_requirements,
        crosses,
        constraint_dofs=constraint_dofs,
        streaming=False,
    ) + 1.10 * extra
    use_streaming = (
        asm == "streaming" or (asm == "auto" and table_double_peak > 2.0)
    )
    worker_count = max(1, int(workers))

    if use_streaming:
        stream_specs_double = tuple(
            (
                int(solver.gen.n_elems),
                int(solver.gen.n_elems),
                bool(mfie or ibc),
                False,
            )
            for solver, efie, mfie, ibc in merged_requirements
            if efie or mfie or ibc
        ) + tuple(
            (
                int(cross.sp.gen.n_elems),
                int(cross.sq.gen.n_elems),
                True,
                False,
            )
            for cross in crosses
        )
        full_far_double = sum(
            estimate_rectangular_streaming_gb(
                nt, ns, int(m_max), rotated, False
            )
            for nt, ns, rotated, _single in stream_specs_double
        )
        use_single = (
            precision == "single"
            or (precision == "auto" and full_far_double > 4.0)
        )
        stream_specs = tuple(
            (nt, ns, rotated, use_single)
            for nt, ns, rotated, _single in stream_specs_double
        )
        mode_block, held_far_gb, effective_workers = (
            plan_combined_streaming_mode_block(
                int(m_max), stream_specs, budget, worker_count
            )
        )
        auxiliary_peak_gb = estimate_bor_operator_storage_gb(
            m_max,
            merged_requirements,
            crosses,
            constraint_dofs=constraint_dofs,
            streaming=True,
        ) + 1.10 * extra
        full_far_gb = full_far_double / (2.0 if use_single else 1.0)
        return {
            "use_streaming": True,
            "use_single": use_single,
            "mode_block": mode_block,
            "workers": effective_workers,
            "tile_budget_gb": BOR_STREAM_TILE_BUDGET_GB,
            "persistent_gb": full_far_gb + auxiliary_peak_gb,
            "held_far_gb": held_far_gb,
            "auxiliary_peak_gb": auxiliary_peak_gb,
            "assembly_peak_gb": (
                held_far_gb + auxiliary_peak_gb + BOR_STREAM_TILE_BUDGET_GB
            ),
        }

    use_single = precision == "single" or (
        precision == "auto" and table_double_peak > 4.0
    )
    for solver, _efie, _mfie, _ibc in merged_requirements:
        solver._table_dtype = np.complex64 if use_single else np.complex128
    table_peak_gb = estimate_bor_operator_storage_gb(
        m_max,
        merged_requirements,
        crosses,
        constraint_dofs=constraint_dofs,
        streaming=False,
    ) + 1.10 * extra
    return {
        "use_streaming": False,
        "use_single": use_single,
        "mode_block": None,
        "workers": worker_count,
        "tile_budget_gb": None,
        "persistent_gb": table_peak_gb,
        "held_far_gb": table_peak_gb,
        "auxiliary_peak_gb": 0.0,
        "assembly_peak_gb": table_peak_gb,
    }


def _validated_bor_aspects(thetas_deg) -> 'np.ndarray':
    """Validate the direct-solver monostatic aspect grid."""

    thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
    if thetas.ndim != 1 or thetas.size == 0:
        raise ValueError("BoR aspect grid must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(thetas)):
        raise ValueError("BoR aspect angles must all be finite.")
    if np.any(thetas < 0.0) or np.any(thetas > 180.0):
        raise ValueError("BoR aspect angles must lie in [0, 180] degrees.")
    return thetas


def _bor_mode_limits(k, rho_max: 'float', thetas,
                     n_modes: 'Optional[int]') -> 'Tuple[int, int]':
    """Return ``(mode_cap, physical_tail_start)`` for a BoR sweep.

    A plane wave incident at polar aspect ``theta`` has useful azimuthal
    content through approximately ``k*rho*sin(theta)``.  The historical
    ``+12`` cap margin is retained, while the adaptive tail test is now held
    off until that physical bandwidth has actually been accumulated.  The
    small 0.05 floor preserves the established axial-look safety margin.
    """

    theta_array = np.atleast_1d(np.asarray(thetas, dtype=float))
    sin_max = float(np.max(np.abs(np.sin(np.radians(theta_array)))))
    bandwidth = int(math.ceil(
        abs(complex(k)) * float(rho_max) * max(sin_max, 0.05)
    ))
    if n_modes is None:
        mode_cap = bandwidth + 12
    else:
        try:
            mode_cap = int(n_modes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("n_modes must be a non-negative integer or None.") from exc
        if mode_cap != n_modes or mode_cap < 0:
            raise ValueError("n_modes must be a non-negative integer or None.")
    return mode_cap, bandwidth


def _block_diagonal_transforms(*blocks: 'np.ndarray') -> 'np.ndarray':
    """Dense block diagonal used by the small number of BoR field families."""

    rows = sum(int(block.shape[0]) for block in blocks)
    columns = sum(int(block.shape[1]) for block in blocks)
    out = np.zeros((rows, columns), dtype=np.complex128)
    row = column = 0
    for block in blocks:
        nr, nc = block.shape
        out[row:row + nr, column:column + nc] = block
        row += nr
        column += nc
    return out


def _apply_regular_axis_rows(Q: 'np.ndarray', columns: 'np.ndarray',
                             offset: 'int', solver: 'BorPecSolver',
                             m: 'int') -> 'None':
    """Insert the exact |m|=1 pole relation into a composite transform."""

    if abs(int(m)) != 1:
        return
    for end, element in ((0, 0), (solver.Nn - 1, solver.gen.n_elems - 1)):
        if not solver.gen.node_on_axis(end):
            continue
        column = int(columns[offset + end])
        if column < 0:
            continue
        radial_sign = 1.0 if solver.gen.trho[element] >= 0.0 else -1.0
        Q[offset + solver.Nn + end, column] = (
            1j * int(m) * radial_sign
        )


@profiled_solve
def solve_bor(points, freq_hz: 'float', thetas_deg, formulation: 'str' = "efie",
              cfie_alpha: 'float' = 0.5, zs=None, n_modes: 'Optional[int]' = None,
              gauss_order: 'int' = 4, mode_tol: 'float' = 1e-6, workers: 'int' = 1,
              progress: 'Optional[Callable]' = None,
              check_abort: 'Optional[Callable]' = None,
              table_precision: 'str' = "auto", assembly: 'str' = "auto",
              stream_budget_gb: 'float' = 8.0, sheet_zs=None) -> 'Dict':
    """
    Monostatic RCS of a closed BoR at aspect angles thetas_deg (from +z).

    formulation: 'efie' (open shells / small bodies), 'cfie' (closed PEC --
    interior-resonance free), 'mfie' (diagnostics only).
    zs: surface impedance -- None (PEC), a complex scalar, or a per-ELEMENT
    complex array (tapered IBC).  IBC uses the EFIE form E_tan = Z_s J;
    lossy Z_s also damps interior resonances.  Nonzero Z_s is implemented
    by EFIE, or by CFIE for a uniform impedance on a closed surface.
    sheet_zs: a transmitting electric sheet impedance, scalar or per element.
    It uses EFIE + Zs mass with no equivalent magnetic current. Zs=0 elements
    can join a sheet to a PEC surface in the same meridian; opaque IBC cannot
    be combined with sheet_zs. Disconnected meridians require a separate solver.
    Returns dict with sigma_vv, sigma_hh (m^2) per angle.
    """

    form = str(formulation).strip().lower()
    if form not in {"efie", "cfie", "mfie"}:
        raise ValueError(
            f"Unsupported BoR formulation '{formulation}'. "
            "Use 'efie', 'cfie', or 'mfie'."
        )
    if form == "cfie":
        cfie_alpha = float(cfie_alpha)
        if not np.isfinite(cfie_alpha) or not (0.0 < cfie_alpha < 1.0):
            raise ValueError("CFIE alpha must be finite and satisfy 0 < alpha < 1.")
    points = _validate_solve_bor_generatrix(points, form)
    solver = BorPecSolver(points, freq_hz, gauss_order=gauss_order)
    k = solver.k
    thetas = _validated_bor_aspects(thetas_deg)
    rho_max = float(np.max(solver.gen.nodes[:, 0]))
    m_max, mode_tail_start = _bor_mode_limits(
        k, rho_max, thetas, n_modes
    )

    # Per-Gauss-point / per-element surface impedance.
    zs_pt = None
    zs_elem = None
    if zs is not None:
        zs_arr = np.asarray(zs, dtype=complex)
        if zs_arr.ndim == 0:
            zs_elem = np.full(solver.gen.n_elems, complex(zs_arr))
        else:
            if len(zs_arr) != solver.gen.n_elems:
                raise ValueError("zs array must have one entry per generatrix element.")
            zs_elem = zs_arr.astype(complex)
        zs_elem = _validate_bor_surface_impedance(
            zs_elem, "BoR surface impedance"
        )
        zs_pt = zs_elem[solver.g.elem]
        if np.all(np.abs(zs_pt) == 0.0):
            zs_pt = None
        elif form != "efie" and not (form == "cfie" and np.all(zs_elem == zs_elem[0])):
            raise ValueError(
                f"{form.upper()} with nonzero surface impedance is not "
                "implemented for this layout; CFIE requires uniform Zs on a closed surface."
            )

    sheet_mass = None
    sheet_weights = None
    if sheet_zs is not None:
        if zs is not None or form != "efie":
            raise ValueError("Free sheets require EFIE and cannot be combined with an opaque IBC.")
        sheet_values = np.asarray(sheet_zs, complex)
        if sheet_values.ndim == 0:
            sheet_values = np.full(solver.gen.n_elems, sheet_values, complex)
        if sheet_values.shape != (solver.gen.n_elems,):
            raise ValueError("sheet_zs must have one value per meridian element.")
        sheet_values = _validate_bor_surface_impedance(sheet_values, "BoR transmitting sheet")
        sheet_weights = sheet_values[solver.g.elem]

    alpha = float(cfie_alpha) if form == "cfie" else (1.0 if form == "efie" else 0.0)
    n_dofs = 2 * solver.Nn

    # -- far-assembly strategy and memory budget --
    from bor_streaming import (
        BOR_STREAM_TILE_BUDGET_GB,
        estimate_streaming_gb,
        plan_streaming_mode_block,
        sampling_backend_name,
    )
    tp = str(table_precision).strip().lower()
    if tp not in ("auto", "single", "double"):
        raise ValueError("table_precision must be 'auto', 'single', or 'double'.")
    asm = str(assembly).strip().lower()
    if asm not in ("auto", "tables", "streaming"):
        raise ValueError("assembly must be 'auto', 'tables', or 'streaming'.")
    est_double = estimate_bor_table_gb(solver.gen.n_elems, m_max, form,
                                       zs_pt is not None, gauss_order, False)
    # auto: switch to the phase-7b streaming path early -- the TABLE builders
    # sample [P, P, n_xi] in one shot, so their construction PEAK is several
    # times the stored table size (a 7 GB table can thrash a 32 GB machine
    # while building); streamed nodal blocks are 16x smaller and build in
    # bounded tiles.
    use_streaming = asm == "streaming" or (asm == "auto" and est_double > 2.0)
    if use_streaming:
        est_full = estimate_streaming_gb(solver.gen.n_elems, m_max, form,
                                         zs_pt is not None, False)
    else:
        est_full = est_double
    use_single = tp == "single" or (tp == "auto" and est_full > 4.0)
    if use_single and not use_streaming:
        solver._table_dtype = np.complex64
    est = est_full / (2.0 if use_single else 1.0)
    # phase-7d mode-block re-sweeps: when even the streamed blocks exceed
    # the budget, hold only an aligned range of modes and re-run the
    # (native, threaded) sampling sweep as the mode loop advances.
    solve_workers = max(1, int(workers))
    mode_block = None
    est_held = est
    if use_streaming:
        mode_block, est_held, solve_workers = plan_streaming_mode_block(
            solver.gen.n_elems,
            m_max,
            form,
            zs_pt is not None,
            use_single,
            stream_budget_gb,
            workers,
        )
    # Non-streaming FFT table builders temporarily retain sampling and
    # contraction arrays in addition to the published tables.  The scheduler
    # reserves 3.5x persistent storage for that construction phase, so the
    # runtime gate must use the same peak before ``prepare`` can allocate.
    # Streamed blocks and detailed multi-surface estimates already include
    # bounded build workspace and must not receive this multiplier again.
    assembly_peak_gb = (
        est_held + BOR_STREAM_TILE_BUDGET_GB + estimate_bor_operator_storage_gb(
            m_max, ((solver, alpha > 0.0, alpha < 1.0,
                     zs_pt is not None and alpha > 0.0),), streaming=True)
        if use_streaming
        else BOR_TABLE_BUILD_PEAK_FACTOR * est_held
    )
    table_note = (f"{'Streamed far blocks' if use_streaming else 'Far kernel tables'} "
                  f"stored in single precision ({est:.1f} GB; double would "
                  f"need {est_full:.1f} GB)."
                  if use_single and tp == "auto" else None)

    def prepare(mm):
        nonlocal sheet_mass
        if sheet_weights is not None:
            # _mode_sweep checks the complete reservation before prepare.
            sheet_mass = solver.mass_blocks(weight=sheet_weights)
        if use_streaming and solver._stream is None:
            solver.enable_streaming(
                mm, efie=alpha > 0.0, mfie=alpha < 1.0,
                ibc_zs_pt=zs_pt if (zs_pt is not None and alpha > 0.0) else None,
                single_blocks=use_single, workers=solve_workers,
                mode_block=mode_block,
                tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB)
        solver.prepare_operators(mm, efie=alpha > 0.0, mfie=alpha < 1.0,
                                 ibc=zs_pt is not None and alpha > 0.0,
                                 workers=solve_workers)

    def assemble(m):
        Z = None
        dual_ibc = None
        if alpha > 0.0:
            Z = solver.assemble_mode(m, m_max)
            if zs_pt is not None and alpha < 1.0:
                # Magnetic trace equation after M=-Zs(n x J), by Maxwell
                # duality: MFIE - (Zs/eta^2) R L R. Here L=-E_s(J),
                # R maps (t,phi) to (-phi,t). Uniform Zs commutes with the
                # source operator; a nodal projection of tapered Zs would not.
                N = solver.Nn
                dual_ibc = np.block([[Z[N:, N:], -Z[N:, :N]],
                                     [-Z[:N, N:], Z[:N, :N]]])
                dual_ibc *= complex(zs_elem[0]) / ETA0**2
            if sheet_mass is not None:
                Z[:solver.Nn, :solver.Nn] += sheet_mass
                Z[solver.Nn:, solver.Nn:] += sheet_mass
            if alpha != 1.0:
                Z *= alpha
            if zs_pt is not None:
                extra = solver.assemble_ibc_extra(m, m_max, zs_pt, zs_elem)
                if alpha != 1.0:
                    extra *= alpha
                Z += extra
        if alpha < 1.0:
            mfie = solver.assemble_mfie_mode(m, m_max)
            if dual_ibc is not None:
                mfie += dual_ibc
            mfie *= (1.0 - alpha) * ETA0
            if Z is None:
                Z = mfie
            else:
                Z += mfie
        if abs(int(m)) == 1:
            Q = solver.basis_transform(m)
            return _reduce_constrained_operator(Z, Q), Q
        # Outside |m|=1 there is no coupled pole relation, so retain the
        # O(N^2) Boolean slice instead of paying for dense O(N^3) products.
        mask = solver.basis_mask(m)
        return Z[np.ix_(mask, mask)], mask

    def rhs(m, th, pol):
        V = np.zeros(n_dofs, dtype=np.complex128)
        if alpha > 0.0:
            V += alpha * solver.rhs_mode(m, th, pol)
        if alpha < 1.0:
            V += (1.0 - alpha) * ETA0 * solver.rhs_mfie_mode(m, th, pol)
        return V

    def farfield(m, full, th, pol):
        fth, fph = solver.farfield_mode(m, full, th, zs_pt=zs_pt)
        return fth if pol == "VV" else fph

    def rhs_batch(m, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        return solver.rhs_vv_hh_batch(
            m,
            batch_thetas,
            efie_scale=alpha,
            mfie_scale=(1.0 - alpha) * ETA0,
        )

    def farfield_batch(m, full, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        return solver.farfield_vv_hh_batch(
            m, full, batch_thetas, zs_pt=zs_pt
        )

    # Interior-resonance guard: a closed body on the plain EFIE with a
    # LOSSLESS (purely reactive) surface impedance has nothing damping the
    # cavity resonances (lossy Z_s damps them; closed PEC bodies use CFIE).
    # A single-frequency condition-number test is not a reliable detector: at
    # practical mesh the resonant mode can rise only ~2x over background.
    # Because an IBC-compatible resonance-free CFIE is not implemented, reject
    # this configuration instead of returning a potentially corrupted field.
    solve_warnings: 'List[str]' = []
    if table_note:
        solve_warnings.append(table_note)
    closed = solver.gen.node_on_axis(0) and solver.gen.node_on_axis(solver.Nn - 1)
    lossless_ibc = (
        zs_pt is not None
        and _effectively_reactive_surface_impedance(zs_pt)
    )
    if form == "efie" and closed and lossless_ibc:
        raise RuntimeError(
            "Closed-body lossless/reactive IBC on the EFIE is unsupported: "
            "undamped interior resonances cannot be ruled out reliably, and "
            "an IBC-compatible resonance-free CFIE is not implemented. Add "
            "physical resistance, use an open surface, or use a validated "
            "full-wave formulation.")

    if sheet_weights is not None:
        assembly_peak_gb += 16.0*solver.Nn**2/1e9
    F, modes_used, stats = _mode_sweep(n_dofs, thetas, ("VV", "HH"), m_max,
                                       mode_tol, assemble, rhs, farfield,
                                       prepare=prepare, workers=solve_workers,
                                       progress=progress,
                                       check_abort=check_abort,
                                       monitor_cond=True,
                                       rhs_batch=rhs_batch,
                                       farfield_batch=farfield_batch,
                                       min_mode_before_tail=mode_tail_start,
                                       assembly_peak_gb=assembly_peak_gb,
                                       memory_context="The PEC/IBC BoR solve",
                                       signed_mode_symmetry=True)
    _require_mode_convergence(stats, mode_tol)
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(n_dofs),
        "formulation": form,
        "boundary_model": "transmitting_electric_sheet" if sheet_zs is not None else "opaque_ibc" if zs_pt is not None else "pec",
        "assembly": "streaming" if use_streaming else "tables",
        "table_precision": "single" if use_single else "double",
        "stream_mode_block": (solver._stream.mode_block
                              if solver._stream is not None else None),
        "stream_sweeps": (solver._stream.n_sweeps
                           if solver._stream is not None else 0),
        "stream_sampling_backend": (
            sampling_backend_name() if use_streaming else None
        ),
        "warnings": solve_warnings,
        **stats,
    }


# -----------------------------------------------------------------------------
# Phase 3: PMCHWT (homogeneous dielectric body)
# -----------------------------------------------------------------------------

@profiled_solve
def solve_bor_dielectric(points, freq_hz: 'float', thetas_deg, eps_r: 'complex',
                         mu_r: 'complex' = 1.0, n_modes: 'Optional[int]' = None,
                         gauss_order: 'int' = 4, mode_tol: 'float' = 1e-6,
                         workers: 'int' = 1, progress: 'Optional[Callable]' = None,
                         check_abort: 'Optional[Callable]' = None,
                         table_precision: 'str' = "auto",
                         assembly: 'str' = "auto",
                         stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """
    Monostatic RCS of a closed homogeneous penetrable BoR via per-mode PMCHWT.

    Unknowns per mode: J (2Nn) and M' = M/eta0 (2Nn).  Combining the
    exterior representation's interior null-field limit with the interior
    representation's exterior limit cancels every identity/jump term and
    leaves (T = EFIE operator in its medium, P = rotated-PV operator):

        [ T_e + T_i              -eta0 (P_e + P_i)      ] [J ]   [  <W, E_inc>     ]
        [ eta0 (P_e + P_i)    T_e + (eta0^2/eta_i^2) T_i ] [M'] = [ eta0 <W, H_inc> ]

    (the eta0 scalings symmetrize the block magnitudes).  The exterior far
    field radiates BOTH J and M in air.
    """

    points = _validate_solve_bor_generatrix(points, "cfie")
    # Validate direct-API constitutive inputs before any operator allocation.
    _causal_medium(eps_r, mu_r)
    se = BorPecSolver(points, freq_hz, gauss_order=gauss_order)
    si = BorPecSolver(points, freq_hz, gauss_order=gauss_order,
                      medium=(eps_r, mu_r))
    k = se.k
    thetas = _validated_bor_aspects(thetas_deg)
    rho_max = float(np.max(se.gen.nodes[:, 0]))
    m_max, mode_tail_start = _bor_mode_limits(
        k, rho_max, thetas, n_modes
    )
    Nn = se.Nn
    eta_ratio2 = (ETA0 / si.eta) ** 2

    # The homogeneous PMCHWT system has two same-surface medium operators.
    # Both admit the bounded nodal far-block stream used by the conductor
    # path; unlike coated/multiregion systems, there are no rectangular
    # cross-surface operators to retain.
    from bor_streaming import (
        BOR_STREAM_TILE_BUDGET_GB,
        estimate_streaming_gb,
        plan_streaming_mode_block,
    )
    tp = str(table_precision).strip().lower()
    if tp not in ("auto", "single", "double"):
        raise ValueError("table_precision must be 'auto', 'single', or 'double'.")
    asm = str(assembly).strip().lower()
    if asm not in ("auto", "tables", "streaming"):
        raise ValueError("assembly must be 'auto', 'tables', or 'streaming'.")
    stream_budget = float(stream_budget_gb)
    if not math.isfinite(stream_budget) or stream_budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")

    table_double = 2.0 * estimate_bor_table_gb(
        se.gen.n_elems, m_max, "efie", True, gauss_order, False
    )
    use_streaming = (
        asm == "streaming" or (asm == "auto" and table_double > 2.0)
    )
    full_double = (
        2.0 * estimate_streaming_gb(
            se.gen.n_elems, m_max, "efie", True, False
        )
        if use_streaming else table_double
    )
    use_single = tp == "single" or (tp == "auto" and full_double > 4.0)
    solve_workers = max(1, int(workers))
    mode_block = None
    if use_streaming:
        # The user budget is the combined retained far-block budget for both
        # medium sides.  Identical topology means one half-budget plan applies
        # exactly to each stream, and both range boundaries stay aligned.
        mode_block, held_one, solve_workers = plan_streaming_mode_block(
            se.gen.n_elems,
            m_max,
            "efie",
            True,
            use_single,
            0.5 * stream_budget,
            solve_workers,
        )
        operator_storage_gb = (
            2.0 * held_one + BOR_STREAM_TILE_BUDGET_GB
            + estimate_bor_operator_storage_gb(m_max,
                ((se, True, False, True), (si, True, False, True)), streaming=True)
        )
    else:
        if use_single:
            se._table_dtype = np.complex64
            si._table_dtype = np.complex64
        operator_storage_gb = estimate_bor_operator_storage_gb(
            m_max,
            (
                (se, True, False, True),
                (si, True, False, True),
            ),
        )

    solve_warnings: 'List[str]' = []
    if use_single and tp == "auto":
        solve_warnings.append(
            f"{'Streamed far blocks' if use_streaming else 'Far kernel tables'} "
            f"stored in single precision ({full_double / 2.0:.1f} GB; "
            f"double would need {full_double:.1f} GB)."
        )

    def prepare(mm):
        if use_streaming:
            for solver in (se, si):
                solver.enable_streaming(
                    mm,
                    efie=True,
                    pmchwt=True,
                    single_blocks=use_single,
                    tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                    workers=solve_workers,
                    mode_block=mode_block,
                )
        se.prepare_operators(mm, efie=True, ibc=True, workers=solve_workers)
        si.prepare_operators(mm, efie=True, ibc=True, workers=solve_workers)

    def assemble(m):
        T_e = se.assemble_mode(m, m_max)
        T_i = si.assemble_mode(m, m_max)
        P_sum = ETA0 * (se.assemble_pmchwt_P(m, m_max) + si.assemble_pmchwt_P(m, m_max))
        A = np.empty((4 * Nn, 4 * Nn), dtype=np.complex128)
        A[: 2 * Nn, : 2 * Nn] = T_e + T_i
        A[: 2 * Nn, 2 * Nn:] = -P_sum
        A[2 * Nn:, : 2 * Nn] = P_sum
        A[2 * Nn:, 2 * Nn:] = T_e + eta_ratio2 * T_i
        if abs(int(m)) == 1:
            q_surface = se.basis_transform(m)
            Q = _block_diagonal_transforms(q_surface, q_surface)
            return _reduce_constrained_operator(A, Q), Q
        mask = np.tile(se.basis_mask(m), 2)
        return A[np.ix_(mask, mask)], mask

    def rhs(m, th, pol):
        return np.concatenate([se.rhs_mode(m, th, pol),
                               ETA0 * se.rhs_h_mode(m, th, pol)])

    def farfield(m, full, th, pol):
        fth, fph = se.farfield_mode(m, full[: 2 * Nn], th,
                                    msol=ETA0 * full[2 * Nn:])
        return fth if pol == "VV" else fph

    def rhs_batch(m, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        electric = se.rhs_vv_hh_batch(m, batch_thetas)
        out = np.empty((4 * Nn, electric.shape[1]), dtype=np.complex128)
        out[:2 * Nn] = electric
        out[2 * Nn:, 0::2] = -electric[:, 1::2]
        out[2 * Nn:, 1::2] = electric[:, 0::2]
        return out

    def farfield_batch(m, full, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        return se.farfield_vv_hh_batch(
            m,
            full[:2 * Nn],
            batch_thetas,
            msolutions=ETA0 * full[2 * Nn:],
        )

    F, modes_used, stats = _mode_sweep(4 * Nn, thetas, ("VV", "HH"), m_max,
                                       mode_tol, assemble, rhs, farfield,
                                       prepare=prepare, workers=solve_workers,
                                       progress=progress,
                                       check_abort=check_abort,
                                       monitor_cond=True,
                                       rhs_batch=rhs_batch,
                                       farfield_batch=farfield_batch,
                                       min_mode_before_tail=mode_tail_start,
                                       assembly_peak_gb=operator_storage_gb,
                                       memory_context="The dielectric BoR solve",
                                       signed_mode_symmetry=True)
    _require_mode_convergence(stats, mode_tol)
    stream_backends = None
    stream_backend = None
    if use_streaming:
        stream_backends = {
            "exterior": "native_c" if se._stream._native is not None else "numpy",
            "interior": "native_c" if si._stream._native is not None else "numpy",
        }
        unique_backends = set(stream_backends.values())
        stream_backend = (
            next(iter(unique_backends)) if len(unique_backends) == 1 else "mixed"
        )
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(4 * Nn),
        "formulation": "pmchwt",
        "eps_r": complex(eps_r),
        "mu_r": complex(mu_r),
        "assembly": "streaming" if use_streaming else "tables",
        "table_precision": "single" if use_single else "double",
        "stream_mode_block": mode_block if use_streaming else None,
        "stream_sweeps": (
            se._stream.n_sweeps + si._stream.n_sweeps
            if use_streaming else 0
        ),
        "stream_sampling_backend": stream_backend,
        "stream_sampling_backends": stream_backends,
        "warnings": solve_warnings,
        **stats,
    }


# -----------------------------------------------------------------------------
# Phase 3: cross-surface operators + coated-PEC multi-region solver
# -----------------------------------------------------------------------------

def _segment_distance(p0, p1, q0, q1) -> 'float':
    """Min distance between two non-intersecting 2D segments (attained at an
    endpoint of one of them)."""

    def pt_seg(c, a, b):
        ab = b - a
        t = float(np.dot(c - a, ab) / max(np.dot(ab, ab), 1e-300))
        t = min(1.0, max(0.0, t))
        return float(np.hypot(*(c - (a + t * ab))))

    return min(pt_seg(p0, q0, q1), pt_seg(p1, q0, q1),
               pt_seg(q0, p0, p1), pt_seg(q1, p0, p1))


def _contract_near_points(gp, e, gq, f, k, m_max, kinds, points):
    """Integrate immediately into 2x2 blocks; never retain raw pair kernels.

    Point tiles and modal projection tiles bound scratch even for a single
    large quadrature record. Output shape per kind is [4, 2m+1, 2, 2].
    EFIE omits j*k*eta*2pi; bracket blocks include 2pi.
    """
    s, t, weight = points
    active = weight > 0
    s, t, weight = s[active], t[active], weight[active]
    modes = np.arange(-m_max, m_max + 1)
    nm = len(modes)
    out = {kind: np.zeros((4, nm, 2, 2), complex) for kind in kinds}
    chunk = max(1, min(256, int(8.0e6 / (16 * nm * 20))))
    contract = lambda left, kernel, right: np.einsum(
        'ip,pm,jp->mij', left, kernel, right, optimize=False)
    for i in range(0, len(s), chunk):
        sl = slice(i, i + chunk)
        rp, zp, trp, tzp, p0, p1, dp0, dp1, lp = _points_on_element(gp, e, s[sl])
        rq, zq, trq, tzq, q0, q1, dq0, dq1, lq = _points_on_element(gq, f, t[sl])
        Tp, Tq = np.array([p0, p1]), np.array([q0, q1])
        Dp, Dq = np.array([dp0, dp1]), np.array([dq0, dq1])
        w = weight[sl] * lp * lq
        rr = (rp * rq * w)[:, None]
        if 'efie' in out:
            G = modal_kernels_near(rp, zp, rq, zq, k, m_max)
            am = np.abs(modes)
            Gn = G[:, am]
            Gc = 0.5 * (G[:, np.abs(am - 1)] + G[:, am + 1])
            Gs = (G[:, np.abs(am - 1)] - G[:, am + 1]) / 2j
            Gs[:, modes < 0] *= -1
            scalar = w[:, None] * Gn / k**2
            out['efie'][0] += contract(Tp, rr * (trp * trq * Gc + tzp * tzq * Gn), Tq) - contract(Dp, scalar, Dq)
            out['efie'][1] += contract(Tp, rr * trp * Gs, Tq) - (1j * modes[:, None, None]) * contract(Dp, scalar, Tq)
            out['efie'][2] += contract(Tp, -rr * trq * Gs, Tq) + (1j * modes[:, None, None]) * contract(Tp, scalar, Dq)
            out['efie'][3] += contract(Tp, rr * Gc, Tq) - (modes[:, None, None]**2) * contract(Tp, scalar, Tq)
            del G, Gn, Gc, Gs, scalar
        for kind in kinds:
            if kind == 'efie':
                continue
            kernel = mfie_kernels_near if kind == 'mfie' else ibc_kernels_near
            values = kernel(rp, zp, trp, tzp, rq, zq, trq, tzq, k, m_max)
            for uv, value in enumerate(values):
                out[kind][uv] += contract(Tp, 2 * np.pi * rr * value, Tq)
            del values
    return out


def _gap_graded_points(gp, e, gq, f, order):
    """Sinh grading around the closest source point for each test point."""
    x, wg = cached_leggauss(order)
    u, w = (x + 1) / 2, wg / 2
    p0, p1 = gp.nodes[e:e + 2]
    q0, q1 = gq.nodes[f:f + 2]
    dp, dq = p1 - p0, q1 - q0
    # Split the outer integral at projections of source endpoints. This
    # resolves changes in the closest-point map on unequal element meshes.
    cuts = np.unique(np.r_[0., 1., np.clip(
        [(q0 - p0) @ dp / (dp @ dp), (q1 - p0) @ dp / (dp @ dp)], 0, 1)])
    # The inner integral also develops endpoint layers in the test variable.
    # Grade both ends of each outer interval to resolve those layers without
    # raising a uniform tensor order over the whole element pair.
    gap = _segment_distance(p0, p1, q0, q1)
    outer_delta = max(gap / np.linalg.norm(dp), 1e-15)
    ss, ww = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        vmax = np.arcsinh((b - a) / (2 * outer_delta))
        v = vmax * u
        distance = outer_delta * np.sinh(v)
        weights = w * vmax * outer_delta * np.cosh(v)
        ss.extend((a + distance, b - distance))
        ww.extend((weights, weights))
    s, ws = np.concatenate(ss), np.concatenate(ww)
    p = p0 + s[:, None] * dp
    center = np.clip((p - q0) @ dq / (dq @ dq), 0, 1)
    distance = np.linalg.norm(p - (q0 + center[:, None] * dq), axis=1)
    delta = np.maximum(distance / np.linalg.norm(dq), 1e-15)
    ts, weights = [], []
    for side, extent in ((-1, center), (1, 1 - center)):
        vmax = np.arcsinh(extent / delta)
        v = vmax[:, None] * u
        ts.append(center[:, None] + side * delta[:, None] * np.sinh(v))
        weights.append(ws[:, None] * w * vmax[:, None] * delta[:, None] * np.cosh(v))
    t = np.concatenate(ts, axis=1)
    return np.broadcast_to(s[:, None], t.shape).ravel(), t.ravel(), np.concatenate(weights, axis=1).ravel()


def _converged_disjoint_blocks(gp, e, gq, f, k, m_max, kinds,
                               order=12, rtol=2e-5, max_order=192):
    """Converge actual EFIE/PV blocks independently of angular integration."""
    requested_order = int(order)
    if requested_order < 2 or not 2 * requested_order <= max_order <= 384 or not 0 < rtol < 1:
        raise ValueError('BoR near quadrature needs order >= 2, 2*order <= max_order <= 384, and 0 < rtol < 1.')
    if tuple(kinds) == ('mfie',):
        nodes = np.vstack((gp.nodes[e:e+2], gq.nodes[f:f+2]))
        # On one flat annulus R and J are coplanar, hence curl(G J) is
        # normal and its tangential MFIE PV trace is exactly zero. Geometry
        # subdivision can leave a few ulps in z; relative convergence of that
        # roundoff (~1e-20 blocks) is meaningless. Use the exact geometric
        # identity, without loosening tolerances for any non-planar pair.
        z_tolerance = 8*np.finfo(float).eps*max(float(np.max(np.abs(nodes))), np.finfo(float).tiny)
        if float(np.ptp(nodes[:, 1])) <= z_tolerance:
            return {'mfie': np.zeros((4, 2*m_max+1, 2, 2), complex)}, int(order), 0.
    gap = _segment_distance(gp.nodes[e], gp.nodes[e + 1], gq.nodes[f], gq.nodes[f + 1])
    graded = gap < 0.25 * max(gp.lengths[e], gq.lengths[f])
    def evaluate(n):
        points = (_gap_graded_points(gp, e, gq, f, n) if graded
                  else _regular_cell_points(n))
        return _contract_near_points(gp, e, gq, f, k, m_max, kinds, points)
    # Smooth disjoint pairs usually converge at the requested order. Compare
    # a cheaper preliminary rule first; never publish below requested_order.
    n = requested_order if graded else max(2, requested_order // 2)
    coarse = evaluate(n)
    while n < max_order:
        n = max(requested_order, min(2 * n, max_order))
        fine = evaluate(n)
        errors = []
        for kind in kinds:
            # Check every vector block, with a floor for symmetry-zero blocks.
            scale = np.max(np.abs(fine[kind]), axis=(1, 2, 3))
            floor = max(float(np.max(scale)) * 1e-8, 1e-280)
            error = np.max(np.abs(fine[kind] - coarse[kind]), axis=(1, 2, 3))
            errors.append(float(np.max(error / np.maximum(scale, floor))))
        error = max(errors)
        if math.isfinite(error) and error <= rtol:
            return fine, n, error
        coarse = fine
    raise ValueError(f'BoR near meridian quadrature did not converge for elements ({e}, {f}); gap={gap:.6g}, order={n}, relative block change={error:.3g}. Refine the mesh or increase near_max_order.')


def _near_quadrature_summary(*operators):
    return {
        'scheme': 'bounded_compact_blocks_with_angular_and_disjoint_meridian_refinement',
        'disjoint_meridian_order_max': max(
            (getattr(op, 'near_quadrature_order_max', 0) for op in operators), default=0),
        'disjoint_meridian_relative_change_max': max(
            (getattr(op, 'near_quadrature_error_max', 0.) for op in operators), default=0.),
        'self_and_junction_rule': 'graded_singular_cells',
    }


class BorCrossOperators:
    """T (EFIE) and P (rotated-PV) Galerkin blocks between two DIFFERENT
    generatrices in one homogeneous medium: test bases on solver sp, source
    bases on solver sq (both BorPecSolver instances with the same medium).

    Pairs closer than near_factor * max(element lengths) are re-integrated
    with gap-graded meridian quadrature and independent block refinement.
    Only compact mode blocks are retained. The surfaces may TOUCH at shared
    endpoints (coating-termination junctions): element pairs sharing such a
    point are log-singular in the Galerkin sense and are routed to the same
    graded corner-cell quadrature the same-surface assembly uses for
    adjacent elements.  Overlapping/crossing interiors remain an error."""

    def __init__(self, sp: 'BorPecSolver', sq: 'BorPecSolver',
                 near_factor: 'float' = 2.0, near_order: 'int' = 12,
                 near_rtol: 'float' = 2e-5, near_max_order: 'int' = 192):
        if not np.isclose(complex(sp.k), complex(sq.k)):
            raise ValueError("Cross operators need both solvers in the same medium.")
        self.sp, self.sq = sp, sq
        self.k, self.eta = sp.k, sp.eta
        self.near_order = int(near_order)
        self.near_rtol = float(near_rtol)
        self.near_max_order = int(near_max_order)
        if self.near_order < 2 or not 2 * self.near_order <= self.near_max_order <= 384 or not 0 < self.near_rtol < 1:
            raise ValueError('BoR near quadrature needs order >= 2, 2*order <= max_order <= 384, and 0 < rtol < 1.')
        self.near_quadrature_order_max = 0
        self.near_quadrature_error_max = 0.0
        # element-pair classification by segment distance / shared endpoints
        gp, gq = sp.gen, sq.gen
        diag = max(float(np.max(gp.nodes)) - float(np.min(gp.nodes)),
                   float(np.max(gq.nodes)) - float(np.min(gq.nodes)), 1e-9)
        touch_tol = 1e-9 * diag
        self.near_pairs = []
        self.pair_kind: 'Dict[Tuple[int, int], Optional[str]]' = {}
        self._far_gap = math.inf
        for e in range(gp.n_elems):
            p_ends = (gp.nodes[gp.elem_n0[e]], gp.nodes[gp.elem_n1[e]])
            for f in range(gq.n_elems):
                q_ends = (gq.nodes[gq.elem_n0[f]], gq.nodes[gq.elem_n1[f]])
                shared = [(a, b) for a in (0, 1) for b in (0, 1)
                          if float(np.hypot(*(p_ends[a] - q_ends[b]))) <= touch_tol]
                d = _segment_distance(p_ends[0], p_ends[1], q_ends[0], q_ends[1])
                if len(shared) > 1:
                    raise ValueError("Cross-operator surfaces share a whole "
                                     "element (overlapping geometry).")
                if shared:
                    a, b = shared[0]
                    self.near_pairs.append((e, f))
                    self.pair_kind[(e, f)] = f"corner{a}{b}"
                elif d <= touch_tol:
                    raise ValueError("Cross-operator surfaces cross or touch "
                                     "away from a shared junction endpoint.")
                elif d < near_factor * max(gp.lengths[e], gq.lengths[f]):
                    self.near_pairs.append((e, f))
                    self.pair_kind[(e, f)] = None
                else:
                    self._far_gap = min(self._far_gap, d)
        if not math.isfinite(self._far_gap):
            self._far_gap = 0.0
        self.near_set = set(self.near_pairs)
        self._G = None
        self._B = None
        self._stream = None
        self._cache: 'Dict' = {}

    def enable_streaming(self, m_max: 'int', single_blocks: 'bool' = False,
                         tile_budget_gb: 'float' = 1.0,
                         workers: 'int' = 1,
                         mode_block: 'Optional[int]' = None) -> 'None':
        """Use bounded rectangular nodal far blocks for this mapping."""

        from bor_streaming import StreamingCrossFarBlocks
        self._stream = StreamingCrossFarBlocks(
            self,
            m_max,
            dtype=np.complex64 if single_blocks else np.complex128,
            tile_budget_gb=tile_budget_gb,
            workers=workers,
            mode_block=mode_block,
        )

    def _tables(self, m_max: 'int'):
        if self._G is not None and self._G.shape[-1] >= m_max + 2:
            return self._G, self._B
        gp, gq = self.sp.g, self.sq.g
        Pp, Pq = len(gp.rho), len(gq.rho)
        rho_scale = max(float(np.max(gp.rho)), float(np.max(gq.rho)))
        n_xi_g = n_xi_for_pairs(self.k, rho_scale, m_max, self._far_gap,
                                bracket=False)
        n_xi_b = n_xi_for_pairs(self.k, rho_scale, m_max, self._far_gap,
                                bracket=True)
        G = modal_kernels_fft(gp.rho[:, None], gp.z[:, None],
                              gq.rho[None, :], gq.z[None, :],
                              self.k, m_max, n_xi=n_xi_g)
        B = ibc_kernels_fft(gp.rho, gp.z, gp.trho, gp.tz,
                            gq.rho, gq.z, gq.trho, gq.tz, self.k, m_max,
                            n_xi=n_xi_b)
        if self.near_pairs:
            near_mask = np.zeros((Pp, Pq), dtype=bool)
            for (e, f) in self.near_pairs:
                near_mask[np.ix_(gp.elem == e, gq.elem == f)] = True
            G[near_mask, :] = 0.0
            B = list(B)
            for i in range(4):
                B[i][near_mask, :] = 0.0
            B = tuple(B)
        table_dtype = self.sp._table_dtype
        self._G = G.astype(table_dtype, copy=False)
        self._B = tuple(value.astype(table_dtype, copy=False) for value in B)
        return self._G, self._B

    def _near_data(self, e, f, m_max):
        """Cache only converged 2x2 EFIE and rotated-PV mode blocks."""
        cache = self._cache.setdefault(m_max, {})
        key = (e, f)
        if key not in cache:
            kind = self.pair_kind.get(key)
            if kind is not None:
                blocks = _contract_near_points(self.sp.gen, e, self.sq.gen, f,
                    self.k, m_max, ('efie', 'ibc'), _cell_points(kind))
            else:
                blocks, order, error = _converged_disjoint_blocks(
                    self.sp.gen, e, self.sq.gen, f, self.k, m_max,
                    ('efie', 'ibc'), self.near_order, self.near_rtol, self.near_max_order)
                self.near_quadrature_order_max = max(self.near_quadrature_order_max, order)
                self.near_quadrature_error_max = max(self.near_quadrature_error_max, error)
            cache[key] = blocks
        return cache[key]

    def assemble_T(self, m: 'int', m_max: 'int') -> 'np.ndarray':
        """Cross EFIE operator [2Np, 2Nq] (same normalization as
        BorPecSolver.assemble_mode, C = j k eta 2pi of this medium)."""

        k = self.k
        gp, gq = self.sp.g, self.sq.g
        if self._stream is not None:
            ztt, ztf, zft, zff = self._stream.efie_blocks(m)
        else:
            G, _ = self._tables(m_max)
            Gm, Gc, Gs = kernels_for_mode(G, m)
            ztt, ztf, zft, zff = _pair_blocks(
                m, k,
                gp.rho, gp.trho, gp.tz, self.sp.B_T, self.sp.B_D, gp.w,
                gq.rho, gq.trho, gq.tz, self.sq.B_T, self.sq.B_D, gq.w,
                Gm, Gc, Gs,
            )
        for e, f in self.near_pairs:
            blocks = self._near_data(e, f, m_max)['efie'][:, m + m_max]
            rc = np.ix_([e, e + 1], [f, f + 1])
            for target, value in zip((ztt, ztf, zft, zff), blocks):
                target[rc] += value

        C = 1j * k * self.eta * 2.0 * np.pi
        Np, Nq = self.sp.Nn, self.sq.Nn
        Z = np.empty((2 * Np, 2 * Nq), dtype=np.complex128)
        Z[:Np, :Nq] = C * ztt
        Z[:Np, Nq:] = C * ztf
        Z[Np:, :Nq] = C * zft
        Z[Np:, Nq:] = C * zff
        return Z

    def assemble_P(self, m: 'int', m_max: 'int') -> 'np.ndarray':
        """Cross rotated-PV operator [2Np, 2Nq] (see assemble_pmchwt_P)."""

        gp, gq = self.sp.g, self.sq.g
        if self._stream is not None:
            blocks = list(self._stream.bracket_blocks(m))
        else:
            _, Bt = self._tables(m_max)
            wrho_p = gp.w * gp.rho
            wrho_q = gq.w * gq.rho
            blocks = []
            for uv in range(4):
                Km = mfie_for_mode(Bt[uv], m, m_max)
                blocks.append(2.0 * np.pi * (self.sp.B_T * wrho_p[None, :]) @ Km @ (self.sq.B_T * wrho_q[None, :]).T)
        for e, f in self.near_pairs:
            near = self._near_data(e, f, m_max)['ibc'][:, m + m_max]
            rc = np.ix_([e, e + 1], [f, f + 1])
            for target, value in zip(blocks, near):
                target[rc] += value
        Btt, Btf, Bft, Bff = blocks
        Np, Nq = self.sp.Nn, self.sq.Nn
        P = np.empty((2 * Np, 2 * Nq), dtype=np.complex128)
        P[:Np, :Nq] = -Btf
        P[:Np, Nq:] = Btt
        P[Np:, :Nq] = -Bff
        P[Np:, Nq:] = Bft
        return P

    def prepare(self, m_max: 'int') -> 'None':
        """Warm every table/near cache (see BorPecSolver.prepare_operators)."""
        if self._stream is None:
            self.sp._ensure_dense_point_matrices()
            self.sq._ensure_dense_point_matrices()
            self._tables(m_max)
        for e, f in self.near_pairs:
            self._near_data(e, f, m_max)


@profiled_solve
def solve_bor_coated_pec(points_outer, points_core, freq_hz: 'float', thetas_deg,
                         eps_r: 'complex', mu_r: 'complex' = 1.0,
                         n_modes: 'Optional[int]' = None, gauss_order: 'int' = 4,
                         mode_tol: 'float' = 1e-6, near_factor: 'float' = 2.0,
                         near_order: 'int' = 12, workers: 'int' = 1,
                         progress: 'Optional[Callable]' = None,
                         check_abort: 'Optional[Callable]' = None,
                         table_precision: 'str' = "auto",
                         assembly: 'str' = "auto",
                         stream_budget_gb: 'float' = 8.0,
                         near_rtol: 'float' = 2e-5,
                         near_max_order: 'int' = 192) -> 'Dict':
    """
    Monostatic RCS of a PEC core (generatrix points_core) fully covered by a
    homogeneous coating with outer surface points_outer (both closed, both
    traversed +z end to -z end so left-of-travel normals face their exterior).

    Unknowns per mode: J_o, M'_o = M_o/eta0 on the outer interface, J_c on
    the core.  PMCHWT rows on the outer interface pick up cross terms from
    J_c radiating in the layer; the core row is the EFIE in the layer:

      [ T_e+T_L          -eta0(P_e+P_L)          -T_L^oc      ] [J_o ]   [ V_E      ]
      [ eta0(P_e+P_L)    T_e+(eta0/eta_L)^2 T_L  -eta0 P_L^oc ] [M'_o] = [ eta0 V_H ]
      [ T_L^co           -eta0 P_L^co            -T_L^cc      ] [J_c ]   [ 0        ]

    Only (J_o, M_o) radiate in air.
    """

    points_outer = _validate_solve_bor_generatrix(points_outer, "cfie")
    points_core = _validate_solve_bor_generatrix(points_core, "cfie")
    _causal_medium(eps_r, mu_r)
    se = BorPecSolver(points_outer, freq_hz, gauss_order=gauss_order)
    sLo = BorPecSolver(points_outer, freq_hz, gauss_order=gauss_order,
                       medium=(eps_r, mu_r))
    sLc = BorPecSolver(points_core, freq_hz, gauss_order=gauss_order,
                       medium=(eps_r, mu_r))
    Xoc = BorCrossOperators(sLo, sLc, near_factor=near_factor, near_order=near_order,
                            near_rtol=near_rtol, near_max_order=near_max_order)
    Xco = BorCrossOperators(sLc, sLo, near_factor=near_factor, near_order=near_order,
                            near_rtol=near_rtol, near_max_order=near_max_order)
    k = se.k
    thetas = _validated_bor_aspects(thetas_deg)
    rho_max = float(np.max(se.gen.nodes[:, 0]))
    m_max, mode_tail_start = _bor_mode_limits(
        k, rho_max, thetas, n_modes
    )
    No, Nc = se.Nn, sLc.Nn
    eta_ratio2 = (ETA0 / sLo.eta) ** 2
    ntot = 4 * No + 2 * Nc

    from bor_streaming import (
        BOR_STREAM_TILE_BUDGET_GB,
        estimate_rectangular_streaming_gb,
        plan_combined_streaming_mode_block,
    )
    tp = str(table_precision).strip().lower()
    if tp not in ("auto", "single", "double"):
        raise ValueError("table_precision must be 'auto', 'single', or 'double'.")
    asm = str(assembly).strip().lower()
    if asm not in ("auto", "tables", "streaming"):
        raise ValueError("assembly must be 'auto', 'tables', or 'streaming'.")
    stream_budget = float(stream_budget_gb)
    if not math.isfinite(stream_budget) or stream_budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")

    ne_outer = se.gen.n_elems
    ne_core = sLc.gen.n_elems
    table_far_double = (
        2.0 * estimate_bor_table_gb(
            ne_outer, m_max, "efie", True, gauss_order, False
        )
        + estimate_bor_table_gb(
            ne_core, m_max, "efie", False, gauss_order, False
        )
        + estimate_bor_cross_table_gb(
            ne_outer, ne_core, m_max, gauss_order, gauss_order, False
        )
        + estimate_bor_cross_table_gb(
            ne_core, ne_outer, m_max, gauss_order, gauss_order, False
        )
    )
    use_streaming = (
        asm == "streaming"
        or (asm == "auto" and table_far_double > 2.0)
    )
    stream_specs_double = (
        (ne_outer, ne_outer, True, False),
        (ne_outer, ne_outer, True, False),
        (ne_core, ne_core, False, False),
        (ne_outer, ne_core, True, False),
        (ne_core, ne_outer, True, False),
    )
    full_stream_double = sum(
        estimate_rectangular_streaming_gb(
            nt, ns, m_max, rotated, single
        )
        for nt, ns, rotated, single in stream_specs_double
    )
    use_single = tp == "single" or (
        tp == "auto" and use_streaming and full_stream_double > 4.0
    )
    solve_workers = max(1, int(workers))
    mode_block = None
    if use_streaming:
        stream_specs = tuple(
            (nt, ns, rotated, use_single)
            for nt, ns, rotated, _single in stream_specs_double
        )
        mode_block, held_blocks_gb, solve_workers = (
            plan_combined_streaming_mode_block(
                m_max, stream_specs, stream_budget, solve_workers
            )
        )
        operator_storage_gb = (
            held_blocks_gb + BOR_STREAM_TILE_BUDGET_GB
        )
    else:
        if use_single:
            for solver in (se, sLo, sLc):
                solver._table_dtype = np.complex64
        operator_storage_gb = (
            estimate_bor_operator_storage_gb(
                m_max,
                (
                    (se, True, False, True),
                    (sLo, True, False, True),
                    (sLc, True, False, False),
                ),
                (Xoc, Xco),
            )
        )

    solve_warnings: 'List[str]' = []
    if use_single and tp == "auto":
        solve_warnings.append(
            "Streamed self/cross far blocks stored in single precision "
            f"({full_stream_double / 2.0:.1f} GB; double would need "
            f"{full_stream_double:.1f} GB)."
        )

    iJ = slice(0, 2 * No); iM = slice(2 * No, 4 * No); iC = slice(4 * No, ntot)

    def prepare(mm):
        if use_streaming:
            se.enable_streaming(
                mm, efie=True, pmchwt=True,
                single_blocks=use_single,
                tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                workers=solve_workers, mode_block=mode_block,
            )
            sLo.enable_streaming(
                mm, efie=True, pmchwt=True,
                single_blocks=use_single,
                tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                workers=solve_workers, mode_block=mode_block,
            )
            sLc.enable_streaming(
                mm, efie=True, single_blocks=use_single,
                tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                workers=solve_workers, mode_block=mode_block,
            )
            for cross in (Xoc, Xco):
                cross.enable_streaming(
                    mm, single_blocks=use_single,
                    tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                    workers=solve_workers, mode_block=mode_block,
                )
        se.prepare_operators(mm, efie=True, ibc=True, workers=solve_workers)
        sLo.prepare_operators(mm, efie=True, ibc=True, workers=solve_workers)
        sLc.prepare_operators(mm, efie=True, workers=solve_workers)
        Xoc.prepare(mm)
        Xco.prepare(mm)

    def assemble(m):
        T_e = se.assemble_mode(m, m_max)
        T_Lo = sLo.assemble_mode(m, m_max)
        P_sum = ETA0 * (se.assemble_pmchwt_P(m, m_max) + sLo.assemble_pmchwt_P(m, m_max))
        A = np.zeros((ntot, ntot), dtype=np.complex128)
        A[iJ, iJ] = T_e + T_Lo
        A[iJ, iM] = -P_sum
        A[iJ, iC] = -Xoc.assemble_T(m, m_max)
        A[iM, iJ] = P_sum
        A[iM, iM] = T_e + eta_ratio2 * T_Lo
        A[iM, iC] = -ETA0 * Xoc.assemble_P(m, m_max)
        A[iC, iJ] = Xco.assemble_T(m, m_max)
        A[iC, iM] = -ETA0 * Xco.assemble_P(m, m_max)
        A[iC, iC] = -sLc.assemble_mode(m, m_max)
        if abs(int(m)) == 1:
            q_outer = se.basis_transform(m)
            Q = _block_diagonal_transforms(
                q_outer, q_outer, sLc.basis_transform(m)
            )
            return _reduce_constrained_operator(A, Q), Q
        mask_o = se.basis_mask(m)
        mask = np.concatenate([mask_o, mask_o, sLc.basis_mask(m)])
        return A[np.ix_(mask, mask)], mask

    def rhs(m, th, pol):
        V = np.zeros(ntot, dtype=np.complex128)
        V[iJ] = se.rhs_mode(m, th, pol)
        V[iM] = ETA0 * se.rhs_h_mode(m, th, pol)
        return V

    def farfield(m, full, th, pol):
        fth, fph = se.farfield_mode(m, full[iJ], th, msol=ETA0 * full[iM])
        return fth if pol == "VV" else fph

    def rhs_batch(m, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        electric = se.rhs_vv_hh_batch(m, batch_thetas)
        out = np.zeros((ntot, electric.shape[1]), dtype=np.complex128)
        out[iJ] = electric
        out[iM, 0::2] = -electric[:, 1::2]
        out[iM, 1::2] = electric[:, 0::2]
        return out

    def farfield_batch(m, full, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        return se.farfield_vv_hh_batch(
            m,
            full[iJ],
            batch_thetas,
            msolutions=ETA0 * full[iM],
        )

    F, modes_used, stats = _mode_sweep(ntot, thetas, ("VV", "HH"), m_max,
                                       mode_tol, assemble, rhs, farfield,
                                       prepare=prepare, workers=solve_workers,
                                       progress=progress,
                                       check_abort=check_abort,
                                       monitor_cond=True,
                                       rhs_batch=rhs_batch,
                                       farfield_batch=farfield_batch,
                                       min_mode_before_tail=mode_tail_start,
                                       assembly_peak_gb=operator_storage_gb,
                                       memory_context="The coated-PEC BoR solve",
                                       signed_mode_symmetry=True)
    _require_mode_convergence(stats, mode_tol)
    stream_backends = None
    stream_backend = None
    if use_streaming:
        stream_objects = {
            "exterior_outer": se._stream,
            "coating_outer": sLo._stream,
            "coating_core": sLc._stream,
            "cross_outer_core": Xoc._stream,
            "cross_core_outer": Xco._stream,
        }
        stream_backends = {
            name: "native_c" if stream._native is not None else "numpy"
            for name, stream in stream_objects.items()
        }
        unique_backends = set(stream_backends.values())
        stream_backend = (
            next(iter(unique_backends)) if len(unique_backends) == 1
            else "mixed"
        )
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(ntot),
        "formulation": "pmchwt-coated",
        "near_quadrature": _near_quadrature_summary(se, sLo, sLc, Xoc, Xco),
        "eps_r": complex(eps_r),
        "mu_r": complex(mu_r),
        "assembly": "streaming" if use_streaming else "tables",
        "table_precision": "single" if use_single else "double",
        "stream_mode_block": mode_block if use_streaming else None,
        "stream_sweeps": (
            se._stream.n_sweeps + sLo._stream.n_sweeps
            + sLc._stream.n_sweeps + Xoc._stream.n_sweeps
            + Xco._stream.n_sweeps
            if use_streaming else 0
        ),
        "stream_sampling_backend": stream_backend,
        "stream_sampling_backends": stream_backends,
        "warnings": solve_warnings,
        **stats,
    }


# -----------------------------------------------------------------------------
# Partial coatings: coating terminating ON the PEC surface (junctions).
# -----------------------------------------------------------------------------

@profiled_solve
def solve_bor_partial_coating(points_interface, points_covered, bare_pieces,
                              freq_hz: 'float', thetas_deg, eps_r: 'complex',
                              mu_r: 'complex' = 1.0, bare_zs=None,
                              n_modes: 'Optional[int]' = None,
                              gauss_order: 'int' = 4, mode_tol: 'float' = 1e-6,
                              near_factor: 'float' = 2.0, near_order: 'int' = 12,
                              workers: 'int' = 1, progress: 'Optional[Callable]' = None,
                              check_abort: 'Optional[Callable]' = None,
                              table_precision: 'str' = "auto",
                              assembly: 'str' = "auto",
                              stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """
    Monostatic RCS of a PEC body PARTIALLY covered by a homogeneous coating:
    the dielectric interface S_d (points_interface) terminates on the PEC
    surface at junction circles where air, coating, and conductor meet.

    points_covered is the coated part of the core, bare_pieces the list of
    uncovered PEC generatrix pieces (0, 1, or 2 -- cap or band coatings).
    All pieces are drawn in the global +z -> -z traversal with left-of-travel
    normals facing away from the surface they bound (exterior/air for S_d
    and the bare pieces, into the coating for the covered core).

    Formulation: exterior region bounded by S_d + bare pieces (currents J_d,
    M_d, J_1p in air), layer region bounded by S_d + covered core (currents
    -J_d, -M_d, J_2 in the coating medium).  PMCHWT rows on S_d, EFIE rows
    on each PEC piece; all with the same phase-3 operator blocks.  Junction
    conditions (per junction circle A):

      * through-current continuity ties the chain-end coefficients:
        J_1p(A) = (+t, +phi) J_d(A)  (air chain runs S_d and the bare piece
        in the same global traversal), and
        J_2(A)  = (+t, -phi) J_d(A)  (the layer chain traverses S_d REVERSED:
        the -J_d current in the flipped frame has +t and -phi components);
      * M_t(A) = 0 (tangential E along the junction circle vanishes on the
        conductor), while M_phi(A) stays free with its natural half-triangle
        end basis (it carries the normal-E wedge behavior).

    The constraints enter Galerkin-style: A_red = Q^T A_full Q -- the tied
    row is the sum of the piece rows, exactly the classical BoR junction
    treatment (Putnam / Medgyesi-Mitschang).

    bare_zs (optional): per-piece Leontovich surface impedance -- a list
    matching bare_pieces of None / complex scalar / per-element complex
    arrays (tapers).  The eliminated magnetic current M_1 = -Z_s n_hat x J_1
    keeps the piece's own operator on the validated Gauss-point IBC path
    (assemble_ibc_extra) and radiates onto OTHER surfaces through the
    existing cross T/P operators via the nodal column map
    (M_1t, M_1phi) = (+Z_s J_1phi, -Z_s J_1t). At a junction adjoining an
    impedance piece, M_t(A) = 0 remains the convergent one-node junction
    approximation. A pointwise M_d,t(A) = Z_s(A) J_phi(A) tie was tested and
    rejected because the wedge-line field limit is singular; see build_Q.
    """

    points_interface = _validate_solve_bor_generatrix(
        points_interface, "efie"
    )
    points_covered = _validate_solve_bor_generatrix(points_covered, "efie")
    bare_pieces = [
        _validate_solve_bor_generatrix(piece, "efie")
        for piece in bare_pieces
    ]
    _causal_medium(eps_r, mu_r)
    sd_e = BorPecSolver(points_interface, freq_hz, gauss_order=gauss_order)
    sd_L = BorPecSolver(points_interface, freq_hz, gauss_order=gauss_order,
                        medium=(eps_r, mu_r))
    s2_L = BorPecSolver(points_covered, freq_hz, gauss_order=gauss_order,
                        medium=(eps_r, mu_r))
    bares = [BorPecSolver(p, freq_hz, gauss_order=gauss_order)
             for p in bare_pieces]

    # -- per-piece surface impedance (None = PEC) --
    if bare_zs is None:
        bare_zs = [None] * len(bares)
    if len(bare_zs) != len(bares):
        raise ValueError("bare_zs must have one entry per bare piece.")
    zs_elems: 'List[Optional[np.ndarray]]' = []
    zs_ptss: 'List[Optional[np.ndarray]]' = []
    S_maps: 'List[Optional[np.ndarray]]' = []
    for b, zs in zip(bares, bare_zs):
        ne = b.gen.n_elems
        zs_arr = None
        if zs is not None:
            za = np.asarray(zs, dtype=complex)
            zs_arr = np.full(ne, complex(za)) if za.ndim == 0 else za.astype(complex)
            if len(zs_arr) != ne:
                raise ValueError("Per-element bare_zs array length must match "
                                 "the piece's element count.")
            zs_arr = _validate_bor_surface_impedance(
                zs_arr, "BoR partial-coating bare surface impedance"
            )
            if not np.any(np.abs(zs_arr) > 0.0):
                zs_arr = None
        zs_elems.append(zs_arr)
        if zs_arr is None:
            zs_ptss.append(None)
            S_maps.append(None)
        else:
            zs_ptss.append(zs_arr[b.g.elem])
            zn = np.empty(b.Nn, dtype=complex)
            zn[0] = zs_arr[0]
            zn[-1] = zs_arr[-1]
            zn[1:-1] = 0.5 * (zs_arr[:-1] + zs_arr[1:])
            S = np.zeros((2 * b.Nn, 2 * b.Nn), dtype=complex)
            S[:b.Nn, b.Nn:] = np.diag(zn)        # M_1t   = +Z_s J_1phi
            S[b.Nn:, :b.Nn] = -np.diag(zn)       # M_1phi = -Z_s J_1t
            S_maps.append(S)

    nonzero_bare_zs = [
        array for array in zs_elems if array is not None
    ]
    if (
        nonzero_bare_zs
        and _effectively_reactive_surface_impedance(
            np.concatenate(nonzero_bare_zs)
        )
    ):
        raise RuntimeError(
            "Closed partial-coated body with lossless/reactive bare IBC on "
            "the EFIE is unsupported: undamped interior resonances cannot "
            "be ruled out reliably, and an IBC-compatible resonance-free "
            "CFIE is not implemented. Add physical resistance or use a "
            "validated full-wave formulation."
        )

    all_nodes = np.vstack([sd_e.gen.nodes, s2_L.gen.nodes] +
                          [b.gen.nodes for b in bares])
    diag = max(float(np.ptp(all_nodes[:, 0])) + float(np.ptp(all_nodes[:, 1])), 1e-9)
    jn_tol = 1e-8 * diag

    # -- junction detection: cluster non-axis chain endpoints --
    def endpoints(solver):
        gen = solver.gen
        out = []
        for node in (0, gen.n_nodes - 1):
            if gen.node_on_axis(node):
                out.append((node, None))
            else:
                out.append((node, gen.nodes[node]))
        return out

    junctions: 'List[Dict]' = []   # {pos, d_node, c_node, bare: (piece, node)}

    def register(kind_key, piece_idx, node, pos):
        for jn in junctions:
            if float(np.hypot(*(jn["pos"] - pos))) <= jn_tol:
                if kind_key in jn:
                    raise ValueError("Two same-role chain ends meet at one junction.")
                jn[kind_key] = (piece_idx, node) if kind_key == "bare" else node
                return
        junctions.append({"pos": pos,
                          kind_key: (piece_idx, node) if kind_key == "bare" else node})

    for node, pos in endpoints(sd_e):
        if pos is not None:
            register("d_node", None, node, pos)
    for node, pos in endpoints(s2_L):
        if pos is not None:
            register("c_node", None, node, pos)
    for bi, b in enumerate(bares):
        for node, pos in endpoints(b):
            if pos is not None:
                register("bare", bi, node, pos)
    for jn in junctions:
        if "d_node" not in jn or "c_node" not in jn or "bare" not in jn:
            raise ValueError(
                "Every off-axis chain endpoint must be a coating-termination "
                "junction where the interface, the covered core, and exactly "
                f"one bare piece meet; found an incomplete junction at "
                f"(rho, z) = ({jn['pos'][0]:.6g}, {jn['pos'][1]:.6g}).")
        bi, bn = jn["bare"]
        za = zs_elems[bi]
        jn["zs"] = complex(za[0 if bn == 0 else -1]) if za is not None else 0.0

    solve_warnings: 'List[str]' = []
    for jn in junctions:
        if abs(jn["zs"]) > 0.02 * ETA0:
            solve_warnings.append(
                f"Surface impedance is {abs(jn['zs']):.1f} ohm at the coating "
                f"junction (rho, z) = ({jn['pos'][0]:.4g}, {jn['pos'][1]:.4g}): "
                "an abrupt Z_s step AT a coating edge is an ill-defined "
                "sheet-model limit (E_phi is discontinuous along the junction "
                "line) and the solution does not mesh-converge there. One "
                "validation case showed a ~0.5 dB mesh/discretization plateau; "
                "that observation is not an accuracy bound. Taper Z_s toward "
                "zero at the junction "
                "(the physical edge treatment) for converged results.")

    # -- cross operators (touching allowed at the junctions) --
    xkw = dict(near_factor=near_factor, near_order=near_order)
    X_d2 = BorCrossOperators(sd_L, s2_L, **xkw)
    X_2d = BorCrossOperators(s2_L, sd_L, **xkw)
    X_d1 = [BorCrossOperators(sd_e, b, **xkw) for b in bares]
    X_1d = [BorCrossOperators(b, sd_e, **xkw) for b in bares]
    X_11 = {(i, j): BorCrossOperators(bares[i], bares[j], **xkw)
            for i in range(len(bares)) for j in range(len(bares)) if i != j}

    k = sd_e.k
    thetas = _validated_bor_aspects(thetas_deg)
    rho_max = max([float(np.max(sd_e.gen.nodes[:, 0])),
                   float(np.max(s2_L.gen.nodes[:, 0]))] +
                  [float(np.max(b.gen.nodes[:, 0])) for b in bares])
    m_max, mode_tail_start = _bor_mode_limits(
        k, rho_max, thetas, n_modes
    )
    eta_ratio2 = (ETA0 / sd_L.eta) ** 2

    Nd, N2 = sd_e.Nn, s2_L.Nn
    N1 = [b.Nn for b in bares]
    off_Jd, off_M, off_J2 = 0, 2 * Nd, 4 * Nd
    off_J1 = []
    acc = 4 * Nd + 2 * N2
    for n in N1:
        off_J1.append(acc)
        acc += 2 * n
    n_full = acc

    def prepare(mm):
        planned_workers = plan["workers"]
        if plan["use_streaming"] and sd_e._stream is None:
            common = dict(
                single_blocks=plan["use_single"],
                tile_budget_gb=plan["tile_budget_gb"],
                workers=planned_workers,
                mode_block=plan["mode_block"],
            )
            sd_e.enable_streaming(mm, efie=True, pmchwt=True, **common)
            sd_L.enable_streaming(mm, efie=True, pmchwt=True, **common)
            s2_L.enable_streaming(mm, efie=True, **common)
            for bi, bare in enumerate(bares):
                bare.enable_streaming(
                    mm,
                    efie=True,
                    ibc_zs_pt=zs_ptss[bi],
                    **common,
                )
            for cross in all_crosses:
                cross.enable_streaming(mm, **common)
        sd_e.prepare_operators(mm, efie=True, ibc=True, workers=planned_workers)
        sd_L.prepare_operators(mm, efie=True, ibc=True, workers=planned_workers)
        s2_L.prepare_operators(mm, efie=True, workers=planned_workers)
        for bi, b in enumerate(bares):
            b.prepare_operators(mm, efie=True, ibc=zs_elems[bi] is not None,
                                workers=planned_workers)
        for X in all_crosses:
            X.prepare(mm)
        # Only the m=0, |m|=1, and |m|>=2 basis categories are distinct.
        # Warm them serially so mode workers never allocate duplicate dense
        # projection matrices concurrently.
        for representative in range(min(int(mm), 2) + 1):
            build_Q(representative)
        if int(mm) >= 1:
            build_Q(-1)

    # -- junction-aware constraint matrix Q(m) --
    _Q_cache: 'Dict[int, np.ndarray]' = {}

    def build_Q(m):
        category = int(m) if abs(int(m)) == 1 else (0 if m == 0 else 2)
        Q = _Q_cache.get(category)
        if Q is not None:
            return Q
        d_jn_nodes = {jn["d_node"] for jn in junctions}
        c_jn_nodes = {jn["c_node"] for jn in junctions}
        b_jn_nodes = {(jn["bare"][0], jn["bare"][1]) for jn in junctions}
        # M_t(A) = 0 is kept at IBC junctions too.  Alternatives tried and
        # rejected against the mixed-impedance sphere cross-check: a pointwise
        # tie M_t(A) = Z_s(A) J_phi(A) fails at +7.6 dB (the wedge-line field
        # limits are singular; the Leontovich relation holds on the impedance
        # sheet, not on the interface's approach to the line), and a free
        # half-basis end DOF fails at 5.5 dB non-convergent (the lone M_t end
        # DOF has no jump partner controlling it).  The E_phi != 0 wall value
        # that M_t(A) = 0 suppresses is finite, so the error is a converging
        # one-node approximation like the corner treatment.

        def surf_mask(solver, jn_nodes, is_m):
            """(t_active, phi_active) with axis rules; junction nodes stay
            active here (masters/M_phi) unless excluded below."""
            Nn = solver.Nn
            t_act = np.ones(Nn, dtype=bool)
            f_act = np.ones(Nn, dtype=bool)
            for end in (0, Nn - 1):
                if solver.gen.node_on_axis(end):
                    t_act[end] = (abs(m) == 1)
                    f_act[end] = False
                elif end in jn_nodes:
                    if is_m:
                        t_act[end] = False   # M_t = 0 at the conductor line
                else:
                    t_act[end] = False       # true open edge (not expected)
        # NOTE: masks for slave nodes are applied by the tie logic below.
            return t_act, f_act

        # column ownership: -1 = masked, -2 = slave, >=0 reduced index
        col = np.full(n_full, -1, dtype=int)
        red = 0

        def assign(offset, acts):
            nonlocal red
            t_act, f_act = acts
            Nn = len(t_act)
            for i in range(Nn):
                if t_act[i]:
                    col[offset + i] = red; red += 1
            for i in range(Nn):
                if f_act[i]:
                    col[offset + Nn + i] = red; red += 1

        assign(off_Jd, surf_mask(sd_e, d_jn_nodes, False))
        assign(off_M, surf_mask(sd_e, d_jn_nodes, True))
        t2, f2 = surf_mask(s2_L, c_jn_nodes, False)
        for jn in junctions:
            t2[jn["c_node"]] = False; f2[jn["c_node"]] = False
        assign(off_J2, (t2, f2))
        b_acts = []
        for bi, b in enumerate(bares):
            tb, fb = surf_mask(b, {n for (p, n) in b_jn_nodes if p == bi}, False)
            for jn in junctions:
                if jn["bare"][0] == bi:
                    tb[jn["bare"][1]] = False; fb[jn["bare"][1]] = False
            b_acts.append((tb, fb))
            assign(off_J1[bi], (tb, fb))

        Q = np.zeros((n_full, red), dtype=np.complex128)
        active = col >= 0
        Q[np.flatnonzero(active), col[active]] = 1.0
        _apply_regular_axis_rows(Q, col, off_Jd, sd_e, m)
        _apply_regular_axis_rows(Q, col, off_M, sd_e, m)
        _apply_regular_axis_rows(Q, col, off_J2, s2_L, m)
        for bi, bare in enumerate(bares):
            _apply_regular_axis_rows(Q, col, off_J1[bi], bare, m)
        # Ties: slaves follow the interface's junction-end coefficients.
        # Earlier gate-only switches could silently disable one or both ties;
        # the validated full continuity relation is now the sole production
        # path.
        for jn in junctions:
            dn = jn["d_node"]
            cn = jn["c_node"]
            bi, bn = jn["bare"]
            master_t = col[off_Jd + dn]
            master_f = col[off_Jd + Nd + dn]
            if master_t >= 0:
                Q[off_J2 + cn, master_t] = 1.0
                Q[off_J1[bi] + bn, master_t] = 1.0
            if master_f >= 0:
                Q[off_J2 + N2 + cn, master_f] = -1.0
                Q[off_J1[bi] + N1[bi] + bn, master_f] = 1.0
        _Q_cache[category] = Q
        return Q

    def assemble(m):
        A = np.zeros((n_full, n_full), dtype=np.complex128)
        sl_Jd = slice(off_Jd, off_Jd + 2 * Nd)
        sl_M = slice(off_M, off_M + 2 * Nd)
        sl_J2 = slice(off_J2, off_J2 + 2 * N2)
        T_e = sd_e.assemble_mode(m, m_max)
        T_L = sd_L.assemble_mode(m, m_max)
        P_sum = ETA0 * (sd_e.assemble_pmchwt_P(m, m_max) + sd_L.assemble_pmchwt_P(m, m_max))
        iE = sl_Jd; iH = sl_M            # row blocks share the column layout
        A[iE, sl_Jd] = T_e + T_L
        A[iE, sl_M] = -P_sum
        A[iE, sl_J2] = -X_d2.assemble_T(m, m_max)
        A[iH, sl_Jd] = P_sum
        A[iH, sl_M] = T_e + eta_ratio2 * T_L
        A[iH, sl_J2] = -ETA0 * X_d2.assemble_P(m, m_max)
        # The covered-core row is -LayEq (NOT +LayEq as in the junction-free
        # coated solver): every row block must carry the same region-equation
        # orientation -- rows here are -(AirEq) on air-bounded surfaces and
        # -(AirEq - LayEq) on the interface -- or the Q^T junction fold sums
        # the layer-region equation with the wrong sign and the through-DOF
        # rows are inconsistent (caught by the eps=1 cap gate at +9 dB).
        A[sl_J2, sl_Jd] = -X_2d.assemble_T(m, m_max)
        A[sl_J2, sl_M] = ETA0 * X_2d.assemble_P(m, m_max)
        A[sl_J2, sl_J2] = s2_L.assemble_mode(m, m_max)
        for bi, b in enumerate(bares):
            sl_b = slice(off_J1[bi], off_J1[bi] + 2 * N1[bi])
            Sb = S_maps[bi]
            # eliminated M_1 = -Z_s n x J_1 radiates onto the other surfaces
            # through the SAME cross operators via the nodal column map Sb
            A[iE, sl_b] = X_d1[bi].assemble_T(m, m_max)
            A[iH, sl_b] = ETA0 * X_d1[bi].assemble_P(m, m_max)
            if Sb is not None:
                A[iE, sl_b] += -X_d1[bi].assemble_P(m, m_max) @ Sb
                A[iH, sl_b] += (1.0 / ETA0) * (X_d1[bi].assemble_T(m, m_max) @ Sb)
            A[sl_b, sl_Jd] = X_1d[bi].assemble_T(m, m_max)
            A[sl_b, sl_M] = -ETA0 * X_1d[bi].assemble_P(m, m_max)
            A[sl_b, sl_b] = b.assemble_mode(m, m_max)
            if zs_elems[bi] is not None:
                A[sl_b, sl_b] += b.assemble_ibc_extra(m, m_max, zs_ptss[bi],
                                                      zs_elems[bi])
            for bj in range(len(bares)):
                if bj != bi:
                    sl_bj = slice(off_J1[bj], off_J1[bj] + 2 * N1[bj])
                    A[sl_b, sl_bj] = X_11[(bi, bj)].assemble_T(m, m_max)
                    if S_maps[bj] is not None:
                        A[sl_b, sl_bj] += -X_11[(bi, bj)].assemble_P(m, m_max) @ S_maps[bj]
        Q = build_Q(m)
        return _reduce_constrained_operator(A, Q), None

    def rhs(m, th, pol):
        V = np.zeros(n_full, dtype=np.complex128)
        V[off_Jd:off_Jd + 2 * Nd] = sd_e.rhs_mode(m, th, pol)
        V[off_M:off_M + 2 * Nd] = ETA0 * sd_e.rhs_h_mode(m, th, pol)
        for bi, b in enumerate(bares):
            V[off_J1[bi]:off_J1[bi] + 2 * N1[bi]] = b.rhs_mode(m, th, pol)
        return build_Q(m).conj().T @ V

    def farfield(m, x_red, th, pol):
        x = build_Q(m) @ x_red
        fth, fph = sd_e.farfield_mode(m, x[off_Jd:off_Jd + 2 * Nd], th,
                                      msol=ETA0 * x[off_M:off_M + 2 * Nd])
        for bi, b in enumerate(bares):
            ft, fp = b.farfield_mode(m, x[off_J1[bi]:off_J1[bi] + 2 * N1[bi]],
                                     th, zs_pt=zs_ptss[bi])
            fth += ft; fph += fp
        return fth if pol == "VV" else fph

    solver_requirements = (
        (sd_e, True, False, True),
        (sd_L, True, False, True),
        (s2_L, True, False, False),
        *(
            (bare, True, False, zs_elems[index] is not None)
            for index, bare in enumerate(bares)
        ),
    )
    all_crosses = (
        X_d2, X_2d, *X_d1, *X_1d, *X_11.values()
    )
    impedance_map_gb = sum(
        matrix.nbytes for matrix in S_maps if matrix is not None
    ) / 1.0e9
    plan = _plan_multisurface_assembly(
        m_max,
        solver_requirements,
        all_crosses,
        constraint_dofs=n_full,
        assembly=assembly,
        table_precision=table_precision,
        stream_budget_gb=stream_budget_gb,
        workers=workers,
        extra_retained_gb=impedance_map_gb,
    )

    F, modes_used, stats = _mode_sweep(n_full, thetas, ("VV", "HH"), m_max,
                                       mode_tol, assemble, rhs, farfield,
                                       prepare=prepare, workers=plan["workers"],
                                       progress=progress,
                                       check_abort=check_abort,
                                       monitor_cond=True,
                                       min_mode_before_tail=mode_tail_start,
                                       assembly_peak_gb=plan["assembly_peak_gb"],
                                       memory_context="The partial-coating BoR solve")
    _require_mode_convergence(stats, mode_tol)
    streams = {
        "interface_exterior": sd_e._stream,
        "interface_coating": sd_L._stream,
        "covered_core": s2_L._stream,
        **{f"bare_{index}": bare._stream for index, bare in enumerate(bares)},
        **{
            f"cross_{index}": cross._stream
            for index, cross in enumerate(all_crosses)
        },
    }
    stream_backends = {
        name: "native_c" if stream is not None and stream._native is not None
        else "numpy"
        for name, stream in streams.items()
    } if plan["use_streaming"] else {}
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(n_full),
        "n_junctions": len(junctions),
        "formulation": "pmchwt-partial-coating",
        "eps_r": complex(eps_r),
        "mu_r": complex(mu_r),
        "assembly": "streaming" if plan["use_streaming"] else "tables",
        "table_precision": "single" if plan["use_single"] else "double",
        "stream_mode_block": plan["mode_block"],
        "stream_sweeps": (
            sum(stream.n_sweeps for stream in streams.values())
            if plan["use_streaming"] else 0
        ),
        "stream_sampling_backend": (
            next(iter(set(stream_backends.values())))
            if plan["use_streaming"]
            and len(set(stream_backends.values())) == 1
            else ("mixed" if plan["use_streaming"] else None)
        ),
        "stream_sampling_backends": stream_backends,
        "stream_auxiliary_peak_gb": plan["auxiliary_peak_gb"],
        "warnings": solve_warnings,
        **stats,
    }


# -----------------------------------------------------------------------------
# Generic multi-region assembly (phase 6: multi-layer coatings).
#
# The phase-5 lesson mechanized: every block and every junction tie follows
# from the REGION PRESCRIPTION.  Each region r contributes, for surfaces
# s, s' on its boundary with orientation weights sigma_rs (+1 if r is the
# surface's reference/exterior side, -1 otherwise):
#
#   A[J_s,  J_s' ] += sigma_rs sigma_rs' T_r^{ss'}
#   A[J_s,  M'_s'] += -eta0 sigma_rs sigma_rs' P_r^{ss'}
#   A[M'_s, J_s' ] += eta0 sigma_rs sigma_rs' P_r^{ss'}
#   A[M'_s, M'_s'] += (eta0/eta_r)^2 sigma_rs sigma_rs' T_r^{ss'}
#
# (M' = M/eta0; conductors carry no M rows/cols).  PMCHWT emerges on every
# interface, the phase-3/5 systems are special cases (including the phase-5
# row-orientation fix, which this construction produces automatically), and
# junction ties follow the sigma/traversal rule below.
# -----------------------------------------------------------------------------

class _MultiRegionBor:
    """surfaces: list of (points, is_conductor).  regions: list of dicts
    {"medium": None|(eps, mu), "bounds": [(surf_idx, sigma), ...],
    "exterior": bool}.  Interfaces bounding the exterior region must carry
    sigma = +1 there (the far field then sums their (J, M) directly)."""

    def __init__(self, surfaces, regions, freq_hz: 'float', gauss_order: 'int' = 4,
                 near_factor: 'float' = 2.0, near_order: 'int' = 12):
        surfaces = [
            (_validate_solve_bor_generatrix(points, "efie"), is_conductor)
            for points, is_conductor in surfaces
        ]
        for region in regions:
            medium = region.get("medium")
            if medium is not None:
                if not isinstance(medium, (tuple, list)) or len(medium) != 2:
                    raise ValueError(
                        "Each BoR region medium must be an (epsilon, mu) pair."
                    )
                _causal_medium(medium[0], medium[1])
        self.regions = regions
        self.n_surf = len(surfaces)
        self.is_cond = [bool(c) for (_, c) in surfaces]
        self.adj: 'List[List[int]]' = [[] for _ in surfaces]   # regions per surface
        self.sigma: 'Dict[Tuple[int, int], int]' = {}
        for ri, reg in enumerate(regions):
            for (si, sg) in reg["bounds"]:
                self.adj[si].append(ri)
                self.sigma[(ri, si)] = int(sg)
        self.ext_region = next(ri for ri, r in enumerate(regions) if r.get("exterior"))
        for (si, sg) in regions[self.ext_region]["bounds"]:
            if sg != +1:
                raise ValueError("Exterior-bounding surfaces must have sigma=+1.")
        # per-(surface, region) solvers
        self.solv: 'Dict[Tuple[int, int], BorPecSolver]' = {}
        for si, (pts, _) in enumerate(surfaces):
            for ri in self.adj[si]:
                self.solv[(si, ri)] = BorPecSolver(
                    pts, freq_hz, gauss_order=gauss_order,
                    medium=regions[ri]["medium"])
        # cross operators per region and ordered surface pair
        self.X: 'Dict[Tuple[int, int, int], BorCrossOperators]' = {}
        for ri, reg in enumerate(regions):
            ids = [si for (si, _) in reg["bounds"]]
            for si in ids:
                for sj in ids:
                    if si != sj:
                        self.X[(ri, si, sj)] = BorCrossOperators(
                            self.solv[(si, ri)], self.solv[(sj, ri)],
                            near_factor=near_factor, near_order=near_order)
        # DOF layout: per surface J (2Nn) then M' (2Nn, interfaces only)
        self.Nn = [self.solv[(si, self.adj[si][0])].Nn for si in range(self.n_surf)]
        self.off_J: 'List[int]' = []
        self.off_M: 'List[Optional[int]]' = []
        acc = 0
        for si in range(self.n_surf):
            self.off_J.append(acc)
            acc += 2 * self.Nn[si]
            if self.is_cond[si]:
                self.off_M.append(None)
            else:
                self.off_M.append(acc)
                acc += 2 * self.Nn[si]
        self.n_full = acc

        # -- junction detection over off-axis endpoints --
        all_pts = np.vstack([self.solv[(si, self.adj[si][0])].gen.nodes
                             for si in range(self.n_surf)])
        diag = max(float(np.ptp(all_pts[:, 0])) + float(np.ptp(all_pts[:, 1])), 1e-9)
        jn_tol = 1e-8 * diag
        self.junctions: 'List[List[Tuple[int, int]]]' = []   # [(surf, node), ...]
        for si in range(self.n_surf):
            gen = self.solv[(si, self.adj[si][0])].gen
            for node in (0, gen.n_nodes - 1):
                if gen.node_on_axis(node):
                    continue
                pos = gen.nodes[node]
                for jn in self.junctions:
                    s0, n0 = jn[0]
                    p0 = self.solv[(s0, self.adj[s0][0])].gen.nodes[n0]
                    if float(np.hypot(*(p0 - pos))) <= jn_tol:
                        jn.append((si, node))
                        break
                else:
                    self.junctions.append([(si, node)])
        for jn in self.junctions:
            if len(jn) < 2:
                si, node = jn[0]
                raise ValueError(f"Surface {si} has an off-axis free endpoint "
                                 "that is not part of a junction.")
            # master = first INTERFACE at the junction (an interface carries
            # the M master DOFs; a pure-conductor junction is just a corner
            # and any member may lead); every slave must share a region.
            mi = next((idx for idx, (si, _) in enumerate(jn)
                       if not self.is_cond[si]), 0)
            jn[0], jn[mi] = jn[mi], jn[0]
            mstr = jn[0]
            for (si, node) in jn[1:]:
                if not set(self.adj[si]) & set(self.adj[mstr[0]]):
                    raise ValueError("Junction surface shares no region with "
                                     "the junction master.")
        self._Q_cache: 'Dict[int, np.ndarray]' = {}

    # -- operator plumbing --
    def enable_streaming(self, m_max: 'int', plan: 'Dict[str, Any]') -> 'None':
        """Attach one aligned far-block stream to every region operator."""

        common = dict(
            single_blocks=bool(plan["use_single"]),
            tile_budget_gb=float(plan["tile_budget_gb"]),
            workers=int(plan["workers"]),
            mode_block=int(plan["mode_block"]),
        )
        for (surface_index, _region_index), solver in self.solv.items():
            solver.enable_streaming(
                m_max,
                efie=True,
                pmchwt=not self.is_cond[surface_index],
                **common,
            )
        for cross in self.X.values():
            cross.enable_streaming(m_max, **common)

    def prepare(self, m_max: 'int', workers: 'int' = 1) -> 'None':
        for (si, ri), s in self.solv.items():
            s.prepare_operators(m_max, efie=True, ibc=not self.is_cond[si],
                                workers=workers)
        for X in self.X.values():
            X.prepare(m_max)
        # Q depends on mode category, not the signed mode number.  Build each
        # possible category once before parallel mode assembly.
        for representative in range(min(int(m_max), 2) + 1):
            self.build_Q(representative)
        if int(m_max) >= 1:
            self.build_Q(-1)

    def _dir(self, si: 'int', node: 'int') -> 'int':
        """+1 if the drawn tangent points INTO the junction node (chain end)."""
        return +1 if node != 0 else -1

    def build_Q(self, m: 'int') -> 'np.ndarray':
        category = int(m) if abs(int(m)) == 1 else (0 if m == 0 else 2)
        Q = self._Q_cache.get(category)
        if Q is not None:
            return Q
        jn_nodes = {(si, node) for jn in self.junctions for (si, node) in jn}
        slave = {(si, node) for jn in self.junctions for (si, node) in jn[1:]}
        # phase-5 condition at CONDUCTOR junctions: E_phi shorts on the metal
        # line, so M_t = 0 for every interface end there (the master keeps
        # its free M_phi half-basis; slaves get only the phi tie back)
        m_t_masked = {(si, node) for jn in self.junctions
                      if any(self.is_cond[sj] for (sj, _) in jn)
                      for (si, node) in jn if not self.is_cond[si]}

        col = np.full(self.n_full, -1, dtype=int)
        red = 0

        def assign(offset, si, is_m):
            nonlocal red
            Nn = self.Nn[si]
            gen = self.solv[(si, self.adj[si][0])].gen
            t_act = np.ones(Nn, dtype=bool)
            f_act = np.ones(Nn, dtype=bool)
            for end in (0, Nn - 1):
                if gen.node_on_axis(end):
                    t_act[end] = (abs(m) == 1)
                    f_act[end] = False
                elif (si, end) in slave:
                    t_act[end] = False
                    f_act[end] = False
                elif (si, end) in jn_nodes:
                    if is_m and (si, end) in m_t_masked:
                        t_act[end] = False   # M_t = 0 at the conductor line
                else:
                    t_act[end] = False       # open edge (not expected)
            for i in range(Nn):
                if t_act[i]:
                    col[offset + i] = red; red += 1
            for i in range(Nn):
                if f_act[i]:
                    col[offset + Nn + i] = red; red += 1

        for si in range(self.n_surf):
            assign(self.off_J[si], si, False)
            if self.off_M[si] is not None:
                assign(self.off_M[si], si, True)

        Q = np.zeros((self.n_full, red), dtype=np.complex128)
        active = col >= 0
        Q[np.flatnonzero(active), col[active]] = 1.0
        for si in range(self.n_surf):
            solver = self.solv[(si, self.adj[si][0])]
            _apply_regular_axis_rows(Q, col, self.off_J[si], solver, m)
            if self.off_M[si] is not None:
                _apply_regular_axis_rows(
                    Q, col, self.off_M[si], solver, m
                )
        # ties: sigma/traversal rule via a region shared with the master.
        #   t:   sigma_rm dir_m J_m,t = -sigma_rs dir_s J_s,t
        #   phi: sigma_rm J_m,phi = sigma_rs J_s,phi
        # (the phi relations around a 3-region cycle are inconsistent -- the
        # projection H.t_hat differs per surface -- so each slave ties via a
        # region it shares with the MASTER and the remaining pairwise
        # relation is left to the Galerkin system.)
        for jn in self.junctions:
            sm, nm = jn[0]
            dir_m = self._dir(sm, nm)
            for (ss, ns) in jn[1:]:
                r = next(iter(set(self.adj[ss]) & set(self.adj[sm])))
                sg_m, sg_s = self.sigma[(r, sm)], self.sigma[(r, ss)]
                ct = -(sg_m * dir_m) / (sg_s * self._dir(ss, ns))
                cf = sg_m / sg_s
                mt = col[self.off_J[sm] + nm]
                mf = col[self.off_J[sm] + self.Nn[sm] + nm]
                if mt >= 0:
                    Q[self.off_J[ss] + ns, mt] = ct
                if mf >= 0:
                    Q[self.off_J[ss] + self.Nn[ss] + ns, mf] = cf
                if self.off_M[sm] is not None and self.off_M[ss] is not None:
                    mmt = col[self.off_M[sm] + nm]
                    mmf = col[self.off_M[sm] + self.Nn[sm] + nm]
                    if mmt >= 0:
                        Q[self.off_M[ss] + ns, mmt] = ct
                    if mmf >= 0:
                        Q[self.off_M[ss] + self.Nn[ss] + ns, mmf] = cf
        self._Q_cache[category] = Q
        return Q

    def assemble(self, m: 'int', m_max: 'int'):
        A = np.zeros((self.n_full, self.n_full), dtype=np.complex128)
        for ri, reg in enumerate(self.regions):
            eta_r = self.solv[(reg["bounds"][0][0], ri)].eta
            eta2 = (ETA0 / eta_r) ** 2
            for (si, sg_i) in reg["bounds"]:
                for (sj, sg_j) in reg["bounds"]:
                    ss = sg_i * sg_j
                    if si == sj:
                        T = self.solv[(si, ri)].assemble_mode(m, m_max)
                        P = self.solv[(si, ri)].assemble_pmchwt_P(m, m_max) \
                            if not self.is_cond[si] else None
                    else:
                        T = self.X[(ri, si, sj)].assemble_T(m, m_max)
                        P = self.X[(ri, si, sj)].assemble_P(m, m_max)
                    slJ_i = slice(self.off_J[si], self.off_J[si] + 2 * self.Nn[si])
                    slJ_j = slice(self.off_J[sj], self.off_J[sj] + 2 * self.Nn[sj])
                    A[slJ_i, slJ_j] += ss * T
                    if self.off_M[sj] is not None:
                        slM_j = slice(self.off_M[sj], self.off_M[sj] + 2 * self.Nn[sj])
                        A[slJ_i, slM_j] += -ETA0 * ss * P
                    if self.off_M[si] is not None:
                        slM_i = slice(self.off_M[si], self.off_M[si] + 2 * self.Nn[si])
                        A[slM_i, slJ_j] += ETA0 * ss * P
                        if self.off_M[sj] is not None:
                            A[slM_i, slM_j] += eta2 * ss * T
        Q = self.build_Q(m)
        return _reduce_constrained_operator(A, Q), None

    def rhs(self, m: 'int', th: 'float', pol: 'str') -> 'np.ndarray':
        V = np.zeros(self.n_full, dtype=np.complex128)
        for (si, _) in self.regions[self.ext_region]["bounds"]:
            s = self.solv[(si, self.ext_region)]
            V[self.off_J[si]:self.off_J[si] + 2 * self.Nn[si]] = s.rhs_mode(m, th, pol)
            if self.off_M[si] is not None:
                V[self.off_M[si]:self.off_M[si] + 2 * self.Nn[si]] = \
                    ETA0 * s.rhs_h_mode(m, th, pol)
        return self.build_Q(m).conj().T @ V

    def farfield(self, m: 'int', x_red: 'np.ndarray', th: 'float', pol: 'str') -> 'complex':
        x = self.build_Q(m) @ x_red
        fth = fph = 0.0
        for (si, _) in self.regions[self.ext_region]["bounds"]:
            s = self.solv[(si, self.ext_region)]
            J = x[self.off_J[si]:self.off_J[si] + 2 * self.Nn[si]]
            msol = (ETA0 * x[self.off_M[si]:self.off_M[si] + 2 * self.Nn[si]]
                    if self.off_M[si] is not None else None)
            ft, fp = s.farfield_mode(m, J, th, msol=msol)
            fth += ft; fph += fp
        return fth if pol == "VV" else fph

    def rho_max(self) -> 'float':
        return max(float(np.max(self.solv[(si, self.adj[si][0])].gen.nodes[:, 0]))
                   for si in range(self.n_surf))


def _solve_multiregion(sys_: '_MultiRegionBor', freq_hz, thetas_deg, n_modes,
                       mode_tol, workers, progress, check_abort,
                       formulation: 'str', extra: 'Dict',
                       table_precision: 'str' = "auto",
                       assembly: 'str' = "auto",
                       stream_budget_gb: 'float' = 8.0) -> 'Dict':
    thetas = _validated_bor_aspects(thetas_deg)
    k = 2.0 * math.pi * freq_hz / C0
    m_max, mode_tail_start = _bor_mode_limits(
        k, sys_.rho_max(), thetas, n_modes
    )
    solver_requirements = tuple(
        (solver, True, False, not sys_.is_cond[surface_index])
        for (surface_index, _region_index), solver in sys_.solv.items()
    )
    plan = _plan_multisurface_assembly(
        m_max,
        solver_requirements,
        tuple(sys_.X.values()),
        constraint_dofs=sys_.n_full,
        assembly=assembly,
        table_precision=table_precision,
        stream_budget_gb=stream_budget_gb,
        workers=workers,
    )

    def prepare(mm):
        if plan["use_streaming"] and all(
            solver._stream is None for solver in sys_.solv.values()
        ):
            sys_.enable_streaming(mm, plan)
        sys_.prepare(mm, workers=plan["workers"])

    F, modes_used, stats = _mode_sweep(
        sys_.n_full, thetas, ("VV", "HH"), m_max, mode_tol,
        lambda m: sys_.assemble(m, m_max), sys_.rhs, sys_.farfield,
        prepare=prepare,
        workers=plan["workers"], progress=progress, check_abort=check_abort,
        monitor_cond=True, min_mode_before_tail=mode_tail_start,
        assembly_peak_gb=plan["assembly_peak_gb"],
        memory_context=f"The {formulation} BoR solve")
    _require_mode_convergence(stats, mode_tol)
    extra = {**extra, **stats}
    warnings = list(extra.get("warnings", []) or [])
    extra["warnings"] = warnings
    streams = {
        **{
            f"surface_{surface_index}_region_{region_index}": solver._stream
            for (surface_index, region_index), solver in sys_.solv.items()
        },
        **{
            f"cross_region_{region_index}_{test_index}_{source_index}": cross._stream
            for (region_index, test_index, source_index), cross in sys_.X.items()
        },
    }
    stream_backends = {
        name: "native_c" if stream is not None and stream._native is not None
        else "numpy"
        for name, stream in streams.items()
    } if plan["use_streaming"] else {}
    unique_backends = set(stream_backends.values())
    out = {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(sys_.n_full),
        "n_junctions": len(sys_.junctions),
        "formulation": formulation,
        "assembly": "streaming" if plan["use_streaming"] else "tables",
        "table_precision": "single" if plan["use_single"] else "double",
        "stream_mode_block": plan["mode_block"],
        "stream_sweeps": (
            sum(stream.n_sweeps for stream in streams.values())
            if plan["use_streaming"] else 0
        ),
        "stream_sampling_backend": (
            next(iter(unique_backends)) if len(unique_backends) == 1
            else ("mixed" if unique_backends else None)
        ),
        "stream_sampling_backends": stream_backends,
        "stream_auxiliary_peak_gb": plan["auxiliary_peak_gb"],
    }
    out.update(extra)
    return out


@profiled_solve
def solve_bor_coated2_pec(points_outer, points_mid, points_core,
                          freq_hz: 'float', thetas_deg,
                          eps_inner: 'complex', mu_inner: 'complex',
                          eps_outer: 'complex', mu_outer: 'complex',
                          n_modes: 'Optional[int]' = None, gauss_order: 'int' = 4,
                          mode_tol: 'float' = 1e-6, near_factor: 'float' = 2.0,
                          near_order: 'int' = 12, workers: 'int' = 1,
                          progress: 'Optional[Callable]' = None,
                          check_abort: 'Optional[Callable]' = None,
                          table_precision: 'str' = "auto",
                          assembly: 'str' = "auto",
                          stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """PEC core under TWO full coating layers (all three generatrices closed
    axis-to-axis, +z -> -z, normals toward the exterior side)."""

    points_outer = _validate_solve_bor_generatrix(points_outer, "cfie")
    points_mid = _validate_solve_bor_generatrix(points_mid, "cfie")
    points_core = _validate_solve_bor_generatrix(points_core, "cfie")
    sys_ = _MultiRegionBor(
        surfaces=[(points_outer, False), (points_mid, False), (points_core, True)],
        regions=[
            {"medium": None, "bounds": [(0, +1)], "exterior": True},
            {"medium": (eps_outer, mu_outer), "bounds": [(0, -1), (1, +1)]},
            {"medium": (eps_inner, mu_inner), "bounds": [(1, -1), (2, +1)]},
        ],
        freq_hz=freq_hz, gauss_order=gauss_order,
        near_factor=near_factor, near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort,
                              "pmchwt-coated-2layer",
                              {"eps_inner": complex(eps_inner),
                               "eps_outer": complex(eps_outer)},
                              table_precision, assembly, stream_budget_gb)


@profiled_solve
def solve_bor_coated_n_pec(interface_points, points_core, freq_hz: 'float',
                           thetas_deg, eps_list, mu_list,
                           n_modes: 'Optional[int]' = None, gauss_order: 'int' = 4,
                           mode_tol: 'float' = 1e-6, near_factor: 'float' = 2.0,
                           near_order: 'int' = 12, workers: 'int' = 1,
                           progress: 'Optional[Callable]' = None,
                           check_abort: 'Optional[Callable]' = None,
                           table_precision: 'str' = "auto",
                           assembly: 'str' = "auto",
                           stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """PEC core under N full coating layers.  interface_points is the list
    of interface generatrices OUTERMOST FIRST; eps_list/mu_list are per
    layer INNERMOST FIRST (matching mie_sphere.sigma_multilayer_pec_sphere)."""

    N = len(interface_points)
    if N < 1:
        raise ValueError("At least one coating interface is required.")
    if len(eps_list) != N or len(mu_list) != N:
        raise ValueError("One (eps, mu) per layer, innermost first.")
    interface_points = [
        _validate_solve_bor_generatrix(points, "cfie")
        for points in interface_points
    ]
    points_core = _validate_solve_bor_generatrix(points_core, "cfie")
    surfaces = [(p, False) for p in interface_points] + [(points_core, True)]
    regions = [{"medium": None, "bounds": [(0, +1)], "exterior": True}]
    for i in range(N):
        # surface i separates layer (N - i) outside from layer (N - i - 1)
        # inside; layer j uses eps_list[j - 1] (innermost first).
        lay = N - i          # region below surface i is layer `lay`
        inner_bound = (i + 1, +1) if i == N - 1 else (i + 1, +1)
        regions.append({"medium": (eps_list[lay - 1], mu_list[lay - 1]),
                        "bounds": [(i, -1), inner_bound]})
    sys_ = _MultiRegionBor(surfaces=surfaces, regions=regions, freq_hz=freq_hz,
                           gauss_order=gauss_order, near_factor=near_factor,
                           near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort,
                              f"pmchwt-coated-{N}layer",
                              {"eps_layers": [complex(e) for e in eps_list]},
                              table_precision, assembly, stream_budget_gb)


@profiled_solve
def solve_bor_coating_patch(points_patch, points_mid_covered, points_mid_bare,
                            points_core, freq_hz: 'float', thetas_deg,
                            eps_inner: 'complex', mu_inner: 'complex',
                            eps_patch: 'complex', mu_patch: 'complex',
                            n_modes: 'Optional[int]' = None, gauss_order: 'int' = 4,
                            mode_tol: 'float' = 1e-6, near_factor: 'float' = 2.0,
                            near_order: 'int' = 12, workers: 'int' = 1,
                            progress: 'Optional[Callable]' = None,
                            check_abort: 'Optional[Callable]' = None,
                            table_precision: 'str' = "auto",
                            assembly: 'str' = "auto",
                            stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """A second-layer coating PATCH terminating on a fully coated PEC body:
    the patch's outer interface (points_patch) meets the inner coating's
    interface at dielectric triple junctions (air / patch / inner coating --
    no conductor on the junction line).  points_mid_covered is the part of
    the inner interface under the patch, points_mid_bare the exposed
    part(s) (a list); the PEC core stays fully covered by the inner layer."""

    points_core = _validate_solve_bor_generatrix(points_core, "cfie")
    bare_list = (points_mid_bare if isinstance(points_mid_bare, (list, tuple))
                 else [points_mid_bare])
    surfaces = [(points_patch, False), (points_mid_covered, False)]
    surfaces += [(p, False) for p in bare_list]
    surfaces.append((points_core, True))
    core_idx = len(surfaces) - 1
    bare_idx = list(range(2, 2 + len(bare_list)))
    regions = [
        {"medium": None, "exterior": True,
         "bounds": [(0, +1)] + [(bi, +1) for bi in bare_idx]},
        {"medium": (eps_patch, mu_patch),
         "bounds": [(0, -1), (1, +1)]},
        {"medium": (eps_inner, mu_inner),
         "bounds": [(1, -1)] + [(bi, -1) for bi in bare_idx] + [(core_idx, +1)]},
    ]
    sys_ = _MultiRegionBor(surfaces=surfaces, regions=regions, freq_hz=freq_hz,
                           gauss_order=gauss_order, near_factor=near_factor,
                           near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort,
                              "pmchwt-coating-patch",
                              {"eps_inner": complex(eps_inner),
                               "eps_patch": complex(eps_patch)},
                              table_precision, assembly, stream_budget_gb)


# -----------------------------------------------------------------------------
# Canonical generatrices for gates
# -----------------------------------------------------------------------------

def sphere_generatrix(a: 'float', n: 'int') -> 'np.ndarray':
    """North pole (+z) to south pole: outward left-normals per convention."""
    th = np.linspace(0.0, math.pi, n + 1)
    return np.column_stack([a * np.sin(th), a * np.cos(th)])


def cylinder_generatrix(a: 'float', L: 'float', n_rad: 'int', n_len: 'int') -> 'np.ndarray':
    """Closed cylinder: top cap center -> rim -> side -> bottom rim -> center."""
    top = np.column_stack([np.linspace(0.0, a, n_rad + 1), np.full(n_rad + 1, L / 2)])
    side = np.column_stack([np.full(n_len - 1, a), np.linspace(L / 2, -L / 2, n_len + 1)[1:-1]])
    bot = np.column_stack([np.linspace(a, 0.0, n_rad + 1), np.full(n_rad + 1, -L / 2)])
    return np.vstack([top, side, bot])
