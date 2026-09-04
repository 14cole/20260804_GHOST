"""
Phase-7b streaming far assembly for the BoR solver.

The table path stores modal kernels at GAUSS-POINT pairs -- [P, P, modes]
with P = gauss_order * N_t -- which is the memory bound at scale.  This
module keeps the FFT-over-azimuth amortization (one xi sweep yields every
mode) but contracts BOTH Galerkin sides immediately, tile by tile, so the
persistent storage is per-mode NODAL blocks:

    EFIE   9 * (m_max + 2) * Nn^2   (order primitives through m_max + 1)
    MFIE   4 * (2 m_max + 1) * Nn^2 (brackets have mixed parity)
    IBC    4 * (2 m_max + 1) * Nn^2 (source Z_s baked into the contraction)

-- a 16x reduction versus the tables (32x with single-precision blocks).

For each tile of test elements the azimuthal integrand is sampled on the
same uniform xi grid the table path uses ([rows, P, n_xi]), FFT'd, near
(local-neighbour element) pair entries are zeroed exactly as the table path zeroes
them, and the modal kernels are contracted through the reference shape
functions:

    out[e + a, f + b] += sum_gh L[a, e, g] K[e, g, f, h] R[b, f, h]

with L/R the per-point nodal weights (shape value x rho w x tangent
component, or the (rho T)' divergence weights).  Because tiles use the same
xi grid, the same FFT, and the same Galerkin points as the table path, the
streamed blocks match the table-path contraction to float roundoff; the
streaming/table regression in tests/test_bor_physics_regression.py checks that.

The near/self machinery (graded cells, adaptive kernels) is untouched: the
solver adds its near corrections on top of these far blocks as before.
"""

import ctypes
import os
import platform
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np

from bor_kernels import _mfie_brackets, _ibc_brackets_grid


# -- phase-7c native sampling kernel (ctypes; NumPy fallback if absent) --

_DLL_DIRECTORY_HANDLES = []


def _prepare_windows_dll_search() -> 'None':
    """Expose trusted compiler-runtime directories for native dependencies."""

    if platform.system().lower() != "windows" or not hasattr(
        os, "add_dll_directory"
    ):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [here]
    configured = os.environ.get("GHOST_NATIVE_DLL_DIR", "").strip()
    if configured:
        candidates.append(configured)
    for compiler_name in ("gcc", "cc", "clang"):
        compiler = shutil.which(compiler_name)
        if compiler:
            candidates.append(os.path.dirname(os.path.abspath(compiler)))
    candidates.extend([
        r"C:\msys64\ucrt64\bin",
        r"C:\msys64\mingw64\bin",
    ])
    seen = set()
    for directory in candidates:
        resolved = os.path.normcase(os.path.abspath(directory))
        if resolved in seen or not os.path.isdir(resolved):
            continue
        seen.add(resolved)
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(resolved))
        except OSError:
            continue

def _native_extensions(system_name: 'str') -> 'tuple[str, ...]':
    """Shared-library suffixes that the current host can safely load."""

    key = str(system_name).strip().lower()
    if key == "windows":
        return (".dll",)
    return (".so",)


def _load_native():
    _prepare_windows_dll_search()
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (f"bor_stream_kernel.{sysname}-{machine}", "bor_stream_kernel"):
        for extension in _native_extensions(sysname):
            path = os.path.join(here, base + extension)
            if not os.path.exists(path):
                continue
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            if not all(
                hasattr(lib, symbol)
                for symbol in ("sample_g", "sample_mfie", "sample_ibc")
            ):
                continue
            dp = ctypes.POINTER(ctypes.c_double)
            ci = ctypes.c_int
            cd = ctypes.c_double
            lib.sample_g.argtypes = [ci, ci, ci, dp, dp, dp, dp, cd, dp, dp]
            lib.sample_g.restype = None
            bracket_args = [ci, ci, ci] + [dp] * 8 + [cd, dp, dp] + [dp] * 4
            lib.sample_mfie.argtypes = bracket_args
            lib.sample_mfie.restype = None
            lib.sample_ibc.argtypes = bracket_args
            lib.sample_ibc.restype = None
            return lib
    return None


_NATIVE = _load_native()
_FALLBACK_NOTICE_SHOWN = False


def sampling_backend_name() -> 'str':
    """Auditable backend label for streamed far-kernel sampling."""

    return "native_c" if _NATIVE is not None else "numpy"


def _notice_numpy_fallback():
    """One-time stderr notice when the streaming build runs on the NumPy
    sampler.  Results are bit-equivalent; assembly is ~2-8x slower.  It also
    diagnoses a native binary copied from the wrong operating system, which
    the loader correctly refuses to load."""
    global _FALLBACK_NOTICE_SHOWN
    if _FALLBACK_NOTICE_SHOWN:
        return
    _FALLBACK_NOTICE_SHOWN = True
    here = os.path.dirname(os.path.abspath(__file__))
    system_name = platform.system().lower()
    tag = f"{system_name}-{platform.machine().lower()}"
    others = [f for f in sorted(os.listdir(here))
              if f.startswith("bor_stream_kernel.")
              and os.path.splitext(f)[1].lower() in {".so", ".dll"}
              and tag not in f]
    hint = (f" (found {', '.join(others)} -- built for a DIFFERENT platform, "
            "so it was correctly skipped)" if others else "")
    print(
        "bor_streaming: native sampling kernel not available for this "
        f"platform{hint}; using the NumPy fallback (bit-equivalent, ~2-8x "
        "slower assembly). Compile and load-check it on THIS machine with:\n"
        "  py build_bor_stream_kernel.py",
        file=sys.stderr, flush=True)


def _dp(a: 'np.ndarray'):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


from bor_kernels import n_xi_for_pairs


BOR_STREAM_TILE_BUDGET_GB = 1.0
# Conservative peak transient storage for one sampled (test, source, xi)
# tuple.  The NumPy MFIE/IBC fallback keeps four real geometry grids, the
# complex Green-function factor, roughly nine real bracket factors and four
# complex outputs live while forming a tile; FFT output and allocator
# temporaries raise the observed peak above the former 176-byte estimate.
# 256 bytes covers that live set and modest FFT/allocator slack.  This is an
# accounting constant only: changing the row/column tiling does not change
# the quadrature samples or modal bins.
BOR_STREAM_TILE_BYTES_PER_SAMPLE = 256.0


def _aligned_stream_mode_block(
    m_max: 'int', mode_block: 'Optional[int]', workers: 'int'
) -> 'int':
    """Return the exact range size shared by planning and runtime."""

    mode_count = int(m_max) + 1
    worker_count = max(1, int(workers))
    if mode_count < 1:
        raise ValueError("Streaming mode maximum must be non-negative.")
    requested = mode_count if mode_block is None else int(mode_block)
    if requested < 1:
        raise ValueError("Streaming mode block must be positive.")
    requested = max(requested, worker_count)
    aligned = ((requested + worker_count - 1) // worker_count) * worker_count
    return min(aligned, mode_count)


def _streaming_worker_count(
    gauss_order: 'int', n_xi: 'int', tile_budget_gb: 'float', workers: 'int'
) -> 'int':
    """Cap simultaneous sampling tiles when the one-column floor requires it."""

    go = int(gauss_order)
    samples = int(n_xi)
    requested = max(1, int(workers))
    budget = float(tile_budget_gb) * 1.0e9
    if go < 1 or samples < 1:
        raise ValueError("Streaming tile dimensions must be positive.")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("Streaming tile budget must be positive and finite.")
    one_tile = go * samples * BOR_STREAM_TILE_BYTES_PER_SAMPLE
    if budget < one_tile:
        raise ValueError(
            "Streaming tile budget is below the modeled one-element, "
            f"one-column minimum of {one_tile / 1.0e9:.6g} GB "
            f"(gauss_order={go}, n_xi={samples})."
        )
    return min(requested, max(1, int(budget / one_tile)))


def _streaming_tile_shape(
    n_elements: 'int', gauss_order: 'int', point_count: 'int', n_xi: 'int',
    tile_budget_gb: 'float', workers: 'int',
) -> 'tuple[int, int]':
    """Return (test-element rows, source-point columns) within the budget.

    The caller must first validate the one-element/one-column floor with
    :func:`_streaming_worker_count`.  Production then caps simultaneous
    sampling workers so the allowance covers every live tile.
    """

    ne = int(n_elements)
    go = int(gauss_order)
    points = int(point_count)
    samples = int(n_xi)
    threads = max(1, int(workers))
    budget = float(tile_budget_gb) * 1.0e9
    if ne < 1 or go < 1 or points < 1 or samples < 1:
        raise ValueError("Streaming tile dimensions must be positive.")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("Streaming tile budget must be positive and finite.")

    rows_max = max(
        go,
        int(
            budget
            / (
                points
                * samples
                * BOR_STREAM_TILE_BYTES_PER_SAMPLE
                * threads
            )
        ),
    )
    tile_elements = min(ne, max(1, rows_max // go))
    source_columns = max(
        1,
        min(
            points,
            int(
                budget
                / (
                    tile_elements
                    * go
                    * samples
                    * BOR_STREAM_TILE_BYTES_PER_SAMPLE
                    * threads
                )
            ),
        ),
    )
    return tile_elements, source_columns


def _n_xi_efie(k: 'complex', rho_max: 'float', m_max: 'int', d_min: 'float' = 0.0) -> 'int':
    return n_xi_for_pairs(k, rho_max, m_max, d_min, bracket=False)


def _n_xi_bracket(k: 'complex', rho_max: 'float', m_max: 'int', d_min: 'float' = 0.0) -> 'int':
    return n_xi_for_pairs(k, rho_max, m_max, d_min, bracket=True)


class StreamingFarBlocks:
    """Per-mode nodal far blocks for one BorPecSolver surface.

    efie=True builds the four EFIE blocks (without the C = jk eta 2pi
    factor, matching _pair_blocks); mfie=True the four MFIE bracket blocks
    (WITH the 2pi Galerkin factor, matching assemble_mfie_mode's far
    contraction); ibc_zs_pt (per-Gauss-point Z_s) the IBC bracket blocks
    with the source weight baked in; pmchwt=True builds the same rotated-PV
    blocks with unit source weight (both matching _rot_pv_blocks)."""

    def __init__(self, solver, m_max: 'int', efie: 'bool' = True,
                 mfie: 'bool' = False, ibc_zs_pt: 'Optional[np.ndarray]' = None,
                 pmchwt: 'bool' = False,
                 dtype=np.complex128,
                 tile_budget_gb: 'float' = BOR_STREAM_TILE_BUDGET_GB,
                 workers: 'int' = 1, mode_block: 'Optional[int]' = None):
        self.solver = solver
        self.m_max = int(m_max)
        self.dtype = dtype
        g = solver.g
        gen = solver.gen
        self.Nn = solver.Nn
        ne = gen.n_elems
        self.go = solver.gauss_order
        P = solver.P
        k = solver.k
        Nn, go, mm = self.Nn, self.go, self.m_max

        # -- per-point nodal weight vectors [2, P] --
        wrho = g.w * g.rho
        self._lv = {
            "r": np.stack([g.T0 * wrho * g.trho, g.T1 * wrho * g.trho]),
            "z": np.stack([g.T0 * wrho * g.tz, g.T1 * wrho * g.tz]),
            "1": np.stack([g.T0 * wrho, g.T1 * wrho]),
            "s": np.stack([g.T0 * g.w, g.T1 * g.w]),
            "d": np.stack([g.dRT0 * g.w, g.dRT1 * g.w]),
        }
        if ibc_zs_pt is not None and pmchwt:
            raise ValueError(
                "Streaming rotated-PV blocks cannot be both IBC-weighted "
                "and unit-weight PMCHWT blocks."
            )
        rv_ibc = None
        if ibc_zs_pt is not None:
            rv_ibc = np.stack([g.T0 * wrho * ibc_zs_pt, g.T1 * wrho * ibc_zs_pt])
        elif pmchwt:
            rv_ibc = np.stack([g.T0 * wrho, g.T1 * wrho])

        # -- configuration (block storage is allocated per MODE RANGE) --
        # EFIE: 9 order-primitives (the Gc/Gs neighbor relations and the
        # mode-dependent scalars commute with contraction, so per-mode blocks
        # are combined on retrieval): rr, zz, r1, 1r, 11, dd, ds, sd, ss.
        self._efie = efie
        self._mfie = mfie
        self._has_ibc = ibc_zs_pt is not None or bool(pmchwt)
        self.rot_pv_unit_source = bool(pmchwt)
        self._rv_ibc = rv_ibc
        self.k = complex(k)
        workers = max(1, int(workers))
        # phase-7d mode-block re-sweeps: hold blocks only for an aligned
        # range of modes; re-run the sampling sweep when the mode loop
        # advances past it (memory / n_ranges at sampling x n_ranges).
        # Ranges are multiples of the engine's worker count so a thread
        # wave never straddles a rebuild.
        self.mode_block = _aligned_stream_mode_block(mm, mode_block, workers)
        self.n_sweeps = 0

        rho_max = float(np.max(gen.nodes[:, 0]))
        # xi grids sized for the FULL mode set AND the closest far pair
        # (identical criteria to the table path for every range, preserving
        # the bit-level equivalence gates)
        gap = solver._far_gap()
        self._nx_e = _n_xi_efie(k, rho_max, mm, gap)
        self._nx_b = _n_xi_bracket(k, rho_max, mm, gap)
        nx_worst = max(self._nx_e,
                       self._nx_b if (mfie or self._has_ibc) else 0)
        self._workers = _streaming_worker_count(
            go, nx_worst, tile_budget_gb, workers
        )
        # Tile the [rows, P, n_xi] sampling footprint.  Each worker thread
        # holds its own tile; the shared helper uses the conservative live-set
        # accounting documented by BOR_STREAM_TILE_BYTES_PER_SAMPLE.
        self._te, self._cols = _streaming_tile_shape(
            ne, go, P, nx_worst, tile_budget_gb, self._workers
        )
        # Fine meshes can push n_xi into the thousands, so row tiling alone
        # may leave a very wide source dimension.  The samplers therefore
        # ALSO chunk over source columns. FFT + mode binning happen per chunk
        # (row/column independent, so bit-identical), and only the small kept
        # [rows, P, modes] slices persist.  The validated one-column floor and
        # capped sampling concurrency keep the modeled live set within budget;
        # cols == P when row tiling alone is sufficient.
        # native (ctypes) sampling kernel: real-k only (the air region --
        # exactly what the solve_bor streaming path serves)
        self._native = (_NATIVE if (_NATIVE is not None and
                                    abs(complex(k).imag) == 0.0) else None)
        if _NATIVE is None:
            _notice_numpy_fallback()
        self._q = tuple(np.ascontiguousarray(v) for v in
                        (g.rho, g.z, g.trho, g.tz))
        self._acc_lock = threading.Lock()
        self._range_lock = threading.RLock()
        self.Z = self.K = self.B = None
        self.lo, self.hi = 1, 0          # empty range
        self._ord_lo = 0
        self._sidx: 'Dict[int, int]' = {}
        self._ensure(0)

    # -- mode-range machinery --
    def _ensure(self, am: 'int') -> 'None':
        if self.lo <= am <= self.hi:
            return
        with self._range_lock:
            if self.lo <= am <= self.hi:
                return
            lo = (am // self.mode_block) * self.mode_block
            hi = min(lo + self.mode_block - 1, self.m_max)
            self._build_range(lo, hi)

    def _build_range(self, lo: 'int', hi: 'int') -> 'None':
        Nn, go, mm = self.Nn, self.go, self.m_max
        ne = self.solver.gen.n_elems
        k = self.solver.k
        ord_lo = max(0, lo - 1)
        n_ord = hi + 2 - ord_lo
        # Release the previous range before allocating its replacement.  A
        # direct ``self.Z = np.zeros(...)`` assignment keeps the old array
        # alive until the new RHS exists, transiently defeating the retained
        # block estimate during every re-sweep.
        self.Z = self.K = self.B = None
        if self._efie:
            self.Z = np.zeros((9, n_ord, Nn, Nn), dtype=self.dtype)
        ms = [m for m in range(-hi, hi + 1) if lo <= abs(m) <= hi]
        self._sidx = {m: i for i, m in enumerate(ms)}
        if self._mfie:
            self.K = np.zeros((4, len(ms), Nn, Nn), dtype=self.dtype)
        if self._has_ibc:
            self.B = np.zeros((4, len(ms), Nn, Nn), dtype=self.dtype)
        self._ord_lo = ord_lo

        orders = np.arange(ord_lo, hi + 2)
        ph_e = np.exp(1j * np.pi * orders) * (2.0 * np.pi / self._nx_e)
        msarr = np.asarray(ms)
        bins_b = np.where(msarr >= 0, msarr, self._nx_b + msarr)
        ph_b = np.exp(1j * np.pi * msarr) * (2.0 * np.pi / self._nx_b)

        def do_tile(e0):
            e1 = min(e0 + self._te, ne)
            rows = slice(e0 * go, e1 * go)
            re = e1 - e0
            if self._efie:
                Gn = self._sample_G(rows, k, self._nx_e, ph_e, ord_lo, hi)
                self._zero_near(Gn, e0, e1)
                self._accumulate_efie(Gn, rows, e0, re, k)
            if self._mfie:
                Fs = self._sample_brackets(
                    "mfie", rows, re, k, self._nx_b, bins_b, ph_b
                )
                self._accumulate_brackets(Fs, self.K, self._lv["1"],
                                          self._lv["1"], rows, e0, re)
            if self._has_ibc:
                Fs = self._sample_brackets(
                    "ibc", rows, re, k, self._nx_b, bins_b, ph_b
                )
                self._accumulate_brackets(Fs, self.B, self._lv["1"],
                                          self._rv_ibc, rows, e0, re)

        starts = list(range(0, ne, self._te))
        if self._workers <= 1:
            for e0 in starts:
                do_tile(e0)
        else:
            with ThreadPoolExecutor(max_workers=self._workers) as ex:
                list(ex.map(do_tile, starts))
        self.lo, self.hi = lo, hi
        self.n_sweeps += 1

    def _sample_brackets(self, which: 'str', rows, re: 'int', k, nx_b: 'int',
                         bins, phase):
        """Return four kept-mode bracket tiles while bounding raw FFT memory.

        Source columns are independent.  Sample and transform at most
        ``self._cols`` columns at a time, then retain only the requested modal
        bins in the full-width output used by the Galerkin contraction.
        """
        g = self.solver.g
        P = self.solver.P
        go = self.go
        nr = re * go
        xi = 2.0 * np.pi * np.arange(nx_b) / nx_b - np.pi
        kept = tuple(np.empty((nr, P, len(bins)), dtype=np.complex128)
                     for _ in range(4))
        rp = np.ascontiguousarray(g.rho[rows])
        zp = np.ascontiguousarray(g.z[rows])
        trp = np.ascontiguousarray(g.trho[rows])
        tzp = np.ascontiguousarray(g.tz[rows])
        cx = np.ascontiguousarray(np.cos(xi))
        sx = np.ascontiguousarray(np.sin(xi))
        for c0 in range(0, P, self._cols):
            c1 = min(c0 + self._cols, P)
            cols = slice(c0, c1)
            nc = c1 - c0
            if self._native is not None:
                rho_q, z_q, tr_q, tz_q = (
                    np.ascontiguousarray(value[cols]) for value in self._q
                )
                sampled = tuple(np.empty((nr, nc, nx_b), dtype=np.complex128)
                                for _ in range(4))
                fn = (self._native.sample_mfie if which == "mfie"
                      else self._native.sample_ibc)
                fn(nr, nc, nx_b, _dp(rp), _dp(zp), _dp(trp), _dp(tzp),
                   _dp(rho_q), _dp(z_q), _dp(tr_q), _dp(tz_q),
                   float(np.real(k)), _dp(cx), _dp(sx),
                   _dp(sampled[0]), _dp(sampled[1]),
                   _dp(sampled[2]), _dp(sampled[3]))
            elif which == "mfie":
                sampled = _mfie_brackets(
                    rp[:, None], zp[:, None], trp[:, None], tzp[:, None],
                    g.rho[None, cols], g.z[None, cols],
                    g.trho[None, cols], g.tz[None, cols], k, xi)
            else:
                pair_shape = (nr, nc)
                sampled = _ibc_brackets_grid(
                    np.broadcast_to(rp[:, None], pair_shape).ravel(),
                    np.broadcast_to(zp[:, None], pair_shape).ravel(),
                    np.broadcast_to(trp[:, None], pair_shape).ravel(),
                    np.broadcast_to(tzp[:, None], pair_shape).ravel(),
                    np.broadcast_to(g.rho[None, cols], pair_shape).ravel(),
                    np.broadcast_to(g.z[None, cols], pair_shape).ravel(),
                    np.broadcast_to(g.trho[None, cols], pair_shape).ravel(),
                    np.broadcast_to(g.tz[None, cols], pair_shape).ravel(),
                    k, np.broadcast_to(xi, (nr * nc, nx_b)))
                sampled = tuple(F.reshape(nr, nc, nx_b) for F in sampled)
            for uv, values in enumerate(sampled):
                spectrum = np.fft.fft(values, axis=-1)
                kept[uv][:, cols] = (
                    spectrum[..., bins] * (2.0 * np.pi * phase)
                )
        return kept

    # -- sampling / masking --
    def _sample_G(self, rows, k, n_xi, phase, ord_lo, hi):
        g = self.solver.g
        xi = 2.0 * np.pi * np.arange(n_xi) / n_xi - np.pi
        rp = np.ascontiguousarray(g.rho[rows])
        zp = np.ascontiguousarray(g.z[rows])
        sin2 = np.ascontiguousarray(np.sin(0.5 * xi) ** 2)
        P = self.solver.P
        kept = np.empty((len(rp), P, hi + 2 - ord_lo),
                        dtype=np.complex128)
        for c0 in range(0, P, self._cols):
            c1 = min(c0 + self._cols, P)
            cols = slice(c0, c1)
            nc = c1 - c0
            if self._native is not None:
                rho_q = np.ascontiguousarray(self._q[0][cols])
                z_q = np.ascontiguousarray(self._q[1][cols])
                gk = np.empty((len(rp), nc, n_xi), dtype=np.complex128)
                self._native.sample_g(len(rp), nc, n_xi,
                                      _dp(rp), _dp(zp), _dp(rho_q),
                                      _dp(z_q), float(np.real(k)),
                                      _dp(sin2), _dp(gk))
            else:
                d2 = (rp[:, None] - g.rho[None, cols]) ** 2 + \
                     (zp[:, None] - g.z[None, cols]) ** 2
                rr4 = 4.0 * rp[:, None] * g.rho[None, cols]
                R = np.sqrt(d2[..., None] + rr4[..., None] * sin2)
                R = np.maximum(R, 1e-300)
                gk = np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R)
            spectrum = np.fft.fft(gk, axis=-1)
            kept[:, cols] = spectrum[..., ord_lo:hi + 2] * phase
        return kept

    def _zero_near(self, Kt, e0, e1):
        go = self.go
        for e in range(e0, e1):
            row = slice((e - e0) * go, (e - e0 + 1) * go)
            for f in self.solver._near_sources_by_element[e]:
                Kt[row, f * go:(f + 1) * go] = 0.0

    # -- contraction: ALL modes/orders in one einsum per weight pair --
    def _contract_all(self, Kn, lv_rows, rv, e0, re, out):
        """out[n_modes, Nn, Nn] += nodal contraction of Kn [rows, P, n_modes]
        with per-point left weights lv_rows [2, rows], right weights rv [2, P]."""
        go = self.go
        ce = self.solver.gen.n_elems
        Kr = Kn.reshape(re, go, ce, go, -1)
        L = lv_rows.reshape(2, re, go)
        R = rv.reshape(2, ce, go)
        M = np.einsum("aeg,egfhm,bfh->maebf", L, Kr, R, optimize=True)
        if out.dtype != M.dtype:
            M = M.astype(out.dtype)
        # adjacent tiles share a boundary node row: serialize the adds
        with self._acc_lock:
            out[:, e0:e0 + re, 0:ce] += M[:, 0, :, 0, :]
            out[:, e0:e0 + re, 1:ce + 1] += M[:, 0, :, 1, :]
            out[:, e0 + 1:e0 + re + 1, 0:ce] += M[:, 1, :, 0, :]
            out[:, e0 + 1:e0 + re + 1, 1:ce + 1] += M[:, 1, :, 1, :]

    _EFIE_COMBOS = (("r", "r"), ("z", "z"), ("r", "1"), ("1", "r"),
                    ("1", "1"), ("d", "d"), ("d", "s"), ("s", "d"), ("s", "s"))

    def _accumulate_efie(self, Gn, rows, e0, re, k):
        for ci, (lx, rx) in enumerate(self._EFIE_COMBOS):
            self._contract_all(Gn, self._lv[lx][:, rows], self._lv[rx],
                               e0, re, self.Z[ci])

    def _accumulate_brackets(self, Fs, store, lv_full, rv, rows, e0, re):
        lv_rows = lv_full[:, rows]
        for uv, Km_all in enumerate(Fs):
            self._zero_near(Km_all, e0, e0 + re)
            self._contract_all(Km_all, lv_rows, rv, e0, re, store[uv])

    # -- per-mode block retrieval (complex128 copies; near loops add onto them) --
    def efie_blocks(self, m: 'int'):
        # primitive order: 0 rr, 1 zz, 2 r1, 3 1r, 4 11, 5 dd, 6 ds, 7 sd, 8 ss
        am = abs(m)
        with self._range_lock:
            self._ensure(am)
            k = self.k
            o = self._ord_lo
            lo, hi = abs(am - 1), am + 1

            def g(ci, n):
                return self.Z[ci, n - o].astype(np.complex128)

            ztt = 0.5 * (g(0, lo) + g(0, hi)) + g(1, am) - (1.0 / k ** 2) * g(5, am)
            ztf = (g(2, lo) - g(2, hi)) / 2j - (1j * am / k ** 2) * g(6, am)
            zft = -(g(3, lo) - g(3, hi)) / 2j + (1j * am / k ** 2) * g(7, am)
            zff = 0.5 * (g(4, lo) + g(4, hi)) - (am ** 2 / k ** 2) * g(8, am)
        if m < 0:
            ztf = -ztf
            zft = -zft
        return ztt, ztf, zft, zff

    def bracket_blocks(self, which: 'str', m: 'int'):
        with self._range_lock:
            self._ensure(abs(m))
            store = self.K if which == "mfie" else self.B
            mi = self._sidx[m]
            return tuple(store[uv, mi].astype(np.complex128) for uv in range(4))

    def memory_gb(self) -> 'float':
        total = 0
        for arr in (self.Z, self.K, self.B):
            if arr is not None:
                total += arr.nbytes
        return total / 1e9


class StreamingCrossFarBlocks:
    """Bounded far blocks for one rectangular BorCrossOperators mapping.

    Test and source generatrices may have different element/node counts.  The
    stored primitives match :class:`StreamingFarBlocks`: nine EFIE order
    blocks and four unit-source rotated-PV blocks.  Cross-operator near pairs
    remain excluded here and are added by ``BorCrossOperators`` with its
    existing high-order/graded quadrature.
    """

    _EFIE_COMBOS = StreamingFarBlocks._EFIE_COMBOS

    def __init__(self, cross, m_max: 'int', dtype=np.complex128,
                 tile_budget_gb: 'float' = BOR_STREAM_TILE_BUDGET_GB,
                 workers: 'int' = 1, mode_block: 'Optional[int]' = None):
        self.cross = cross
        self.sp, self.sq = cross.sp, cross.sq
        self.m_max = int(m_max)
        self.dtype = dtype
        gp, gq = self.sp.g, self.sq.g
        self.Np, self.Nq = self.sp.Nn, self.sq.Nn
        self.go_p, self.go_q = self.sp.gauss_order, self.sq.gauss_order
        self.Pq = self.sq.P
        self.k = complex(cross.k)

        def weights(g):
            wrho = g.w * g.rho
            return {
                "r": np.stack([g.T0 * wrho * g.trho,
                               g.T1 * wrho * g.trho]),
                "z": np.stack([g.T0 * wrho * g.tz,
                               g.T1 * wrho * g.tz]),
                "1": np.stack([g.T0 * wrho, g.T1 * wrho]),
                "s": np.stack([g.T0 * g.w, g.T1 * g.w]),
                "d": np.stack([g.dRT0 * g.w, g.dRT1 * g.w]),
            }

        self._lv = weights(gp)
        self._rv = weights(gq)
        workers = max(1, int(workers))
        self.mode_block = _aligned_stream_mode_block(
            self.m_max, mode_block, workers
        )
        self.n_sweeps = 0
        rho_max = max(
            float(np.max(self.sp.gen.nodes[:, 0])),
            float(np.max(self.sq.gen.nodes[:, 0])),
        )
        self._nx_e = _n_xi_efie(
            self.k, rho_max, self.m_max, cross._far_gap
        )
        self._nx_b = _n_xi_bracket(
            self.k, rho_max, self.m_max, cross._far_gap
        )
        nx_worst = max(self._nx_e, self._nx_b)
        self._workers = _streaming_worker_count(
            max(self.go_p, self.go_q), nx_worst, tile_budget_gb, workers
        )
        self._te, self._cols = _streaming_tile_shape(
            self.sp.gen.n_elems, self.go_p, self.Pq, nx_worst,
            tile_budget_gb, self._workers,
        )
        self._native = (
            _NATIVE if _NATIVE is not None and abs(self.k.imag) == 0.0
            else None
        )
        if _NATIVE is None:
            _notice_numpy_fallback()
        self._q = tuple(np.ascontiguousarray(value) for value in
                        (gq.rho, gq.z, gq.trho, gq.tz))
        self._near_sources = {
            e: [] for e in range(self.sp.gen.n_elems)
        }
        for e, f in cross.near_pairs:
            self._near_sources[e].append(f)
        self._acc_lock = threading.Lock()
        self._range_lock = threading.RLock()
        self.Z = self.B = None
        self.lo, self.hi = 1, 0
        self._ord_lo = 0
        self._sidx: 'Dict[int, int]' = {}
        self._ensure(0)

    def _ensure(self, am: 'int') -> 'None':
        if self.lo <= am <= self.hi:
            return
        with self._range_lock:
            if self.lo <= am <= self.hi:
                return
            lo = (am // self.mode_block) * self.mode_block
            hi = min(lo + self.mode_block - 1, self.m_max)
            self._build_range(lo, hi)

    def _build_range(self, lo: 'int', hi: 'int') -> 'None':
        ne = self.sp.gen.n_elems
        ord_lo = max(0, lo - 1)
        n_ord = hi + 2 - ord_lo
        self.Z = self.B = None
        self.Z = np.zeros(
            (9, n_ord, self.Np, self.Nq), dtype=self.dtype
        )
        modes = [m for m in range(-hi, hi + 1) if lo <= abs(m) <= hi]
        self._sidx = {m: index for index, m in enumerate(modes)}
        self.B = np.zeros(
            (4, len(modes), self.Np, self.Nq), dtype=self.dtype
        )
        self._ord_lo = ord_lo
        orders = np.arange(ord_lo, hi + 2)
        phase_e = (
            np.exp(1j * np.pi * orders) * (2.0 * np.pi / self._nx_e)
        )
        mode_array = np.asarray(modes)
        bins_b = np.where(
            mode_array >= 0, mode_array, self._nx_b + mode_array
        )
        phase_b = (
            np.exp(1j * np.pi * mode_array) * (2.0 * np.pi / self._nx_b)
        )

        def do_tile(e0):
            e1 = min(e0 + self._te, ne)
            rows = slice(e0 * self.go_p, e1 * self.go_p)
            re = e1 - e0
            Gn = self._sample_G(rows, phase_e, ord_lo, hi)
            self._zero_near(Gn, e0, e1)
            self._accumulate_efie(Gn, rows, e0, re)
            brackets = self._sample_brackets(
                rows, re, bins_b, phase_b
            )
            self._accumulate_brackets(
                brackets, rows, e0, re
            )

        starts = list(range(0, ne, self._te))
        if self._workers <= 1:
            for e0 in starts:
                do_tile(e0)
        else:
            with ThreadPoolExecutor(max_workers=self._workers) as executor:
                list(executor.map(do_tile, starts))
        self.lo, self.hi = lo, hi
        self.n_sweeps += 1

    def _sample_G(self, rows, phase, ord_lo: 'int', hi: 'int'):
        gp = self.sp.g
        xi = 2.0 * np.pi * np.arange(self._nx_e) / self._nx_e - np.pi
        rp = np.ascontiguousarray(gp.rho[rows])
        zp = np.ascontiguousarray(gp.z[rows])
        sin2 = np.ascontiguousarray(np.sin(0.5 * xi) ** 2)
        kept = np.empty(
            (len(rp), self.Pq, hi + 2 - ord_lo), dtype=np.complex128
        )
        for c0 in range(0, self.Pq, self._cols):
            c1 = min(c0 + self._cols, self.Pq)
            cols = slice(c0, c1)
            nc = c1 - c0
            if self._native is not None:
                rho_q = np.ascontiguousarray(self._q[0][cols])
                z_q = np.ascontiguousarray(self._q[1][cols])
                sampled = np.empty(
                    (len(rp), nc, self._nx_e), dtype=np.complex128
                )
                self._native.sample_g(
                    len(rp), nc, self._nx_e, _dp(rp), _dp(zp),
                    _dp(rho_q), _dp(z_q), float(np.real(self.k)),
                    _dp(sin2), _dp(sampled),
                )
            else:
                d2 = (
                    (rp[:, None] - self.sq.g.rho[None, cols]) ** 2
                    + (zp[:, None] - self.sq.g.z[None, cols]) ** 2
                )
                rr4 = 4.0 * rp[:, None] * self.sq.g.rho[None, cols]
                radius = np.sqrt(
                    d2[..., None] + rr4[..., None] * sin2
                )
                radius = np.maximum(radius, 1.0e-300)
                sampled = (
                    np.exp(-1j * self.k * radius)
                    / (4.0 * np.pi * radius)
                )
            spectrum = np.fft.fft(sampled, axis=-1)
            kept[:, cols] = spectrum[..., ord_lo:hi + 2] * phase
        return kept

    def _sample_brackets(self, rows, re: 'int', bins, phase):
        gp, gq = self.sp.g, self.sq.g
        nr = re * self.go_p
        xi = 2.0 * np.pi * np.arange(self._nx_b) / self._nx_b - np.pi
        kept = tuple(
            np.empty((nr, self.Pq, len(bins)), dtype=np.complex128)
            for _ in range(4)
        )
        rp = np.ascontiguousarray(gp.rho[rows])
        zp = np.ascontiguousarray(gp.z[rows])
        trp = np.ascontiguousarray(gp.trho[rows])
        tzp = np.ascontiguousarray(gp.tz[rows])
        cos_xi = np.ascontiguousarray(np.cos(xi))
        sin_xi = np.ascontiguousarray(np.sin(xi))
        for c0 in range(0, self.Pq, self._cols):
            c1 = min(c0 + self._cols, self.Pq)
            cols = slice(c0, c1)
            nc = c1 - c0
            if self._native is not None:
                rho_q, z_q, tr_q, tz_q = (
                    np.ascontiguousarray(value[cols]) for value in self._q
                )
                sampled = tuple(
                    np.empty((nr, nc, self._nx_b), dtype=np.complex128)
                    for _ in range(4)
                )
                self._native.sample_ibc(
                    nr, nc, self._nx_b,
                    _dp(rp), _dp(zp), _dp(trp), _dp(tzp),
                    _dp(rho_q), _dp(z_q), _dp(tr_q), _dp(tz_q),
                    float(np.real(self.k)), _dp(cos_xi), _dp(sin_xi),
                    _dp(sampled[0]), _dp(sampled[1]),
                    _dp(sampled[2]), _dp(sampled[3]),
                )
            else:
                pair_shape = (nr, nc)
                sampled = _ibc_brackets_grid(
                    np.broadcast_to(rp[:, None], pair_shape).ravel(),
                    np.broadcast_to(zp[:, None], pair_shape).ravel(),
                    np.broadcast_to(trp[:, None], pair_shape).ravel(),
                    np.broadcast_to(tzp[:, None], pair_shape).ravel(),
                    np.broadcast_to(gq.rho[None, cols], pair_shape).ravel(),
                    np.broadcast_to(gq.z[None, cols], pair_shape).ravel(),
                    np.broadcast_to(gq.trho[None, cols], pair_shape).ravel(),
                    np.broadcast_to(gq.tz[None, cols], pair_shape).ravel(),
                    self.k,
                    np.broadcast_to(xi, (nr * nc, self._nx_b)),
                )
                sampled = tuple(
                    value.reshape(nr, nc, self._nx_b)
                    for value in sampled
                )
            for uv, values in enumerate(sampled):
                spectrum = np.fft.fft(values, axis=-1)
                kept[uv][:, cols] = (
                    spectrum[..., bins] * (2.0 * np.pi * phase)
                )
        return kept

    def _zero_near(self, values, e0: 'int', e1: 'int') -> 'None':
        for e in range(e0, e1):
            row = slice(
                (e - e0) * self.go_p, (e - e0 + 1) * self.go_p
            )
            for f in self._near_sources[e]:
                values[
                    row, f * self.go_q:(f + 1) * self.go_q
                ] = 0.0

    def _contract_all(self, kernels, left, right, e0: 'int', re: 'int', out):
        ne_q = self.sq.gen.n_elems
        reshaped = kernels.reshape(
            re, self.go_p, ne_q, self.go_q, -1
        )
        left = left.reshape(2, re, self.go_p)
        right = right.reshape(2, ne_q, self.go_q)
        block = np.einsum(
            "aeg,egfhm,bfh->maebf",
            left, reshaped, right, optimize=True,
        )
        if out.dtype != block.dtype:
            block = block.astype(out.dtype)
        with self._acc_lock:
            out[:, e0:e0 + re, 0:ne_q] += block[:, 0, :, 0, :]
            out[:, e0:e0 + re, 1:ne_q + 1] += block[:, 0, :, 1, :]
            out[:, e0 + 1:e0 + re + 1, 0:ne_q] += block[:, 1, :, 0, :]
            out[:, e0 + 1:e0 + re + 1, 1:ne_q + 1] += block[:, 1, :, 1, :]

    def _accumulate_efie(self, kernels, rows, e0: 'int', re: 'int'):
        for index, (left_name, right_name) in enumerate(self._EFIE_COMBOS):
            self._contract_all(
                kernels,
                self._lv[left_name][:, rows],
                self._rv[right_name],
                e0, re, self.Z[index],
            )

    def _accumulate_brackets(self, brackets, rows, e0: 'int', re: 'int'):
        for uv, kernels in enumerate(brackets):
            self._zero_near(kernels, e0, e0 + re)
            self._contract_all(
                kernels,
                self._lv["1"][:, rows],
                self._rv["1"],
                e0, re, self.B[uv],
            )

    def efie_blocks(self, m: 'int'):
        am = abs(m)
        with self._range_lock:
            self._ensure(am)
            order_offset = self._ord_lo

            def primitive(index, order):
                return self.Z[index, order - order_offset].astype(
                    np.complex128
                )

            lower, upper = abs(am - 1), am + 1
            ztt = (
                0.5 * (primitive(0, lower) + primitive(0, upper))
                + primitive(1, am)
                - primitive(5, am) / self.k ** 2
            )
            ztf = (
                (primitive(2, lower) - primitive(2, upper)) / 2j
                - (1j * am / self.k ** 2) * primitive(6, am)
            )
            zft = (
                -(primitive(3, lower) - primitive(3, upper)) / 2j
                + (1j * am / self.k ** 2) * primitive(7, am)
            )
            zff = (
                0.5 * (primitive(4, lower) + primitive(4, upper))
                - (am ** 2 / self.k ** 2) * primitive(8, am)
            )
        if m < 0:
            ztf = -ztf
            zft = -zft
        return ztt, ztf, zft, zff

    def bracket_blocks(self, m: 'int'):
        with self._range_lock:
            self._ensure(abs(m))
            index = self._sidx[m]
            return tuple(
                self.B[uv, index].astype(np.complex128)
                for uv in range(4)
            )

    def memory_gb(self) -> 'float':
        return sum(
            value.nbytes for value in (self.Z, self.B)
            if value is not None
        ) / 1.0e9


def estimate_rectangular_streaming_gb(
    n_test_elements: 'int', n_source_elements: 'int', m_max: 'int',
    has_rotated_pv: 'bool' = True, single_blocks: 'bool' = False,
) -> 'float':
    """All-mode retained blocks for one rectangular EFIE/PV mapping."""

    test_nodes = float(int(n_test_elements) + 1)
    source_nodes = float(int(n_source_elements) + 1)
    modes = int(m_max)
    item_bytes = 8.0 if single_blocks else 16.0
    total = 9.0 * test_nodes * source_nodes * (modes + 2) * item_bytes
    if has_rotated_pv:
        total += (
            4.0 * test_nodes * source_nodes
            * (2 * modes + 1) * item_bytes
        )
    return total / 1.0e9


def estimate_rectangular_streaming_block_gb(
    n_test_elements: 'int', n_source_elements: 'int', m_max: 'int',
    mode_block: 'int', has_rotated_pv: 'bool' = True,
    single_blocks: 'bool' = False,
) -> 'float':
    """Worst retained range for one rectangular EFIE/PV mapping."""

    nt = int(n_test_elements)
    ns = int(n_source_elements)
    mm = int(m_max)
    block = int(mode_block)
    if nt < 1 or ns < 1 or mm < 0 or block < 1 or block > mm + 1:
        raise ValueError("Rectangular streaming estimate dimensions are invalid.")
    node_pairs = float((nt + 1) * (ns + 1))
    item_bytes = 8.0 if single_blocks else 16.0
    worst = 0.0
    for lo in range(0, mm + 1, block):
        hi = min(lo + block - 1, mm)
        order_lo = max(0, lo - 1)
        efie_orders = hi + 2 - order_lo
        signed_modes = (2 * hi + 1) if lo == 0 else 2 * (hi - lo + 1)
        total = 9.0 * efie_orders * node_pairs * item_bytes
        if has_rotated_pv:
            total += 4.0 * signed_modes * node_pairs * item_bytes
        worst = max(worst, total)
    return worst / 1.0e9


def plan_combined_streaming_mode_block(
    m_max: 'int', requirements, stream_budget_gb: 'float', workers: 'int',
) -> 'tuple[int, float, int]':
    """Plan one aligned range shared by several self/cross far streams.

    Each requirement is ``(test_elements, source_elements, has_rotated_pv,
    single_blocks)``.  The returned retained peak is the sum of every stream's
    current block; transient sampling tiles are budgeted separately by the
    caller because streams are constructed sequentially.
    """

    mm = int(m_max)
    budget = float(stream_budget_gb)
    specs = list(requirements)
    if mm < 0 or not specs:
        raise ValueError("Combined streaming planning needs modes and streams.")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("Streaming block budget must be positive and finite.")

    def retained(block):
        return sum(
            estimate_rectangular_streaming_block_gb(
                nt, ns, mm, block, bool(rotated), bool(single)
            )
            for nt, ns, rotated, single in specs
        )

    minimum = retained(1)
    if minimum > budget:
        raise ValueError(
            "Combined streaming block budget is below the modeled one-mode "
            f"retained minimum of {minimum:.6g} GB."
        )
    mode_count = mm + 1
    low, high, max_safe = 1, mode_count, 1
    while low <= high:
        candidate = (low + high) // 2
        if retained(candidate) <= budget:
            max_safe = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    effective_workers = min(max(1, int(workers)), max_safe)
    if max_safe == mode_count:
        aligned = mode_count
    else:
        aligned = (max_safe // effective_workers) * effective_workers
    held = retained(aligned)
    if held > budget:
        raise RuntimeError(
            "Internal combined streaming planner error: aligned block "
            "exceeds its retained-memory budget."
        )
    return aligned, held, effective_workers


def estimate_streaming_gb(n_elems: 'int', m_max: 'int', formulation: 'str' = "cfie",
                          has_ibc: 'bool' = False,
                          single_blocks: 'bool' = False) -> 'float':
    """Persistent per-mode nodal block memory (GB) for the streaming path."""
    Nn = float(n_elems + 1)
    per = 8.0 if single_blocks else 16.0
    # _build_range stores nine EFIE order primitives through m_max + 1
    # inclusive. This estimate drives the memory gate, precision choice, and
    # mode-block sizing, so it must mirror that allocation exactly.
    total = 9.0 * Nn * Nn * (m_max + 2) * per
    if formulation in ("cfie", "mfie"):
        total += 4.0 * Nn * Nn * (2 * m_max + 1) * per
    if has_ibc:
        total += 4.0 * Nn * Nn * (2 * m_max + 1) * per
    return total / 1e9


def estimate_streaming_block_gb(
    n_elems: 'int', m_max: 'int', mode_block: 'int',
    formulation: 'str' = "cfie", has_ibc: 'bool' = False,
    single_blocks: 'bool' = False,
) -> 'float':
    """Worst retained range allocation for an already-aligned mode block.

    This mirrors :meth:`StreamingFarBlocks._build_range`, including the two
    overlapping EFIE neighbor orders and both signs of MFIE/IBC modes.  It is
    deliberately conservative for the diagnostics-only pure-MFIE path in the
    same way as :func:`estimate_streaming_gb`.
    """

    ne = int(n_elems)
    mm = int(m_max)
    block = int(mode_block)
    if ne < 1 or mm < 0 or block < 1 or block > mm + 1:
        raise ValueError("Streaming block estimate dimensions are invalid.")
    nodes = float(ne + 1)
    item_bytes = 8.0 if single_blocks else 16.0
    worst = 0.0
    for lo in range(0, mm + 1, block):
        hi = min(lo + block - 1, mm)
        order_lo = max(0, lo - 1)
        efie_orders = hi + 2 - order_lo
        signed_modes = (2 * hi + 1) if lo == 0 else 2 * (hi - lo + 1)
        total = 9.0 * efie_orders * nodes * nodes * item_bytes
        if formulation in ("cfie", "mfie"):
            total += 4.0 * signed_modes * nodes * nodes * item_bytes
        if has_ibc:
            total += 4.0 * signed_modes * nodes * nodes * item_bytes
        worst = max(worst, total)
    return worst / 1.0e9


def plan_streaming_mode_block(
    n_elems: 'int', m_max: 'int', formulation: 'str', has_ibc: 'bool',
    single_blocks: 'bool', stream_budget_gb: 'float', workers: 'int',
) -> 'tuple[int, float, int]':
    """Return a budget-safe block, retained peak, and effective workers.

    A streaming range cannot be smaller than the number of simultaneously
    solved outer modes: otherwise a worker wave can straddle two ranges and
    force a rebuild while another worker still reads the old range.  Treat
    ``stream_budget_gb`` as a hard retained-block limit by reducing that
    outer concurrency when necessary.  Only a budget below the true
    one-mode retained minimum is rejected.
    """

    budget = float(stream_budget_gb)
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("Streaming block budget must be positive and finite.")
    mode_count = int(m_max) + 1
    minimum = estimate_streaming_block_gb(
        n_elems, m_max, 1, formulation, has_ibc, single_blocks
    )
    if minimum > budget:
        raise ValueError(
            "Streaming block budget is below the modeled one-mode retained "
            f"minimum of {minimum:.6g} GB for this geometry/formulation."
        )

    # The exact retained allocation is monotone in the block size.  Find the
    # largest budget-safe size without walking every mode on large jobs.
    low, high = 1, mode_count
    max_safe = 1
    while low <= high:
        candidate = (low + high) // 2
        retained = estimate_streaming_block_gb(
            n_elems, m_max, candidate, formulation, has_ibc, single_blocks
        )
        if retained <= budget:
            max_safe = candidate
            low = candidate + 1
        else:
            high = candidate - 1

    effective_workers = min(max(1, int(workers)), max_safe)
    if max_safe == mode_count:
        aligned = mode_count
    else:
        # Keep complete outer-worker waves inside one retained range.
        aligned = (max_safe // effective_workers) * effective_workers
    retained = estimate_streaming_block_gb(
        n_elems, m_max, aligned, formulation, has_ibc, single_blocks
    )
    if retained > budget:  # defensive invariant; should be unreachable
        raise RuntimeError(
            "Internal streaming planner error: aligned block exceeds its "
            "retained-memory budget."
        )
    return aligned, retained, effective_workers
