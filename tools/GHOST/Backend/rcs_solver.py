
"""
2D boundary-integral / MoM RCS solver.

High-level workflow:
1) Parse geometry and material definitions into boundary primitives.
2) Build boundary-integral operators (single-layer plus normal-derivative terms).
3) Select the supported Robin, dielectric, multi-region, or sheet
   formulation and solve it with continuous linear Galerkin basis/testing.
4) Post-process the solved boundary unknowns into monostatic far-field RCS.

Notes:
- Uses e^{+j omega t} engineering convention: outgoing Green's function is
  G = (j/4) H_0^(2)(kR), lossy media have eps = eps' - j*eps'' (negative
  imaginary part), and incident plane waves use exp(+j k . r).
- Supports lossy media via complex wavenumber in the coupled formulation.
- Discretization uses continuous two-node linear boundary elements.
"""

import cmath
import csv
import ctypes
import ctypes.util
import math
import os
import subprocess
import sys
import threading
from ghost_runtime import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from solver_metrics import active_metrics, profiled_solve, timed_stage
from thin_sheet import ThinLayerDefinition, layer_for_mesh, solve_thin_layer_fields
from refined_lu import RefinedLU, requested_precision
from cpu_execution import (experimental_monostatic, current_state, requested_cpu,
                           select_formulation, select_solver, EXPERIMENTAL_METHOD)
from geometry_io import (
    material_filename_from_row,
)

# Compatibility imports retain the established module entrypoints.
from rcs_special import (
    _BESSEL,
    _BesselBackend,
    _MPMATH,
    _SCIPY_SPECIAL,
    _complex_hankel_backend_name,
    _hankel2_0,
    _hankel2_1,
    _j0_fallback,
    _j1_fallback,
    _raise_if_untrusted_math_backends,
    _y0_fallback,
    _y1_fallback,
)
from rcs_constants import (
    C0,
    CFIE_ALPHA_DEFAULT,
    DEFAULT_PANELS_PER_WAVELENGTH,
    DENSE_GPU_BACKEND_ENV,
    DENSE_GPU_MIN_N_DEFAULT,
    DENSE_GPU_MIN_N_ENV,
    DENSE_GPU_PROBE_TIMEOUT_S,
    DENSE_LINEAR_BACKWARD_ERROR_MAX,
    EPS,
    ETA0,
    EULER_GAMMA,
    MATERIAL_SINGULAR_TOL,
    MAX_PANELS_DEFAULT,
    MIN_EXPLICIT_PANELS_PER_WAVELENGTH,
    RCS_AMPLITUDE_CONVENTION,
    RCS_AMPLITUDE_VERSION,
    RCS_DB_FLOOR_LINEAR,
    RCS_NORM_MODE_DEFAULT,
    RCS_NORM_MODE_PHYSICAL,
    RCS_NORM_NUMERATOR,
    VIRTUAL_SHEET_REGION_START,
)
from rcs_geometry import (
    ComplexTable,
    ImpedanceTaper,
    LinearElement,
    LinearMesh,
    LinearNode,
    MaterialLibrary,
    MediumTable,
    Panel,
    PanelCoupledInfo,
    _apply_user_convention_flip,
    _build_coupled_panel_info,
    _build_linear_mesh,
    _build_linear_mesh_interface_aware,
    _build_panels,
    _causal_medium_index,
    _check_segment_orientation_or_raise,
    _conservative_mesh_wavelength_for_frequencies,
    _discretize_primitive,
    _ensure_finite_complex,
    _impedance_to_admittance,
    _linear_node_snap_key,
    _linear_panel_signature_from_info,
    _linear_shape_values,
    _load_dielectric_csv,
    _load_impedance_csv,
    _material_base_dir_for_snapshot,
    _medium_eta,
    _medium_n,
    _medium_wavenumber,
    _mesh_wavelength_for_snapshot,
    _normalize_segment_orientation,
    _panel_count_from_n,
    _parse_flag,
    _parse_float,
    _parse_geometry_float,
    _parse_geometry_integer,
    _parse_int,
    _parse_material_definition_flag,
    _parse_material_float,
    _passivity_tolerance,
    _points_close,
    _primitive_length,
    _q_plus_beta,
    _read_csv_numeric_rows,
    _region_medium,
    _resolve_material_file,
    _reverse_point_pairs,
    _safe_complex_div,
    _segment_intersects_strict,
    _snapshot_segments,
    _solver_point_key,
    _surface_robin_alpha,
    _unit_scale_to_meters,
    _validate_passive_medium,
    _validate_passive_surface_impedance,
    validate_geometry_snapshot_for_solver,
)
from rcs_operators import (
    NEAR_PAIR_QUADRATURE_MAX_DEPTH,
    NEAR_PAIR_QUADRATURE_RTOL,
    _ASSEMBLY_COMPACT_BELOW,
    _ASSEMBLY_THREADS,
    _ASSEMBLY_TILE,
    _ASSEMBLY_TILE_TARGET_BYTES,
    _FAR_GRADED,
    _FAR_ORDER_TABLE,
    _FAR_QUAD_ORDER,
    _QUAD_CACHE,
    _QUAD_LOCK,
    _TANGENT_OUTER,
    _assemble_linear_hypersingular_matrix,
    _assemble_linear_mass_matrix,
    _assemble_linear_operator_matrices,
    _assemble_linear_operator_matrices_multi,
    _assemble_linear_weighted_mass_matrix,
    _assembly_tile_size,
    _axpy_into,
    _build_linear_junction_constraints,
    _dgreen_dn_obs_array,
    _dgreen_dn_src_array,
    _ensure_finite_linear_system,
    _env_positive_int,
    _expand_near_chunks,
    _far_green_into,
    _far_hankel1_into,
    _far_kernel_argument,
    _farfield_linear_density_many,
    _get_quadrature,
    _graded_far_order,
    _green_2d,
    _green_2d_array,
    _hankel2_0_array,
    _hankel2_1_array,
    _hypersingular_block_from_s_block,
    _integrate_linear_pair_adaptive_sk,
    _integrate_linear_pair_box,
    _integrate_linear_pair_box_sk_vectorized,
    _integrate_linear_pair_generic,
    _integrate_linear_pair_recursive,
    _integrate_linear_pairs_box_sk_batched,
    _integrate_linear_self_duffy,
    _integrate_linear_touching_duffy,
    _integrate_linear_touching_duffy_sk_vectorized,
    _linear_coupled_interface_signature,
    _linear_coupled_node_report,
    _linear_element_incident_dn_load_many,
    _linear_element_incident_load_many,
    _linear_interval_length,
    _linear_interval_midpoint,
    _linear_interval_point,
    _linear_map_local_to_parent,
    _linear_mass_block,
    _linear_param_to_point,
    _linear_shared_interval_endpoint_info,
    _near_singular_scheme,
    _quadrature_nodes,
    _robin_alpha_elements,
    _run_tiled_obs_blocks,
    _single_layer_block_linear,
    _single_layer_self_block_exact,
    _sk_blocks_near_linear,
    _stable_hankel2_array,
    _warn_far_quadrature_override,
    _wavenumber_is_real,
    get_assembly_threads,
    set_assembly_compaction,
    set_assembly_threads,
    set_far_quadrature_grading,
    set_far_quadrature_order,
)


try:
    from scipy import linalg as _SCIPY_LINALG
except Exception:
    _SCIPY_LINALG = None
try:
    from scipy.sparse import linalg as _SCIPY_SPARSE_LINALG
except Exception:
    _SCIPY_SPARSE_LINALG = None


_DENSE_BACKEND_LOCAL = threading.local()
_CUPY_PROBE_LOCK = threading.Lock()
_CUPY_PROBE_RESULT: 'Optional[Tuple[bool, str]]' = None


def _reset_dense_backend_telemetry() -> 'None':
    _DENSE_BACKEND_LOCAL.events = []


def _record_dense_backend_event(**event: 'Any') -> 'None':
    events = getattr(_DENSE_BACKEND_LOCAL, "events", None)
    if events is None:
        events = []
        _DENSE_BACKEND_LOCAL.events = events
    events.append(dict(event))


def _dense_backend_summary() -> 'Dict[str, Any]':
    events = list(getattr(_DENSE_BACKEND_LOCAL, "events", []) or [])
    used = sorted({str(row.get("used", "cpu")) for row in events})
    reasons = []
    for row in events:
        reason = str(row.get("fallback_reason", "") or "")
        if reason and reason not in reasons:
            reasons.append(reason)
    devices = []
    for row in events:
        device = str(row.get("gpu_device", "") or "")
        if device and device not in devices:
            devices.append(device)
    return {
        "linear_backend": (
            used[0] if len(used) == 1 else ("mixed" if used else "cpu")
        ),
        "dense_gpu_solve_count": sum(
            1 for row in events if row.get("used") == "gpu_cupy"
        ),
        "dense_cpu_solve_count": sum(
            1 for row in events if str(row.get("used", "")).startswith("cpu")
        ),
        "dense_mixed_precision_solve_count": sum(1 for row in events if row.get("used") == "cpu_mixed_lu"),
        "dense_fallback_reasons": reasons,
        "dense_gpu_fallback_reasons": reasons,
        "dense_gpu_devices": devices,
        "dense_largest_system": max(
            [int(row.get("n", 0)) for row in events] or [0]
        ),
    }


def _requested_dense_backend() -> 'Tuple[str, int]':
    if requested_cpu():
        return "cpu", DENSE_GPU_MIN_N_DEFAULT
    backend = os.environ.get(DENSE_GPU_BACKEND_ENV, "cpu").strip().lower()
    if backend not in {"cpu", "auto", "gpu"}:
        raise ValueError(
            f"{DENSE_GPU_BACKEND_ENV} must be cpu, auto, or gpu."
        )
    try:
        minimum_n = int(os.environ.get(
            DENSE_GPU_MIN_N_ENV, str(DENSE_GPU_MIN_N_DEFAULT)
        ))
    except (TypeError, ValueError):
        raise ValueError(
            f"{DENSE_GPU_MIN_N_ENV} must be a positive integer."
        ) from None
    if minimum_n < 1:
        raise ValueError(
            f"{DENSE_GPU_MIN_N_ENV} must be a positive integer."
        )
    return backend, minimum_n


def _probe_cupy_backend() -> 'Tuple[bool, str]':
    """Run one isolated cuSOLVER operation so a bad driver cannot hang GHOST."""

    global _CUPY_PROBE_RESULT
    with _CUPY_PROBE_LOCK:
        if _CUPY_PROBE_RESULT is not None:
            return _CUPY_PROBE_RESULT
        probe = (
            "import cupy as c; "
            "a=c.asarray([[3+0j,1],[1,2]],dtype=c.complex128); "
            "b=c.asarray([1+0j,0]); "
            "x=c.asnumpy(c.linalg.solve(a,b)); "
            "assert abs(x[0]-0.4)<1e-12 and abs(x[1]+0.2)<1e-12"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                universal_newlines=True,
                timeout=DENSE_GPU_PROBE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            _CUPY_PROBE_RESULT = (
                False,
                "CuPy cuSOLVER health probe timed out",
            )
            return _CUPY_PROBE_RESULT
        if completed.returncode != 0:
            detail = str(completed.stderr or "").strip().splitlines()
            suffix = detail[-1] if detail else "unknown CuPy error"
            _CUPY_PROBE_RESULT = (
                False,
                "CuPy cuSOLVER health probe failed: " + suffix,
            )
            return _CUPY_PROBE_RESULT
        _CUPY_PROBE_RESULT = (True, "")
        return _CUPY_PROBE_RESULT


def _solve_dense_gpu(a_eval: 'np.ndarray', rhs_eval: 'np.ndarray'):
    healthy, reason = _probe_cupy_backend()
    if not healthy:
        raise RuntimeError(reason)
    try:
        import cupy as cp
        import cupyx
    except Exception as exc:
        raise RuntimeError(f"CuPy import failed: {exc}") from exc
    free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    # Matrix, RHS, solution, cuSOLVER work area, and allocator fragmentation.
    required_bytes = int(
        4.0 * a_eval.nbytes + 3.0 * rhs_eval.nbytes + 64.0 * 1024.0 ** 2
    )
    if required_bytes > 0.8 * float(free_bytes):
        raise MemoryError(
            "GPU dense solve needs an estimated "
            f"{required_bytes / 1024.0 ** 3:.3f} GiB, exceeding 80% of "
            f"the {float(free_bytes) / 1024.0 ** 3:.3f} GiB currently free."
        )
    try:
        with cupyx.errstate(linalg="raise"):
            gpu_a = cp.asarray(a_eval)
            gpu_b = cp.asarray(rhs_eval)
            gpu_x = cp.linalg.solve(gpu_a, gpu_b)
            solution = cp.asnumpy(gpu_x)
        cp.cuda.Stream.null.synchronize()
        properties = cp.cuda.runtime.getDeviceProperties(
            cp.cuda.runtime.getDevice()
        )
        raw_name = properties.get("name", "CUDA GPU")
        device_name = (
            raw_name.decode(errors="replace")
            if isinstance(raw_name, bytes) else str(raw_name)
        )
    except Exception as exc:
        raise RuntimeError(f"CuPy dense solve failed: {exc}") from exc
    return np.asarray(solution, dtype=np.complex128), device_name


def _canonical_user_polarization_label(label: 'Optional[str]') -> 'str':
    text = str(label or '').strip().upper()
    if text in {'TM', 'HH', 'H', 'HORIZONTAL'}:
        return 'TM'
    if text in {'TE', 'VV', 'V', 'VERTICAL'}:
        return 'TE'
    raise ValueError(f"Unsupported polarization '{label}'. Use TM/TE or VV/HH.")

def _primary_alias_for_user_polarization(label: 'str') -> 'str':
    # Elevation-cut convention: z is horizontal, so E_z (TM) == HH.
    return 'HH' if _canonical_user_polarization_label(label) == 'TM' else 'VV'

def _normalize_polarization(polarization: 'str') -> 'str':
    """
    Normalize user-facing polarization labels without swapping TM and TE.

    Radar-alias convention in this project (2D geometries are elevation cuts,
    out-of-plane z axis is HORIZONTAL):
    - TM, HH, H, HORIZONTAL -> TM  (E along z = horizontal = HH)
    - TE, VV, V, VERTICAL   -> TE  (H along z; E in-plane has vertical component = VV)
    """

    pol = (polarization or "").strip().upper()
    if pol in {"TM", "HH", "H", "HORIZONTAL"}:
        return "TM"
    if pol in {"TE", "VV", "V", "VERTICAL"}:
        return "TE"
    raise ValueError(f"Unsupported polarization '{polarization}'. Use TM/TE or VV/HH.")


def _build_linear_coupled_infos(
    mesh: 'LinearMesh',
    materials: 'MaterialLibrary',
    freq_ghz: 'float',
    pol: 'str',
    k0: 'float',
) -> 'List[PanelCoupledInfo]':
    pseudo_panels = [
        Panel(
            name=e.name,
            seg_type=e.seg_type,
            ibc_flag=e.ibc_flag,
            pos_mat=e.pos_mat,
            neg_mat=e.neg_mat,
            p0=e.p0,
            p1=e.p1,
            center=e.center,
            tangent=e.tangent,
            normal=e.normal,
            length=e.length,
            arc_s_center=float(e.arc_s_center),
        )
        for e in mesh.elements
    ]
    return _build_coupled_panel_info(pseudo_panels, materials, freq_ghz, pol, k0)

def _residual_norm(a_mat: 'np.ndarray', x: 'np.ndarray', b: 'np.ndarray') -> 'float':
    denom = float(np.linalg.norm(b))
    if denom <= EPS:
        denom = 1.0
    return float(np.linalg.norm(a_mat @ x - b) / denom)

def _residual_norm_many(a_mat: 'np.ndarray', x_mat: 'np.ndarray', b_mat: 'np.ndarray') -> 'np.ndarray':
    """Vectorized residual norms for matrix right-hand-sides."""

    x_eval = np.asarray(x_mat)
    b_eval = np.asarray(b_mat)
    if x_eval.ndim == 1:
        return np.asarray([_residual_norm(a_mat, x_eval, b_eval)], dtype=float)

    residual = a_mat @ x_eval - b_eval
    num = np.linalg.norm(residual, axis=0)
    den = np.linalg.norm(b_eval, axis=0)
    den = np.where(den <= EPS, 1.0, den)
    return np.asarray(num / den, dtype=float)

def _summarize_residuals(values: 'List[float]') -> 'Tuple[float, float, int]':
    """Return finite max/mean and the number of non-finite residuals."""

    residuals = np.asarray(values, dtype=float).reshape(-1)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        max_value = float("nan")
        mean_value = float("nan")
    else:
        max_value = float(np.max(finite))
        mean_value = float(np.mean(finite))
    return max_value, mean_value, int(residuals.size - finite.size)


def _equilibrated_scaling_and_norm_1(
    a_mat: 'np.ndarray',
    max_block_bytes: 'int' = 16 * 1024 * 1024,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """Return row/column scales and the equilibrated matrix 1-norm.

    The dense matrix and its LU already dominate solve memory.  Forming
    ``abs(A)``, the row-equilibrated matrix, and a second scaled temporary used
    to add several more full N-by-N arrays during certification.  Two
    column-blocked passes compute the identical scales and column sums while
    bounding the extra real workspace to ``max_block_bytes``.
    """

    a_eval = np.asarray(a_mat, dtype=np.complex128)
    if a_eval.ndim != 2 or a_eval.shape[0] != a_eval.shape[1]:
        raise ValueError("Condition estimation requires a square matrix.")
    n = int(a_eval.shape[0])
    if n < 1:
        raise ValueError("Condition estimation requires a non-empty matrix.")
    workspace_bytes = max(8, int(max_block_bytes))
    block_columns = max(1, min(n, workspace_bytes // (8 * n)))

    row_scale = np.zeros(n, dtype=float)
    for start in range(0, n, block_columns):
        stop = min(n, start + block_columns)
        magnitude = np.abs(a_eval[:, start:stop])
        np.maximum(row_scale, np.max(magnitude, axis=1), out=row_scale)
    row_scale = np.where(row_scale > 0.0, row_scale, 1.0)

    col_scale = np.ones(n, dtype=float)
    norm_a = 0.0
    for start in range(0, n, block_columns):
        stop = min(n, start + block_columns)
        equilibrated = np.abs(a_eval[:, start:stop])
        equilibrated /= row_scale[:, None]
        local_col_scale = np.max(equilibrated, axis=0)
        local_col_scale = np.where(
            local_col_scale > 0.0, local_col_scale, 1.0
        )
        col_scale[start:stop] = local_col_scale
        equilibrated /= local_col_scale[None, :]
        norm_a = max(
            norm_a,
            float(np.max(np.sum(equilibrated, axis=0))),
        )
    return row_scale, col_scale, norm_a


def _equilibrated_condition_from_lu(
    a_mat: 'np.ndarray',
    lu: 'np.ndarray',
    piv: 'np.ndarray',
    solve_override=None,
) -> 'float':
    """Estimate cond_1 of a row/column-equilibrated matrix from one LU.

    The raw block systems mix trace and flux equations with different scales,
    so their unscaled condition numbers are not comparable.  The inverse
    1-norm estimator needs only a handful of LU solves and avoids the second
    cubic SVD that certification previously performed.
    """

    if _SCIPY_LINALG is None or _SCIPY_SPARSE_LINALG is None:
        raise RuntimeError("equilibrated condition estimation requires SciPy")
    a_eval = np.asarray(a_mat, dtype=np.complex128)
    row_scale, col_scale, norm_a = _equilibrated_scaling_and_norm_1(a_eval)
    n = int(a_eval.shape[0])

    def _inverse_matvec(vector):
        rhs = row_scale * np.asarray(
            vector, dtype=np.complex128
        ).reshape(-1)
        solved = solve_override(rhs) if solve_override is not None else _SCIPY_LINALG.lu_solve((lu, piv), rhs)
        return col_scale * solved

    def _inverse_rmatvec(vector):
        rhs = col_scale * np.asarray(
            vector, dtype=np.complex128
        ).reshape(-1)
        solved = solve_override(rhs, trans=2) if solve_override is not None else _SCIPY_LINALG.lu_solve((lu, piv), rhs, trans=2)
        return row_scale * solved

    inverse = _SCIPY_SPARSE_LINALG.LinearOperator(
        (n, n), matvec=_inverse_matvec, rmatvec=_inverse_rmatvec,
        dtype=np.complex128,
    )
    inverse_norm = float(_SCIPY_SPARSE_LINALG.onenormest(inverse))
    estimate = norm_a * inverse_norm
    return estimate if math.isfinite(estimate) else float("inf")


@timed_stage("linear_solve")
def _solve_dense_system(
    a_mat: 'np.ndarray',
    rhs: 'np.ndarray',
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
    label: 'str' = "dense system",
) -> 'np.ndarray':
    """Factor once, solve all RHS columns, and optionally estimate condition."""

    a_eval = np.asarray(a_mat, dtype=np.complex128)
    rhs_eval = np.asarray(rhs, dtype=np.complex128)
    requested_backend, gpu_min_n = _requested_dense_backend()
    system_n = int(a_eval.shape[0]) if a_eval.ndim == 2 else 0
    backend_used = "cpu"
    fallback_reason = ""
    gpu_device = ""
    lu = piv = None
    solution = None
    gpu_candidate = requested_backend in {"auto", "gpu"}
    if requested_precision() == "mixed":
        if requested_backend == "gpu":
            raise ValueError("Mixed LU currently requires the CPU backend; choose double precision for GPU solves.")
        gpu_candidate = False
    if gpu_candidate and condition_diagnostics is not None:
        fallback_reason = "CPU LU required for certified condition estimation"
        gpu_candidate = False
    if (
        gpu_candidate
        and requested_backend == "auto"
        and system_n < gpu_min_n
    ):
        fallback_reason = (
            f"system order {system_n} is below the auto-GPU threshold "
            f"{gpu_min_n}"
        )
        gpu_candidate = False
    if gpu_candidate:
        try:
            solution, gpu_device = _solve_dense_gpu(a_eval, rhs_eval)
            backend_used = "gpu_cupy"
        except Exception as exc:
            fallback_reason = str(exc)
            if requested_backend == "gpu":
                _record_dense_backend_event(
                    requested=requested_backend,
                    used="gpu_failed",
                    n=system_n,
                    label=str(label),
                    fallback_reason=fallback_reason,
                )
                raise RuntimeError(
                    f"{label} requested the GPU backend but it was not "
                    f"usable: {fallback_reason}"
                ) from exc

    if solution is None and requested_precision() == "mixed" and _SCIPY_LINALG is not None:
        try:
            factor = RefinedLU(a_eval)
            solution = factor.solve(rhs_eval)
            if condition_diagnostics is not None:
                estimate = _equilibrated_condition_from_lu(a_eval, factor.lu, factor.piv, factor.solve)
                condition_diagnostics.update(condition_est=estimate, condition_method="equilibrated_1norm_refined_inverse", mixed_precision_corrections=factor.max_corrections)
            backend_used = "cpu_mixed_lu"
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning) as exc:
            solution = None
            fallback_reason = f"Mixed precision fell back to double LU: {exc}"

    if solution is None and _SCIPY_LINALG is None:
        solution = np.linalg.solve(a_eval, rhs_eval)
        if condition_diagnostics is not None:
            # Production environments require SciPy, but keep a fail-closed
            # diagnostic fallback for minimal installations.
            row = np.max(np.abs(a_eval), axis=1)
            row = np.where(row > 0.0, row, 1.0)
            row_eq = a_eval / row[:, None]
            col = np.max(np.abs(row_eq), axis=0)
            col = np.where(col > 0.0, col, 1.0)
            try:
                estimate = float(np.linalg.cond(row_eq / col[None, :], p=1))
            except np.linalg.LinAlgError:
                estimate = float("inf")
            condition_diagnostics["condition_est"] = estimate
            condition_diagnostics["condition_method"] = (
                "equilibrated_1norm_numpy_fallback"
            )
    elif solution is None:
        lu, piv = timed_stage("factorization")(_SCIPY_LINALG.lu_factor)(a_eval)
        solution = timed_stage("rhs_solve")(_SCIPY_LINALG.lu_solve)((lu, piv), rhs_eval)
        if condition_diagnostics is not None:
            condition_diagnostics["condition_est"] = (
                _equilibrated_condition_from_lu(a_eval, lu, piv)
            )
            condition_diagnostics["condition_method"] = (
                "equilibrated_1norm_lu_onenormest"
            )

    def backward_metrics(candidate):
        x_columns = np.asarray(candidate, dtype=np.complex128)
        b_columns = rhs_eval
        if x_columns.ndim == 1:
            x_columns = x_columns[:, None]
            b_columns = b_columns[:, None]
        residual = a_eval @ x_columns - b_columns
        residual_inf = np.max(np.abs(residual), axis=0)
        matrix_inf = float(np.linalg.norm(a_eval, ord=np.inf))
        solution_inf = np.max(np.abs(x_columns), axis=0)
        rhs_inf = np.max(np.abs(b_columns), axis=0)
        denominator = matrix_inf * solution_inf + rhs_inf
        errors = np.divide(
            residual_inf,
            denominator,
            out=np.zeros_like(residual_inf, dtype=float),
            where=denominator > 0.0,
        )
        errors[(denominator <= 0.0) & (residual_inf > 0.0)] = math.inf
        return residual, float(np.max(errors))

    residual_matrix, backward_error = backward_metrics(solution)
    refinement_steps = 0
    for _attempt in range(2):
        if backward_error <= DENSE_LINEAR_BACKWARD_ERROR_MAX:
            break
        correction_rhs = -residual_matrix
        if np.asarray(solution).ndim == 1:
            correction_rhs = correction_rhs[:, 0]
        correction = (
            np.linalg.solve(a_eval, correction_rhs)
            if lu is None
            else _SCIPY_LINALG.lu_solve((lu, piv), correction_rhs)
        )
        candidate = solution + correction
        candidate_residual, candidate_error = backward_metrics(candidate)
        if candidate_error >= backward_error:
            break
        solution = candidate
        residual_matrix = candidate_residual
        backward_error = candidate_error
        refinement_steps += 1

    if not math.isfinite(backward_error):
        raise RuntimeError(
            f"{label} produced a non-finite normwise backward error."
        )
    if backward_error > DENSE_LINEAR_BACKWARD_ERROR_MAX:
        raise RuntimeError(
            f"{label} normwise backward error {backward_error:.6g} exceeds "
            f"the release limit {DENSE_LINEAR_BACKWARD_ERROR_MAX:.6g}."
        )
    if condition_diagnostics is not None:
        condition_diagnostics["condition_label"] = str(label)
        condition_diagnostics["linear_backward_error"] = float(
            backward_error
        )
        condition_diagnostics["linear_backward_error_limit"] = float(
            DENSE_LINEAR_BACKWARD_ERROR_MAX
        )
        condition_diagnostics["linear_refinement_steps"] = int(
            refinement_steps
        )
    _record_dense_backend_event(
        requested=requested_backend,
        used=backend_used,
        n=system_n,
        label=str(label),
        fallback_reason=fallback_reason,
        gpu_device=gpu_device,
        refinement_steps=int(refinement_steps),
    )
    return np.asarray(solution, dtype=np.complex128)


def _consume_condition_estimate(
    values: 'List[float]',
    diagnostics: 'Optional[Dict[str, Any]]',
    label: 'str',
) -> 'None':
    """Append a requested estimate, refusing a silently unimplemented path."""

    if diagnostics is None:
        return
    if "condition_est" not in diagnostics:
        raise RuntimeError(
            f"{label} did not produce the requested condition-number "
            "diagnostic; no field is returned."
        )
    values.append(float(diagnostics["condition_est"]))

def _normalize_rcs_normalization_mode(mode: 'Optional[str]') -> 'str':
    """Accept only physical sigma_2d normalization aliases."""

    text = str(mode or "").strip().lower().replace("-", "_")
    if text in {"", "physical", "divide_by_k", "with_k", "k", "derived", "width", "sigma_2d"}:
        return RCS_NORM_MODE_PHYSICAL
    raise ValueError(
        f"Unsupported rcs_normalization_mode '{mode}'. This solver now supports only physical normalization "
        "sigma_2d = |A|^2 / (4k)."
    )

def _normalize_public_2d_solver_method(method: 'Any') -> 'str':
    """Accept the supported direct methods before allocating solver arrays."""

    normalized = str(method).strip().lower()
    if normalized == "fmm":
        raise ValueError(
            "The FMM solver has been removed. Use solver_method='auto' or "
            "'direct' for dense LU."
        )
    if normalized not in {"auto", "direct", EXPERIMENTAL_METHOD}:
        raise ValueError(
            f"Unsupported 2-D solver_method {method!r}; expected 'auto', 'direct', or 'experimental_cpu'."
        )
    return normalized


def _validate_disabled_2d_cfie_alpha(value: 'Any') -> 'float':
    """Require the exact disabled value for the unimplemented 2-D CFIE knob.

    A tolerance check is inappropriate for an algorithm selector: accepting a
    tiny nonzero value would silently ignore a requested formulation change.
    ``NaN`` also compares false to ordinary magnitude thresholds, so validate
    finiteness explicitly before any geometry or operator work begins.
    """

    try:
        alpha = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "cfie_alpha is not implemented by any active 2-D formulation; "
            "use the exact disabled value cfie_alpha=0."
        ) from exc
    if not math.isfinite(alpha) or alpha != 0.0:
        raise ValueError(
            "cfie_alpha is not implemented by any active 2-D formulation; "
            "use the exact disabled value cfie_alpha=0. No unchanged field "
            "was returned under a different solver setting."
        )
    return 0.0

def _rcs_sigma_from_amp(
    amp_vec: 'np.ndarray',
    k_value: 'float',
) -> 'np.ndarray':
    """
    Apply physical 2D scattering-width normalization to the far-field amplitude.

    Linear scattering width is not presentation data: exact zeros and finite
    deep nulls are retained.  Only conversion to dB applies a display floor.
    """

    amp_eval = np.asarray(amp_vec, dtype=np.complex128)
    if not np.all(np.isfinite(amp_eval.real) & np.isfinite(amp_eval.imag)):
        raise FloatingPointError("Far-field amplitude contains non-finite value(s).")
    k_eval = float(k_value)
    if not math.isfinite(k_eval) or k_eval <= 0.0:
        raise ValueError(f"RCS normalization requires positive finite k; got {k_value!r}.")
    scale = float(RCS_NORM_NUMERATOR) / k_eval
    sigma_lin = scale * (np.abs(amp_eval) ** 2)
    if not np.all(np.isfinite(sigma_lin)):
        raise FloatingPointError("Computed linear scattering width contains non-finite value(s).")
    return np.asarray(sigma_lin, dtype=float)

def _rcs_db_from_sigma(
    sigma_linear: 'Union[float, np.ndarray]',
    floor_linear: 'float' = RCS_DB_FLOOR_LINEAR,
) -> 'np.ndarray':
    """Convert non-negative linear RCS to display dB with a display-only floor."""

    sigma = np.asarray(sigma_linear, dtype=float)
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError("Linear RCS must contain finite non-negative values.")
    floor_eval = float(floor_linear)
    if not math.isfinite(floor_eval) or floor_eval <= 0.0:
        raise ValueError("RCS dB display floor must be positive and finite.")
    return np.asarray(10.0 * np.log10(np.maximum(sigma, floor_eval)), dtype=float)


def evaluate_quality_gate(
    metadata: 'Dict[str, Any]',
    thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
) -> 'Dict[str, Any]':
    """
    Evaluate a lightweight numeric quality gate from solver metadata.

    This does not prove correctness; it catches obvious numerical-risk runs.
    """

    defaults: 'Dict[str, Union[float, int]]' = {
        "residual_norm_max": 1.0e-6,
        "constraint_residual_norm_max": 1.0e-8,
        "condition_est_max": 1.0e6,
        "warnings_max": 10,
    }
    merged = dict(defaults)
    if thresholds:
        supplied = dict(thresholds)
        unknown = sorted(set(supplied) - set(defaults))
        if unknown:
            raise ValueError(
                "Unknown 2-D quality threshold field(s): "
                + ", ".join(str(key) for key in unknown)
            )
        merged.update(supplied)

    residual_limit = float(merged.get("residual_norm_max", defaults["residual_norm_max"]))
    constraint_limit = float(merged.get("constraint_residual_norm_max", defaults["constraint_residual_norm_max"]))
    condition_limit = float(merged.get("condition_est_max", defaults["condition_est_max"]))
    warnings_raw = merged.get("warnings_max", defaults["warnings_max"])
    try:
        warnings_float = float(warnings_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("warnings_max must be a finite non-negative integer.") from exc
    if (
        not math.isfinite(residual_limit)
        or residual_limit < 0.0
        or not math.isfinite(constraint_limit)
        or constraint_limit < 0.0
        or not math.isfinite(condition_limit)
        or condition_limit < 0.0
    ):
        raise ValueError(
            "2-D residual and condition quality thresholds must be finite "
            "non-negative values."
        )
    if (
        not math.isfinite(warnings_float)
        or warnings_float < 0.0
        or not warnings_float.is_integer()
    ):
        raise ValueError("warnings_max must be a finite non-negative integer.")
    warnings_limit = int(warnings_float)

    residual_raw = metadata.get("residual_norm_max")
    try:
        residual_value = (
            float(residual_raw) if residual_raw is not None else float("nan")
        )
    except (TypeError, ValueError):
        residual_value = float("nan")
    residual_nonfinite_raw = metadata.get("residual_nonfinite_count", 0)
    try:
        residual_nonfinite_count = int(residual_nonfinite_raw)
    except (TypeError, ValueError, OverflowError):
        residual_nonfinite_count = -1
    constraint_value = float(metadata.get("constraint_residual_norm_max", 0.0) or 0.0)
    condition_raw = metadata.get("condition_est_max")
    try:
        condition_value = float(condition_raw) if condition_raw is not None else float("nan")
    except (TypeError, ValueError):
        condition_value = float("nan")
    if "condition_est_computed" in metadata:
        condition_computed = bool(metadata.get("condition_est_computed"))
    else:
        # Backward-compatible inference for older result dictionaries: a
        # finite estimate is usable, while a missing/NaN placeholder means
        # the condition number was not computed and must not fail the gate.
        condition_computed = math.isfinite(condition_value)
    warnings_count = len(list(metadata.get("warnings", []) or []))

    violations: 'List[str]' = []
    if not math.isfinite(residual_value) or residual_value > residual_limit:
        violations.append(
            f"residual_norm_max={residual_value:.6g} exceeds limit {residual_limit:.6g}"
        )
    if residual_nonfinite_count != 0:
        if residual_nonfinite_count > 0:
            violations.append(
                f"residual_nonfinite_count={residual_nonfinite_count} must be zero"
            )
        else:
            violations.append(
                "residual_nonfinite_count is missing a valid non-negative integer value"
            )
    if bool(metadata.get("junction_constraints_applied", False)) and (
        (not math.isfinite(constraint_value)) or constraint_value > constraint_limit
    ):
        violations.append(
            f"constraint_residual_norm_max={constraint_value:.6g} exceeds limit {constraint_limit:.6g}"
        )
    if condition_computed and (not math.isfinite(condition_value) or condition_value > condition_limit):
        violations.append(
            f"condition_est_max={condition_value:.6g} exceeds limit {condition_limit:.6g}"
        )
    if warnings_count > warnings_limit:
        violations.append(
            f"warnings_count={warnings_count} exceeds limit {warnings_limit}"
        )

    return {
        "passed": len(violations) == 0,
        "thresholds": {
            "residual_norm_max": residual_limit,
            "constraint_residual_norm_max": constraint_limit,
            "condition_est_max": condition_limit,
            "warnings_max": warnings_limit,
        },
        "values": {
            "residual_norm_max": residual_value,
            "residual_nonfinite_count": residual_nonfinite_count,
            "constraint_residual_norm_max": constraint_value,
            "condition_est_max": condition_value,
            "condition_est_computed": condition_computed,
            "warnings_count": warnings_count,
        },
        "violations": violations,
        "certification_scope": (
            "discrete_linear_system_residual_and_condition"
            if condition_computed
            else "discrete_linear_system_residual_only"
        ),
        "mesh_convergence_certified": bool(
            metadata.get("mesh_convergence_certified", False)
        ),
        "reason": (
            "; ".join(violations)
            if violations
            else (
                "discrete linear-system quality thresholds satisfied; "
                + (
                    "condition number was not requested; "
                    if not condition_computed else ""
                )
                + "mesh convergence is separately certified by the production workflow"
            )
        ),
    }


def _is_all_robin(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """Return True if every element uses a Robin BC (PEC or IBC, no dielectric)."""
    return all(info.bc_kind == 'robin' for info in infos)

def _assert_supported_te_type2_contours(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
) -> 'None':
    """
    Reject open TYPE 2 contours before applying a closed-obstacle TE MFIE.

    Geometric endpoint keys are used instead of linear node IDs because the
    interface-aware mesh deliberately splits a shared node when two stitched
    TYPE 2 segments use different IBC flags. Such stitched contours are still
    physically closed and must remain supported.
    """

    if pol != "TE" or not _is_all_robin(infos):
        return

    type2_degree: 'Dict[Tuple[int, int], int]' = {}
    for elem, info in zip(mesh.elements, infos):
        if int(info.seg_type) != 2:
            continue
        for nid in elem.node_ids:
            key = mesh.nodes[int(nid)].key
            type2_degree[key] = type2_degree.get(key, 0) + 1

    open_endpoint_count = sum(1 for degree in type2_degree.values() if degree == 1)
    if open_endpoint_count > 0:
        raise ValueError(
            "Open TYPE 2 PEC/IBC contours are not supported for TE polarization: "
            "the available TE Robin MFIE is a closed-obstacle formulation and "
            f"the geometry has {open_endpoint_count} open TYPE 2 endpoint(s). "
            "Close/stitch the obstacle contour, or use a physically appropriate "
            "TYPE 1 sheet model for an open impedance card."
        )

_BYTES_PER_GIB = 1024.0 ** 3


def _psutil_available_bytes() -> 'Optional[int]':
    """Host-available bytes from psutil, without making it mandatory."""

    try:
        import psutil

        value = int(psutil.virtual_memory().available)
    except Exception:
        return None
    return value if value >= 0 else None


def _windows_available_bytes() -> 'Optional[int]':
    """Windows ``ullAvailPhys`` fallback when psutil is unavailable."""

    if os.name != "nt":
        return None

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    try:
        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
            ctypes.byref(status)
        )
    except Exception:
        return None
    if not success:
        return None
    return int(status.ullAvailPhys)


def _posix_available_bytes() -> 'Optional[int]':
    """Linux/proc and POSIX sysconf availability fallbacks."""

    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return max(0, int(line.split()[1]) * 1024)
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    value = pages * page_size
    return value if value >= 0 else None


def _process_rss_bytes() -> 'int':
    """Best-effort resident memory used inside a scheduler allocation."""

    try:
        import psutil

        return max(0, int(psutil.Process(os.getpid()).memory_info().rss))
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as stream:
            resident_pages = int(stream.read().split()[1])
        return max(0, resident_pages * int(os.sysconf("SC_PAGE_SIZE")))
    except (AttributeError, OSError, TypeError, ValueError, IndexError):
        return 0


def _slurm_available_bytes() -> 'Optional[int]':
    """Remaining bytes in the active SLURM task allocation, if declared."""

    capacity_mb = None
    raw = os.environ.get("SLURM_MEM_PER_NODE", "").strip()
    if raw.isdigit() and int(raw) > 0:
        capacity_mb = int(raw)
    else:
        raw = os.environ.get("SLURM_MEM_PER_CPU", "").strip()
        cpus_text = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
        # Slurm's documented default is one CPU per task when
        # SLURM_CPUS_PER_TASK is not exported.  Ignoring MEM_PER_CPU in that
        # common case would let a desktop/HPC solve use host memory beyond its
        # allocation.  An explicitly malformed/non-positive CPU count remains
        # untrusted and is not guessed.
        if raw.isdigit() and int(raw) > 0:
            if not cpus_text:
                cpus = 1
            elif cpus_text.isdigit() and int(cpus_text) > 0:
                cpus = int(cpus_text)
            else:
                cpus = None
            if cpus is not None:
                capacity_mb = int(raw) * cpus
    if capacity_mb is None:
        return None
    capacity = capacity_mb * 1024 * 1024
    return max(0, capacity - _process_rss_bytes())


def _read_cgroup_int(path: 'str') -> 'Optional[int]':
    try:
        with open(path) as stream:
            text = stream.read().strip()
    except OSError:
        return None
    if not text.isdigit():
        return None
    value = int(text)
    # cgroup v1 represents "unlimited" with a huge integer near LONG_MAX.
    if value < 0 or value >= 2 ** 60:
        return None
    return value


def _cgroup_available_bytes() -> 'Optional[int]':
    """Remaining bytes under a cgroup v2 or v1 memory limit."""

    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        limit = _read_cgroup_int(limit_path)
        if limit is None:
            continue
        usage = _read_cgroup_int(usage_path)
        if usage is None:
            # A capacity without current usage is not actionable headroom.
            # Treat an unreadable constrained cgroup as exhausted instead of
            # silently permitting the full limit as one new allocation.
            return 0
        return max(0, limit - usage)
    return None


def _detect_available_gb() -> 'float':
    """Memory this process may safely allocate now, in GiB.

    ``psutil`` is preferred on desktops.  Native Windows/POSIX
    fallbacks keep the GUI safe without the optional dependency.  Scheduler
    and cgroup headroom are additional bounds, so the tightest known limit
    wins instead of a machine-total or historical fixed allowance.
    """

    host_available = _psutil_available_bytes()
    if host_available is None:
        host_available = _windows_available_bytes()
    if host_available is None:
        host_available = _posix_available_bytes()

    bounds = []
    if host_available is not None:
        bounds.append(max(0, host_available))
    slurm_available = _slurm_available_bytes()
    if slurm_available is not None:
        bounds.append(max(0, slurm_available))
    cgroup_available = _cgroup_available_bytes()
    if cgroup_available is not None:
        bounds.append(max(0, cgroup_available))
    if not bounds:
        return 0.0
    return float(min(bounds)) / _BYTES_PER_GIB


# Hard ceiling on one solve's estimated dense footprint.
#
# A solve may use only a fraction of currently available memory so the GUI,
# plotting stack, and operating system retain headroom.  If detection fails,
# the limit is zero and dense allocation fails closed; an informed user or
# scheduler may provide an explicit GHOST_MAX_SOLVE_GB reservation.
_MEMORY_LIMIT_FRACTION = 0.9


def _solve_memory_limit_gb() -> 'float':
    override = os.environ.get("GHOST_MAX_SOLVE_GB", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            value = 0.0
        if math.isfinite(value) and value > 0.0:
            return value
    detected = _detect_available_gb()
    return _MEMORY_LIMIT_FRACTION * detected if detected > 0.0 else 0.0


def _memory_gate_message(
    required_gb: 'float',
    limit_gb: 'float',
    context: 'str',
    details: 'str' = "",
    remedies: 'str' = "Reduce the mesh size, frequency, or solve scope.",
) -> 'str':
    """Build a required-versus-available, actionable allocation error."""

    available_gb = _detect_available_gb()
    if available_gb > 0.0:
        availability = f"{available_gb:.2f} GB is currently available"
    else:
        availability = "available memory could not be detected"
    detail_text = f" {details.strip()}" if details.strip() else ""
    return (
        f"{context} requires an estimated {required_gb:.2f} GB, but "
        f"{availability}; the safe allocation limit is {limit_gb:.2f} GB."
        f"{detail_text} {remedies.strip()} If a larger allocation is confirmed, "
        "set GHOST_MAX_SOLVE_GB to that explicit per-process limit and retry."
    )


def _estimate_memory_gb(
    nnodes: 'int',
    use_cfie: 'bool',
    n_regions: 'int' = 1,
    system_dofs: 'Optional[int]' = None,
    operator_matrices: 'Optional[int]' = None,
    n_rhs: 'int' = 1000,
    solver_method: 'str' = 'direct',
    formulation: 'Optional[str]' = None,
) -> 'float':
    """
    Estimate peak memory for the dense BIE/MoM solve in GB.

    Accounts for: system matrix, region operators, RHS, solution, factorization.
    """

    bytes_per_complex = 16  # complex128
    # System matrix + factorization copy.  The historical default is a 2N
    # coupled system; formulation-aware planning supplies the active system
    # dimension (N for sheet/Robin, 2N for a single dielectric, and the exact
    # interface-side DOF count for multi-region geometries).
    sys_size = (
        2 * nnodes if system_dofs is None else max(1, int(system_dofs))
    )
    sys_bytes = 2 * sys_size * sys_size * bytes_per_complex
    # Dense global operators retained while the system is formed.  Callers
    # that know the formulation provide the actual conservative count.
    if operator_matrices is None:
        ops_per_region = 4 if not use_cfie else 8
        operator_matrices = max(1, int(n_regions)) * ops_per_region
    region_bytes = (
        max(0, int(operator_matrices))
        * nnodes * nnodes * bytes_per_complex
    )
    # RHS + solution
    misc_bytes = 4 * sys_size * bytes_per_complex * max(1, int(n_rhs))
    extra_bytes = 0
    if solver_method == EXPERIMENTAL_METHOD:
        from cpu_execution import BATCH_SIZE, CACHE_BYTES, TABLE_BYTES, STREAMED_FORMULATIONS
        # Cache/table retention can overlap a later fallback or refined solve.
        extra_bytes = CACHE_BYTES + TABLE_BYTES
        if formulation in STREAMED_FORMULATIONS:
            misc_bytes = 8 * sys_size * bytes_per_complex * min(BATCH_SIZE, max(1, int(n_rhs)))
            # Compact arrays plus canonical Python result records for both channels.
            extra_bytes += max(1, int(n_rhs)) * 4096
    total = sys_bytes + region_bytes + misc_bytes + extra_bytes
    return total / (1024 ** 3)


def _solve_te_robin_mfie(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    solver_method: 'str' = "auto",
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
    operator_cache: 'Optional[Dict[Any, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve TE Robin (PEC or IBC) problems via a generalized MFIE.

    Uses the single-layer potential representation u_scat = SLP(sigma).
    The exterior-limit Robin BC gives:

        (-1/2 M + K' + alpha.S) sigma = -(du_inc/dn + alpha.u_inc)

    where alpha is retained as a piecewise-constant element coefficient inside
    the Galerkin observation integral (0 for PEC, nonzero for IBC).
    K' is the adjoint double-layer operator (obs_normal_deriv=True).

    Uses dense LU for both solver_method="auto" and "direct".

    Returns (rcs_linear, amplitude, residual_norm) arrays over elevations.
    """
    _normalize_public_2d_solver_method(solver_method)

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    # The impedance model is sampled at element centers, so alpha is a
    # piecewise-constant coefficient in the discrete weak form.  Keep it
    # inside the observation and RHS element integrals; nodal row scaling is
    # not a Galerkin treatment when alpha varies spatially.
    alpha_elements, _ = _robin_alpha_elements(mesh, infos, pol)
    has_ibc = bool(np.any(np.abs(alpha_elements) > EPS))

    # RHS: -(du_inc/dn + alpha * u_inc)
    rhs_mfie = np.zeros((nnodes, elev.size), dtype=np.complex128)
    for eidx, elem in enumerate(mesh.elements):
        ids = np.asarray(elem.node_ids, dtype=int)
        load_dn = _linear_element_incident_dn_load_many(elem, k_air=k0, elevations_deg=elev)
        rhs_mfie[ids, :] -= load_dn
        if has_ibc:
            load_u = _linear_element_incident_load_many(elem, k_air=k0, elevations_deg=elev)
            rhs_mfie[ids, :] -= complex(alpha_elements[eidx]) * load_u

    s_alpha_mat, kp_mat = _assemble_linear_operator_matrices(
        mesh, k0, obs_normal_deriv=True,
        obs_order=obs_order, src_order=src_order,
        compute_single_layer=bool(has_ibc),
        single_layer_observation_coefficients=(
            alpha_elements if has_ibc else None
        ),
    )
    if operator_cache is not None and has_ibc:
        cache_key = (
            "air_kp",
            id(mesh),
            complex(k0),
            int(obs_order),
            int(src_order),
        )
        if cache_key not in operator_cache:
            operator_cache[cache_key] = kp_mat
            operator_cache["_stores"] = int(
                operator_cache.get("_stores", 0)
            ) + 1
    mass_mat = _assemble_linear_mass_matrix(mesh)
    a_mfie = -0.5 * mass_mat + kp_mat
    if has_ibc:
        a_mfie += s_alpha_mat
    _ensure_finite_linear_system(a_mfie, rhs_mfie, label="TE Robin MFIE system")
    sigma_mat = _solve_dense_system(
        a_mfie, rhs_mfie, condition_diagnostics,
        "TE Robin MFIE system",
    )
    residual = np.linalg.norm(a_mfie @ sigma_mat - rhs_mfie, axis=0)

    rhs_norm = np.linalg.norm(rhs_mfie, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, sigma_mat, k0, elev, "SLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k0)
    return rcs_lin, amp, float(np.max(residual_vec))

def _has_sheet(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """True if any element is a TYPE 1 free-floating resistive/reactive sheet.

    Sheets carry their impedance via q_plus_gamma = 1/Z_s, and correctly
    modelling them requires a formulation that uses that term.  The
    dielectric-indirect and multi-region-indirect solvers do not -- and
    neither does the current coupled trace formulation, which also has
    pre-existing sign/normalization issues in the sheet case that produce
    unphysical results.

    The public RCS dispatch routes all-sheet and sheet + pure-PEC geometries
    to dedicated sheet solvers. It rejects TYPE 1 mixed with an IBC body,
    dielectric body, or layered coating rather than silently sending that
    combination through an operator that omits the sheet admittance. For a
    tapered resistance treatment on a conducting body, use TYPE 2 with a
    tapered IBC instead--that path is validated.
    """
    return any(int(info.seg_type) == 1 for info in infos)


def _assert_air_exterior(infos: 'List[PanelCoupledInfo]') -> 'None':
    """
    Reject geometries with no air-facing boundary.

    Every formulation in this solver poses the scattering problem in a free
    space background: the incident plane wave, the exterior Green's function,
    and the far-field projection all use the air wavenumber k0.  A geometry
    whose boundaries never touch region 0 (e.g. a TYPE 5-only contour with
    dielectric on BOTH sides) describes a non-air background, which the
    dispatch predicates would otherwise mis-capture: `_solve_dielectric_
    indirect` would silently treat the outer dielectric as air and solve a
    different problem.
    """

    for info in infos:
        if info.minus_region == 0 or info.plus_region == 0:
            return
    raise ValueError(
        "Geometry has no air-facing boundary: every interface separates "
        "non-air media (e.g. a TYPE 5 dielectric/dielectric contour with no "
        "enclosing TYPE 2/3 boundary). This solver poses scattering in a "
        "free-space background, so the unbounded exterior region must be "
        "air -- add the body's outer air boundary (TYPE 2/3), or model the "
        "background medium explicitly as an enclosing region."
    )


def _is_single_dielectric_body(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """Return True if every element is a transmission interface (TYPE 3 dielectric).

    Excludes TYPE 1 sheets: those have bc_kind == 'transmission' but their
    impedance semantics live in q_plus_gamma, which the dielectric-indirect
    solver does not evaluate.  In the RCS dispatch, supported sheet
    geometries are handled by the dedicated sheet solvers (and unsupported
    mixes rejected by ``_assert_no_type1_sheet_for_mixed``) before this
    predicate, so it should never see a sheet; the ``_has_sheet`` guard
    below is belt-and-braces.
    """
    if _has_sheet(infos):
        return False
    return all(info.bc_kind == 'transmission' for info in infos)


def _assert_no_type1_sheet_for_mixed(infos: 'List[PanelCoupledInfo]') -> 'None':
    """Reject mixed TYPE 1 + non-sheet geometries that aren't supported.

    Supported geometries containing TYPE 1 sheets:
      - All-sheet (solved by _solve_tm_sheet / _solve_te_sheet)
      - Sheet + pure-PEC TYPE 2 body (solved by _solve_mixed_sheet_pec)

    Everything else -- sheet + IBC-coated body, sheet + dielectric body,
    sheet + layered coating -- still needs bespoke coupling work and is
    rejected.  The error message points at the workaround (TYPE 2 with
    tapered IBC for edge treatments on bodies).
    """
    if any(info.bc_kind == "thin_layer" for info in infos) and not all(
            info.bc_kind == "thin_layer" for info in infos):
        raise ValueError("Thin dielectric layers currently require an all-layer geometry; coupling to other boundary models is not implemented.")
    if _has_sheet(infos) and not _is_all_sheet(infos) and not _is_sheet_plus_pec(infos):
        raise ValueError(
            "Mixed TYPE 1 sheet + (IBC-coated body / dielectric body / "
            "layered coating) geometries are not currently supported. "
            "Supported options: all-sheet (any number of TYPE 1 sheets, "
            "each with its own Z_s), or sheet + pure-PEC TYPE 2 body (mixed "
            "sheet+PEC is handled by _solve_mixed_sheet_pec).  For a "
            "tapered resistive treatment on a coated body, use TYPE 2 with "
            "a tapered IBC row -- that's the physically correct model for "
            "'resistance transitioning from air to a conducting body'."
        )


def _assert_no_type1_sheet(infos: 'List[PanelCoupledInfo]') -> 'None':
    """Raise if any element is a TYPE 1 sheet (boundary-density export path).

    TYPE 1 free-floating sheets ARE supported for RCS, via the dedicated
    sheet BIEs (``_solve_tm_sheet`` / ``_solve_te_sheet`` /
    ``_solve_mixed_sheet_pec``) reached through ``solve_monostatic_rcs_2d``
    and ``solve_bistatic_rcs_2d``.  This guard only protects
    ``compute_boundary_densities``, whose dispatch covers just the robin /
    multi-region / coupled-trace solvers -- none of which build the sheet
    representation, so they would ignore or mishandle the sheet admittance
    q_plus_gamma.  Rather than return wrong densities, fail fast here and
    point the user at the RCS entry points.
    """
    if _has_sheet(infos):
        raise ValueError(
            "TYPE 1 free-floating sheets are not supported by "
            "compute_boundary_densities (the boundary-density export path).  "
            "They ARE supported for RCS: use solve_monostatic_rcs_2d or "
            "solve_bistatic_rcs_2d, which route sheet geometries to the "
            "dedicated sheet BIEs (_solve_tm_sheet / _solve_te_sheet / "
            "_solve_mixed_sheet_pec)."
        )

def _solve_dielectric_indirect(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve dielectric scattering via the indirect two-density formulation.

    The coupled trace formulation degenerates for dielectrics because the exterior
    BIE alone determines the far-field amplitude regardless of the flux continuity
    parameter beta.  This indirect formulation uses separate densities:

        u_scat(r) = DL_0(mu)   (exterior, double-layer at k0)
        u_int(r)  = SL_1(sigma) (interior, single-layer at k1)

    Trace continuity:  u_inc + (mu/2 + K0*mu) = S1*sigma
    Flux continuity:   W0*mu + factor*(sigma/2 + K'1*sigma) = du_inc/dn

    The Maue matrix is W0 = -d_n DL_0. The interior derivative of SL_1
    is (1/2 + K'1). Thus both rows below follow directly from continuity;
    the physical exterior density is mu, without a far-field sign change.

    where factor = mu_ext/mu_int for E_z, eps_ext/eps_int for H_z.

    Far-field: A = integral jk0*(d.n)*mu * exp(jk0 d.r') ds'
    """

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    # Determine interior wavenumber from coupled infos.
    k1_vals = {complex(info.k_plus) for info in infos if info.plus_region > 0}
    if not k1_vals:
        k1_vals = {complex(info.k_minus) for info in infos if info.minus_region > 0}
    if not k1_vals:
        raise ValueError("Dielectric indirect solver requires at least one dielectric region.")
    k1 = k1_vals.pop()

    # Determine flux scaling factor.
    info0 = infos[0]
    if pol == 'TM':
        # E_z: flux uses 1/mu -> factor = mu_ext/mu_int
        factor = complex(info0.mu_minus / info0.mu_plus) if abs(info0.mu_plus) > EPS else 1.0
    else:
        # H_z: flux uses 1/eps -> factor = eps_ext/eps_int
        factor = complex(info0.eps_minus / info0.eps_plus) if abs(info0.eps_plus) > EPS else 1.0

    # Assemble operators.
    _, K0 = _assemble_linear_operator_matrices(
        mesh, k0, obs_normal_deriv=False,
        obs_order=obs_order, src_order=src_order,
        compute_single_layer=False,
    )
    S1, Kp1 = _assemble_linear_operator_matrices(
        mesh, k1, obs_normal_deriv=True,
        obs_order=obs_order, src_order=src_order,
        compute_single_layer=True,
    )
    D0 = _assemble_linear_hypersingular_matrix(
        mesh, k0, obs_order=obs_order, src_order=src_order,
    )
    M = _assemble_linear_mass_matrix(mesh)

    # Build system: 2N x 2N.
    a_sys = np.zeros((2 * nnodes, 2 * nnodes), dtype=np.complex128)
    # Row 1 (trace): 0.5*M*mu + K0*mu - S1*sigma = -bu
    a_sys[:nnodes, :nnodes] = 0.5 * M + K0
    a_sys[:nnodes, nnodes:] = -S1
    # Row 2 (flux): W0*mu + factor*(0.5*M + K'1)*sigma = bdn
    a_sys[nnodes:, :nnodes] = D0
    a_sys[nnodes:, nnodes:] = factor * (0.5 * M + Kp1)

    # Build RHS for all elevations.
    rhs_sys = np.zeros((2 * nnodes, elev.size), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        load_u = _linear_element_incident_load_many(elem, k_air=k0, elevations_deg=elev)
        load_dn = _linear_element_incident_dn_load_many(elem, k_air=k0, elevations_deg=elev)
        rhs_sys[ids, :] -= load_u
        rhs_sys[nnodes + ids, :] += load_dn

    _ensure_finite_linear_system(a_sys, rhs_sys, label="dielectric indirect system")
    sol = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics,
        "dielectric indirect system",
    )
    if sol.ndim == 1:
        sol = sol.reshape(-1, 1)

    mu_mat = sol[:nnodes, :]  # DL density

    # Residual.
    residual = np.linalg.norm(a_sys @ sol - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, mu_mat, k0, elev, "DLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k0)
    return rcs_lin, amp, float(np.max(residual_vec))

def _geometric_sheet_endpoint_nodes(
    mesh: 'LinearMesh',
    infos: 'Optional[List[PanelCoupledInfo]]' = None,
) -> 'np.ndarray':
    """Return node IDs that are geometric open-strip endpoints (Meixner pin targets).

    A node is a strip endpoint iff only one sheet-element endpoint lands on
    its geometric key (across the whole mesh, regardless of signature).

    The signature-based mesh builder creates distinct node IDs for
    geometrically-coincident panels that have different material signatures
    (e.g., adjacent stair-step tapered-IBC segments with different flags);
    per-node incidence counting would wrongly flag every such node as an
    endpoint and pin mu=0 everywhere.  Counting by geometric key avoids this.

    When ``infos`` is provided, only elements with ``info.seg_type == 1``
    contribute -- so sheet endpoints that touch a PEC body are still flagged
    as endpoints (correct for Meixner), while PEC-body-internal nodes don't
    qualify.  When ``infos`` is None, all elements contribute (appropriate
    for all-sheet geometries).
    """
    geom_count: 'Dict[Tuple[int, int], int]' = {}
    for eidx, elem in enumerate(mesh.elements):
        if infos is not None and int(infos[eidx].seg_type) != 1:
            continue
        for nid in elem.node_ids:
            gk = tuple(mesh.nodes[int(nid)].key)
            geom_count[gk] = geom_count.get(gk, 0) + 1
    endpoint_ids: 'List[int]' = []
    for nid in range(len(mesh.nodes)):
        gk = tuple(mesh.nodes[int(nid)].key)
        if geom_count.get(gk, 0) == 1:
            endpoint_ids.append(int(nid))
    return np.asarray(endpoint_ids, dtype=np.int64)


def _is_all_sheet(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """True if every element is a TYPE 1 free-floating sheet."""
    if not infos:
        return False
    return all(int(info.seg_type) == 1 for info in infos)


def _solve_tm_sheet(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve TM (E_z axial) scattering from a thin resistive/reactive sheet.

    Derivation under e^{+jwt}:
        - E_z is continuous across the sheet; J_z = E_z / Z_s on the sheet.
        - Scattered field from an axial current:  u_s = jketa . Int G . J_z dr'.
        - SIBC on the sheet:                      u_inc + u_s = Z_s . J_z.

    Introducing sigma = jketa . J_z so that u_s = SLP(sigma) matches the existing SLP
    far-field projector, the SIBC becomes
        (S - (Z_s / jketa) M) sigma = -RHS_uinc
    in nodal Galerkin form, where S is the single-layer operator at k0 and
    M is the consistent boundary mass matrix.  Per-node Z_s (which may
    vary on a tapered sheet) enters as a diagonal scaling on M.

    Limit cases:
        Z_s -> 0  (PEC sheet):        S sigma = -u_inc          (TM EFIE)
        Z_s -> inf (transparent):     sigma -> 0                (no scattering)

    Returns (rcs_lin, amp, residual_norm_max).  Far field uses the standard
    SLP projector already validated for TM PEC Mie cases.
    """

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    z_elements = np.asarray(
        [complex(info.robin_impedance) for info in infos],
        dtype=np.complex128,
    )

    # Operators and mass.
    S_mat, _ = _assemble_linear_operator_matrices(
        mesh, k0, obs_normal_deriv=False,
        obs_order=obs_order, src_order=src_order,
        compute_double_layer=False)
    # SIBC coefficient remains inside the element weak integral.
    denom = 1j * float(k0) * ETA0
    sigma_factor_elements = z_elements / denom
    weighted_mass = _assemble_linear_weighted_mass_matrix(
        mesh, sigma_factor_elements
    )

    a_sys = S_mat - weighted_mass

    # RHS: -<phi, u_inc>.
    rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        load_u = _linear_element_incident_load_many(elem, k_air=float(k0), elevations_deg=elev)
        rhs_sys[ids, :] -= load_u

    _ensure_finite_linear_system(a_sys, rhs_sys, label="TM sheet system")
    sigma_mat = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics, "TM sheet system"
    )
    if sigma_mat.ndim == 1:
        sigma_mat = sigma_mat.reshape(-1, 1)

    residual = np.linalg.norm(a_sys @ sigma_mat - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, sigma_mat, float(k0), elev, "SLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, float(k0))
    return rcs_lin, amp, float(np.max(residual_vec))


def _solve_te_sheet(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve TE (H_z axial) scattering from a thin resistive/reactive sheet.

    Derivation under e^{+jwt}:
        - E_tangent is continuous across the sheet.
        - H_z jumps across the sheet by the induced tangential surface
          current K:  [H_z]+ - [H_z]- = K.
        - Ohm's law: K = E_tangent / Z_s.
        - E_tangent related to H_z via E_x = -(1/jomegaeps) dH_z/dy, which is
          continuous across the sheet (same on both sides).

    Represent u_s by a double-layer potential with density mu = K:
        u_s(r) = Int (dG(r,r')/dn_src) . mu(r') dr'.
    Then u_s jumps across the sheet by exactly mu (as required), and the
    normal derivative of u_s is continuous and equals -(W mu)(r), where W
    is the hypersingular matrix assembled through the Maue identity.

    At the sheet:  q = q_inc - W mu,  E_tangent relates to q through the
    local frame, and K = Y . E_tangent gives:
        (W - jomegaeps . Z_s . I) mu = q_inc

    (Using omegaeps = k/eta with eta = free-space impedance.  The sign of the Z_s
    term mirrors the validated TM sheet system S - (Z_s/jketa).M: this
    code's Green's function is G = +(j/4)H0^(2) -- the negative of the
    textbook e^{+jomegat} fundamental solution -- which flips the sign of the
    operator terms relative to the mass term.  Validated against the
    analytic resistive-sheet jump-BC series; the naive '+' sign makes a
    passive sheet scatter above the PEC level.)

    Limit cases:
        Z_s -> 0  (PEC sheet):        W mu = q_inc           (TE PEC Neumann)
        Z_s -> inf (transparent):     mu -> 0                (no scattering)

    Returns (rcs_lin, amp, residual_norm_max).  Far field uses the standard
    DLP projector.
    """

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    z_elements = np.asarray(
        [complex(info.robin_impedance) for info in infos],
        dtype=np.complex128,
    )

    # Operators: hypersingular via Maue, plus mass matrix.
    N_mat = _assemble_linear_hypersingular_matrix(
        mesh, k0, obs_order=obs_order, src_order=src_order)
    # Coefficient: jomegaeps . Z_s = (jk/eta) . Z_s, retained inside
    # each element weak integral.
    coeff_elements = (1j * float(k0) / ETA0) * z_elements
    weighted_mass = _assemble_linear_weighted_mass_matrix(
        mesh, coeff_elements
    )

    a_sys = N_mat - weighted_mass

    # W = -d_n D: the physical DLP density has RHS +<phi, du_inc/dn>.
    rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        load_dn = _linear_element_incident_dn_load_many(elem, k_air=float(k0), elevations_deg=elev)
        rhs_sys[ids, :] += load_dn

    # Meixner edge condition: at open-strip endpoints mu -> 0 (H_z is
    # continuous across the strip edge, so the jump density vanishes).
    # See _geometric_sheet_endpoint_nodes for the subtle point about
    # signature-split nodes in stair-stepped tapers.
    endpoint_nodes = _geometric_sheet_endpoint_nodes(mesh)
    if endpoint_nodes.size > 0:
        a_sys[endpoint_nodes, :] = 0.0
        a_sys[endpoint_nodes, endpoint_nodes] = 1.0
        rhs_sys[endpoint_nodes, :] = 0.0

    _ensure_finite_linear_system(a_sys, rhs_sys, label="TE sheet system")
    mu_mat = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics, "TE sheet system"
    )
    if mu_mat.ndim == 1:
        mu_mat = mu_mat.reshape(-1, 1)

    residual = np.linalg.norm(a_sys @ mu_mat - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, mu_mat, float(k0), elev, "DLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, float(k0))
    return rcs_lin, amp, float(np.max(residual_vec))


def _is_sheet_plus_pec(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """True if every element is either a TYPE 1 sheet or a pure-PEC TYPE 2.

    "Pure-PEC TYPE 2" means bc_kind == 'robin' with zero impedance, i.e., the
    Leontovich coefficient reduces to the Dirichlet (TM) / Neumann (TE)
    limit.  Such elements have no IBC layer -- they're hard PEC surfaces.
    """
    if not infos:
        return False
    has_sheet = False
    has_pec = False
    for info in infos:
        if int(info.seg_type) == 1:
            has_sheet = True
        elif info.bc_kind == 'robin' and abs(complex(info.robin_impedance)) <= EPS:
            has_pec = True
        else:
            # Anything else (dielectric, coated IBC, etc.) disqualifies.
            return False
    return has_sheet and has_pec


def _solve_mixed_sheet_pec(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve mixed TYPE 1 sheet + TYPE 2 PEC body geometries in a single block.

    The key insight is that both the sheet BIE and the PEC body BIE can
    share a single boundary unknown and representation, differing only in a
    coefficient-weighted element mass term:

        TM (single-layer representation, u_s = S sigma):
            PEC   nodes:  row = S                          RHS = -<phi, u_inc>
            sheet nodes:  row = S - (Z_s / jketa) M          RHS = -<phi, u_inc>

            Unified: (S - diag(alpha_TM) M) sigma = -RHS_u
            where alpha_TM[i] = 0 on PEC, Z_s[i]/(jketa) on sheet.

        TE (double-layer representation, u_s = D mu):
            PEC   nodes:  row = N                          RHS = -<phi, dn_u_inc>
            sheet nodes:  row = N - (jk/eta) Z_s M           RHS = -<phi, dn_u_inc>

            Unified: (N - diag(alpha_TE) M) mu = -RHS_dn_u
            where alpha_TE[i] = 0 on PEC, (jk/eta) Z_s[i] on sheet.

    Because both the sheet and the PEC body share the same representation,
    the cross-coupling between sources on one and observations on the other
    is automatic -- the S (resp. N) matrix is assembled over ALL elements
    (sheet + PEC), and the BC is applied row-by-row.

    Caveat: the TM PEC body is solved via plain SLP-EFIE here, which can
    suffer interior-resonance issues for electrically large closed bodies.
    Condition diagnostics and mesh convergence must therefore be retained
    for production use.  Solving the sheet and body separately and adding
    their far fields is NOT a valid workaround in general because it omits
    their mutual multiple-scattering interaction.
    """
    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    z_elements = np.asarray([
        complex(info.robin_impedance) if int(info.seg_type) == 1 else 0.0 + 0.0j
        for info in infos
    ], dtype=np.complex128)

    if pol == "TM":
        S_mat, _ = _assemble_linear_operator_matrices(
            mesh, k0, obs_normal_deriv=False,
            obs_order=obs_order, src_order=src_order,
            compute_double_layer=False,
        )
        weighted_mass = _assemble_linear_weighted_mass_matrix(
            mesh, z_elements / (1j * float(k0) * ETA0)
        )
        a_sys = S_mat - weighted_mass

        # RHS: -<phi, u_inc>
        rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
        for elem in mesh.elements:
            ids = np.asarray(elem.node_ids, dtype=int)
            load_u = _linear_element_incident_load_many(
                elem, k_air=float(k0), elevations_deg=elev,
            )
            rhs_sys[ids, :] -= load_u

        solve_label = "mixed sheet+PEC TM system"
        _ensure_finite_linear_system(a_sys, rhs_sys, label=solve_label)

    else:   # TE
        N_mat = _assemble_linear_hypersingular_matrix(
            mesh, k0, obs_order=obs_order, src_order=src_order,
        )
        # Sign matches _solve_te_sheet: N - (jk/eta)Z_s.M (see derivation there).
        weighted_mass = _assemble_linear_weighted_mass_matrix(
            mesh, (1j * float(k0) / ETA0) * z_elements
        )
        a_sys = N_mat - weighted_mass

        # RHS: +<phi, du_inc/dn>, since W = -d_n D.
        rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
        for elem in mesh.elements:
            ids = np.asarray(elem.node_ids, dtype=int)
            load_dn = _linear_element_incident_dn_load_many(
                elem, k_air=float(k0), elevations_deg=elev,
            )
            rhs_sys[ids, :] += load_dn

        # Meixner edge condition on open sheet endpoints: mu=0 (H_z continuous
        # at the strip edge).  Applied only to nodes that are geometric
        # endpoints of sheet elements -- not to closed-PEC-body nodes, and not
        # to stair-step-segment-boundary nodes that are geometrically interior
        # but have distinct signatures.  See _geometric_sheet_endpoint_nodes.
        endpoint_nodes = _geometric_sheet_endpoint_nodes(mesh, infos)
        if endpoint_nodes.size > 0:
            a_sys[endpoint_nodes, :] = 0.0
            a_sys[endpoint_nodes, endpoint_nodes] = 1.0
            rhs_sys[endpoint_nodes, :] = 0.0

        solve_label = "mixed sheet+PEC TE system"
        _ensure_finite_linear_system(a_sys, rhs_sys, label=solve_label)

    sol_mat = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics, solve_label
    )

    if sol_mat.ndim == 1:
        sol_mat = sol_mat.reshape(-1, 1)

    residual = np.linalg.norm(a_sys @ sol_mat - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, sol_mat, float(k0), elev,
        "SLP" if pol == "TM" else "DLP", order=obs_order,
    )

    rcs_lin = _rcs_sigma_from_amp(amp, float(k0))
    return rcs_lin, amp, float(np.max(residual_vec))


def _assemble_robin_bie_system(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    operator_cache: 'Optional[Dict[Any, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """
    Assemble the all-Robin SLP BIE system shared by the monostatic and
    bistatic solvers (see `_solve_robin_bie` for the derivation).

    Returns (a_sys, alpha_elements, pec_node) where a_sys already contains the
    per-row TM-PEC EFIE override, alpha_elements is the piecewise-constant
    Robin coefficient used inside the weak observation integral, and pec_node
    marks nodes incident on a PEC (Z_s = 0) element.

    INTERIOR RESONANCES (why there is deliberately NO CFIE here): the
    system's conditioning spikes at the cavity's interior Dirichlet
    eigenfrequencies (both TM-EFIE S and TE-MFIE K'-1/2 share that
    resonance set), but with the INDIRECT SLP ansatz the exterior far field
    is immune: a resonant null density sigma_0 has S sigma_0 = 0 on the
    contour, so by exterior uniqueness S sigma_0 vanishes IDENTICALLY
    outside -- the null space radiates nothing, and a direct solve stays
    accurate (validated at the first interior resonance of a PEC circle for
    both polarizations, including complex field and base/fine convergence;
    see ``PecInteriorResonanceAcceptanceTests`` in
    tests/test_2d_capability_acceptance.py).  A Robin-style "CFIE"
    combination of trace and normal-derivative rows was tried and REJECTED:
    with the SLP ansatz it imposes a second, physically false boundary
    condition on PEC (u = 0 for TE, du/dn = 0 for TM) and shifts the RCS by
    ~9.5 dB everywhere.  A genuine resonance-free indirect scheme needs the
    Brakhage-Werner combined-SOURCE ansatz (double-layer + hypersingular
    operators); the supported direct solves do not need it.
    The public 2-D entry points reject nonzero ``cfie_alpha`` so this setting
    cannot silently leave the physical operator unchanged.
    """

    nnodes = len(mesh.nodes)

    alpha_elements, pec_elements = _robin_alpha_elements(mesh, infos, pol)
    pec_node = np.zeros(nnodes, dtype=bool)
    for eidx, elem in enumerate(mesh.elements):
        if bool(pec_elements[eidx]):
            for nid in elem.node_ids:
                pec_node[int(nid)] = True

    has_ibc = bool(np.any(np.abs(alpha_elements) > EPS))
    all_tm_pec = bool(pol == 'TM' and np.all(pec_elements))

    # Assemble the single-layer and adjoint double-layer operators together.
    # S is independent of the normal-derivative selection, so the former two
    # full passes over every element pair were redundant.  Pure TM-PEC uses
    # only S because every row is replaced by the EFIE limit.
    need_kp = not all_tm_pec
    cache_key = (
        "air_kp",
        id(mesh),
        complex(k0),
        int(obs_order),
        int(src_order),
    )
    cached_kp = (
        operator_cache.get(cache_key)
        if operator_cache is not None and need_kp else None
    )
    if cached_kp is not None:
        # TE has already assembled the same air-side K' on the shared mesh.
        # Assemble only the polarization-specific weighted S term for TM.
        S_alpha, _unused = _assemble_linear_operator_matrices(
            mesh, k0, obs_normal_deriv=True,
            obs_order=obs_order, src_order=src_order,
            compute_single_layer=(has_ibc or all_tm_pec),
            compute_double_layer=False,
            single_layer_observation_coefficients=(
                alpha_elements if has_ibc and not all_tm_pec else None
            ),
        )
        Kp_mat = cached_kp
        operator_cache["_hits"] = int(operator_cache.get("_hits", 0)) + 1
    else:
        S_alpha, Kp_mat = _assemble_linear_operator_matrices(
            mesh, k0, obs_normal_deriv=True,
            obs_order=obs_order, src_order=src_order,
            compute_single_layer=(has_ibc or all_tm_pec),
            compute_double_layer=need_kp,
            single_layer_observation_coefficients=(
                alpha_elements if has_ibc and not all_tm_pec else None
            ),
        )
        if operator_cache is not None and need_kp:
            operator_cache[cache_key] = Kp_mat
            operator_cache["_stores"] = int(
                operator_cache.get("_stores", 0)
            ) + 1
    M_mat = _assemble_linear_mass_matrix(mesh)

    # Default Robin row: (-1/2 M + K' + S_alpha) sigma,
    # where alpha remains inside each observation-element integral.
    a_sys = -0.5 * M_mat + Kp_mat + (S_alpha if has_ibc else 0.0)

    # TM PEC override: replace those rows with the EFIE  S sigma = -u_inc.
    # This is the alpha -> infinity limit of the Robin BIE, divided by alpha
    # to recover a well-conditioned operator.  For TE PEC, alpha = 0 already
    # gives the correct MFIE, so no override is needed there.
    tm_pec_rows = np.flatnonzero(pec_node) if pol == 'TM' else np.zeros(0, dtype=np.int64)
    if tm_pec_rows.size > 0:
        if all_tm_pec:
            S_plain = S_alpha
        else:
            S_plain, _ = _assemble_linear_operator_matrices(
                mesh, k0, obs_normal_deriv=False,
                obs_order=obs_order, src_order=src_order,
                compute_double_layer=False,
            )
        a_sys[tm_pec_rows, :] = S_plain[tm_pec_rows, :]

    return a_sys, alpha_elements, pec_node


def _robin_bie_rhs_many(
    mesh: 'LinearMesh',
    alpha_elements: 'np.ndarray',
    pec_node: 'np.ndarray',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
) -> 'np.ndarray':
    """RHS columns for `_assemble_robin_bie_system`, one per elevation angle."""

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
    tm_pec_rows = (
        np.flatnonzero(pec_node)
        if pol == 'TM' else np.zeros(0, dtype=np.int64)
    )
    all_tm_pec = bool(pol == 'TM' and np.all(pec_node))
    pec_rhs = (
        np.zeros_like(rhs_sys) if tm_pec_rows.size > 0 else None
    )
    alpha_eval = np.asarray(alpha_elements, dtype=np.complex128).reshape(-1)
    if alpha_eval.size != len(mesh.elements):
        raise ValueError("Robin RHS coefficient count must match mesh elements.")
    for eidx, elem in enumerate(mesh.elements):
        ids = np.asarray(elem.node_ids, dtype=int)
        load_u = _linear_element_incident_load_many(elem, k_air=k0, elevations_deg=elev)
        if not all_tm_pec:
            load_dn = _linear_element_incident_dn_load_many(
                elem, k_air=k0, elevations_deg=elev
            )
            rhs_sys[ids, :] -= load_dn
            rhs_sys[ids, :] -= complex(alpha_eval[eidx]) * load_u
        if pec_rhs is not None:
            for local_index, node_id in enumerate(ids):
                if pec_node[int(node_id)]:
                    pec_rhs[int(node_id), :] -= load_u[local_index, :]

    # TM PEC override on RHS: the EFIE row uses -u_inc, not -(du_inc/dn + alpha*u_inc).
    if tm_pec_rows.size > 0:
        rhs_sys[tm_pec_rows, :] = pec_rhs[tm_pec_rows, :]

    return rhs_sys


def _solve_robin_bie(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
    operator_cache: 'Optional[Dict[Any, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve all-Robin (PEC, IBC, or mixed PEC+IBC) scattering with SLP representation.

    Uses u_scat = SLP(sigma) and the exterior-limit Robin BC
    du/dn + alpha*u = 0 in its element-weighted Galerkin form:

        (-1/2 M + K' + S_alpha) sigma = -(du_inc/dn + alpha*u_inc)

    The Robin coefficient alpha is computed per element (function of pol, the
    medium adjacent to the surface, and Z_s) and retained inside that
    observation element's weak integral. This is the consistent Galerkin form
    for the element-center-sampled impedance model, including tapered IBCs.

    Polarisation/PEC handling (per row, so mixed PEC + IBC at TYPE 2 / TYPE 4
    interfaces are supported correctly):

      - TM PEC node (Z_s = 0):  alpha is formally infinite.  Divide the BC by
        alpha to obtain the well-conditioned EFIE row  S sigma = -u_inc.
      - TM IBC node (Z_s != 0): finite alpha_TM = j k_med eta_med / Z_s.
        Standard Robin BIE row above.
      - TE PEC node (Z_s = 0):  alpha = 0 already gives the correct MFIE row
        (-1/2 M + K') sigma = -du_inc/dn.
      - TE IBC node (Z_s != 0): finite alpha_TE = +j k_med Z_s / eta_med.
        Standard Robin BIE row above.

    Without the TM-PEC row override the alpha->infty limit cannot be taken
    numerically (the original implementation silently degenerated TM PEC to
    the TE MFIE, giving the wrong boundary condition and ~0.3-1.0 dB Mie
    error depending on ka).

    Far-field: SLP projector  A = integral sigma * exp(jk d.r') ds'.
    """

    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    a_sys, alpha_elements, pec_node = _assemble_robin_bie_system(
        mesh, infos, pol, k0, obs_order=obs_order, src_order=src_order,
        operator_cache=operator_cache)
    rhs_sys = _robin_bie_rhs_many(
        mesh, alpha_elements, pec_node, pol, k0, elev
    )

    _ensure_finite_linear_system(a_sys, rhs_sys, label="Robin-BIE IBC system")
    sigma_mat = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics, "Robin-BIE IBC system"
    )
    if sigma_mat.ndim == 1:
        sigma_mat = sigma_mat.reshape(-1, 1)

    # Residual.
    residual = np.linalg.norm(a_sys @ sigma_mat - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, sigma_mat, k0, elev, "SLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k0)
    return rcs_lin, amp, float(np.max(residual_vec))


def _make_elem_mask(elem_ids, n_total):
    mask = np.zeros(n_total, dtype=bool)
    for eidx in elem_ids: mask[eidx] = True
    return mask

def _count_distinct_regions(infos):
    regions = set()
    for info in infos:
        if info.minus_region >= 0: regions.add(info.minus_region)
        if info.plus_region >= 0: regions.add(info.plus_region)
    return len(regions)

def _is_multi_region(infos):
    """True if geometry needs multi-region solver (layered, coated, or mixed PEC+diel).

    Excludes any geometry containing a TYPE 1 sheet -- the multi-region
    indirect solver treats its transmission interfaces as pure-medium
    boundaries and would ignore the sheet's impedance.  In the RCS dispatch,
    supported sheet geometries are routed to the dedicated sheet solvers
    (and unsupported mixes rejected by ``_assert_no_type1_sheet_for_mixed``)
    before this predicate; the ``_has_sheet`` check below is belt-and-braces.
    """
    if _has_sheet(infos):
        return False
    n_regions = _count_distinct_regions(infos)
    if n_regions > 2:
        return True
    # Mixed PEC+dielectric also needs multi-region because interior PEC
    # boundaries require interior wavenumber, not k_air.
    has_transmission = any(info.bc_kind == 'transmission' for info in infos)
    has_robin = any(info.bc_kind == 'robin' for info in infos)
    return has_transmission and has_robin


def _dense_formulation_resources(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
) -> 'Dict[str, Any]':
    """Return the dense system size and retained-operator budget.

    This is the single formulation classifier used by both the solver's
    pre-allocation memory gate and the sweep scheduler.  Keeping it beside the
    active dispatch predicates prevents a scheduler estimate from silently
    drifting back to the old generic 2N coupled-system assumption.
    """

    nnodes = int(len(mesh.nodes))
    regions = {
        int(region)
        for info in infos
        for region in (info.minus_region, info.plus_region)
        if int(region) >= 0
    }

    if any(info.bc_kind == "thin_layer" for info in infos):
        formulation = "thin_dielectric_layer"
        system_dofs = 2 * nnodes
        operator_matrices = 20
    elif _is_all_sheet(infos):
        formulation = "sheet"
        system_dofs = nnodes
        operator_matrices = 3
    elif _is_sheet_plus_pec(infos):
        formulation = "mixed_sheet_pec"
        system_dofs = nnodes
        operator_matrices = 3
    elif pol == "TE" and _is_all_robin(infos):
        formulation = "te_robin"
        system_dofs = nnodes
        operator_matrices = 3
    elif _is_multi_region(infos):
        interface_nodes = {}  # type: Dict[Tuple[int, int], Set[int]]
        for elem, info in zip(mesh.elements, infos):
            key = (int(info.minus_region), int(info.plus_region))
            interface_nodes.setdefault(key, set()).update(
                int(node) for node in elem.node_ids
            )
        formulation = "multi_region"
        system_dofs = sum(
            len(nodes) * int(r_minus >= 0)
            + len(nodes) * int(r_plus >= 0)
            for (r_minus, r_plus), nodes in interface_nodes.items()
        )
        operator_matrices = 4 * max(1, len(regions))
    elif _is_single_dielectric_body(infos):
        formulation = "single_dielectric"
        system_dofs = 2 * nnodes
        operator_matrices = 5
    elif _is_all_robin(infos):
        formulation = "robin"
        system_dofs = nnodes
        operator_matrices = 3
    else:
        raise ValueError(
            "Geometry does not match a supported dense 2-D formulation."
        )

    return {
        "nodes": nnodes,
        "n_regions": int(len(regions)),
        "formulation": formulation,
        "system_dofs": int(system_dofs),
        "operator_matrices": int(operator_matrices),
    }

def _solve_multi_region_indirect(
    mesh,
    infos,
    pol,
    k0,
    elevations_deg,
    obs_order=8,
    src_order=8,
    solver_method="auto",
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
):
    r"""Multi-region indirect SLP formulation for layered dielectric coatings.

    BIE sign convention (validated against single-region solvers):
    - Element normal n points from minus_region toward plus_region.
    - Density on minus_region side: flux = (-1/2 M + K') * sigma
    - Density on plus_region side:  flux = (+1/2 M + K') * tau
    - Cross-interface operators (source != observer): no +/-1/2 jump.

    Uses dense LU for both solver_method="auto" and "direct".
    """
    _normalize_public_2d_solver_method(solver_method)
    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    elements = list(mesh.elements)
    nelems = len(elements)

    # 1. Discover regions.
    region_props = {}
    for info in infos:
        for rid, k, eps, mu, has_inc in [
            (info.minus_region, info.k_minus, info.eps_minus, info.mu_minus, info.minus_has_incident),
            (info.plus_region, info.k_plus, info.eps_plus, info.mu_plus, info.plus_has_incident),
        ]:
            if rid >= 0 and rid not in region_props:
                region_props[rid] = {'k': complex(k), 'eps': complex(eps), 'mu': complex(mu), 'has_incident': bool(has_inc)}

    # 2. Discover interfaces.
    iface_elems = {}
    for eidx, info in enumerate(infos):
        iface_elems.setdefault((info.minus_region, info.plus_region), []).append(eidx)

    ifaces = []
    for (r_m, r_p), eids in sorted(iface_elems.items()):
        nodes = sorted({nid for ei in eids for nid in elements[ei].node_ids})
        pec_minus = (r_m < 0)
        pec_plus = (r_p < 0)
        robin_alpha = np.zeros(len(nodes), dtype=np.complex128)
        robin_alpha_elements = np.zeros(nelems, dtype=np.complex128)
        if pec_minus or pec_plus:
            diel_rid = r_p if pec_minus else r_m
            if diel_rid >= 0 and diel_rid in region_props:
                rp = region_props[diel_rid]
                # The element normal points minus_region -> plus_region.  The
                # Robin rows below are written as q + alpha*u = 0, which is the
                # Leontovich BC when the normal points INTO the impedance
                # backing (pec_plus, TYPE 2 orientation).  For pec_minus
                # (TYPE 4) the normal points into the dielectric field region
                # instead, so the flux term flips sign: q - alpha*u = 0.  Bake
                # the side sign into the stored alpha so the matrix rows
                # and RHS stay consistent.
                side_sign = -1.0 if pec_minus else 1.0
                for ei in eids:
                    z_s = complex(infos[ei].robin_impedance)
                    if abs(z_s) > EPS:
                        robin_alpha_elements[ei] = (
                            side_sign
                            * _surface_robin_alpha(
                                pol, rp['eps'], rp['mu'], rp['k'], z_s
                            )
                        )
                # Nodal alpha identifies the TM PEC rows below. The weak
                # observation integral uses robin_alpha_elements directly.
                for ni, nid in enumerate(nodes):
                    incident = [
                        robin_alpha_elements[ei]
                        for ei in eids if nid in elements[ei].node_ids
                    ]
                    if incident:
                        robin_alpha[ni] = sum(incident) / len(incident)
        ifaces.append({'r_m': r_m, 'r_p': r_p, 'eids': eids, 'nodes': nodes, 'n': len(nodes),
                       'pec_minus': pec_minus, 'pec_plus': pec_plus,
                       'robin_alpha': robin_alpha,
                       'robin_alpha_elements': robin_alpha_elements,
                       'mask': _make_elem_mask(eids, nelems)})

    region_ifaces = {}
    for mi, ifc in enumerate(ifaces):
        for rid in [ifc['r_m'], ifc['r_p']]:
            if rid >= 0: region_ifaces.setdefault(rid, []).append(mi)

    # 3. DOF layout: one density per dielectric side per interface.
    dof_map = {}
    n_dof = 0
    for mi, ifc in enumerate(ifaces):
        if ifc['r_m'] >= 0:
            dof_map[(mi, 'minus')] = (n_dof, ifc['n']); n_dof += ifc['n']
        if ifc['r_p'] >= 0:
            dof_map[(mi, 'plus')] = (n_dof, ifc['n']); n_dof += ifc['n']

    # 4. Dense operator cache.
    M_global = _assemble_linear_mass_matrix(mesh)


    op_cache = {}
    def get_ops(k_val, src_mask):
        key = (complex(k_val), id(src_mask))
        if key not in op_cache:
            S, Kp = _assemble_linear_operator_matrices(
                mesh, k_val, True, obs_order, src_order,
                source_element_mask=src_mask,
            )
            op_cache[key] = (S, Kp)
        return op_cache[key]

    weighted_s_cache = {}
    def get_weighted_s(k_val, src_mask, obs_coeff):
        coeff_eval = np.asarray(obs_coeff, dtype=np.complex128)
        key = (complex(k_val), id(src_mask), id(obs_coeff))
        if not np.any(np.abs(coeff_eval) > EPS):
            return np.broadcast_to(
                np.zeros((), dtype=np.complex128),
                (nnodes, nnodes),
            )
        if key not in weighted_s_cache:
            S_alpha, _ = _assemble_linear_operator_matrices(
                mesh, k_val, True, obs_order, src_order,
                source_element_mask=src_mask,
                compute_double_layer=False,
                single_layer_observation_coefficients=coeff_eval,
            )
            weighted_s_cache[key] = S_alpha
        return weighted_s_cache[key]

    # Prefetch every unweighted S/K' output and every Robin-weighted S
    # output in one traversal per distinct wavenumber. Kernel values and
    # near-pair blocks depend on geometry and k, not on which source mask
    # or observation coefficient consumes them.
    _requests_by_k = {}
    _request_seen = {}
    for _rid, _mis in region_ifaces.items():
        _k = complex(region_props[_rid]['k'])
        _slot = _requests_by_k.setdefault(_k, [])
        _seen = _request_seen.setdefault(_k, set())
        for _mi in _mis:
            _mask = ifaces[_mi]['mask']
            _token = ("plain", id(_mask))
            if _token in _seen:
                continue
            _seen.add(_token)
            _slot.append(("plain", _mask, None))

    for _mi, _ifc in enumerate(ifaces):
        if not (_ifc['pec_minus'] or _ifc['pec_plus']):
            continue
        _rid = _ifc['r_p'] if _ifc['pec_minus'] else _ifc['r_m']
        _k = complex(region_props[_rid]['k'])
        _slot = _requests_by_k.setdefault(_k, [])
        _seen = _request_seen.setdefault(_k, set())
        for _mj in region_ifaces.get(_rid, []):
            _src_mask = ifaces[_mj]['mask']
            _obs_coeff = _ifc['robin_alpha_elements']
            if not np.any(np.abs(_obs_coeff) > EPS):
                continue
            _token = ("weighted", id(_src_mask), id(_obs_coeff))
            if _token in _seen:
                continue
            _seen.add(_token)
            _slot.append(("weighted", _src_mask, _obs_coeff))

    for _k, _requests in _requests_by_k.items():
        _masks = [request[1] for request in _requests]
        _coeffs = [request[2] for request in _requests]
        _want_k = [request[0] == "plain" for request in _requests]
        _outputs = _assemble_linear_operator_matrices_multi(
            mesh=mesh,
            k0=_k,
            obs_normal_deriv=True,
            source_element_masks=_masks,
            obs_order=obs_order,
            src_order=src_order,
            compute_double_layer=True,
            compute_double_layer_many=_want_k,
            single_layer_observation_coefficients_many=_coeffs,
        )
        for (_kind, _mask, _coeff), (_s_mat, _k_mat) in zip(
            _requests, _outputs
        ):
            if _kind == "plain":
                op_cache[(_k, id(_mask))] = (_s_mat, _k_mat)
            else:
                weighted_s_cache[
                    (_k, id(_mask), id(_coeff))
                ] = _s_mat
    # Incident field.
    bu = np.zeros((nnodes, elev.size), dtype=np.complex128)
    bdn = np.zeros((nnodes, elev.size), dtype=np.complex128)
    for elem in elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        bu[ids] += _linear_element_incident_load_many(elem, k_air=k0, elevations_deg=elev)
        bdn[ids] += _linear_element_incident_dn_load_many(elem, k_air=k0, elevations_deg=elev)

    # 5. Assemble RHS.
    Brhs = np.zeros((n_dof, elev.size), dtype=np.complex128)

    for mi, ifc in enumerate(ifaces):
        obs_n = ifc['nodes']; nm = ifc['n']
        r_m, r_p = ifc['r_m'], ifc['r_p']
        alpha = ifc['robin_alpha']

        if ifc['pec_minus'] or ifc['pec_plus']:
            dof_side = 'plus' if ifc['pec_minus'] else 'minus'
            dm = dof_map[(mi, dof_side)]
            rid = r_p if ifc['pec_minus'] else r_m
            # Per-row TM PEC mask: TM nodes with alpha = 0 use EFIE RHS
            # (-u_inc); all other rows use Robin BIE RHS -(q_inc + alpha*u_inc).
            tm_pec_mask = (np.abs(alpha) <= EPS) if pol == 'TM' else np.zeros(nm, dtype=bool)
            if region_props[rid].get('has_incident'):
                # Default: Robin BIE RHS.
                alpha_bu_global = np.zeros_like(bu)
                alpha_e = ifc['robin_alpha_elements']
                for ei in ifc['eids']:
                    elem = elements[ei]
                    ids = np.asarray(elem.node_ids, dtype=int)
                    alpha_bu_global[ids] += (
                        complex(alpha_e[ei])
                        * _linear_element_incident_load_many(
                            elem, k_air=k0, elevations_deg=elev
                        )
                    )
                alpha_bu = alpha_bu_global[obs_n]
                rhs_block = bdn[obs_n] + alpha_bu
                # TM PEC override (per row).
                if np.any(tm_pec_mask):
                    rhs_block[tm_pec_mask] = bu[obs_n][tm_pec_mask]
                Brhs[dm[0]:dm[0]+nm] -= rhs_block
        else:
            d_sigma = dof_map[(mi, 'minus')]; d_tau = dof_map[(mi, 'plus')]
            if pol == 'TM':
                beta = complex(region_props[r_p]['mu'] / region_props[r_m]['mu']) if abs(region_props[r_m]['mu']) > EPS else 1.0+0j
            else:
                beta = complex(region_props[r_p]['eps'] / region_props[r_m]['eps']) if abs(region_props[r_m]['eps']) > EPS else 1.0+0j
            if abs(beta) <= EPS: beta = 1.0+0j
            inv_beta = 1.0 / beta
            if region_props[r_m].get('has_incident'):
                Brhs[d_sigma[0]:d_sigma[0]+nm] -= bdn[obs_n]
                Brhs[d_tau[0]:d_tau[0]+nm]     -= bu[obs_n]
            if region_props[r_p].get('has_incident'):
                Brhs[d_sigma[0]:d_sigma[0]+nm] += inv_beta * bdn[obs_n]
                Brhs[d_tau[0]:d_tau[0]+nm]     += bu[obs_n]

    Asys = np.zeros((n_dof, n_dof), dtype=np.complex128)
    def sub(mat, obs_n, src_n):
        return mat[np.ix_(obs_n, src_n)]

    def _add_robin_block_dense(mi, ifc, dof_side, region_id, jump_sign):
        dm = dof_map[(mi, dof_side)]
        obs_n = ifc['nodes']; nm = ifc['n']
        k_d = region_props[region_id]['k']
        S_self, Kp_self = get_ops(k_d, ifc['mask'])
        S_alpha_self = get_weighted_s(
            k_d, ifc['mask'], ifc['robin_alpha_elements']
        )
        M_s = sub(M_global, obs_n, obs_n)
        alpha = ifc['robin_alpha']
        # Per-node TM PEC mask: nodes where alpha = 0 and pol = TM use
        # the EFIE row  S sigma = -u_inc (the Z_s -> 0, alpha -> infty
        # limit of the Robin BIE divided through by alpha).  Without
        # this override, alpha = 0 silently collapses to the TE MFIE
        # row, which is the wrong boundary condition for TM PEC and
        # produces ~0.3-1.0 dB Mie error.  For TE the alpha = 0 case
        # already gives the correct MFIE, so no override is needed.
        tm_pec_mask = (np.abs(alpha) <= EPS) if pol == 'TM' else np.zeros(nm, dtype=bool)

        S_sub = sub(S_self, obs_n, obs_n)
        Kp_sub = sub(Kp_self, obs_n, obs_n)
        block = (
            jump_sign * 0.5 * M_s
            + Kp_sub
            + sub(S_alpha_self, obs_n, obs_n)
        )
        if np.any(tm_pec_mask):
            block[tm_pec_mask, :] = S_sub[tm_pec_mask, :]
        Asys[dm[0]:dm[0]+nm, dm[0]:dm[0]+nm] += block

        for mj in region_ifaces.get(region_id, []):
            if mj == mi: continue
            ifj = ifaces[mj]
            side_j = 'minus' if ifj['r_m'] == region_id else 'plus'
            dj = dof_map.get((mj, side_j))
            if dj is None: continue
            S_x, Kp_x = get_ops(k_d, ifj['mask']); src_n = ifj['nodes']
            S_alpha_x = get_weighted_s(
                k_d, ifj['mask'], ifc['robin_alpha_elements']
            )
            S_x_sub = sub(S_x, obs_n, src_n)
            Kp_x_sub = sub(Kp_x, obs_n, src_n)
            cross_block = (
                Kp_x_sub + sub(S_alpha_x, obs_n, src_n)
            )
            if np.any(tm_pec_mask):
                cross_block[tm_pec_mask, :] = S_x_sub[tm_pec_mask, :]
            Asys[dm[0]:dm[0]+nm, dj[0]:dj[0]+dj[1]] += cross_block

    for mi, ifc in enumerate(ifaces):
        r_m, r_p = ifc['r_m'], ifc['r_p']
        if ifc['pec_minus']:
            _add_robin_block_dense(mi, ifc, 'plus', r_p, +1.0)
        elif ifc['pec_plus']:
            _add_robin_block_dense(mi, ifc, 'minus', r_m, -1.0)
        else:
            obs_n = ifc['nodes']; nm = ifc['n']
            d_sigma = dof_map[(mi, 'minus')]; d_tau = dof_map[(mi, 'plus')]
            k_m_val = region_props[r_m]['k']; k_p_val = region_props[r_p]['k']
            if pol == 'TM':
                beta = complex(region_props[r_p]['mu'] / region_props[r_m]['mu']) if abs(region_props[r_m]['mu']) > EPS else 1.0+0j
            else:
                beta = complex(region_props[r_p]['eps'] / region_props[r_m]['eps']) if abs(region_props[r_m]['eps']) > EPS else 1.0+0j
            if abs(beta) <= EPS: beta = 1.0+0j
            inv_beta = 1.0 / beta
            S_m, Kp_m = get_ops(k_m_val, ifc['mask'])
            S_p, Kp_p = get_ops(k_p_val, ifc['mask'])
            M_s = sub(M_global, obs_n, obs_n)
            Asys[d_sigma[0]:d_sigma[0]+nm, d_sigma[0]:d_sigma[0]+nm] += -0.5*M_s + sub(Kp_m, obs_n, obs_n)
            Asys[d_sigma[0]:d_sigma[0]+nm, d_tau[0]:d_tau[0]+nm]     -= inv_beta*(0.5*M_s + sub(Kp_p, obs_n, obs_n))
            Asys[d_tau[0]:d_tau[0]+nm, d_sigma[0]:d_sigma[0]+nm]     += sub(S_m, obs_n, obs_n)
            Asys[d_tau[0]:d_tau[0]+nm, d_tau[0]:d_tau[0]+nm]         -= sub(S_p, obs_n, obs_n)
            for mj in region_ifaces.get(r_m, []):
                if mj == mi: continue
                ifj = ifaces[mj]; side_j = 'minus' if ifj['r_m'] == r_m else 'plus'
                dj = dof_map.get((mj, side_j))
                if dj is None: continue
                S_x, Kp_x = get_ops(k_m_val, ifj['mask']); src_n = ifj['nodes']
                Asys[d_sigma[0]:d_sigma[0]+nm, dj[0]:dj[0]+dj[1]] += sub(Kp_x, obs_n, src_n)
                Asys[d_tau[0]:d_tau[0]+nm, dj[0]:dj[0]+dj[1]]     += sub(S_x, obs_n, src_n)
            for mj in region_ifaces.get(r_p, []):
                if mj == mi: continue
                ifj = ifaces[mj]; side_j = 'minus' if ifj['r_m'] == r_p else 'plus'
                dj = dof_map.get((mj, side_j))
                if dj is None: continue
                S_x, Kp_x = get_ops(k_p_val, ifj['mask']); src_n = ifj['nodes']
                Asys[d_sigma[0]:d_sigma[0]+nm, dj[0]:dj[0]+dj[1]] -= inv_beta * sub(Kp_x, obs_n, src_n)
                Asys[d_tau[0]:d_tau[0]+nm, dj[0]:dj[0]+dj[1]]     -= sub(S_x, obs_n, src_n)

    # 6. Solve (dense).
    _ensure_finite_linear_system(Asys, Brhs, label="multi-region indirect system")
    sol = _solve_dense_system(
        Asys, Brhs, condition_diagnostics,
        "multi-region indirect system",
    )
    if sol.ndim == 1: sol = sol.reshape(-1, 1)
    residual = np.linalg.norm(Asys @ sol - Brhs, axis=0)
    rhs_norm = np.linalg.norm(Brhs, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    max_res = float(np.max(residual / rhs_norm))

    # 7. Extract exterior SLP density mapped to global node IDs.
    ext_rid = next((rid for rid, rp in region_props.items() if rp.get('has_incident')), 0)
    ext_density_global = np.zeros((nnodes, elev.size), dtype=np.complex128)
    ext_elem_mask = np.zeros(nelems, dtype=bool)

    for mi, ifc in enumerate(ifaces):
        if ifc['r_m'] == ext_rid:
            side = 'minus'
        elif ifc['r_p'] == ext_rid:
            side = 'plus'
        else:
            continue
        dm = dof_map.get((mi, side))
        if dm is None:
            continue
        density = sol[dm[0]:dm[0]+dm[1], :]
        for li, nid in enumerate(ifc['nodes']):
            # These equations use u_scat = S[sigma_ext]. Preserve the
            # physical SLP density in all projections and density exports.
            ext_density_global[nid, :] += density[li, :]
        for eidx in ifc['eids']:
            ext_elem_mask[eidx] = True

    # 8. Far-field from exterior SLP density.
    amp = _farfield_linear_density_many(
        mesh,
        ext_density_global,
        k0,
        elev,
        "SLP",
        order=obs_order,
        element_mask=ext_elem_mask,
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k0)
    return rcs_lin, amp, max_res, ext_density_global

@profiled_solve
@experimental_monostatic
def solve_monostatic_rcs_2d_single_polarization(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    strict_quality_gate: 'bool' = True,
    compute_condition_number: 'bool' = False,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    rcs_normalization_mode: 'str' = RCS_NORM_MODE_DEFAULT,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
    _shared_discretization_cache: 'Optional[Dict[str, Any]]' = None,
) -> 'Dict[str, Any]':
    """
    Explicit single-polarization diagnostic monostatic 2-D solve.

    Production callers should use :func:`solve_monostatic_rcs_2d` (or its
    certified/survey variants), which always solves and returns both physical
    co-polarized channels.  This function remains available for focused
    formulation tests, density diagnostics, and compatibility with specialist
    code that genuinely needs one scalar Helmholtz problem.

    Per frequency:
    - build the boundary discretization,
    - assemble the selected linear/Galerkin boundary-integral system,
    - solve all requested elevations,
    - compute monostatic backscatter RCS.

    The returned quality gate certifies the assembled discrete linear system,
    not mesh convergence.  Production callers must perform the base/fine
    complex-field mesh comparison implemented by the certified solve entry
    points and selected by the general-purpose local/HPC drivers.

    Angle convention (coming-from):
    - 0 deg: from right to left
    - +90 deg: from top to bottom
    - -90 deg: from bottom to top
    """

    if not frequencies_ghz:
        raise ValueError("At least one frequency is required.")
    if not elevations_deg:
        raise ValueError("At least one elevation angle is required.")

    frequencies = [float(f) for f in frequencies_ghz]
    elevations = [float(e) for e in elevations_deg]
    if any((not math.isfinite(f)) or f <= 0.0 for f in frequencies):
        raise ValueError("Frequencies must be positive finite GHz values.")
    if any(not math.isfinite(e) for e in elevations):
        raise ValueError("Elevation angles must all be finite.")

    cfie_alpha = _validate_disabled_2d_cfie_alpha(cfie_alpha)

    mesh_ref_ghz: 'Optional[float]' = None
    if mesh_reference_ghz is not None:
        mesh_ref_ghz = float(mesh_reference_ghz)
        if (not math.isfinite(mesh_ref_ghz)) or mesh_ref_ghz <= 0.0:
            raise ValueError("mesh_reference_ghz must be a positive finite GHz value.")

    rcs_norm_mode = _normalize_rcs_normalization_mode(rcs_normalization_mode)
    solver_method = _normalize_public_2d_solver_method(solver_method)
    _raise_if_untrusted_math_backends()
    _reset_dense_backend_telemetry()

    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)

    shared_cache = (
        _shared_discretization_cache
        if isinstance(_shared_discretization_cache, dict)
        else None
    )
    shared_operator_cache = (
        shared_cache.setdefault("operators", {})
        if shared_cache is not None else None
    )
    prepared = shared_cache.get("prepared") if shared_cache is not None else None
    if prepared is None:
        base_dir = _material_base_dir_for_snapshot(
            geometry_snapshot, material_base_dir
        )
        preflight_report = validate_geometry_snapshot_for_solver(
            geometry_snapshot, base_dir=base_dir, meters_scale=unit_scale
        )
        materials = MaterialLibrary.from_entries(
            geometry_snapshot.get("ibcs", []) or [],
            geometry_snapshot.get("dielectrics", []) or [],
            base_dir=base_dir,
        )
        if shared_cache is not None:
            shared_cache["prepared"] = (
                base_dir, preflight_report, materials, float(unit_scale)
            )
    else:
        base_dir, preflight_report, materials, prepared_scale = prepared
        if abs(float(prepared_scale) - float(unit_scale)) > EPS:
            raise ValueError(
                "A shared 2-D discretization cache cannot be reused across "
                "different geometry units."
            )
    for _msg in list(preflight_report.get('warnings', []) or []):
        materials.warn_once(str(_msg))
    _warn_far_quadrature_override(materials)

    samples: 'List[Dict[str, Any]]' = []
    total_steps = len(frequencies) * (len(elevations) + 1)
    done_steps = 0

    residual_values: 'List[float]' = []
    constraint_residual_values: 'List[float]' = []
    cond_values: 'List[float]' = []
    mesh_reference_values: 'List[float]' = []
    mesh_wavelength_values: 'List[float]' = []
    mesh_max_index_values: 'List[float]' = []
    mesh_material_flags_used: 'Set[int]' = set()
    panel_count_values: 'List[int]' = []
    panel_length_min_values: 'List[float]' = []
    panel_length_max_values: 'List[float]' = []
    elevations_arr = np.asarray(elevations, dtype=float)
    reused_matrix_solve_count = 0
    max_parallel_workers_used = 1
    formulation_label = "2D BIE/MoM coupled dielectric trace formulation (linear Galerkin)"
    thin_layer_evidence = []
    junction_treatment = "none"
    junction_constraints_applied = False
    junction_constraint_residual_applicable = False
    junction_stats = {
        "junction_nodes": 0,
        "junction_constraints": 0,
        "junction_panels": 0,
        "junction_trace_constraints": 0,
        "junction_flux_constraints": 0,
        "junction_orientation_conflict_nodes": 0,
    }

    progress_floor = [0]

    def emit_progress(message: 'str') -> 'None':
        if progress_callback is None:
            return
        try:
            progress_floor[0] = max(progress_floor[0], done_steps)
            progress_callback(progress_floor[0], total_steps, message)
        except Exception:
            pass

    def check_abort() -> 'None':
        if abort_event is not None and abort_event.is_set():
            raise InterruptedError("Solve cancelled by user.")

    check_abort()
    emit_progress("Initializing solver")

    # --- Mesh caching: when mesh_reference_ghz is set, the mesh topology is
    # frequency-independent and can be built once before the frequency loop. ---
    cached_panels: 'Optional[List[Any]]' = None
    cached_mesh: 'Any' = None
    cached_mesh_stats: 'Optional[Dict[str, Any]]' = None
    cached_junction_constraints: 'Optional[np.ndarray]' = None
    cached_junction_stats: 'Optional[Dict[str, Any]]' = None
    cached_mesh_wavelength: 'Optional[float]' = None
    cached_mesh_max_index: 'Optional[float]' = None
    cached_mesh_material_flags: 'List[int]' = []

    fixed_mesh_record = (
        shared_cache.get("fixed_mesh") if shared_cache is not None else None
    )
    if fixed_mesh_record is not None:
        (
            cached_panels,
            cached_mesh,
            cached_mesh_stats,
            cached_mesh_wavelength,
            cached_mesh_max_index,
            cached_mesh_material_flags,
        ) = fixed_mesh_record
    elif mesh_ref_ghz is not None and len(frequencies) > 1:
        (
            ref_lambda,
            cached_mesh_max_index,
            cached_mesh_material_flags,
        ) = _conservative_mesh_wavelength_for_frequencies(
            geometry_snapshot,
            materials,
            set(frequencies) | {mesh_ref_ghz},
        )
        cached_mesh_wavelength = ref_lambda
        ref_k0 = 2.0 * math.pi * mesh_ref_ghz * 1e9 / C0
        cached_panels = _build_panels(
            geometry_snapshot, unit_scale, ref_lambda, max_panels=max_panels,
        )
        # Build preview infos at reference frequency for interface-aware splitting.
        ref_infos = _build_coupled_panel_info(cached_panels, materials, mesh_ref_ghz, pol, ref_k0)
        cached_mesh, cached_mesh_stats = _build_linear_mesh_interface_aware(
            cached_panels, ref_infos,
        )
        cached_mesh_stats = dict(cached_mesh_stats)
        cached_mesh_stats.update(_linear_coupled_node_report(
            cached_mesh,
            _build_linear_coupled_infos(cached_mesh, materials, mesh_ref_ghz, pol, ref_k0),
        ))
        # Junction constraints depend on coupled_infos which may be freq-dependent.
        # Build once at reference freq; topology-based constraints are stable.
        ref_coupled = _build_linear_coupled_infos(cached_mesh, materials, mesh_ref_ghz, pol, ref_k0)
        cached_junction_constraints, cached_junction_stats = _build_linear_junction_constraints(
            cached_mesh, ref_coupled, materialize=False,
        )
        if shared_cache is not None:
            # Panel geometry and interface-aware node topology are independent
            # of TE/TM.  Constitutive factors, junction flux coefficients, all
            # operators, RHS vectors, and factorizations remain channel-local.
            shared_cache["fixed_mesh"] = (
                cached_panels,
                cached_mesh,
                dict(cached_mesh_stats),
                float(cached_mesh_wavelength),
                float(cached_mesh_max_index),
                list(cached_mesh_material_flags),
            )
        materials.warn_once(
            f"Mesh topology cached using the shortest referenced-material "
            f"wavelength across the requested frequencies and the "
            f"{mesh_ref_ghz:g} GHz reference "
            f"({len(cached_panels)} panels, {len(cached_mesh.nodes)} nodes). "
            f"Reusing for {len(frequencies)} frequencies."
        )

    for freq_ghz in frequencies:
        # Specialist callers can also reuse this cache. Keep dense operators
        # only for the current frequency; preserve scalar telemetry.
        if shared_operator_cache is not None:
            if shared_operator_cache.get('_frequency_ghz') != freq_ghz:
                for cache_key in list(shared_operator_cache):
                    if isinstance(cache_key, tuple):
                        del shared_operator_cache[cache_key]
                shared_operator_cache['_frequency_ghz'] = freq_ghz
        check_abort()
        freq_hz = freq_ghz * 1e9
        k0 = 2.0 * math.pi * freq_hz / C0
        mesh_freq_ghz = mesh_ref_ghz if mesh_ref_ghz is not None else float(freq_ghz)

        if cached_panels is not None and cached_mesh is not None:
            panels = cached_panels
            mesh = cached_mesh
            linear_mesh_stats_local = dict(cached_mesh_stats or {})
            lambda_min = float(cached_mesh_wavelength)
            mesh_max_index = float(cached_mesh_max_index)
            mesh_material_flags = list(cached_mesh_material_flags)
        else:
            shared_frequency_meshes = (
                shared_cache.setdefault("frequency_meshes", {})
                if shared_cache is not None else None
            )
            mesh_cache_key = (
                round(float(mesh_freq_ghz), 12),
                int(max_panels),
            )
            frequency_mesh_record = (
                shared_frequency_meshes.get(mesh_cache_key)
                if shared_frequency_meshes is not None else None
            )
            if frequency_mesh_record is not None:
                (
                    panels,
                    mesh,
                    linear_mesh_stats_local,
                    lambda_min,
                    mesh_max_index,
                    mesh_material_flags,
                ) = frequency_mesh_record
                linear_mesh_stats_local = dict(linear_mesh_stats_local)
            else:
                (
                    lambda_min,
                    mesh_max_index,
                    mesh_material_flags,
                ) = _mesh_wavelength_for_snapshot(
                    geometry_snapshot, materials, mesh_freq_ghz
                )
                panels = _build_panels(
                    geometry_snapshot, unit_scale, lambda_min,
                    max_panels=max_panels,
                )
                preview_infos = _build_coupled_panel_info(
                    panels, materials, freq_ghz, pol, k0
                )
                mesh, linear_mesh_stats_local = (
                    _build_linear_mesh_interface_aware(panels, preview_infos)
                )
                linear_mesh_stats_local = dict(linear_mesh_stats_local)
                if shared_frequency_meshes is not None:
                    shared_frequency_meshes.clear()
                    shared_frequency_meshes[mesh_cache_key] = (
                        panels,
                        mesh,
                        dict(linear_mesh_stats_local),
                        float(lambda_min),
                        float(mesh_max_index),
                        list(mesh_material_flags),
                    )

        panel_lengths = np.asarray([p.length for p in panels], dtype=float)
        mesh_reference_values.append(float(mesh_freq_ghz))
        mesh_wavelength_values.append(float(lambda_min))
        mesh_max_index_values.append(float(mesh_max_index))
        mesh_material_flags_used.update(int(flag) for flag in mesh_material_flags)
        panel_count_values.append(int(len(panels)))
        panel_length_min_values.append(float(np.min(panel_lengths)) if len(panel_lengths) else 0.0)
        panel_length_max_values.append(float(np.max(panel_lengths)) if len(panel_lengths) else 0.0)

        coupled_infos = _build_linear_coupled_infos(mesh, materials, freq_ghz, pol, k0)
        _assert_no_type1_sheet_for_mixed(coupled_infos)
        _assert_air_exterior(coupled_infos)
        _assert_supported_te_type2_contours(mesh, coupled_infos, pol)

        # Refuse before any formulation allocates its dense operators.  The
        # classifier is shared with the scheduler, including N-DOF sheet and
        # Robin systems and exact interface-side DOFs for multi-region solves.
        resources = _dense_formulation_resources(mesh, coupled_infos, pol)
        def batch_progress(completed, total):
            if progress_callback is not None:
                progress_floor[0] = max(progress_floor[0], done_steps + completed)
                progress_callback(progress_floor[0], total_steps,
                    "Experimental CPU: solved {} of {} angles at {} GHz".format(completed, total, freq_ghz))
        select_formulation(resources, batch_progress)
        est_gb = _estimate_memory_gb(
            resources["nodes"],
            use_cfie=False,
            n_regions=max(1, resources["n_regions"]),
            system_dofs=resources["system_dofs"],
            operator_matrices=resources["operator_matrices"],
            n_rhs=max(1, len(elevations)),
            solver_method=solver_method, formulation=resources["formulation"],
        )
        memory_limit_gb = _solve_memory_limit_gb()
        if est_gb > memory_limit_gb:
            raise MemoryError(
                _memory_gate_message(
                    est_gb,
                    memory_limit_gb,
                    f"The {resources['formulation']} 2-D solve",
                    (
                        f"Planned system: {resources['system_dofs']} DOFs "
                        f"across {resources['n_regions']} region(s)."
                    ),
                    (
                        "Reduce panel count or frequency, set an appropriate "
                        "mesh_reference_ghz, or reduce the angle batch."
                    ),
                )
            )
        if est_gb > 8.0:
            materials.warn_once(
                f"Estimated peak memory {est_gb:.1f} GB for "
                f"{resources['system_dofs']} {resources['formulation']} "
                "system DOFs. Large problems may cause slowdowns or "
                "out-of-memory errors."
            )

        # --- TYPE 1 sheet dispatch ---
        junction_stats.update(linear_mesh_stats_local)
        junction_stats["linear_node_count"] = int(len(mesh.nodes))
        junction_stats["linear_element_count"] = int(len(mesh.elements))
        # Pure-sheet geometries use a dedicated sheet BIE derived directly
        # from Maxwell's equations (see _solve_tm_sheet / _solve_te_sheet).
        # Mixed sheet+body is rejected above by the _for_mixed guard.
        if _is_all_sheet(coupled_infos):
            formulation_label = (
                "2D sheet BIE (TM: single-layer representation)"
                if pol == "TM"
                else "2D sheet BIE (TE: double-layer / hypersingular representation)"
            )
            sheet_solver = _solve_tm_sheet if pol == "TM" else _solve_te_sheet
            condition_diagnostics = {} if compute_condition_number else None
            if any(info.bc_kind == "thin_layer" for info in coupled_infos):
                eps, mu, thickness = layer_for_mesh(mesh, materials, freq_ghz)
                rcs_lin_vec, amp_vec, sheet_residual, layer_evidence = solve_thin_layer_fields(
                    mesh, k0, elevations_arr, pol, eps, mu, thickness,
                    condition_diagnostics=condition_diagnostics)
                formulation_label = "2D transmitting thin dielectric layer (normal and tangential polarization)"
                thin_layer_evidence.append(dict(layer_evidence, frequency_ghz=float(freq_ghz), polarization=pol))
            else:
                rcs_lin_vec, amp_vec, sheet_residual = sheet_solver(
                    mesh=mesh, infos=coupled_infos, k0=k0,
                    elevations_deg=elevations_arr,
                    condition_diagnostics=condition_diagnostics)
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), sheet_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append({
                    "frequency_ghz": float(freq_ghz),
                    "theta_inc_deg": float(elev_deg),
                    "theta_scat_deg": float(elev_deg),
                    "rcs_linear": float(rcs_lin_vec[idx]),
                    "rcs_db": float(rcs_db_vec[idx]),
                    "rcs_amp_real": float(np.real(amp_val)),
                    "rcs_amp_imag": float(np.imag(amp_val)),
                    "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                    "linear_residual": residual_local,
                })
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Sheet BIE solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue
        # --- Mixed TYPE 1 sheet + TYPE 2 PEC dispatch ---
        # Handled with a unified SLP (TM) or DLP (TE) representation over
        # both surface types with an element-weighted impedance term.
        if _is_sheet_plus_pec(coupled_infos):
            formulation_label = (
                "2D mixed sheet+PEC BIE (TM: unified SLP representation)"
                if pol == "TM"
                else "2D mixed sheet+PEC BIE (TE: unified DLP / hypersingular representation)"
            )
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, mixed_residual = _solve_mixed_sheet_pec(
                mesh=mesh, infos=coupled_infos, pol=pol,
                k0=k0, elevations_deg=elevations_arr,
                condition_diagnostics=condition_diagnostics,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), mixed_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append({
                    "frequency_ghz": float(freq_ghz),
                    "theta_inc_deg": float(elev_deg),
                    "theta_scat_deg": float(elev_deg),
                    "rcs_linear": float(rcs_lin_vec[idx]),
                    "rcs_db": float(rcs_db_vec[idx]),
                    "rcs_amp_real": float(np.real(amp_val)),
                    "rcs_amp_imag": float(np.imag(amp_val)),
                    "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                    "linear_residual": residual_local,
                })
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Mixed sheet+PEC BIE solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        if cached_panels is None:
            linear_mesh_stats_local.update(_linear_coupled_node_report(mesh, coupled_infos))
        done_steps += 1
        emit_progress(f"Assembled linear/Galerkin coupled operators at {freq_ghz:g} GHz")

        if cached_junction_constraints is not None and cached_junction_stats is not None:
            linear_junction_constraints = cached_junction_constraints
            linear_junction_stats = dict(cached_junction_stats)
        else:
            linear_junction_constraints, linear_junction_stats = _build_linear_junction_constraints(
                mesh, coupled_infos, materialize=False,
            )
        junction_stats.update(linear_mesh_stats_local)
        junction_stats.update(linear_junction_stats)
        orientation_conflicts = int(linear_junction_stats.get("junction_orientation_conflict_nodes", 0))
        if orientation_conflicts > 0:
            raise ValueError(
                f"Detected {orientation_conflicts} cross-segment junction node(s) with "
                "inconsistent segment orientation. Refusing to solve because "
                "the material-side trace assignment is physically ambiguous; "
                "fix the geometry so shared junctions have a consistent "
                "plus/minus side assignment."
            )
        if int(linear_junction_stats.get("junction_constraints", 0)) > 0:
            candidate_count = int(linear_junction_stats.get("junction_constraints", 0))
            if _is_multi_region(coupled_infos):
                junction_treatment = "implicit_multi_region_indirect"
                materials.warn_once(
                    (
                        f"Detected {candidate_count} physical trace/flux junction "
                        "relation(s). The active multi-region indirect formulation "
                        "uses interface-specific SLP densities, so that diagnostic "
                        "trace/flux matrix is not applied to its different unknowns; "
                        "junction coupling is represented implicitly by the shared "
                        "regional potentials."
                    )
                )
            else:
                junction_treatment = "diagnostic_only"
                materials.warn_once(
                    (
                        f"Detected {candidate_count} physical trace/flux junction "
                        "relation(s); the diagnostic matrix is not applied by the "
                        "active formulation."
                    )
                )

        check_abort()

        # --- TE Robin path: MFIE for PEC and IBC surfaces ---
        use_te_robin_mfie = (pol == 'TE' and _is_all_robin(coupled_infos))

        if use_te_robin_mfie:
            formulation_label = "2D MFIE TE Robin (SLP representation)"
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, mfie_residual = select_solver(_solve_te_robin_mfie)(
                mesh=mesh,
                infos=coupled_infos,
                pol=pol,
                k0=k0,
                elevations_deg=elevations_arr,
                solver_method=solver_method,
                condition_diagnostics=condition_diagnostics,
                operator_cache=shared_operator_cache,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), mfie_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"MFIE solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        # --- Multi-region indirect formulation (layered coatings) ---
        use_multi_region = _is_multi_region(coupled_infos)

        if use_multi_region:
            formulation_label = "2D multi-region indirect SLP formulation (layered coating)"
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, multi_residual, _ = select_solver(_solve_multi_region_indirect)(
                mesh=mesh,
                infos=coupled_infos,
                pol=pol,
                k0=k0,
                elevations_deg=elevations_arr,
                solver_method=solver_method,
                condition_diagnostics=condition_diagnostics,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), multi_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Multi-region solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        # --- Dielectric indirect formulation ---
        use_dielectric_indirect = _is_single_dielectric_body(coupled_infos)

        if use_dielectric_indirect:
            formulation_label = "2D indirect two-density dielectric formulation"
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, diel_residual = select_solver(_solve_dielectric_indirect)(
                mesh=mesh,
                infos=coupled_infos,
                pol=pol,
                k0=k0,
                elevations_deg=elevations_arr,
                condition_diagnostics=condition_diagnostics,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), diel_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Dielectric solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        # --- All-Robin (PEC, IBC, or mixed PEC+IBC) SLP formulation ---
        # _solve_robin_bie handles every all-robin case correctly: TE PEC via
        # the alpha=0 MFIE limit, TM IBC via the standard Robin BIE, and TM
        # PEC via the per-row EFIE override (the alpha->infty limit).  TE
        # all-robin already short-circuits to the dedicated MFIE solver
        # above, so we only need to dispatch the remaining all-robin cases
        # here (primarily TM PEC bodies, which the coupled-trace path
        # does not handle correctly).
        use_robin_bie = _is_all_robin(coupled_infos)

        if use_robin_bie:
            formulation_label = (
                "2D Robin-BIE (SLP representation; element-weighted IBC, TM-PEC EFIE override)"
                if pol == 'TM'
                else "2D Robin-BIE IBC formulation (SLP representation)"
            )
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, robin_residual = select_solver(_solve_robin_bie)(
                mesh=mesh,
                infos=coupled_infos,
                pol=pol,
                k0=k0,
                elevations_deg=elevations_arr,
                condition_diagnostics=condition_diagnostics,
                operator_cache=shared_operator_cache,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), robin_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Robin-BIE solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        # --- No formulation matched ---
        # The dispatch above is exhaustive for physical inputs: sheets, TE
        # all-Robin (MFIE), multi-region (layered / mixed robin+transmission /
        # multiple bodies), single dielectric body, and TM/mixed all-Robin.
        # Only degenerate configurations (e.g. TYPE 5 interfaces with no
        # exterior boundary) can reach this point.  The old coupled-trace
        # fallback that lived here had inconsistent Green-identity signs and
        # a BC row that solved the wrong problem for TM PEC, so failing
        # loudly is strictly better than running it.
        raise ValueError(
            "Geometry did not match any supported monostatic formulation "
            "(sheet, all-Robin PEC/IBC, dielectric body, or multi-region). "
            "Check that the geometry encloses regions with a boundary to air "
            "(TYPE 5-only configurations without an exterior interface are "
            "not solvable)."
        )

    residual_norm_max, residual_norm_mean, residual_nonfinite_count = (
        _summarize_residuals(residual_values)
    )
    condition_est_computed = bool(compute_condition_number)

    metadata: 'Dict[str, Any]' = {
        "source_path": str(geometry_snapshot.get("source_path", "") or ""),
        "segment_count": int(len(geometry_snapshot.get("segments", []) or [])),
        "panel_count": int(np.max(panel_count_values)) if panel_count_values else 0,
        "panel_count_min": int(np.min(panel_count_values)) if panel_count_values else 0,
        "panel_count_max": int(np.max(panel_count_values)) if panel_count_values else 0,
        "panel_length_min_m": float(np.min(panel_length_min_values)) if panel_length_min_values else 0.0,
        "panel_length_max_m": float(np.max(panel_length_max_values)) if panel_length_max_values else 0.0,
        "mesh_reference_ghz": float(mesh_reference_values[0]) if len(set(round(v, 12) for v in mesh_reference_values)) == 1 and mesh_reference_values else None,
        "mesh_reference_ghz_min": float(np.min(mesh_reference_values)) if mesh_reference_values else 0.0,
        "mesh_reference_ghz_max": float(np.max(mesh_reference_values)) if mesh_reference_values else 0.0,
        "mesh_wavelength_m": float(mesh_wavelength_values[0]) if len(set(round(v, 15) for v in mesh_wavelength_values)) == 1 and mesh_wavelength_values else None,
        "mesh_wavelength_min_m": float(np.min(mesh_wavelength_values)) if mesh_wavelength_values else 0.0,
        "mesh_wavelength_max_m": float(np.max(mesh_wavelength_values)) if mesh_wavelength_values else 0.0,
        "mesh_max_refractive_index": float(np.max(mesh_max_index_values)) if mesh_max_index_values else 1.0,
        "mesh_material_flags": sorted(mesh_material_flags_used),
        "polarization_internal": pol,
        "polarization_user": _canonical_user_polarization_label(polarization),
        "polarization_aliases": [_canonical_user_polarization_label(polarization)],
        "polarization_export": _canonical_user_polarization_label(polarization),
        "polarization_export_alias": _primary_alias_for_user_polarization(polarization),
        "rcs_normalization_mode": rcs_norm_mode,
        "formulation": formulation_label,
        "solver_method": "dense_lu",
        "solver_method_requested": str(solver_method),
        "residual_norm_max": residual_norm_max,
        "residual_norm_mean": residual_norm_mean,
        "residual_nonfinite_count": residual_nonfinite_count,
        "constraint_residual_norm_max": float(np.max(constraint_residual_values)) if constraint_residual_values else 0.0,
        "constraint_residual_norm_mean": float(np.mean(constraint_residual_values)) if constraint_residual_values else 0.0,
        "constraint_residual_applicable": bool(junction_constraint_residual_applicable),
        "condition_est_max": float(np.max(cond_values)) if cond_values else float("nan"),
        "condition_est_mean": float(np.mean(cond_values)) if cond_values else float("nan"),
        "condition_est_computed": bool(condition_est_computed),
        "condition_estimator": (
            "equilibrated_1norm_lu_onenormest"
            if condition_est_computed else "not_requested"
        ),
        "thin_layer": thin_layer_evidence,
        "warnings": list(materials.warnings),
        "warning_count": int(len(materials.warnings)),
        "math_backend_real_bessel": _BESSEL.backend_name,
        "math_backend_complex_hankel": _complex_hankel_backend_name(),
        "reused_matrix_solve_count": int(reused_matrix_solve_count),
        "shared_operator_cache_enabled": bool(
            shared_operator_cache is not None
        ),
        "shared_operator_cache_hits": int(
            shared_operator_cache.get("_hits", 0)
            if shared_operator_cache is not None else 0
        ),
        "shared_operator_cache_stores": int(
            shared_operator_cache.get("_stores", 0)
            if shared_operator_cache is not None else 0
        ),
        "parallel_elevation_solve_count": 0,
        "max_parallel_workers_used": int(max_parallel_workers_used),
        "mesh_reference_frequency_used": bool(mesh_ref_ghz is not None),
        "cfie_alpha": float(cfie_alpha),
        "junction_nodes": int(junction_stats.get("junction_nodes", 0)),
        "junction_constraints": int(junction_stats.get("junction_constraints", 0)),
        "junction_constraint_candidates": int(junction_stats.get("junction_constraints", 0)),
        "junction_constraints_applied": bool(junction_constraints_applied),
        "junction_constraints_applied_count": 0,
        "junction_treatment": junction_treatment,
        "junction_panels": int(junction_stats.get("junction_panels", 0)),
        "junction_trace_constraints": int(junction_stats.get("junction_trace_constraints", 0)),
        "junction_flux_constraints": int(junction_stats.get("junction_flux_constraints", 0)),
        "junction_orientation_conflict_nodes": int(junction_stats.get("junction_orientation_conflict_nodes", 0)),
        "linear_node_count": int(junction_stats.get("linear_node_count", 0)),
        "linear_element_count": int(junction_stats.get("linear_element_count", 0)),
        "shared_node_count": int(junction_stats.get("shared_node_count", 0)),
        "split_node_count": int(junction_stats.get("split_node_count", 0)),
        "split_boundary_primitive_count": int(junction_stats.get("split_boundary_primitive_count", 0)),
        "multi_signature_node_count": int(junction_stats.get("multi_signature_node_count", 0)),
        "preflight": dict(preflight_report),
        **_dense_backend_summary(),
    }

    metadata["amplitude_version"] = RCS_AMPLITUDE_VERSION
    quality_gate = evaluate_quality_gate(metadata, thresholds=quality_thresholds)
    metadata["quality_gate"] = quality_gate
    if strict_quality_gate and not bool(quality_gate.get("passed", False)):
        reason = str(quality_gate.get("reason", "quality gate failed"))
        raise ValueError(f"Quality gate failed: {reason}")

    return {
        "solver": "2d_bie_mom_rcs",
        "scattering_mode": "monostatic",
        "amplitude_convention": RCS_AMPLITUDE_CONVENTION,
        "amplitude_version": RCS_AMPLITUDE_VERSION,
        "polarization": _canonical_user_polarization_label(polarization),
        "polarization_export": _canonical_user_polarization_label(polarization),
        "samples": samples,
        "metadata": metadata,
    }


_CO_POLARIZED_2D_CHANNELS = (("VV", "TE"), ("HH", "TM"))


def _co_polarized_progress_callback(
    progress_callback: 'Optional[Callable[[int, int, str], None]]',
    channel_index: 'int',
    export_polarization: 'str',
) -> 'Optional[Callable[[int, int, str], None]]':
    """Map one scalar-channel progress stream onto the combined solve."""

    if progress_callback is None:
        return None

    def _mapped(done: 'int', total: 'int', message: 'str') -> 'None':
        total_i = max(1, int(total))
        done_i = max(0, min(int(done), total_i))
        try:
            progress_callback(
                int(channel_index) * total_i + done_i,
                len(_CO_POLARIZED_2D_CHANNELS) * total_i,
                f"{export_polarization}: {message}",
            )
        except Exception:
            pass

    return _mapped


def _finite_metadata_max(
    channel_metadata: 'Dict[str, Dict[str, Any]]',
    key: 'str',
    default: 'float' = 0.0,
) -> 'float':
    values = []
    for metadata in channel_metadata.values():
        try:
            value = float(metadata.get(key, float("nan")))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(max(values)) if values else float(default)


def _merge_co_polarized_2d_results(
    channel_results: 'Dict[str, Dict[str, Any]]',
) -> 'Dict[str, Any]':
    """Merge exact TE/TM solves without inventing a selected polarization."""

    expected_channels = [item[0] for item in _CO_POLARIZED_2D_CHANNELS]
    if set(channel_results) != set(expected_channels):
        raise ValueError(
            "A co-polarized 2-D result requires exactly VV<-TE and HH<-TM."
        )

    co_solved_samples: 'Dict[str, List[Dict[str, Any]]]' = {}
    channel_keys: 'Optional[Set[Tuple[float, float, float]]]' = None
    flattened: 'List[Dict[str, Any]]' = []
    channel_metadata: 'Dict[str, Dict[str, Any]]' = {}
    scattering_modes: 'Set[str]' = set()
    solver_names: 'Set[str]' = set()
    amplitude_conventions: 'Set[str]' = set()

    for export_pol, internal_pol in _CO_POLARIZED_2D_CHANNELS:
        result = channel_results[export_pol]
        raw_samples = list(result.get("samples", []) or [])
        if not raw_samples:
            raise ValueError(
                f"The co-polarized 2-D solve returned no {export_pol} samples."
            )
        labeled_samples = []
        keys: 'Set[Tuple[float, float, float]]' = set()
        for row in raw_samples:
            copied = dict(row)
            copied["polarization"] = export_pol
            copied["polarization_internal"] = internal_pol
            key = (
                float(copied["frequency_ghz"]),
                float(copied["theta_inc_deg"]),
                float(copied["theta_scat_deg"]),
            )
            if key in keys:
                raise ValueError(
                    f"Duplicate {export_pol} 2-D sample at f/inc/scat={key}."
                )
            keys.add(key)
            labeled_samples.append(copied)
        if channel_keys is None:
            channel_keys = keys
        elif keys != channel_keys:
            missing = sorted(channel_keys - keys)
            extra = sorted(keys - channel_keys)
            raise ValueError(
                "TE/TM 2-D solves did not return the same physical grid "
                f"(first missing={missing[:1]}, first extra={extra[:1]})."
            )
        labeled_samples.sort(key=lambda row: (
            float(row["frequency_ghz"]),
            float(row["theta_inc_deg"]),
            float(row["theta_scat_deg"]),
        ))
        co_solved_samples[export_pol] = labeled_samples
        flattened.extend(labeled_samples)
        channel_metadata[export_pol] = dict(result.get("metadata", {}) or {})
        scattering_modes.add(str(result.get("scattering_mode", "")))
        solver_names.add(str(result.get("solver", "")))
        amplitude_conventions.add(str(result.get("amplitude_convention", "")))

    if len(scattering_modes) != 1 or len(solver_names) != 1 \
            or len(amplitude_conventions) != 1:
        raise ValueError(
            "TE/TM channel results disagree on solver, scattering mode, or "
            "complex-amplitude convention."
        )

    channel_order = {label: index for index, label in enumerate(expected_channels)}
    flattened.sort(key=lambda row: (
        float(row["frequency_ghz"]),
        float(row["theta_inc_deg"]),
        float(row["theta_scat_deg"]),
        channel_order[str(row["polarization"])],
    ))

    warnings = []
    gpu_fallback_reasons = []
    gpu_devices = []
    for export_pol in expected_channels:
        for warning in list(channel_metadata[export_pol].get("warnings", []) or []):
            text = str(warning)
            if text not in warnings:
                warnings.append(text)
        for reason in list(channel_metadata[export_pol].get(
            "dense_gpu_fallback_reasons", []
        ) or []):
            text = str(reason)
            if text and text not in gpu_fallback_reasons:
                gpu_fallback_reasons.append(text)
        for device in list(channel_metadata[export_pol].get(
            "dense_gpu_devices", []
        ) or []):
            text = str(device)
            if text and text not in gpu_devices:
                gpu_devices.append(text)

    quality_by_channel = {
        export_pol: dict(
            channel_metadata[export_pol].get("quality_gate", {}) or {}
        )
        for export_pol in expected_channels
    }
    quality_passed = all(
        bool(quality_by_channel[label].get("passed", False))
        for label in expected_channels
    )
    quality_gate = {
        "passed": bool(quality_passed),
        "channels": quality_by_channel,
        "reason": (
            "Both VV<-TE and HH<-TM discrete linear-system quality gates passed"
            if quality_passed else
            "At least one co-polarized 2-D channel failed its quality gate"
        ),
    }

    mesh_by_channel = {
        export_pol: dict(
            channel_metadata[export_pol].get("mesh_convergence", {}) or {}
        )
        for export_pol in expected_channels
        if channel_metadata[export_pol].get("mesh_convergence")
    }
    mesh_certified = bool(mesh_by_channel) and all(
        bool(channel_metadata[label].get("mesh_convergence_certified", False))
        and bool(mesh_by_channel.get(label, {}).get("passed", False))
        for label in expected_channels
    )

    first_metadata = channel_metadata[expected_channels[0]]
    linear_backends = {
        str(channel_metadata[label].get("linear_backend", "cpu"))
        for label in expected_channels
    }
    metadata: 'Dict[str, Any]' = {
        "source_path": first_metadata.get("source_path", ""),
        "segment_count": first_metadata.get("segment_count", 0),
        "panel_count": int(_finite_metadata_max(channel_metadata, "panel_count")),
        "panel_count_min": int(_finite_metadata_max(
            channel_metadata, "panel_count_min"
        )),
        "panel_count_max": int(_finite_metadata_max(
            channel_metadata, "panel_count_max"
        )),
        "polarizations": list(expected_channels),
        "polarization_internal": [item[1] for item in _CO_POLARIZED_2D_CHANNELS],
        "polarization_mapping": {"VV": "TE", "HH": "TM"},
        "formulation": "co-polarized 2-D BIE/MoM",
        "formulations": {
            label: channel_metadata[label].get("formulation", "")
            for label in expected_channels
        },
        "solver_method": first_metadata.get("solver_method", ""),
        "solver_method_requested": first_metadata.get(
            "solver_method_requested", ""
        ),
        "residual_norm_max": _finite_metadata_max(
            channel_metadata, "residual_norm_max"
        ),
        "constraint_residual_norm_max": _finite_metadata_max(
            channel_metadata, "constraint_residual_norm_max"
        ),
        "condition_est_max": _finite_metadata_max(
            channel_metadata, "condition_est_max", default=float("nan")
        ),
        "condition_est_computed": all(
            bool(channel_metadata[label].get("condition_est_computed", False))
            for label in expected_channels
        ),
        "warnings": warnings,
        "warning_count": len(warnings),
        "preflight": dict(first_metadata.get("preflight", {}) or {}),
        "quality_gate": quality_gate,
        "channel_metadata": channel_metadata,
        "thin_layer": {label: channel_metadata[label].get("thin_layer", []) for label in expected_channels},
        "amplitude_version": RCS_AMPLITUDE_VERSION,
        "co_solve_shared_discretization": True,
        "shared_operator_cache_enabled": any(
            bool(channel_metadata[label].get(
                "shared_operator_cache_enabled", False
            ))
            for label in expected_channels
        ),
        "shared_operator_cache_hits": int(_finite_metadata_max(
            channel_metadata, "shared_operator_cache_hits"
        )),
        "shared_operator_cache_stores": int(_finite_metadata_max(
            channel_metadata, "shared_operator_cache_stores"
        )),
        "linear_backend": (
            next(iter(linear_backends))
            if len(linear_backends) == 1 else "mixed"
        ),
        "dense_gpu_solve_count": int(sum(
            int(channel_metadata[label].get("dense_gpu_solve_count", 0))
            for label in expected_channels
        )),
        "dense_cpu_solve_count": int(sum(
            int(channel_metadata[label].get("dense_cpu_solve_count", 0))
            for label in expected_channels
        )),
        "dense_gpu_fallback_reasons": gpu_fallback_reasons,
        "dense_gpu_devices": gpu_devices,
        "dense_largest_system": int(_finite_metadata_max(
            channel_metadata, "dense_largest_system"
        )),
        "mesh_convergence_certified": mesh_certified,
        "certified_entry_point": all(
            bool(channel_metadata[label].get("certified_entry_point", False))
            for label in expected_channels
        ),
        "survey_mode": all(
            bool(channel_metadata[label].get("survey_mode", False))
            for label in expected_channels
        ),
        "published_mesh": first_metadata.get("published_mesh", ""),
    }
    if mesh_by_channel:
        base_quality_by_channel = {
            label: dict(
                mesh_by_channel[label].get("base_quality_gate", {}) or {}
            )
            for label in expected_channels
        }
        fine_quality_by_channel = {
            label: dict(
                mesh_by_channel[label].get("fine_quality_gate", {}) or {}
            )
            for label in expected_channels
        }
        aggregate_mesh = {
            "schema": "ghost.solver.mesh-convergence.co-polarized.v1",
            "passed": mesh_certified,
            "published_mesh": "fine" if mesh_certified else "",
            "channels": mesh_by_channel,
            # ``polarizations`` is retained as a semantic alias for older
            # strict workflow consumers that predate the result-level
            # ``co_solved_samples`` contract.
            "polarizations": mesh_by_channel,
            "base_quality_gate": {
                "passed": all(
                    bool(base_quality_by_channel[label].get("passed", False))
                    for label in expected_channels
                ),
                "channels": base_quality_by_channel,
            },
            "fine_quality_gate": {
                "passed": all(
                    bool(fine_quality_by_channel[label].get("passed", False))
                    for label in expected_channels
                ),
                "channels": fine_quality_by_channel,
            },
            "reason": (
                "VV<-TE and HH<-TM mesh-convergence gates passed"
                if mesh_certified else
                "At least one co-polarized channel lacks a passing mesh certificate"
            ),
        }
        for metric in (
            "rms_db", "max_abs_db", "complex_rms", "complex_max",
            "phase_rms_deg", "phase_max_deg", "base_panel_count",
            "fine_panel_count", "panel_refinement_ratio", "fine_factor",
        ):
            values = []
            for channel_mesh in mesh_by_channel.values():
                try:
                    value = float(channel_mesh.get(metric, float("nan")))
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            if values:
                aggregate_mesh[metric] = max(values)
        metadata["mesh_convergence"] = aggregate_mesh

    return {
        "solver": next(iter(solver_names)),
        "scattering_mode": next(iter(scattering_modes)),
        "amplitude_convention": next(iter(amplitude_conventions)),
        "amplitude_version": RCS_AMPLITUDE_VERSION,
        "rcs_log_unit": "dBke",
        "rcs_linear_quantity": "sigma_2d",
        "polarizations": list(expected_channels),
        "polarization_mapping": {"VV": "TE", "HH": "TM"},
        "samples": flattened,
        "co_solved_samples": co_solved_samples,
        "metadata": metadata,
    }


def _frequency_local_co_solve(solve, kwargs):
    """Finish both channels (and both certification meshes) per frequency."""
    kwargs = dict(kwargs)
    frequencies = list(kwargs.pop('frequencies_ghz'))
    if len({float(value) for value in frequencies}) != len(frequencies):
        raise ValueError('Duplicate frequencies are not supported in a co-polarized result grid.')
    progress = kwargs.pop('progress_callback', None)
    results = []
    for index, frequency in enumerate(frequencies):
        def report(done, total, message):
            if progress is not None:
                progress(index * 1000 + int(1000 * done / max(total, 1)),
                         1000 * len(frequencies), message)
        results.append(solve(frequencies_ghz=[frequency], progress_callback=report, **kwargs))

    def combine(items, field=''):
        first = items[0]
        if all(isinstance(v, dict) for v in items):
            return {key: combine([v[key] for v in items if key in v], key)
                    for key in dict.fromkeys(k for v in items for k in v)}
        if all(isinstance(v, bool) for v in items):
            return all(items)
        if all(isinstance(v, (int, float)) for v in items):
            finite = [v for v in items if math.isfinite(v)]
            if finite and ('_min' in field or field.startswith('min_')):
                return min(finite)
            if finite and field.endswith('_mean'):
                return sum(finite) / len(finite)
            return max(finite) if finite else float('nan')
        if all(isinstance(v, list) for v in items):
            merged = []
            for value in items:
                for entry in value:
                    if entry not in merged:
                        merged.append(entry)
            return merged
        return first

    result = dict(results[0])
    result['samples'] = [row for value in results for row in value['samples']]
    result['co_solved_samples'] = {
        pol: [row for value in results for row in value['co_solved_samples'][pol]]
        for pol in ('VV', 'HH')}
    key = lambda row: (float(row['frequency_ghz']), float(row['theta_inc_deg']),
                       float(row['theta_scat_deg']), 0 if row['polarization'] == 'VV' else 1)
    result['samples'].sort(key=key)
    for rows in result['co_solved_samples'].values():
        rows.sort(key=key)
    metadata = combine([value['metadata'] for value in results])
    # Preserve every frequency's evidence, including its channel mesh gates.
    metadata['frequency_metadata'] = [
        {'frequency_ghz': float(frequency), 'metadata': value['metadata']}
        for frequency, value in zip(frequencies, results)]
    metadata['operator_cache_scope'] = 'one_frequency'
    metadata['panel_count_min'] = min(value['metadata']['panel_count_min'] for value in results)
    for key in ('shared_operator_cache_hits', 'shared_operator_cache_stores',
                'reused_matrix_solve_count', 'residual_nonfinite_count'):
        metadata[key] = sum(value['metadata'].get(key, 0) for value in results)
    metadata['warning_count'] = len(metadata.get('warnings', []))
    result['metadata'] = metadata
    return result


@profiled_solve
@experimental_monostatic
def solve_monostatic_rcs_2d(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    strict_quality_gate: 'bool' = True,
    compute_condition_number: 'bool' = False,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    rcs_normalization_mode: 'str' = RCS_NORM_MODE_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "direct",
) -> 'Dict[str, Any]':
    """Solve the complete 2-D monostatic co-polarized response.

    This is the canonical low-level result contract: VV is the TE scalar
    problem, HH is the TM scalar problem, and neither channel can be omitted or
    selected by a user-facing polarization argument.  Geometry validation,
    materials, panels, and interface-aware mesh topology are reused; physical
    operators and factorizations remain separate because their boundary
    conditions differ.
    """

    if len(frequencies_ghz) > 1:
        return _frequency_local_co_solve(solve_monostatic_rcs_2d, locals())
    shared_cache: 'Dict[str, Any]' = {}
    channel_results = {}
    for index, (export_pol, internal_pol) in enumerate(
        _CO_POLARIZED_2D_CHANNELS
    ):
        channel_results[export_pol] = solve_monostatic_rcs_2d_single_polarization(
            geometry_snapshot=geometry_snapshot,
            frequencies_ghz=frequencies_ghz,
            elevations_deg=elevations_deg,
            polarization=internal_pol,
            geometry_units=geometry_units,
            material_base_dir=material_base_dir,
            progress_callback=_co_polarized_progress_callback(
                progress_callback, index, export_pol
            ),
            quality_thresholds=quality_thresholds,
            strict_quality_gate=strict_quality_gate,
            compute_condition_number=compute_condition_number,
            max_panels=max_panels,
            mesh_reference_ghz=mesh_reference_ghz,
            rcs_normalization_mode=rcs_normalization_mode,
            cfie_alpha=0.0,
            abort_event=abort_event,
            solver_method=solver_method,
            _shared_discretization_cache=shared_cache,
        )
    return _merge_co_polarized_2d_results(channel_results)


@profiled_solve
def solve_bistatic_rcs_2d_single_polarization(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    incidence_angles_deg: 'List[float]',
    observation_angles_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    strict_quality_gate: 'bool' = True,
    compute_condition_number: 'bool' = False,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
) -> 'Dict[str, Any]':
    """
    Explicit single-polarization bistatic 2-D RCS diagnostic.

    Production callers should use :func:`solve_bistatic_rcs_2d`, which always
    returns both VV<-TE and HH<-TM channels on the same physical grid.

    For each frequency and incidence angle, solves the boundary integral equation
    and evaluates the far-field RCS at all requested observation angles.

    Returns samples with ``theta_inc_deg != theta_scat_deg`` in general.
    Compatible with ``export_result_to_grim`` which splits by incidence angle.
    """
    if str(solver_method).strip().lower() == EXPERIMENTAL_METHOD:
        raise ValueError("Experimental CPU supports 2D monostatic fields only.")

    if not frequencies_ghz:
        raise ValueError("At least one frequency is required.")
    if not incidence_angles_deg:
        raise ValueError("At least one incidence angle is required.")
    if not observation_angles_deg:
        raise ValueError("At least one observation angle is required.")

    frequencies = [float(f) for f in frequencies_ghz]
    inc_angles = [float(a) for a in incidence_angles_deg]
    obs_angles = [float(a) for a in observation_angles_deg]
    if any((not math.isfinite(f)) or f <= 0.0 for f in frequencies):
        raise ValueError("Frequencies must be positive finite GHz values.")
    if any(not math.isfinite(a) for a in inc_angles):
        raise ValueError("Incidence angles must all be finite.")
    if any(not math.isfinite(a) for a in obs_angles):
        raise ValueError("Observation angles must all be finite.")

    cfie_alpha = _validate_disabled_2d_cfie_alpha(cfie_alpha)

    solver_method = _normalize_public_2d_solver_method(solver_method)
    _raise_if_untrusted_math_backends()
    _reset_dense_backend_telemetry()
    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)
    base_dir = _material_base_dir_for_snapshot(
        geometry_snapshot, material_base_dir
    )

    mesh_ref_ghz = float(mesh_reference_ghz) if mesh_reference_ghz is not None else None
    if mesh_ref_ghz is not None and (
        not math.isfinite(mesh_ref_ghz) or mesh_ref_ghz <= 0.0
    ):
        raise ValueError("mesh_reference_ghz must be a positive finite GHz value.")

    preflight_report = validate_geometry_snapshot_for_solver(geometry_snapshot, base_dir=base_dir, meters_scale=unit_scale)
    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    for _msg in list(preflight_report.get("warnings", []) or []):
        materials.warn_once(str(_msg))
    _warn_far_quadrature_override(materials)

    samples: 'List[Dict[str, Any]]' = []
    residual_values: 'List[float]' = []
    cond_values: 'List[float]' = []
    total_steps = len(frequencies) * len(inc_angles)
    done_steps = 0
    obs_arr = np.asarray(obs_angles, dtype=float)
    thin_layer_evidence = []
    mesh_wavelength_values: 'List[float]' = []
    mesh_max_index_values: 'List[float]' = []
    mesh_material_flags_used: 'Set[int]' = set()
    conservative_mesh = None
    if mesh_ref_ghz is not None:
        conservative_mesh = _conservative_mesh_wavelength_for_frequencies(
            geometry_snapshot,
            materials,
            set(frequencies) | {mesh_ref_ghz},
        )

    def check_abort() -> 'None':
        if abort_event is not None and abort_event.is_set():
            raise InterruptedError("Solve cancelled by user.")

    def emit_progress(msg: 'str') -> 'None':
        if progress_callback is not None:
            try:
                progress_callback(done_steps, total_steps, msg)
            except Exception:
                pass

    for freq_ghz in frequencies:
        frequency_condition_recorded = False
        check_abort()
        freq_hz = freq_ghz * 1e9
        k0 = 2.0 * math.pi * freq_hz / C0
        mesh_freq_ghz = mesh_ref_ghz if mesh_ref_ghz is not None else float(freq_ghz)
        if conservative_mesh is None:
            (
                lambda_min,
                mesh_max_index,
                mesh_material_flags,
            ) = _mesh_wavelength_for_snapshot(
                geometry_snapshot, materials, mesh_freq_ghz
            )
        else:
            (
                lambda_min,
                mesh_max_index,
                mesh_material_flags,
            ) = conservative_mesh
        mesh_wavelength_values.append(float(lambda_min))
        mesh_max_index_values.append(float(mesh_max_index))
        mesh_material_flags_used.update(int(flag) for flag in mesh_material_flags)

        panels = _build_panels(geometry_snapshot, unit_scale, lambda_min, max_panels=max_panels)
        preview_infos = _build_coupled_panel_info(panels, materials, freq_ghz, pol, k0)
        mesh, _ = _build_linear_mesh_interface_aware(panels, preview_infos)
        coupled_infos = _build_linear_coupled_infos(mesh, materials, freq_ghz, pol, k0)
        _assert_no_type1_sheet_for_mixed(coupled_infos)
        _assert_air_exterior(coupled_infos)
        _assert_supported_te_type2_contours(mesh, coupled_infos, pol)
        nnodes = len(mesh.nodes)

        use_sheet = _is_all_sheet(coupled_infos)
        use_mixed_sheet = _is_sheet_plus_pec(coupled_infos)
        use_te_robin_mfie = (pol == 'TE' and _is_all_robin(coupled_infos))
        # TM all-Robin (PEC, IBC, or mixed) must use the Robin BIE, exactly as
        # in the monostatic dispatch: the coupled-trace fallback below applies
        # a mass row to q for TYPE 2 surfaces, which ignores the impedance
        # entirely (TM IBC used to return bit-for-bit the PEC answer here).
        use_tm_robin_bie = (pol == 'TM' and _is_all_robin(coupled_infos))
        use_diel_indirect = _is_single_dielectric_body(coupled_infos) and not _is_multi_region(coupled_infos)
        use_multi_region = _is_multi_region(coupled_infos)

        resources = _dense_formulation_resources(mesh, coupled_infos, pol)
        est_gb = _estimate_memory_gb(
            resources["nodes"],
            use_cfie=False,
            n_regions=max(1, resources["n_regions"]),
            system_dofs=resources["system_dofs"],
            operator_matrices=resources["operator_matrices"],
            n_rhs=len(inc_angles),
        )
        # Returned dictionaries persist across frequencies; include them as
        # well as the rectangular complex/power projection workspace.
        est_gb += (len(inc_angles) * len(obs_angles) *
                   (1024 * len(frequencies) + 64)) / (1024 ** 3)
        memory_limit_gb = _solve_memory_limit_gb()
        if est_gb > memory_limit_gb:
            raise MemoryError(
                _memory_gate_message(
                    est_gb,
                    memory_limit_gb,
                    f"The {resources['formulation']} bistatic 2-D solve",
                    (
                        f"Planned system: {resources['system_dofs']} DOFs "
                        f"across {resources['n_regions']} region(s)."
                    ),
                    (
                        "Reduce panel count or frequency, set an appropriate "
                        "mesh_reference_ghz, or reduce the incidence/observation grid."
                    ),
                )
            )
        if est_gb > 8.0:
            materials.warn_once(
                f"Estimated peak memory {est_gb:.1f} GB for "
                f"{resources['system_dofs']} {resources['formulation']} "
                "system DOFs. Large problems may cause slowdowns or "
                "out-of-memory errors."
            )

        use_thin_sheet = any(info.bc_kind == "thin_layer" for info in coupled_infos)
        # Pre-assemble system matrices (reused across incidence angles).

        # --- TM Robin BIE pre-assembly (shared with _solve_robin_bie) ---
        robin_sys = None
        robin_alpha_elements = None
        robin_pec_node = None
        if use_tm_robin_bie:
            robin_sys, robin_alpha_elements, robin_pec_node = _assemble_robin_bie_system(
                mesh, coupled_infos, pol, k0)

        # --- TE Robin MFIE pre-assembly ---
        mfie_sys = None
        mfie_alpha_elements = None
        if use_te_robin_mfie:
            mfie_alpha_elements, _ = _robin_alpha_elements(
                mesh, coupled_infos, pol
            )
            has_mfie_ibc = bool(np.any(np.abs(mfie_alpha_elements) > EPS))
            s_alpha, Kp = _assemble_linear_operator_matrices(
                mesh, k0, obs_normal_deriv=True,
                compute_single_layer=has_mfie_ibc,
                single_layer_observation_coefficients=(
                    mfie_alpha_elements if has_mfie_ibc else None
                ),
            )
            M_mat = _assemble_linear_mass_matrix(mesh)
            mfie_sys = -0.5 * M_mat + Kp
            if has_mfie_ibc:
                mfie_sys = mfie_sys + s_alpha

        # Pre-assemble sheet system operators once (reused across inc angles).
        # Handles both pure-sheet (all TYPE 1) and mixed sheet+PEC geometries.
        # Mixed uses the unified SLP (TM) / DLP (TE) representation with
        # element coefficient Z/(jketa) on sheets and zero on PEC elements.
        sheet_a_sys = None
        sheet_endpoint_nodes = None
        if (use_sheet or use_mixed_sheet) and not use_thin_sheet:
            z_elements_sheet = np.asarray([
                complex(info.robin_impedance)
                if int(info.seg_type) == 1 else 0.0 + 0.0j
                for info in coupled_infos
            ], dtype=np.complex128)
            if pol == "TM":
                S_sheet, _ = _assemble_linear_operator_matrices(
                    mesh, k0, obs_normal_deriv=False,
                    compute_double_layer=False)
                weighted_mass = _assemble_linear_weighted_mass_matrix(
                    mesh, z_elements_sheet / (1j * float(k0) * ETA0)
                )
                sheet_a_sys = S_sheet - weighted_mass
            else:
                N_sheet = _assemble_linear_hypersingular_matrix(mesh, k0)
                # Sign matches _solve_te_sheet: N - (jk/eta)Z_s.M.
                weighted_mass = _assemble_linear_weighted_mass_matrix(
                    mesh, (1j * float(k0) / ETA0) * z_elements_sheet
                )
                sheet_a_sys = N_sheet - weighted_mass
                # Meixner pin: mu=0 at OPEN-STRIP endpoints only.  See
                # _geometric_sheet_endpoint_nodes for why we count by geometric
                # key (handles signature-split nodes in stair-stepped tapers).
                sheet_endpoint_nodes = _geometric_sheet_endpoint_nodes(mesh, coupled_infos)
                if sheet_endpoint_nodes.size > 0:
                    sheet_a_sys[sheet_endpoint_nodes, :] = 0.0
                    sheet_a_sys[sheet_endpoint_nodes, sheet_endpoint_nodes] = 1.0

        # Assemble every incidence-angle RHS and solve it as one dense
        # multi-RHS system.  np.linalg.solve then performs one factorization
        # per frequency/formulation instead of refactoring the same matrix for
        # every incidence angle.  The monostatic path already follows this
        # pattern; keeping bistatic consistent is both faster and numerically
        # equivalent.
        inc_all_arr = np.asarray(inc_angles, dtype=float)
        batch_solution = None
        batch_residuals = None
        batch_exterior_density = None

        if use_thin_sheet:
            eps, mu, thickness = layer_for_mesh(mesh, materials, freq_ghz)
            condition_diagnostics = {} if compute_condition_number else None
            _, batch_amp, residual, evidence = solve_thin_layer_fields(
                mesh, k0, inc_all_arr, pol, eps, mu, thickness,
                observation_angles_deg=obs_arr,
                condition_diagnostics=condition_diagnostics)
            thin_layer_evidence.append(dict(evidence, frequency_ghz=float(freq_ghz), polarization=pol))
            batch_residuals = np.full(inc_all_arr.size, residual)
            if condition_diagnostics is not None:
                _consume_condition_estimate(cond_values, condition_diagnostics, "bistatic thin layer")
                frequency_condition_recorded = True
        elif use_sheet or use_mixed_sheet:
            rhs_all = np.zeros(
                (nnodes, inc_all_arr.size), dtype=np.complex128
            )
            for elem in mesh.elements:
                ids = np.asarray(elem.node_ids, dtype=int)
                if pol == "TM":
                    rhs_all[ids, :] -= _linear_element_incident_load_many(
                        elem, k_air=float(k0), elevations_deg=inc_all_arr
                    )
                else:
                    rhs_all[ids, :] += _linear_element_incident_dn_load_many(
                        elem, k_air=float(k0), elevations_deg=inc_all_arr
                    )
            if sheet_endpoint_nodes is not None and sheet_endpoint_nodes.size > 0:
                rhs_all[sheet_endpoint_nodes, :] = 0.0
            _ensure_finite_linear_system(
                sheet_a_sys, rhs_all, label="bistatic sheet system"
            )
            condition_diagnostics = {} if compute_condition_number else None
            batch_solution = _solve_dense_system(
                sheet_a_sys, rhs_all, condition_diagnostics,
                "bistatic sheet system",
            )
            if condition_diagnostics is not None:
                cond_values.append(float(condition_diagnostics["condition_est"]))
                frequency_condition_recorded = True
            batch_residuals = _residual_norm_many(
                sheet_a_sys, batch_solution, rhs_all
            )

        elif use_multi_region:
            condition_diagnostics = {} if compute_condition_number else None
            _, _, multi_residual, batch_exterior_density = (
                _solve_multi_region_indirect(
                    mesh,
                    coupled_infos,
                    pol,
                    k0,
                    inc_all_arr,
                    condition_diagnostics=condition_diagnostics,
                )
            )
            batch_residuals = np.full(
                inc_all_arr.size, float(multi_residual), dtype=float
            )
            if condition_diagnostics is not None:
                _consume_condition_estimate(
                    cond_values,
                    condition_diagnostics,
                    "bistatic multi-region formulation",
                )
                frequency_condition_recorded = True

        elif use_te_robin_mfie:
            rhs_all = np.zeros(
                (nnodes, inc_all_arr.size), dtype=np.complex128
            )
            for eidx, elem in enumerate(mesh.elements):
                ids = np.asarray(elem.node_ids, dtype=int)
                rhs_all[ids, :] -= _linear_element_incident_dn_load_many(
                    elem, k_air=k0, elevations_deg=inc_all_arr,
                )
                if (
                    mfie_alpha_elements is not None
                    and abs(complex(mfie_alpha_elements[eidx])) > EPS
                ):
                    rhs_all[ids, :] -= (
                        complex(mfie_alpha_elements[eidx])
                        * _linear_element_incident_load_many(
                            elem, k_air=k0, elevations_deg=inc_all_arr,
                        )
                    )
            _ensure_finite_linear_system(
                mfie_sys, rhs_all, label="bistatic TE Robin MFIE system"
            )
            condition_diagnostics = {} if compute_condition_number else None
            batch_solution = _solve_dense_system(
                mfie_sys, rhs_all, condition_diagnostics,
                "bistatic TE Robin MFIE system",
            )
            if condition_diagnostics is not None:
                cond_values.append(float(condition_diagnostics["condition_est"]))
                frequency_condition_recorded = True
            batch_residuals = _residual_norm_many(
                mfie_sys, batch_solution, rhs_all
            )

        elif use_tm_robin_bie:
            rhs_all = _robin_bie_rhs_many(
                mesh,
                robin_alpha_elements,
                robin_pec_node,
                pol,
                k0,
                inc_all_arr,
            )
            _ensure_finite_linear_system(
                robin_sys, rhs_all, label="bistatic TM Robin BIE system"
            )
            condition_diagnostics = {} if compute_condition_number else None
            batch_solution = _solve_dense_system(
                robin_sys, rhs_all, condition_diagnostics,
                "bistatic Robin system",
            )
            if condition_diagnostics is not None:
                cond_values.append(float(condition_diagnostics["condition_est"]))
                frequency_condition_recorded = True
            batch_residuals = _residual_norm_many(
                robin_sys, batch_solution, rhs_all
            )

        elif use_diel_indirect:
            info0 = coupled_infos[0]
            k1_vals = {
                complex(info.k_plus)
                for info in coupled_infos
                if info.plus_region > 0
            }
            k1 = k1_vals.pop() if k1_vals else k0
            factor = (
                complex(info0.mu_minus / info0.mu_plus)
                if pol == "TM"
                else complex(info0.eps_minus / info0.eps_plus)
            )

            _, K0 = _assemble_linear_operator_matrices(
                mesh, k0, obs_normal_deriv=False,
                compute_single_layer=False,
            )
            S1, Kp1 = _assemble_linear_operator_matrices(
                mesh, k1, obs_normal_deriv=True,
                compute_single_layer=True,
            )
            D0 = _assemble_linear_hypersingular_matrix(mesh, k0)
            M = _assemble_linear_mass_matrix(mesh)
            diel_sys = np.zeros(
                (2 * nnodes, 2 * nnodes), dtype=np.complex128
            )
            diel_sys[:nnodes, :nnodes] = 0.5 * M + K0
            diel_sys[:nnodes, nnodes:] = -S1
            diel_sys[nnodes:, :nnodes] = D0
            diel_sys[nnodes:, nnodes:] = factor * (0.5 * M + Kp1)

            rhs_all = np.zeros(
                (2 * nnodes, inc_all_arr.size), dtype=np.complex128
            )
            for elem in mesh.elements:
                ids = np.asarray(elem.node_ids, dtype=int)
                rhs_all[ids, :] -= _linear_element_incident_load_many(
                    elem, k0, inc_all_arr
                )
                rhs_all[nnodes + ids, :] += (
                    _linear_element_incident_dn_load_many(
                        elem, k0, inc_all_arr
                    )
                )
            _ensure_finite_linear_system(
                diel_sys, rhs_all, label="bistatic dielectric indirect system"
            )
            condition_diagnostics = {} if compute_condition_number else None
            batch_solution = _solve_dense_system(
                diel_sys, rhs_all, condition_diagnostics,
                "bistatic dielectric indirect system",
            )
            if condition_diagnostics is not None:
                cond_values.append(float(condition_diagnostics["condition_est"]))
                frequency_condition_recorded = True
            batch_residuals = _residual_norm_many(
                diel_sys, batch_solution, rhs_all
            )

        else:
            raise ValueError(
                "Geometry did not match any supported bistatic formulation "
                "(sheet, all-Robin PEC/IBC, dielectric body, or multi-region). "
                "No deprecated coupled-trace fallback was performed."
            )

        # Project every solved incidence density at every observation angle
        # in one tiled pass. The geometry/quadrature phase matrix depends only
        # on observation angle, so rebuilding it once per incidence repeated
        # identical work for a rectangular bistatic sweep.
        if use_thin_sheet:
            batch_far_density = None  # Both single- and double-layer projections are already combined.
        elif use_sheet or use_mixed_sheet:
            batch_far_density = batch_solution
            batch_far_potential = "SLP" if pol == "TM" else "DLP"
        elif use_multi_region:
            batch_far_density = batch_exterior_density
            batch_far_potential = "SLP"
        elif use_te_robin_mfie or use_tm_robin_bie:
            batch_far_density = batch_solution
            batch_far_potential = "SLP"
        elif use_diel_indirect:
            batch_far_density = batch_solution[:nnodes, :]
            batch_far_potential = "DLP"
        else:
            raise AssertionError("Unreachable bistatic far-field dispatch.")

        if not use_thin_sheet:
            batch_amp = _farfield_linear_density_many(
                mesh, batch_far_density, k0, obs_arr, batch_far_potential,
                projection="grid")
        batch_rcs_lin = _rcs_sigma_from_amp(batch_amp, k0)
        batch_rcs_db = _rcs_db_from_sigma(batch_rcs_lin)

        for inc_index, inc_deg in enumerate(inc_angles):
            check_abort()
            residual_local = float(batch_residuals[inc_index])
            amp = batch_amp[inc_index, :]
            rcs_lin = batch_rcs_lin[inc_index, :]
            rcs_db = batch_rcs_db[inc_index, :]
            residual_values.append(float(residual_local))
            for idx, obs_deg in enumerate(obs_angles):
                amp_val = complex(amp[idx])
                samples.append({
                    "frequency_ghz": float(freq_ghz),
                    "theta_inc_deg": float(inc_deg),
                    "theta_scat_deg": float(obs_deg),
                    "rcs_linear": float(rcs_lin[idx]),
                    "rcs_db": float(rcs_db[idx]),
                    "rcs_amp_real": float(np.real(amp_val)),
                    "rcs_amp_imag": float(np.imag(amp_val)),
                    "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                    "linear_residual": float(residual_local),
                })

            done_steps += 1
            emit_progress(f"Bistatic {freq_ghz:g} GHz inc={inc_deg:g} deg")

        if compute_condition_number and not frequency_condition_recorded:
            raise RuntimeError(
                "Bistatic 2-D solve did not produce the requested "
                "condition-number diagnostic; no field is returned."
            )

    residual_norm_max, residual_norm_mean, residual_nonfinite_count = (
        _summarize_residuals(residual_values)
    )
    metadata: 'Dict[str, Any]' = {
        "formulation": "bistatic 2D BIE/MoM",
        "cfie_alpha": float(cfie_alpha),
        "solver_method": "dense_lu",
        "solver_method_requested": str(solver_method),
        "mesh_wavelength_m": float(mesh_wavelength_values[0]) if len(set(round(v, 15) for v in mesh_wavelength_values)) == 1 and mesh_wavelength_values else None,
        "mesh_wavelength_min_m": float(np.min(mesh_wavelength_values)) if mesh_wavelength_values else 0.0,
        "mesh_wavelength_max_m": float(np.max(mesh_wavelength_values)) if mesh_wavelength_values else 0.0,
        "mesh_max_refractive_index": float(np.max(mesh_max_index_values)) if mesh_max_index_values else 1.0,
        "mesh_material_flags": sorted(mesh_material_flags_used),
        "residual_norm_max": residual_norm_max,
        "residual_norm_mean": residual_norm_mean,
        "residual_nonfinite_count": residual_nonfinite_count,
        "constraint_residual_norm_max": 0.0,
        "constraint_residual_norm_mean": 0.0,
        "condition_est_max": float(np.max(cond_values)) if cond_values else float("nan"),
        "condition_est_mean": float(np.mean(cond_values)) if cond_values else float("nan"),
        "condition_est_computed": bool(compute_condition_number),
        "condition_estimator": (
            "equilibrated_1norm_lu_onenormest"
            if compute_condition_number else "not_requested"
        ),
        "thin_layer": thin_layer_evidence,
        "warnings": list(materials.warnings),
        "warning_count": int(len(materials.warnings)),
        "preflight": dict(preflight_report),
        **_dense_backend_summary(),
    }
    metadata["amplitude_version"] = RCS_AMPLITUDE_VERSION
    quality_gate = evaluate_quality_gate(metadata, thresholds=quality_thresholds)
    metadata["quality_gate"] = quality_gate
    if strict_quality_gate and not bool(quality_gate.get("passed", False)):
        reason = str(quality_gate.get("reason", "quality gate failed"))
        raise ValueError(f"Quality gate failed: {reason}")

    return {
        "solver": "2d_bie_mom_rcs",
        "scattering_mode": "bistatic",
        "amplitude_convention": RCS_AMPLITUDE_CONVENTION,
        "amplitude_version": RCS_AMPLITUDE_VERSION,
        "polarization": _canonical_user_polarization_label(polarization),
        "polarization_export": _canonical_user_polarization_label(polarization),
        "samples": samples,
        "metadata": metadata,
    }


@profiled_solve
def solve_bistatic_rcs_2d(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    incidence_angles_deg: 'List[float]',
    observation_angles_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    strict_quality_gate: 'bool' = True,
    compute_condition_number: 'bool' = False,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    abort_event: 'Optional[threading.Event]' = None,
) -> 'Dict[str, Any]':
    """Solve both physical co-polarized bistatic 2-D channels."""

    channel_results = {}
    for index, (export_pol, internal_pol) in enumerate(
        _CO_POLARIZED_2D_CHANNELS
    ):
        channel_results[export_pol] = solve_bistatic_rcs_2d_single_polarization(
            geometry_snapshot=geometry_snapshot,
            frequencies_ghz=frequencies_ghz,
            incidence_angles_deg=incidence_angles_deg,
            observation_angles_deg=observation_angles_deg,
            polarization=internal_pol,
            geometry_units=geometry_units,
            material_base_dir=material_base_dir,
            progress_callback=_co_polarized_progress_callback(
                progress_callback, index, export_pol
            ),
            quality_thresholds=quality_thresholds,
            strict_quality_gate=strict_quality_gate,
            compute_condition_number=compute_condition_number,
            max_panels=max_panels,
            mesh_reference_ghz=mesh_reference_ghz,
            cfie_alpha=0.0,
            abort_event=abort_event,
            solver_method="direct",
        )
    merged = _merge_co_polarized_2d_results(channel_results)
    # The bistatic implementation still constructs channel-local meshes.  Do
    # not claim the monostatic shared-discretization optimization in metadata.
    merged["metadata"]["co_solve_shared_discretization"] = False
    return merged


def _run_certified_2d_pair(
    low_level_solver: 'Callable[..., Dict[str, Any]]',
    geometry_snapshot: 'Dict[str, Any]',
    solver_kwargs: 'Dict[str, Any]',
    mesh_convergence_policy: 'Optional[Dict[str, Any]]',
    progress_callback: 'Optional[Callable[[int, int, str], None]]',
    shared_discretization_caches: 'Optional[Tuple[Dict[str, Any], Dict[str, Any]]]' = None,
) -> 'Dict[str, Any]':
    """Run base/fine 2-D solves and publish only a certified fine result."""

    from solver_quality import (
        evaluate_mesh_convergence,
        scale_snapshot_panel_density,
        validate_mesh_convergence_policy,
    )

    policy = validate_mesh_convergence_policy(mesh_convergence_policy)

    def _phase_callback(
        phase: 'str',
    ) -> 'Optional[Callable[[int, int, str], None]]':
        if progress_callback is None:
            return None

        def _mapped(done: 'int', total: 'int', message: 'str') -> 'None':
            total_i = max(1, int(total))
            done_i = max(0, min(int(done), total_i))
            if phase == "base":
                mapped_done = done_i
            else:
                mapped_done = total_i + done_i
            try:
                progress_callback(
                    mapped_done,
                    2 * total_i,
                    f"{phase.capitalize()} mesh: {message}",
                )
            except Exception:
                pass

        return _mapped

    common = dict(solver_kwargs)
    common["solver_method"] = _normalize_public_2d_solver_method(
        common.get("solver_method", "auto")
    )
    # Certification is intentionally non-optional here.  Both discrete
    # systems must pass their algebraic gate and supply condition telemetry
    # before their complex fields are compared.
    common["strict_quality_gate"] = True
    common["compute_condition_number"] = True

    base_kwargs = dict(common)
    base_kwargs["geometry_snapshot"] = geometry_snapshot
    base_kwargs["progress_callback"] = _phase_callback("base")
    if shared_discretization_caches is not None:
        base_kwargs["_shared_discretization_cache"] = (
            shared_discretization_caches[0]
        )
    base_result = low_level_solver(**base_kwargs)

    fine_snapshot = scale_snapshot_panel_density(
        geometry_snapshot, policy["fine_factor"]
    )
    # Keep the shared property scaling for BoR and provenance compatibility,
    # while giving the 2-D panel builder the original per-segment controls so
    # it can refine the realized base count of every primitive exactly.
    base_segment_n = []
    for segment in list(geometry_snapshot.get("segments", []) or []):
        props = list(segment.get("properties", []) or [])
        base_segment_n.append(props[1] if len(props) > 1 else 0)
    fine_snapshot["_2d_certification_refinement_factor"] = float(
        policy["fine_factor"]
    )
    fine_snapshot["_2d_certification_base_segment_n"] = base_segment_n
    fine_kwargs = dict(common)
    fine_kwargs["geometry_snapshot"] = fine_snapshot
    fine_kwargs["progress_callback"] = _phase_callback("fine")
    if shared_discretization_caches is not None:
        fine_kwargs["_shared_discretization_cache"] = (
            shared_discretization_caches[1]
        )
    fine_result = low_level_solver(**fine_kwargs)

    base_panel_count = int(
        base_result.get("metadata", {}).get("panel_count", 0) or 0
    )
    fine_panel_count = int(
        fine_result.get("metadata", {}).get("panel_count", 0) or 0
    )
    if base_panel_count > 0 and fine_panel_count <= base_panel_count:
        raise ValueError(
            "Certified 2-D mesh refinement failed: the fine solve used "
            f"{fine_panel_count} panels versus {base_panel_count} on the base "
            "mesh. A mesh-convergence certificate requires a genuinely "
            "refined discretization."
        )

    mesh_gate = evaluate_mesh_convergence(
        base_result=base_result,
        fine_result=fine_result,
        rms_limit_db=policy["rms_limit_db"],
        max_abs_limit_db=policy["max_abs_limit_db"],
        complex_rms_limit=policy["complex_rms_limit"],
        complex_max_limit=policy["complex_max_limit"],
        phase_rms_limit_deg=policy["phase_rms_limit_deg"],
        phase_max_limit_deg=policy["phase_max_limit_deg"],
        phase_floor_relative=policy["phase_floor_relative"],
    )
    mesh_gate["schema"] = "ghost.solver.mesh-convergence.v1"
    mesh_gate["fine_factor"] = policy["fine_factor"]
    mesh_gate["published_mesh"] = "fine"
    mesh_gate["geometry_model"] = "piecewise_linear_input"
    mesh_gate["geometry_approximation_certified"] = False
    mesh_gate["base_quality_gate"] = dict(
        base_result.get("metadata", {}).get("quality_gate", {}) or {}
    )
    mesh_gate["fine_quality_gate"] = dict(
        fine_result.get("metadata", {}).get("quality_gate", {}) or {}
    )
    mesh_gate["base_panel_count"] = base_panel_count
    mesh_gate["fine_panel_count"] = fine_panel_count
    mesh_gate["panel_refinement_ratio"] = (
        float(fine_panel_count) / float(base_panel_count)
        if base_panel_count > 0 else float("nan")
    )

    if not bool(mesh_gate.get("passed", False)):
        raise ValueError(
            "Certified 2-D mesh convergence failed: "
            + str(mesh_gate.get("reason", "unknown convergence failure"))
        )

    result = fine_result
    metadata = result.setdefault("metadata", {})
    metadata["mesh_convergence"] = mesh_gate
    metadata["mesh_convergence_certified"] = True
    metadata["certified_entry_point"] = True
    metadata["published_mesh"] = "fine"
    quality_gate = metadata.get("quality_gate")
    if isinstance(quality_gate, dict):
        quality_gate["mesh_convergence_certified"] = True
        quality_gate["certification_scope"] = (
            "discrete_linear_system_and_mesh_convergence"
        )
        if bool(quality_gate.get("passed", False)):
            quality_gate["reason"] = (
                "discrete linear-system quality thresholds and production "
                "mesh-convergence certification satisfied"
            )
    return result


@profiled_solve
@experimental_monostatic
def solve_monostatic_rcs_2d_certified_single_polarization(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    mesh_convergence_policy: 'Optional[Dict[str, Any]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    rcs_normalization_mode: 'str' = RCS_NORM_MODE_DEFAULT,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
    _shared_discretization_caches: 'Optional[Tuple[Dict[str, Any], Dict[str, Any]]]' = None,
) -> 'Dict[str, Any]':
    """Explicit single-polarization algebraic plus mesh certification."""

    return _run_certified_2d_pair(
        solve_monostatic_rcs_2d_single_polarization,
        geometry_snapshot,
        {
            "frequencies_ghz": frequencies_ghz,
            "elevations_deg": elevations_deg,
            "polarization": polarization,
            "geometry_units": geometry_units,
            "material_base_dir": material_base_dir,
            "quality_thresholds": quality_thresholds,
            "max_panels": max_panels,
            "mesh_reference_ghz": mesh_reference_ghz,
            "rcs_normalization_mode": rcs_normalization_mode,
            "cfie_alpha": cfie_alpha,
            "abort_event": abort_event,
            "solver_method": solver_method,
        },
        mesh_convergence_policy,
        progress_callback,
        _shared_discretization_caches,
    )


@profiled_solve
def solve_bistatic_rcs_2d_certified_single_polarization(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    incidence_angles_deg: 'List[float]',
    observation_angles_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    mesh_convergence_policy: 'Optional[Dict[str, Any]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
) -> 'Dict[str, Any]':
    """Explicit single-polarization bistatic mesh certification."""
    if str(solver_method).strip().lower() == EXPERIMENTAL_METHOD:
        raise ValueError("Experimental CPU supports 2D monostatic fields only.")

    return _run_certified_2d_pair(
        solve_bistatic_rcs_2d_single_polarization,
        geometry_snapshot,
        {
            "frequencies_ghz": frequencies_ghz,
            "incidence_angles_deg": incidence_angles_deg,
            "observation_angles_deg": observation_angles_deg,
            "polarization": polarization,
            "geometry_units": geometry_units,
            "material_base_dir": material_base_dir,
            "quality_thresholds": quality_thresholds,
            "max_panels": max_panels,
            "mesh_reference_ghz": mesh_reference_ghz,
            "cfie_alpha": cfie_alpha,
            "abort_event": abort_event,
            "solver_method": solver_method,
        },
        mesh_convergence_policy,
        progress_callback,
    )


@profiled_solve
@experimental_monostatic
def solve_monostatic_rcs_2d_certified(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    mesh_convergence_policy: 'Optional[Dict[str, Any]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    rcs_normalization_mode: 'str' = RCS_NORM_MODE_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "direct",
) -> 'Dict[str, Any]':
    """Canonical production monostatic entry; both channels must certify."""

    if len(frequencies_ghz) > 1:
        return _frequency_local_co_solve(solve_monostatic_rcs_2d_certified, locals())
    channel_results = {}
    shared_discretization_caches = ({}, {})
    for index, (export_pol, internal_pol) in enumerate(
        _CO_POLARIZED_2D_CHANNELS
    ):
        channel_results[export_pol] = (
            solve_monostatic_rcs_2d_certified_single_polarization(
                geometry_snapshot=geometry_snapshot,
                frequencies_ghz=frequencies_ghz,
                elevations_deg=elevations_deg,
                polarization=internal_pol,
                geometry_units=geometry_units,
                material_base_dir=material_base_dir,
                progress_callback=_co_polarized_progress_callback(
                    progress_callback, index, export_pol
                ),
                quality_thresholds=quality_thresholds,
                mesh_convergence_policy=mesh_convergence_policy,
                max_panels=max_panels,
                mesh_reference_ghz=mesh_reference_ghz,
                rcs_normalization_mode=rcs_normalization_mode,
                cfie_alpha=0.0,
                abort_event=abort_event,
                solver_method=solver_method,
                _shared_discretization_caches=shared_discretization_caches,
            )
        )
    result = _merge_co_polarized_2d_results(channel_results)
    result["metadata"]["co_solve_shared_discretization"] = True
    return result


def _mark_co_polarized_survey_result(
    result: 'Dict[str, Any]',
    warning: 'str',
) -> 'Dict[str, Any]':
    metadata = result.setdefault("metadata", {})
    metadata["mesh_convergence_certified"] = False
    metadata["certified_entry_point"] = False
    metadata["published_mesh"] = "base"
    metadata["survey_mode"] = True
    warnings = metadata.setdefault("warnings", [])
    if warning not in warnings:
        warnings.append(warning)
    metadata["warning_count"] = len(warnings)
    quality_gate = metadata.get("quality_gate")
    if isinstance(quality_gate, dict):
        quality_gate["mesh_convergence_certified"] = False
        quality_gate["certification_scope"] = "discrete_linear_system_only"
    return result


@profiled_solve
@experimental_monostatic
def solve_monostatic_rcs_2d_survey(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    rcs_normalization_mode: 'str' = RCS_NORM_MODE_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "direct",
) -> 'Dict[str, Any]':
    """Co-polarized single-mesh survey with explicit non-certification."""

    result = solve_monostatic_rcs_2d(
        geometry_snapshot=geometry_snapshot,
        frequencies_ghz=frequencies_ghz,
        elevations_deg=elevations_deg,
        geometry_units=geometry_units,
        material_base_dir=material_base_dir,
        progress_callback=progress_callback,
        quality_thresholds=quality_thresholds,
        strict_quality_gate=True,
        compute_condition_number=True,
        max_panels=max_panels,
        mesh_reference_ghz=mesh_reference_ghz,
        rcs_normalization_mode=rcs_normalization_mode,
        abort_event=abort_event,
        solver_method=solver_method,
    )
    return _mark_co_polarized_survey_result(
        result,
        "SURVEY MODE: VV and HH were solved on the base mesh only; no "
        "mesh-convergence certificate exists for either channel.",
    )


@profiled_solve
def solve_bistatic_rcs_2d_certified(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    incidence_angles_deg: 'List[float]',
    observation_angles_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    mesh_convergence_policy: 'Optional[Dict[str, Any]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    abort_event: 'Optional[threading.Event]' = None,
) -> 'Dict[str, Any]':
    """Canonical certified bistatic entry; both channels must certify."""

    channel_results = {}
    for index, (export_pol, internal_pol) in enumerate(
        _CO_POLARIZED_2D_CHANNELS
    ):
        channel_results[export_pol] = (
            solve_bistatic_rcs_2d_certified_single_polarization(
                geometry_snapshot=geometry_snapshot,
                frequencies_ghz=frequencies_ghz,
                incidence_angles_deg=incidence_angles_deg,
                observation_angles_deg=observation_angles_deg,
                polarization=internal_pol,
                geometry_units=geometry_units,
                material_base_dir=material_base_dir,
                progress_callback=_co_polarized_progress_callback(
                    progress_callback, index, export_pol
                ),
                quality_thresholds=quality_thresholds,
                mesh_convergence_policy=mesh_convergence_policy,
                max_panels=max_panels,
                mesh_reference_ghz=mesh_reference_ghz,
                cfie_alpha=0.0,
                abort_event=abort_event,
                solver_method="direct",
            )
        )
    result = _merge_co_polarized_2d_results(channel_results)
    result["metadata"]["co_solve_shared_discretization"] = False
    return result


@profiled_solve
def solve_bistatic_rcs_2d_survey(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    incidence_angles_deg: 'List[float]',
    observation_angles_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    abort_event: 'Optional[threading.Event]' = None,
) -> 'Dict[str, Any]':
    """Co-polarized bistatic base-mesh survey."""

    result = solve_bistatic_rcs_2d(
        geometry_snapshot=geometry_snapshot,
        frequencies_ghz=frequencies_ghz,
        incidence_angles_deg=incidence_angles_deg,
        observation_angles_deg=observation_angles_deg,
        geometry_units=geometry_units,
        material_base_dir=material_base_dir,
        progress_callback=progress_callback,
        quality_thresholds=quality_thresholds,
        strict_quality_gate=True,
        compute_condition_number=True,
        max_panels=max_panels,
        mesh_reference_ghz=mesh_reference_ghz,
        abort_event=abort_event,
    )
    return _mark_co_polarized_survey_result(
        result,
        "SURVEY MODE: bistatic VV and HH were solved on the base mesh only; "
        "no mesh-convergence certificate exists for either channel.",
    )


def compute_boundary_densities(
    geometry_snapshot: 'Dict[str, Any]',
    frequency_ghz: 'float',
    elevation_deg: 'float',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
) -> 'Dict[str, Any]':
    """
    Compute formulation-specific boundary-integral unknowns for visualization.

    These SLP/DLP layer densities are mathematical representation unknowns;
    they are not generally physical electric or magnetic surface-current
    densities. Returns element-center positions, layer density, panel normals,
    and the formulation used for a single-frequency, single-angle debug solve.
    """

    def check_abort() -> 'None':
        if abort_event is not None and abort_event.is_set():
            raise InterruptedError(
                "Boundary-density calculation canceled by user."
            )

    check_abort()
    cfie_alpha = _validate_disabled_2d_cfie_alpha(cfie_alpha)
    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)
    base_dir = _material_base_dir_for_snapshot(
        geometry_snapshot, material_base_dir
    )
    frequency_ghz = float(frequency_ghz)
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0.0:
        raise ValueError("frequency_ghz must be a positive finite value.")
    freq_hz = frequency_ghz * 1e9
    k0 = 2.0 * math.pi * freq_hz / C0

    preflight = validate_geometry_snapshot_for_solver(geometry_snapshot, base_dir=base_dir, meters_scale=unit_scale)
    check_abort()
    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    check_abort()
    lambda_min, mesh_max_index, mesh_material_flags = _mesh_wavelength_for_snapshot(
        geometry_snapshot, materials, frequency_ghz
    )
    panels = _build_panels(geometry_snapshot, unit_scale, lambda_min, max_panels=max_panels)
    check_abort()
    preview_infos = _build_coupled_panel_info(panels, materials, frequency_ghz, pol, k0)
    mesh, _ = _build_linear_mesh_interface_aware(panels, preview_infos)
    check_abort()
    coupled_infos = _build_linear_coupled_infos(mesh, materials, frequency_ghz, pol, k0)
    _assert_no_type1_sheet(coupled_infos)
    _assert_air_exterior(coupled_infos)
    _assert_supported_te_type2_contours(mesh, coupled_infos, pol)
    nnodes = len(mesh.nodes)
    elev_arr = np.asarray([elevation_deg], dtype=float)

    centers = np.asarray([e.center for e in mesh.elements], dtype=float)
    normals = np.asarray([e.normal for e in mesh.elements], dtype=float)
    lengths = np.asarray([e.length for e in mesh.elements], dtype=float)

    use_multi = _is_multi_region(coupled_infos)
    use_diel = _is_single_dielectric_body(coupled_infos) and not use_multi
    # All-Robin covers PEC, IBC (constant or tapered), and mixed PEC+IBC.
    # It reuses the shared Robin-BIE assembly (element-weighted alpha,
    # adjacent-medium wavenumber, per-row TM-PEC EFIE override) so the
    # visualized layer densities come from exactly the formulation the RCS solvers
    # use.  The previous branches applied a single alpha from element 0 to
    # the whole body (ignoring tapers, using raw k0) and dropped mixed
    # PEC+IBC TM bodies into a pure-PEC EFIE that ignored the impedance.
    use_robin = _is_all_robin(coupled_infos)

    resources = _dense_formulation_resources(mesh, coupled_infos, pol)
    check_abort()
    est_gb = _estimate_memory_gb(
        resources["nodes"],
        use_cfie=False,
        n_regions=max(1, resources["n_regions"]),
        system_dofs=resources["system_dofs"],
        operator_matrices=resources["operator_matrices"],
        n_rhs=1,
    )
    memory_limit_gb = _solve_memory_limit_gb()
    if est_gb > memory_limit_gb:
        raise MemoryError(
            _memory_gate_message(
                est_gb,
                memory_limit_gb,
                "Boundary-density diagnostics",
                (
                    f"Planned {resources['formulation']} system: "
                    f"{resources['system_dofs']} DOFs."
                ),
                "Reduce panel count or frequency before opening the density view.",
            )
        )

    if use_multi:
        # Multi-region: extract exterior SLP density.
        check_abort()
        _, _, _, ext_density = _solve_multi_region_indirect(
            mesh, coupled_infos, pol, k0, elev_arr)
        check_abort()
        sigma_nodes = ext_density[:, 0]
        density = np.asarray([
            0.5 * (sigma_nodes[e.node_ids[0]] + sigma_nodes[e.node_ids[1]])
            for e in mesh.elements
        ], dtype=np.complex128)
        formulation = "Multi-region indirect (exterior SLP density)"

    elif use_diel:
        # Assemble once and retain the DLP density directly.  The previous
        # implementation first called the complete RCS solver, discarded its
        # field, then rebuilt every dense operator and solved the identical
        # system a second time solely to expose this density.
        info0 = coupled_infos[0]
        k1_vals = {complex(i.k_plus) for i in coupled_infos if i.plus_region > 0}
        k1 = k1_vals.pop() if k1_vals else k0
        factor = complex(info0.mu_minus / info0.mu_plus) if pol == 'TM' else complex(info0.eps_minus / info0.eps_plus)
        _, K0 = _assemble_linear_operator_matrices(
            mesh, k0, False, compute_single_layer=False
        )
        check_abort()
        S1, Kp1 = _assemble_linear_operator_matrices(
            mesh, k1, True, compute_single_layer=True
        )
        check_abort()
        D0 = _assemble_linear_hypersingular_matrix(mesh, k0)
        check_abort()
        M = _assemble_linear_mass_matrix(mesh)
        a = np.zeros((2*nnodes, 2*nnodes), dtype=np.complex128)
        a[:nnodes,:nnodes] = 0.5*M+K0; a[:nnodes,nnodes:] = -S1
        a[nnodes:,:nnodes] = D0; a[nnodes:,nnodes:] = factor*(0.5*M+Kp1)
        rhs = np.zeros(2*nnodes, dtype=np.complex128)
        for elem in mesh.elements:
            ids = np.asarray(elem.node_ids, dtype=int)
            rhs[ids] -= _linear_element_incident_load_many(elem, k0, elev_arr)[:,0]
            rhs[nnodes+ids] += _linear_element_incident_dn_load_many(elem, k0, elev_arr)[:,0]
        check_abort()
        sol = np.linalg.solve(a, rhs)
        check_abort()
        mu_nodes = sol[:nnodes]
        density = np.asarray([
            0.5*(mu_nodes[e.node_ids[0]]+mu_nodes[e.node_ids[1]])
            for e in mesh.elements
        ], dtype=np.complex128)
        formulation = "Indirect dielectric (DLP density)"

    elif use_robin:
        # Shared Robin-BIE assembly: element-weighted alpha (tapered IBC),
        # adjacent-medium wavenumber, per-row TM-PEC EFIE override -- the same
        # system _solve_robin_bie / the bistatic dispatch solve.  For pure
        # PEC this reduces to the EFIE (TM) / MFIE (TE) exactly.
        a_sys, alpha_elements, pec_node = _assemble_robin_bie_system(
            mesh, coupled_infos, pol, k0
        )
        check_abort()
        rhs = _robin_bie_rhs_many(
            mesh, alpha_elements, pec_node, pol, k0, elev_arr
        )
        check_abort()
        sigma_nodes = np.linalg.solve(a_sys, rhs)[:, 0]
        check_abort()
        density = np.asarray([
            0.5*(sigma_nodes[e.node_ids[0]]+sigma_nodes[e.node_ids[1]])
            for e in mesh.elements
        ], dtype=np.complex128)
        formulation = (
            "Robin BIE (SLP density; element-weighted alpha, TM-PEC EFIE rows)"
            if pol == "TM" else "Robin BIE / MFIE (SLP density; element-weighted alpha)"
        )

    else:
        # Safety net -- the dispatch above is exhaustive (all-Robin,
        # single-dielectric, multi-region), so this should be unreachable.
        raise ValueError(
            "compute_boundary_densities: geometry did not match any supported "
            "formulation (all-Robin, single dielectric, or multi-region)."
        )

    return {
        "quantity": "boundary_integral_layer_density",
        "is_physical_surface_current": False,
        "interpretation": (
            "Formulation-specific SLP/DLP representation density; do not "
            "interpret as electric or magnetic surface current without a "
            "formulation- and polarization-specific trace conversion."
        ),
        "formulation": formulation,
        "frequency_ghz": float(frequency_ghz),
        "elevation_deg": float(elevation_deg),
        "polarization": pol,
        "coordinate_units": "meters",
        "length_units": "meters",
        "mesh_wavelength_m": float(lambda_min),
        "mesh_max_refractive_index": float(mesh_max_index),
        "mesh_material_flags": list(mesh_material_flags),
        "element_count": int(len(mesh.elements)),
        "node_count": int(nnodes),
        "centers_x": centers[:, 0].tolist(),
        "centers_y": centers[:, 1].tolist(),
        "normals_x": normals[:, 0].tolist(),
        "normals_y": normals[:, 1].tolist(),
        "lengths": lengths.tolist(),
        "density_real": np.real(density).tolist(),
        "density_imag": np.imag(density).tolist(),
        "density_abs": np.abs(density).tolist(),
        "density_phase_deg": np.degrees(np.angle(density)).tolist(),
        "amplitude_version": RCS_AMPLITUDE_VERSION,
    }
