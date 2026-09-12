"""2-D boundary-integral RCS solves and mesh certification."""
from ghost_backend.twod.preparation import prepared_execution, prepare_geometry
from ghost_backend.twod.meshing import segment_wavelengths
from ghost_backend.execution.options import configured_execution, environment_value, current_options

import cmath
import csv
import ctypes
import ctypes.util
import math
import os
import subprocess
import sys
import threading
from ghost_backend.execution.runtime import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from ghost_backend.linalg.workspace import matrix_inf_norm
from ghost_backend.twod.assembly.mass import add_mass
from ghost_backend.twod.assembly.session import shared_assembly
from ghost_backend.execution.metrics import active_metrics, profiled_solve, timed_stage
from ghost_backend.twod.formulations.thin_layer import (
    ThinLayerDefinition,
    layer_for_mesh,
    solve_thin_layer_fields,
)
from ghost_backend.linalg.refined_lu import RefinedLU, requested_precision
from ghost_backend.execution.cpu import (
    experimental_monostatic,
    current_state,
    requested_cpu,
    select_formulation,
    EXPERIMENTAL_METHOD,
)
from ghost_backend.geometry.io import material_filename_from_row


from ghost_backend.twod.special import (
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
from ghost_backend.twod.constants import (
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
from ghost_backend.twod.geometry import (
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
from ghost_backend.twod.operators import (
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
    def unique_records(key):
        seen, records = set(), []
        for event in events:
            record = event.get(key)
            if record is not None and id(record) not in seen:
                seen.add(id(record))
                records.append(dict(record))
        return records
    from ghost_backend.linalg.hierarchical import factor_mode
    from ghost_backend.linalg.sweep import mode as compression_mode
    return {
        "cpu_factorization_requested": factor_mode(),
        "cpu_rhs_compression_requested": compression_mode(),
        "hierarchical_factors": unique_records('hierarchical'),
        "compressed_factors": unique_records('compressed'),
        "sweep_compression": unique_records('sweep_compression'),
        "dense_condition_methods": sorted({row['condition_method'] for row in events if row.get('condition_method')}),
        "linear_backend": (
            used[0] if len(used) == 1 else ("mixed" if used else "cpu")
        ),
        "dense_gpu_solve_count": sum(
            1 for row in events if row.get("used") == "gpu_cupy"
        ),
        "dense_cpu_solve_count": sum(
            1 for row in events if str(row.get("used", "")).startswith("cpu")
        ),
        "dense_factorization_count": sum(int(row.get('factorizations', 1)) for row in events),
        "dense_rhs_batch_count": len(events),
        "dense_rhs_column_count": sum(int(row.get('rhs_columns', 0)) for row in events),
        "dense_max_rhs_columns": max([int(row.get('rhs_columns', 0)) for row in events] or [0]),
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
    backend = environment_value(DENSE_GPU_BACKEND_ENV, "cpu").strip().lower()
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
    """Estimate the row/column-equilibrated matrix 1-norm condition number using LU solves."""

    if _SCIPY_LINALG is None or _SCIPY_SPARSE_LINALG is None:
        raise RuntimeError("equilibrated condition estimation requires SciPy")
    a_eval = np.asarray(a_mat, dtype=np.complex128)
    row_scale, col_scale, norm_a = _equilibrated_scaling_and_norm_1(a_eval)
    n = int(a_eval.shape[0])

    def _inverse_matvec(vector):
        rhs = row_scale * np.asarray(
            vector, dtype=np.complex128
        ).reshape(-1)
        solved = solve_override(rhs) if solve_override is not None else _SCIPY_LINALG.lu_solve((lu, piv), rhs, check_finite=False)
        return col_scale * solved

    def _inverse_rmatvec(vector):
        rhs = col_scale * np.asarray(
            vector, dtype=np.complex128
        ).reshape(-1)
        solved = solve_override(rhs, trans=2) if solve_override is not None else _SCIPY_LINALG.lu_solve((lu, piv), rhs, trans=2, check_finite=False)
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
    residual_diagnostics=None,
) -> 'np.ndarray':
    """Factor once, solve all RHS columns, and optionally estimate condition."""

    a_eval = np.asarray(a_mat, dtype=np.complex128)
    rhs_eval = np.asarray(rhs, dtype=np.complex128)
    requested, threshold = _requested_dense_backend()
    from ghost_backend.linalg.hierarchical import factor_mode
    factorization = factor_mode()
    if factorization != 'dense' and requested == 'gpu':
        raise ValueError('Hierarchical factorization requires the CPU dense backend.')
    if requested_precision() == 'mixed' and requested == 'gpu':
        raise ValueError('Mixed LU currently requires the CPU backend; choose double precision for GPU solves.')
    if (factorization != 'dense' or requested == 'cpu' or condition_diagnostics is not None or requested_precision() == 'mixed'
            or requested == 'auto' and len(a_eval) < threshold):
        from ghost_backend.linalg.dense import DenseFactor
        factor = DenseFactor(a_eval, condition_diagnostics, label)
        solution = factor.solve(rhs_eval)
        if residual_diagnostics is not None:
            residual_diagnostics['relative_residual'] = factor.relative_residual
        return solution
    _ensure_finite_linear_system(a_eval, rhs_eval, label=label)
    matrix_inf = matrix_inf_norm(a_eval)
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

    if solution is None:

        from ghost_backend.linalg.dense import DenseFactor
        factor = DenseFactor(a_eval, condition_diagnostics, label)
        factor.fallback_reason = fallback_reason
        solution = factor.solve(rhs_eval)
        if residual_diagnostics is not None:
            residual_diagnostics['relative_residual'] = factor.relative_residual
        return solution

    def backward_metrics(candidate):
        x_columns = np.asarray(candidate, dtype=np.complex128)
        b_columns = rhs_eval
        if x_columns.ndim == 1:
            x_columns = x_columns[:, None]
            b_columns = b_columns[:, None]
        residual = a_eval @ x_columns - b_columns
        residual_inf = np.max(np.abs(residual), axis=0)
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
        if lu is None and _SCIPY_LINALG is not None:


            lu, piv = timed_stage('factorization')(_SCIPY_LINALG.lu_factor)(a_eval, check_finite=False)
        correction = (np.linalg.solve(a_eval, correction_rhs) if lu is None else
                      _SCIPY_LINALG.lu_solve((lu, piv), correction_rhs, check_finite=False))
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
        factorizations=1 + int(lu is not None),
        rhs_columns=1 if rhs_eval.ndim == 1 else rhs_eval.shape[1],
        n=system_n,
        label=str(label),
        fallback_reason=fallback_reason,
        gpu_device=gpu_device,
        refinement_steps=int(refinement_steps),
    )
    if residual_diagnostics is not None:
        b_columns = rhs_eval[:, None] if rhs_eval.ndim == 1 else rhs_eval
        norm = np.linalg.norm(b_columns, axis=0)
        residual_diagnostics['relative_residual'] = np.linalg.norm(residual_matrix, axis=0) / np.where(norm <= EPS, 1., norm)
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


            return 0
        return max(0, limit - usage)
    return None


def _detect_available_gb() -> 'float':
    """Return the tightest available-memory bound in GiB.

    Use psutil or native Windows/POSIX probes, then apply scheduler and cgroup limits.
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


_MEMORY_LIMIT_FRACTION = 0.9


def _solve_memory_limit_gb() -> 'float':
    override = environment_value("GHOST_MAX_SOLVE_GB", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            value = 0.0
        if math.isfinite(value) and value > 0.0:
            available = _detect_available_gb() if current_options() is not None else 0.0
            return min(value, _MEMORY_LIMIT_FRACTION * available) if available > 0 else value
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
        f"{detail_text} {remedies.strip()} "
        + ("Review the saved RAM budget and current free memory before retrying."
           if current_options() is not None else
           "If a larger allocation is confirmed, set GHOST_MAX_SOLVE_GB to that explicit per-process limit and retry.")
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
    dense_resources=None,
) -> 'float':
    """
    Estimate peak memory for the dense BIE/MoM solve in GB.

    Accounts for: system matrix, region operators, RHS, solution, factorization.
    """

    from ghost_backend.execution.cpu import configured_batch_size, CACHE_BYTES, TABLE_BYTES
    from ghost_backend.twod.operators import get_assembly_threads
    n = int(nnodes)
    d = 2*n if system_dofs is None else max(1, int(system_dofs))
    resources = dense_resources or {}
    kind = resources.get('formulation', formulation)
    count = max(1, int(n_rhs))
    if resources.get('analytic_zero'):
        return (64*1024**2 + 1024*n + count*4096) / 1024**3
    requested, threshold = _requested_dense_backend()
    from ghost_backend.linalg.hierarchical import factor_mode
    factorization = factor_mode()

    gpu = factorization == 'dense' and (requested == 'gpu' or requested == 'auto' and d >= threshold)
    batch = count if gpu else min(configured_batch_size(), count)
    matrix = 16*d*d
    if factorization == 'compressed':
        from ghost_backend.compressed.runtime import storage_budget
        from ghost_backend.linalg.refined_lu import requested_precision
        if requested_precision()!='double':raise ValueError('Compressed factorization requires double precision.')
        if requested == 'gpu':raise ValueError('Compressed factorization requires CPU execution.')
        from ghost_backend.compressed.memory import forecast
        plan = forecast(n, d, count, batch, get_assembly_threads(), storage_budget(), resources)
        if dense_resources is not None:
            dense_resources['memory_estimate'] = plan
        from ghost_backend.execution.cpu import current_state
        state = current_state()
        if state is not None and plan not in state.memory_estimates:
            state.memory_estimates.append(plan)
        return plan['peak_bytes']/1024**3

    solve = 2*matrix + 16*12*d*batch + 64*1024**2
    if factorization == 'hierarchical':


        from ghost_backend.linalg.hierarchical import factor_storage_budget
        solve = matrix + factor_storage_budget(matrix) + 16*12*d*max(256,batch) + 64*1024**2
    workspace = (64 + 128*get_assembly_threads()) * 1024**2 + 16*512*n
    if kind == 'multi_region' and 'operator_entries' in resources:
        assembly = matrix + 16*resources.get('assembly_operator_entries', resources['operator_entries']) + resources['operator_map_bytes']
        assembly += resources['mass_workspace_bytes'] + max(
            resources['block_workspace_bytes'], resources['assembly_workspace_bytes'])
    else:
        slots = {'single_dielectric': 0, 'sheet': 0, 'mixed_sheet_pec': 0,
                 'te_robin': 0, 'robin': 0, 'thin_dielectric_layer': 0}.get(kind,
                 max(0, int(operator_matrices)) if operator_matrices is not None else max(1,n_regions)*(8 if use_cfie else 4))
        assembly = matrix + 16*slots*n*n + workspace

    extra = CACHE_BYTES + TABLE_BYTES if solver_method == EXPERIMENTAL_METHOD else 0
    return (max(assembly, solve) + extra + count*4096) / 1024**3


def _solve_te_robin_mfie(mesh, infos, pol, k0, elevations_deg, obs_order=8, src_order=8,
                         solver_method="auto", condition_diagnostics=None, operator_cache=None):
    """Generalized TE MFIE with one factorization and bounded angle batches."""
    _normalize_public_2d_solver_method(solver_method)
    return _solve_robin_bie(mesh, infos, pol, k0, elevations_deg, obs_order, src_order,
                            condition_diagnostics, operator_cache)

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
    """Return True for a body containing only TYPE 3 dielectric interfaces.

    TYPE 1 sheets are excluded because their impedance terms require the
    sheet formulations."""
    if _has_sheet(infos):
        return False
    return all(info.bc_kind == 'transmission' for info in infos)


def _assert_no_type1_sheet_for_mixed(infos: 'List[PanelCoupledInfo]') -> 'None':
    """Validate TYPE 1 sheet combinations before operator assembly.

    Accept all-sheet geometries and sheets combined with pure PEC TYPE 2
    bodies. Raise ValueError for sheets combined with IBC, dielectric, or
    layered boundaries."""
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

def _solve_dielectric_indirect(mesh, infos, pol, k0, elevations_deg, obs_order=8, src_order=8,
                                condition_diagnostics=None):
    """Two-density dielectric equations with shared assembly and bounded solves."""
    from ghost_backend.twod.formulations.dielectric import assemble_system, rhs_many
    from ghost_backend.twod.fields import solve_fields
    matrix = assemble_system(mesh, infos, pol, k0, obs_order, src_order)
    return solve_fields(mesh, matrix, k0, elevations_deg,
        lambda angles: rhs_many(mesh, k0, angles), condition_diagnostics,
        'dielectric indirect system', potential='DLP', order=obs_order)[:3]

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


def _solve_tm_sheet(mesh, infos, k0, elevations_deg, obs_order=8, src_order=8,
                    condition_diagnostics=None):
    return _solve_mixed_sheet_pec(mesh, infos, 'TM', k0, elevations_deg,
                                 obs_order, src_order, condition_diagnostics)


def _solve_te_sheet(mesh, infos, k0, elevations_deg, obs_order=8, src_order=8,
                    condition_diagnostics=None):
    return _solve_mixed_sheet_pec(mesh, infos, 'TE', k0, elevations_deg,
                                 obs_order, src_order, condition_diagnostics)


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

            return False
    return has_sheet and has_pec


def _solve_mixed_sheet_pec(mesh, infos, pol, k0, elevations_deg, obs_order=8, src_order=8,
                           condition_diagnostics=None):
    """Unified sheet/PEC equations with local impedance terms and edge pins."""
    from ghost_backend.twod.formulations.sheet import assemble_system, rhs_many
    from ghost_backend.twod.fields import solve_fields
    matrix, endpoints = assemble_system(mesh, infos, pol, k0, obs_order, src_order)
    return solve_fields(mesh, matrix, k0, elevations_deg,
        lambda angles: rhs_many(mesh, k0, angles, pol, endpoints), condition_diagnostics,
        'sheet {} system'.format(pol), potential='SLP' if pol == 'TM' else 'DLP', order=obs_order)[:3]


def _assemble_robin_bie_system(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    operator_cache: 'Optional[Dict[Any, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Assemble the Robin single-layer-potential boundary system.

    Return (a_sys, alpha_elements, pec_node). The matrix includes the
    per-row TM-PEC EFIE terms. alpha_elements contains the constant Robin
    coefficient for each element's weak observation integral; pec_node
    marks nodes incident on a PEC element."""

    from ghost_backend.twod.formulations.robin import assemble_system
    return assemble_system(mesh, infos, pol, k0, obs_order, src_order, operator_cache)


def _robin_bie_rhs_many(mesh, alpha_elements, pec_node, pol, k0, elevations_deg):
    """One moment traversal for the needed incident traces and weighted loads."""
    from ghost_backend.twod.assembly.kernels import incident_loads
    elev = np.asarray(elevations_deg, float).reshape(-1)
    alpha = np.asarray(alpha_elements, complex).reshape(-1)
    if len(alpha) != len(mesh.elements):
        raise ValueError('Robin RHS coefficient count must match mesh elements.')
    all_pec = pol == 'TM' and np.all(pec_node)
    bu, bdn = incident_loads(mesh, k0, elev, want_u=bool(pol == 'TM' and np.any(pec_node)),
                            want_dn=not all_pec, observation_coefficients=None if all_pec else alpha)
    if all_pec:
        return -bu
    rhs = -bdn
    if pol == 'TM' and np.any(pec_node):
        rhs[pec_node] = -bu[pec_node]
    return rhs


def _solve_robin_bie(mesh, infos, pol, k0, elevations_deg, obs_order=8, src_order=8,
                     condition_diagnostics=None, operator_cache=None):
    """SLP Robin equations, including the TM PEC EFIE limit."""
    from ghost_backend.twod.fields import solve_fields
    matrix, alpha, pec = _assemble_robin_bie_system(mesh, infos, pol, k0, obs_order, src_order,
                                                    operator_cache=operator_cache)
    return solve_fields(mesh, matrix, k0, elevations_deg,
        lambda angles: _robin_bie_rhs_many(mesh, alpha, pec, pol, k0, angles),
        condition_diagnostics, 'Robin-BIE IBC system', order=obs_order)[:3]


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
    """Return True for layered, coated, or mixed PEC/dielectric geometry.

    TYPE 1 sheets are excluded because their impedance terms require the
    sheet formulations."""
    if _has_sheet(infos):
        return False
    n_regions = _count_distinct_regions(infos)
    if n_regions > 2:
        return True


    has_transmission = any(info.bc_kind == 'transmission' for info in infos)
    has_robin = any(info.bc_kind == 'robin' for info in infos)
    return has_transmission and has_robin


def _dense_formulation_resources(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    thin_parameters=None,
) -> 'Dict[str, Any]':
    """Return system dimensions and retained-operator/workspace requirements for the
    selected formulation.
    """

    nnodes = int(len(mesh.nodes))
    storage = {}
    regions = {
        int(region)
        for info in infos
        for region in (info.minus_region, info.plus_region)
        if int(region) >= 0
    }

    if any(info.bc_kind == "thin_layer" for info in infos):
        formulation = "thin_dielectric_layer"
        system_dofs = 2 * nnodes
        if thin_parameters is not None:
            eps, mu, thickness = thin_parameters
            if (mu if pol == 'TM' else eps) == 1:
                system_dofs = nnodes
            if eps == 1 and mu == 1:
                system_dofs = 0
                storage['analytic_zero'] = True
        operator_matrices = 2
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
        from ghost_backend.twod.formulations.regions import build_layout, storage_resources
        layout = build_layout(mesh, infos, pol)
        formulation = "multi_region"
        system_dofs = layout['n_dof']
        storage = storage_resources(mesh, layout)
        operator_matrices = storage['operator_matrices']
    elif _is_single_dielectric_body(infos):
        formulation = "single_dielectric"
        system_dofs = 2 * nnodes
        operator_matrices = 2
    elif _is_all_robin(infos):
        formulation = "robin"
        system_dofs = nnodes
        operator_matrices = 3
    else:
        raise ValueError(
            "Geometry does not match a supported dense 2-D formulation."
        )

    result = {
        "nodes": nnodes,
        "n_regions": int(len(regions)),
        "formulation": formulation,
        "system_dofs": int(system_dofs),
        "operator_matrices": int(operator_matrices),
        **storage,
    }
    from ghost_backend.compressed.runtime import enabled as compressed_enabled
    if compressed_enabled() and not storage.get('analytic_zero'):
        from ghost_backend.compressed.memory import geometry_storage
        air_k = next((i.k_plus if i.plus_region == 0 else i.k_minus for i in infos
                      if i.plus_region == 0 or i.minus_region == 0), None)
        if air_k is None:
            raise ValueError('Compressed resource planning requires an air exterior.')
        result['compressed_storage'] = geometry_storage(mesh, infos, pol, formulation,
            float(complex(air_k).real), thin_parameters, system_dofs)
    return result

def _solve_multi_region_indirect(mesh, infos, pol, k0, elevations_deg,
    obs_order=8, src_order=8, solver_method="auto", condition_diagnostics=None,
    return_density=True, observation_angles_deg=None, project=True):
    """Shared multi-region assembly, bounded solves and exterior projection."""
    _normalize_public_2d_solver_method(solver_method)
    from ghost_backend.twod.formulations.regions import (
        assemble_system,
        rhs_many,
        exterior_projection,
        dof_coordinates,
    )
    from ghost_backend.twod.fields import solve_fields
    matrix, layout = assemble_system(mesh, infos, pol, obs_order, src_order)
    mask, density = exterior_projection(mesh, layout)
    return solve_fields(mesh, matrix, k0, elevations_deg,
        lambda angles: rhs_many(mesh, layout, k0, angles), condition_diagnostics,
        "multi-region indirect system", density_builder=density, element_mask=mask,
        observation_angles=observation_angles_deg, order=obs_order,
        return_density=return_density, project=project, coordinates=dof_coordinates(mesh, layout))

@prepared_execution
@configured_execution
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
        base_dir, preflight_report, materials, _ = prepare_geometry(
            geometry_snapshot, material_base_dir, geometry_units)
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
            segment_wavelengths=segment_wavelengths(geometry_snapshot, materials, set(frequencies) | {mesh_ref_ghz}, unit_scale, ref_lambda),
        )

        ref_infos = _build_coupled_panel_info(cached_panels, materials, mesh_ref_ghz, pol, ref_k0)
        cached_mesh, cached_mesh_stats = _build_linear_mesh_interface_aware(
            cached_panels, ref_infos,
        )
        cached_mesh_stats = dict(cached_mesh_stats)
        cached_mesh_stats.update(_linear_coupled_node_report(
            cached_mesh,
            _build_linear_coupled_infos(cached_mesh, materials, mesh_ref_ghz, pol, ref_k0),
        ))


        ref_coupled = _build_linear_coupled_infos(cached_mesh, materials, mesh_ref_ghz, pol, ref_k0)
        cached_junction_constraints, cached_junction_stats = _build_linear_junction_constraints(
            cached_mesh, ref_coupled, materialize=False,
        )
        if shared_cache is not None:


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
                    segment_wavelengths=segment_wavelengths(geometry_snapshot, materials, [mesh_freq_ghz], unit_scale, lambda_min),
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


        resources = _dense_formulation_resources(mesh, coupled_infos, pol,
            layer_for_mesh(mesh, materials, freq_ghz) if any(i.bc_kind == 'thin_layer' for i in coupled_infos) else None)
        def batch_progress(completed, total):
            if progress_callback is not None:
                progress_floor[0] = max(progress_floor[0], done_steps + completed)
                progress_callback(progress_floor[0], total_steps,
                    "Experimental CPU: solved {} of {} angles at {} GHz".format(completed, total, freq_ghz))
        select_formulation(resources, batch_progress)
        from ghost_backend.compressed.runtime import enabled as compressed_enabled
        if compressed_enabled() and pol == 'TE':
            from ghost_backend.twod.assembly.session import current_session
            assembly=current_session()
            if assembly is not None:
                assembly.compressed_partner=(mesh,_build_linear_coupled_infos(mesh,materials,freq_ghz,'TM',k0))
        est_gb = _estimate_memory_gb(
            resources["nodes"],
            use_cfie=False,
            n_regions=max(1, resources["n_regions"]),
            system_dofs=resources["system_dofs"],
            operator_matrices=resources["operator_matrices"],
            dense_resources=resources,
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


        junction_stats.update(linear_mesh_stats_local)
        junction_stats["linear_node_count"] = int(len(mesh.nodes))
        junction_stats["linear_element_count"] = int(len(mesh.elements))


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


        use_te_robin_mfie = (pol == 'TE' and _is_all_robin(coupled_infos))

        if use_te_robin_mfie:
            formulation_label = "2D MFIE TE Robin (SLP representation)"
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, mfie_residual = _solve_te_robin_mfie(
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


        use_multi_region = _is_multi_region(coupled_infos)

        if use_multi_region:
            formulation_label = "2D multi-region indirect SLP formulation (layered coating)"
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, multi_residual, _ = _solve_multi_region_indirect(
                mesh=mesh,
                infos=coupled_infos,
                pol=pol,
                k0=k0,
                elevations_deg=elevations_arr,
                solver_method=solver_method,
                condition_diagnostics=condition_diagnostics,
                return_density=False,
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


        use_dielectric_indirect = _is_single_dielectric_body(coupled_infos)

        if use_dielectric_indirect:
            formulation_label = "2D indirect two-density dielectric formulation"
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, diel_residual = _solve_dielectric_indirect(
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


        use_robin_bie = _is_all_robin(coupled_infos)

        if use_robin_bie:
            formulation_label = (
                "2D Robin-BIE (SLP representation; element-weighted IBC, TM-PEC EFIE override)"
                if pol == 'TM'
                else "2D Robin-BIE IBC formulation (SLP representation)"
            )
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, robin_residual = _solve_robin_bie(
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
            (condition_diagnostics or {}).get('condition_method', 'equilibrated_1norm_lu_onenormest')
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

    if metadata.get('compressed_factors'):metadata['solver_method']='compressed_cpu'
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
        "cpu_factorization_requested": first_metadata.get('cpu_factorization_requested', 'dense'),
        "compressed_factors": [v for m in channel_metadata.values() for v in m.get('compressed_factors',[])],
        "cpu_rhs_compression_requested": first_metadata.get('cpu_rhs_compression_requested', 'auto'),
        "hierarchical_factors": {label: channel_metadata[label].get('hierarchical_factors', []) for label in expected_channels},
        "sweep_compression": {label: channel_metadata[label].get('sweep_compression', []) for label in expected_channels},
        "dense_factorization_count": sum(m.get('dense_factorization_count', 0) for m in channel_metadata.values()),
        "dense_rhs_batch_count": sum(m.get('dense_rhs_batch_count', 0) for m in channel_metadata.values()),
        "dense_rhs_column_count": sum(m.get('dense_rhs_column_count', 0) for m in channel_metadata.values()),
        "dense_max_rhs_columns": max(m.get('dense_max_rhs_columns', 0) for m in channel_metadata.values()),
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
    def completed():
        for index, frequency in enumerate(frequencies):
            def report(done, total, message):
                if progress is not None:
                    progress(index * 1000 + int(1000 * done / max(total, 1)),
                             1000 * len(frequencies), message)
            yield solve(frequencies_ghz=[frequency], progress_callback=report, **kwargs)
    return _merge_frequency_results(completed(), frequencies)


def _merge_frequency_results(results, frequencies):
    """Consume each frequency once, retaining rows and metadata without child results."""
    import json
    result, records = None, []
    for value in results:
        if result is None:
            result = {key: entry for key,entry in value.items() if key not in ('samples', 'co_solved_samples', 'metadata')}
            result['samples'] = []
            result['co_solved_samples'] = {pol: [] for pol in ('VV', 'HH')}
        result['samples'].extend(value['samples'])
        for pol in ('VV', 'HH'):
            result['co_solved_samples'][pol].extend(value['co_solved_samples'][pol])
        records.append(value['metadata'])
    if result is None or len(records) != len(frequencies):
        raise ValueError('Expected one completed result per requested frequency.')

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
            merged, seen = [], set()
            for value in items:
                for entry in value:
                    identity = json.dumps(entry, sort_keys=True, default=lambda v: v.item())
                    if identity not in seen:
                        seen.add(identity)
                        merged.append(entry)
            return merged
        return first

    key = lambda row: (float(row['frequency_ghz']), float(row['theta_inc_deg']),
                       float(row['theta_scat_deg']), 0 if row['polarization'] == 'VV' else 1)
    result['samples'].sort(key=key)
    for rows in result['co_solved_samples'].values():
        rows.sort(key=key)
    metadata = combine(records)
    selections = [record['backend_selection'] for record in records if record.get('backend_selection')]
    selected = sorted(set(record['selected'] for record in selections))
    if len(selected) > 1:
        metadata['backend_selection'] = dict(requested='adaptive', selected='mixed', choices=selected,
            reason='Selected separately per frequency; see frequency metadata for forecasts and decisions.')
        if metadata.get('requested_execution_options'):
            metadata['execution_options'] = dict(metadata['requested_execution_options'])
    strategies = sorted(set(record['mesh_strategy_used'] for record in records if record.get('mesh_strategy_used')))
    if len(strategies) > 1:
        metadata['mesh_strategy_used'] = 'mixed (see frequency metadata)'

    metadata['frequency_metadata'] = [
        {'frequency_ghz': float(frequency), 'metadata': value}
        for frequency, value in zip(frequencies, records)]
    metadata['operator_cache_scope'] = 'one_frequency'
    metadata['panel_count_min'] = min(value.get('panel_count_min', value.get('panel_count', 0)) for value in records)
    for key in ('dense_factorization_count', 'dense_rhs_batch_count', 'dense_rhs_column_count',
                'assembled_system_reuses', 'shared_operator_cache_hits', 'shared_operator_cache_stores',
                'reused_matrix_solve_count', 'residual_nonfinite_count'):
        metadata[key] = sum(value.get(key, 0) for value in records)
    metadata['warning_count'] = len(metadata.get('warnings', []))
    result['metadata'] = metadata
    return result


@prepared_execution
@configured_execution
@profiled_solve
@experimental_monostatic
@shared_assembly
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


@prepared_execution
@configured_execution
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
    panel_counts = []
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


        use_tm_robin_bie = (pol == 'TM' and _is_all_robin(coupled_infos))
        use_diel_indirect = _is_single_dielectric_body(coupled_infos) and not _is_multi_region(coupled_infos)
        use_multi_region = _is_multi_region(coupled_infos)

        panel_counts.append(len(mesh.elements))
        resources = _dense_formulation_resources(mesh, coupled_infos, pol,
            layer_for_mesh(mesh, materials, freq_ghz) if any(i.bc_kind == 'thin_layer' for i in coupled_infos) else None)
        est_gb = _estimate_memory_gb(
            resources["nodes"],
            use_cfie=False,
            n_regions=max(1, resources["n_regions"]),
            system_dofs=resources["system_dofs"],
            operator_matrices=resources["operator_matrices"],
            dense_resources=resources,
            n_rhs=len(inc_angles),
        )


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


        from ghost_backend.twod.fields import solve_fields
        inc_all_arr = np.asarray(inc_angles, dtype=float)
        condition_diagnostics = {} if compute_condition_number else None
        if use_thin_sheet:
            eps, mu, thickness = layer_for_mesh(mesh, materials, freq_ghz)
            _, batch_amp, residual, evidence = solve_thin_layer_fields(
                mesh, k0, inc_all_arr, pol, eps, mu, thickness,
                observation_angles_deg=obs_arr, condition_diagnostics=condition_diagnostics)
            thin_layer_evidence.append(dict(evidence, frequency_ghz=float(freq_ghz), polarization=pol))
        elif use_multi_region:
            _, batch_amp, residual, _ = _solve_multi_region_indirect(
                mesh, coupled_infos, pol, k0, inc_all_arr,
                condition_diagnostics=condition_diagnostics, return_density=False,
                observation_angles_deg=obs_arr)
        else:
            if use_sheet or use_mixed_sheet:
                from ghost_backend.twod.formulations.sheet import assemble_system, rhs_many
                matrix, endpoints = assemble_system(mesh, coupled_infos, pol, k0)
                potential = 'SLP' if pol == 'TM' else 'DLP'
                rhs_builder = lambda angles: rhs_many(mesh, k0, angles, pol, endpoints)
            elif use_te_robin_mfie or use_tm_robin_bie:
                matrix, alpha_elements, pec_nodes = _assemble_robin_bie_system(mesh, coupled_infos, pol, k0)
                rhs_builder = lambda angles: _robin_bie_rhs_many(mesh, alpha_elements, pec_nodes, pol, k0, angles)
                potential = 'SLP'
            elif use_diel_indirect:
                from ghost_backend.twod.formulations.dielectric import assemble_system, rhs_many
                matrix = assemble_system(mesh, coupled_infos, pol, k0)
                rhs_builder = lambda angles: rhs_many(mesh, k0, angles)
                potential = 'DLP'
            else:
                raise ValueError("Geometry did not match a supported bistatic formulation.")
            _, batch_amp, residual, _ = solve_fields(mesh, matrix, k0, inc_all_arr,
                rhs_builder, condition_diagnostics, "bistatic " + pol + " system",
                potential=potential, observation_angles=obs_arr)
            matrix = rhs_builder = None
        batch_residuals = np.full(inc_all_arr.size, residual)
        if condition_diagnostics is not None:
            _consume_condition_estimate(cond_values, condition_diagnostics, "bistatic system")
            frequency_condition_recorded = True
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
        "panel_count": max(panel_counts) if panel_counts else 0,
        "panel_count_min": min(panel_counts) if panel_counts else 0,
        "panel_count_max": max(panel_counts) if panel_counts else 0,
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
            (condition_diagnostics or {}).get('condition_method', 'equilibrated_1norm_lu_onenormest')
            if compute_condition_number else "not_requested"
        ),
        "thin_layer": thin_layer_evidence,
        "warnings": list(materials.warnings),
        "warning_count": int(len(materials.warnings)),
        "preflight": dict(preflight_report),
        **_dense_backend_summary(),
    }
    if metadata.get('compressed_factors'):metadata['solver_method']='compressed_cpu'
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


@prepared_execution
@configured_execution
@profiled_solve
@shared_assembly
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

    if len(frequencies_ghz) > 1:
        return _frequency_local_co_solve(solve_bistatic_rcs_2d, locals())
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


    merged["metadata"]["co_solve_shared_discretization"] = False
    return merged


def _run_certified_2d_pair(*args, **kwargs):
    from ghost_backend.execution.options import option, execution_scope
    try:
        return _run_certified_2d_pair_impl(*args, **kwargs)
    except ValueError as exc:
        if option('mesh_strategy', 'global') != 'local' or not str(exc).startswith('Certified 2-D mesh convergence failed:'):
            raise
        reason = str(exc)
    progress = kwargs.get('progress_callback', args[4] if len(args) > 4 else None)
    if progress is not None:
        progress(0, 1, 'Local mesh comparison failed; retrying global material sizing.')
    options = dict(current_options(), mesh_strategy='global')
    with execution_scope(options):
        result = _run_certified_2d_pair_impl(*args, **kwargs)
    result['metadata']['local_mesh_fallback'] = reason
    result['metadata']['mesh_strategy_used'] = 'global'
    return result


def _run_certified_2d_pair_impl(
    low_level_solver: 'Callable[..., Dict[str, Any]]',
    geometry_snapshot: 'Dict[str, Any]',
    solver_kwargs: 'Dict[str, Any]',
    mesh_convergence_policy: 'Optional[Dict[str, Any]]',
    progress_callback: 'Optional[Callable[[int, int, str], None]]',
    shared_discretization_caches: 'Optional[Tuple[Dict[str, Any], Dict[str, Any]]]' = None,
) -> 'Dict[str, Any]':
    """Run base/fine 2-D solves and publish only a certified fine result."""

    from ghost_backend.runs.quality import (
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


    common["strict_quality_gate"] = True
    common["compute_condition_number"] = True

    base_kwargs = dict(common)
    base_kwargs["geometry_snapshot"] = geometry_snapshot
    base_kwargs["progress_callback"] = _phase_callback("base")
    if shared_discretization_caches is not None:
        base_kwargs["_shared_discretization_cache"] = (
            shared_discretization_caches[0]
        )
    from ghost_backend.execution.metrics import solve_phase
    with solve_phase('Base mesh'):
        base_result = low_level_solver(**base_kwargs)

    fine_snapshot = scale_snapshot_panel_density(
        geometry_snapshot, policy["fine_factor"]
    )


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
    with solve_phase('Refined mesh'):
        fine_result = low_level_solver(**fine_kwargs)

    if "co_solved_samples" in base_result:
        channels = {}
        for export_pol, internal_pol in _CO_POLARIZED_2D_CHANNELS:
            def channel(result):
                value = dict(result)
                value.pop('co_solved_samples', None)
                value['samples'] = result['co_solved_samples'][export_pol]
                value['metadata'] = dict(result['metadata']['channel_metadata'][export_pol])
                return value
            channels[export_pol] = _certify_2d_results(channel(base_result), channel(fine_result), policy)
        result = _merge_co_polarized_2d_results(channels)
        result['metadata']['assembled_system_reuses'] = sum(
            r['metadata'].get('assembled_system_reuses', 0) for r in (base_result, fine_result))
        counters = ('dense_factorization_count', 'dense_rhs_batch_count', 'dense_rhs_column_count')
        result['metadata']['certification_phase_counters'] = {
            phase: {key: run['metadata'].get(key, 0) for key in counters}
            for phase, run in (('base', base_result), ('fine', fine_result))}
        for key in counters:
            result['metadata'][key] = sum(run['metadata'].get(key, 0) for run in (base_result, fine_result))
        result['metadata']['certification_solve_order'] = ['base_TE', 'base_TM', 'fine_TE', 'fine_TM']
        return result
    return _certify_2d_results(base_result, fine_result, policy)


@timed_stage('mesh_certification')
def _certify_2d_results(base_result, fine_result, policy):
    from ghost_backend.runs.quality import evaluate_mesh_convergence
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
    from ghost_backend.execution.options import option
    metadata['mesh_strategy_used'] = option('mesh_strategy', 'global')
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


@prepared_execution
@configured_execution
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


@prepared_execution
@configured_execution
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


@prepared_execution
@configured_execution
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
    kwargs = dict(locals())
    kwargs.pop('geometry_snapshot')
    kwargs.pop('mesh_convergence_policy')
    kwargs.pop('progress_callback')
    def solve_phase(**kw):
        return solve_monostatic_rcs_2d(**kw)
    return _run_certified_2d_pair(solve_phase, geometry_snapshot, kwargs,
                                 mesh_convergence_policy, progress_callback)


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


@prepared_execution
@configured_execution
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


@prepared_execution
@configured_execution
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

    if len(frequencies_ghz) > 1:
        return _frequency_local_co_solve(solve_bistatic_rcs_2d_certified, locals())
    kwargs = dict(locals())
    kwargs.pop('geometry_snapshot')
    kwargs.pop('mesh_convergence_policy')
    kwargs.pop('progress_callback')
    def solve_phase(**kw):
        kw.pop('solver_method', None)
        return solve_bistatic_rcs_2d(**kw)
    return _run_certified_2d_pair(solve_phase, geometry_snapshot, kwargs,
                                 mesh_convergence_policy, progress_callback)


@prepared_execution
@configured_execution
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


@shared_assembly
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


    use_robin = _is_all_robin(coupled_infos)

    resources = _dense_formulation_resources(mesh, coupled_infos, pol)
    check_abort()
    est_gb = _estimate_memory_gb(
        resources["nodes"],
        use_cfie=False,
        n_regions=max(1, resources["n_regions"]),
        system_dofs=resources["system_dofs"],
        operator_matrices=resources["operator_matrices"],
        dense_resources=resources,
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


    from ghost_backend.compressed.runtime import enabled as compressed_enabled
    if compressed_enabled():
        from ghost_backend.compressed.factor import CompressedFactor as DensityFactor
    else:
        from ghost_backend.linalg.dense import DenseFactor as DensityFactor

    if use_multi:

        check_abort()
        _, _, _, ext_density = _solve_multi_region_indirect(
            mesh, coupled_infos, pol, k0, elev_arr, project=False)
        check_abort()
        sigma_nodes = ext_density[:, 0]
        density = np.asarray([
            0.5 * (sigma_nodes[e.node_ids[0]] + sigma_nodes[e.node_ids[1]])
            for e in mesh.elements
        ], dtype=np.complex128)
        formulation = "Multi-region indirect (exterior SLP density)"

    elif use_diel:
        from ghost_backend.twod.formulations.dielectric import assemble_system, rhs_many
        matrix = assemble_system(mesh, coupled_infos, pol, k0)
        check_abort()
        mu_nodes = DensityFactor(matrix, checkpoint=check_abort).solve(rhs_many(mesh, k0, elev_arr))[:nnodes, 0]
        density = np.asarray([
            0.5*(mu_nodes[e.node_ids[0]]+mu_nodes[e.node_ids[1]])
            for e in mesh.elements
        ], dtype=np.complex128)
        formulation = "Indirect dielectric (DLP density)"

    elif use_robin:


        a_sys, alpha_elements, pec_node = _assemble_robin_bie_system(
            mesh, coupled_infos, pol, k0
        )
        check_abort()
        rhs = _robin_bie_rhs_many(
            mesh, alpha_elements, pec_node, pol, k0, elev_arr
        )
        check_abort()
        sigma_nodes = DensityFactor(a_sys, checkpoint=check_abort).solve(rhs)[:, 0]
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
