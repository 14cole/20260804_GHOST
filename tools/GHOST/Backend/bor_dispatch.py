"""
BoR dispatch (phase 4): solve .geo geometry snapshots with the BoR-MoM solver.

The drawing's (x, y) plane is reinterpreted as the (rho, z) half-plane:
x = rho (distance from the rotation axis, must be >= 0) and y = z (the
rotation axis, drawn vertically).  A closed body of revolution is an OPEN
generatrix polyline whose two endpoints lie ON the axis, traversed from the
+z end (nose) to the -z end (tail) so the left-of-travel normal faces the
exterior (BOR_CONVENTIONS.md).  Wrong traversal or non-axis endpoints are
hard preflight errors, never silently corrected -- same philosophy as the 2D
solver's orientation checks.

Supported material configurations (segment TYPE semantics shared with the
2D solver / MaterialLibrary):

  * all TYPE 2 (PEC, optionally with IBC flags incl. tapers and material
    tables)                      -> CFIE (pure PEC) / IBC-EFIE (any Z_s != 0)
  * all TYPE 3, one pos_mat     -> homogeneous penetrable body (PMCHWT)
  * TYPE 3 outer + TYPE 4 core,
    matching pos_mat            -> coated PEC (multi-region PMCHWT)
  * TYPE 2 + TYPE 3 + TYPE 4   -> partial coating with junction circles
  * TYPE 3 + TYPE 5... + TYPE 4
                                -> enumerated layered stacks and coating patches
  * supported TYPE 2/3/4/5 band layouts
                                -> side-by-side coating bands

Anything else--including TYPE 1 sheets and arbitrary mixed PEC/dielectric
graphs--raises with a named error. TYPE 5 is accepted only in the explicitly
classified layouts above, not as a general interface graph.

The production entry points share the 2-D solver's geometry/frequency/aspect
argument names and dual-channel result schema.  BoR-specific controls remain
explicit; sigma is 3-D RCS in m^2 (dBsm).
"""

import cmath
import math
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from bor_kernels import C0
from bor_solver import (BOR_CONDITION_EST_MAX,
                        BOR_LINEAR_BACKWARD_ERROR_MAX,
                        BOR_LINEAR_RESIDUAL_MAX,
                        solve_bor,
                        solve_bor_dielectric, solve_bor_coated_pec,
                        solve_bor_partial_coating, solve_bor_coated2_pec,
                        solve_bor_coated_n_pec, solve_bor_coating_patch,
                        _MultiRegionBor, _solve_multiregion,
                        _bor_mode_limits)
from rcs_solver import (
    MaterialLibrary,
    _conservative_mesh_wavelength_for_frequencies,
    _material_base_dir_for_snapshot,
    validate_geometry_snapshot_for_solver,
)
from solver_quality import (
    evaluate_mesh_convergence,
    scale_snapshot_panel_density,
    validate_mesh_convergence_policy,
)

DEFAULT_ELEMENTS_PER_WAVELENGTH = 20
MAX_ELEMENTS_DEFAULT = 50_000
BOR_POWER_CONSISTENCY_RTOL = 2.0e-8
BOR_STREAM_BUDGET_GB_DEFAULT = 8.0


def _unit_scale_to_meters(units: 'str') -> 'float':
    value = str(units or "").strip().lower()
    if value in {"inch", "inches", "in"}:
        return 0.0254
    if value in {"meter", "meters", "m"}:
        return 1.0
    raise ValueError(f"Unsupported geometry units '{units}'. Use inches or meters.")


def _parse_flag(tok: 'Any', default: 'int' = 0) -> 'int':
    try:
        text = str(tok).strip().lower()
        if text.startswith("mat."):
            text = text[4:]
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _parse_int(tok: 'Any', default: 'int' = 0) -> 'int':
    try:
        text = str(tok).strip()
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _reject_unsupported_bor_ibc_interfaces(
    geometry_snapshot: 'Dict[str, Any]',
) -> 'None':
    """Reject impedance flags that the selected BoR material paths ignore.

    TYPE 2 is the only BoR segment whose IBC is currently assembled.  The
    2-D solver supports a TYPE 4 Robin backing, but the BoR coated/multiregion
    formulations currently model TYPE 4 as PEC, so accepting its flag would
    silently change the requested boundary condition.
    """

    for seg_idx, seg in enumerate(geometry_snapshot.get("segments", []) or []):
        props = list(seg.get("properties", []) or [])
        seg_type = _parse_flag(
            props[0] if len(props) > 0 and str(props[0]).strip()
            else seg.get("seg_type", 2),
            2,
        )
        ibc_flag = _parse_flag(props[2] if len(props) > 2 else 0)
        if ibc_flag > 0 and seg_type in (3, 4, 5):
            name = str(seg.get("name", f"segment_{seg_idx + 1}"))
            raise ValueError(
                f"BoR TYPE {seg_type} segment '{name}' assigns IBC flag "
                f"{ibc_flag}, but impedance on TYPE {seg_type} is not "
                "implemented by the BoR material formulation. Remove the "
                "flag or place the IBC on a supported TYPE 2 conductor."
            )


class _SegChain:
    """One segment's polyline in scaled (rho, z) coordinates."""

    def __init__(self, name: 'str', seg_type: 'int', n_prop: 'int', ibc_flag: 'int',
                 pos_mat: 'int', neg_mat: 'int', pts: 'np.ndarray'):
        self.name = name
        self.seg_type = seg_type
        self.n_prop = n_prop
        self.ibc_flag = ibc_flag
        self.pos_mat = pos_mat
        self.neg_mat = neg_mat
        self.pts = pts                       # (N, 2) columns rho, z
        d = np.diff(pts, axis=0)
        self.prim_lengths = np.hypot(d[:, 0], d[:, 1])
        self.length = float(np.sum(self.prim_lengths))


def _chains_from_snapshot(snapshot: 'Dict[str, Any]', scale: 'float') -> 'List[_SegChain]':
    chains: 'List[_SegChain]' = []
    for seg_idx, seg in enumerate(snapshot.get("segments", []) or []):
        props = list(seg.get("properties", []) or [])
        seg_type = _parse_flag(
            props[0] if len(props) > 0 and str(props[0]).strip() else seg.get("seg_type", 2), 2)
        n_prop = _parse_int(props[1] if len(props) > 1 else 0, 0)
        ibc_flag = _parse_flag(props[2] if len(props) > 2 else 0)
        pos_mat = _parse_flag(props[3] if len(props) > 3 else 0)
        neg_mat = _parse_flag(props[4] if len(props) > 4 else 0)
        pts: 'List[Tuple[float, float]]' = []
        for i, pair in enumerate(list(seg.get("point_pairs", []) or [])):
            x1 = float(pair.get("x1", 0.0)) * scale
            y1 = float(pair.get("y1", 0.0)) * scale
            x2 = float(pair.get("x2", 0.0)) * scale
            y2 = float(pair.get("y2", 0.0)) * scale
            if i == 0:
                pts.append((x1, y1))
            elif math.hypot(x1 - pts[-1][0], y1 - pts[-1][1]) > 0:
                # primitives inside one segment must chain head-to-tail
                raise ValueError(
                    f"Segment '{seg.get('name', seg_idx)}': primitives do not chain "
                    f"head-to-tail at ({x1 / scale:.6g}, {y1 / scale:.6g}).")
            pts.append((x2, y2))
        if len(pts) < 2:
            continue
        chains.append(_SegChain(str(seg.get("name", f"segment_{seg_idx + 1}")),
                                seg_type, n_prop, ibc_flag, pos_mat, neg_mat,
                                np.asarray(pts, dtype=float)))
    if not chains:
        raise ValueError("Geometry contains no usable segments.")
    return chains


def _stitch_generatrix(chains: 'List[_SegChain]', what: 'str',
                       tol: 'float') -> 'List[_SegChain]':
    """Order chains head-to-tail into a single generatrix run.  Junctions
    must match END of one chain to START of the next AS DRAWN (reversing a
    segment silently would flip its normal and its taper direction)."""

    if len(chains) == 1:
        return chains

    def key(p) -> 'Tuple[int, int]':
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))

    start_of = {}
    end_of = {}
    for c in chains:
        ks, ke = key(c.pts[0]), key(c.pts[-1])
        if ks in start_of or ke in end_of:
            raise ValueError(f"The {what} segments do not form a single chain "
                             "(two segments start or end at the same point).")
        start_of[ks] = c
        end_of[ke] = c
    heads = [c for c in chains if key(c.pts[0]) not in end_of]
    if len(heads) != 1:
        raise ValueError(
            f"The {what} segments do not chain head-to-tail into one generatrix. "
            "Check that consecutive segments share endpoints and that each "
            "segment is drawn in the same traversal direction (a start-to-start "
            "or end-to-end meeting means one segment's endpoint order must be "
            "reversed).")
    ordered = [heads[0]]
    while True:
        nxt = start_of.get(key(ordered[-1].pts[-1]))
        if nxt is None:
            break
        if nxt is ordered[0]:
            raise ValueError(f"The {what} segments form a closed loop in the "
                             "(rho, z) plane; a BoR generatrix must be an open "
                             "polyline with both endpoints on the axis.")
        ordered.append(nxt)
    if len(ordered) != len(chains):
        raise ValueError(f"The {what} segments split into multiple disconnected "
                         "chains; expected one generatrix.")
    return ordered


def _preflight_generatrix(ordered: 'List[_SegChain]', what: 'str', tol: 'float') -> 'None':
    pts = np.vstack([ordered[0].pts] + [c.pts[1:] for c in ordered[1:]])
    rho, z = pts[:, 0], pts[:, 1]
    if np.any(rho < -tol):
        bad = pts[np.argmin(rho)]
        raise ValueError(
            f"The {what} generatrix crosses the rotation axis (rho = x = "
            f"{bad[0]:.6g} < 0). Draw the half-profile entirely at x >= 0.")
    if rho[0] > tol or rho[-1] > tol:
        raise ValueError(
            f"The {what} generatrix endpoints must lie ON the rotation axis "
            f"(x = 0) to close the body of revolution; got start rho = "
            f"{rho[0]:.6g}, end rho = {rho[-1]:.6g}. Open BoR shells are not "
            "supported in phase 4.")
    if z[0] <= z[-1]:
        raise ValueError(
            f"The {what} generatrix must be traversed from the +z (top) axis "
            "end to the -z (bottom) axis end so the left-of-travel normal "
            f"faces the exterior; it is drawn bottom-to-top (z {z[0]:.6g} -> "
            f"{z[-1]:.6g}). Reverse the segment endpoint order.")


def _element_count(n_prop: 'int', prim_len: 'float', lam_target: 'float') -> 'int':
    if prim_len <= 0.0:
        return 1
    if n_prop > 0:
        return max(1, n_prop)
    n_wave = abs(n_prop) if n_prop < 0 else DEFAULT_ELEMENTS_PER_WAVELENGTH
    target = max(lam_target / max(1, n_wave), prim_len / 2000.0)
    return max(1, int(math.ceil(prim_len / target)))


def _mesh_generatrix(ordered: 'List[_SegChain]', lam_target: 'float',
                     max_elements: 'int', axis_tol: 'float'):
    """Subdivide the ordered chains into elements.  Returns (points [Nn,2],
    elem_seg [Ne] chain index, elem_arc_s [Ne] normalized arc position of the
    element midpoint along its own segment -- the taper coordinate)."""

    points: 'List[Tuple[float, float]]' = []
    elem_seg: 'List[int]' = []
    elem_arc: 'List[float]' = []
    for ci, c in enumerate(ordered):
        arc0 = 0.0
        for pi in range(len(c.pts) - 1):
            p0, p1 = c.pts[pi], c.pts[pi + 1]
            plen = c.prim_lengths[pi]
            cnt = _element_count(c.n_prop, plen, lam_target)
            for i in range(cnt):
                q0 = p0 + (p1 - p0) * (i / cnt)
                if not points:
                    points.append(tuple(q0))
                elem_seg.append(ci)
                elem_arc.append((arc0 + plen * (i + 0.5) / cnt) / max(c.length, 1e-300))
                q1 = p0 + (p1 - p0) * ((i + 1) / cnt)
                points.append(tuple(q1))
            arc0 += plen
    pts = np.asarray(points, dtype=float)
    # snap near-axis coordinates exactly onto the axis (Generatrix requires rho >= 0)
    pts[np.abs(pts[:, 0]) <= axis_tol, 0] = 0.0
    pts[:, 0] = np.maximum(pts[:, 0], 0.0)
    if len(pts) - 1 > max_elements:
        raise ValueError(f"BoR mesh would need {len(pts) - 1} elements "
                         f"(> max {max_elements}). Reduce frequency or density.")
    return pts, np.asarray(elem_seg, dtype=int), np.asarray(elem_arc, dtype=float)


def _classify(chains: 'List[_SegChain]') -> 'str':
    types = {c.seg_type for c in chains}
    if types == {2}:
        return "conductor"
    if types == {3}:
        if len({c.pos_mat for c in chains}) != 1:
            raise ValueError("All TYPE 3 segments of a homogeneous BoR body "
                             "must reference the same pos_mat material.")
        return "dielectric"
    if types in ({3, 4}, {2, 3, 4}):
        pm3 = {c.pos_mat for c in chains if c.seg_type == 3}
        pm4 = {c.pos_mat for c in chains if c.seg_type == 4}
        if len(pm3) != 1 or len(pm4) != 1 or pm3 != pm4:
            raise ValueError("Coated BoR: the TYPE 3 interface and the TYPE 4 "
                             "covered core must reference the same pos_mat "
                             "coating material.")
        if 2 in types:
            return "partial"
        return "coated"
    if types == {2, 3, 4, 5}:
        # banded design: side-by-side coatings on a PEC core with TYPE 5
        # walls between adjacent bands, plus bare conductor pieces
        return "banded"
    if types == {3, 4, 5}:
        if len({c.pos_mat for c in chains if c.seg_type == 4}) > 1:
            return "banded"          # multiple band materials on the core
        pm5 = {(c.pos_mat, c.neg_mat) for c in chains if c.seg_type == 5}
        if len(pm5) > 1:
            return "layered_n"
        outer_flag, inner_flag = next(iter(pm5))
        if outer_flag <= 0 or inner_flag <= 0 or outer_flag == inner_flag:
            raise ValueError("TYPE 5 needs distinct positive pos_mat (outer "
                             "layer) and neg_mat (inner layer) flags.")
        pm4 = {c.pos_mat for c in chains if c.seg_type == 4}
        if pm4 != {inner_flag}:
            raise ValueError("Layered BoR: the TYPE 4 core's pos_mat must be "
                             "the TYPE 5 interface's neg_mat (inner layer).")
        for c in chains:
            if c.seg_type == 3 and c.pos_mat not in (outer_flag, inner_flag):
                raise ValueError("Layered BoR: every TYPE 3 pos_mat must be "
                                 "the outer-layer or inner-layer flag.")
        return "layered"
    unsupported = types - {2, 3, 4, 5}
    if unsupported:
        raise ValueError(f"Segment TYPE(s) {sorted(unsupported)} are not "
                         "supported by the BoR solver (supported: TYPE 2 "
                         "PEC/IBC, TYPE 3 dielectric, TYPE 3+4 coated, "
                         "TYPE 2+3+4 partially coated, TYPE 3+5+4 layered).")
    raise ValueError("Unsupported BoR material combination: TYPE 2 "
                     "conductors can only mix with dielectric interfaces via "
                     "the TYPE 2+3+4 partial-coating layout.")


def _validate_bor_far_controls(kind: 'str', assembly: 'str',
                               table_precision: 'str',
                               stream_budget_gb: 'float') -> 'None':
    """Apply the same far-table control policy to preview and solve paths."""

    if kind == "conductor":
        return
    unsupported = []
    if assembly != "auto":
        unsupported.append(f"assembly={assembly!r}")
    if table_precision != "auto":
        unsupported.append(f"table_precision={table_precision!r}")
    if stream_budget_gb != BOR_STREAM_BUDGET_GB_DEFAULT:
        unsupported.append(f"stream_budget_gb={stream_budget_gb:g}")
    if unsupported:
        raise ValueError(
            f"BoR {kind} formulation does not implement the conductor "
            "far-table/streaming controls "
            f"({', '.join(unsupported)}). Leave assembly and "
            "table_precision at 'auto' and stream_budget_gb at its "
            f"{BOR_STREAM_BUDGET_GB_DEFAULT:g} GB default."
        )


def estimate_bor_resources(
    geometry_snapshot: 'Dict[str, Any]',
    frequency_ghz: 'float',
    aspects_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    n_modes: 'Optional[int]' = None,
    max_elements: 'int' = MAX_ELEMENTS_DEFAULT,
    workers: 'int' = 1,
    table_precision: 'str' = "auto",
    assembly: 'str' = "auto",
    stream_budget_gb: 'float' = BOR_STREAM_BUDGET_GB_DEFAULT,
    mesh_certification: 'bool' = True,
    fine_factor: 'float' = 1.5,
) -> 'Dict[str, Any]':
    """Preview the peak allocation used for memory-aware scheduling.

    Material interpolation and panel-count arithmetic match the solve path,
    but this function performs no quadrature or matrix assembly. The peak is
    intentionally conservative and is a reservation, not an RSS promise.
    """

    frequency = float(frequency_ghz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_ghz must be positive and finite.")
    aspects = np.asarray(aspects_deg, dtype=float)
    if (
        aspects.ndim != 1 or aspects.size == 0
        or not np.all(np.isfinite(aspects))
        or np.any((aspects < 0.0) | (aspects > 180.0))
    ):
        raise ValueError("aspects_deg must be a nonempty finite [0, 180] axis.")
    worker_count = max(1, int(workers))
    precision = str(table_precision).strip().lower()
    assembly_key = str(assembly).strip().lower()
    stream_budget = float(stream_budget_gb)
    if precision not in {"auto", "single", "double"}:
        raise ValueError("table_precision must be auto, single, or double.")
    if assembly_key not in {"auto", "tables", "streaming"}:
        raise ValueError("assembly must be auto, tables, or streaming.")
    if not math.isfinite(stream_budget) or stream_budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")

    preview_snapshot = (
        scale_snapshot_panel_density(geometry_snapshot, float(fine_factor))
        if mesh_certification else geometry_snapshot
    )
    scale = _unit_scale_to_meters(geometry_units)
    base_dir = _material_base_dir_for_snapshot(
        preview_snapshot, material_base_dir
    )
    _reject_unsupported_bor_ibc_interfaces(preview_snapshot)
    validate_geometry_snapshot_for_solver(
        preview_snapshot,
        base_dir=base_dir,
        meters_scale=scale,
    )
    materials = MaterialLibrary.from_entries(
        preview_snapshot.get("ibcs", []) or [],
        preview_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    wavelength, _max_index, _flags = (
        _conservative_mesh_wavelength_for_frequencies(
            preview_snapshot, materials, [frequency]
        )
    )
    chains = _chains_from_snapshot(preview_snapshot, scale)
    kind = _classify(chains)
    _validate_bor_far_controls(
        kind, assembly_key, precision, stream_budget
    )
    groups, _tol, _axis_tol = _prepare_bor_groups(chains, kind)
    surface_layout = _bor_surface_layout(groups, kind, wavelength)
    element_count = _enforce_total_element_limit(
        surface_layout, max_elements
    )
    radius = max(float(np.max(chain.pts[:, 0])) for chain in chains)
    k0 = 2.0 * math.pi * frequency * 1.0e9 / C0
    mode_cap, mode_tail_start = _bor_mode_limits(
        k0, radius, aspects, n_modes
    )
    from bor_solver import (
        BOR_TABLE_BUILD_PEAK_FACTOR,
        estimate_bor_dense_peak_gb,
        estimate_bor_total_peak_gb,
    )

    unknowns = sum(
        (2 if is_conductor else 4) * (int(elements) + 1)
        for elements, is_conductor in surface_layout
    )
    persistent_gb = 0.0
    held_assembly_gb = 0.0
    stream_mode_block = None
    effective_workers = worker_count
    estimated_assembly = "dense-direct"
    estimated_precision = "double"
    if kind == "conductor":
        from bor_solver import estimate_bor_table_gb
        from bor_streaming import (
            BOR_STREAM_TILE_BUDGET_GB,
            estimate_streaming_gb,
            plan_streaming_mode_block,
        )

        has_ibc = any(chain.ibc_flag > 0 for chain in chains)
        formulation = "efie" if has_ibc else "cfie"
        table_double = estimate_bor_table_gb(
            element_count, mode_cap, formulation, has_ibc, 4, False
        )
        use_streaming = (
            assembly_key == "streaming"
            or (assembly_key == "auto" and table_double > 2.0)
        )
        full_double = (
            estimate_streaming_gb(
                element_count, mode_cap, formulation, has_ibc, False
            )
            if use_streaming else table_double
        )
        use_single = (
            precision == "single"
            or (precision == "auto" and full_double > 4.0)
        )
        persistent_gb = full_double / (2.0 if use_single else 1.0)
        held_assembly_gb = persistent_gb
        if use_streaming:
            (
                stream_mode_block,
                held_assembly_gb,
                effective_workers,
            ) = plan_streaming_mode_block(
                element_count,
                mode_cap,
                formulation,
                has_ibc,
                use_single,
                stream_budget,
                worker_count,
            )
        elif not use_streaming:
            held_assembly_gb = BOR_TABLE_BUILD_PEAK_FACTOR * persistent_gb
        estimated_assembly = "streaming" if use_streaming else "tables"
        estimated_precision = "single" if use_single else "double"
        assembly_peak_gb = (
            held_assembly_gb + BOR_STREAM_TILE_BUDGET_GB
            if use_streaming else held_assembly_gb
        )
    else:
        # Penetrable and multi-surface formulations currently retain dense
        # all-mode self/cross-operator tables.  Count every surface's two
        # medium-side self operators and every directed cross-surface pair.
        # Some layouts share fewer regions, so this is deliberately a safe
        # scheduler reservation rather than an optimistic RSS prediction.
        persistent_gb = _estimate_multisurface_operator_gb(
            surface_layout, mode_cap
        )
        held_assembly_gb = BOR_TABLE_BUILD_PEAK_FACTOR * persistent_gb
        assembly_peak_gb = held_assembly_gb
        estimated_assembly = "dense-all-mode-tables"

    active_modes = min(effective_workers, mode_cap + 1)
    # Use the exact same dense-work reservation and effective outer-mode
    # concurrency as the runtime gate.  Keeping a second scheduler planner can
    # over-admit jobs that runtime rejects or collectively exhaust memory.
    dense_peak_gb = estimate_bor_dense_peak_gb(
        unknowns,
        2 * int(aspects.size),
        workers=effective_workers,
        mode_tasks=mode_cap + 1,
    )

    peak_gb = estimate_bor_total_peak_gb(
        assembly_peak_gb,
        dense_peak_gb,
    )
    return {
        "frequency_ghz": frequency,
        "geometry_kind": kind,
        "mesh_elements": int(element_count),
        "surface_count": int(len(surface_layout)),
        "n_unknowns_estimate": int(unknowns),
        "mode_cap_estimate": int(mode_cap),
        "mode_tail_start_estimate": int(mode_tail_start),
        "active_mode_workers": int(active_modes),
        "assembly_estimate": estimated_assembly,
        "table_precision_estimate": estimated_precision,
        "persistent_assembly_gb": float(persistent_gb),
        "held_assembly_gb": float(held_assembly_gb),
        "stream_mode_block_estimate": (
            int(stream_mode_block) if stream_mode_block is not None else None
        ),
        "estimated_peak_gb": float(peak_gb),
        "mesh_certification": bool(mesh_certification),
    }


def _stitch_pieces(chains: 'List[_SegChain]', what: 'str',
                   tol: 'float') -> 'List[List[_SegChain]]':
    """Order chains head-to-tail into MULTIPLE maximal open runs (used for
    the bare-conductor pieces of a partial coating)."""

    def key(p) -> 'Tuple[int, int]':
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))

    start_of = {}
    end_of = {}
    for c in chains:
        ks, ke = key(c.pts[0]), key(c.pts[-1])
        if ks in start_of or ke in end_of:
            raise ValueError(f"Two {what} segments start or end at the same point.")
        start_of[ks] = c
        end_of[ke] = c
    heads = [c for c in chains if key(c.pts[0]) not in end_of]
    runs: 'List[List[_SegChain]]' = []
    used = 0
    for head in heads:
        run = [head]
        while True:
            nxt = start_of.get(key(run[-1].pts[-1]))
            if nxt is None or nxt is head:
                break
            run.append(nxt)
        runs.append(run)
        used += len(run)
    if used != len(chains):
        raise ValueError(f"The {what} segments contain a closed loop or a "
                         "branching junction; expected open head-to-tail runs.")
    return runs


def _prepare_bor_groups(chains: 'List[_SegChain]', kind: 'str'):
    """Build and preflight the material-specific generatrix layout.

    Resource preview and the actual solve must interpret a geometry in exactly
    the same way.  Keeping this topology construction in one place prevents a
    scheduler estimate from counting a different set of surfaces than the
    formulation that will eventually be assembled.
    """

    diag = max(
        float(np.ptp(np.vstack([chain.pts for chain in chains]), axis=0).max()),
        1e-9,
    )
    tol = max(1e-12, 1e-9 * diag)
    axis_tol = 1e-6 * diag

    if kind == "coated":
        outer_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 3],
            "outer-interface (TYPE 3)", tol,
        )
        core_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 4],
            "core (TYPE 4)", tol,
        )
        _preflight_generatrix(outer_chains, "outer-interface", axis_tol)
        _preflight_generatrix(core_chains, "core", axis_tol)
        groups = [outer_chains, core_chains]
    elif kind == "partial":
        iface_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 3],
            "coating-interface (TYPE 3)", tol,
        )
        cov_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 4],
            "covered-core (TYPE 4)", tol,
        )
        bare_runs = _stitch_pieces(
            [chain for chain in chains if chain.seg_type == 2],
            "bare-conductor (TYPE 2)", tol,
        )
        merged = _stitch_generatrix(
            cov_chains + [chain for run in bare_runs for chain in run],
            "PEC core (TYPE 2 + TYPE 4)", tol,
        )
        _preflight_generatrix(merged, "PEC core", axis_tol)
        groups = [iface_chains, cov_chains, bare_runs]
    elif kind == "layered":
        outer_flag, inner_flag = next(iter({
            (chain.pos_mat, chain.neg_mat)
            for chain in chains if chain.seg_type == 5
        }))
        mid5_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 5],
            "layer-interface (TYPE 5)", tol,
        )
        core_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 4],
            "core (TYPE 4)", tol,
        )
        patch_chains = _stitch_generatrix(
            [
                chain for chain in chains
                if chain.seg_type == 3 and chain.pos_mat == outer_flag
            ],
            "outer-interface (TYPE 3, outer layer)", tol,
        )
        bare_mid_runs = _stitch_pieces(
            [
                chain for chain in chains
                if chain.seg_type == 3 and chain.pos_mat == inner_flag
            ],
            "exposed-inner-interface (TYPE 3, inner layer)", tol,
        )
        _preflight_generatrix(core_chains, "core", axis_tol)
        merged_mid = _stitch_generatrix(
            mid5_chains
            + [chain for run in bare_mid_runs for chain in run],
            "inner-layer interface (TYPE 5 + TYPE 3)", tol,
        )
        _preflight_generatrix(
            merged_mid, "inner-layer interface", axis_tol
        )
        if not bare_mid_runs:
            _preflight_generatrix(
                patch_chains, "outer interface", axis_tol
            )
        groups = [
            patch_chains, mid5_chains, bare_mid_runs, core_chains,
            (outer_flag, inner_flag),
        ]
    elif kind == "layered_n":
        type3 = [chain for chain in chains if chain.seg_type == 3]
        top_flags = {chain.pos_mat for chain in type3}
        if len(top_flags) != 1:
            raise ValueError(
                "N-layer stacks (multiple TYPE 5 flag pairs) support full "
                "coverage only: all TYPE 3 chains must reference the "
                "outermost layer flag (patch layouts are limited to two "
                "layers)."
            )
        top_flag = next(iter(top_flags))
        core_flags = {
            chain.pos_mat for chain in chains if chain.seg_type == 4
        }
        if len(core_flags) != 1:
            raise ValueError(
                "The TYPE 4 core segments must share one pos_mat."
            )
        bottom_flag = next(iter(core_flags))
        pair_map = {}
        for outer, inner in {
            (chain.pos_mat, chain.neg_mat)
            for chain in chains if chain.seg_type == 5
        }:
            if outer in pair_map:
                raise ValueError(
                    f"Two TYPE 5 interfaces claim outer flag {outer}."
                )
            pair_map[outer] = inner
        flag_order = [top_flag]
        while flag_order[-1] != bottom_flag:
            next_flag = pair_map.pop(flag_order[-1], None)
            if next_flag is None:
                raise ValueError(
                    "Layer-flag chain broken: no TYPE 5 interface has "
                    f"pos_mat {flag_order[-1]} (walking outer flag "
                    f"{top_flag} toward core flag {bottom_flag})."
                )
            flag_order.append(next_flag)
        if pair_map:
            raise ValueError(
                f"TYPE 5 interfaces with flags {sorted(pair_map)} are not "
                "part of the outer-to-core layer chain."
            )
        interface_groups = [
            _stitch_generatrix(
                type3, "outer-interface (TYPE 3)", tol
            )
        ]
        for outer, inner in zip(flag_order[:-1], flag_order[1:]):
            type5 = [
                chain for chain in chains
                if chain.seg_type == 5
                and (chain.pos_mat, chain.neg_mat) == (outer, inner)
            ]
            interface_groups.append(_stitch_generatrix(
                type5,
                f"layer-interface (TYPE 5, {outer}|{inner})",
                tol,
            ))
        core_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 4],
            "core (TYPE 4)", tol,
        )
        for index, group in enumerate(interface_groups):
            _preflight_generatrix(group, f"interface {index}", axis_tol)
        _preflight_generatrix(core_chains, "core", axis_tol)
        groups = [interface_groups, core_chains, flag_order]
    elif kind == "banded":
        def runs_of(predicate, what):
            subset = [chain for chain in chains if predicate(chain)]
            return _stitch_pieces(subset, what, tol) if subset else []

        covered_runs = []
        for flag in sorted({
            chain.pos_mat for chain in chains if chain.seg_type == 4
        }):
            covered_runs += [(flag, run) for run in runs_of(
                lambda chain, material=flag: (
                    chain.seg_type == 4 and chain.pos_mat == material
                ),
                f"TYPE 4 band (mat {flag})",
            )]
        outer_runs = []
        for flag in sorted({
            chain.pos_mat for chain in chains if chain.seg_type == 3
        }):
            outer_runs += [(flag, run) for run in runs_of(
                lambda chain, material=flag: (
                    chain.seg_type == 3 and chain.pos_mat == material
                ),
                f"TYPE 3 band surface (mat {flag})",
            )]
        wall_runs = []
        for outer, inner in sorted({
            (chain.pos_mat, chain.neg_mat)
            for chain in chains if chain.seg_type == 5
        }):
            if outer == inner:
                raise ValueError(
                    "A TYPE 5 band wall needs two DIFFERENT material flags "
                    "(adjacent bands of the same material are one band)."
                )
            wall_runs += [((outer, inner), run) for run in runs_of(
                lambda chain, positive=outer, negative=inner: (
                    chain.seg_type == 5
                    and (chain.pos_mat, chain.neg_mat)
                    == (positive, negative)
                ),
                f"TYPE 5 band wall ({outer}|{inner})",
            )]
        bare_runs = runs_of(
            lambda chain: chain.seg_type == 2, "bare (TYPE 2)"
        )
        for run in bare_runs:
            for chain in run:
                if chain.ibc_flag > 0:
                    raise ValueError(
                        "Banded layouts: IBC on bare TYPE 2 pieces is not "
                        "supported yet (PEC only)."
                    )
        groups = [covered_runs, outer_runs, wall_runs, bare_runs]
    else:
        ordered = _stitch_generatrix(
            chains, "TYPE 2" if kind == "conductor" else "TYPE 3", tol
        )
        _preflight_generatrix(ordered, "body", axis_tol)
        groups = [ordered]

    return groups, tol, axis_tol


def _run_element_count(run: 'List[_SegChain]', wavelength: 'float') -> 'int':
    return sum(
        _element_count(chain.n_prop, float(length), wavelength)
        for chain in run for length in chain.prim_lengths
    )


def _bor_surface_layout(groups, kind: 'str', wavelength: 'float'):
    """Return ``[(element_count, is_conductor), ...]`` for a prepared job."""

    def record(run, conductor):
        return (_run_element_count(run, wavelength), bool(conductor))

    if kind == "conductor":
        return [record(groups[0], True)]
    if kind == "dielectric":
        return [record(groups[0], False)]
    if kind == "coated":
        return [record(groups[0], False), record(groups[1], True)]
    if kind == "partial":
        return [record(groups[0], False), record(groups[1], True)] + [
            record(run, True) for run in groups[2]
        ]
    if kind == "layered":
        return [record(groups[0], False), record(groups[1], False)] + [
            record(run, False) for run in groups[2]
        ] + [record(groups[3], True)]
    if kind == "layered_n":
        return [record(run, False) for run in groups[0]] + [
            record(groups[1], True)
        ]
    if kind == "banded":
        covered, outer, walls, bare = groups
        return (
            [record(run, True) for _flag, run in covered]
            + [record(run, False) for _flag, run in outer]
            + [record(run, False) for _flags, run in walls]
            + [record(run, True) for run in bare]
        )
    raise ValueError(f"Unsupported BoR resource layout kind {kind!r}.")


def _estimate_multisurface_operator_gb(surface_layout, mode_cap: 'int',
                                       gauss_order: 'int' = 4) -> 'float':
    """Conservative retained dense-table storage for nonconductor BoR.

    Each PMCHWT medium-side/self or cross operator retains one scalar modal
    table plus four rotated-principal-value tables.  Interface surfaces have
    two medium sides; conductor surfaces have one.  Counting every directed
    cross-surface pair is exact for a fully connected region and conservative
    for layered/banded topologies whose surfaces do not all share a region.
    """

    mode_cap = max(0, int(mode_cap))
    points = [
        float(gauss_order) * float(elements)
        for elements, _conductor in surface_layout
    ]
    complex_bytes = 16.0

    def operator_gb(index_a, index_b):
        pair_points = points[index_a] * points[index_b]
        scalar = pair_points * float(mode_cap + 2)
        rotated = 4.0 * pair_points * float(2 * mode_cap + 1)
        return complex_bytes * (scalar + rotated) / 1.0e9

    total = 0.0
    for index, (_elements, is_conductor) in enumerate(surface_layout):
        total += (1.0 if is_conductor else 2.0) * operator_gb(
            index, index
        )
    for index_a in range(len(surface_layout)):
        for index_b in range(len(surface_layout)):
            if index_a != index_b:
                total += operator_gb(index_a, index_b)
    return total


def _enforce_total_element_limit(surface_layout, max_elements: 'int') -> 'int':
    total = sum(int(elements) for elements, _conductor in surface_layout)
    limit = int(max_elements)
    if total > limit:
        raise ValueError(
            f"BoR mesh would need approximately {total} total elements "
            f"across {len(surface_layout)} surface(s) (> max {limit}). "
            "Reduce frequency or density, or raise max_elements after "
            "reviewing the resource estimate."
        )
    return total


def solve_monostatic_rcs_bor(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    mesh_reference_ghz: 'Optional[float]' = None,
    cfie_alpha: 'float' = 0.5,
    n_modes: 'Optional[int]' = None,
    mode_tol: 'float' = 1e-6,
    max_elements: 'int' = MAX_ELEMENTS_DEFAULT,
    workers: 'Optional[int]' = None,
    abort_event: 'Optional[threading.Event]' = None,
    table_precision: 'str' = "auto",
    assembly: 'str' = "auto",
    expand_to_360: 'bool' = False,
    stream_budget_gb: 'float' = BOR_STREAM_BUDGET_GB_DEFAULT,
) -> 'Dict[str, Any]':
    """
    Monostatic 3-D RCS (m^2 / dBsm) of an axisymmetric body described by a
    .geo geometry snapshot.  `elevations_deg` are ASPECT angles measured from
    the +z rotation axis (0 = nose-on, 90 = broadside, 180 = tail-on); the
    same argument name as the 2-D entry point is kept for a consistent UI.
    VV and HH are always co-solved and returned together; there is no channel
    selector in the production API.

    expand_to_360=True mirrors the samples about the axis to fill the full
    polar cut: sigma(360 - theta) = sigma(theta) -- EXACT for a body of
    revolution (rotating the problem 180 deg about z maps the body, the
    directions, and the polarization basis onto themselves), including the
    complex amplitudes.  The seam directions 0/360 and 180 are not
    duplicated.  Note this is a property of the axisymmetric MODEL: it does
    not conjure the effect of any non-axisymmetric feature the BoR cannot
    represent, and it is NOT the nose<->tail flip (theta -> 180 - theta),
    which is only valid for fore-aft symmetric bodies.
    """

    if not frequencies_ghz:
        raise ValueError("At least one frequency is required.")
    if not elevations_deg:
        raise ValueError("At least one aspect angle is required.")
    cfie_alpha = float(cfie_alpha)
    if not math.isfinite(cfie_alpha) or not (0.0 < cfie_alpha < 1.0):
        raise ValueError(
            "BoR CFIE alpha must be finite and satisfy 0 < alpha < 1. "
            "Use the explicit EFIE formulation API when pure EFIE is intended."
        )
    frequencies = [float(f) for f in frequencies_ghz]
    if any((not math.isfinite(f)) or f <= 0.0 for f in frequencies):
        raise ValueError("Frequencies must be positive finite GHz values.")
    mesh_ref_ghz = None
    if mesh_reference_ghz is not None:
        mesh_ref_ghz = float(mesh_reference_ghz)
        if (not math.isfinite(mesh_ref_ghz)) or mesh_ref_ghz <= 0.0:
            raise ValueError(
                "mesh_reference_ghz must be a positive finite GHz value."
            )
    aspects = [float(a) for a in elevations_deg]
    if any((not math.isfinite(a)) or a < 0.0 or a > 180.0 for a in aspects):
        raise ValueError("Aspect angles must lie in [0, 180] degrees from +z.")
    assembly_key = str(assembly).strip().lower()
    if assembly_key not in {"auto", "tables", "streaming"}:
        raise ValueError(
            "assembly must be 'auto', 'tables', or 'streaming'."
        )
    table_precision_key = str(table_precision).strip().lower()
    if table_precision_key not in {"auto", "single", "double"}:
        raise ValueError(
            "table_precision must be 'auto', 'single', or 'double'."
        )
    stream_budget = float(stream_budget_gb)
    if not math.isfinite(stream_budget) or stream_budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")
    scale = _unit_scale_to_meters(geometry_units)
    base_dir = _material_base_dir_for_snapshot(
        geometry_snapshot, material_base_dir
    )
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    _reject_unsupported_bor_ibc_interfaces(geometry_snapshot)
    preflight = validate_geometry_snapshot_for_solver(
        geometry_snapshot,
        base_dir=base_dir,
        meters_scale=scale,
    )

    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    # Mesh each solve frequency from its own material wavelength. A previous
    # sweep-wide minimum made a 1 GHz solve use the 18 GHz mesh whenever both
    # appeared in one GUI call. ``mesh_reference_ghz`` remains an explicit
    # user request to impose an additional common reference frequency.
    mesh_control_frequencies = set(frequencies)
    if mesh_ref_ghz is not None:
        mesh_control_frequencies.add(mesh_ref_ghz)
    mesh_controls = {}
    for solve_frequency in frequencies:
        control_frequencies = {float(solve_frequency)}
        if mesh_ref_ghz is not None:
            control_frequencies.add(mesh_ref_ghz)
        mesh_controls[float(solve_frequency)] = (
            _conservative_mesh_wavelength_for_frequencies(
                geometry_snapshot,
                materials,
                control_frequencies,
            )
        )
    mesh_wavelength_m = min(value[0] for value in mesh_controls.values())
    mesh_max_refractive_index = max(
        value[1] for value in mesh_controls.values()
    )
    mesh_material_flags = sorted({
        flag
        for value in mesh_controls.values()
        for flag in value[2]
    })

    chains = _chains_from_snapshot(geometry_snapshot, scale)
    kind = _classify(chains)
    _validate_bor_far_controls(
        kind, assembly_key, table_precision_key, stream_budget
    )
    groups, tol, axis_tol = _prepare_bor_groups(chains, kind)

    def check_abort():
        if abort_event is not None and abort_event.is_set():
            raise InterruptedError("Solve cancelled by user.")

    samples_by_pol: 'Dict[str, List[Dict[str, Any]]]' = {
        "VV": [],
        "HH": [],
    }
    per_freq_meta: 'List[Dict[str, Any]]' = []
    formulation_label = ""
    total_steps = len(frequencies)
    negative_rcs_count = 0
    power_amplitude_inconsistent_count = 0
    nonfinite_expected_power_count = 0

    for fi, freq_ghz in enumerate(frequencies):
        check_abort()
        freq_hz = freq_ghz * 1e9
        current_mesh = mesh_controls[float(freq_ghz)]
        lam0 = float(current_mesh[0])
        surface_layout = _bor_surface_layout(groups, kind, lam0)
        total_mesh_elements = _enforce_total_element_limit(
            surface_layout, max_elements
        )

        def report(modes_done, m_cap):
            if progress_callback is not None:
                try:
                    progress_callback(fi, total_steps,
                                     f"{freq_ghz:g} GHz: mode {modes_done}/{m_cap}")
                except Exception:
                    pass

        if kind == "conductor":
            ordered = groups[0]
            pts, elem_seg, elem_arc = _mesh_generatrix(ordered, lam0,
                                                       max_elements, axis_tol)
            zs_elem = np.zeros(len(pts) - 1, dtype=complex)
            for ei in range(len(zs_elem)):
                c = ordered[elem_seg[ei]]
                if c.ibc_flag > 0:
                    zs_elem[ei] = materials.get_impedance(
                        c.ibc_flag, freq_ghz, arc_s=float(elem_arc[ei]))
            has_ibc = bool(np.any(np.abs(zs_elem) > 0.0))
            form = "efie" if has_ibc else "cfie"
            out = solve_bor(pts, freq_hz, aspects, formulation=form,
                            cfie_alpha=cfie_alpha,
                            zs=zs_elem if has_ibc else None,
                            n_modes=n_modes, mode_tol=mode_tol,
                            workers=workers, progress=report,
                            check_abort=check_abort,
                            table_precision=table_precision_key,
                            assembly=assembly_key,
                            stream_budget_gb=stream_budget)
            actual_assembly = str(out.get("assembly", "")).strip().lower()
            actual_precision = str(
                out.get("table_precision", "")
            ).strip().lower()
            if (
                assembly_key != "auto"
                and actual_assembly != assembly_key
            ):
                raise RuntimeError(
                    "BoR conductor solver did not attest the requested "
                    f"assembly={assembly_key!r}; reported "
                    f"{actual_assembly or 'missing'!r}."
                )
            if (
                table_precision_key != "auto"
                and actual_precision != table_precision_key
            ):
                raise RuntimeError(
                    "BoR conductor solver did not attest the requested "
                    f"table_precision={table_precision_key!r}; reported "
                    f"{actual_precision or 'missing'!r}."
                )
            formulation_label = ("BoR-MoM IBC-EFIE (Leontovich)" if has_ibc
                                 else f"BoR-MoM PEC {form.upper()}")
        elif kind == "dielectric":
            eps, mu = materials.get_medium(groups[0][0].pos_mat, freq_ghz)
            pts, _, _ = _mesh_generatrix(
                groups[0], lam0, max_elements, axis_tol
            )
            out = solve_bor_dielectric(pts, freq_hz, aspects, eps, mu,
                                       n_modes=n_modes, mode_tol=mode_tol,
                                       workers=workers, progress=report,
                                       check_abort=check_abort)
            formulation_label = "BoR-MoM PMCHWT (homogeneous dielectric)"
        elif kind == "coated":
            eps, mu = materials.get_medium(groups[0][0].pos_mat, freq_ghz)
            pts_o, _, _ = _mesh_generatrix(groups[0], lam0, max_elements, axis_tol)
            pts_c, _, _ = _mesh_generatrix(groups[1], lam0, max_elements, axis_tol)
            out = solve_bor_coated_pec(pts_o, pts_c, freq_hz, aspects, eps, mu,
                                       n_modes=n_modes, mode_tol=mode_tol,
                                       workers=workers, progress=report,
                                       check_abort=check_abort)
            formulation_label = "BoR-MoM PMCHWT coated PEC (multi-region)"
        elif kind == "partial":
            eps, mu = materials.get_medium(groups[0][0].pos_mat, freq_ghz)
            pts_i, _, _ = _mesh_generatrix(groups[0], lam0, max_elements, axis_tol)
            pts_c, _, _ = _mesh_generatrix(groups[1], lam0, max_elements, axis_tol)
            bare_pts = []
            bare_zs = []
            any_ibc = False
            for run in groups[2]:
                pts_b, elem_seg, elem_arc = _mesh_generatrix(
                    run, lam0, max_elements, axis_tol)
                zs_elem = np.zeros(len(pts_b) - 1, dtype=complex)
                for ei in range(len(zs_elem)):
                    c = run[elem_seg[ei]]
                    if c.ibc_flag > 0:
                        zs_elem[ei] = materials.get_impedance(
                            c.ibc_flag, freq_ghz, arc_s=float(elem_arc[ei]))
                bare_pts.append(pts_b)
                has = bool(np.any(np.abs(zs_elem) > 0.0))
                any_ibc |= has
                bare_zs.append(zs_elem if has else None)
            out = solve_bor_partial_coating(pts_i, pts_c, bare_pts, freq_hz,
                                            aspects, eps, mu, bare_zs=bare_zs,
                                            n_modes=n_modes,
                                            mode_tol=mode_tol, workers=workers,
                                            progress=report,
                                            check_abort=check_abort)
            for w in out.get("warnings", []):
                materials.warn_once(w)
            formulation_label = ("BoR-MoM PMCHWT partial coating "
                                 f"({out['n_junctions']} junction(s)"
                                 f"{', IBC bare' if any_ibc else ''})")
        elif kind == "layered":
            outer_flag, inner_flag = groups[4]
            eps_o, mu_o = materials.get_medium(outer_flag, freq_ghz)
            eps_i, mu_i = materials.get_medium(inner_flag, freq_ghz)
            pts_p, _, _ = _mesh_generatrix(groups[0], lam0, max_elements, axis_tol)
            pts_m, _, _ = _mesh_generatrix(groups[1], lam0, max_elements, axis_tol)
            pts_c, _, _ = _mesh_generatrix(groups[3], lam0, max_elements, axis_tol)
            bare_pts = [_mesh_generatrix(run, lam0, max_elements, axis_tol)[0]
                        for run in groups[2]]
            if bare_pts:
                out = solve_bor_coating_patch(pts_p, pts_m, bare_pts, pts_c,
                                              freq_hz, aspects, eps_i, mu_i,
                                              eps_o, mu_o, n_modes=n_modes,
                                              mode_tol=mode_tol, workers=workers,
                                              progress=report,
                                              check_abort=check_abort)
                formulation_label = ("BoR-MoM PMCHWT coating patch "
                                     f"({out['n_junctions']} junction(s))")
            else:
                out = solve_bor_coated2_pec(pts_p, pts_m, pts_c, freq_hz,
                                            aspects, eps_i, mu_i, eps_o, mu_o,
                                            n_modes=n_modes, mode_tol=mode_tol,
                                            workers=workers, progress=report,
                                            check_abort=check_abort)
                formulation_label = "BoR-MoM PMCHWT two-layer coated PEC"
        elif kind == "layered_n":
            iface_groups, core_chains, flag_order = groups
            media = [materials.get_medium(fl, freq_ghz) for fl in flag_order]
            iface_pts = [_mesh_generatrix(g, lam0, max_elements, axis_tol)[0]
                         for g in iface_groups]
            pts_c, _, _ = _mesh_generatrix(core_chains, lam0, max_elements,
                                           axis_tol)
            # solver wants eps/mu INNERMOST first; flag_order is outer->inner
            eps_list = [media[i][0] for i in range(len(media) - 1, -1, -1)]
            mu_list = [media[i][1] for i in range(len(media) - 1, -1, -1)]
            out = solve_bor_coated_n_pec(iface_pts, pts_c, freq_hz, aspects,
                                         eps_list, mu_list, n_modes=n_modes,
                                         mode_tol=mode_tol, workers=workers,
                                         progress=report,
                                         check_abort=check_abort)
            formulation_label = (f"BoR-MoM PMCHWT {len(iface_pts)}-layer "
                                 "coated PEC")
        elif kind == "banded":
            cov_runs, out_runs, wall_runs, bare_band_runs = groups

            def run_ends(run):
                return (run[0].pts[0], run[-1].pts[-1])

            def key(p):
                return (int(round(p[0] / tol)), int(round(p[1] / tol)))

            def lam_for(flags):
                # The global conservative wavelength already includes every
                # referenced flag at every mesh-control frequency.
                del flags
                return lam0

            # mesh every piece; build the surface list (interfaces first,
            # then conductors -- order is arbitrary but stable)
            surfaces = []
            piece_ends = []
            piece_tag = []          # ("out", flag) | ("wall", (po, ne)) | ("cov", flag) | ("bare", None)
            for flag, run in out_runs:
                pts, _, _ = _mesh_generatrix(run, lam_for([flag]),
                                             max_elements, axis_tol)
                surfaces.append((pts, False))
                piece_ends.append(run_ends(run))
                piece_tag.append(("out", flag))
            for pair, run in wall_runs:
                pts, _, _ = _mesh_generatrix(run, lam_for(list(pair)),
                                             max_elements, axis_tol)
                surfaces.append((pts, False))
                piece_ends.append(run_ends(run))
                piece_tag.append(("wall", pair))
            for flag, run in cov_runs:
                pts, _, _ = _mesh_generatrix(run, lam_for([flag]),
                                             max_elements, axis_tol)
                surfaces.append((pts, True))
                piece_ends.append(run_ends(run))
                piece_tag.append(("cov", flag))
            for run in bare_band_runs:
                pts, _, _ = _mesh_generatrix(run, lam0, max_elements, axis_tol)
                surfaces.append((pts, True))
                piece_ends.append(run_ends(run))
                piece_tag.append(("bare", None))

            # air region: every TYPE 3 surface and every bare conductor
            regions = [{"medium": None, "exterior": True,
                        "bounds": [(i, +1) for i, tg in enumerate(piece_tag)
                                   if tg[0] in ("out", "bare")]}]
            # one region per covered run, discovered by endpoint connectivity
            for ci, tg in enumerate(piece_tag):
                if tg[0] != "cov":
                    continue
                flag = tg[1]
                cand = [i for i, t2 in enumerate(piece_tag)
                        if (t2[0] == "out" and t2[1] == flag)
                        or (t2[0] == "wall" and flag in t2[1])]
                comp = {ci}
                grew = True
                while grew:
                    grew = False
                    kset = {key(p) for i in comp for p in piece_ends[i]}
                    for i in cand:
                        if i not in comp and any(key(p) in kset
                                                 for p in piece_ends[i]):
                            comp.add(i)
                            grew = True
                bounds = []
                for i in sorted(comp):
                    t2 = piece_tag[i]
                    if t2[0] == "cov":
                        bounds.append((i, +1))
                    elif t2[0] == "out":
                        bounds.append((i, -1))
                    else:                     # wall: +1 on its pos_mat side
                        bounds.append((i, +1 if t2[1][0] == flag else -1))
                if len(bounds) < 2:
                    raise ValueError(
                        f"Band (mat {flag}) covered piece has no attached "
                        "TYPE 3/5 boundary -- check junction coordinates "
                        "coincide exactly.")
                eps_b, mu_b = materials.get_medium(flag, freq_ghz)
                regions.append({"medium": (eps_b, mu_b), "bounds": bounds})

            sys_ = _MultiRegionBor(surfaces=surfaces, regions=regions,
                                   freq_hz=freq_hz)
            out = _solve_multiregion(sys_, freq_hz, aspects, n_modes, mode_tol,
                                     workers, report, check_abort,
                                     "pmchwt-banded", {})
            formulation_label = (f"BoR-MoM PMCHWT banded coatings "
                                 f"({len(regions) - 1} band(s), "
                                 f"{out['n_junctions']} junction(s))")

        for w in out.get("warnings", []) or []:
            materials.warn_once(str(w))
        # Missing residual telemetry is not evidence of a good solve.  Preserve
        # it as non-finite so the release gate below rejects the result.
        residual = float(out.get("linear_residual", math.nan))
        backward_error = float(
            out.get("linear_backward_error", math.nan)
        )
        for channel, sigma_key, amp_key in (
            ("VV", "sigma_vv", "amp_vv"),
            ("HH", "sigma_hh", "amp_hh"),
        ):
            sig = out[sigma_key]
            amp = out[amp_key]
            for ai, aspect in enumerate(aspects):
                raw_lin = float(sig[ai])
                a_val = complex(amp[ai])
                if math.isfinite(raw_lin) and raw_lin < 0.0:
                    negative_rcs_count += 1
                amp_abs2 = (
                    a_val.real * a_val.real + a_val.imag * a_val.imag
                )
                expected_lin = 4.0 * math.pi * amp_abs2
                if not math.isfinite(expected_lin):
                    nonfinite_expected_power_count += 1
                if (
                    math.isfinite(raw_lin)
                    and math.isfinite(expected_lin)
                    and abs(raw_lin - expected_lin)
                    > (
                        BOR_POWER_CONSISTENCY_RTOL
                        * max(raw_lin, expected_lin)
                        + np.finfo(float).tiny
                    )
                ):
                    power_amplitude_inconsistent_count += 1
                lin = max(raw_lin, 0.0)
                samples_by_pol[channel].append({
                    "frequency_ghz": float(freq_ghz),
                    "theta_inc_deg": float(aspect),
                    "theta_scat_deg": float(aspect),
                    "rcs_linear": lin,
                    # Preserve exact/deep physical nulls in linear units.  A
                    # numerical floor belongs only in logarithmic presentation.
                    "rcs_db": 10.0 * math.log10(max(lin, 1e-300)),
                    "rcs_amp_real": float(a_val.real),
                    "rcs_amp_imag": float(a_val.imag),
                    "rcs_amp_phase_deg": float(math.degrees(cmath.phase(a_val))),
                    "linear_residual": residual,
                    "linear_backward_error": backward_error,
                })
        per_freq_meta.append({
            "frequency_ghz": float(freq_ghz),
            "modes_used": int(out["modes_used"]),
            "mode_cap": int(out.get("mode_cap", out["modes_used"])),
            "mode_tail_start": int(out.get("mode_tail_start", 0)),
            "mode_converged": bool(out.get("mode_converged", False)),
            "mode_quiet_count": int(out.get("mode_quiet_count", 0)),
            "mode_last_relative_increment": float(
                out.get("mode_last_relative_increment", math.inf)
            ),
            "n_unknowns": int(out["n_unknowns"]),
            "linear_residual": residual,
            "linear_backward_error": backward_error,
            "linear_refinement_steps": int(
                out.get("linear_refinement_steps", 0)
            ),
            "max_cond": float(out["max_cond"]) if "max_cond" in out else None,
            "condition_est_computed": bool(
                out.get("condition_est_computed", "max_cond" in out)
            ),
            "condition_est_method": out.get("condition_est_method"),
            "condition_est_limit": float(
                out.get("condition_est_limit", BOR_CONDITION_EST_MAX)
            ),
            "assembly": (
                str(out.get("assembly", "") or "") or None
            ),
            "table_precision": (
                str(out.get("table_precision", "") or "") or None
            ),
            "stream_mode_block": out.get("stream_mode_block"),
            "stream_sweeps": out.get("stream_sweeps"),
            "mesh_elements_total": int(total_mesh_elements),
            "mesh_surface_count": int(len(surface_layout)),
            "mesh_wavelength_m": float(current_mesh[0]),
            "mesh_max_refractive_index": float(current_mesh[1]),
            "mesh_material_flags": list(current_mesh[2]),
        })
        if progress_callback is not None:
            try:
                progress_callback(fi + 1, total_steps, f"Solved {freq_ghz:g} GHz")
            except Exception:
                pass

    if expand_to_360:
        for channel in ("VV", "HH"):
            mirrored = []
            for sample in samples_by_pol[channel]:
                th = float(sample["theta_inc_deg"])
                if 0.0 < th < 180.0:        # 0/360 and 180 are seam directions
                    duplicate = dict(sample)
                    duplicate["theta_inc_deg"] = duplicate["theta_scat_deg"] = (
                        360.0 - th
                    )
                    mirrored.append(duplicate)
            samples_by_pol[channel] = sorted(
                samples_by_pol[channel] + mirrored,
                key=lambda row: (
                    row["frequency_ghz"],
                    row["theta_inc_deg"],
                ),
            )

    samples = []
    for channel in ("VV", "HH"):
        for source in samples_by_pol[channel]:
            row = dict(source)
            row["polarization"] = channel
            samples.append(row)
    samples.sort(key=lambda row: (
        row["frequency_ghz"], row["polarization"], row["theta_inc_deg"]
    ))
    all_physical_samples = samples

    residual_values = np.asarray(
        [row.get("linear_residual", math.nan) for row in per_freq_meta],
        dtype=float,
    )
    finite_residuals = residual_values[np.isfinite(residual_values)]
    residual_nonfinite_count = int(
        residual_values.size - finite_residuals.size
    )
    residual_max = (
        float(np.max(finite_residuals))
        if finite_residuals.size
        else math.nan
    )
    backward_error_values = np.asarray(
        [
            row.get("linear_backward_error", math.nan)
            for row in per_freq_meta
        ],
        dtype=float,
    )
    finite_backward_errors = backward_error_values[
        np.isfinite(backward_error_values)
    ]
    backward_error_nonfinite_count = int(
        backward_error_values.size - finite_backward_errors.size
    )
    backward_error_max = (
        float(np.max(finite_backward_errors))
        if finite_backward_errors.size
        else math.nan
    )
    condition_values = np.asarray(
        [
            row.get("max_cond", math.nan)
            if bool(row.get("condition_est_computed", False))
            else math.nan
            for row in per_freq_meta
        ],
        dtype=float,
    )
    finite_conditions = condition_values[np.isfinite(condition_values)]
    condition_missing_count = int(
        condition_values.size - finite_conditions.size
    )
    condition_max = (
        float(np.max(finite_conditions))
        if finite_conditions.size
        else math.nan
    )
    unconverged_modes = int(sum(
        not bool(row.get("mode_converged", False))
        for row in per_freq_meta
    ))
    nonfinite_samples = int(sum(
        not all(math.isfinite(float(sample[key])) for key in (
            "rcs_linear",
            "rcs_amp_real",
            "rcs_amp_imag",
        ))
        for sample in all_physical_samples
    ))
    quality_violations: 'List[str]' = []
    if residual_nonfinite_count:
        quality_violations.append(
            f"residual_nonfinite_count={residual_nonfinite_count} must be zero"
        )
    if backward_error_nonfinite_count:
        quality_violations.append(
            "backward_error_nonfinite_count="
            f"{backward_error_nonfinite_count} must be zero"
        )
    if (
        not math.isfinite(backward_error_max)
        or backward_error_max > BOR_LINEAR_BACKWARD_ERROR_MAX
    ):
        quality_violations.append(
            f"backward_error_max={backward_error_max:.6g} exceeds limit "
            f"{BOR_LINEAR_BACKWARD_ERROR_MAX:.6g}"
        )
    if condition_missing_count:
        quality_violations.append(
            "condition_missing_or_nonfinite_count="
            f"{condition_missing_count} must be zero"
        )
    if (
        not math.isfinite(condition_max)
        or condition_max > BOR_CONDITION_EST_MAX
    ):
        quality_violations.append(
            f"condition_est_max={condition_max:.6g} exceeds limit "
            f"{BOR_CONDITION_EST_MAX:.6g}"
        )
    if unconverged_modes:
        quality_violations.append(
            f"mode_unconverged_count={unconverged_modes} must be zero"
        )
    if nonfinite_samples:
        quality_violations.append(
            f"nonfinite_sample_count={nonfinite_samples} must be zero"
        )
    if negative_rcs_count:
        quality_violations.append(
            f"negative_rcs_count={negative_rcs_count} must be zero"
        )
    if power_amplitude_inconsistent_count:
        quality_violations.append(
            "power_amplitude_inconsistent_count="
            f"{power_amplitude_inconsistent_count} must be zero"
        )
    if nonfinite_expected_power_count:
        quality_violations.append(
            "nonfinite_expected_power_count="
            f"{nonfinite_expected_power_count} must be zero"
        )
    quality_gate = {
        "passed": not quality_violations,
        "thresholds": {
            "residual_norm_refinement_advisory": BOR_LINEAR_RESIDUAL_MAX,
            "backward_error_max": BOR_LINEAR_BACKWARD_ERROR_MAX,
            "condition_est_max": BOR_CONDITION_EST_MAX,
            "mode_unconverged_count": 0,
            "nonfinite_sample_count": 0,
            "negative_rcs_count": 0,
            "nonfinite_expected_power_count": 0,
            "power_amplitude_consistency_rtol": (
                BOR_POWER_CONSISTENCY_RTOL
            ),
        },
        "values": {
            "residual_norm_max": residual_max,
            "residual_nonfinite_count": residual_nonfinite_count,
            "backward_error_max": backward_error_max,
            "backward_error_nonfinite_count": (
                backward_error_nonfinite_count
            ),
            "condition_est_max": condition_max,
            "condition_missing_or_nonfinite_count":
                condition_missing_count,
            "mode_unconverged_count": unconverged_modes,
            "nonfinite_sample_count": nonfinite_samples,
            "negative_rcs_count": negative_rcs_count,
            "power_amplitude_inconsistent_count": (
                power_amplitude_inconsistent_count
            ),
            "nonfinite_expected_power_count": (
                nonfinite_expected_power_count
            ),
        },
        "violations": quality_violations,
        "reason": (
            "; ".join(quality_violations)
            if quality_violations
            else "BoR linear backward-error, conditioning, modal, and "
                 "field-consistency thresholds satisfied"
        ),
    }
    if quality_violations:
        raise RuntimeError(
            "BoR quality gate failed: " + quality_gate["reason"]
        )

    return {
        "solver": "bor_mom_rcs",
        "scattering_mode": "monostatic",
        "polarizations": ["VV", "HH"],
        "polarization_mapping": {"VV": "VV", "HH": "HH"},
        "rcs_log_unit": "dBsm",
        "rcs_linear_quantity": "sigma_3d",
        "samples": samples,
        # The low-level BoR solve always computes both polarizations with the
        # same modal matrix factorization.  Preserve both so body workflows do
        # not repeat the complete assembly and solve merely to select the
        # other output channel.
        "co_solved_samples": samples_by_pol,
        "metadata": {
            "formulation": formulation_label,
            "geometry_kind": kind,
            "frequency_count": int(len(frequencies)),
            "aspect_count": int(len(aspects)),
            "elevation_count": int(len(aspects)),
            "output_aspect_count": int(len({
                float(sample["theta_inc_deg"]) for sample in samples
            })),
            "expanded_to_360": bool(expand_to_360),
            "per_frequency": per_freq_meta,
            "residual_norm_max": residual_max,
            "residual_nonfinite_count": residual_nonfinite_count,
            "residual_norm_refinement_advisory": (
                BOR_LINEAR_RESIDUAL_MAX
            ),
            "backward_error_max": backward_error_max,
            "backward_error_nonfinite_count": (
                backward_error_nonfinite_count
            ),
            "condition_est_max": condition_max,
            "condition_missing_or_nonfinite_count":
                condition_missing_count,
            "condition_est_computed": bool(
                condition_missing_count == 0
                and len(per_freq_meta) > 0
            ),
            "condition_est_method": "lapack_gecon_1norm",
            "quality_gate": quality_gate,
            "cfie_alpha": float(cfie_alpha),
            "workers": int(workers),
            "far_table_controls_applicable": bool(kind == "conductor"),
            "assembly_requested": assembly_key,
            "table_precision_requested": table_precision_key,
            "stream_budget_gb": stream_budget,
            "mesh_reference_ghz": mesh_ref_ghz,
            "mesh_control_frequencies_ghz": sorted(mesh_control_frequencies),
            "mesh_wavelength_m": float(mesh_wavelength_m),
            "mesh_max_refractive_index": float(mesh_max_refractive_index),
            "mesh_material_flags": list(mesh_material_flags),
            "warnings": list(materials.warnings),
            "preflight": preflight,
        },
    }


def _bor_channel_result(
    result: 'Dict[str, Any]',
    polarization: 'str',
) -> 'Dict[str, Any]':
    """Expose one co-solved BoR polarization to the shared mesh comparator."""

    channels = result.get("co_solved_samples", {}) or {}
    samples = list(channels.get(polarization, []) or [])
    if not samples:
        raise ValueError(
            "Certified BoR solve is missing co-solved "
            f"{polarization} samples."
        )
    return {"samples": samples}


def solve_monostatic_rcs_bor_certified(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    mesh_reference_ghz: 'Optional[float]' = None,
    cfie_alpha: 'float' = 0.5,
    n_modes: 'Optional[int]' = None,
    mode_tol: 'float' = 1e-6,
    max_elements: 'int' = MAX_ELEMENTS_DEFAULT,
    workers: 'Optional[int]' = None,
    abort_event: 'Optional[threading.Event]' = None,
    table_precision: 'str' = "auto",
    assembly: 'str' = "auto",
    expand_to_360: 'bool' = False,
    stream_budget_gb: 'float' = BOR_STREAM_BUDGET_GB_DEFAULT,
    mesh_convergence_policy: 'Optional[Dict[str, Any]]' = None,
) -> 'Dict[str, Any]':
    """Production BoR entry point.

    Solve the requested grid on the user mesh and one internally refined
    mesh.  Both co-solved VV/HH complex fields must pass the fixed production
    convergence policy.  Only the refined result is returned.
    """

    policy = validate_mesh_convergence_policy(mesh_convergence_policy)

    common = dict(
        frequencies_ghz=frequencies_ghz,
        elevations_deg=elevations_deg,
        geometry_units=geometry_units,
        material_base_dir=material_base_dir,
        mesh_reference_ghz=mesh_reference_ghz,
        cfie_alpha=cfie_alpha,
        n_modes=n_modes,
        mode_tol=mode_tol,
        max_elements=max_elements,
        workers=workers,
        abort_event=abort_event,
        table_precision=table_precision,
        assembly=assembly,
        expand_to_360=expand_to_360,
        stream_budget_gb=stream_budget_gb,
    )

    def phase_progress(phase):
        if progress_callback is None:
            return None

        def report(done, total, message):
            total_value = max(int(total), 1)
            done_value = max(0, min(int(done), total_value))
            progress_callback(
                phase * total_value + done_value,
                2 * total_value,
                ("Base mesh: " if phase == 0 else "Certified mesh: ")
                + str(message),
            )

        return report

    base_result = solve_monostatic_rcs_bor(
        geometry_snapshot=geometry_snapshot,
        progress_callback=phase_progress(0),
        **common
    )
    fine_snapshot = scale_snapshot_panel_density(
        geometry_snapshot, policy["fine_factor"]
    )
    fine_result = solve_monostatic_rcs_bor(
        geometry_snapshot=fine_snapshot,
        progress_callback=phase_progress(1),
        **common
    )

    per_polarization = {}
    violations = []
    for channel in ("VV", "HH"):
        gate = evaluate_mesh_convergence(
            _bor_channel_result(base_result, channel),
            _bor_channel_result(fine_result, channel),
            rms_limit_db=policy["rms_limit_db"],
            max_abs_limit_db=policy["max_abs_limit_db"],
            complex_rms_limit=policy["complex_rms_limit"],
            complex_max_limit=policy["complex_max_limit"],
            phase_rms_limit_deg=policy["phase_rms_limit_deg"],
            phase_max_limit_deg=policy["phase_max_limit_deg"],
            phase_floor_relative=policy["phase_floor_relative"],
        )
        per_polarization[channel] = gate
        if not bool(gate.get("passed", False)):
            violations.append(
                channel + ": "
                + str(gate.get("reason", "mesh convergence failed"))
            )

    mesh_gate = {
        "schema": "ghost.solver.mesh-convergence.v1",
        "passed": not violations,
        "fine_factor": policy["fine_factor"],
        "published_mesh": "fine",
        "co_solved_polarizations": ["VV", "HH"],
        "policy": policy,
        "polarizations": per_polarization,
        "violations": violations,
        "reason": (
            "; ".join(violations)
            if violations
            else "BoR VV/HH complex-field mesh convergence passed"
        ),
    }
    if violations:
        raise RuntimeError(
            "Certified BoR mesh convergence failed: "
            + mesh_gate["reason"]
        )

    metadata = dict(fine_result.get("metadata", {}) or {})
    metadata["mesh_convergence"] = mesh_gate
    metadata["mesh_convergence_certified"] = True
    metadata["certified_entry_point"] = True
    mesh_gate["base_quality_gate"] = dict(
        (base_result.get("metadata", {}) or {}).get("quality_gate", {}) or {}
    )
    mesh_gate["fine_quality_gate"] = dict(
        metadata.get("quality_gate", {}) or {}
    )
    quality_gate = dict(metadata.get("quality_gate", {}) or {})
    quality_gate["mesh_convergence_certified"] = True
    quality_gate["certification_scope"] = (
        "discrete_linear_system_modal_truncation_and_mesh_convergence"
    )
    metadata["quality_gate"] = quality_gate
    fine_result["metadata"] = metadata
    return fine_result


def solve_monostatic_rcs_bor_survey(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    **kwargs: 'Any',
) -> 'Dict[str, Any]':
    """Single-mesh BoR solve with explicit uncertified-result metadata."""

    result = solve_monostatic_rcs_bor(
        geometry_snapshot=geometry_snapshot,
        frequencies_ghz=frequencies_ghz,
        elevations_deg=elevations_deg,
        **kwargs,
    )
    metadata = result.setdefault("metadata", {})
    metadata["mesh_convergence_certified"] = False
    metadata["certified_entry_point"] = False
    metadata["published_mesh"] = "base"
    metadata["survey_mode"] = True
    warning = (
        "SURVEY MODE: solved on the base BoR mesh only. No mesh-convergence "
        "certificate exists for this field."
    )
    warnings = metadata.setdefault("warnings", [])
    if warning not in warnings:
        warnings.append(warning)
    quality_gate = metadata.get("quality_gate")
    if isinstance(quality_gate, dict):
        quality_gate["mesh_convergence_certified"] = False
        quality_gate["certification_scope"] = (
            "discrete_linear_system_and_modal_truncation_only"
        )
    return result
