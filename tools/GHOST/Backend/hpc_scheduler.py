#!/usr/bin/env python3
"""
Shared scheduling core for the HPC sweep drivers.

The 2-D driver expands a sweep into dual-channel (geometry x frequency) units;
the BoR driver retains legacy per-channel output units while co-solving their
paired physics. Both hand their work to a pile of nodes. Getting that right is
almost entirely a scheduling problem, and the three things that decide how many
node-hours a sweep burns are:

1. LOAD BALANCE.  Unit cost is dominated by the dense boundary-integral
   assembly, which grows like N^2 in the number of boundary nodes, and N grows
   linearly with frequency.  Across a 2-18 GHz sweep the most expensive unit is
   roughly 80x the cheapest, so handing out units round-robin over an index --
   the obvious scheme, and the one this replaces -- routinely leaves one slot
   holding every high-frequency unit while the rest idle.  Units are instead
   costed at submit time and dealt out longest-processing-time-first, then
   rebalanced at run time by stealing (below).

2. MEMORY.  A node with 64-96 cores and 375-750 GB cannot run one solve per
   core if each solve peaks at several GB: the cgroup OOM killer takes out a
   pool worker, and an OOM kill (unlike a Python MemoryError) can wedge the
   pool.  Workers are therefore admitted against a memory budget using the
   solver's own peak estimate for the unit, not just a core count.

3. PER-UNIT OVERHEAD.  Provenance is re-verified around every unit.  Hashing
   the whole backend tree per unit costs more than a small unit's solve once
   you multiply it by the units, the workers, and a shared filesystem.  The
   fingerprint cache below keeps the checks while making the repeat ones cheap.

Everything here is filesystem-coordinated and stateless between processes: any
number of array tasks may join or leave a run at any time, and a task that dies
loses at most the units it had in flight.
"""

import errno
import json
import math
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Polarization labels
# -----------------------------------------------------------------------------

# The radar spellings of the same two physical channels. These 2-D geometries
# are elevation cuts, so the out-of-plane z axis is HORIZONTAL: E along z (TM)
# is HH, and TE's in-plane E carries the vertical component, so TE is VV. This
# mirrors rcs_solver._normalize_polarization without importing the numerical
# solver during resource planning.
_POLARIZATION_ALIASES = {
    "TM": "TM", "HH": "TM", "H": "TM", "HORIZONTAL": "TM",
    "TE": "TE", "VV": "TE", "V": "TE", "VERTICAL": "TE",
}


def canonical_polarization(label: 'str') -> 'str':
    """Canonical TM/TE channel for a user-facing polarization label."""

    text = str(label or "").strip().upper()
    try:
        return _POLARIZATION_ALIASES[text]
    except KeyError:
        raise ValueError(
            f"Unsupported polarization {label!r}. Use TM/TE or the radar "
            "aliases VV/HH (VV = TE, HH = TM)."
        ) from None


def distinct_polarization_channels(
    labels: 'Sequence[str]',
    canonical: 'Optional[Callable[[str], str]]' = None,
) -> 'List[str]':
    """Validate 2-D resource-planning labels and return one per channel.

    Rejects both an unknown label and two spellings of the *same* channel. That
    second case is the one worth catching: ``["VV", "TE"]`` looks like two
    polarizations and is one, so without this a cost planner could reserve the
    same physical channel twice.

    Labels are returned as written by default so caller-owned record keys stay
    stable. Pass ``canonical`` to return a fixed spelling instead.
    """

    resolved = []  # type: List[str]
    seen = {}  # type: Dict[str, str]
    for label in labels:
        channel = canonical_polarization(label)
        if channel in seen:
            raise ValueError(
                f"{label!r} and {seen[channel]!r} are the same physical "
                f"channel ({channel}); list each channel once."
            )
        seen[channel] = str(label)
        resolved.append(str(label) if canonical is None else canonical(label))
    if not resolved:
        raise ValueError("no polarizations given.")
    return resolved


# -----------------------------------------------------------------------------
# Node resources
# -----------------------------------------------------------------------------

_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def pin_blas_threads(count: 'int') -> 'None':
    """Pin every BLAS/OpenMP backend to ``count`` threads.

    Must run before numpy/scipy import to take effect on all backends, so the
    drivers call it at module import and again in each pool worker.
    """

    value = str(max(1, int(count)))
    for name in _BLAS_THREAD_VARS:
        os.environ[name] = value


def detect_cores() -> 'int':
    """Cores actually allocated to this process (SLURM first, then affinity)."""

    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.environ.get(name, "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def detect_memory_gb() -> 'float':
    """Memory this task may use, in GB.

    SLURM's allocation is authoritative when present: a node with 750 GB
    installed may still have been given a 64 GB cgroup, and /proc/meminfo
    reports the machine, not the cgroup.
    """

    raw = os.environ.get("SLURM_MEM_PER_NODE", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return float(int(raw)) / 1024.0
    raw = os.environ.get("SLURM_MEM_PER_CPU", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return float(int(raw)) * float(detect_cores()) / 1024.0
    for path, scale in (
        ("/sys/fs/cgroup/memory.max", 1.0),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", 1.0),
    ):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit():
            limit = float(text) * scale / (1024.0 ** 3)
            # Unlimited cgroups report a huge sentinel; ignore those.
            if 0.5 < limit < 1.0e6:
                return limit
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / (1024.0 ** 2)
    except (OSError, ValueError, IndexError):
        pass
    return 8.0


# -----------------------------------------------------------------------------
# Provenance fingerprint cache
# -----------------------------------------------------------------------------

class FingerprintCache:
    """Memoized file hashing for the per-unit provenance checks.

    The drivers verify the solver source and the frozen geometry inputs before
    and after every unit, which is the right check to make -- a run whose
    source changed underneath it must not publish fields.  Recomputing it from
    scratch each time means re-reading several MB of backend sources per unit
    per worker, which on a shared filesystem costs more than the check is
    worth once a sweep has thousands of units.

    Repeat hashes are served from an (inode identity, size, mtime) key, and the
    whole cache is dropped every ``full_recheck_seconds`` so a full re-read
    still happens regularly inside a long-lived worker.  A fresh worker process
    always starts with an empty cache.
    """

    def __init__(self, full_recheck_seconds: 'float' = 300.0) -> 'None':
        self._entries: 'Dict[str, Tuple[Tuple[int, int, int, int], str]]' = {}
        self._full_recheck_seconds = float(full_recheck_seconds)
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()

    def sha256_file(self, path: 'str') -> 'str':
        import hashlib

        abs_path = os.path.abspath(path)
        stat = os.stat(abs_path)
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        now = time.monotonic()
        with self._lock:
            if now - self._last_flush >= self._full_recheck_seconds:
                self._entries.clear()
                self._last_flush = now
            cached = self._entries.get(abs_path)
            if cached is not None and cached[0] == key:
                return cached[1]
        digest = hashlib.sha256()
        with open(abs_path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        with self._lock:
            self._entries[abs_path] = (key, value)
        return value


_FINGERPRINT_CACHE: 'Optional[FingerprintCache]' = None


def install_fingerprint_cache(full_recheck_seconds: 'float' = 300.0) -> 'None':
    """Route ``workflow_provenance.sha256_file`` through the cache.

    Called by the drivers in the worker process.  Submission stays uncached so
    the manifest is always written from freshly read bytes.
    """

    global _FINGERPRINT_CACHE
    import workflow_provenance

    if _FINGERPRINT_CACHE is None:
        _FINGERPRINT_CACHE = FingerprintCache(full_recheck_seconds)
    workflow_provenance.sha256_file = _FINGERPRINT_CACHE.sha256_file


# -----------------------------------------------------------------------------
# Cost and memory model
# -----------------------------------------------------------------------------

def _same_interface_topology(
    rcs_solver: 'Any',
    panels: 'Sequence[Any]',
    left: 'Sequence[Any]',
    right: 'Sequence[Any]',
) -> 'bool':
    """Whether two polarization previews produce the same nodal topology.

    Today the signature depends only on geometry/interface flags, not material
    values or polarization. Keeping the comparison explicit makes the reuse
    fail-safe if a future formulation introduces a polarization-dependent
    interface: that channel simply receives its own mesh.
    """

    if len(left) != len(right) or len(left) != len(panels):
        return False
    return all(
        rcs_solver._linear_panel_signature_from_info(panel, l_info)
        == rcs_solver._linear_panel_signature_from_info(panel, r_info)
        for panel, l_info, r_info in zip(panels, left, right)
    )


def _resource_records_for_frequency(
    rcs_solver: 'Any',
    snapshot: 'Dict[str, Any]',
    materials: 'Any',
    frequency_ghz: 'float',
    polarizations: 'Sequence[Tuple[str, str]]',
    unit_scale: 'float',
    max_panels: 'int',
) -> 'Dict[str, Dict[str, Any]]':
    """Build one exact mesh per distinct interface topology at a frequency."""

    freq_ghz = float(frequency_ghz)
    k0 = 2.0 * math.pi * freq_ghz * 1.0e9 / rcs_solver.C0
    lambda_min, _, _ = rcs_solver._mesh_wavelength_for_snapshot(
        snapshot, materials, freq_ghz
    )
    panels = rcs_solver._build_panels(
        snapshot, unit_scale, lambda_min, max_panels=int(max_panels)
    )

    # Each group holds a representative panel-info list and the exact mesh
    # built from it. VV/HH currently share one group; the fallback remains
    # exact if that ever stops being true.
    topology_groups = []  # type: List[Tuple[List[Any], Any]]
    records = {}  # type: Dict[str, Dict[str, Any]]
    for requested_pol, canonical_pol in polarizations:
        preview = rcs_solver._build_coupled_panel_info(
            panels, materials, freq_ghz, canonical_pol, k0
        )
        mesh = None
        for representative, candidate_mesh in topology_groups:
            if _same_interface_topology(
                rcs_solver, panels, representative, preview
            ):
                mesh = candidate_mesh
                break
        if mesh is None:
            mesh, _stats = rcs_solver._build_linear_mesh_interface_aware(
                panels, preview
            )
            topology_groups.append((preview, mesh))

        # ``preview`` already came from the production coefficient builder.
        # Interface-aware meshing preserves the panels one-for-one and in the
        # same order, changing only their endpoint node IDs.  Rebuilding the
        # material records from those copied elements would therefore produce
        # identical records while repeating every material-table lookup and
        # passivity check on large meshes.
        infos = preview
        rcs_solver._assert_no_type1_sheet_for_mixed(infos)
        rcs_solver._assert_air_exterior(infos)
        rcs_solver._assert_supported_te_type2_contours(
            mesh, infos, canonical_pol
        )
        resources = rcs_solver._dense_formulation_resources(
            mesh, infos, canonical_pol
        )
        records[requested_pol] = {
            "panels": int(len(panels)),
            **resources,
        }
    return records


def predict_2d_resources_many(
    geometry_path: 'str',
    frequencies_ghz: 'Sequence[float]',
    polarizations: 'Sequence[str]',
    geometry_units: 'str',
    max_panels: 'int',
    fine_factor: 'float' = 1.0,
    n_angles: 'int' = 1,
    safety: 'float' = 1.35,
    floor_gb: 'float' = 0.6,
    progress: 'Optional[Callable[[float, str], None]]' = None,
) -> 'Dict[Tuple[float, str], Dict[str, Any]]':
    """Plan a geometry sweep using the exact solver mesh and formulation.

    Geometry validation, material-table loading, and certification refinement
    are performed once per geometry. Panels are built once per frequency and
    the interface-aware mesh is shared across polarizations only after their
    complete topology signatures compare equal. Material coefficients,
    formulation checks, DOF counts, and memory estimates remain specific to
    every frequency/polarization unit. Errors still propagate so an
    unsupported unit cannot enter a sweep with a zero-GB reservation.
    """

    import rcs_solver
    from geometry_io import parse_geometry, build_geometry_snapshot
    from solver_quality import scale_snapshot_panel_density

    frequencies = [float(value) for value in frequencies_ghz]
    if not frequencies:
        raise ValueError("2-D resource planning requires at least one frequency.")
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
    ):
        raise ValueError("Planning frequencies must be finite, positive, and unique.")
    requested_pols = distinct_polarization_channels(
        [str(value) for value in polarizations]
    )
    normalized_pols = [
        (label, rcs_solver._normalize_polarization(label))
        for label in requested_pols
    ]

    path = Path(geometry_path)
    title, segments, ibcs, dielectrics = parse_geometry(path.read_text())
    base_snapshot = build_geometry_snapshot(
        title, segments, ibcs, dielectrics
    )
    base_dir = str(path.parent)
    unit_scale = rcs_solver._unit_scale_to_meters(geometry_units)
    materials = rcs_solver.MaterialLibrary.from_entries(
        base_snapshot.get("ibcs", []) or [],
        base_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    rcs_solver.validate_geometry_snapshot_for_solver(
        base_snapshot,
        base_dir=base_dir,
        meters_scale=unit_scale,
        material_library=materials,
    )

    if float(fine_factor) <= 1.0:
        fine_snapshot = base_snapshot
    else:
        fine_snapshot = scale_snapshot_panel_density(
            base_snapshot, float(fine_factor)
        )
        base_segment_n = []
        for segment in list(base_snapshot.get("segments", []) or []):
            props = list(segment.get("properties", []) or [])
            base_segment_n.append(props[1] if len(props) > 1 else 0)
        fine_snapshot["_2d_certification_refinement_factor"] = float(
            fine_factor
        )
        fine_snapshot["_2d_certification_base_segment_n"] = base_segment_n
        rcs_solver.validate_geometry_snapshot_for_solver(
            fine_snapshot,
            base_dir=base_dir,
            meters_scale=unit_scale,
            material_library=materials,
        )

    planned = {}  # type: Dict[Tuple[float, str], Dict[str, Any]]
    for freq_ghz in frequencies:
        base_records = _resource_records_for_frequency(
            rcs_solver,
            base_snapshot,
            materials,
            freq_ghz,
            normalized_pols,
            unit_scale,
            max_panels,
        )
        fine_records = (
            base_records
            if fine_snapshot is base_snapshot
            else _resource_records_for_frequency(
                rcs_solver,
                fine_snapshot,
                materials,
                freq_ghz,
                normalized_pols,
                unit_scale,
                max_panels,
            )
        )
        for requested_pol in requested_pols:
            base = base_records[requested_pol]
            fine = fine_records[requested_pol]
            if fine["formulation"] != base["formulation"]:
                raise RuntimeError(
                    "base/fine resource planning selected different formulations"
                )
            dense_gb = rcs_solver._estimate_memory_gb(
                fine["nodes"],
                use_cfie=False,
                n_regions=max(1, fine["n_regions"]),
                system_dofs=fine["system_dofs"],
                operator_matrices=fine["operator_matrices"],
                n_rhs=max(1, int(n_angles)),
            )
            planned[(freq_ghz, requested_pol)] = {
                "nodes": int(base["nodes"]),
                "panels": int(base["panels"]),
                "base_system_dofs": int(base["system_dofs"]),
                "base_operator_matrices": int(base["operator_matrices"]),
                "fine_nodes": int(fine["nodes"]),
                "fine_panels": int(fine["panels"]),
                "fine_system_dofs": int(fine["system_dofs"]),
                "fine_operator_matrices": int(fine["operator_matrices"]),
                "n_regions": int(fine["n_regions"]),
                "formulation": str(fine["formulation"]),
                # Compatibility aliases: peak planning is governed by the fine mesh.
                "system_dofs": int(fine["system_dofs"]),
                "operator_matrices": int(fine["operator_matrices"]),
                "peak_gb": float(floor_gb) + float(safety) * float(dense_gb),
            }
            if progress is not None:
                progress(freq_ghz, requested_pol)
    return planned


def predict_2d_resources(
    geometry_path: 'str',
    frequency_ghz: 'float',
    polarization: 'str',
    geometry_units: 'str',
    max_panels: 'int',
    fine_factor: 'float' = 1.0,
    n_angles: 'int' = 1,
    safety: 'float' = 1.35,
    floor_gb: 'float' = 0.6,
) -> 'Dict[str, Any]':
    """Backward-compatible one-unit exact resource prediction."""

    key = (float(frequency_ghz), str(polarization))
    return predict_2d_resources_many(
        geometry_path,
        [key[0]],
        [key[1]],
        geometry_units,
        max_panels,
        fine_factor=fine_factor,
        n_angles=n_angles,
        safety=safety,
        floor_gb=floor_gb,
    )[key]


def unit_cost(
    nodes: 'int',
    n_angles: 'int',
    fine_factor: 'float' = 2.0,
    fine_nodes: 'Optional[int]' = None,
    system_dofs: 'Optional[int]' = None,
    fine_system_dofs: 'Optional[int]' = None,
    operator_matrices: 'int' = 3,
    fine_operator_matrices: 'Optional[int]' = None,
) -> 'float':
    """Relative wall-clock cost of one certified unit.

    Only ratios matter -- this feeds bin packing, not a walltime request.  The
    Terms are the three that actually scale: retained operator assembly
    (operator count times N^2 element pairs), LU factorization (system DOFs
    cubed), and multi-RHS solve/residual work (system DOFs squared per angle).
    This distinction matters for coated bodies whose interface-side system can
    be much larger than the boundary-node count. A certified solve runs the
    base mesh and a refined one, so both are counted.
    """

    angles = max(1.0, float(n_angles))

    def _one(n_nodes: 'float', n_dofs: 'float', n_operators: 'int') -> 'float':
        n = max(1.0, float(n_nodes))
        d = max(1.0, float(n_dofs))
        # Three retained matrices is the calibrated N-unknown Robin/sheet
        # baseline, preserving the historical cost for that common case.
        assembly = n * n * max(1.0, float(n_operators) / 3.0)
        rhs_solve = d * d * angles / 1500.0
        factorization = (d ** 3) / 13000.0
        return assembly + rhs_solve + factorization

    base_dofs = nodes if system_dofs is None else int(system_dofs)
    base = _one(nodes, base_dofs, operator_matrices)
    if float(fine_factor) <= 1.0:
        # Survey mode: one mesh, no convergence pair.
        return base
    refined_nodes = (
        max(1.0, float(nodes) * float(fine_factor))
        if fine_nodes is None else max(1.0, float(fine_nodes))
    )
    refined_dofs = (
        max(1.0, float(base_dofs) * float(fine_factor))
        if fine_system_dofs is None else max(1.0, float(fine_system_dofs))
    )
    refined_operators = (
        int(operator_matrices)
        if fine_operator_matrices is None else int(fine_operator_matrices)
    )
    return base + _one(refined_nodes, refined_dofs, refined_operators)


def unit_peak_gb(
    nodes: 'int',
    fine_factor: 'float' = 2.0,
    n_regions: 'int' = 1,
    safety: 'float' = 1.35,
    floor_gb: 'float' = 0.6,
) -> 'float':
    """Peak resident memory for one certified unit, in GB.

    Built on the solver's own dense-storage estimate for the *fine* mesh (the
    larger of the two solves) so the scheduler and the solver's internal memory
    gate cannot disagree about what a unit costs.  ``floor_gb`` covers the
    interpreter, numpy/scipy, and the forked snapshot; ``safety`` covers
    allocator slack and the transient copies a factorization makes.
    """

    fine_nodes = max(1, int(math.ceil(float(nodes) * float(fine_factor))))
    try:
        import rcs_solver

        dense_gb = float(
            rcs_solver._estimate_memory_gb(
                fine_nodes, use_cfie=False, n_regions=max(1, int(n_regions))
            )
        )
    except Exception:
        dense_gb = (128.0 * fine_nodes * fine_nodes) / (1024.0 ** 3)
    return float(floor_gb) + float(safety) * dense_gb


def assembly_threads_for_unit(
    cores: 'int',
    max_concurrent: 'int',
    budget_gb: 'float',
    peak_gb: 'float',
    configured: 'Any' = "auto",
) -> 'int':
    """Choose a CPU reservation/thread count for one memory-sized solve.

    Homogeneous units fill the node without oversubscription: if only four
    copies of a 70 GB unit fit, each gets one quarter of the cores; if many
    cheap units fit, each gets correspondingly fewer.  The dispatcher reserves
    these CPU counts as well as memory, so mixed heavy/light backfill cannot
    multiply the heavy-unit thread count by every cheap process admitted.
    """

    cores = max(1, int(cores))
    max_concurrent = max(1, int(max_concurrent))
    if str(configured).strip().lower() != "auto":
        return max(1, min(cores, int(configured)))
    peak_gb = max(0.0, float(peak_gb))
    if peak_gb <= 0.0:
        concurrency = max_concurrent
    else:
        concurrency = max(
            1,
            min(max_concurrent, int(max(0.0, float(budget_gb)) // peak_gb)),
        )
    return max(1, cores // concurrency)


def predict_bor_extent(
    geometry_path: 'str',
    geometry_units: 'str',
) -> 'Tuple[float, float]':
    """(generatrix arc length, maximum radius) of a BoR geometry, in metres.

    Read straight off the .geo point pairs (x = rho, y = z), so this costs a
    file parse rather than a mesh build.  Returns (0, 0) when the geometry
    cannot be read, which the caller treats as "unknown".
    """

    try:
        import rcs_solver
        from geometry_io import parse_geometry, build_geometry_snapshot

        scale = rcs_solver._unit_scale_to_meters(geometry_units)
        title, segments, ibcs, dielectrics = parse_geometry(
            Path(geometry_path).read_text()
        )
        snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
        arc = 0.0
        radius = 0.0
        for segment in snapshot.get("segments", []) or []:
            for pair in segment.get("point_pairs", []) or []:
                x1 = float(pair.get("x1", 0.0)) * scale
                y1 = float(pair.get("y1", 0.0)) * scale
                x2 = float(pair.get("x2", 0.0)) * scale
                y2 = float(pair.get("y2", 0.0)) * scale
                arc += math.hypot(x2 - x1, y2 - y1)
                radius = max(radius, abs(x1), abs(x2))
        return float(arc), float(radius)
    except Exception:
        return 0.0, 0.0


def bor_unit_cost(
    arc_length_m: 'float',
    radius_m: 'float',
    frequency_ghz: 'float',
    n_aspects: 'int',
) -> 'float':
    """Relative wall-clock cost of one body-of-revolution unit.

    A BoR solve factors one dense system per azimuthal mode.  Generatrix
    elements scale with arc length in wavelengths, and the number of modes that
    must be kept scales with the circumference in wavelengths -- so cost grows
    roughly as (elements^3) x (modes), i.e. the fourth power of frequency.
    Only the ratio matters here; it feeds bin packing, not a walltime request.

    Deliberately coarser than the 2-D model, which builds the real mesh: there
    is no equally cheap way to predict a BoR discretization exactly, and an
    approximate plan plus run-time stealing beats an exact plan that costs a
    solve to compute.  A geometry that cannot be read falls back to unit cost.
    """

    if arc_length_m <= 0.0 or frequency_ghz <= 0.0:
        return 1.0
    wavelength = 299_792_458.0 / (float(frequency_ghz) * 1e9)
    elements = max(1.0, 20.0 * float(arc_length_m) / wavelength)
    modes = max(1.0, 2.0 * math.pi * max(radius_m, wavelength / 20.0) / wavelength + 6.0)
    return (elements ** 3) * modes * (1.0 + max(1, int(n_aspects)) / 500.0)


def balance_units(
    units: 'Sequence[Dict[str, Any]]',
    n_slots: 'int',
    cost_key: 'str' = "cost",
) -> 'List[int]':
    """Assign each unit a slot, longest-processing-time-first.

    Returns a list of slot indices parallel to ``units``.  LPT is the standard
    greedy makespan heuristic (never worse than 4/3 of optimal), and on a
    frequency sweep it is dramatically better than round-robin because it puts
    the handful of expensive high-frequency units on different slots first and
    fills the gaps with cheap ones.
    """

    slots = max(1, int(n_slots))
    order = sorted(
        range(len(units)),
        key=lambda i: (-float(units[i].get(cost_key, 1.0)), i),
    )
    loads = [0.0] * slots
    assignment = [0] * len(units)
    for index in order:
        target = min(range(slots), key=lambda s: (loads[s], s))
        assignment[index] = target
        loads[target] += float(units[index].get(cost_key, 1.0))
    return assignment


def slot_plan_summary(
    units: 'Sequence[Dict[str, Any]]',
    assignment: 'Sequence[int]',
    n_slots: 'int',
) -> 'Dict[str, Any]':
    """Predicted per-slot load, for the submit-time report.

    ``imbalance`` is the plan's makespan against the best any schedule could
    do, not against the mean load.  The mean is the wrong yardstick whenever a
    single unit costs more than an even share -- with 6 units across 50 slots,
    or with one dominant high-frequency unit, a perfect plan still shows a
    large max/mean ratio, and reporting that as imbalance would send you
    tuning a scheduler that is already optimal.  Against the lower bound
    max(total/slots, dearest unit), 1.00 means "nothing left to gain".
    """

    slots = max(1, int(n_slots))
    loads = [0.0] * slots
    counts = [0] * slots
    for unit, slot in zip(units, assignment):
        loads[slot] += float(unit.get("cost", 1.0))
        counts[slot] += 1
    total = sum(loads)
    dearest = max((float(u.get("cost", 1.0)) for u in units), default=0.0)
    lower_bound = max(total / slots, dearest)
    makespan = max(loads) if loads else 0.0
    return {
        "slot_units": counts,
        "idle_slots": sum(1 for count in counts if count == 0),
        "makespan": makespan,
        "lower_bound": lower_bound,
        "imbalance": (makespan / lower_bound) if lower_bound > 0 else 1.0,
        # Kept for anyone reading the raw plan: this is the ratio the old
        # report printed, which is only meaningful when units outnumber slots.
        "max_over_mean": (makespan / (total / slots)) if total > 0 else 1.0,
    }


# -----------------------------------------------------------------------------
# Filesystem claim broker
# -----------------------------------------------------------------------------

class ClaimBroker:
    """Atomic cross-node unit claiming, with steal-on-stale recovery.

    ``O_CREAT | O_EXCL`` is atomic on every filesystem a cluster will put a run
    directory on, which is all the coordination a sweep needs: no scheduler
    process, no message passing, and array tasks that can join or die freely.

    Claims carry a heartbeat (the file's mtime, refreshed by
    :meth:`start_heartbeat`).  A claim whose heartbeat has gone quiet for
    ``stale_seconds`` belongs to a task that was killed or preempted, and any
    other task may take it over.  The result file remains the real idempotency
    guard, so the worst case for a mistakenly stolen unit is that it is solved
    twice and written once.
    """

    def __init__(
        self,
        claims_dir: 'os.PathLike',
        stale_seconds: 'float' = 3600.0,
        heartbeat_seconds: 'float' = 60.0,
    ) -> 'None':
        self.dir = Path(claims_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stale_seconds = float(stale_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self._held: 'Dict[str, Path]' = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: 'Optional[threading.Thread]' = None

    def _path(self, key: 'str') -> 'Path':
        return self.dir / f"{key}.claim"

    def try_claim(self, key: 'str') -> 'bool':
        """Take ``key`` if it is unclaimed or its holder has gone quiet."""

        path = self._path(key)
        payload = json.dumps({
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "job": os.environ.get("SLURM_JOB_ID", ""),
            "array_task": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            "claimed_at": time.time(),
        }).encode("utf-8")
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            if not self._is_stale(path):
                return False
            # Steal: re-create under a private name, then swap it in.  Two
            # stealers race on the rename and one of them wins; the loser's
            # temporary is removed.  The result file still gates the work.
            temporary = self.dir / f".steal.{os.getpid()}.{key}"
            try:
                with open(temporary, "wb") as stream:
                    stream.write(payload)
                os.replace(str(temporary), str(path))
            except OSError:
                try:
                    temporary.unlink()
                except OSError:
                    pass
                return False
            with self._lock:
                self._held[key] = path
            return True
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        with self._lock:
            self._held[key] = path
        return True

    def _is_stale(self, path: 'Path') -> 'bool':
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return True
        return age > self.stale_seconds

    def release(self, key: 'str') -> 'None':
        """Drop the heartbeat for a finished unit (the claim file stays as a
        record that it was done here)."""

        with self._lock:
            self._held.pop(key, None)

    def abandon(self, key: 'str') -> 'None':
        """Give a unit back after a failure, so another task can retry it."""

        with self._lock:
            path = self._held.pop(key, None)
        if path is None:
            path = self._path(key)
        try:
            path.unlink()
        except OSError:
            pass

    def start_heartbeat(self) -> 'None':
        if self._thread is not None:
            return

        def _beat() -> 'None':
            while not self._stop.wait(self.heartbeat_seconds):
                now = time.time()
                with self._lock:
                    paths = list(self._held.values())
                for path in paths:
                    try:
                        os.utime(str(path), (now, now))
                    except OSError:
                        pass

        self._thread = threading.Thread(target=_beat, name="claim-heartbeat", daemon=True)
        self._thread.start()

    def stop_heartbeat(self) -> 'None':
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


# -----------------------------------------------------------------------------
# Memory-aware dispatch
# -----------------------------------------------------------------------------

class MemoryAwareDispatcher:
    """Run units across a process pool under a node memory budget.

    A fixed pool size is the wrong control for a sweep whose units differ by
    two orders of magnitude in footprint: sized for the big units it wastes the
    node on the small ones, and sized for the small ones it OOM-kills on the
    big ones.  This admits work while ``sum(estimated peak) <= budget`` and,
    when CPU requests are supplied, ``sum(assembly threads) <= cores``.  A
    node therefore runs many cheap units concurrently, narrows for expensive
    ones, and can backfill spare resources without oversubscribing either.

    One unit is always admitted when nothing is running, so a unit larger than
    the whole budget still runs (and fails loudly with a MemoryError from the
    solver's own gate) instead of deadlocking the node.
    """

    def __init__(
        self,
        pool: 'Any',
        budget_gb: 'float',
        max_concurrent: 'int',
        cpu_budget: 'Optional[int]' = None,
        poll_seconds: 'float' = 0.05,
    ) -> 'None':
        self.pool = pool
        self.budget_gb = float(budget_gb)
        self.max_concurrent = max(1, int(max_concurrent))
        self.cpu_budget = (
            None if cpu_budget is None else max(1, int(cpu_budget))
        )
        self.poll_seconds = float(poll_seconds)

    def run(
        self,
        candidates: 'Sequence[Dict[str, Any]]',
        prepare: 'Callable[[Dict[str, Any]], Optional[Tuple[str, float, Any]]]',
        on_result: 'Callable[[str, Any], None]',
        on_error: 'Callable[[str, BaseException], None]',
        resource_request: 'Optional[Callable[[Dict[str, Any]], Tuple[float, int]]]' = None,
    ) -> 'None':
        """Work through ``candidates`` in order and drain the pool.

        ``prepare(unit)`` claims the unit and returns
        (key, estimated_gb, apply_async_arguments), or None when the unit is
        already finished or held by another task.  It runs only when there is
        room to start something, so claims are taken at the moment work
        actually begins.

        When ``resource_request`` is supplied, it returns ``(GB, CPUs)`` before
        a unit is claimed.  The dispatcher scans past a temporarily blocked
        large unit and backfills the first smaller unit that fits both budgets.
        Deferred units keep their original order and are reconsidered whenever
        work completes.  Without the callback the legacy memory-only,
        no-backfill behaviour is retained for callers with fixed-size work.
        """

        if resource_request is not None:
            self._run_with_backfill(
                candidates, prepare, on_result, on_error, resource_request
            )
            return

        inflight: 'List[Tuple[str, float, Any]]' = []
        reserved = 0.0
        held: 'Optional[Tuple[str, float, Any]]' = None
        cursor = 0

        while True:
            while len(inflight) < self.max_concurrent:
                if held is None:
                    while cursor < len(candidates):
                        prepared = prepare(candidates[cursor])
                        cursor += 1
                        if prepared is not None:
                            held = prepared
                            break
                    if held is None:
                        break
                gb = max(0.0, float(held[1]))
                if inflight and reserved + gb > self.budget_gb:
                    break
                key, _gb, args = held
                held = None
                inflight.append((key, gb, self.pool.apply_async(*args)))
                reserved += gb

            if not inflight:
                if held is None and cursor >= len(candidates):
                    return
                time.sleep(self.poll_seconds)
                continue

            progressed = False
            for index in range(len(inflight) - 1, -1, -1):
                key, gb, handle = inflight[index]
                if not handle.ready():
                    continue
                inflight.pop(index)
                reserved -= gb
                progressed = True
                try:
                    on_result(key, handle.get())
                except BaseException as exc:  # noqa: BLE001 - reported, not raised
                    on_error(key, exc)
            if not progressed:
                time.sleep(self.poll_seconds)

    def _run_with_backfill(
        self,
        candidates: 'Sequence[Dict[str, Any]]',
        prepare: 'Callable[[Dict[str, Any]], Optional[Tuple[str, float, Any]]]',
        on_result: 'Callable[[str, Any], None]',
        on_error: 'Callable[[str, BaseException], None]',
        resource_request: 'Callable[[Dict[str, Any]], Tuple[float, int]]',
    ) -> 'None':
        """Resource-aware admission with claim-safe, order-preserving backfill."""

        pending: 'Deque[Dict[str, Any]]' = deque(candidates)
        deferred: 'List[Dict[str, Any]]' = []
        inflight: 'List[Tuple[str, float, int, Any]]' = []
        reserved_gb = 0.0
        reserved_cpus = 0

        while True:
            while len(inflight) < self.max_concurrent:
                selected = None
                selected_gb = 0.0
                selected_cpus = 1
                while pending:
                    unit = pending.popleft()
                    requested_gb, requested_cpus = resource_request(unit)
                    requested_gb = max(0.0, float(requested_gb))
                    requested_cpus = max(1, int(requested_cpus))
                    memory_fits = reserved_gb + requested_gb <= self.budget_gb
                    cpu_fits = (
                        self.cpu_budget is None
                        or reserved_cpus + requested_cpus <= self.cpu_budget
                    )
                    # An over-budget unit runs alone and reaches the solver's
                    # own memory/error gate instead of deadlocking this loop.
                    if not inflight or (memory_fits and cpu_fits):
                        selected = unit
                        selected_gb = requested_gb
                        selected_cpus = requested_cpus
                        break
                    deferred.append(unit)
                if selected is None:
                    break

                prepared = prepare(selected)
                if prepared is None:
                    continue
                key, prepared_gb, args = prepared
                if not math.isclose(
                    max(0.0, float(prepared_gb)), selected_gb,
                    rel_tol=1.0e-12, abs_tol=1.0e-12,
                ):
                    on_error(
                        key,
                        ValueError(
                            "resource_request and prepare returned different "
                            f"memory estimates ({selected_gb:g} vs "
                            f"{float(prepared_gb):g} GB)"
                        ),
                    )
                    continue
                handle = self.pool.apply_async(*args)
                inflight.append(
                    (key, selected_gb, selected_cpus, handle)
                )
                reserved_gb += selected_gb
                reserved_cpus += selected_cpus

            if not inflight:
                if deferred:
                    pending.extendleft(reversed(deferred))
                    deferred = []
                    continue
                if not pending:
                    return

            progressed = False
            for index in range(len(inflight) - 1, -1, -1):
                key, gb, cpus, handle = inflight[index]
                if not handle.ready():
                    continue
                inflight.pop(index)
                reserved_gb -= gb
                reserved_cpus -= cpus
                progressed = True
                try:
                    on_result(key, handle.get())
                except BaseException as exc:  # noqa: BLE001 - reported, not raised
                    on_error(key, exc)
            if progressed and deferred:
                pending.extendleft(reversed(deferred))
                deferred = []
            if not progressed:
                time.sleep(self.poll_seconds)


# -----------------------------------------------------------------------------
# SLURM script generation
# -----------------------------------------------------------------------------

def build_sbatch_script(
    *,
    job_name: 'str',
    run_dir: 'os.PathLike',
    script_path: 'os.PathLike',
    array_size: 'int',
    array_throttle: 'Optional[int]',
    partition: 'str',
    cpus_per_node: 'Optional[int]',
    mem_per_node: 'Optional[str]',
    walltime: 'Optional[str]',
    account: 'Optional[str]',
    qos: 'Optional[str]',
    mail_type: 'Optional[str]',
    mail_user: 'Optional[str]',
    extra_sbatch: 'Sequence[str]',
    prologue: 'Sequence[str]',
    python_exe: 'str',
    worker_args: 'str',
    submission_index: 'int',
    blas_threads: 'int',
    extra_env: 'Optional[Dict[str, str]]' = None,
) -> 'str':
    """Write one array-job script.

    Array tasks are interchangeable: every task runs the same worker, which
    pulls units from the shared claim directory.  That means the array size is
    a *parallelism* knob rather than a partitioning key -- oversubscribing or
    cancelling part of an array cannot strand work, and a second submission on
    a different partition can join the same run at any time.
    """

    import shlex

    run_dir = Path(run_dir)
    script_path = Path(script_path)
    array = f"0-{max(1, int(array_size)) - 1}"
    if array_throttle and int(array_throttle) > 0:
        array += f"%{int(array_throttle)}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={array}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --output={run_dir}/logs/sub{submission_index}_%A_%a.out",
        f"#SBATCH --error={run_dir}/logs/sub{submission_index}_%A_%a.err",
        # A requeued task re-joins the run and picks up whatever is unclaimed,
        # so preemption costs only the units that were in flight.
        "#SBATCH --requeue",
        "#SBATCH --open-mode=append",
    ]
    if cpus_per_node is not None:
        lines.append(f"#SBATCH --cpus-per-task={int(cpus_per_node)}")
    else:
        lines.append("#SBATCH --exclusive")
    if mem_per_node:
        lines.append(f"#SBATCH --mem={mem_per_node}")
    if walltime:
        lines.append(f"#SBATCH --time={walltime}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if mail_type:
        lines.append(f"#SBATCH --mail-type={mail_type}")
    if mail_user:
        lines.append(f"#SBATCH --mail-user={mail_user}")
    for extra in extra_sbatch:
        text = str(extra).strip()
        if not text:
            continue
        lines.append(text if text.startswith("#SBATCH") else f"#SBATCH {text}")

    lines += ["", "set -euo pipefail", f"cd {shlex.quote(str(script_path.parent))}"]
    # Pinned before the interpreter starts: several BLAS backends read these
    # only at load time, and an unpinned OpenMP runtime will start one thread
    # per core inside every pool worker.
    for name in _BLAS_THREAD_VARS:
        lines.append(f"export {name}={max(1, int(blas_threads))}")
    for name, value in sorted((extra_env or {}).items()):
        lines.append(f"export {name}={shlex.quote(str(value))}")
    lines += list(prologue)
    lines += [
        (f"exec {shlex.quote(python_exe)} {shlex.quote(str(script_path))} "
         f"{worker_args}"),
        "",
    ]
    return "\n".join(lines)
