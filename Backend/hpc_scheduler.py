#!/usr/bin/env python3
"""
Shared scheduling core for the HPC sweep drivers.

The drivers expand a sweep into (geometry x frequency x polarization) units and
hand them to a pile of nodes.  Getting that right is almost entirely a
scheduling problem, and the three things that decide how many node-hours a
sweep burns are:

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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Node resources
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Provenance fingerprint cache
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Cost and memory model
# ─────────────────────────────────────────────────────────────────────────────

def predict_2d_nodes(
    geometry_path: 'str',
    frequency_ghz: 'float',
    polarization: 'str',
    geometry_units: 'str',
    max_panels: 'int',
) -> 'int':
    """Boundary-node count the 2-D solver will actually discretize to.

    Uses the solver's own meshing rule rather than a proxy, so the cost and
    memory numbers below track what the solve really does (including
    material-dependent wavelength shortening inside dielectrics).  Returns 0
    when the geometry cannot be meshed, which the caller treats as "unknown".
    """

    try:
        import rcs_solver
        from geometry_io import parse_geometry, build_geometry_snapshot

        title, segments, ibcs, dielectrics = parse_geometry(
            Path(geometry_path).read_text()
        )
        snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
        materials = rcs_solver.MaterialLibrary.from_entries(
            snapshot.get("ibcs", []) or [],
            snapshot.get("dielectrics", []) or [],
            base_dir=str(Path(geometry_path).parent),
        )
        lambda_min, _, _ = rcs_solver._mesh_wavelength_for_snapshot(
            snapshot, materials, float(frequency_ghz)
        )
        panels = rcs_solver._build_panels(
            snapshot,
            rcs_solver._unit_scale_to_meters(geometry_units),
            lambda_min,
            max_panels=int(max_panels),
        )
        return int(len(panels))
    except Exception:
        return 0


def unit_cost(nodes: 'int', n_angles: 'int', fine_factor: 'float' = 2.0) -> 'float':
    """Relative wall-clock cost of one certified unit.

    Only ratios matter -- this feeds bin packing, not a walltime request.  The
    terms are the three that actually scale: operator assembly (N^2 element
    pairs), the LU factorization (N^3, but with threaded BLAS behind it, so it
    only overtakes assembly for very large N), and the multi-RHS solve plus
    residual check across the angle sweep (N^2 per angle).  A certified solve
    runs the base mesh and a refined one, so both are counted.
    """

    angles = max(1.0, float(n_angles))

    def _one(n_nodes: 'float') -> 'float':
        n = max(1.0, float(n_nodes))
        return n * n * (1.0 + angles / 1500.0) + (n ** 3) / 13000.0

    base = _one(nodes)
    return base + _one(max(1.0, float(nodes) * float(fine_factor)))


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
    """Predicted per-slot load, for the submit-time report."""

    loads = [0.0] * max(1, int(n_slots))
    counts = [0] * max(1, int(n_slots))
    for unit, slot in zip(units, assignment):
        loads[slot] += float(unit.get("cost", 1.0))
        counts[slot] += 1
    total = sum(loads) or 1.0
    return {
        "slot_units": counts,
        "max_slot_share": max(loads) / total,
        "mean_slot_share": (total / len(loads)) / total,
        "imbalance": (max(loads) / (total / len(loads))) if total > 0 else 1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem claim broker
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Memory-aware dispatch
# ─────────────────────────────────────────────────────────────────────────────

class MemoryAwareDispatcher:
    """Run units across a process pool under a node memory budget.

    A fixed pool size is the wrong control for a sweep whose units differ by
    two orders of magnitude in footprint: sized for the big units it wastes the
    node on the small ones, and sized for the small ones it OOM-kills on the
    big ones.  This admits work while ``sum(estimated peak) <= budget``, so a
    node runs many cheap units concurrently and automatically narrows to a few
    when the expensive ones come up.

    One unit is always admitted when nothing is running, so a unit larger than
    the whole budget still runs (and fails loudly with a MemoryError from the
    solver's own gate) instead of deadlocking the node.
    """

    def __init__(
        self,
        pool: 'Any',
        budget_gb: 'float',
        max_concurrent: 'int',
        poll_seconds: 'float' = 0.05,
    ) -> 'None':
        self.pool = pool
        self.budget_gb = float(budget_gb)
        self.max_concurrent = max(1, int(max_concurrent))
        self.poll_seconds = float(poll_seconds)

    def run(
        self,
        candidates: 'Sequence[Dict[str, Any]]',
        prepare: 'Callable[[Dict[str, Any]], Optional[Tuple[str, float, Any]]]',
        on_result: 'Callable[[str, Any], None]',
        on_error: 'Callable[[str, BaseException], None]',
    ) -> 'None':
        """Work through ``candidates`` in order and drain the pool.

        ``prepare(unit)`` claims the unit and returns
        (key, estimated_gb, apply_async_arguments), or None when the unit is
        already finished or held by another task.  It runs only when there is
        room to start something, so claims are taken at the moment work
        actually begins.

        A prepared unit that does not fit the remaining budget is *held*, not
        skipped: skipping it would let a queue of cheap units run ahead of the
        expensive one that the longest-processing-time ordering deliberately
        put first, which is exactly the tail imbalance this is here to avoid.
        """

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


# ─────────────────────────────────────────────────────────────────────────────
# SLURM script generation
# ─────────────────────────────────────────────────────────────────────────────

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
    lines += list(prologue)
    lines += [
        (f"exec {shlex.quote(python_exe)} {shlex.quote(str(script_path))} "
         f"{worker_args}"),
        "",
    ]
    return "\n".join(lines)
