#!/usr/bin/env python3
"""
Integration test for the HPC sweep scheduler.

Exercises the parts that are easy to get subtly wrong and expensive to debug
on a cluster:

* `hpc_common.configure_driver` can still rewrite every CONFIG constant.
* Submission builds a manifest, a schedule, and sbatch scripts, and the
  scheduling plan is genuinely balanced by cost rather than by index.
* Several array tasks working the same run in parallel each solve every unit at
  most once (atomic claims), between them cover the run exactly, and the
  results verify their attestations.
* A task that re-joins a finished run skips everything.
* Stale claims are stealable, so a killed task's units are not stranded.
* The memory-aware dispatcher respects its budget and still makes progress on
  a unit larger than the whole budget.

Usage:
    python tests/test_hpc_scheduling.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
sys.path.insert(0, str(BACKEND))

import hpc_common  # noqa: E402
import hpc_scheduler  # noqa: E402

FAILURES = []


def check(condition, message):
    if condition:
        print(f"  ok   {message}")
    else:
        FAILURES.append(message)
        print(f"  FAIL {message}")


# --- unit-level checks ------------------------------------------------------

def test_balance():
    print("\nload balancing")
    # A frequency sweep: cost grows steeply with frequency, and the unit list
    # is ordered geometry-major, which is exactly the shape that defeats
    # round-robin when the slot count divides the frequency count.
    freqs = [2.0, 4.0, 6.0, 8.0]
    units = []
    for _geom in range(6):
        for f in freqs:
            units.append({"cost": f ** 2, "name": f"g{_geom}_f{f}"})
    n_slots = 4

    assignment = hpc_scheduler.balance_units(units, n_slots)
    lpt = [0.0] * n_slots
    for unit, slot in zip(units, assignment):
        lpt[slot] += unit["cost"]

    rr = [0.0] * n_slots
    for index, unit in enumerate(units):
        rr[index % n_slots] += unit["cost"]

    # Measured against the makespan lower bound the reporting uses, so the
    # test and the number printed at submit time mean the same thing.
    total = sum(u["cost"] for u in units)
    lower_bound = max(total / n_slots, max(u["cost"] for u in units))
    lpt_imbalance = max(lpt) / lower_bound
    rr_imbalance = max(rr) / lower_bound
    print(f"       round-robin makespan factor {rr_imbalance:.2f}, "
          f"LPT {lpt_imbalance:.2f}")
    check(lpt_imbalance <= 1.02, "LPT plan is within 2% of optimal")
    check(rr_imbalance > 1.5,
          "round-robin really is badly imbalanced on this shape "
          f"({rr_imbalance:.2f}x)")
    check(len(set(assignment)) == n_slots, "every slot receives work")

    summary = hpc_scheduler.slot_plan_summary(units, assignment, n_slots)
    check(abs(summary["imbalance"] - lpt_imbalance) < 1e-9,
          "the reported balance matches the plan actually produced")

    # More slots than units: a perfect plan still has a large max/mean ratio,
    # so the report must not read that as imbalance.
    sparse = [{"cost": c} for c in (100.0, 100.0, 25.0, 4.0)]
    sparse_assignment = hpc_scheduler.balance_units(sparse, 50)
    sparse_summary = hpc_scheduler.slot_plan_summary(sparse, sparse_assignment, 50)
    check(abs(sparse_summary["imbalance"] - 1.0) < 1e-9,
          "50 slots for 4 units reports optimal, not spurious imbalance")
    check(sparse_summary["idle_slots"] == 46, "idle slots are reported")


def test_windows_lock_initialization():
    """The Windows coordination byte must be seeded under its own lock."""

    print("\nWindows lock initialization")

    class OrderedWindowsLock:
        LK_LOCK = 1
        LK_NBLCK = 2
        LK_UNLCK = 3

        def __init__(self, events, *, unavailable=False):
            self.events = events
            self.unavailable = unavailable

        def locking(self, _fd, mode, _count):
            self.events.append("unlock" if mode == self.LK_UNLCK else "lock")
            if self.unavailable and mode == self.LK_NBLCK:
                raise OSError(hpc_scheduler.errno.EACCES, "lock held")

    with tempfile.TemporaryDirectory() as tmp:
        operation_path = Path(tmp) / "operation.lock"
        events = []
        fake_lock = OrderedWindowsLock(events)
        real_write = os.write

        def record_write(fd, payload):
            events.append("write")
            return real_write(fd, payload)

        with mock.patch.object(hpc_scheduler, "fcntl", None), \
                mock.patch.object(hpc_scheduler, "msvcrt", fake_lock), \
                mock.patch.object(hpc_scheduler.os, "write", side_effect=record_write):
            fd = hpc_scheduler.ClaimBroker._try_lock_file(operation_path)
            check(fd is not None and events[:2] == ["lock", "write"],
                  "an empty Windows operation file is seeded only after locking")
            hpc_scheduler.ClaimBroker._unlock_file(fd)
        check(operation_path.read_bytes() == b"\0" and events[-1] == "unlock",
              "the initialized Windows operation lock releases cleanly")

        # A nonblocking contender must report ordinary unavailability and close
        # the descriptor it opened, rather than leaking it or raising.
        captured_fds = []
        real_open = os.open

        def record_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            captured_fds.append(fd)
            return fd

        unavailable_lock = OrderedWindowsLock([], unavailable=True)
        with mock.patch.object(hpc_scheduler, "fcntl", None), \
                mock.patch.object(hpc_scheduler, "msvcrt", unavailable_lock), \
                mock.patch.object(hpc_scheduler.os, "open", side_effect=record_open):
            result = hpc_scheduler.ClaimBroker._try_lock_file(operation_path)
        check(result is None, "a busy nonblocking Windows operation lock is unavailable")
        try:
            os.fstat(captured_fds[-1])
        except OSError:
            closed = True
        else:
            closed = False
            os.close(captured_fds[-1])
        check(closed, "a busy Windows operation-lock attempt closes its descriptor")


def test_claims():
    print("\nclaim broker")
    with tempfile.TemporaryDirectory() as tmp:
        broker = hpc_scheduler.ClaimBroker(Path(tmp) / "claims", stale_seconds=1.0)
        try:
            broker.try_claim("../escape")
        except ValueError:
            check(True, "claim keys cannot traverse outside the claims directory")
        else:
            check(False, "claim keys cannot traverse outside the claims directory")
        check(broker.try_claim("unit-a"), "first claim succeeds")
        check(not broker.try_claim("unit-a"), "second claim on the same unit fails")

        other = hpc_scheduler.ClaimBroker(Path(tmp) / "claims", stale_seconds=1.0)
        check(not other.try_claim("unit-a"), "another task cannot take a live claim")

        stale_path = Path(tmp) / "claims" / "unit-a.claim"
        old = time.time() - 120.0
        os.utime(stale_path, (old, old))
        check(other.try_claim("unit-a"), "a stale claim is stealable")
        replacement = json.loads(stale_path.read_text(encoding="utf-8"))
        check(replacement.get("schema") == "ghost.hpc.claim.v2",
              "claims carry the versioned ownership schema")
        check(replacement.get("owner_token") == other.owner_token,
              "stale takeover installs the new owner's token")
        check(replacement.get("generation") == 2,
              "stale takeover advances the claim generation")

        replacement_mtime = stale_path.stat().st_mtime_ns
        broker._heartbeat_once()
        check(stale_path.stat().st_mtime_ns == replacement_mtime,
              "the former owner's heartbeat cannot refresh a replacement claim")
        broker.abandon("unit-a")
        check(stale_path.is_file(),
              "the former owner cannot abandon a replacement claim")

        broker.abandon("unit-b")  # must not raise on an unheld key
        check(broker.try_claim("unit-b"), "claim after abandon succeeds")
        broker.abandon("unit-b")
        check(broker.try_claim("unit-b"), "abandon releases the unit again")

        # Two stale takers must not both believe they won.  The short operation
        # lease serializes their generation check and replacement.
        check(broker.try_claim("unit-race"), "race fixture claim succeeds")
        race_path = Path(tmp) / "claims" / "unit-race.claim"
        os.utime(race_path, (old, old))
        contenders = [
            hpc_scheduler.ClaimBroker(Path(tmp) / "claims", stale_seconds=1.0)
            for _ in range(2)
        ]
        barrier = threading.Barrier(2)
        outcomes = []

        def contend(candidate):
            barrier.wait()
            outcomes.append(candidate.try_claim("unit-race"))

        threads = [threading.Thread(target=contend, args=(candidate,))
                   for candidate in contenders]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
        check(outcomes.count(True) == 1 and outcomes.count(False) == 1,
              "exactly one concurrent stale taker wins")

        # A paused mutation owner must remain fenced even when its stable lock
        # file looks arbitrarily old.  Time-based unlinking used to let both
        # contenders return True in this interleaving.
        check(broker.try_claim("unit-paused"), "paused-owner fixture claim succeeds")
        paused_path = Path(tmp) / "claims" / "unit-paused.claim"
        os.utime(paused_path, (old, old))
        paused_operation = contenders[0]._acquire_operation("unit-paused")
        check(paused_operation is not None, "paused contender owns mutation lock")
        operation_path = contenders[0]._operation_path("unit-paused")
        os.utime(operation_path, (old, old))
        check(not contenders[1].try_claim("unit-paused"),
              "an old-looking live mutation lock cannot be stolen")
        contenders[0]._release_operation(paused_operation)
        check(contenders[1].try_claim("unit-paused"),
              "claim takeover proceeds after the paused owner releases its lock")

        check(broker.try_claim("unit-abandon-busy"),
              "busy-abandon fixture claim succeeds")
        abandon_path = Path(tmp) / "claims" / "unit-abandon-busy.claim"
        with mock.patch.object(broker, "_acquire_operation", return_value=None):
            broker.abandon("unit-abandon-busy")
        check(abandon_path.is_file() and "unit-abandon-busy" in broker._held,
              "a busy abandon retains ownership tracking for retry")
        broker.abandon("unit-abandon-busy")
        check(not abandon_path.exists() and "unit-abandon-busy" not in broker._held,
              "a retried abandon removes the claim and local tracking")

        heartbeat = hpc_scheduler.ClaimBroker(
            Path(tmp) / "heartbeat-claims",
            stale_seconds=1.0,
            heartbeat_seconds=0.01,
        )
        check(heartbeat.try_claim("unit-io"), "heartbeat I/O fixture claim succeeds")
        recovered = threading.Event()
        heartbeat_calls = [0]
        original_heartbeat_once = heartbeat._heartbeat_once

        def flaky_heartbeat():
            heartbeat_calls[0] += 1
            if heartbeat_calls[0] == 1:
                raise OSError("transient shared-filesystem error")
            original_heartbeat_once()
            recovered.set()

        with mock.patch.object(heartbeat, "_heartbeat_once", side_effect=flaky_heartbeat):
            heartbeat.start_heartbeat()
            check(recovered.wait(1.0),
                  "claim heartbeat retries after a transient filesystem error")
            check(heartbeat._thread is not None and heartbeat._thread.is_alive(),
                  "claim heartbeat thread remains alive after transient I/O")
            heartbeat.stop_heartbeat()

        read_heartbeat = hpc_scheduler.ClaimBroker(
            Path(tmp) / "heartbeat-read-claims",
            stale_seconds=1.0,
            heartbeat_seconds=0.01,
        )
        check(read_heartbeat.try_claim("unit-read-io"),
              "heartbeat read-I/O fixture claim succeeds")
        original_read = read_heartbeat._read_document
        read_recovered = threading.Event()
        read_calls = [0]

        def flaky_read(path):
            read_calls[0] += 1
            if read_calls[0] == 1:
                raise OSError("transient claim read error")
            document = original_read(path)
            read_recovered.set()
            return document

        with mock.patch.object(read_heartbeat, "_read_document", side_effect=flaky_read):
            read_heartbeat.start_heartbeat()
            check(read_recovered.wait(1.0),
                  "claim heartbeat retries after transient claim-file read error")
            check("unit-read-io" in read_heartbeat._held,
                  "transient claim-file read error does not drop ownership")
            check(read_heartbeat._thread is not None
                  and read_heartbeat._thread.is_alive(),
                  "claim heartbeat remains alive after claim-file read error")
            read_heartbeat.stop_heartbeat()


class _FakeHandle:
    def __init__(self, value):
        self._value = value

    def ready(self):
        return True

    def get(self):
        return self._value


class _FakePool:
    """Runs work inline, recording how much was in flight at once."""

    def __init__(self):
        self.order = []

    def apply_async(self, fn, args):
        self.order.append(args[0])
        return _FakeHandle(fn(*args))


def test_memory_admission():
    print("\nmemory-aware admission")
    # A budget of 10 GB, one 25 GB unit (bigger than the whole budget) and a
    # pile of 4 GB ones. Everything must run, and nothing may be dropped.
    units = [{"name": "huge", "gb": 25.0}] + [
        {"name": f"small{i}", "gb": 4.0} for i in range(6)
    ]
    pool = _FakePool()
    dispatcher = hpc_scheduler.MemoryAwareDispatcher(
        pool, budget_gb=10.0, max_concurrent=8, poll_seconds=0.0
    )
    seen = []

    def prepare(unit):
        return (unit["name"], unit["gb"], (lambda name: name, (unit["name"],)))

    dispatcher.run(units, prepare, lambda key, value: seen.append(value),
                   lambda key, exc: seen.append(("error", key, exc)))
    check(sorted(seen) == sorted(u["name"] for u in units),
          "every unit ran exactly once, including one over budget")

    # A unit the caller declines (already done, or claimed elsewhere) is skipped
    # without stalling the queue.
    pool = _FakePool()
    dispatcher = hpc_scheduler.MemoryAwareDispatcher(
        pool, budget_gb=100.0, max_concurrent=4, poll_seconds=0.0
    )
    seen = []

    def prepare_partial(unit):
        if unit["name"].endswith(("1", "3")):
            return None
        return (unit["name"], unit["gb"], (lambda name: name, (unit["name"],)))

    dispatcher.run(units, prepare_partial, lambda key, value: seen.append(value),
                   lambda key, exc: seen.append(("error", key, exc)))
    check(len(seen) == len(units) - 2, "declined units are skipped cleanly")


def test_memory_cpu_backfill():
    print("\nmemory/CPU-aware backfill")

    class _Handle:
        def __init__(self, owner, payload):
            self.owner = owner
            self.payload = payload
            self.polls = 0
            self.finished = False

        def ready(self):
            self.polls += 1
            if self.polls <= 2:
                return False
            if not self.finished:
                self.finished = True
                _name, gb, cpus = self.payload
                self.owner.live_gb -= gb
                self.owner.live_cpus -= cpus
            return True

        def get(self):
            return self.payload[0]

    class _TrackingPool:
        def __init__(self):
            self.order = []
            self.live_gb = 0.0
            self.live_cpus = 0
            self.peak_gb = 0.0
            self.peak_cpus = 0

        def apply_async(self, fn, args):
            payload = args[0]
            self.order.append(payload[0])
            self.live_gb += payload[1]
            self.live_cpus += payload[2]
            self.peak_gb = max(self.peak_gb, self.live_gb)
            self.peak_cpus = max(self.peak_cpus, self.live_cpus)
            return _Handle(self, payload)

    units = [
        {"name": "heavy0", "gb": 6.0, "cpus": 4},
        {"name": "heavy1", "gb": 6.0, "cpus": 4},
        {"name": "small0", "gb": 4.0, "cpus": 2},
        {"name": "small1", "gb": 4.0, "cpus": 2},
    ]
    pool = _TrackingPool()
    dispatcher = hpc_scheduler.MemoryAwareDispatcher(
        pool, budget_gb=10.0, max_concurrent=8, cpu_budget=6,
        poll_seconds=0.0,
    )
    prepared = []
    seen = []

    def prepare(unit):
        prepared.append(unit["name"])
        payload = (unit["name"], unit["gb"], unit["cpus"])
        return (unit["name"], unit["gb"], (lambda value: value, (payload,)))

    dispatcher.run(
        units, prepare, lambda key, value: seen.append(value),
        lambda key, exc: seen.append(("error", key, exc)),
        resource_request=lambda unit: (unit["gb"], unit["cpus"]),
    )
    check(pool.order[:2] == ["heavy0", "small0"],
          "a fitting small unit backfills behind a blocked heavy unit")
    check(prepared[:2] == ["heavy0", "small0"],
          "a blocked unit is not claimed before it can be dispatched")
    check(pool.peak_gb <= 10.0 and pool.peak_cpus <= 6,
          "backfill respects both memory and CPU reservations")
    check(sorted(seen) == sorted(unit["name"] for unit in units),
          "backfill runs every unit exactly once")

    heavy_threads = hpc_scheduler.assembly_threads_for_unit(
        96, 96, 318.75, 70.0
    )
    cheap_threads = hpc_scheduler.assembly_threads_for_unit(
        96, 96, 318.75, 10.0
    )
    check(heavy_threads == 24 and cheap_threads == 3,
          "per-unit threads shrink as memory permits more concurrent solves")


def test_formulation_resource_plan():
    print("\nformulation-aware resource planning")
    pec = hpc_scheduler.predict_2d_resources(
        str(REPO / "geometries" / "body.geo"), 3.0, "TM", "meters",
        50000, fine_factor=1.5, n_angles=19,
    )
    check(pec["formulation"] == "robin",
          "PEC is planned as the active Robin formulation")
    check(pec["system_dofs"] == pec["fine_nodes"],
          "PEC reserves an N-by-N system rather than a generic 2N system")
    check(pec["fine_nodes"] > pec["nodes"],
          "the planner builds the genuinely refined certification mesh")

    with tempfile.TemporaryDirectory() as tmp:
        coating = Path(tmp) / "coating.geo"
        coating.write_text(
            """Title: resource-plan coating
Segment: outer 3
properties: 3 12 0 1 0
-1 -1  -1 1
-1 1  1 1
1 1  1 -1
1 -1  -1 -1
Segment: inner 4
properties: 4 12 0 1 0
-0.8 -0.8  -0.8 0.8
-0.8 0.8  0.8 0.8
0.8 0.8  0.8 -0.8
0.8 -0.8  -0.8 -0.8
IBCS_Resistances:
Dielectrics:
1 2.8 -0.06 1 0
"""
        )
        layered = hpc_scheduler.predict_2d_resources(
            str(coating), 0.1, "TM", "meters", 50000,
            fine_factor=1.5, n_angles=19,
        )
    check(layered["formulation"] == "multi_region",
          "a coated PEC is planned as multi-region")
    check(layered["n_regions"] == 2,
          "the planner counts exterior and dielectric regions")
    check(layered["system_dofs"] > layered["fine_nodes"],
          "multi-region planning uses interface-side DOFs, not panel count")
    node_only_cost = hpc_scheduler.unit_cost(
        layered["nodes"], 19, 1.5,
        fine_nodes=layered["fine_nodes"],
    )
    formulation_cost = hpc_scheduler.unit_cost(
        layered["nodes"], 19, 1.5,
        fine_nodes=layered["fine_nodes"],
        system_dofs=layered["base_system_dofs"],
        fine_system_dofs=layered["fine_system_dofs"],
        operator_matrices=layered["base_operator_matrices"],
        fine_operator_matrices=layered["fine_operator_matrices"],
    )
    check(formulation_cost > node_only_cost,
          "multi-region load balancing charges its larger system and operators")


def test_batched_exact_resource_plan():
    print("\nbatched exact resource planning")
    import rcs_solver

    with tempfile.TemporaryDirectory() as tmp:
        closed = Path(tmp) / "closed_pec.geo"
        closed.write_text(
            """Title: closed PEC batch planner
Segment: square 2
properties: 2 0 0 0 0
-1 -1  -1 1
-1 1  1 1
1 1  1 -1
1 -1  -1 -1
IBCS_Resistances:
Dielectrics:
"""
        )

        validate_original = rcs_solver.validate_geometry_snapshot_for_solver
        mesh_original = rcs_solver._build_linear_mesh_interface_aware
        info_original = rcs_solver._build_coupled_panel_info
        calls = {"validate": 0, "mesh": 0, "info": 0}

        def counted_validate(*args, **kwargs):
            calls["validate"] += 1
            return validate_original(*args, **kwargs)

        def counted_mesh(*args, **kwargs):
            calls["mesh"] += 1
            return mesh_original(*args, **kwargs)

        def counted_info(*args, **kwargs):
            calls["info"] += 1
            return info_original(*args, **kwargs)

        rcs_solver.validate_geometry_snapshot_for_solver = counted_validate
        rcs_solver._build_linear_mesh_interface_aware = counted_mesh
        rcs_solver._build_coupled_panel_info = counted_info
        progress = []
        try:
            batched = hpc_scheduler.predict_2d_resources_many(
                str(closed), [1.0, 2.0], ["TM", "TE"], "meters", 50000,
                fine_factor=1.0, n_angles=19,
                progress=lambda frequency, polarization: progress.append(
                    (frequency, polarization)
                ),
            )
        finally:
            rcs_solver.validate_geometry_snapshot_for_solver = validate_original
            rcs_solver._build_linear_mesh_interface_aware = mesh_original
            rcs_solver._build_coupled_panel_info = info_original

        check(calls["validate"] == 1,
              "survey planning validates geometry once, not once per unit")
        check(calls["mesh"] == 2,
              "TM/TE share one exact topology mesh per frequency")
        check(calls["info"] == 4,
              "each unit builds its material coefficients exactly once")
        check(len(progress) == 4,
              "progress reports every frequency/polarization resource record")

        exact = True
        for frequency in (1.0, 2.0):
            for polarization in ("TM", "TE"):
                single = hpc_scheduler.predict_2d_resources(
                    str(closed), frequency, polarization, "meters", 50000,
                    fine_factor=1.0, n_angles=19,
                )
                exact = exact and batched[(frequency, polarization)] == single
        check(exact,
              "batched reuse is numerically identical to isolated exact planning")


def test_survey_mode_cost():
    print("\nsurvey mode (MESH_CERTIFICATION = False)")
    # fine_factor <= 1 means "one mesh", not "a second mesh the same size".
    certified = hpc_scheduler.unit_cost(2000, 19, 1.5)
    survey = hpc_scheduler.unit_cost(2000, 19, 1.0)
    check(abs(survey - hpc_scheduler.unit_cost(2000, 19, 0.5)) < 1e-9,
          "any fine_factor <= 1 costs a single mesh")
    check(2.5 < certified / survey < 4.0,
          f"certification is modelled as ~3x the work ({certified / survey:.2f}x)")
    big_c = hpc_scheduler.unit_peak_gb(5000, 1.5)
    big_s = hpc_scheduler.unit_peak_gb(5000, 1.0)
    check(1.9 < big_c / big_s < 2.3,
          f"the refined mesh drives the memory reservation ({big_c / big_s:.2f}x)")


def test_memory_heavy_geometry():
    """Concurrency must follow per-unit memory, not the core count."""

    print("\nmemory-heavy geometry")
    import threading

    class _Pool:
        def __init__(self):
            self.live = 0
            self.peak = 0
            self.lock = threading.Lock()

        def apply_async(self, fn, args):
            with self.lock:
                self.live += 1
                self.peak = max(self.peak, self.live)
            handle = type("H", (), {})()
            handle._n = 0
            name = args[0]

            def ready(_h=handle):
                _h._n += 1
                if _h._n > 2:
                    with self.lock:
                        self.live -= 1
                    return True
                return False

            handle.ready = ready
            handle.get = lambda: name
            return handle

    def peak_concurrency(budget, per_unit_gb, count):
        pool = _Pool()
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=budget, max_concurrent=96, poll_seconds=0.0
        )
        units = [{"name": f"u{i}", "gb": per_unit_gb} for i in range(count)]
        done = []
        dispatcher.run(
            units,
            lambda u: (u["name"], u["gb"], (lambda a: a, (u["name"],))),
            lambda k, v: done.append(v),
            lambda k, e: done.append(("ERR", k)),
        )
        return pool.peak, len(done)

    light, n_light = peak_concurrency(100.0, 0.7, 20)
    heavy, n_heavy = peak_concurrency(100.0, 45.0, 20)
    check(n_light == 20 and n_heavy == 20, "every unit runs at both weights")
    check(light > heavy,
          f"heavy units run less concurrently ({light} light vs {heavy} heavy)")
    check(heavy * 45.0 <= 100.0,
          f"the heavy case stays inside the budget ({heavy} x 45 GB)")

    over, n_over = peak_concurrency(100.0, 400.0, 3)
    check(over == 1 and n_over == 3,
          "a unit larger than the whole budget runs alone, without deadlock")


def test_resource_detection():
    print("\nresource detection")
    saved = {name: os.environ.get(name) for name in
             ("SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU")}
    try:
        os.environ["SLURM_CPUS_PER_TASK"] = "96"
        os.environ["SLURM_MEM_PER_NODE"] = str(750 * 1024)
        os.environ.pop("SLURM_MEM_PER_CPU", None)
        check(hpc_scheduler.detect_cores() == 96, "SLURM core count is honoured")
        check(abs(hpc_scheduler.detect_memory_gb() - 750.0) < 1.0,
              "SLURM node memory is honoured over /proc/meminfo")
        os.environ.pop("SLURM_MEM_PER_NODE")
        os.environ["SLURM_MEM_PER_CPU"] = "4096"
        check(abs(hpc_scheduler.detect_memory_gb() - 384.0) < 1.0,
              "per-CPU memory allocation is scaled by the core count")
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_fingerprint_cache():
    print("\nfingerprint cache")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "source.py"
        target.write_text("x = 1\n")
        cache = hpc_scheduler.FingerprintCache(full_recheck_seconds=1e9)
        first = cache.sha256_file(str(target))
        second = cache.sha256_file(str(target))
        check(first == second, "repeat hash is stable")
        time.sleep(0.01)
        target.write_text("x = 2\n")
        os.utime(target, None)
        check(cache.sha256_file(str(target)) != first,
              "a changed file is re-hashed, not served from cache")


# --- end-to-end sweep -------------------------------------------------------

DRIVER_SETTINGS = {
    "FREQUENCIES_GHZ": [2.0, 3.0, 4.0],
    "AZIMUTHS_DEG": [0.0, 45.0, 90.0],
    "GEOMETRY_UNITS": "meters",
    "MAX_PANELS": 20000,
    "N_NODES": 2,
    "N_JOBS": 1,
    "SUBMIT": False,
    "TASKS_PER_CHILD": 2,
    "MAX_WORKERS_PER_NODE": 2,
    "CLAIM_STALE_SECONDS": 60,
}


def _write_closed_2d_geometry(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Title: closed 2-D PEC test\n"
        "Segment: body 2\n"
        "properties: 2 0 0 0 0\n"
        "-0.02 -0.02 -0.02 0.02\n"
        "-0.02 0.02 0.02 0.02\n"
        "0.02 0.02 0.02 -0.02\n"
        "0.02 -0.02 -0.02 -0.02\n"
        "IBCS_Resistances:\n"
        "Dielectrics:\n",
        encoding="utf-8",
    )
    return path


def _stage_run(root):
    """Configure a private copy of the driver, as hpc_common intends."""

    geom_root = root / "geometries"
    source = _write_closed_2d_geometry(root / "closed_body.geo")
    sources = [source]
    hpc_common.stage_geometry(sources, geom_root)
    # A second geometry so the sweep has more than one stem.
    shutil.copyfile(str(sources[0]), str(geom_root / "FRD" / "body_two.geo"))

    settings = dict(DRIVER_SETTINGS)
    settings["FRD_DIR"] = str(geom_root / "FRD")
    settings["OPN_DIR"] = str(geom_root / "OPN")
    settings["OUTPUT_DIR"] = str(root / "runs")
    return hpc_common.configure_driver(
        BACKEND / "run_hpc_monostatic.py", root / "driver.py", settings
    )


def _subprocess_env():
    """Keep the caller's numerical runtime while adding the staged backend."""

    inherited = str(os.environ.get("PYTHONPATH", "")).strip()
    pythonpath = str(BACKEND)
    if inherited:
        pythonpath = os.pathsep.join((pythonpath, inherited))
    return {**os.environ, "PYTHONPATH": pythonpath}


def _run(driver, args, timeout=1800):
    return subprocess.run(
        [sys.executable, str(driver)] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=timeout,
        env=_subprocess_env(),
    )


def test_no_task_starvation():
    """A fast starter must not claim work planned for its peers.

    Concurrency used to be sized from the whole sweep rather than from a
    task's own share, and the dispatcher fills its pool from the head of the
    candidate list -- so the first task to start reached straight past its
    share into everyone else's. On a 96-core node with a 40-unit sweep that
    meant one task took the entire run and the other nine exited having
    written nothing.
    """

    print("\nstarvation (many tasks, one small sweep)")
    root = Path(tempfile.mkdtemp(prefix="ghost-starve-"))
    try:
        geom_root = root / "geometries"
        source = _write_closed_2d_geometry(root / "closed_body.geo")
        hpc_common.stage_geometry([source], geom_root)
        shutil.copyfile(str(source), str(geom_root / "FRD" / "body_two.geo"))
        tasks = 6
        settings = dict(DRIVER_SETTINGS)
        settings.update({
            "FRD_DIR": str(geom_root / "FRD"),
            "OPN_DIR": str(geom_root / "OPN"),
            "OUTPUT_DIR": str(root / "runs"),
            "FREQUENCIES_GHZ": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "AZIMUTHS_DEG": [0.0, 90.0],
            "N_NODES": tasks,
            "MAX_WORKERS_PER_NODE": None,
        })
        driver = hpc_common.configure_driver(
            BACKEND / "run_hpc_monostatic.py", root / "driver.py", settings
        )
        env = _subprocess_env()
        if subprocess.run([sys.executable, str(driver)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env=env).returncode:
            check(False, "submit failed")
            return
        run_dir = hpc_common.latest_run_dir(root / "runs")
        procs = [
            subprocess.Popen(
                [sys.executable, str(driver), "--worker", str(run_dir), "0", str(t)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, env=env,
            )
            for t in range(tasks)
        ]
        outputs = [p.communicate()[0] for p in procs]
        wrote = [int(o.split("wrote=")[1].split(",")[0]) for o in outputs
                 if "wrote=" in o]
        check(len(wrote) == tasks, f"all {tasks} tasks reported ({len(wrote)})")
        check(sum(wrote) == 12, f"all 12 units solved (got {sum(wrote)})")
        check(all(n > 0 for n in wrote),
              f"no task was starved by a faster peer (per-task writes: {wrote})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_end_to_end():
    print("\nend-to-end sweep")
    root = Path(tempfile.mkdtemp(prefix="ghost-hpc-test-"))
    try:
        try:
            driver = _stage_run(root)
        except ValueError as exc:
            check(False, f"configure_driver rejected a CONFIG name: {exc}")
            return
        check(True, "configure_driver rewrote every requested CONFIG constant")

        result = _run(driver, [])
        if result.returncode != 0:
            check(False, f"submit failed:\n{result.stdout}")
            return
        check(True, "submit completed")

        run_dir = hpc_common.latest_run_dir(root / "runs")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        expected_units = 2 * 3  # geometries x frequencies; each holds VV+HH
        check(manifest["n_units"] == expected_units,
              f"manifest holds {expected_units} units")
        check((run_dir / "schedule.json").is_file(), "schedule.json was written")
        schedule = json.loads((run_dir / "schedule.json").read_text())
        check(schedule.get("planning", {}).get("method") == "batched_exact",
              "schedule records the batched exact planning method")
        check("Planning exact resources" in result.stdout
              and "Resource plan ready" in result.stdout,
              "submit output reports resource-planning progress and timing")
        check(all(int(r["nodes"]) > 0 for r in schedule["units"]),
              "every unit was pre-meshed for its cost estimate")
        check(all(float(r["peak_gb"]) > 0 for r in schedule["units"]),
              "every unit carries a memory estimate")
        costs = {r["unit"]: r["cost"] for r in schedule["units"]}
        by_freq = {}
        for name, cost in costs.items():
            by_freq.setdefault(name.split("_")[0], []).append(cost)
        check(max(map(max, by_freq.values())) > 2.0 * min(map(min, by_freq.values())),
              "the cost model separates cheap and expensive frequencies")
        check(len(list(run_dir.glob("submit_job*.slurm"))) == 1,
              "one sbatch script per submission")
        slurm_text = (run_dir / "submit_job0.slurm").read_text()
        check("--array=0-1" in slurm_text, "array size matches N_NODES")
        check("--requeue" in slurm_text, "array tasks are requeueable")
        check("OMP_NUM_THREADS=1" in slurm_text,
              "BLAS threads are pinned before the interpreter starts")

        # A worker must never recreate the old zero-reservation fallback when
        # its submit-time plan is missing or corrupt.
        schedule_path = run_dir / "schedule.json"
        schedule_text = schedule_path.read_text()
        schedule_path.unlink()
        missing_plan = _run(
            driver, ["--worker", str(run_dir), "0", "0"]
        )
        check(
            missing_plan.returncode != 0
            and "refusing to run without per-unit memory reservations"
            in missing_plan.stdout,
            "a missing schedule fails closed before any unit starts",
        )
        schedule_path.write_text("{not-json")
        corrupt_plan = _run(
            driver, ["--worker", str(run_dir), "0", "0"]
        )
        check(
            corrupt_plan.returncode != 0
            and "refusing to run without per-unit memory reservations"
            in corrupt_plan.stdout,
            "a corrupt schedule fails closed before any unit starts",
        )
        schedule_path.write_text(schedule_text)

        # Two array tasks working the same run at the same time.
        started = time.time()
        procs = [
            subprocess.Popen(
                [sys.executable, str(driver), "--worker", str(run_dir), "0", str(task)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=_subprocess_env(),
            )
            for task in (0, 1)
        ]
        outputs = [p.communicate()[0] for p in procs]
        codes = [p.returncode for p in procs]
        elapsed = time.time() - started
        if any(codes):
            check(False, f"a worker exited non-zero {codes}:\n{outputs[0]}\n{outputs[1]}")
            return
        check(True, f"both array tasks completed ({elapsed:.1f}s)")

        written = sum(out.count("written") for out in outputs)
        result_paths = sorted((run_dir / "results").rglob("*.grim"))
        results = [p.name for p in result_paths]
        check(len(results) == expected_units,
              f"results/ holds exactly {expected_units} grims (got {len(results)})")
        check(written == expected_units,
              f"each unit was solved exactly once across tasks (got {written})")
        check(all(int(out.split("wrote=")[1].split(",")[0]) > 0 for out in outputs),
              "both tasks did real work rather than one doing everything")

        # Attestations must verify, which is what makes a restart safe.
        hpc_common.require_hpc_run_provenance(manifest, "ghost.hpc.2d-run.v2")
        hpc_common.require_hpc_output_attestations(run_dir, manifest)
        check(True, "manifest provenance and every output attestation verify")

        sidecars = list((run_dir / "results").rglob("*.provenance.json"))
        check(not sidecars,
              f"results/ holds no sidecar files (found {len(sidecars)})")
        check(len(result_paths) == expected_units,
              "results/ holds exactly one file per unit")
        check(all(path.parent.name in {"FRD", "OPN"} for path in result_paths),
              "results preserve their FRD/OPN role for downstream tools")
        first_unit = manifest["units"][0]
        check("azimuths_deg" not in first_unit,
              "units do not repeat the shared angular grid")
        check(isinstance(manifest.get("azimuths_deg"), list),
              "the angular grid is recorded once at manifest level")
        from workflow_provenance import read_embedded_attestation
        sample = result_paths[0]
        embedded = read_embedded_attestation(str(sample))
        check(embedded.get("run_id") == manifest["run_id"],
              "each result carries its run binding inside the artifact")

        status = hpc_common.run_status(run_dir)
        check(status["pending"] == 0, "run_status reports the run complete")

        # Re-joining a finished run must skip, not redo.
        rerun = _run(driver, ["--worker", str(run_dir), "0", "0"])
        skipped = int(rerun.stdout.split("skipped=")[1].split(",")[0])
        check(rerun.returncode == 0 and skipped > 0,
              f"a task re-joining a finished run re-verifies and skips its share "
              f"({skipped} units)")
        check("wrote=0" in rerun.stdout, "no unit is solved twice on a restart")
        check("failed=0" in rerun.stdout,
              "every completed output still passes its attestation check")

        units = hpc_common.read_unit_grims(run_dir / "results")
        check(len(units) == expected_units, "every grim reads back cleanly")
        check(all(u["amp"].size for u in units),
              "complex amplitudes were preserved in every grim")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    test_balance()
    test_windows_lock_initialization()
    test_claims()
    test_memory_admission()
    test_memory_cpu_backfill()
    test_formulation_resource_plan()
    test_batched_exact_resource_plan()
    test_survey_mode_cost()
    test_memory_heavy_geometry()
    test_resource_detection()
    test_fingerprint_cache()
    test_no_task_starvation()
    test_end_to_end()
    print(f"\n{len(FAILURES)} failure(s)")
    for message in FAILURES:
        print(f"  {message}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
