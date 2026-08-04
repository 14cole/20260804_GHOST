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
import time
from pathlib import Path

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


# ─── unit-level checks ──────────────────────────────────────────────────────

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


def test_claims():
    print("\nclaim broker")
    with tempfile.TemporaryDirectory() as tmp:
        broker = hpc_scheduler.ClaimBroker(Path(tmp) / "claims", stale_seconds=1.0)
        check(broker.try_claim("unit-a"), "first claim succeeds")
        check(not broker.try_claim("unit-a"), "second claim on the same unit fails")

        other = hpc_scheduler.ClaimBroker(Path(tmp) / "claims", stale_seconds=1.0)
        check(not other.try_claim("unit-a"), "another task cannot take a live claim")

        stale_path = Path(tmp) / "claims" / "unit-a.claim"
        old = time.time() - 120.0
        os.utime(stale_path, (old, old))
        check(other.try_claim("unit-a"), "a stale claim is stealable")

        broker.abandon("unit-b")  # must not raise on an unheld key
        check(broker.try_claim("unit-b"), "claim after abandon succeeds")
        broker.abandon("unit-b")
        check(broker.try_claim("unit-b"), "abandon releases the unit again")


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


# ─── end-to-end sweep ───────────────────────────────────────────────────────

DRIVER_SETTINGS = {
    "FREQUENCIES_GHZ": [2.0, 3.0, 4.0],
    "AZIMUTHS_DEG": [0.0, 45.0, 90.0],
    "POLARIZATIONS": ["TM"],
    "GEOMETRY_UNITS": "meters",
    "MAX_PANELS": 20000,
    "N_NODES": 2,
    "N_JOBS": 1,
    "SUBMIT": False,
    "TASKS_PER_CHILD": 2,
    "MAX_WORKERS_PER_NODE": 2,
    "CLAIM_STALE_SECONDS": 60,
}


def _stage_run(root):
    """Configure a private copy of the driver, as hpc_common intends."""

    geom_root = root / "geometries"
    sources = [REPO / "geometries" / "body.geo"]
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


def _run(driver, args, timeout=1800):
    return subprocess.run(
        [sys.executable, str(driver)] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(BACKEND)},
    )


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
        expected_units = 2 * 3 * 1  # geometries x frequencies x polarizations
        check(manifest["n_units"] == expected_units,
              f"manifest holds {expected_units} units")
        check((run_dir / "schedule.json").is_file(), "schedule.json was written")
        schedule = json.loads((run_dir / "schedule.json").read_text())
        check(all(int(r["nodes"]) > 0 for r in schedule["units"]),
              "every unit was pre-meshed for its cost estimate")
        check(all(float(r["peak_gb"]) > 0 for r in schedule["units"]),
              "every unit carries a memory estimate")
        costs = {r["unit"]: r["cost"] for r in schedule["units"]}
        by_freq = {}
        for name, cost in costs.items():
            by_freq.setdefault(name.split("_")[1], []).append(cost)
        check(max(map(max, by_freq.values())) > 2.0 * min(map(min, by_freq.values())),
              "the cost model separates cheap and expensive frequencies")
        check(len(list(run_dir.glob("submit_job*.slurm"))) == 1,
              "one sbatch script per submission")
        slurm_text = (run_dir / "submit_job0.slurm").read_text()
        check("--array=0-1" in slurm_text, "array size matches N_NODES")
        check("--requeue" in slurm_text, "array tasks are requeueable")
        check("OMP_NUM_THREADS=1" in slurm_text,
              "BLAS threads are pinned before the interpreter starts")

        # Two array tasks working the same run at the same time.
        started = time.time()
        procs = [
            subprocess.Popen(
                [sys.executable, str(driver), "--worker", str(run_dir), "0", str(task)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True,
                env={**os.environ, "PYTHONPATH": str(BACKEND)},
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
        results = sorted(p.name for p in (run_dir / "results").glob("*.grim"))
        check(len(results) == expected_units,
              f"results/ holds exactly {expected_units} grims (got {len(results)})")
        check(written == expected_units,
              f"each unit was solved exactly once across tasks (got {written})")
        check(all(int(out.split("wrote=")[1].split(",")[0]) > 0 for out in outputs),
              "both tasks did real work rather than one doing everything")

        # Attestations must verify, which is what makes a restart safe.
        hpc_common.require_hpc_run_provenance(manifest, "ghost.hpc.2d-run.v1")
        hpc_common.require_hpc_output_attestations(run_dir, manifest)
        check(True, "manifest provenance and every output attestation verify")

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
    test_claims()
    test_memory_admission()
    test_survey_mode_cost()
    test_resource_detection()
    test_fingerprint_cache()
    test_end_to_end()
    print(f"\n{len(FAILURES)} failure(s)")
    for message in FAILURES:
        print(f"  {message}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
