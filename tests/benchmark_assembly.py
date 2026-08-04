#!/usr/bin/env python3
"""
Benchmark the boundary-operator assembly: time, peak memory, thread scaling.

Assembly is where a large 2-D solve spends its time -- the linear solve is
O(N^3) but runs on threaded BLAS, while assembly is O(N^2 Q^2) Hankel
evaluations driven from Python. It is also what decides how many solves fit on
a node at once, which on a 96-core box matters as much as the speed.

Pass a pristine pre-optimization rcs_solver.py to get a side-by-side; without
one, only the current solver is measured.

Usage:
    python tests/benchmark_assembly.py [reference_rcs_solver.py]
"""

import gc
import importlib.util
import math
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
sys.path.insert(0, str(BACKEND))

GEOMETRY = REPO / "geometries" / "body.geo"
FREQUENCIES = [10.0, 20.0, 40.0, 80.0]


def load_reference(path):
    spec = importlib.util.spec_from_file_location("rcs_solver_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rcs_solver_reference"] = module
    spec.loader.exec_module(module)
    return module


def build_mesh(module, freq, pol="TM"):
    from geometry_io import parse_geometry, build_geometry_snapshot

    title, segments, ibcs, dielectrics = parse_geometry(GEOMETRY.read_text())
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    materials = module.MaterialLibrary.from_entries(
        [], [], base_dir=str(GEOMETRY.parent)
    )
    k0 = 2.0 * math.pi * freq * 1e9 / module.C0
    lambda_min, _, _ = module._mesh_wavelength_for_snapshot(snapshot, materials, freq)
    panels = module._build_panels(snapshot, 1.0, lambda_min, max_panels=50_000)
    infos = module._build_coupled_panel_info(panels, materials, freq, pol, k0)
    mesh, _ = module._build_linear_mesh_interface_aware(panels, infos)
    return mesh, k0


def timed(fn):
    gc.collect()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.time()
    out = fn()
    elapsed = time.time() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    del out
    gc.collect()
    return elapsed, max(0, peak - before) / 1024.0


def main():
    import rcs_solver as new

    ref_path = sys.argv[1] if len(sys.argv) > 1 else None
    ref = load_reference(ref_path) if ref_path and Path(ref_path).is_file() else None

    print("Single-/double-layer assembly (body.geo, TM, one operator pair)\n")
    header = f"{'N':>6} {'operators':>10} {'new s':>9} {'new MB':>8}"
    if ref is not None:
        header += f" {'ref s':>9} {'ref MB':>8} {'speedup':>8} {'mem saved':>10}"
    print(header)
    for freq in FREQUENCIES:
        mesh, k0 = build_mesh(new, freq)
        nelems = len(mesh.elements)
        for want_k, label in ((False, "S"), (True, "S+K")):
            new_time, new_mb = timed(lambda: new._assemble_linear_operator_matrices(
                mesh, k0, obs_normal_deriv=True, compute_double_layer=want_k))
            row = f"{nelems:6d} {label:>10} {new_time:9.2f} {new_mb:8.1f}"
            if ref is not None:
                ref_mesh, ref_k0 = build_mesh(ref, freq)
                ref_time, ref_mb = timed(
                    lambda: ref._assemble_linear_operator_matrices(
                        ref_mesh, ref_k0, obs_normal_deriv=True,
                        compute_double_layer=want_k))
                row += (f" {ref_time:9.2f} {ref_mb:8.1f} "
                        f"{ref_time / max(new_time, 1e-9):7.2f}x "
                        f"{max(0.0, ref_mb - new_mb):9.1f}M")
            print(row, flush=True)

    print("\nHypersingular assembly (Maue identity, TE sheet / dielectric paths)\n")
    print(f"{'N':>6} {'new s':>9}" + (f" {'ref s':>10} {'speedup':>8}" if ref else ""))
    for freq in FREQUENCIES[:3]:
        mesh, k0 = build_mesh(new, freq, pol="TE")
        nelems = len(mesh.elements)
        new_time, _ = timed(
            lambda: new._assemble_linear_hypersingular_matrix(mesh, k0))
        row = f"{nelems:6d} {new_time:9.2f}"
        if ref is not None and nelems <= 250:
            ref_mesh, ref_k0 = build_mesh(ref, freq, pol="TE")
            ref_time, _ = timed(
                lambda: ref._assemble_linear_hypersingular_matrix(ref_mesh, ref_k0))
            row += f" {ref_time:10.2f} {ref_time / max(new_time, 1e-9):7.0f}x"
        elif ref is not None:
            row += f" {'(skipped)':>10} {'':>8}"
        print(row, flush=True)

    cores = len(getattr(__import__("os"), "sched_getaffinity", lambda _: range(1))(0))
    print(f"\nAssembly thread scaling (this host reports {cores} usable cores)\n")
    print(f"{'N':>6} {'threads':>8} {'s':>9} {'speedup':>8}")
    mesh, k0 = build_mesh(new, FREQUENCIES[-1])
    nelems = len(mesh.elements)
    baseline = None
    for threads in (1, 2, 4, max(4, cores)):
        if threads > max(4, cores):
            continue
        new.set_assembly_threads(threads)
        elapsed, _ = timed(lambda: new._assemble_linear_operator_matrices(
            mesh, k0, obs_normal_deriv=True, compute_double_layer=False))
        baseline = baseline or elapsed
        print(f"{nelems:6d} {threads:8d} {elapsed:9.2f} "
              f"{baseline / max(elapsed, 1e-9):7.2f}x", flush=True)
    new.set_assembly_threads(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
