#!/usr/bin/env python3
"""
Numerical equivalence check for the optimized boundary-operator assembly.

The tiled far-field engine and the batched hypersingular assembly reorganize
how the element-pair quadrature is staged; they must not change what it
computes.  This compares the working solver against a pristine copy of the
pre-optimization module, operator by operator, on real meshes.

Case sizes are chosen around the *reference* cost: its hypersingular assembly
is an O(N^2) interpreted loop, so those cases stay small, while the vectorized
S/K comparison can afford larger meshes.

Usage:
    python tests/test_assembly_equivalence.py /path/to/rcs_solver_reference.py
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
sys.path.insert(0, str(BACKEND))

TOL = 5.0e-11


def load_reference(path):
    spec = importlib.util.spec_from_file_location("rcs_solver_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rcs_solver_reference"] = module
    spec.loader.exec_module(module)
    return module


def rel_err(new, ref):
    denom = float(np.max(np.abs(ref)))
    if denom <= 0.0:
        denom = 1.0
    return float(np.max(np.abs(new - ref))) / denom


def build_mesh(module, geo_path, freq_ghz, pol, units):
    from geometry_io import parse_geometry, build_geometry_snapshot

    title, segments, ibcs, dielectrics = parse_geometry(Path(geo_path).read_text())
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    materials = module.MaterialLibrary.from_entries(
        snapshot.get("ibcs", []) or [],
        snapshot.get("dielectrics", []) or [],
        base_dir=str(Path(geo_path).parent),
    )
    scale = module._unit_scale_to_meters(units)
    k0 = 2.0 * math.pi * freq_ghz * 1e9 / module.C0
    lambda_min, _, _ = module._mesh_wavelength_for_snapshot(
        snapshot, materials, freq_ghz
    )
    panels = module._build_panels(snapshot, scale, lambda_min, max_panels=50_000)
    infos = module._build_coupled_panel_info(panels, materials, freq_ghz, pol, k0)
    mesh, _ = module._build_linear_mesh_interface_aware(panels, infos)
    return mesh, k0


# (geometry, frequency GHz, polarization, units, run hypersingular too)
#
# The hypersingular flag is set only on small meshes: the *reference* builds
# that operator with an O(N^2) interpreted loop over element pairs, so a
# few-hundred-element case takes tens of minutes on the reference side alone.
# The optimized path is exercised at realistic sizes by benchmark_assembly.py.
CASES = [
    ("geometries/body.geo", 3.0, "TM", "meters", True),
    ("geometries/body.geo", 8.0, "TM", "meters", True),
    ("geometries/body.geo", 12.0, "TE", "meters", False),
    ("geometries/body.geo", 30.0, "TM", "meters", False),
    ("1b_solve_2d_hpc/geometries/FRD/SEAL-00-01_0.006gap.geo", 2.0, "TM", "meters", False),
    ("1b_solve_2d_hpc/geometries/OPN/SEAL-00-01_0.010gap.geo", 6.0, "TE", "meters", False),
]

# (source mask?, lossy wavenumber?, obs_normal_deriv, want_S, want_K)
OPERATOR_VARIANTS = [
    (False, False, True, True, True),
    (False, False, False, True, True),
    (False, False, True, True, False),
    (False, False, False, False, True),
    (True, False, True, True, True),
    (True, False, False, True, True),
    (False, True, True, True, True),
    (False, True, False, True, True),
    (True, True, False, True, True),
]


def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not ref_path or not Path(ref_path).is_file():
        raise SystemExit("usage: test_assembly_equivalence.py <reference rcs_solver.py>")

    import rcs_solver as new

    ref = load_reference(ref_path)
    print(f"reference: {ref_path}")
    print(f"assembly threads: {new.get_assembly_threads()}\n")

    failures = []
    checks = 0
    for rel_geo, freq, pol, units, do_hyper in CASES:
        geo = REPO / rel_geo
        if not geo.is_file():
            print(f"  [skip] {rel_geo} not present")
            continue
        mesh_new, k0 = build_mesh(new, geo, freq, pol, units)
        mesh_ref, k0_ref = build_mesh(ref, geo, freq, pol, units)
        assert len(mesh_new.elements) == len(mesh_ref.elements)
        assert abs(k0 - k0_ref) <= 1e-12 * abs(k0)
        nelems = len(mesh_new.elements)
        print(f"{rel_geo} @ {freq} GHz {pol}: {nelems} elements", flush=True)

        rng = np.random.default_rng(20260804)
        sub_mask = rng.random(nelems) < 0.6
        lossy = complex(k0, -0.05 * k0)

        for use_mask, use_lossy, obs_dn, want_s, want_k in OPERATOR_VARIANTS:
            mask = sub_mask if use_mask else None
            kval = lossy if use_lossy else k0
            s_new, k_new = new._assemble_linear_operator_matrices(
                mesh_new, kval, obs_normal_deriv=obs_dn,
                source_element_mask=mask,
                compute_single_layer=want_s, compute_double_layer=want_k,
            )
            s_ref, k_ref = ref._assemble_linear_operator_matrices(
                mesh_ref, kval, obs_normal_deriv=obs_dn,
                source_element_mask=mask,
                compute_single_layer=want_s, compute_double_layer=want_k,
            )
            checks += 1
            variant = (f"mask={'sub' if use_mask else 'all'} "
                       f"k={'lossy' if use_lossy else 'real'} "
                       f"obs_dn={int(obs_dn)} S={int(want_s)} K={int(want_k)}")
            for label, a, b in (("S", s_new, s_ref), ("K", k_new, k_ref)):
                err = rel_err(a, b)
                tag = f"  {variant} {label}"
                if not math.isfinite(err) or err > TOL:
                    failures.append((f"{rel_geo} {variant} {label}", err))
                    print(f"  FAIL{tag}: {err:.3e}", flush=True)
                else:
                    print(f"  ok  {tag}: {err:.3e}", flush=True)

        if do_hyper:
            for use_mask in (False, True):
                mask = sub_mask if use_mask else None
                d_new = new._assemble_linear_hypersingular_matrix(
                    mesh_new, k0, source_element_mask=mask
                )
                d_ref = ref._assemble_linear_hypersingular_matrix(
                    mesh_ref, k0, source_element_mask=mask
                )
                checks += 1
                err = rel_err(d_new, d_ref)
                tag = f"  mask={'sub' if use_mask else 'all'} D"
                if not math.isfinite(err) or err > TOL:
                    failures.append((f"{rel_geo} hypersingular", err))
                    print(f"  FAIL{tag}: {err:.3e}", flush=True)
                else:
                    print(f"  ok  {tag}: {err:.3e}", flush=True)
        print(flush=True)

    print(f"{checks} comparisons, {len(failures)} failure(s), tolerance {TOL:g}")
    for tag, err in failures:
        print(f"  {tag}: {err:.3e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
