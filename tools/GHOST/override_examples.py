#!/usr/bin/env python3
"""Local/HPC override cookbook: 28 commented, independently usable recipes.

Run from this file's directory:
    python override_examples.py --list
    python override_examples.py --example fmm_pulse_low_ram
    python override_examples.py --all
    python override_examples.py --validate

Running this file prints settings or validates their syntax and compatibility.
It does not run the solver, submit jobs, edit drivers, or create scratch folders.

HOW TO USE A RECIPE
-------------------
1. Choose ONE recipe below, or print its block with --example NAME.
2. Replace the corresponding existing assignments in run_local_monostatic.py
   or run_hpc_monostatic.py. MAX_SOLVE_GB is farther down in each driver: replace
   that assignment too, rather than leaving a later assignment to overwrite it.
3. Leave the driver's legacy EXECUTION_OPTIONS = None when using these named
   presets. An explicit legacy profile takes precedence over these overrides.
4. If launching with --config or an adjacent .config.json, change its settings
   object instead: it overrides the script's CONFIG assignments. Use JSON null,
   true, and false in place of Python None, True, and False.
5. Keep your geometry paths, units, frequencies, angles, and output paths.
   Keep MESH_CERTIFICATION = True; choose ACCURACY_TARGET separately.

WHAT THE MAIN KNOBS MEAN
-----------------------
* SOLVE_PRESET = "auto" compares eligible dense, compressed, and FMM backends
  within your chosen discretization. Its stored factorization is "adaptive".
  ADVANCED_OVERRIDES = {"factorization": "auto"} is a DIFFERENT, legacy option:
  hierarchical factorization with dense fallback. See the last recipe.
* "balanced" uses streaming assembly and dense factorization; "large" uses
  compressed assembly; "small" uses the reference dense Galerkin kernel.
  All named presets use double precision. Explicit overrides change their
  performance settings; unspecified fields retain that preset's defaults.
* Pulse uses constant panel densities. Galerkin supports polynomial bases.
  Pulse is currently supported for 2D monostatic PEC/impedance bodies, bulk
  dielectrics, layered/coated bodies, and mixed bodies. Pulse thin sheets,
  thin-layer approximations, bistatic, BOR, and density export are unsupported.
  Pulse requires basis_order = 1; the inherited adaptive mesh setting becomes
  global panel refinement. Galerkin's default adaptive mesh may enrich its
  polynomial basis during certification.
* Pulse can reduce assembly work. Dense Pulse still stores a quadratic-size
  matrix; it does not inherently solve dense RAM scaling. FMM avoids the full
  dense matrix, but setup and iterative work can lose time on small problems.
  Explicit FMM requires the platform's native FMM library on the execution host.
  Automatic excludes FMM if unavailable. Copy the complete updated backend.
* MAX_SOLVE_GB is a per-solve admission budget in GiB, NOT a hard process limit
  or a request for node RAM. compressed_storage_mib caps retained numeric
  storage (including FMM near storage); workspaces need additional RAM. Lower
  caps can reject a run rather than compress it further. 1024 MiB = 1 GiB.
  Set the per-solve budget with MAX_SOLVE_GB here; do not also supply a conflicting
  ram_budget_gib override. All numeric budgets below are illustrative.
* assembly_threads builds interactions; blas_threads controls native matrix
  algebra. More threads cost workspaces and may contend for memory bandwidth.
  "auto" assembly threads follows the batch worker's CPU allocation.
* angle_batch_size bounds simultaneous incident-angle work, not the requested
  angle spacing or count. rhs_compression attempts a smaller independent set of
  incident fields and reconstructs all requested angles with numerical checks.
  FMM recycle vectors are a separate iterative-solver cache.
* Driver-only settings stay OUTSIDE ADVANCED_OVERRIDES: WORKERS for local runs;
  MAX_WORKERS_PER_NODE, CORES_PER_NODE, MEM_PER_NODE, N_NODES, and N_JOBS for HPC.
  Match these to the machine or SLURM allocation. A per-solve thread recipe does
  not change how many nodes or CPUs SLURM allocates.

COMPARE FAIRLY
--------------
Keep physical inputs and accuracy targets matched. Equal panel counts do not
imply equal Pulse/Galerkin accuracy, especially for TE and dielectric cases.
Compare certified wall time, peak RAM, and fields; extra Pulse refinement may
erase its assembly advantage. Tight linear/FMM tolerances do not replace mesh
convergence or accurate geometry. These are starting points, not speed promises.
"""

import argparse
from pathlib import Path
import sys
import textwrap


def recipe(why, tradeoff, overrides, preset="auto", ram=None):
    """Keep each recipe self-contained; no recipe inherits another recipe."""
    return {
        "why": why,
        "tradeoff": tradeoff,
        "settings": {
            "SOLVE_PRESET": preset,
            "ADVANCED_OVERRIDES": overrides,
            "MAX_SOLVE_GB": ram,
        },
    }


EXAMPLES = {
    # 01-02: General starting points. Automatic chooses the backend, not the basis.
    "automatic_galerkin": recipe(
        "Start here for a general Galerkin run, including adaptive polynomial meshing.",
        "Selection uses forecasts; benchmark your case before forcing a backend.",
        {"discretization": "galerkin"},
    ),
    "automatic_pulse": recipe(
        "Try Pulse while letting the scheduler choose an eligible backend.",
        "Check certified accuracy: some problems need substantially more Pulse panels.",
        {"discretization": "pulse"},
    ),

    # 03-04: Modest per-solve budgets and smaller temporary workspaces.
    "laptop_galerkin": recipe(
        "Reduce per-solve thread and angle workspaces on a smaller workstation.",
        "The 4 GiB budget can reject oversized cases; lower WORKERS separately if needed.",
        {"discretization": "galerkin", "assembly_threads": 2, "blas_threads": 1,
         "angle_batch_size": 64, "compressed_storage_mib": 512},
        ram=4,
    ),
    "laptop_pulse": recipe(
        "Try Pulse with the same modest thread, batching, and storage allowances.",
        "Automatic may still choose dense when it fits; Pulse alone does not bound RAM.",
        {"discretization": "pulse", "assembly_threads": 2, "blas_threads": 1,
         "angle_batch_size": 64, "compressed_storage_mib": 512},
        ram=4,
    ),

    # 05-08: Trade parallel solves against threads devoted to each solve.
    "parallel_sweep_galerkin": recipe(
        "Run many independent geometry/frequency units without each spawning many threads.",
        "Single-unit latency may rise. Set local/HPC worker counts for aggregate throughput.",
        {"discretization": "galerkin", "assembly_threads": 1, "blas_threads": 1,
         "angle_batch_size": 64, "compressed_storage_mib": 1024},
        ram=8,
    ),
    "parallel_sweep_pulse": recipe(
        "Use the same many-worker strategy with Pulse.",
        "Budget for all concurrent workers; 8 GiB here is per solve, not per node.",
        {"discretization": "pulse", "assembly_threads": 1, "blas_threads": 1,
         "angle_batch_size": 64, "compressed_storage_mib": 1024},
        ram=8,
    ),
    "hpc_allocated_threads": recipe(
        "Let each batch worker use its allocated assembly concurrency on varying nodes.",
        "This follows the scheduler allocation; it does not request extra SLURM CPUs.",
        {"discretization": "pulse", "assembly_threads": "auto", "blas_threads": 1,
         "angle_batch_size": 128, "compressed_storage_mib": 2048},
        ram=16,
    ),
    "workstation_dense": recipe(
        "Try more assembly and BLAS threads for a few substantial dense solves that fit RAM.",
        "Use a suitable CPU allocation and few concurrent workers; more threads may be slower.",
        {"discretization": "galerkin", "factorization": "dense", "mesh_strategy": "global",
         "assembly_threads": 8, "blas_threads": 4, "angle_batch_size": 128},
        preset="balanced", ram=32,
    ),

    # 09-10: Explicit linear/constant dense baselines with illumination reuse off.
    "reference_galerkin": recipe(
        "Establish a linear Galerkin dense reference for backend comparisons.",
        "Reference assembly and independent angle solves can cost time and dense RAM.",
        {"discretization": "galerkin", "factorization": "dense", "mesh_strategy": "global",
         "basis_order": 1, "rhs_compression": "off", "assembly_threads": 1, "blas_threads": 1},
        preset="small", ram=16,
    ),
    "reference_pulse": recipe(
        "Establish a Pulse dense baseline before comparing Pulse compressed or FMM results.",
        "This is Pulse's own discretization; agreement with Galerkin requires a convergence study.",
        {"discretization": "pulse", "factorization": "dense", "mesh_strategy": "global",
         "basis_order": 1, "rhs_compression": "off", "assembly_threads": 1, "blas_threads": 1},
        preset="balanced", ram=16,
    ),

    # 11-12: Explicit compressed assembly, retaining a bounded numeric payload.
    "compressed_galerkin": recipe(
        "Try compressed Galerkin when dense matrix storage is the limiting cost.",
        "Compression/inverse work can be slower; 4096 MiB is payload, not total RAM.",
        {"discretization": "galerkin", "factorization": "compressed",
         "compressed_storage_mib": 4096, "assembly_threads": 4, "blas_threads": 2,
         "rhs_compression": "auto", "angle_batch_size": 128},
        preset="large", ram=16,
    ),
    "compressed_pulse": recipe(
        "Combine Pulse assembly with compressed storage when a dense Pulse matrix is too large.",
        "A poorly compressible case can hit the cap; a lower cap does not force better compression.",
        {"discretization": "pulse", "factorization": "compressed",
         "compressed_storage_mib": 4096, "assembly_threads": 4, "blas_threads": 2,
         "rhs_compression": "auto", "angle_batch_size": 128},
        preset="large", ram=16,
    ),

    # 13-14: Explicit FMM, with the default qualified kernel/linear tolerances.
    "fmm_galerkin": recipe(
        "Use matrix-free FMM for a large linear Galerkin system with costly dense storage.",
        "Native FMM is required; setup and iterations may dominate smaller systems.",
        {"discretization": "galerkin", "factorization": "fmm", "mesh_strategy": "global",
         "basis_order": 1, "assembly_threads": 4, "blas_threads": 1,
         "fmm_pec_cfie": "auto", "fmm_recycle_vectors": "auto", "angle_batch_size": 128},
        ram=8,
    ),
    "fmm_pulse": recipe(
        "Combine Pulse with FMM to avoid a full dense matrix on a large panel mesh.",
        "Measure total certified time; Pulse refinement and iterative work still matter.",
        {"discretization": "pulse", "factorization": "fmm", "assembly_threads": 4,
         "blas_threads": 1, "pulse_pec_cfie": True,
         "fmm_recycle_vectors": "auto", "angle_batch_size": 128},
        ram=8,
    ),

    # 15-16: Reduce Krylov (iterative-search) and angle workspaces, not tolerances.
    "fmm_galerkin_low_ram": recipe(
        "Reduce linear Galerkin FMM working memory when the normal FMM recipe is too large.",
        "A shorter search space and disabled reuse can require more iterations/angle solves.",
        {"discretization": "galerkin", "factorization": "fmm", "mesh_strategy": "global",
         "basis_order": 1, "assembly_threads": 2, "blas_threads": 1,
         "compressed_storage_mib": 512, "angle_batch_size": 16, "rhs_compression": "off",
         "fmm_restart": 40, "fmm_recycle_vectors": 0},
        ram=4,
    ),
    "fmm_pulse_low_ram": recipe(
        "Try a smaller Pulse FMM workspace while retaining the normal accuracy tolerances.",
        "The near representation must still fit its cap; slower convergence can erase time savings.",
        {"discretization": "pulse", "factorization": "fmm", "assembly_threads": 2,
         "blas_threads": 1, "compressed_storage_mib": 512, "angle_batch_size": 16,
         "rhs_compression": "off", "fmm_restart": 40, "fmm_recycle_vectors": 0},
        ram=4,
    ),

    # 17-20: Incident-angle reuse and batching are separate from mesh/basis choice.
    "fmm_pulse_long_sweep": recipe(
        "Attempt more illumination reuse for a long, related incident-angle sweep.",
        "Larger batches retain more workspace; reuse is guarded and may not help every case.",
        {"discretization": "pulse", "factorization": "fmm", "rhs_compression": "on",
         "angle_batch_size": 256, "fmm_recycle_vectors": "auto",
         "assembly_threads": 4, "blas_threads": 1},
        ram=16,
    ),
    "fmm_material_recycling": recipe(
        "Tune iterative augmentation for a material Galerkin case after observing costly iterations.",
        "Twelve saved vectors cost RAM and can hurt PEC performance; auto is the starting point.",
        {"discretization": "galerkin", "factorization": "fmm", "mesh_strategy": "global",
         "fmm_recycle_vectors": 12, "fmm_restart": 100, "rhs_compression": "on",
         "angle_batch_size": 128, "assembly_threads": 4, "blas_threads": 1},
        ram=16,
    ),
    "dense_long_sweep": recipe(
        "Keep a reusable dense factor and attempt illumination compression for many angles.",
        "The matrix must fit RAM; large batches trade workspace for fewer batch overheads.",
        {"discretization": "galerkin", "factorization": "dense", "rhs_compression": "on",
         "angle_batch_size": 256, "assembly_threads": 4, "blas_threads": 2},
        preset="balanced", ram=32,
    ),
    "compressed_small_angle_batches": recipe(
        "Reduce angular workspaces in a compressed Pulse sweep without changing output angles.",
        "Smaller batches add overhead and can reduce illumination-reuse opportunities.",
        {"discretization": "pulse", "factorization": "compressed", "rhs_compression": "auto",
         "angle_batch_size": 32, "compressed_storage_mib": 1024,
         "assembly_threads": 2, "blas_threads": 1},
        preset="large", ram=8,
    ),

    # 21-23: Mesh and density-space changes need physical convergence checks.
    "local_material_mesh": recipe(
        "Try local wavelength sizing for separated regions with different material demands.",
        "Little benefit is expected for all-PEC cases; failed certification can retry global sizing.",
        {"discretization": "galerkin", "mesh_strategy": "local", "basis_order": 1,
         "assembly_threads": 4, "blas_threads": 2},
        ram=16,
    ),
    "quadratic_galerkin": recipe(
        "Try quadratic panel densities when a smoother solution may need fewer geometric panels.",
        "Higher degree adds unknowns and integration work per panel; Pulse cannot use this setting.",
        {"discretization": "galerkin", "mesh_strategy": "global", "basis_order": 2,
         "factorization": "dense", "assembly_threads": 4, "blas_threads": 2},
        preset="balanced", ram=16,
    ),
    "cubic_fmm_galerkin": recipe(
        "Explore cubic Galerkin densities with FMM when higher-order dense storage is excessive.",
        "Higher-order quadrature and more unknowns cost RAM/time; compare at matched accuracy.",
        {"discretization": "galerkin", "mesh_strategy": "global", "basis_order": 3,
         "factorization": "fmm", "fmm_quadrature_order": 8,
         "assembly_threads": 4, "blas_threads": 1},
        ram=16,
    ),

    # 24-25: Accuracy diagnostics. Tightening these is not a speed optimization.
    "fmm_strict_tolerances": recipe(
        "Check whether FMM/linear-solve error contributes to a small observed field difference.",
        "More accurate kernels and solves cost time; they cannot fix a coarse physical mesh.",
        {"discretization": "galerkin", "factorization": "fmm", "mesh_strategy": "global",
         "fmm_tolerance": 1e-12, "fmm_solver_tolerance": 1e-11,
         "fmm_quadrature_order": 12, "assembly_threads": 4, "blas_threads": 1},
        ram=16,
    ),
    "fmm_pulse_quadrature_check": recipe(
        "Compare automatic Pulse FMM quadrature with an explicit minimum of eight source points.",
        "The actual order may rise for long electrical panels; extra points add work and RAM.",
        {"discretization": "pulse", "factorization": "fmm", "fmm_quadrature_order": 8,
         "fmm_tolerance": 1e-10, "fmm_solver_tolerance": 1e-9,
         "assembly_threads": 4, "blas_threads": 1},
        ram=8,
    ),

    # 26-27: Replace these placeholder paths with EXISTING execution-host folders.
    "compressed_windows_scratch": recipe(
        "Put compressed polarization spool files on an available local SSD scratch directory.",
        "Replace the path and create it yourself; disk spooling does not eliminate RAM needs.",
        {"discretization": "pulse", "factorization": "compressed",
         "temporary_directory": "D:/ghost_scratch", "compressed_storage_mib": 2048,
         "angle_batch_size": 128, "assembly_threads": 4, "blas_threads": 2},
        preset="large", ram=8,
    ),
    "compressed_hpc_scratch": recipe(
        "Use an existing node-local scratch directory to avoid a busy shared filesystem for spools.",
        "Replace the path on every execution node; environment-variable text is not expanded here.",
        {"discretization": "galerkin", "factorization": "compressed",
         "temporary_directory": "/scratch/your_user/ghost", "compressed_storage_mib": 4096,
         "angle_batch_size": 128, "assembly_threads": "auto", "blas_threads": 1},
        preset="large", ram=16,
    ),

    # 28: Distinguish legacy factorization "auto" from the modern "auto" preset.
    "legacy_hierarchical_fallback": recipe(
        "Reproduce a legacy hierarchical-factorization experiment with dense fallback allowed.",
        "This can allocate a dense matrix; it is unsuitable when dense fallback cannot fit RAM.",
        {"discretization": "galerkin", "factorization": "auto", "mesh_strategy": "global",
         "assembly_threads": 4, "blas_threads": 2},
        preset="balanced", ram=32,
    ),
}


def print_example(name):
    entry = EXAMPLES[name]
    print("# " + name)
    for label, field in (("Use when: ", "why"), ("Tradeoff: ", "tradeoff")):
        for line in textwrap.wrap(label + entry[field], width=86):
            print("# " + line)
    settings = entry["settings"]
    print("SOLVE_PRESET = " + repr(settings["SOLVE_PRESET"]))
    overrides = settings["ADVANCED_OVERRIDES"]
    if overrides:
        print("ADVANCED_OVERRIDES = {")
        for field, value in overrides.items():
            print("    {!r}: {!r},".format(field, value))
        print("}")
    else:
        print("ADVANCED_OVERRIDES = {}")
    print("MAX_SOLVE_GB = " + repr(settings["MAX_SOLVE_GB"]))


def validate_examples():
    # This imports only configuration code. It needs no geometry or native FMM.
    # Path syntax is checked; scratch existence and solve capacity are not.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ghost_backend.runs.execution import driver_options, reconcile_settings
    from ghost_backend.runs.presets import resolve_preset

    for name, entry in EXAMPLES.items():
        settings = entry["settings"]
        resolved = resolve_preset(settings)
        profile = resolved["EXECUTION_OPTIONS"]
        if driver_options(settings) != profile:
            raise ValueError(name + ": direct-script and preset settings disagree")
        if reconcile_settings(settings)["EXECUTION_OPTIONS"] != profile:
            raise ValueError(name + ": JSON configuration and preset settings disagree")
    print("Validated {} recipes through preset, driver, and JSON settings resolution.".format(len(EXAMPLES)))
    print("Configuration check only: no solves, native-library checks, or scratch-directory checks.")


def main():
    parser = argparse.ArgumentParser(
        description="Print local/HPC 2D override recipes. No solves or job submissions.",
        epilog="Copy one recipe into the existing driver assignments; keep mesh certification enabled.",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--list", action="store_true", help="list recipe names and purposes (default)")
    display.add_argument("--example", choices=tuple(EXAMPLES), metavar="NAME", help="print one copyable Python block")
    display.add_argument("--all", action="store_true", help="print every recipe and its tradeoff")
    parser.add_argument("--validate", action="store_true", help="validate all recipes with the adjacent backend")
    args = parser.parse_args()
    if args.validate:
        validate_examples()
        if not (args.list or args.example or args.all):
            return
    if args.example:
        print_example(args.example)
    elif args.all:
        for index, name in enumerate(EXAMPLES):
            if index:
                print()
            print_example(name)
    else:
        for name, entry in EXAMPLES.items():
            print(name + "\n    " + entry["why"])
        print("\nPrint a complete block: python override_examples.py --example NAME")


if __name__ == "__main__":
    main()
