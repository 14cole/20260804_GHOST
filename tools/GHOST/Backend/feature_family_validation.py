"""Parameter studies for four feature families, graded against supplied truth.

This creates study definitions, never synthetic full-wave reference fields.
Acceptance includes clean-body agreement, featured-total agreement, isolated
complex delta agreement, and two successive reference mesh refinements.
"""
from pathlib import Path
import argparse
import json
import math
from validate_feature_reconstruction import compare_feature_case, compare_grims, CASE_REQUIRED_PATHS
from workflow_provenance import sha256_file

FAMILIES = {
    "corner": ("included_angle_deg", (30., 60., 90.), "Two intersecting PEC groove runs, with a resolved sharp corner; compare to independent straight-run expansion."),
    "termination": ("run_length_wavelengths", (1., 3., 6.), "A finite straight PEC groove with both physical end walls; compare to a truncated line expansion."),
    "curved_seam": ("radius_wavelengths", (1., 3., 10.), "A circular groove on a smooth PEC host; compare to curved line expansion using the same local normals."),
    "nearby_pair": ("separation_wavelengths", (.1, .25, .5, 1.), "Two parallel finite PEC grooves solved together; compare to the coherent sum of independently characterized runs."),
}


def study_template():
    cases = []
    for family, (parameter, values, geometry) in FAMILIES.items():
        for index, value in enumerate(values, 1):
            identity = f"{family}-{index:02d}"
            paths = {key: f"{identity}/{key}.grim" for key in CASE_REQUIRED_PATHS}
            cases.append(dict(id=identity, family=family, parameter=parameter, value=value,
                              geometry=geometry, paths=paths,
                              reference_refinements={kind: [f"{identity}/{kind}_coarse.grim",f"{identity}/{kind}_medium.grim",paths[f"{kind}_truth"]] for kind in ("clean","featured")}))
    return dict(schema="ghost.validation.feature-family-study.v1",
                status="definition_only_no_reference_results",
                reference_solver="", host_material="PEC", frequency_ghz=10.,
                fixed_geometry={"groove_width_wavelengths":.03,"groove_depth_wavelengths":.03,
                                "run_length_wavelengths":6.,"host":"Converged finite PEC plate or a separately documented curved PEC host; use identical host geometry for clean and featured runs."},
                conventions="exp(+j omega t), coming-from radar directions, global phase origin; physical F, sigma=4*pi*|F|^2; VV/HH/VH on identical grids",
                cases=cases)


def validate_study(path, cancel_check=lambda: False):
    source = Path(path).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema") != "ghost.validation.feature-family-study.v1":
        raise ValueError("Unsupported feature-family study schema.")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("A family study needs at least one case.")
    rows, seen = [], set()
    for case in cases:
        if cancel_check():
            raise InterruptedError("Family study cancelled.")
        family, identity = str(case.get("family","")), str(case.get("id",""))
        if family not in FAMILIES or not identity or identity in seen:
            raise ValueError("Family cases need unique IDs and a supported family.")
        seen.add(identity)
        parameter = FAMILIES[family][0]
        value = float(case.get("value",float("nan")))
        if case.get("parameter") != parameter or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{identity}: invalid {parameter}.")
        paths = {key:(source.parent/str(case["paths"][key])).resolve() for key in CASE_REQUIRED_PATHS}
        levels = {kind:[(source.parent/str(p)).resolve() for p in case["reference_refinements"][kind]] for kind in ("clean","featured")}
        for kind, files in levels.items():
            if len(files) != 3 or len(set(files)) != 3 or files[-1] != paths[f"{kind}_truth"]:
                raise ValueError(f"{identity}: {kind} reference needs three distinct mesh levels ending at its truth artifact.")
        all_paths=set(paths.values())|{p for files in levels.values() for p in files}
        missing=[str(p) for p in sorted(all_paths) if not p.is_file()]
        row=dict(id=identity,family=family,parameter=parameter,value=value,passed=False)
        if missing:
            row.update(status="awaiting_reference_artifacts",missing=missing)
        elif not str(document.get("reference_solver","")).strip():
            row.update(status="reference_solver_not_declared")
        else:
            convergence={}
            for kind, files in levels.items():
                convergence[kind]=[compare_grims(files[i+1],files[i],max_normalized_rms=.02,max_magnitude_p95_db=.5,max_phase_rms_deg=3.,min_coherence=.999) for i in (0,1)]
            result=compare_feature_case(**paths)
            row.update(status="passed" if result["passed"] and all(item["passed"] for steps in convergence.values() for item in steps) else "failed",
                       reference_convergence=convergence,comparison=result,
                       artifact_sha256={str(p):sha256_file(str(p)) for p in sorted(all_paths)})
            row["passed"]=row["status"]=="passed"
        rows.append(row)
    coverage={family:dict(parameter=spec[0],tested_values=[row["value"] for row in rows if row["family"]==family and row["passed"]],
                          requested_values=[row["value"] for row in rows if row["family"]==family]) for family,spec in FAMILIES.items()}
    return dict(schema="ghost.validation.feature-family-report.v1",passed=all(row["passed"] for row in rows),
                source=str(source),source_sha256=sha256_file(str(source)),reference_solver=str(document.get("reference_solver","")),
                cases=rows,coverage=coverage,
                interpretation="Pass applies only to tested geometry, material, and sampled radar grid. No interpolation envelope or mutual-coupling model is inferred. A failing nearby-pair case needs a joint characterized response or full-wave coupling.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-template")
    parser.add_argument("--study")
    parser.add_argument("--report")
    args=parser.parse_args(argv)
    if args.create_template:
        path=Path(args.create_template)
        with path.open("x",encoding="utf-8") as stream:
            json.dump(study_template(),stream,indent=2)
        return 0
    if not args.study or not args.report:
        parser.error("use --create-template FILE or --study FILE --report FILE")
    # Exclusive creation protects both the study and every existing source
    # artifact, including paths reached through Windows case aliases.
    report=validate_study(args.study)
    with Path(args.report).open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"{sum(case['passed'] for case in report['cases'])}/{len(report['cases'])} family cases passed; report: {args.report}")
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
