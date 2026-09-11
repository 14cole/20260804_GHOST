"""Feature library identities, applicability checks, and contracts."""
from typing import Any, Callable, Mapping, Optional, Sequence

def _apply_feature_library_contracts(
    *,
    line_placements: 'Sequence[Mapping[str, Any]]',
    line_records: 'Sequence[dict[str, Any]]',
    point_placements: 'Sequence[Mapping[str, Any]]',
    point_records: 'Sequence[dict[str, Any]]',
    radar_grid: 'Mapping[str, Any]',
    require_manifests: 'bool',
    cancel_check: 'Optional[Callable[[], bool]]' = None,
    host_material: 'str' = "",
    host_stack_id: 'str' = "",
    host_minimum_radius_m: 'Optional[float]' = None,
) -> 'tuple[dict[str, Any], list[str], dict[str, str], set[str]]':
    """Bind manifests, applicability gates, and component identities to a plan."""
    from ghost_backend.assembly.workflow import (
        Any,
        FEATURE_LIBRARY_MANIFEST_SCHEMA,
        Optional,
        _canonical_point_roll,
        _component_clearance,
        _component_signature,
        _footprint_candidate_pairs,
        _line_applicability_metrics,
        _line_self_footprint_overlap,
        _load_grim,
        _prepared_line_response_physics_sha256,
        _prepared_point_response_physics_sha256,
        _validate_point_requested_support,
        _vehicle_radar_directions,
        feature_response_content_sha256,
        load_feature_library_manifest,
        load_seam_from_grim,
        math,
        np,
        validate_installed_host,
    )

    frequencies = np.asarray(radar_grid["frequencies_ghz"], dtype=float)
    directions = _vehicle_radar_directions(radar_grid)
    contracts: 'dict[str, Any]' = {}
    warnings: 'list[str]' = []
    source_hashes: 'dict[str, str]' = {}
    absent_source_paths: 'set[str]' = set()
    seen_components: 'dict[str, tuple[str, str]]' = {}
    footprint_components: 'list[dict[str, Any]]' = []
    groups = (
        ("line", line_placements, line_records),
        ("point", point_placements, point_records),
    )
    for feature_kind, placements, records in groups:
        if len(placements) != len(records):
            raise ValueError(
                f"Prepared {feature_kind} placement/record counts disagree."
            )
        manifests: 'dict[str, Optional[dict[str, Any]]]' = {}
        response_identities: 'dict[str, str]' = {}
        response_content_identities: 'dict[str, str]' = {}
        response_frequency_bounds: 'dict[str, tuple[float, float]]' = {}
        line_coefficients: 'dict[str, tuple[Any, ...]]' = {}
        for placement, record in zip(placements, records):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Feature placement validation cancelled.")
            dataset_id = str(record["dataset_id"])
            dataset_value = record.get("dataset")
            dataset_digest = record.get("dataset_sha256")
            required_geometry = (
                {"perimeter", "segment_normals"}
                if feature_kind == "line"
                else {"location", "aperture_normal", "roll_ref"}
            )
            if (
                dataset_value is None
                or dataset_digest is None
                or not required_geometry.issubset(placement)
            ):
                message = (
                    f"Injected/custom {feature_kind} dataset {dataset_id!r} "
                    "predates the production manifest/component-identity "
                    "contract."
                )
                if require_manifests:
                    raise ValueError(message)
                warnings.append(message)
                contracts[f"{feature_kind}:{dataset_id}"] = {
                    "status": "legacy_injected_placement",
                }
                continue
            dataset = str(dataset_value)
            contract_key = f"{feature_kind}:{dataset_id}"
            if dataset_id not in manifests:
                try:
                    manifest, sources = load_feature_library_manifest(
                        dataset, dataset_id=dataset_id, feature_kind=feature_kind
                    )
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    if require_manifests:
                        raise
                    manifest, sources = None, []
                    warnings.append(f"Metadata advisory for {feature_kind} {dataset_id!r}: {exc}; response samples remain usable.")
                advisory_manifest = manifest if not require_manifests else None
                if not require_manifests:
                    manifest = None
                for source in sources:
                    if source.get("absent") == "true" and "path" in source:
                        absent_source_paths.add(str(source["path"]))
                    elif "path" in source and "sha256" in source:
                        source_hashes[str(source["path"])] = str(
                            source["sha256"]
                        )
                manifests[dataset_id] = manifest
                if manifest is not None:
                    response_content_identities[dataset_id] = str(
                        manifest["response_content_sha256"]
                    )
                else:
                    try:
                        response_content_identities[dataset_id] = (
                            feature_response_content_sha256(dataset)
                        )
                    except ValueError:


                        response_content_identities[dataset_id] = str(
                            dataset_digest
                        )
                if manifest is None:
                    description = "has an advisory" if advisory_manifest is not None else "has no"
                    message = (
                        f"{feature_kind} dataset {dataset_id!r} {description} "
                        "feature-library manifest; phase/frame are accepted "
                        "through the selected dataset role; host/curvature/"
                        "footprint annotations are advisory."
                    )
                    if require_manifests:
                        raise ValueError(message)
                    warnings.append(message)
                    contracts[contract_key] = {
                        "status": "metadata_advisory",
                        "dataset": dataset,
                        "source_manifest": advisory_manifest,
                    }
                else:
                    if manifest["schema"] != FEATURE_LIBRARY_MANIFEST_SCHEMA:
                        message = (
                            f"{feature_kind} dataset {dataset_id!r} uses "
                            f"Legacy manifest schema {manifest['schema']!r}; "
                            f"Production requires {FEATURE_LIBRARY_MANIFEST_SCHEMA!r} "
                            "so passing full-wave cases, all four artifacts, "
                            "and the exact exercised response are bound. "
                            "Migrate it with the supported manifest tool."
                        )
                        if require_manifests:
                            raise ValueError(message)
                        warnings.append(message)
                    contracts[contract_key] = {
                        "status": manifest["validation"]["status"],
                        "dataset": dataset,
                        "manifest": manifest,
                        "sources": sources,
                    }
                    if manifest["validation"]["status"] != "validated":
                        message = (
                            f"{feature_kind} dataset {dataset_id!r} manifest "
                            f"is {manifest['validation']['status']}, not validated."
                        )
                        if require_manifests:
                            raise ValueError(message)
                        warnings.append(message)

                if feature_kind == "line":
                    response_payload = _load_grim(dataset)
                    response_frequencies = np.asarray(
                        response_payload["frequencies"], dtype=float
                    )
                    prepared_coefficients = []
                    for requested_frequency in frequencies:
                        prepared_coefficients.append(load_seam_from_grim(
                            dataset,
                            float(requested_frequency),
                            declared_coherent_delta=True,
                            delta_sign=1.0,
                            _grim_payload=response_payload,
                        ))
                    response_identities[dataset_id] = (
                        _prepared_line_response_physics_sha256(
                            prepared_coefficients
                        )
                    )
                    line_coefficients[dataset_id] = tuple(prepared_coefficients)
                    response_frequency_bounds[dataset_id] = (
                        float(np.min(response_frequencies)),
                        float(np.max(response_frequencies)),
                    )
                else:
                    response_identities[dataset_id] = (
                        _prepared_point_response_physics_sha256(
                            placement["pattern"]
                        )
                    )
                    point_frequencies = np.asarray(
                        placement["pattern"].frequencies, dtype=float
                    )
                    response_frequency_bounds[dataset_id] = (
                        float(np.min(point_frequencies)),
                        float(np.max(point_frequencies)),
                    )
            manifest = manifests[dataset_id]
            if feature_kind == "point":
                point_support = _validate_point_requested_support(
                    placement, directions, frequencies, dataset_id=dataset_id
                )
                record.update(point_support)
                if point_support["illuminated_requested_look_count"] == 0:
                    warnings.append(
                        f"Point {str(record['placement_id'])!r} has zero "
                        "illuminated requested looks, so its enabled response "
                        "contributes zero on this radar grid. Review its "
                        "outward normal/orientation and requested aperture. If "
                        "intentional, review and accept the existing one-time "
                        "release-warning waiver before Build."
                    )
            if manifest is not None:
                host_result = validate_installed_host(
                    manifest, material=host_material, stack_id=host_stack_id,
                    minimum_radius_m=host_minimum_radius_m,
                    required=require_manifests, label=f"{feature_kind} dataset {dataset_id!r}",
                )
                record["host_applicability"] = host_result
                for message in host_result["warnings"]:
                    if message not in warnings:
                        warnings.append(message)
                applicability = manifest["applicability"]
                frequency_range = applicability["frequency_ghz"]
                if (
                    float(np.min(frequencies)) < frequency_range["min"] - 1e-12
                    or float(np.max(frequencies)) > frequency_range["max"] + 1e-12
                ):
                    raise ValueError(
                        f"{feature_kind} dataset {dataset_id!r} is certified "
                        f"only for {frequency_range['min']:g}-"
                        f"{frequency_range['max']:g} GHz; the Assembly grid is "
                        f"{float(np.min(frequencies)):g}-"
                        f"{float(np.max(frequencies)):g} GHz."
                    )
                response_min, response_max = response_frequency_bounds[dataset_id]
                if (
                    frequency_range["min"] < response_min - 1.0e-12
                    or frequency_range["max"] > response_max + 1.0e-12
                ):
                    raise ValueError(
                        f"{feature_kind} dataset {dataset_id!r} manifest declares "
                        f"{frequency_range['min']:g}-{frequency_range['max']:g} "
                        "GHz applicability, outside the bound response data "
                        f"range {response_min:g}-{response_max:g} GHz."
                    )

            if feature_kind == "line":
                metrics = _line_applicability_metrics(
                    placement,
                    directions,
                    requested_frequencies_ghz=frequencies,
                    cancel_check=cancel_check,
                )
                installed_radius = float(metrics[
                    "estimated_min_along_line_normal_turn_radius_m"
                ])
                public_metrics = dict(metrics)
                if not math.isfinite(installed_radius):


                    public_metrics[
                        "estimated_min_along_line_normal_turn_radius_m"
                    ] = None
                record.update(public_metrics)
                if metrics["illuminated_requested_look_count"] == 0:
                    warnings.append(
                        f"Line {str(record['line_id'])!r} has zero illuminated "
                        "requested looks, so its enabled response contributes "
                        "zero on this radar grid. Review its endpoint normals "
                        "and requested aperture. If intentional, review and "
                        "accept the existing one-time release-warning waiver "
                        "before Build."
                    )
                for coefficient, cut_range in zip(
                    line_coefficients[dataset_id],
                    metrics["required_cut_angle_ranges_deg"],
                ):
                    cut_min = cut_range["minimum_deg"]
                    cut_max = cut_range["maximum_deg"]
                    if cut_min is None:
                        continue
                    support_min = float(coefficient.phi_deg[0])
                    support_max = float(coefficient.phi_deg[-1])
                    if (
                        float(cut_min) < support_min - 1.0e-9
                        or float(cut_max) > support_max + 1.0e-9
                    ):
                        raise ValueError(
                            f"line dataset {dataset_id!r} at "
                            f"{float(coefficient.frequency_ghz):g} GHz covers "
                            f"cut angles [{support_min:g}, {support_max:g}] deg, "
                            f"but installed line {record['line_id']!r} needs "
                            f"[{float(cut_min):.6g}, {float(cut_max):.6g}] deg "
                            "over lit requested looks. Extend the coupon sweep "
                            "or change the requested/installed envelope."
                        )
                if manifest is not None:
                    applicability = manifest["applicability"]
                    conical_limit = applicability[
                        "maximum_conical_incidence_deg"
                    ]
                    if (
                        metrics["maximum_requested_conical_incidence_deg"]
                        > conical_limit + 1.0e-9
                    ):
                        raise ValueError(
                            f"line dataset {dataset_id!r} is certified through "
                            f"{conical_limit:g} deg conical incidence, but line "
                            f"{record['line_id']!r} reaches "
                            f"{metrics['maximum_requested_conical_incidence_deg']:.3g} "
                            "deg over illuminated requested looks. The 2-D line "
                            "coefficient lookup does not model arbitrary d.t."
                        )
                    curvature_limit = applicability[
                        "minimum_along_line_normal_turn_radius_m"
                    ]
                    if (
                        installed_radius + 1.0e-12 < curvature_limit
                    ):
                        raise ValueError(
                            f"line dataset {dataset_id!r} requires an along-line "
                            f"normal-turn radius >= {curvature_limit:g} m, but line "
                            f"{record['line_id']!r} is approximately "
                            f"{installed_radius:.3g} "
                            "m. This does not certify transverse/principal host "
                            "curvature."
                        )
                    path_turn_limit = applicability[
                        "maximum_path_vertex_turn_deg"
                    ]
                    installed_path_turn = float(
                        metrics["maximum_path_vertex_turn_deg"]
                    )
                    if installed_path_turn > path_turn_limit + 1.0e-9:
                        raise ValueError(
                            f"line dataset {dataset_id!r} permits at most "
                            f"{path_turn_limit:g} deg path turn at a shared "
                            f"vertex, but line {record['line_id']!r} reaches "
                            f"{installed_path_turn:.6g} deg. Split/validate the "
                            "corner as its own interaction or use matching "
                            "corner evidence."
                        )
                    footprint_radius = float(
                        applicability["footprint_radius_m"]
                    )
                    self_overlap = _line_self_footprint_overlap(
                        np.asarray(placement["perimeter"], dtype=float),
                        footprint_radius,
                        cancel_check=cancel_check,
                    )
                    if self_overlap is not None:
                        left_index, right_index, clearance = self_overlap
                        message = (
                            f"Line {record['line_id']!r} folds back within its "
                            "own applicability footprint: segments "
                            f"{left_index + 1} and {right_index + 1} are "
                            f"nonlocally {clearance:.6g} m apart, below "
                            f"{2.0 * footprint_radius:.6g} m. Independent "
                            "straight-seam superposition omits this self/corner "
                            "coupling."
                        )
                        if require_manifests:
                            raise ValueError(message)
                        warnings.append(message)
                signature = _component_signature(
                    feature_kind,
                    response_identities[dataset_id],
                    placement["perimeter"],
                    placement["segment_normals"],
                )
                instance_id = str(record["line_id"])
            else:
                effective_roll = _canonical_point_roll(
                    np.asarray(placement["aperture_normal"], dtype=float),
                    np.asarray(placement["roll_ref"], dtype=float),
                )
                signature = _component_signature(
                    feature_kind,
                    response_identities[dataset_id],
                    placement["location"],
                    placement["aperture_normal"],
                    effective_roll,
                )
                instance_id = str(record["placement_id"])
            prior = seen_components.get(signature)
            if prior is not None:
                raise ValueError(
                    f"Duplicate physical feature component: {feature_kind} "
                    f"{instance_id!r} repeats {prior[0]} {prior[1]!r} with the "
                    "same response, location/path, and orientation."
                )
            seen_components[signature] = (feature_kind, instance_id)
            record["component_signature"] = signature
            record["dataset_content_sha256"] = response_content_identities[
                dataset_id
            ]
            record["dataset_physics_sha256"] = response_identities[dataset_id]
            if manifest is not None:
                record["feature_library_manifest_schema"] = manifest["schema"]
                record["feature_library_validation_status"] = manifest[
                    "validation"
                ]["status"]
                record["feature_library_footprint_radius_m"] = manifest[
                    "applicability"
                ]["footprint_radius_m"]
                footprint = {
                    "kind": feature_kind,
                    "instance_id": instance_id,
                    "radius_m": float(manifest["applicability"][
                        "footprint_radius_m"
                    ]),
                }
                if feature_kind == "point":
                    footprint["location"] = np.asarray(
                        placement["location"], dtype=float
                    )
                else:
                    footprint["segments"] = np.asarray(
                        placement["perimeter"], dtype=float
                    )
                footprint_components.append(footprint)

    point_ids = {str(record["placement_id"]) for record in point_records}
    line_ids = {str(record["line_id"]) for record in line_records}
    collisions = sorted(point_ids & line_ids)
    if collisions:
        raise ValueError(
            "Point placement_id and line_id values share one Assembly identity "
            f"namespace; rename duplicate ID(s) {collisions}."
        )
    overlap_warning_limit = 100
    overlap_count = 0
    for left, right in _footprint_candidate_pairs(
        footprint_components, cancel_check=cancel_check
    ):
        clearance = _component_clearance(left, right)
        required_clearance = left["radius_m"] + right["radius_m"]
        if clearance + 1.0e-12 >= required_clearance:
            continue
        overlap_count += 1
        message = (
            f"Feature applicability footprints overlap: {left['kind']} "
            f"{left['instance_id']!r} and {right['kind']} "
            f"{right['instance_id']!r} are {clearance:.6g} m apart, below "
            f"their combined {required_clearance:.6g} m footprint. "
            "Independent superposition omits cluster coupling."
        )
        if require_manifests:
            raise ValueError(message)
        if overlap_count <= overlap_warning_limit:
            warnings.append(message)
        elif overlap_count == overlap_warning_limit + 1:
            warnings.append(
                "More than 100 feature applicability-footprint overlaps were "
                "found; additional pairs are not rendered. Use Production "
                "validation and resolve clustered-feature coupling before release."
            )
            break
    return contracts, warnings, source_hashes, absent_source_paths
