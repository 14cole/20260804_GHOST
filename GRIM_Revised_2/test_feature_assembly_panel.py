from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

# Select a headless Qt platform before this module conditionally imports Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

GHOST_BACKEND = (
    Path(__file__).resolve().parents[1] / "tools" / "GHOST" / "Backend"
)

import feature_assembly_panel as feature_panel_module  # noqa: E402
from feature_assembly_panel import (  # noqa: E402
    GUI_AVAILABLE,
    LINE_PLACEMENT_COLUMNS,
    POINT_PLACEMENT_COLUMNS,
    FeatureAssemblyFormModel,
    FeatureAssemblyPanel,
    FeatureAssemblyValues,
    FeatureBuildDispatch,
    FeatureWorkflowAdapter,
    LoadedDatasetEntry,
    FEATURE_RECIPE_SCHEMA,
    FEATURE_RECIPE_VERSION,
    assess_surface_binding_readiness,
    preflight_base_grim,
    read_feature_assembly_recipe,
    _normalized_grim_output_path,
    placement_csv_template_text,
    write_feature_assembly_recipe,
    write_placement_csv_template,
)


def _write_minimal_base_grim(path: Path, *, embedded_bor: bool = False) -> None:
    """Write enough native GRIM structure for the lightweight body preflight."""

    payload = {
        "azimuths": np.asarray([0.0]),
        "elevations": np.asarray([0.0]),
        "frequencies": np.asarray([1.0]),
        "polarizations": np.asarray(["HH"]),
        "rcs_power": np.zeros((1, 1, 1, 1), dtype=np.float32),
        "rcs_phase": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }
    if embedded_bor:
        payload.update(
            body_profile_rho_m=np.asarray([0.0, 0.1, 0.0]),
            body_profile_z_m=np.asarray([-0.1, 0.0, 0.1]),
            requested_radar_grid_json=np.asarray("{}"),
        )
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _close_panel_without_prompt(panel) -> None:
    """Close a disposable test panel without exercising its close prompt."""

    panel._recipe_dirty = False
    panel.close()


@contextmanager
def _isolated_ghost_backend():
    """Import the real backend without leaking generic modules to later tests."""

    prior_modules = set(sys.modules)
    prior_path = list(sys.path)
    backend_root = GHOST_BACKEND.resolve()
    sys.path.insert(0, str(backend_root))
    try:
        workflow = importlib.import_module("feature_workflow")
        physics = importlib.import_module("feature_sum")
        line_model = importlib.import_module("line_expand")
        yield SimpleNamespace(
            feature_workflow=workflow,
            feature_sum=physics,
            c0=line_model.C0,
            psi_hh_deg=line_model.PSI_HH_DEG,
            psi_vv_deg=line_model.PSI_VV_DEG,
        )
    finally:
        sys.path[:] = prior_path
        for name in set(sys.modules) - prior_modules:
            source = getattr(sys.modules.get(name), "__file__", None)
            if not source:
                continue
            try:
                Path(source).resolve().relative_to(backend_root)
            except (OSError, ValueError):
                continue
            sys.modules.pop(name, None)


def _write_closed_box_facet(path: Path) -> None:
    """Write a small, outward-wound non-BoR rectangular-box surface."""

    vertices = np.asarray(
        [
            [-0.20, -0.30, -0.10],
            [0.20, -0.30, -0.10],
            [0.20, 0.30, -0.10],
            [-0.20, 0.30, -0.10],
            [-0.20, -0.30, 0.10],
            [0.20, -0.30, 0.10],
            [0.20, 0.30, 0.10],
            [-0.20, 0.30, 0.10],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 3, 2], [0, 2, 1],  # bottom, -z
            [4, 5, 6], [4, 6, 7],  # top, +z
            [0, 1, 5], [0, 5, 4],  # rear, -y
            [3, 7, 6], [3, 6, 2],  # nose, +y
            [1, 2, 6], [1, 6, 5],  # right, +x
            [0, 4, 7], [0, 7, 3],  # left, -x
        ],
        dtype=int,
    )
    rows = [f"{len(vertices)} {len(faces)}"]
    rows.extend(
        f"{index + 1} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}"
        for index, point in enumerate(vertices)
    )
    rows.extend(
        f"{index + 1} {face[0] + 1} {face[1] + 1} {face[2] + 1}"
        for index, face in enumerate(faces)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_isotropic_line_delta(
    path: Path,
    *,
    frequency_ghz: float,
    installed_coefficient: complex,
    c0: float,
    psi_hh_deg: float,
    psi_vv_deg: float,
) -> None:
    """Write one strict two-channel line response shared by both instances."""

    angles = np.asarray([0.0, 90.0, 180.0])
    wave_number = 2.0 * math.pi * frequency_ghz * 1.0e9 / c0
    raw_te = installed_coefficient * np.exp(-1j * math.radians(psi_vv_deg))
    raw_tm = installed_coefficient * np.exp(-1j * math.radians(psi_hh_deg))
    amplitude = np.empty((len(angles), 1, 1, 2), dtype=np.complex128)
    amplitude[..., 0] = raw_te
    amplitude[..., 1] = raw_tm
    payload = {
        "azimuths": angles,
        "elevations": np.asarray([0.0]),
        "frequencies": np.asarray([frequency_ghz]),
        "polarizations": np.asarray(["VV", "HH"]),
        "rcs_power": (
            np.abs(amplitude) ** 2 / (4.0 * wave_number)
        ).astype(np.float32),
        "rcs_phase": np.angle(amplitude).astype(np.float32),
        "units": np.asarray(json.dumps({
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_linear_quantity": "sigma_2d",
        })),
        "raw_complex_amplitude_preserved": np.asarray(True),
        "rcs_amp_real": amplitude.real.astype(np.float64),
        "rcs_amp_imag": amplitude.imag.astype(np.float64),
    }
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


class _CapturedRequest:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _FakeWorkflow:
    FeatureAssemblyRequest = _CapturedRequest

    def __init__(self):
        self.calls = []
        self.requirements = SimpleNamespace(
            point_dataset_ids=("fastener",),
            line_dataset_ids=("panel_gap",),
        )

    def discover_feature_dataset_ids(self, **kwargs):
        self.calls.append(("discover", dict(kwargs)))
        return self.requirements

    def prepare_feature_assembly(self, request):
        self.calls.append(("prepare", request))
        return SimpleNamespace(request=request, preview_geometry="preview")

    def prepare_feature_input_preview(self, **kwargs):
        self.calls.append(("input_preview", dict(kwargs)))
        return SimpleNamespace(preview_stage="inputs", **kwargs)

    def execute_feature_assembly(self, plan):
        self.calls.append(("execute", plan))
        return plan.request.kwargs["output_grim"]


def _ready_point_model() -> FeatureAssemblyFormModel:
    values = FeatureAssemblyValues(
        base_grim="clean_body.grim",
        output_grim="assembled.grim",
        coordinate_units="millimeters",
        surface_mesh="body.stl",
        surface_units="meters",
        flip_surface_normals=True,
        shadow=True,
        shadow_bias_m=2.5e-5,
        point_locations_csv="points.csv",
        skin_tol_m=8.0e-4,
        skin_phase_tol_deg=12.0,
        normal_tol_deg=9.0,
        expected_host_material="PEC",
    )
    model = FeatureAssemblyFormModel(values)
    model.update_dataset_requirements(
        {"point_dataset_ids": ("fastener",), "line_dataset_ids": ()}
    )
    model.set_point_dataset("fastener", "fastener_opn_minus_frd.grim")
    return model


class FeatureAssemblyModelTests(unittest.TestCase):
    def test_versioned_recipe_round_trip_preserves_all_effective_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "inputs"
            recipes = root / "recipes"
            inputs.mkdir()
            recipes.mkdir()
            for name, content in (
                ("body.grim", b"clean response"),
                ("body.stl", b"solid body"),
                ("points.csv", b"point placement bytes"),
                ("lines.csv", b"line placement bytes"),
                ("fastener.grim", b"point delta"),
                ("seal.grim", b"line delta"),
            ):
                (inputs / name).write_bytes(content)
            values = FeatureAssemblyValues(
                base_grim="inputs/body.grim",
                output_grim="outputs/vehicle_features.grim",
                coordinate_units="millimeters",
                surface_mesh="inputs/body.stl",
                surface_units="meters",
                flip_surface_normals=True,
                shadow=True,
                shadow_bias_m=2.5e-5,
                point_locations_csv="inputs/points.csv",
                line_locations_csv="inputs/lines.csv",
                skin_tol_m=7.5e-4,
                skin_phase_tol_deg=11.0,
                normal_tol_deg=8.0,
                allow_legacy_base_metadata=False,
                require_feature_manifests=True,
                expected_host_material="paint-stack-v3",
                base_dir=str(root),
                point_datasets={"fastener": "inputs/fastener.grim"},
                line_datasets={"seal": "inputs/seal.grim"},
                point_host_materials={"fastener": "paint-stack-v3"},
                line_host_materials={"seal": "rubber-seal-host-v1"},
                excluded_point_placement_ids={"bolt_002"},
                excluded_line_ids={"rear_door"},
            )
            saved = write_feature_assembly_recipe(
                values,
                recipes / "vehicle",
                name="Full vehicle",
                variant="Doors + fasteners",
            )

            self.assertTrue(str(saved).endswith(".assembly.json"))
            document = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(document["schema"], FEATURE_RECIPE_SCHEMA)
            self.assertEqual(document["version"], FEATURE_RECIPE_VERSION)
            self.assertFalse(Path(document["values"]["base_grim"]).is_absolute())
            self.assertTrue(document["source_manifest"])
            self.assertTrue(
                all("sha256" in record for record in document["source_manifest"])
            )

            loaded = read_feature_assembly_recipe(saved)
            self.assertEqual(loaded.name, "Full vehicle")
            self.assertEqual(loaded.variant, "Doors + fasteners")
            self.assertEqual(loaded.source_warnings, ())
            restored = loaded.values
            self.assertEqual(restored.coordinate_units, "millimeters")
            self.assertEqual(restored.surface_units, "meters")
            self.assertTrue(restored.flip_surface_normals)
            self.assertTrue(restored.shadow)
            self.assertEqual(restored.shadow_bias_m, 2.5e-5)
            self.assertEqual(restored.skin_tol_m, 7.5e-4)
            self.assertEqual(restored.skin_phase_tol_deg, 11.0)
            self.assertEqual(restored.normal_tol_deg, 8.0)
            self.assertFalse(restored.allow_legacy_base_metadata)
            self.assertTrue(restored.require_feature_manifests)
            self.assertEqual(restored.expected_host_material, "paint-stack-v3")
            self.assertEqual(
                restored.point_host_materials, {"fastener": "paint-stack-v3"}
            )
            self.assertEqual(
                restored.line_host_materials, {"seal": "rubber-seal-host-v1"}
            )
            self.assertEqual(
                Path(restored.base_grim), (inputs / "body.grim").resolve()
            )
            self.assertEqual(
                Path(restored.output_grim),
                (root / "outputs" / "vehicle_features.grim").resolve(),
            )
            self.assertEqual(
                Path(restored.point_datasets["fastener"]),
                (inputs / "fastener.grim").resolve(),
            )
            self.assertEqual(restored.excluded_point_placement_ids, {"bolt_002"})
            self.assertEqual(restored.excluded_line_ids, {"rear_door"})

    def test_recipe_reports_content_change_without_silently_rejecting_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            body = root / "body.grim"
            points = root / "points.csv"
            body.write_bytes(b"body")
            points.write_bytes(b"AAAA")
            recipe = write_feature_assembly_recipe(
                FeatureAssemblyValues(
                    base_grim=str(body), point_locations_csv=str(points)
                ),
                root / "trade",
                name="Vehicle",
                variant="Baseline",
            )
            # Same length defeats size-only identity but must be found by SHA-256.
            points.write_bytes(b"BBBB")

            loaded = read_feature_assembly_recipe(recipe)

            self.assertEqual(loaded.values.point_locations_csv, str(points.resolve()))
            self.assertTrue(
                any("content changed" in warning for warning in loaded.source_warnings)
            )

    def test_recipe_rejects_unknown_schema_version(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "future.assembly.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": FEATURE_RECIPE_SCHEMA,
                        "version": 99,
                        "name": "Future",
                        "variant": "Option",
                        "values": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported.*version"):
                read_feature_assembly_recipe(path)

    def test_v1_recipe_migrates_without_inventing_host_material_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recipe = write_feature_assembly_recipe(
                FeatureAssemblyValues(),
                root / "legacy",
                name="Vehicle",
                variant="Legacy library",
            )
            document = json.loads(recipe.read_text(encoding="utf-8"))
            document["version"] = 1
            document["values"].pop("expected_host_material")
            recipe.write_text(json.dumps(document), encoding="utf-8")

            loaded = read_feature_assembly_recipe(recipe)

            self.assertEqual(loaded.values.expected_host_material, "")
            self.assertTrue(
                any("predates host material" in value for value in loaded.source_warnings)
            )

    def test_v2_recipe_migrates_global_host_as_default_with_review_warning(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recipe = write_feature_assembly_recipe(
                FeatureAssemblyValues(
                    point_datasets={"fastener": "fastener.grim"},
                    expected_host_material="PEC",
                ),
                root / "v2",
                name="Vehicle",
                variant="One host",
            )
            document = json.loads(recipe.read_text(encoding="utf-8"))
            document["version"] = 2
            document["values"].pop("point_host_materials")
            document["values"].pop("line_host_materials")
            recipe.write_text(json.dumps(document), encoding="utf-8")

            loaded = read_feature_assembly_recipe(recipe)

            self.assertEqual(loaded.values.expected_host_material, "PEC")
            self.assertEqual(loaded.values.point_host_materials, {})
            self.assertTrue(
                any("only one global" in value for value in loaded.source_warnings)
            )

    def test_base_grim_preflight_distinguishes_external_bor_and_malformed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            external = root / "external.grim"
            embedded = root / "bor.grim"
            malformed = root / "malformed.grim"
            _write_minimal_base_grim(external)
            _write_minimal_base_grim(embedded, embedded_bor=True)
            malformed.write_bytes(b"not a ZIP")

            external_result = preflight_base_grim(external)
            embedded_result = preflight_base_grim(embedded)
            malformed_result = preflight_base_grim(malformed)

            self.assertTrue(external_result.valid)
            self.assertTrue(external_result.requires_surface_mesh)
            self.assertFalse(external_result.embedded_bor)
            self.assertTrue(embedded_result.valid)
            self.assertTrue(embedded_result.embedded_bor)
            self.assertFalse(embedded_result.requires_surface_mesh)
            self.assertFalse(malformed_result.valid)
            self.assertIn("Invalid GRIM container", malformed_result.summary)

    def test_external_binding_readiness_is_explicit_cached_and_stat_invalidated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            external = root / "clean_vehicle.grim"
            embedded = root / "clean_bor.grim"
            surface = root / "vehicle.facet"
            _write_minimal_base_grim(external)
            _write_minimal_base_grim(embedded, embedded_bor=True)
            surface.write_text("4 2\nmesh\n", encoding="utf-8")

            missing = assess_surface_binding_readiness(
                base_grim=external,
                surface_mesh=surface,
                surface_units="meters",
                production_profile=True,
            )
            self.assertEqual(missing.code, "missing")
            self.assertFalse(missing.ready)
            self.assertTrue(missing.required)

            sidecar = Path(str(surface) + ".assembly.json")
            sidecar.write_text("{}\n", encoding="utf-8")
            unchecked = assess_surface_binding_readiness(
                base_grim=external,
                surface_mesh=surface,
                surface_units="meters",
                production_profile=True,
            )
            self.assertEqual(unchecked.code, "unchecked")
            self.assertFalse(unchecked.ready)
            self.assertIsNotNone(unchecked.identity_key)

            checked = assess_surface_binding_readiness(
                base_grim=external,
                surface_mesh=surface,
                surface_units="meters",
                production_profile=True,
                checked_key=unchecked.identity_key,
                checked_binding={
                    "geometry_id": "vehicle-r7",
                    "attestation_case_id": "registration-42",
                },
            )
            self.assertEqual(checked.code, "valid")
            self.assertTrue(checked.ready)
            self.assertIn("vehicle-r7", checked.message)

            stale = assess_surface_binding_readiness(
                base_grim=external,
                surface_mesh=surface,
                surface_units="inches",
                production_profile=True,
                checked_key=unchecked.identity_key,
                checked_binding={
                    "geometry_id": "vehicle-r7",
                    "attestation_case_id": "registration-42",
                },
            )
            self.assertEqual(stale.code, "stale")
            self.assertFalse(stale.ready)

            self_bound = assess_surface_binding_readiness(
                base_grim=embedded,
                surface_mesh="",
                surface_units="meters",
                production_profile=True,
            )
            self.assertEqual(self_bound.code, "not_required")
            self.assertTrue(self_bound.ready)

    def test_loaded_dataset_entry_requires_a_clean_existing_grim_file(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / "saved.grim"
            saved.write_bytes(b"artifact")

            self.assertEqual(
                LoadedDatasetEntry("saved-id", "Saved", str(saved)).usable_path,
                str(saved),
            )
            dirty = LoadedDatasetEntry(
                "dirty-id", "Derived", str(saved), dirty=True
            )
            self.assertEqual(dirty.usable_path, "")
            self.assertIn("save unsaved derived", dirty.unavailable_reason)
            missing = LoadedDatasetEntry(
                "missing-id", "Missing", str(Path(folder) / "missing.grim")
            )
            self.assertEqual(missing.usable_path, "")
            self.assertIn("missing", missing.unavailable_reason)

    def test_displayed_templates_are_the_strict_shared_placement_contracts(self):
        self.assertEqual(
            POINT_PLACEMENT_COLUMNS,
            (
                "placement_id", "dataset_id", "x", "y", "z", "nx", "ny",
                "nz", "roll_x", "roll_y", "roll_z",
            ),
        )
        self.assertEqual(
            LINE_PLACEMENT_COLUMNS,
            (
                "line_id", "dataset_id", "segment_index", "x1", "y1", "z1",
                "x2", "y2", "z2", "n1x", "n1y", "n1z", "n2x", "n2y",
                "n2z",
            ),
        )
        self.assertEqual(
            placement_csv_template_text("point"),
            ",".join(POINT_PLACEMENT_COLUMNS) + "\n",
        )
        self.assertEqual(
            placement_csv_template_text("line"),
            ",".join(LINE_PLACEMENT_COLUMNS) + "\n",
        )

        with tempfile.TemporaryDirectory() as directory:
            target = write_placement_csv_template(
                "point", Path(directory) / "placements"
            )
            self.assertEqual(target.suffix, ".csv")
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                placement_csv_template_text("point"),
            )

    def test_adapter_rejects_backend_csv_contract_drift(self):
        workflow = _FakeWorkflow()
        workflow.POINT_CSV_COLUMNS = ("different",)

        with self.assertRaisesRegex(RuntimeError, "no longer matches"):
            FeatureWorkflowAdapter.from_module(workflow)

    def test_input_preview_does_not_require_output_or_response_mapping(self):
        workflow = _FakeWorkflow()
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                base_grim="body.grim",
                surface_mesh="body.stl",
                coordinate_units="millimeters",
                surface_units="meters",
                point_locations_csv="points.csv",
                line_locations_csv="lines.csv",
            )
        )

        preview = model.prepare_input_preview(workflow)

        self.assertEqual(preview.preview_stage, "inputs")
        self.assertEqual(
            workflow.calls,
            [
                (
                    "input_preview",
                    {
                        "base_grim": "body.grim",
                        "surface_mesh": "body.stl",
                        "coordinate_units": "millimeters",
                        "surface_units": "meters",
                        "point_locations_csv": "points.csv",
                        "line_locations_csv": "lines.csv",
                        "enabled_point_placement_ids": None,
                        "enabled_line_ids": None,
                        "base_dir": None,
                    },
                )
            ],
        )

    def test_invalidating_one_csv_discards_only_its_mapping_rows(self):
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                point_datasets={"fastener": "fastener.grim"},
                line_datasets={"gap": "gap.grim"},
            )
        )
        model.update_dataset_requirements(
            {"point_dataset_ids": ("fastener",), "line_dataset_ids": ("gap",)}
        )

        model.invalidate_dataset_requirements("point")

        self.assertEqual(model.point_dataset_ids, ())
        self.assertEqual(model.values.point_datasets, {})
        self.assertEqual(model.line_dataset_ids, ("gap",))
        self.assertEqual(model.values.line_datasets, {"gap": "gap.grim"})

    def test_enabled_instances_define_required_mappings_and_request_snapshot(self):
        workflow = _FakeWorkflow()
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                base_grim="body.grim",
                output_grim="assembled.grim",
                point_locations_csv="points.csv",
                expected_host_material="PEC",
            )
        )
        model.update_dataset_requirements(
            {
                "point_dataset_ids": ("active", "disabled"),
                "line_dataset_ids": (),
                "point_instances": (
                    ("keep", "active"),
                    ("omit", "disabled"),
                ),
            }
        )
        model.set_point_dataset("active", "active.grim")
        model.set_feature_instance_enabled("point", "omit", False)

        self.assertEqual(model.missing_dataset_mappings(), ())
        self.assertEqual(model.active_point_dataset_ids(), ("active",))
        self.assertIn("point=[omit]", model.feature_selection_summary())
        request = model.build_request(workflow)
        self.assertEqual(request.kwargs["point_datasets"], {"active": "active.grim"})
        self.assertEqual(
            request.kwargs["enabled_point_placement_ids"], ("keep",)
        )

    def test_selection_summary_can_bound_display_without_losing_full_record(self):
        point_instances = tuple(
            (f"fastener_{index:03d}", "fastener") for index in range(12)
        )
        line_instances = tuple(
            (f"seal_{index:03d}", "seal", 1) for index in range(5)
        )
        model = FeatureAssemblyFormModel()
        model.update_dataset_requirements(
            {
                "point_dataset_ids": ("fastener",),
                "line_dataset_ids": ("seal",),
                "point_instances": point_instances,
                "line_instances": line_instances,
            }
        )
        model.set_excluded_feature_instances(
            point_ids=[value[0] for value in point_instances[:10]],
            line_ids=[value[0] for value in line_instances],
        )

        display = model.feature_selection_summary(max_disabled_ids_per_kind=3)
        full = model.feature_selection_summary()

        self.assertIn("fastener_000", display)
        self.assertIn("… +7 more", display)
        self.assertIn("… +2 more", display)
        self.assertIn("use Copy full selection", display)
        self.assertNotIn("fastener_009", display)
        self.assertIn("fastener_009", full)
        self.assertIn("seal_004", full)
        self.assertNotIn("more", full)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            model.feature_selection_summary(max_disabled_ids_per_kind=-1)

    def test_all_disabled_spatial_configuration_is_rejected(self):
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                base_grim="body.grim",
                output_grim="assembled.grim",
                point_locations_csv="points.csv",
                expected_host_material="PEC",
            )
        )
        model.update_dataset_requirements(
            {
                "point_dataset_ids": ("fastener",),
                "line_dataset_ids": (),
                "point_instances": (("p1", "fastener"),),
            }
        )
        model.set_feature_instance_enabled("point", "p1", False)

        with self.assertRaisesRegex(ValueError, "No enabled spatial features"):
            model.validate()

    def test_all_disabled_configuration_can_preview_the_clean_body(self):
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                base_grim="body.grim",
                point_locations_csv="points.csv",
            )
        )
        model.update_dataset_requirements(
            {
                "point_dataset_ids": ("fastener",),
                "line_dataset_ids": (),
                "point_instances": (("p1", "fastener"),),
            }
        )
        model.set_feature_instance_enabled("point", "p1", False)
        workflow = _FakeWorkflow()

        preview = model.prepare_input_preview(workflow)

        self.assertEqual(preview.enabled_point_placement_ids, ())
        self.assertIsNone(preview.enabled_line_ids)
        self.assertEqual(workflow.calls[-1][0], "input_preview")

    def test_same_source_rescan_keeps_surviving_exclusions_and_enables_new_ids(self):
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(point_locations_csv="points.csv")
        )
        model.update_dataset_requirements(
            {
                "point_dataset_ids": ("family",),
                "line_dataset_ids": (),
                "point_instances": (("p1", "family"), ("p2", "family")),
            }
        )
        model.set_feature_instance_enabled("point", "p1", False)

        model.update_dataset_requirements(
            {
                "point_dataset_ids": ("family",),
                "line_dataset_ids": (),
                "point_instances": (("p1", "family"), ("p3", "family")),
            }
        )

        self.assertEqual(model.values.excluded_point_placement_ids, {"p1"})
        self.assertEqual(model.enabled_point_placement_ids, ("p3",))

    def test_prepare_rejects_selection_change_instead_of_mislabeling_cache(self):
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                base_grim="body.grim",
                output_grim="assembled.grim",
                point_locations_csv="points.csv",
                expected_host_material="PEC",
            )
        )
        model.update_dataset_requirements(
            {
                "point_dataset_ids": ("family",),
                "line_dataset_ids": (),
                "point_instances": (("p1", "family"), ("p2", "family")),
            }
        )
        model.set_point_dataset("family", "family.grim")

        class MutatingWorkflow(_FakeWorkflow):
            def prepare_feature_assembly(self, request):
                self.calls.append(("prepare", request))
                model.set_feature_instance_enabled("point", "p1", False)
                return SimpleNamespace(request=request, preview_geometry="stale")

        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            model.prepare_preview(MutatingWorkflow())
        self.assertIsNone(model._prepared_plan_cache)

    def test_input_preview_rejects_selection_change_during_load(self):
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(point_locations_csv="points.csv")
        )
        model.update_dataset_requirements(
            {
                "point_dataset_ids": ("family",),
                "line_dataset_ids": (),
                "point_instances": (("p1", "family"), ("p2", "family")),
            }
        )

        class MutatingWorkflow(_FakeWorkflow):
            def prepare_feature_input_preview(self, **kwargs):
                self.calls.append(("input_preview", dict(kwargs)))
                model.set_feature_instance_enabled("point", "p1", False)
                return SimpleNamespace(preview_stage="inputs", **kwargs)

        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            model.prepare_input_preview(MutatingWorkflow())

    def test_changed_csv_path_cannot_reuse_previous_discovery(self):
        model = _ready_point_model()
        model.values.point_locations_csv = "replacement_points.csv"

        with self.assertRaisesRegex(ValueError, "changed after its last"):
            model.validate()

    def test_in_place_csv_edit_cannot_reuse_previous_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            point_csv = Path(folder) / "points.csv"
            point_csv.write_bytes(b"alpha\n")
            original = point_csv.stat()
            model = _ready_point_model()
            model.values.point_locations_csv = str(point_csv)
            model.update_dataset_requirements(
                {"point_dataset_ids": ("fastener",), "line_dataset_ids": ()}
            )
            model.set_point_dataset("fastener", "fastener.grim")
            self.assertTrue(model.requirements_are_current("point"))

            point_csv.write_bytes(b"bravo\n")
            os.utime(
                point_csv,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )

            with self.assertRaisesRegex(ValueError, "changed after its last"):
                model.validate()

    def test_output_aliases_are_rejected_before_backend_prepare(self):
        model = _ready_point_model()
        model.values.base_grim = "body.grim"
        model.values.output_grim = "body"
        with self.assertRaisesRegex(ValueError, "clean-body response"):
            model.validate()

        model.values.output_grim = "fastener_opn_minus_frd"
        with self.assertRaisesRegex(ValueError, "point response"):
            model.validate()

    def test_cached_preview_rejects_late_hardlink_output_alias_without_execute(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "body.grim"
            points = root / "points.csv"
            response = root / "fastener.grim"
            output = root / "assembled.grim"
            base.write_bytes(b"base response")
            points.write_bytes(b"stable placement bytes")
            response.write_bytes(b"feature response")

            model = FeatureAssemblyFormModel(
                FeatureAssemblyValues(
                    base_grim=str(base),
                    output_grim=str(output),
                    point_locations_csv=str(points),
                    expected_host_material="PEC",
                )
            )
            model.update_dataset_requirements(
                {"point_dataset_ids": ("fastener",), "line_dataset_ids": ()}
            )
            model.set_point_dataset("fastener", str(response))
            workflow = _FakeWorkflow()
            model.prepare_preview(workflow)

            try:
                os.link(base, output)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"hard links are unavailable on this filesystem: {exc}")

            with self.assertRaisesRegex(ValueError, "clean-body response"):
                model.assemble(workflow)

            self.assertEqual(
                [name for name, _value in workflow.calls],
                ["prepare"],
            )

    def test_output_normalization_resolves_final_symlink_before_adding_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "existing.grim"
            link = root / "output_link"
            target.write_bytes(b"existing result")
            try:
                link.symlink_to(target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlinks are unavailable on this platform: {exc}")

            self.assertEqual(
                _normalized_grim_output_path(link),
                target.resolve(),
            )

    def test_input_preview_rejects_csv_changed_while_backend_reads_it(self):
        class MutatingInputPreviewWorkflow(_FakeWorkflow):
            def prepare_feature_input_preview(self, **kwargs):
                self.calls.append(("input_preview", dict(kwargs)))
                point_csv = Path(kwargs["point_locations_csv"])
                original = point_csv.stat()
                point_csv.write_bytes(b"bravo\n")
                os.utime(
                    point_csv,
                    ns=(original.st_atime_ns, original.st_mtime_ns),
                )
                return SimpleNamespace(preview_stage="inputs", **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            point_csv = Path(folder) / "points.csv"
            point_csv.write_bytes(b"alpha\n")
            model = FeatureAssemblyFormModel(
                FeatureAssemblyValues(point_locations_csv=str(point_csv))
            )
            model.update_dataset_requirements(
                {"point_dataset_ids": ("fastener",), "line_dataset_ids": ()}
            )
            model.set_point_dataset("fastener", "fastener.grim")
            workflow = MutatingInputPreviewWorkflow()

            with self.assertRaisesRegex(RuntimeError, "changed while"):
                model.prepare_input_preview(workflow)

            self.assertEqual(
                [name for name, _value in workflow.calls],
                ["input_preview"],
            )
            self.assertEqual(model.point_dataset_ids, ())
            self.assertEqual(model.values.point_datasets, {})

    def test_discovery_invalidates_ids_when_csv_changes_during_parse(self):
        class MutatingDiscoveryWorkflow(_FakeWorkflow):
            def discover_feature_dataset_ids(self, **kwargs):
                self.calls.append(("discover", dict(kwargs)))
                point_csv = Path(kwargs["point_locations_csv"])
                original = point_csv.stat()
                point_csv.write_bytes(b"bravo\n")
                os.utime(
                    point_csv,
                    ns=(original.st_atime_ns, original.st_mtime_ns),
                )
                return self.requirements

        with tempfile.TemporaryDirectory() as folder:
            point_csv = Path(folder) / "points.csv"
            point_csv.write_bytes(b"alpha\n")
            model = FeatureAssemblyFormModel(
                FeatureAssemblyValues(point_locations_csv=str(point_csv))
            )
            model.update_dataset_requirements(
                {"point_dataset_ids": ("fastener",), "line_dataset_ids": ()}
            )
            model.set_point_dataset("fastener", "fastener.grim")

            with self.assertRaisesRegex(RuntimeError, "changed while"):
                model.query_dataset_ids(MutatingDiscoveryWorkflow())

            self.assertEqual(model.point_dataset_ids, ())
            self.assertEqual(model.values.point_datasets, {})

    def test_normal_tolerance_must_keep_frames_outward_facing(self):
        model = _ready_point_model()
        model.values.normal_tol_deg = 90.0
        with self.assertRaisesRegex(ValueError, "less than 90"):
            model.validate()

    def test_production_host_ids_resolve_per_response_with_global_fallback(self):
        model = _ready_point_model()
        model.values.expected_host_material = ""
        with self.assertRaisesRegex(ValueError, "point:fastener"):
            model.validate()

        model.set_point_host_material("fastener", "coated-aluminum-v2")
        model.validate()
        self.assertEqual(
            model.effective_host_materials(),
            {"point:fastener": "coated-aluminum-v2"},
        )

    def test_point_and_line_can_share_dataset_id_with_distinct_host_stacks(self):
        model = FeatureAssemblyFormModel(FeatureAssemblyValues(
            expected_host_material="  PEC   outer skin  ",
        ))
        model.update_dataset_requirements({
            "point_dataset_ids": ("shared",),
            "line_dataset_ids": ("shared",),
        })
        model.set_point_host_material("shared", "pec OUTER skin")
        model.set_line_host_material("shared", "rubber   seal stack v2")

        self.assertEqual(model.effective_host_materials(), {
            "point:shared": "PEC outer skin",
            "line:shared": "rubber seal stack v2",
        })
        self.assertEqual(model.missing_host_material_mappings(), ())

    def test_output_alias_guard_includes_mesh_and_placement_csvs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = {
                "surface mesh": root / "body.facet",
                "point placement CSV": root / "points.csv",
                "line placement CSV": root / "lines.csv",
            }
            for path in sources.values():
                path.write_bytes(b"stable source")
            model = FeatureAssemblyFormModel(
                FeatureAssemblyValues(
                    surface_mesh=str(sources["surface mesh"]),
                    point_locations_csv=str(sources["point placement CSV"]),
                    line_locations_csv=str(sources["line placement CSV"]),
                )
            )
            for label, source in sources.items():
                output = root / f"{label.replace(' ', '_')}.grim"
                try:
                    os.link(source, output)
                except (NotImplementedError, OSError) as exc:
                    self.skipTest(f"hard links are unavailable: {exc}")
                model.values.output_grim = str(output)
                with self.assertRaisesRegex(ValueError, label):
                    model._validate_output_target()

    def test_request_construction_uses_exact_mapping_and_controls(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()

        request = model.build_request(workflow)

        self.assertIsInstance(request, _CapturedRequest)
        self.assertEqual(request.kwargs["base_grim"], "clean_body.grim")
        self.assertEqual(request.kwargs["output_grim"], "assembled.grim")
        self.assertEqual(request.kwargs["coordinate_units"], "millimeters")
        self.assertEqual(request.kwargs["surface_mesh"], "body.stl")
        self.assertEqual(request.kwargs["surface_units"], "meters")
        self.assertTrue(request.kwargs["flip_surface_normals"])
        self.assertTrue(request.kwargs["shadow"])
        self.assertEqual(request.kwargs["shadow_bias_m"], 2.5e-5)
        self.assertEqual(request.kwargs["point_locations_csv"], "points.csv")
        self.assertEqual(
            request.kwargs["point_datasets"],
            {"fastener": "fastener_opn_minus_frd.grim"},
        )
        self.assertIsNone(request.kwargs["line_locations_csv"])
        self.assertEqual(request.kwargs["line_datasets"], {})
        self.assertEqual(request.kwargs["skin_tol_m"], 8.0e-4)
        self.assertEqual(request.kwargs["skin_phase_tol_deg"], 12.0)
        self.assertEqual(request.kwargs["normal_tol_deg"], 9.0)
        self.assertFalse(request.kwargs["allow_legacy_base_metadata"])
        self.assertTrue(request.kwargs["require_feature_manifests"])
        self.assertEqual(request.kwargs["expected_host_material"], "PEC")
        self.assertEqual(
            request.kwargs["expected_host_materials"], {"point:fastener": "PEC"}
        )

    def test_discovery_updates_ids_and_preserves_surviving_paths(self):
        workflow = _FakeWorkflow()
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                point_locations_csv="points.csv",
                line_locations_csv="lines.csv",
            )
        )

        model.discover_dataset_ids(workflow)
        model.set_point_dataset("fastener", "fastener.grim")
        model.set_line_dataset("panel_gap", "gap.grim")

        workflow.requirements = SimpleNamespace(
            point_dataset_ids=("fastener", "antenna"),
            line_dataset_ids=("seam",),
        )
        model.discover_dataset_ids(workflow)

        self.assertEqual(model.point_dataset_ids, ("fastener", "antenna"))
        self.assertEqual(model.line_dataset_ids, ("seam",))
        self.assertEqual(
            model.values.point_datasets,
            {"fastener": "fastener.grim", "antenna": ""},
        )
        self.assertEqual(model.values.line_datasets, {"seam": ""})
        discover_call = workflow.calls[-1]
        self.assertEqual(discover_call[0], "discover")
        self.assertEqual(
            discover_call[1],
            {
                "point_locations_csv": "points.csv",
                "line_locations_csv": "lines.csv",
                "base_dir": None,
            },
        )

    def test_preview_and_build_dispatch_through_injected_service(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()

        preview_plan = model.prepare_preview(workflow)
        dispatch = model.assemble(workflow)

        self.assertEqual(preview_plan.preview_geometry, "preview")
        self.assertIsInstance(dispatch, FeatureBuildDispatch)
        self.assertEqual(dispatch.output_path, "assembled.grim")
        self.assertTrue(dispatch.reused_validated_plan)
        self.assertEqual(
            [name for name, _value in workflow.calls],
            ["prepare", "execute"],
        )
        self.assertIs(workflow.calls[-1][1], dispatch.plan)

    def test_validated_publish_never_prepares_an_unreviewed_replacement(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()

        with self.assertRaisesRegex(RuntimeError, "have not been validated"):
            model.assemble_validated(workflow)
        self.assertEqual(workflow.calls, [])

        model.prepare_preview(workflow)
        model.values.normal_tol_deg = 8.0
        with self.assertRaisesRegex(RuntimeError, "Validate placements"):
            model.assemble_validated(workflow)

        self.assertEqual(
            [name for name, _value in workflow.calls],
            ["prepare"],
        )

    def test_build_forwards_progress_and_cooperative_cancellation_hooks(self):
        class HookWorkflow(_FakeWorkflow):
            def execute_feature_assembly(
                self, plan, *, cancel_check=None, progress_callback=None
            ):
                self.calls.append(("execute", plan))
                self.received_cancel = cancel_check
                self.received_progress = progress_callback
                progress_callback(2, 4, "placing points")
                return plan.request.kwargs["output_grim"]

        workflow = HookWorkflow()
        model = _ready_point_model()
        progress = []
        dispatch = model.assemble(
            workflow,
            cancel_check=lambda: False,
            progress_callback=lambda done, total, message: progress.append(
                (done, total, message)
            ),
        )

        self.assertEqual(dispatch.output_path, "assembled.grim")
        self.assertFalse(workflow.received_cancel())
        self.assertEqual(progress, [(2, 4, "placing points")])

    def test_validation_forwards_progress_and_cooperative_cancellation_hooks(self):
        class HookPrepareWorkflow(_FakeWorkflow):
            def prepare_feature_assembly(
                self, request, *, cancel_check=None, progress_callback=None
            ):
                self.calls.append(("prepare", request))
                self.received_cancel = cancel_check
                self.received_progress = progress_callback
                progress_callback(3, 5, "Checking line paths")
                return SimpleNamespace(request=request, validation_warnings=())

        workflow = HookPrepareWorkflow()
        model = _ready_point_model()
        progress = []

        plan = model.prepare_preview(
            workflow,
            cancel_check=lambda: False,
            progress_callback=lambda done, total, message: progress.append(
                (done, total, message)
            ),
        )

        self.assertIsNotNone(plan)
        self.assertTrue(callable(workflow.received_cancel))
        self.assertTrue(callable(workflow.received_progress))
        self.assertEqual(progress, [(3, 5, "Checking line paths")])
        self.assertTrue(model.validated_plan_is_current(workflow))

    def test_cancelled_validation_discards_the_previous_cached_plan(self):
        model = _ready_point_model()
        first_workflow = _FakeWorkflow()
        model.prepare_preview(first_workflow)
        self.assertIsNotNone(model._prepared_plan_cache)

        class CancellingPrepareWorkflow(_FakeWorkflow):
            def prepare_feature_assembly(
                self, request, *, cancel_check=None, progress_callback=None
            ):
                self.calls.append(("prepare", request))
                progress_callback(1, 4, "Checking body surface")
                raise InterruptedError("operator cancelled validation")

        workflow = CancellingPrepareWorkflow()
        with self.assertRaisesRegex(InterruptedError, "operator cancelled"):
            model.prepare_preview(
                workflow,
                cancel_check=lambda: False,
                progress_callback=lambda *_args: None,
            )

        self.assertIsNone(model._prepared_plan_cache)
        self.assertFalse(model.validated_plan_is_current(workflow))

    def test_cancel_after_final_fingerprint_never_caches_reviewed_plan(self):
        model = _ready_point_model()
        workflow = _FakeWorkflow()
        calls = 0

        def cancel_at_final_cache_boundary():
            nonlocal calls
            calls += 1
            return calls == 3

        with self.assertRaisesRegex(InterruptedError, "no reviewed plan"):
            model.prepare_preview(
                workflow,
                cancel_check=cancel_at_final_cache_boundary,
            )

        self.assertEqual(calls, 3)
        self.assertIsNone(model._prepared_plan_cache)
        self.assertFalse(model.validated_plan_is_current(workflow))

    def test_cancel_before_execute_keeps_legacy_service_untouched(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            model.assemble(workflow, cancel_check=lambda: True)

        self.assertEqual(
            [name for name, _value in workflow.calls],
            ["prepare"],
        )

    def test_fingerprint_passes_are_bounded_for_cold_and_cached_builds(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()

        with mock.patch.object(
            model,
            "_source_fingerprints",
            wraps=model._source_fingerprints,
        ) as fingerprints:
            cold = model.assemble(workflow)
        self.assertFalse(cold.reused_validated_plan)
        self.assertEqual(fingerprints.call_count, 2)

        with mock.patch.object(
            model,
            "_source_fingerprints",
            wraps=model._source_fingerprints,
        ) as fingerprints:
            warm = model.assemble(workflow)
        self.assertTrue(warm.reused_validated_plan)
        self.assertEqual(fingerprints.call_count, 1)

    def test_changed_controls_force_a_new_prepare_before_build(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()

        model.prepare_preview(workflow)
        model.values.normal_tol_deg = 8.0
        dispatch = model.assemble(workflow)

        self.assertFalse(dispatch.reused_validated_plan)
        self.assertEqual(
            [name for name, _value in workflow.calls],
            ["prepare", "prepare", "execute"],
        )

    def test_discovered_counts_are_available_for_the_gui_summary(self):
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(point_locations_csv="points.csv")
        )
        model.update_dataset_requirements(
            {
                "point_dataset_ids": ("a", "b"),
                "line_dataset_ids": ("edge",),
                "point_placement_count": 12,
                "line_path_count": 3,
                "line_segment_count": 27,
            }
        )

        self.assertEqual(model.point_placement_count, 12)
        self.assertEqual(model.line_path_count, 3)
        self.assertEqual(model.line_segment_count, 27)

    def test_missing_mapping_is_rejected_before_backend_prepare(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()
        model.set_point_dataset("fastener", "")

        with self.assertRaisesRegex(ValueError, "point:fastener"):
            model.prepare_preview(workflow)

        self.assertEqual(workflow.calls, [])

    def test_adapter_accepts_small_grim_service_contract(self):
        class Service:
            def make_request(self, **kwargs):
                return kwargs

            def discover_dataset_ids(self, point_csv=None, line_csv=None):
                return (point_csv, line_csv)

            def prepare(self, request):
                return request

            def execute(self, plan):
                return "out.grim"

        adapter = FeatureWorkflowAdapter.from_service(Service())
        self.assertEqual(
            adapter.discover(
                point_locations_csv="p.csv",
                line_locations_csv="l.csv",
                base_dir="ignored",
            ),
            ("p.csv", "l.csv"),
        )


@unittest.skipUnless(GUI_AVAILABLE, "PySide6 GUI dependency unavailable")
class FeatureAssemblyPanelQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_surface_binding_dialog_requires_ids_and_explicit_attestation(self):
        from PySide6.QtWidgets import QDialogButtonBox, QWidget

        parent = QWidget()
        dialog = feature_panel_module._SurfaceBindingDialog(parent)
        accept = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.assertFalse(accept.isEnabled())
        dialog.geometry_id_edit.setText("vehicle-r7")
        dialog.case_id_edit.setText("registration-case-42")
        self.assertFalse(accept.isEnabled())
        dialog.attestation.setChecked(True)
        self.assertTrue(accept.isEnabled())
        self.assertEqual(
            dialog.binding_values(), ("vehicle-r7", "registration-case-42")
        )
        dialog.close()
        parent.close()

    def test_external_body_binding_actions_gate_production_and_refresh_stale_files(self):
        from PySide6.QtWidgets import QMessageBox

        ghost_context = _isolated_ghost_backend()
        ghost = ghost_context.__enter__()
        self.addCleanup(ghost_context.__exit__, None, None, None)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "clean_vehicle.grim"
            surface = root / "vehicle.facet"
            _write_minimal_base_grim(base)
            surface.write_text("4 2\nexact mesh revision 7\n", encoding="utf-8")
            panel = FeatureAssemblyPanel(service=ghost.feature_workflow)
            try:
                panel.set_base_grim(str(base))
                panel.set_surface_mesh(str(surface))
                panel.surface_units.setCurrentIndex(
                    panel.surface_units.findData("meters")
                )
                panel._update_workflow_readiness()
                self.assertIn("Binding missing", panel.surface_binding_status.text())
                self.assertFalse(panel.check_surface_binding_button.isEnabled())
                self.assertTrue(panel.bind_surface_button.isEnabled())
                self.assertIn("reviewed body binding", panel.readiness_label.text())
                self.assertFalse(panel.preview_button.isEnabled())

                def run_synchronously(kind, operation, **_kwargs):
                    panel._active_kind = kind
                    try:
                        panel._operation_succeeded(operation())
                    finally:
                        panel._active_kind = ""

                with mock.patch.object(
                    panel,
                    "_prompt_surface_binding_details",
                    return_value=("vehicle-r7", "registration-case-42"),
                ), mock.patch.object(
                    panel, "_start_operation", side_effect=run_synchronously
                ):
                    panel.bind_selected_surface()

                sidecar = Path(str(surface) + ".assembly.json")
                self.assertTrue(sidecar.is_file())
                self.assertIn("Current reviewed binding", panel.surface_binding_status.text())
                self.assertIn("vehicle-r7", panel.surface_binding_status.text())

                # A stat change immediately makes the cached green check stale;
                # no repeated content hashing occurs in the readiness refresh.
                surface.write_text(
                    "4 2\nexact mesh revision 8 changed\n", encoding="utf-8"
                )
                panel._update_workflow_readiness()
                self.assertIn("check is stale", panel.surface_binding_status.text())
                self.assertFalse(panel.preview_button.isEnabled())

                with mock.patch.object(
                    panel,
                    "_prompt_surface_binding_details",
                    return_value=("vehicle-r8", "registration-case-43"),
                ), mock.patch.object(
                    QMessageBox,
                    "warning",
                    return_value=QMessageBox.StandardButton.Yes,
                ) as overwrite_prompt, mock.patch.object(
                    panel, "_start_operation", side_effect=run_synchronously
                ):
                    panel.bind_selected_surface()
                overwrite_prompt.assert_called_once()
                self.assertIn("vehicle-r8", panel.surface_binding_status.text())

                # The separate Check action also restores readiness from an
                # existing canonical sidecar without rewriting it.
                panel._surface_binding_checked_key = None
                panel._surface_binding_checked = None
                panel._update_workflow_readiness()
                self.assertIn("not checked", panel.surface_binding_status.text())
                self.assertTrue(panel.check_surface_binding_button.isEnabled())
                before = sidecar.read_bytes()
                with mock.patch.object(
                    panel, "_start_operation", side_effect=run_synchronously
                ):
                    panel.check_selected_surface_binding()
                self.assertEqual(sidecar.read_bytes(), before)
                self.assertIn("Current reviewed binding", panel.surface_binding_status.text())
            finally:
                _close_panel_without_prompt(panel)

    def test_recipe_load_restores_named_variant_and_tracks_later_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            body = root / "body.grim"
            points = root / "points.csv"
            response = root / "fastener.grim"
            body.write_bytes(b"body")
            points.write_bytes(b"points")
            response.write_bytes(b"delta")
            recipe = write_feature_assembly_recipe(
                FeatureAssemblyValues(
                    base_grim=str(body),
                    output_grim=str(root / "assembled.grim"),
                    coordinate_units="meters",
                    point_locations_csv=str(points),
                    point_datasets={"fastener": str(response)},
                    excluded_point_placement_ids={"bolt_002"},
                    expected_host_material="coated-aluminum-v2",
                ),
                root / "vehicle",
                name="Test vehicle",
                variant="No rear fastener",
            )
            panel = FeatureAssemblyPanel(service=_FakeWorkflow())
            try:
                loaded = panel.load_recipe_path(recipe, refresh=False)
                self.assertEqual(loaded.variant, "No rear fastener")
                self.assertEqual(panel.recipe_name_edit.text(), "Test vehicle")
                self.assertEqual(
                    panel.recipe_variant_edit.text(), "No rear fastener"
                )
                self.assertFalse(panel._recipe_dirty)
                self.assertEqual(panel.skin_tol.value(), 1.0)
                self.assertEqual(
                    panel.expected_host_material.text(), "coated-aluminum-v2"
                )
                self.assertIn("saved", panel.recipe_status_label.text())
                self.assertEqual(
                    panel.point_mapping.mapping(), {"fastener": str(response.resolve())}
                )
                self.assertEqual(
                    panel.model.values.excluded_point_placement_ids, {"bolt_002"}
                )

                panel.recipe_variant_edit.setFocus()
                panel.recipe_variant_edit.insert(" updated")
                self.app.processEvents()

                self.assertTrue(panel._recipe_dirty)
                self.assertIn("modified", panel.recipe_status_label.text())
            finally:
                _close_panel_without_prompt(panel)

    def test_validation_qa_rows_link_back_to_spatial_instances(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        try:
            panel.model.update_dataset_requirements(
                {
                    "point_dataset_ids": ("fastener",),
                    "line_dataset_ids": ("seal",),
                    "point_instances": (("bolt_1", "fastener"),),
                    "line_instances": (("door_gap", "seal", 2),),
                }
            )
            panel._apply_requirements_to_tables()
            plan = SimpleNamespace(
                skin_limit_m=1.0e-3,
                validation_warnings=(
                    "fastener response has no certified library manifest",
                ),
                point_records=(
                    {
                        "placement_id": "bolt_1",
                        "dataset_id": "fastener",
                        "skin_offset_m": 2.0e-4,
                    },
                ),
                line_records=(
                    {
                        "line_id": "door_gap",
                        "dataset_id": "seal",
                        "max_skin_offset_m": 3.0e-4,
                        "max_normal_error_deg": 2.5,
                    },
                ),
            )

            panel._show_validation_qa(plan)

            self.assertEqual(panel.validation_qa_table.rowCount(), 2)
            self.assertIn("2 enabled", panel.validation_qa_label.text())
            self.assertIn("1 production QA warning", panel.validation_qa_label.text())
            self.assertFalse(panel.validation_warning_label.isHidden())
            self.assertIn("RELEASE WARNINGS", panel.validation_warning_label.text())
            self.assertFalse(panel.validation_warning_ack.isHidden())
            self.assertFalse(panel.validation_warning_ack.isChecked())
            self.assertIn(
                "no certified library manifest",
                panel.validation_warning_label.text(),
            )
            panel._qa_row_clicked(0, 0)
            current = panel.spatial_feature_tree.currentItem()
            self.assertIsNotNone(current)
            self.assertIn("door_gap", current.text(0))
        finally:
            _close_panel_without_prompt(panel)

    def test_public_busy_and_close_contract_starts_idle(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        self.assertFalse(panel.is_busy())
        self.assertTrue(panel.can_close())
        self.assertIn("locally and on HPC", panel.point_help_label.text())
        self.assertIn(",".join(POINT_PLACEMENT_COLUMNS), panel.point_schema_label.text())
        self.assertIn(",".join(LINE_PLACEMENT_COLUMNS), panel.line_schema_label.text())
        self.assertEqual(panel.input_preview_button.text(), "Preview geometry")
        self.assertEqual(panel.preview_button.text(), "Validate placements")
        self.assertIn("No Assembly operation", panel.status_label.text())
        self.assertFalse(panel.advanced_section.header.isChecked())
        self.assertTrue(panel.skin_tol.isEnabled())
        self.assertEqual(panel.validation_profile.currentData()[0], "production")
        self.assertEqual(panel._validation_profile_flags(), (False, True))
        self.assertEqual(panel.skin_tol.suffix().strip(), "mm")
        self.assertAlmostEqual(panel.skin_tol.value(), 1.0)
        self.assertLessEqual(panel.skin_tol.singleStep(), 0.01)
        self.assertEqual(panel.phase_tol.maximum(), 90.0)
        self.assertFalse(panel.scan_button.isEnabled())
        self.assertFalse(panel.preview_button.isEnabled())
        self.assertFalse(panel.build_button.isEnabled())
        self.assertFalse(panel.operation_progress.isVisible())
        self.assertFalse(panel.cancel_operation_button.isVisible())
        self.assertIn("Preview Layers → Show", panel.preview_help_label.text())
        self.assertIn(
            "Spatial Feature Configuration → Use",
            panel.preview_help_label.text(),
        )
        _close_panel_without_prompt(panel)

    def test_qa_defaults_reset_in_display_units_without_changing_profile(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        panel.skin_tol.setValue(22.0)
        panel.phase_tol.setValue(75.0)
        panel.normal_tol.setValue(33.0)
        panel.validation_profile.setCurrentIndex(1)

        panel.reset_qa_defaults_button.click()
        panel._pull_values()

        self.assertAlmostEqual(panel.skin_tol.value(), 1.0)
        self.assertAlmostEqual(panel.model.values.skin_tol_m, 1.0e-3)
        self.assertEqual(panel.phase_tol.value(), 15.0)
        self.assertEqual(panel.normal_tol.value(), 15.0)
        self.assertEqual(panel._validation_profile_flags(), (True, False))
        _close_panel_without_prompt(panel)

    def test_dirty_recipe_close_and_load_offer_save_discard_cancel(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            load_target = write_feature_assembly_recipe(
                FeatureAssemblyValues(),
                root / "load_target",
                name="Load target",
                variant="Baseline",
            )
            panel = FeatureAssemblyPanel(service=_FakeWorkflow())
            panel._recipe_dirty = True

            with mock.patch.object(
                QMessageBox,
                "warning",
                return_value=QMessageBox.StandardButton.Cancel,
            ):
                self.assertFalse(panel.request_close())
            self.assertTrue(panel._recipe_dirty)

            with mock.patch.object(
                QFileDialog,
                "getOpenFileName",
                return_value=(str(load_target), ""),
            ), mock.patch.object(
                QMessageBox,
                "warning",
                return_value=QMessageBox.StandardButton.Cancel,
            ), mock.patch.object(panel, "load_recipe_path") as load_mock:
                panel._load_recipe_dialog()
                load_mock.assert_not_called()

            panel._recipe_path = root / "saved.assembly.json"
            with mock.patch.object(
                QMessageBox,
                "warning",
                return_value=QMessageBox.StandardButton.Save,
            ):
                self.assertTrue(panel.request_close())
            self.assertTrue(panel._recipe_path.is_file())
            self.assertFalse(panel._recipe_dirty)

            panel._recipe_dirty = True
            with mock.patch.object(
                QMessageBox,
                "warning",
                return_value=QMessageBox.StandardButton.Discard,
            ):
                self.assertTrue(panel.request_close())
            _close_panel_without_prompt(panel)

    def test_body_preflight_controls_readiness_for_external_bor_and_malformed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            external = root / "external.grim"
            embedded = root / "bor.grim"
            malformed = root / "bad.grim"
            _write_minimal_base_grim(external)
            _write_minimal_base_grim(embedded, embedded_bor=True)
            malformed.write_bytes(b"broken")
            panel = FeatureAssemblyPanel(service=_FakeWorkflow())
            panel.validation_profile.setCurrentIndex(1)

            panel.base_picker.set_path(str(external))
            panel._update_workflow_readiness()
            self.assertIn("✓ valid body GRIM", panel.readiness_label.text())
            self.assertIn("surface mesh", panel.next_step_label.text())

            panel.base_picker.set_path(str(embedded))
            panel._update_workflow_readiness()
            self.assertIn("✓ valid body GRIM", panel.readiness_label.text())
            self.assertNotIn("surface mesh", panel.next_step_label.text())

            panel.base_picker.set_path(str(malformed))
            panel._update_workflow_readiness()
            self.assertIn("○ valid body GRIM", panel.readiness_label.text())
            self.assertIn("Invalid GRIM container", panel.next_step_label.text())
            _close_panel_without_prompt(panel)

    def test_assembly_button_requires_current_validation_and_warning_waiver(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "body.grim"
            surface = root / "body.facet"
            points = root / "points.csv"
            response = root / "fastener.grim"
            _write_minimal_base_grim(base, embedded_bor=True)
            surface.write_text("0 0\n", encoding="utf-8")
            points.write_bytes(b"stable placements")
            response.write_bytes(b"stable response")
            workflow = _FakeWorkflow()
            panel = FeatureAssemblyPanel(service=workflow)
            panel.base_picker.set_path(str(base))
            panel.surface_picker.set_path(str(surface))
            panel.point_csv_picker.set_path(str(points))
            panel.output_picker.set_path(str(root / "assembled.grim"))
            panel.expected_host_material.setText("")
            panel._pull_values()
            panel.model.update_dataset_requirements(
                {"point_dataset_ids": ("fastener",), "line_dataset_ids": ()}
            )
            panel.model.set_point_dataset("fastener", str(response))
            panel._apply_requirements_to_tables()
            panel.point_mapping.set_host_material("fastener", "paint-stack-v3")
            panel._update_workflow_readiness()

            self.assertEqual(
                panel.model.values.point_host_materials,
                {"fastener": "paint-stack-v3"},
            )

            self.assertTrue(panel.preview_button.isEnabled())
            self.assertFalse(panel.build_button.isEnabled())
            self.assertIn("run Validate", panel.next_step_label.text())

            plan = panel.model.prepare_preview(workflow)
            panel._show_validation_qa(plan)
            panel._validated_plan_current = True
            panel._update_workflow_readiness()
            self.assertTrue(panel.build_button.isEnabled())

            warning_plan = SimpleNamespace(
                validation_warnings=("legacy response applicability is unproven",),
                point_records=(),
                line_records=(),
            )
            panel._show_validation_qa(warning_plan)
            panel._validated_plan_current = True
            panel._update_workflow_readiness()
            self.assertFalse(panel.build_button.isEnabled())
            panel.validation_warning_ack.setChecked(True)
            self.assertTrue(panel.build_button.isEnabled())

            panel.normal_tol.setValue(16.0)
            self.app.processEvents()
            self.assertFalse(panel._validated_plan_current)
            self.assertFalse(panel.build_button.isEnabled())
            self.assertFalse(panel.validation_warning_ack.isChecked())
            _close_panel_without_prompt(panel)

    def test_progress_display_and_cancel_request_are_explicit(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        worker = SimpleNamespace(request_cancel=mock.Mock())
        try:
            panel._active_kind = "build"
            panel._worker = worker
            panel.operation_progress.setVisible(True)
            panel.cancel_operation_button.setVisible(True)
            panel.cancel_operation_button.setEnabled(True)

            panel._operation_progress(37, "Expanding line door_gap")
            self.assertEqual(panel.operation_progress.value(), 37)
            self.assertIn("door_gap", panel.operation_progress.format())

            panel.request_cancel()
            worker.request_cancel.assert_called_once_with()
            self.assertFalse(panel.cancel_operation_button.isEnabled())
            self.assertIn(
                "Cancelling assembly safely", panel.operation_progress.format()
            )
            self.assertIn("no partial output", panel.status_label.text())
        finally:
            panel._active_kind = ""
            panel._worker = None
            _close_panel_without_prompt(panel)

    def test_validation_worker_exposes_progress_and_cooperative_cancel(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        worker = SimpleNamespace(request_cancel=mock.Mock())
        try:
            panel._active_kind = "preview"
            panel._worker = worker
            panel._validated_plan_current = True
            panel._preview_is_current = True
            panel.model._prepared_plan_cache = object()
            panel._set_busy(True)

            self.assertFalse(panel.cancel_operation_button.isHidden())
            self.assertEqual(
                panel.cancel_operation_button.text(), "Cancel validation"
            )
            panel._operation_progress(55, "Checking line paths")
            self.assertEqual(panel.operation_progress.value(), 55)
            self.assertIn("Checking line paths", panel.operation_progress.format())

            panel.request_cancel()
            worker.request_cancel.assert_called_once_with()
            self.assertFalse(panel.cancel_operation_button.isEnabled())
            self.assertIn(
                "Cancelling validation safely", panel.operation_progress.format()
            )
            self.assertIn("no reviewed plan", panel.status_label.text())

            panel._operation_cancelled("operator cancelled validation")
            self.assertIsNone(panel.model._prepared_plan_cache)
            self.assertFalse(panel._validated_plan_current)
            self.assertFalse(panel._preview_is_current)
            self.assertFalse(panel.build_button.isEnabled())
            self.assertIn("operator cancelled", panel.status_label.text())
        finally:
            panel._active_kind = ""
            panel._worker = None
            panel._set_busy(False)
            _close_panel_without_prompt(panel)

    def test_validate_action_starts_a_cooperative_worker(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        panel.model._prepared_plan_cache = object()
        try:
            with mock.patch.object(panel, "_start_operation") as start:
                panel.validate_and_preview()

            start.assert_called_once()
            self.assertEqual(start.call_args.args[0], "preview")
            self.assertTrue(start.call_args.kwargs["cooperative"])
            self.assertIsNone(panel.model._prepared_plan_cache)
            self.assertFalse(panel._validated_plan_current)
            self.assertIn("Assembly remains locked", panel.validation_qa_label.text())
        finally:
            _close_panel_without_prompt(panel)

    def test_reselecting_unchanged_csv_preserves_response_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "points.csv"
            csv_path.write_text("stable bytes\n", encoding="utf-8")
            panel = FeatureAssemblyPanel(service=_FakeWorkflow())
            panel.point_csv_picker.set_path(str(csv_path))
            panel.model.values.point_locations_csv = str(csv_path)
            panel.model.update_dataset_requirements(
                {"point_dataset_ids": ("fastener",), "line_dataset_ids": ()}
            )
            panel.model.set_point_dataset("fastener", "fastener.grim")
            panel._apply_requirements_to_tables()

            panel._placement_csv_changed("point")

            self.assertEqual(
                panel.point_mapping.mapping(), {"fastener": "fastener.grim"}
            )
            self.assertEqual(panel.model.point_dataset_ids, ("fastener",))
            self.assertFalse(panel.job_is_running())
            _close_panel_without_prompt(panel)

    def test_selecting_different_csv_resets_same_named_feature_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.csv"
            second = Path(folder) / "second.csv"
            first.write_text("first bytes\n", encoding="utf-8")
            second.write_text("second bytes\n", encoding="utf-8")
            panel = FeatureAssemblyPanel(service=_FakeWorkflow())
            panel.point_csv_picker.set_path(str(first))
            panel.model.values.point_locations_csv = str(first)
            panel.model.update_dataset_requirements(
                {
                    "point_dataset_ids": ("family",),
                    "line_dataset_ids": (),
                    "point_instances": (("p1", "family"),),
                }
            )
            panel.model.set_feature_instance_enabled("point", "p1", False)
            panel._apply_requirements_to_tables()

            panel.point_csv_picker.set_path(str(second))
            with mock.patch.object(panel, "refresh_dataset_ids") as refresh:
                panel._placement_csv_changed("point")

            self.assertEqual(panel.model.values.excluded_point_placement_ids, set())
            self.assertEqual(panel.model.point_dataset_ids, ())
            refresh.assert_called_once_with()
            _close_panel_without_prompt(panel)

    def test_spatial_tree_group_use_recursively_updates_model_membership(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        panel.model.values.point_locations_csv = "points.csv"
        panel.point_csv_picker.set_path("points.csv")
        panel.model.update_dataset_requirements(
            {
                "point_dataset_ids": ("family",),
                "line_dataset_ids": (),
                "point_instances": (("p1", "family"), ("p2", "family")),
            }
        )
        panel._apply_requirements_to_tables()
        body = panel.spatial_feature_tree.topLevelItem(0)
        point_root = body.child(0)
        self.assertEqual(body.text(0), "Body")
        self.assertEqual(point_root.text(0), "Point features (2)")
        family = point_root.child(0)

        from PySide6.QtCore import Qt

        family.setCheckState(2, Qt.CheckState.Unchecked)
        self.app.processEvents()

        self.assertEqual(
            panel.model.values.excluded_point_placement_ids,
            {"p1", "p2"},
        )
        self.assertEqual(panel.model.enabled_point_placement_ids, ())
        self.assertFalse(panel.input_preview_button.isEnabled())
        self.assertIn("enable at least one", panel.status_label.text().lower())
        _close_panel_without_prompt(panel)

    def test_tree_leaf_selection_reaches_real_backend_field_and_provenance(self):
        """Close the UI-to-saved-artifact loop on a non-BoR external body."""

        ghost_context = _isolated_ghost_backend()
        ghost = ghost_context.__enter__()
        self.addCleanup(ghost_context.__exit__, None, None, None)
        feature_sum = ghost.feature_sum
        feature_workflow = ghost.feature_workflow
        frequency_ghz = 2.0
        azimuths_deg = [0.0, 45.0]
        elevations_deg = [30.0, 60.0]
        adapter = FeatureWorkflowAdapter.from_module(feature_workflow)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "clean_box.grim"
            surface = root / "rectangular_box.facet"
            coordinates = root / "top_seals.csv"
            response = root / "shared_seal_delta.grim"
            selected_output = root / "selected_seal.grim"
            all_output = root / "all_seals.grim"
            disabled_output = root / "disabled_seal_only.grim"

            _write_closed_box_facet(surface)
            _write_isotropic_line_delta(
                response,
                frequency_ghz=frequency_ghz,
                installed_coefficient=0.31 - 0.12j,
                c0=ghost.c0,
                psi_hh_deg=ghost.psi_hh_deg,
                psi_vv_deg=ghost.psi_vv_deg,
            )
            coordinates.write_text(
                ",".join(LINE_PLACEMENT_COLUMNS)
                + "\n"
                + "seal_keep,door_seal,1,-0.08,-0.10,0.10,"
                + "0.08,-0.10,0.10,0,0,1,0,0,1\n"
                + "seal_drop,door_seal,1,-0.08,0.10,0.10,"
                + "0.08,0.10,0.10,0,0,1,0,0,1\n",
                encoding="utf-8",
            )
            feature_sum.export_radar_grim(
                str(base),
                bor_result=None,
                placements=[],
                frequencies_ghz=[frequency_ghz],
                azimuths_deg=azimuths_deg,
                elevations_deg=elevations_deg,
                axis_az_deg=0.0,
                axis_el_deg=0.0,
                roll_deg=0.0,
                history="zero-field non-BoR rectangular-box fixture",
            )

            panel = FeatureAssemblyPanel(service=adapter)
            try:
                panel.set_base_grim(str(base))
                panel.set_surface_mesh(str(surface))
                panel.set_line_csv(str(coordinates), discover=False)
                panel.set_output_grim(str(selected_output))
                panel.coordinate_units.setCurrentIndex(
                    panel.coordinate_units.findData("meters")
                )
                panel.surface_units.setCurrentIndex(
                    panel.surface_units.findData("meters")
                )
                # This fixture intentionally predates certified feature manifests.
                panel.validation_profile.setCurrentIndex(1)
                panel.skin_tol.setValue(1.0e-5)  # millimeters = 1e-8 meters
                panel.phase_tol.setValue(0.1)
                panel.normal_tol.setValue(2.0)
                panel._pull_values()

                requirements = panel.model.discover_dataset_ids(adapter)
                self.assertEqual(requirements.line_dataset_ids, ("door_seal",))
                self.assertEqual(
                    panel.model.line_instances,
                    (
                        ("seal_keep", "door_seal", 1),
                        ("seal_drop", "door_seal", 1),
                    ),
                )
                panel.model.set_line_dataset("door_seal", str(response))
                panel._apply_requirements_to_tables()

                body = panel.spatial_feature_tree.topLevelItem(0)
                line_root = body.child(1)
                family = line_root.child(0)
                self.assertEqual(family.childCount(), 2)
                self.assertEqual(family.child(0).text(0), "seal_keep (1 segment(s))")
                self.assertEqual(family.child(1).text(0), "seal_drop (1 segment(s))")

                from PySide6.QtCore import Qt

                family.child(1).setCheckState(2, Qt.CheckState.Unchecked)
                self.app.processEvents()
                self.assertEqual(panel.model.enabled_line_ids, ("seal_keep",))
                self.assertEqual(
                    panel.model.values.excluded_line_ids, {"seal_drop"}
                )

                # This is deliberately synchronous: it exercises the same
                # panel model and real service as the GUI worker without
                # introducing thread timing into the numerical regression.
                dispatch = panel.model.assemble(adapter)
                self.assertEqual(Path(dispatch.output_path), selected_output.resolve())
                self.assertEqual(
                    dispatch.plan.dataset_requirements.line_instances,
                    (("seal_keep", "door_seal", 1),),
                )

                # Real-backend counterfactuals identify the disabled leaf's
                # field. Coherent linearity requires full-selected to equal
                # disabled-only minus the same clean external body.
                all_request = replace(
                    dispatch.plan.request,
                    output_grim=all_output,
                    enabled_line_ids=None,
                )
                disabled_request = replace(
                    dispatch.plan.request,
                    output_grim=disabled_output,
                    enabled_line_ids=("seal_drop",),
                )
                adapter.execute(adapter.prepare(all_request))
                adapter.execute(adapter.prepare(disabled_request))

                def load_complex(path: Path) -> np.ndarray:
                    with np.load(path, allow_pickle=False) as payload:
                        return np.asarray(
                            payload["rcs_amp_real"]
                            + 1j * payload["rcs_amp_imag"],
                            dtype=np.complex128,
                        )

                clean_field = load_complex(base)
                selected_field = load_complex(selected_output)
                all_field = load_complex(all_output)
                disabled_field = load_complex(disabled_output)
                disabled_contribution = disabled_field - clean_field
                self.assertGreater(
                    float(np.linalg.norm(selected_field - clean_field)), 1.0e-10
                )
                self.assertGreater(
                    float(np.linalg.norm(disabled_contribution)), 1.0e-10
                )
                np.testing.assert_allclose(
                    all_field - selected_field,
                    disabled_contribution,
                    rtol=2.0e-12,
                    atol=2.0e-13,
                )

                with np.load(selected_output, allow_pickle=False) as payload:
                    raw_provenance = np.asarray(
                        payload["feature_provenance_json"]
                    ).reshape(()).item()
                if isinstance(raw_provenance, bytes):
                    raw_provenance = raw_provenance.decode("utf-8")
                provenance = json.loads(str(raw_provenance))[-1]
                details = provenance["details"]
                self.assertEqual(provenance["line_feature_count"], 1)
                self.assertEqual(
                    details["enabled_selection"],
                    {
                        "point_placement_ids": [],
                        "line_ids": ["seal_keep"],
                    },
                )
                self.assertEqual(
                    [record["line_id"] for record in details["placements"]],
                    ["seal_keep"],
                )
            finally:
                _close_panel_without_prompt(panel)

    def test_spatial_tree_filter_matches_instance_dataset_and_response(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        panel.model.update_dataset_requirements(
            {
                "point_dataset_ids": ("hardware", "latch"),
                "line_dataset_ids": ("seal",),
                "point_instances": (
                    ("bolt_left", "hardware"),
                    ("rivet_right", "hardware"),
                    ("latch_center", "latch"),
                ),
                "line_instances": (("door_gap", "seal", 3),),
            }
        )
        panel.model.set_point_dataset(
            "hardware", "C:/responses/shared_fastener.grim"
        )
        panel.model.set_point_dataset("latch", "C:/responses/latch.grim")
        panel.model.set_line_dataset("seal", "C:/responses/door_seal.grim")
        panel._apply_requirements_to_tables()

        body = panel.spatial_feature_tree.topLevelItem(0)
        point_root = body.child(0)
        line_root = body.child(1)
        hardware = point_root.child(0)
        latch = point_root.child(1)
        seal = line_root.child(0)
        self.assertTrue(body.isExpanded())
        self.assertTrue(point_root.isExpanded())
        self.assertTrue(line_root.isExpanded())
        self.assertFalse(hardware.isExpanded())
        self.assertFalse(latch.isExpanded())
        self.assertFalse(seal.isExpanded())

        panel.spatial_feature_filter.setText("rivet_right")
        self.app.processEvents()

        self.assertFalse(body.isHidden())
        self.assertFalse(point_root.isHidden())
        self.assertFalse(hardware.isHidden())
        self.assertTrue(hardware.child(0).isHidden())
        self.assertFalse(hardware.child(1).isHidden())
        self.assertTrue(latch.isHidden())
        self.assertTrue(line_root.isHidden())
        self.assertTrue(hardware.isExpanded())
        self.assertEqual(panel.model.values.excluded_point_placement_ids, set())

        panel.spatial_feature_filter.setText("door_seal.grim")
        self.app.processEvents()

        self.assertTrue(point_root.isHidden())
        self.assertFalse(line_root.isHidden())
        self.assertFalse(seal.isHidden())
        self.assertFalse(seal.child(0).isHidden())

        panel.spatial_feature_filter.clear()
        self.app.processEvents()

        self.assertFalse(point_root.isHidden())
        self.assertFalse(line_root.isHidden())
        self.assertFalse(hardware.child(0).isHidden())
        self.assertTrue(body.isExpanded())
        self.assertTrue(point_root.isExpanded())
        self.assertTrue(line_root.isExpanded())
        self.assertFalse(hardware.isExpanded())
        self.assertFalse(latch.isExpanded())
        self.assertFalse(seal.isExpanded())
        _close_panel_without_prompt(panel)

    def test_copy_full_selection_keeps_large_trade_study_membership_exact(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        instances = tuple(
            (f"fastener_{index:03d}", "fastener") for index in range(12)
        )
        panel.model.update_dataset_requirements(
            {
                "point_dataset_ids": ("fastener",),
                "line_dataset_ids": (),
                "point_instances": instances,
            }
        )
        panel.model.set_excluded_feature_instances(
            point_ids=[value[0] for value in instances[:11]], line_ids=()
        )
        panel._apply_requirements_to_tables()

        displayed = panel.spatial_selection_summary.text()
        self.assertIn("… +3 more", displayed)
        self.assertNotIn("fastener_010", displayed)
        self.assertTrue(panel.copy_spatial_selection_button.isEnabled())

        self.app.clipboard().clear()
        panel.copy_spatial_selection_button.click()
        self.app.processEvents()

        copied = self.app.clipboard().text()
        self.assertEqual(copied, panel.model.feature_selection_summary())
        self.assertIn("fastener_010", copied)
        self.assertNotIn("more", copied)
        self.assertIn("Copied the full", panel.status_label.text())
        _close_panel_without_prompt(panel)

    def test_input_change_marks_a_current_preview_stale(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        messages = []
        panel.preview_stale.connect(messages.append)
        panel._preview_is_current = True

        panel.set_surface_mesh("body.stl")

        self.assertFalse(panel._preview_is_current)
        self.assertEqual(len(messages), 1)
        self.assertIn("out of date", messages[0])
        self.assertIn("out of date", panel.status_label.text())
        _close_panel_without_prompt(panel)

    def test_readiness_disables_alias_output_and_missing_mapped_response(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "body.grim"
            surface = root / "body.facet"
            points = root / "points.csv"
            response = root / "fastener.grim"
            _write_minimal_base_grim(base, embedded_bor=True)
            surface.write_text("0 0\n", encoding="utf-8")
            points.write_bytes(b"stable placement bytes")
            response.write_bytes(b"feature response")

            panel = FeatureAssemblyPanel(service=_FakeWorkflow())
            panel.base_picker.set_path(str(base))
            panel.surface_picker.set_path(str(surface))
            panel.expected_host_material.setText("PEC")
            panel.point_csv_picker.set_path(str(points))
            panel.output_picker.set_path(str(base))
            panel.model.values.point_locations_csv = str(points)
            panel.model.update_dataset_requirements(
                {"point_dataset_ids": ("fastener",), "line_dataset_ids": ()}
            )
            panel.model.set_point_dataset("fastener", str(response))
            panel._apply_requirements_to_tables()
            panel._update_workflow_readiness()

            self.assertFalse(panel.preview_button.isEnabled())
            self.assertFalse(panel.build_button.isEnabled())
            self.assertIn("does not alias", panel.next_step_label.text())

            panel.output_picker.set_path(str(root / "assembled.grim"))
            panel.point_mapping.set_path(
                "fastener", str(root / "missing_response.grim")
            )
            panel._update_workflow_readiness()

            self.assertFalse(panel.preview_button.isEnabled())
            self.assertFalse(panel.build_button.isEnabled())
            self.assertIn("existing .grim", panel.next_step_label.text())
            _close_panel_without_prompt(panel)

    def test_loaded_catalog_selects_only_saved_grim_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / "clean body.grim"
            saved.write_bytes(b"artifact")
            dirty_path = Path(folder) / "old source.grim"
            dirty_path.write_bytes(b"old artifact")
            wrong_type = Path(folder) / "response.csv"
            wrong_type.write_text("not a grim", encoding="utf-8")

            panel = FeatureAssemblyPanel(service=_FakeWorkflow())
            panel.set_loaded_dataset_catalog(
                [
                    {
                        "stable_id": "clean-id",
                        "display_name": "Clean body",
                        "file_path": str(saved),
                    },
                    {
                        "id": "dirty-id",
                        "name": "Derived result",
                        "source_path": str(dirty_path),
                        "is_dirty": True,
                    },
                    ("wrong-id", "Wrong type", str(wrong_type), False),
                    SimpleNamespace(
                        dataset_id="missing-id",
                        name="Missing artifact",
                        source=str(Path(folder) / "missing.grim"),
                    ),
                ]
            )

            self.assertEqual(
                [entry.dataset_id for entry in panel.loaded_dataset_catalog()],
                ["clean-id", "dirty-id", "wrong-id", "missing-id"],
            )
            chooser = panel.base_picker.loaded_button
            self.assertIsNotNone(chooser)
            actions = chooser.catalog_menu().actions()
            selectable = [
                action
                for action in actions
                if action.isEnabled() and not action.isSeparator()
            ]
            self.assertEqual(len(selectable), 1)
            self.assertEqual(selectable[0].data(), "clean-id")
            self.assertTrue(
                any(
                    "Save unsaved derived datasets first" in action.text()
                    for action in actions
                )
            )

            selectable[0].trigger()
            self.assertEqual(panel.base_picker.path(), str(saved))
            self.assertEqual(
                panel.output_picker.path(),
                str(saved.with_name("clean body_features.grim")),
            )

            panel.point_mapping.set_dataset_ids(("fastener",))
            response_chooser = panel.point_mapping.loaded_dataset_button(
                "fastener"
            )
            response_action = next(
                action
                for action in response_chooser.catalog_menu().actions()
                if action.data() == "clean-id"
            )
            response_action.trigger()
            self.assertEqual(
                panel.point_mapping.mapping()["fastener"], str(saved)
            )
            panel.line_mapping.set_dataset_ids(("panel_gap",))
            line_action = next(
                action
                for action in panel.line_mapping.loaded_dataset_button(
                    "panel_gap"
                ).catalog_menu().actions()
                if action.data() == "clean-id"
            )
            line_action.trigger()
            self.assertEqual(
                panel.line_mapping.mapping()["panel_gap"], str(saved)
            )

            chooser.catalog_menu().aboutToShow.emit()
            self.assertIn(
                "Save unsaved derived datasets first", panel.status_label.text()
            )
            _close_panel_without_prompt(panel)

    def test_loaded_catalog_rejects_duplicate_stable_ids_atomically(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        panel.set_loaded_dataset_catalog(
            [{"id": "one", "name": "First", "path": ""}]
        )

        with self.assertRaisesRegex(ValueError, "must be unique"):
            panel.set_loaded_dataset_catalog(
                [
                    {"id": "same", "name": "First", "path": ""},
                    {"id": "same", "name": "Second", "path": ""},
                ]
            )

        self.assertEqual(
            [entry.dataset_id for entry in panel.loaded_dataset_catalog()],
            ["one"],
        )
        _close_panel_without_prompt(panel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
