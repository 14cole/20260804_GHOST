from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

import isar_artifact
from grim_dataset import RcsGrid
from isar_artifact import (
    ISAR_ARTIFACT_SCHEMA,
    build_isar_manifest,
    load_isar_artifact,
    save_isar_artifact,
)


class TestIsarArtifact(unittest.TestCase):
    def _case(self):
        dataset = RcsGrid(
            [-1.0, 0.0, 1.0],
            [0.0],
            [9.0, 9.5, 10.0],
            ["VV"],
            rcs=np.ones((3, 1, 3, 1), dtype=np.complex64),
            source_path="source.grim",
            history="loaded and calibrated",
            units={"frequency": "GHz", "time_convention": "exp(+jwt)"},
            extra={
                "phase_reference": "fixed origin",
                "large_passthrough": np.ones(1000),
            },
        )
        params = {
            "bands": [[0, 1, 2]],
            "freq_indices_sorted": [0, 1, 2],
            "freq_hz": np.asarray([9.0e9, 9.5e9, 10.0e9]),
            "elev_idx": 0,
            "elevation_deg": 0.0,
            "pol_idx": 0,
            "recon": "accurate",
            "window_name": "Hamming",
            "unit_name": "m",
            "az_target_deg": None,
            "az_center_deg": None,
            "l1_strength": 0.05,
            "l1_iters": 100,
            "flip_x": False,
            "flip_y": False,
            "isar_contract_assumptions": [
                "far-field monostatic acquisition geometry",
                "a stable or motion-compensated phase center",
            ],
            "legacy_metadata_attested": False,
        }
        complex_image = np.asarray([[1 + 2j, 3 + 4j]], dtype=np.complex64)
        band = {
            "magnitude": np.abs(complex_image),
            "complex_image": complex_image,
            "x_range": np.asarray([-0.5]),
            "y_range": np.asarray([-1.0, 1.0]),
            "az_values": np.asarray([-1.0, 0.0, 1.0]),
            "phase_coverage": 1.0,
            "source_selection_digest": "abc123",
            "sampling": {"range_resolution": 0.25},
        }
        return dataset, params, [band]

    def test_round_trip_preserves_complex_image_and_manifest(self):
        dataset, params, bands = self._case()
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        self.assertEqual(manifest["schema"], ISAR_ARTIFACT_SCHEMA)
        self.assertNotIn("large_passthrough", manifest["source"]["metadata"])
        self.assertTrue(manifest["formation"]["isar_contract_user_assumed"])
        self.assertEqual(
            manifest["formation"]["isar_contract_undeclared_fields"],
            params["isar_contract_assumptions"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_isar_artifact(Path(directory) / "result", bands, manifest)
            self.assertTrue(str(path).endswith(".isar.npz"))
            with np.load(path, allow_pickle=False) as archive:
                self.assertNotIn("band_0_magnitude", archive.files)
            restored_manifest, restored_bands = load_isar_artifact(path)
        self.assertEqual(restored_manifest["source"]["band_content_digests"], ["abc123"])
        self.assertEqual(
            restored_manifest["source"]["selected_frequency_values_native"],
            [9.0, 9.5, 10.0],
        )
        self.assertEqual(
            restored_manifest["source"]["selected_frequency_values_hz"],
            [9.0e9, 9.5e9, 10.0e9],
        )
        self.assertEqual(restored_manifest["source"]["selected_polarization"], "VV")
        np.testing.assert_allclose(
            restored_bands[0]["complex_image"], bands[0]["complex_image"]
        )
        np.testing.assert_allclose(
            restored_bands[0]["magnitude"], bands[0]["magnitude"]
        )

    def test_small_complex_metadata_is_json_safe_and_preserved_explicitly(self):
        dataset, params, bands = self._case()
        dataset.extra["complex_scalar"] = np.complex64(1.5 - 2.25j)
        dataset.extra["complex_vector"] = np.asarray(
            [1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64
        )
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        metadata = manifest["source"]["metadata"]
        self.assertEqual(metadata["complex_scalar"]["__complex__"], [1.5, -2.25])
        self.assertEqual(
            metadata["complex_vector"][1]["__complex__"], [3.0, -4.0]
        )
        json.dumps(manifest, allow_nan=False)

    def test_v1_artifact_with_redundant_magnitude_remains_loadable(self):
        dataset, params, bands = self._case()
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        legacy_manifest = json.loads(json.dumps(manifest))
        legacy_manifest["schema"] = "grim.isar-result.v1"
        legacy_manifest["bands"][0].pop("magnitude_storage")
        with tempfile.TemporaryDirectory() as directory:
            path = save_isar_artifact(
                Path(directory) / "legacy",
                bands,
                legacy_manifest,
                compressed=False,
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertIn("band_0_magnitude", archive.files)
            restored_manifest, restored = load_isar_artifact(path)
        self.assertEqual(restored_manifest["schema"], "grim.isar-result.v1")
        np.testing.assert_allclose(restored[0]["magnitude"], bands[0]["magnitude"])

    def test_adaptive_storage_skips_deflate_for_noise_and_uses_it_for_zeros(self):
        dataset, params, bands = self._case()
        rng = np.random.default_rng(42)
        shape = (256, 256)
        x_axis = np.linspace(-1.0, 1.0, shape[0])
        y_axis = np.linspace(-2.0, 2.0, shape[1])

        def artifact(field):
            band = dict(bands[0])
            band["complex_image"] = np.asarray(field, dtype=np.complex64)
            band["magnitude"] = np.abs(band["complex_image"])
            band["x_range"] = x_axis
            band["y_range"] = y_axis
            manifest = build_isar_manifest(dataset, params, [band], 0.25)
            return [band], manifest

        noise = (
            rng.standard_normal(shape, dtype=np.float32)
            + 1j * rng.standard_normal(shape, dtype=np.float32)
        )
        zero = np.zeros(shape, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            noise_bands, noise_manifest = artifact(noise)
            noise_path = save_isar_artifact(
                Path(directory) / "noise", noise_bands, noise_manifest
            )
            zero_bands, zero_manifest = artifact(zero)
            zero_path = save_isar_artifact(
                Path(directory) / "zero", zero_bands, zero_manifest
            )
            with zipfile.ZipFile(noise_path) as archive:
                self.assertEqual(
                    archive.getinfo("band_0_complex_image.npy").compress_type,
                    zipfile.ZIP_STORED,
                )
            with zipfile.ZipFile(zero_path) as archive:
                self.assertEqual(
                    archive.getinfo("band_0_complex_image.npy").compress_type,
                    zipfile.ZIP_DEFLATED,
                )

    def test_adaptive_storage_samples_negative_stride_images(self):
        dataset, params, bands = self._case()
        rng = np.random.default_rng(7)
        noise = (
            rng.standard_normal((256, 256), dtype=np.float32)
            + 1j * rng.standard_normal((256, 256), dtype=np.float32)
        )[::-1, :]
        self.assertFalse(noise.flags.c_contiguous)
        decision = isar_artifact._compression_decision(
            [
                np.linspace(-1.0, 1.0, 256),
                np.linspace(-2.0, 2.0, 256),
                noise,
            ]
        )
        self.assertGreater(decision["sample_bytes"], 500_000)
        self.assertFalse(decision["compressed"])

        band = dict(bands[0])
        band["complex_image"] = noise
        band["magnitude"] = np.abs(noise)
        band["x_range"] = np.linspace(-1.0, 1.0, 256)
        band["y_range"] = np.linspace(-2.0, 2.0, 256)
        manifest = build_isar_manifest(dataset, params, [band], 0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = save_isar_artifact(
                Path(directory) / "flipped-noise",
                [band],
                manifest,
            )
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(
                    archive.getinfo("band_0_complex_image.npy").compress_type,
                    zipfile.ZIP_STORED,
                )
            _restored_manifest, restored = load_isar_artifact(path)
        np.testing.assert_array_equal(restored[0]["complex_image"], noise)

    def test_artifact_rejects_complex_axes_and_complex64_overflow(self):
        dataset, params, bands = self._case()
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        with tempfile.TemporaryDirectory() as directory:
            complex_axis = [dict(bands[0])]
            complex_axis[0]["x_range"] = np.asarray([-0.5 + 1.0j])
            with self.assertRaisesRegex(ValueError, "x_range.*real"):
                save_isar_artifact(
                    Path(directory) / "complex-axis",
                    complex_axis,
                    manifest,
                )

            overflow = [dict(bands[0])]
            overflow[0]["complex_image"] = np.asarray(
                [[1.0e300 + 0.0j, 1.0 + 0.0j]],
                dtype=np.complex128,
            )
            with self.assertRaisesRegex(ValueError, "finite complex64"):
                save_isar_artifact(
                    Path(directory) / "overflow",
                    overflow,
                    manifest,
                )

    def test_source_metadata_is_recursively_bounded(self):
        dataset, params, bands = self._case()
        deep = "leaf"
        for _ in range(12):
            deep = {"child": deep}
        dataset.extra.update(
            {
                "small_nested": {"labels": ["a", "b"], "gain": 2.0},
                "nested_grid": {"samples": np.ones(100_000)},
                "oversized_list": list(range(300)),
                "oversized_string": "x" * 40_000,
                "too_deep": deep,
            }
        )
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        metadata = manifest["source"]["metadata"]
        self.assertEqual(metadata["small_nested"]["labels"], ["a", "b"])
        self.assertNotIn("nested_grid", metadata)
        self.assertNotIn("oversized_list", metadata)
        self.assertNotIn("oversized_string", metadata)
        self.assertNotIn("too_deep", metadata)
        self.assertLess(
            len(json.dumps(metadata).encode("utf-8")),
            isar_artifact._MAX_METADATA_JSON_BYTES,
        )

    def test_loader_enforces_array_budget_before_numpy_load(self):
        dataset, params, bands = self._case()
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = save_isar_artifact(
                Path(directory) / "bounded",
                bands,
                manifest,
                compressed=False,
            )
            with (
                mock.patch.object(
                    isar_artifact,
                    "_MAX_ARTIFACT_ARRAY_CELLS",
                    1,
                ),
                mock.patch.object(
                    isar_artifact.np,
                    "load",
                    side_effect=AssertionError("np.load ran before preflight"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "declares 2 cells"):
                    load_isar_artifact(path)

    def test_save_enforces_numerical_payload_budget(self):
        dataset, params, bands = self._case()
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                isar_artifact,
                "_MAX_ARTIFACT_NUMERIC_BYTES",
                8,
            ):
                with self.assertRaisesRegex(ValueError, "numerical payload"):
                    save_isar_artifact(
                        Path(directory) / "too-large",
                        bands,
                        manifest,
                    )

    def test_v1_redundant_magnitude_is_validated(self):
        dataset, params, bands = self._case()
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        legacy_manifest = json.loads(json.dumps(manifest))
        legacy_manifest["schema"] = "grim.isar-result.v1"
        legacy_manifest["bands"][0].pop("magnitude_storage")

        def write(path, magnitude):
            np.savez(
                path,
                manifest_json=np.asarray(json.dumps(legacy_manifest)),
                band_0_x_range=bands[0]["x_range"],
                band_0_y_range=bands[0]["y_range"],
                band_0_magnitude=np.asarray(magnitude),
                band_0_complex_image=bands[0]["complex_image"],
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.isar.npz"
            write(path, [[np.nan, np.nan]])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_isar_artifact(path)

            write(path, [[99.0, 99.0]])
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_isar_artifact(path)

    def test_manifest_refuses_unbound_source_content(self):
        dataset, params, bands = self._case()
        del bands[0]["source_selection_digest"]
        with self.assertRaisesRegex(ValueError, "source digest"):
            build_isar_manifest(dataset, params, bands, 0.25)

    def test_complex_image_controls_numerical_resolution_not_display_magnitude(self):
        dataset, params, bands = self._case()
        bands[0]["magnitude"] = np.asarray([[99.0]], dtype=np.float32)
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        self.assertEqual(manifest["bands"][0]["shape"], [1, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = save_isar_artifact(Path(directory) / "result", bands, manifest)
            _restored_manifest, restored = load_isar_artifact(path)
        np.testing.assert_allclose(
            restored[0]["magnitude"], np.abs(bands[0]["complex_image"])
        )

    def test_save_rejects_manifest_band_count_and_axis_shape_mismatch(self):
        dataset, params, bands = self._case()
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        with tempfile.TemporaryDirectory() as directory:
            extra_manifest = dict(manifest)
            extra_manifest["bands"] = list(manifest["bands"]) * 2
            with self.assertRaisesRegex(ValueError, "band count"):
                save_isar_artifact(
                    Path(directory) / "wrong-count", bands, extra_manifest
                )

            malformed = [dict(bands[0])]
            malformed[0]["x_range"] = np.asarray([-1.0, 1.0])
            with self.assertRaisesRegex(
                ValueError, "does not match axes|shape disagrees"
            ):
                save_isar_artifact(
                    Path(directory) / "wrong-shape", malformed, manifest
                )

    def test_save_rejects_nonfinite_or_nonmonotonic_numerical_arrays(self):
        dataset, params, bands = self._case()
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        with tempfile.TemporaryDirectory() as directory:
            nonfinite = [dict(bands[0])]
            nonfinite[0].pop("complex_image")
            nonfinite[0]["magnitude"] = np.asarray([[np.nan, 1.0]])
            nonfinite_manifest = dict(manifest)
            nonfinite_manifest["bands"] = [dict(manifest["bands"][0])]
            nonfinite_manifest["bands"][0]["complex_image_available"] = False
            nonfinite_manifest["bands"][0]["magnitude_storage"] = "stored"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                save_isar_artifact(
                    Path(directory) / "nonfinite", nonfinite, nonfinite_manifest
                )

            nonmonotonic = [dict(bands[0])]
            nonmonotonic[0]["y_range"] = np.asarray([1.0, -1.0])
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                save_isar_artifact(
                    Path(directory) / "nonmonotonic", nonmonotonic, manifest
                )

    def test_loader_rejects_arrays_that_disagree_with_manifest(self):
        dataset, params, bands = self._case()
        manifest = build_isar_manifest(dataset, params, bands, 0.25)
        manifest["bands"][0]["shape"] = [2, 1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.isar.npz"
            np.savez_compressed(
                path,
                manifest_json=np.asarray(
                    json.dumps(manifest, allow_nan=False)
                ),
                band_0_x_range=bands[0]["x_range"],
                band_0_y_range=bands[0]["y_range"],
                band_0_complex_image=bands[0]["complex_image"],
            )
            with self.assertRaisesRegex(ValueError, "shape disagrees"):
                load_isar_artifact(path)


if __name__ == "__main__":
    unittest.main()
