"""Response-role and provenance safety regressions for Assembly Tree."""

from __future__ import annotations

import json
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QAbstractItemView, QApplication

from assembly_tree import (
    AssemblyTree,
    BuildDialog,
    _ASSEMBLY_PROVENANCE_KEY,
    _ASSEMBLY_BASE_SHA256_KEY,
    _ASSEMBLY_BASE_RESPONSE_SHA256_KEY,
    _COMBINE_ROLE_KEY,
    _FEATURE_PROVENANCE_KEY,
    _MODE_AUTO,
    _MODE_COH,
    _MODE_INCOH,
    _RESPONSE_ROLE_BODY_PLUS_FEATURES,
    _RESPONSE_ROLE_FEATURES_ONLY_DELTA,
    _RESPONSE_ROLE_KEY,
    _ROLE_GRID,
    _SOURCE_MONOSTATIC_SHA256_KEY,
    _TYPE_BRANCH,
    _TYPE_ROOT,
    _assembly_response_physics_sha256,
    _attach,
    _b64_to_grid,
    _grid_to_b64,
    _dict_to_item,
    _item_to_dict,
    _node_mode,
    _set_node_mode,
    build_assembly_grid,
)
from grim_dataset import RcsGrid


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "e" * 64
_RESPONSE_HASH_A = "c" * 64
_RESPONSE_HASH_B = "d" * 64
_SIGNATURE_A = "1" * 64
_SIGNATURE_B = "2" * 64
_SIGNATURE_C = "3" * 64
_SIGNATURE_D = "4" * 64


def _grid(amplitude=1.0, *, extra=None):
    return RcsGrid(
        [0.0],
        [0.0],
        [10.0],
        ["VV"],
        rcs=np.asarray([[[[complex(amplitude)]]]], dtype=np.complex128),
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        },
        extra=extra,
    )


def _coherent_extra(**additional):
    extra = {
        _COMBINE_ROLE_KEY: "coherent",
        "phase_reference": "vehicle origin",
        "time_convention": "exp(+jwt)",
        "polarization_basis": "earth V/H",
        "amplitude_convention": "sqrt(sigma_3d)",
        "complex_field_domain": "vehicle-frame far field",
    }
    extra.update(additional)
    return extra


def _body_total(
    amplitude,
    digest=_HASH_A,
    signature=_SIGNATURE_A,
    response_digest=_RESPONSE_HASH_A,
):
    record = [{
        "schema": "ghost.workflow.coherent-feature-addition.v1",
        _SOURCE_MONOSTATIC_SHA256_KEY: digest,
        "details": {"component_signature": signature},
    }]
    return _grid(
        amplitude,
        extra=_coherent_extra(
            **{
                _RESPONSE_ROLE_KEY: _RESPONSE_ROLE_BODY_PLUS_FEATURES,
                _SOURCE_MONOSTATIC_SHA256_KEY: digest,
                _ASSEMBLY_BASE_SHA256_KEY: digest,
                _ASSEMBLY_BASE_RESPONSE_SHA256_KEY: response_digest,
                _FEATURE_PROVENANCE_KEY: json.dumps(record),
            }
        ),
    )


def _feature_delta(
    amplitude,
    digest=_HASH_A,
    signature=_SIGNATURE_B,
    response_digest=_RESPONSE_HASH_A,
):
    record = [{
        "schema": "ghost.workflow.coherent-feature-addition.v1",
        _SOURCE_MONOSTATIC_SHA256_KEY: digest,
        "details": {"component_signature": signature},
    }]
    return _grid(
        amplitude,
        extra=_coherent_extra(
            **{
                _RESPONSE_ROLE_KEY: _RESPONSE_ROLE_FEATURES_ONLY_DELTA,
                _ASSEMBLY_BASE_SHA256_KEY: digest,
                _ASSEMBLY_BASE_RESPONSE_SHA256_KEY: response_digest,
                _FEATURE_PROVENANCE_KEY: json.dumps(record),
            }
        ),
    )


def _authoritative_clean_body(amplitude=2.0):
    field = np.asarray(
        [[[[amplitude, 0.5 * amplitude, -0.25 * amplitude]]]],
        dtype=np.complex128,
    )
    raw = field / np.sqrt(4.0 * np.pi)
    return RcsGrid(
        [0.0],
        [0.0],
        [10.0],
        ["VV", "HH", "VH"],
        rcs_power=np.asarray(np.abs(field) ** 2, dtype=np.float64),
        rcs_phase=np.asarray(np.angle(field), dtype=np.float64),
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        },
        extra=_coherent_extra(
            rcs_amp_real=raw.real,
            rcs_amp_imag=raw.imag,
            raw_complex_amplitude_preserved=True,
        ),
    )


def _matching_feature_delta(
    clean,
    amplitude=0.25,
    *,
    digest=_HASH_A,
    signature=_SIGNATURE_B,
):
    field = np.full(clean.rcs_power.shape, complex(amplitude), dtype=np.complex128)
    record = [{
        "schema": "ghost.workflow.coherent-feature-addition.v1",
        _SOURCE_MONOSTATIC_SHA256_KEY: digest,
        "details": {"component_signature": signature},
    }]
    return RcsGrid(
        clean.azimuths,
        clean.elevations,
        clean.frequencies,
        clean.polarizations,
        rcs=field,
        units=dict(clean.units),
        extra=_coherent_extra(
            **{
                _RESPONSE_ROLE_KEY: _RESPONSE_ROLE_FEATURES_ONLY_DELTA,
                _ASSEMBLY_BASE_SHA256_KEY: digest,
                _ASSEMBLY_BASE_RESPONSE_SHA256_KEY: (
                    _assembly_response_physics_sha256(clean)
                ),
                _FEATURE_PROVENANCE_KEY: json.dumps(record),
            }
        ),
    )


class AssemblyTreeSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _root_with(self, *named_grids):
        tree = AssemblyTree()
        root = tree._make_node("Vehicle", _TYPE_ROOT, edit=False)
        leaves = []
        for name, grid in named_grids:
            leaf = tree._make_leaf(name, grid)
            _attach(tree, leaf, root)
            leaves.append(leaf)
        return tree, root, leaves

    def test_untyped_leaf_defaults_power_but_declared_role_is_adopted(self):
        _tree, _root, leaves = self._root_with(
            ("untyped", _grid()),
            ("field", _grid(extra=_coherent_extra())),
            ("power", _grid(extra={_COMBINE_ROLE_KEY: "power"})),
        )

        self.assertEqual(_node_mode(leaves[0]), _MODE_INCOH)
        self.assertEqual(_node_mode(leaves[1]), _MODE_COH)
        self.assertEqual(_node_mode(leaves[2]), _MODE_INCOH)

    def test_auto_branch_preserves_nested_clean_body_plus_feature_field(self):
        clean = _authoritative_clean_body(2.0)
        delta = _matching_feature_delta(clean, 0.25)
        tree = AssemblyTree()
        root = tree._make_node("Vehicle", _TYPE_ROOT, edit=False)
        _attach(tree, tree._make_leaf("clean body", clean), root)
        feature_branch = tree._make_node(
            "Door features", _TYPE_BRANCH, parent=root, edit=False
        )
        _attach(tree, tree._make_leaf("gap delta", delta), feature_branch)

        self.assertEqual(_node_mode(feature_branch), _MODE_AUTO)
        serialized = _item_to_dict(root)
        self.assertEqual(serialized["children"][1]["mode"], _MODE_AUTO)
        restored_root = _dict_to_item(serialized)
        restored_branch = restored_root.child(1)
        self.assertEqual(_node_mode(restored_branch), _MODE_AUTO)
        legacy_serialized = json.loads(json.dumps(serialized))
        legacy_serialized["children"][1].pop("mode")
        legacy_root = _dict_to_item(legacy_serialized)
        self.assertEqual(_node_mode(legacy_root.child(1)), _MODE_AUTO)

        result, history = build_assembly_grid(
            restored_root,
            axis_mode="strict",
            coherent_metadata_attested=True,
        )
        expected = clean.rcs + delta.rcs
        np.testing.assert_allclose(result.rcs, expected, rtol=1.0e-13)
        self.assertIn("Auto -> field +", history)
        self.assertEqual(
            result.extra[_RESPONSE_ROLE_KEY],
            _RESPONSE_ROLE_BODY_PLUS_FEATURES,
        )
        _set_node_mode(feature_branch, _MODE_INCOH)
        overridden = _dict_to_item(_item_to_dict(root))
        self.assertEqual(_node_mode(overridden.child(1)), _MODE_INCOH)

    def test_auto_branch_mode_matrix_follows_built_child_semantics(self):
        scenarios = (
            (
                "single coherent passthrough",
                ((2.0, _coherent_extra()),),
                "coherent",
                4.0,
            ),
            (
                "multiple coherent children",
                ((2.0, _coherent_extra()), (3.0, _coherent_extra())),
                "coherent",
                25.0,
            ),
            (
                "mixed coherent and power children",
                ((2.0, _coherent_extra()), (3.0, None)),
                "power",
                13.0,
            ),
            (
                "power children",
                ((2.0, None), (3.0, None)),
                "power",
                13.0,
            ),
        )
        for label, child_specs, expected_role, expected_power in scenarios:
            with self.subTest(label=label):
                tree = AssemblyTree()
                root = tree._make_node("Vehicle", _TYPE_ROOT, edit=False)
                branch = tree._make_node(
                    "Auto group", _TYPE_BRANCH, parent=root, edit=False
                )
                for index, (amplitude, extra) in enumerate(child_specs):
                    child = tree._make_leaf(
                        f"response {index + 1}",
                        _grid(amplitude, extra=extra),
                    )
                    _attach(tree, child, branch)

                result, history = build_assembly_grid(
                    root,
                    axis_mode="strict",
                    coherent_metadata_attested=True,
                )

                self.assertEqual(_node_mode(branch), _MODE_AUTO)
                self.assertEqual(result.extra[_COMBINE_ROLE_KEY], expected_role)
                self.assertAlmostEqual(
                    float(result.rcs_power.item()), expected_power
                )
                expected_mode = (
                    "field +" if expected_role == "coherent" else "power +"
                )
                self.assertIn(f"Auto -> {expected_mode}", history)

    def test_mode_column_cannot_be_edited_as_free_text(self):
        tree = AssemblyTree()
        root = tree._make_node("Vehicle", _TYPE_ROOT, edit=False)
        branch = tree._make_node(
            "Features", _TYPE_BRANCH, parent=root, edit=False
        )
        mode_index = tree.indexFromItem(branch, 1)

        accepted = tree.edit(
            mode_index,
            QAbstractItemView.DoubleClicked,
            None,
        )

        self.assertFalse(accepted)
        self.assertEqual(_node_mode(branch), _MODE_AUTO)
        self.assertEqual(branch.text(1), "auto")

    def test_shared_body_plus_features_total_is_hard_refused(self):
        _tree, root, _leaves = self._root_with(
            ("fastener total", _body_total(2.0)),
            ("gap total", _body_total(3.0)),
        )

        with self.assertRaisesRegex(ValueError, "count the body twice"):
            build_assembly_grid(
                root,
                axis_mode="strict",
                coherent_metadata_attested=True,
            )

    def test_body_total_plus_same_source_feature_delta_is_safe(self):
        _tree, root, _leaves = self._root_with(
            ("body plus fasteners", _body_total(2.0)),
            ("gap delta", _feature_delta(3.0)),
        )

        result, _history = build_assembly_grid(
            root,
            axis_mode="strict",
            coherent_metadata_attested=True,
        )

        self.assertAlmostEqual(float(result.rcs_power.item()), 25.0)
        self.assertEqual(
            result.extra[_RESPONSE_ROLE_KEY],
            _RESPONSE_ROLE_BODY_PLUS_FEATURES,
        )
        self.assertEqual(result.extra[_SOURCE_MONOSTATIC_SHA256_KEY], _HASH_A)
        self.assertEqual(result.extra[_ASSEMBLY_BASE_SHA256_KEY], _HASH_A)
        provenance = json.loads(result.extra[_ASSEMBLY_PROVENANCE_KEY])
        self.assertEqual(
            provenance["body_plus_features_source_sha256"], [_HASH_A]
        )

    def test_clean_body_plus_matching_delta_becomes_durable_total(self):
        raw = np.asarray(
            [[[[1.0 + 0.0j, 0.5 + 0.25j, -0.25 + 0.1j]]]],
            dtype=np.complex128,
        )
        clean = RcsGrid(
            [0.0], [0.0], [10.0], ["VV", "HH", "VH"],
            rcs_power=np.asarray(4.0 * np.pi * np.abs(raw) ** 2, dtype=np.float32),
            rcs_phase=np.asarray(np.angle(raw), dtype=np.float32),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
            },
            extra=_coherent_extra(
                rcs_amp_real=raw.real,
                rcs_amp_imag=raw.imag,
                raw_complex_amplitude_preserved=True,
            ),
        )
        response_digest = _assembly_response_physics_sha256(clean)
        delta = _feature_delta(
            0.1,
            _HASH_A,
            _SIGNATURE_B,
            response_digest,
        )
        _tree, root, _leaves = self._root_with(
            ("clean body", clean), ("new feature delta", delta)
        )

        aggregate, _history = build_assembly_grid(
            root,
            axis_mode="intersect",
            coherent_metadata_attested=True,
        )

        self.assertEqual(
            aggregate.extra[_RESPONSE_ROLE_KEY],
            _RESPONSE_ROLE_BODY_PLUS_FEATURES,
        )
        self.assertEqual(aggregate.extra[_ASSEMBLY_BASE_SHA256_KEY], _HASH_A)
        self.assertEqual(
            aggregate.extra[_ASSEMBLY_BASE_RESPONSE_SHA256_KEY], response_digest
        )

        restored = _b64_to_grid(_grid_to_b64(aggregate))
        other_total = _body_total(
            2.0,
            _HASH_A,
            _SIGNATURE_C,
            response_digest,
        )
        _tree, duplicate_root, _leaves = self._root_with(
            ("restored aggregate", restored), ("another total", other_total)
        )
        with self.assertRaisesRegex(ValueError, "count the body twice"):
            build_assembly_grid(
                duplicate_root,
                axis_mode="strict",
                coherent_metadata_attested=True,
            )

    def test_different_body_totals_remain_traceable_after_one_build(self):
        _tree, root, _leaves = self._root_with(
            ("vehicle A", _body_total(2.0, _HASH_A)),
            (
                "vehicle B",
                _body_total(
                    3.0,
                    _HASH_B,
                    _SIGNATURE_C,
                    _RESPONSE_HASH_B,
                ),
            ),
        )

        result, _history = build_assembly_grid(
            root,
            axis_mode="strict",
            coherent_metadata_attested=True,
        )

        provenance = json.loads(result.extra[_ASSEMBLY_PROVENANCE_KEY])
        self.assertEqual(
            provenance["body_plus_features_source_sha256"],
            [_HASH_A, _HASH_B],
        )
        self.assertEqual(
            provenance["body_plus_features_base_response_sha256"],
            [_RESPONSE_HASH_A, _RESPONSE_HASH_B],
        )

    def test_same_physical_body_is_rejected_after_source_file_repackage(self):
        _tree, root, _leaves = self._root_with(
            (
                "original package",
                _body_total(
                    2.0,
                    _HASH_A,
                    _SIGNATURE_A,
                    _RESPONSE_HASH_A,
                ),
            ),
            (
                "resaved package",
                _body_total(
                    3.0,
                    _HASH_B,
                    _SIGNATURE_C,
                    _RESPONSE_HASH_A,
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "physical base-response"):
            build_assembly_grid(
                root,
                axis_mode="strict",
                coherent_metadata_attested=True,
            )

    def test_aggregate_round_trip_retains_physical_body_occupancy(self):
        _tree, first_root, _leaves = self._root_with(
            (
                "body A",
                _body_total(
                    2.0,
                    _HASH_A,
                    _SIGNATURE_A,
                    _RESPONSE_HASH_A,
                ),
            ),
            (
                "body B",
                _body_total(
                    3.0,
                    _HASH_B,
                    _SIGNATURE_C,
                    _RESPONSE_HASH_B,
                ),
            ),
        )
        aggregate, _history = build_assembly_grid(
            first_root,
            axis_mode="strict",
            coherent_metadata_attested=True,
        )
        restored = _b64_to_grid(_grid_to_b64(aggregate))
        repackaged_a = _body_total(
            4.0,
            _HASH_C,
            _SIGNATURE_D,
            _RESPONSE_HASH_A,
        )
        _tree, second_root, _leaves = self._root_with(
            ("restored aggregate", restored),
            ("repackaged body A", repackaged_a),
        )

        with self.assertRaisesRegex(ValueError, "physical base-response"):
            build_assembly_grid(
                second_root,
                axis_mode="strict",
                coherent_metadata_attested=True,
            )

    def test_untagged_clean_body_plus_total_is_rejected_after_asy_round_trip(self):
        clean = _authoritative_clean_body(2.0)
        clean_digest = _assembly_response_physics_sha256(clean)
        total = _body_total(
            3.0,
            _HASH_A,
            _SIGNATURE_A,
            clean_digest,
        )
        tree, root, _leaves = self._root_with(
            ("untagged clean body", clean),
            ("body plus features", total),
        )

        for candidate in (root, _dict_to_item(_item_to_dict(root))):
            with self.assertRaisesRegex(
                ValueError, "physical response SHA-256"
            ):
                build_assembly_grid(
                    candidate,
                    axis_mode="strict",
                    coherent_metadata_attested=True,
                )

    def test_power_transition_retains_feature_delta_base_context(self):
        clean = _authoritative_clean_body(2.0)
        delta = _matching_feature_delta(clean, 0.25)
        _tree, root, leaves = self._root_with(("feature delta", delta))
        _set_node_mode(leaves[0], _MODE_INCOH)

        power_delta, _history = build_assembly_grid(root, axis_mode="strict")

        self.assertEqual(
            power_delta.extra[_ASSEMBLY_BASE_SHA256_KEY], _HASH_A
        )
        self.assertEqual(
            power_delta.extra[_ASSEMBLY_BASE_RESPONSE_SHA256_KEY],
            _assembly_response_physics_sha256(clean),
        )
        provenance = json.loads(
            power_delta.extra[_ASSEMBLY_PROVENANCE_KEY]
        )
        self.assertTrue(provenance["feature_delta_only"])
        self.assertEqual(
            provenance["feature_delta_base_sha256"], [_HASH_A]
        )

        legacy_delta = _b64_to_grid(_grid_to_b64(power_delta))
        legacy_record_raw = np.asarray(
            legacy_delta.extra[_ASSEMBLY_PROVENANCE_KEY]
        ).reshape(-1)[0]
        if isinstance(legacy_record_raw, np.generic):
            legacy_record_raw = legacy_record_raw.item()
        legacy_record = json.loads(legacy_record_raw)
        for key in (
            "feature_delta_base_sha256",
            "feature_delta_base_response_sha256",
            "feature_delta_only",
        ):
            legacy_record.pop(key)
        legacy_delta.extra[_ASSEMBLY_PROVENANCE_KEY] = json.dumps(
            legacy_record
        )
        legacy_delta.extra.pop(_ASSEMBLY_BASE_SHA256_KEY, None)

        for candidate in (power_delta, legacy_delta):
            _tree, unrelated_root, _leaves = self._root_with(
                ("power-domain delta", candidate),
                (
                    "unrelated body",
                    _body_total(
                        3.0,
                        _HASH_B,
                        _SIGNATURE_C,
                        _RESPONSE_HASH_B,
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "different assembly base"):
                build_assembly_grid(unrelated_root, axis_mode="strict")

    def test_distinct_feature_deltas_on_one_base_remain_combinable(self):
        clean = _authoritative_clean_body(2.0)
        first = _matching_feature_delta(
            clean, 0.25, signature=_SIGNATURE_B
        )
        second = _matching_feature_delta(
            clean, 0.5, signature=_SIGNATURE_C
        )
        _tree, root, _leaves = self._root_with(
            ("gap delta", first),
            ("fastener delta", second),
        )

        result, _history = build_assembly_grid(
            root,
            axis_mode="strict",
            coherent_metadata_attested=True,
        )

        self.assertEqual(
            result.extra[_RESPONSE_ROLE_KEY],
            _RESPONSE_ROLE_FEATURES_ONLY_DELTA,
        )
        np.testing.assert_allclose(result.rcs, first.rcs + second.rcs)

    def test_distinct_tree_built_featured_bodies_remain_combinable(self):
        clean_a = _authoritative_clean_body(2.0)
        clean_b = _authoritative_clean_body(3.0)
        delta_a = _matching_feature_delta(
            clean_a,
            0.25,
            digest=_HASH_A,
            signature=_SIGNATURE_B,
        )
        delta_b = _matching_feature_delta(
            clean_b,
            0.5,
            digest=_HASH_B,
            signature=_SIGNATURE_C,
        )
        totals = []
        for clean, delta in ((clean_a, delta_a), (clean_b, delta_b)):
            _tree, body_root, _leaves = self._root_with(
                ("clean body", clean),
                ("feature delta", delta),
            )
            total, _history = build_assembly_grid(
                body_root,
                axis_mode="strict",
                coherent_metadata_attested=True,
            )
            totals.append(total)

        _tree, root, _leaves = self._root_with(
            ("featured body A", totals[0]),
            ("featured body B", totals[1]),
        )
        result, _history = build_assembly_grid(
            root,
            axis_mode="strict",
            coherent_metadata_attested=True,
        )

        provenance = json.loads(result.extra[_ASSEMBLY_PROVENANCE_KEY])
        self.assertEqual(
            provenance["body_plus_features_source_sha256"],
            [_HASH_A, _HASH_B],
        )
        self.assertEqual(
            provenance["feature_delta_base_sha256"],
            [_HASH_A, _HASH_B],
        )

    def test_feature_delta_from_different_base_is_refused(self):
        _tree, root, _leaves = self._root_with(
            ("vehicle A", _body_total(2.0, _HASH_A)),
            (
                "vehicle B delta",
                _feature_delta(
                    3.0,
                    _HASH_B,
                    _SIGNATURE_B,
                    _RESPONSE_HASH_B,
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "different assembly base hashes"):
            build_assembly_grid(
                root,
                axis_mode="strict",
                coherent_metadata_attested=True,
            )

    def test_total_plus_its_published_delta_is_refused(self):
        _tree, root, _leaves = self._root_with(
            ("featured total", _body_total(2.0, _HASH_A, _SIGNATURE_A)),
            (
                "same features again",
                _feature_delta(1.0, _HASH_A, _SIGNATURE_A),
            ),
        )

        with self.assertRaisesRegex(ValueError, "same placed feature component"):
            build_assembly_grid(
                root,
                axis_mode="strict",
                coherent_metadata_attested=True,
            )

    def test_aggregate_provenance_blocks_readding_one_delta(self):
        _tree, first_root, _leaves = self._root_with(
            ("first total", _body_total(2.0, _HASH_A, _SIGNATURE_A)),
            ("second delta", _feature_delta(1.0, _HASH_A, _SIGNATURE_B)),
        )
        aggregate, _history = build_assembly_grid(
            first_root,
            axis_mode="strict",
            coherent_metadata_attested=True,
        )

        _tree, second_root, _leaves = self._root_with(
            ("aggregate", aggregate),
            ("second delta again", _feature_delta(1.0, _HASH_A, _SIGNATURE_B)),
        )
        with self.assertRaisesRegex(ValueError, "same placed feature component"):
            build_assembly_grid(
                second_root,
                axis_mode="strict",
                coherent_metadata_attested=True,
            )

    def test_contradictory_base_hash_metadata_is_refused(self):
        bad = _feature_delta(1.0, _HASH_A)
        bad.extra[_SOURCE_MONOSTATIC_SHA256_KEY] = _HASH_B
        _tree, root, _leaves = self._root_with(("bad delta", bad))

        with self.assertRaisesRegex(ValueError, "contradictory|exactly one consistent"):
            build_assembly_grid(root, axis_mode="strict")

    def test_single_untyped_sentri_leaf_keeps_coordinate_provenance(self):
        sentri = _grid(
            extra={
                "source_format": "SENTRi exported RCS table",
                "assembly_angular_coordinate_contract": "conic waterline elevation",
                "sentri_elevation_convention": "grim_waterline_zero_top_positive",
                "sentri_coordinate_mapping": "elevation=90-theta; azimuth=wrapped phi",
                "sentri_units_row_present": True,
            }
        )
        _tree, root, _leaves = self._root_with(("SENTRi", sentri))

        result, _history = build_assembly_grid(root, axis_mode="strict")

        self.assertTrue(np.isnan(result.rcs_phase.item()))
        for key in (
            "source_format",
            "assembly_angular_coordinate_contract",
            "sentri_elevation_convention",
            "sentri_coordinate_mapping",
            "sentri_units_row_present",
        ):
            self.assertEqual(result.extra[key], sentri.extra[key])

    def test_single_coherent_feature_leaf_is_exact_metadata_passthrough(self):
        featured = _body_total(2.0)
        _tree, root, _leaves = self._root_with(("featured vehicle", featured))

        result, _history = build_assembly_grid(root, axis_mode="strict")

        self.assertIs(result, featured)
        self.assertIn(_FEATURE_PROVENANCE_KEY, result.extra)
        self.assertEqual(
            result.extra[_RESPONSE_ROLE_KEY], _RESPONSE_ROLE_BODY_PLUS_FEATURES
        )

    def test_coordinate_contract_conflict_is_not_silently_dropped(self):
        left = _grid(extra={"assembly_angular_coordinate_contract": "contract A"})
        right = _grid(extra={"assembly_angular_coordinate_contract": "contract B"})
        _tree, root, _leaves = self._root_with(("left", left), ("right", right))

        with self.assertRaisesRegex(
            ValueError, "contradictory assembly_angular_coordinate_contract"
        ):
            build_assembly_grid(root, axis_mode="strict")

    def test_one_sided_units_convention_is_not_laundered_to_whole_result(self):
        sentri = _grid()
        sentri.units["elevation_coordinate_convention"] = "sentri_theta_top_zero"
        ordinary = _grid(2.0)
        _tree, root, _leaves = self._root_with(
            ("SENTRi", sentri), ("ordinary", ordinary)
        )

        result, _history = build_assembly_grid(root, axis_mode="strict")

        self.assertNotIn("elevation_coordinate_convention", result.units)
        self.assertNotIn("elevation_coordinate_convention", result.extra)

    def test_power_transition_scrubs_stale_dynamic_metadata_from_units(self):
        source = _grid(2.0, extra=_coherent_extra())
        stale_dynamic = {
            _COMBINE_ROLE_KEY: "coherent",
            "combine_role_note": "stale source role",
            _RESPONSE_ROLE_KEY: "coherent_field_sum",
            _ASSEMBLY_PROVENANCE_KEY: json.dumps({"schema": "stale"}),
            _FEATURE_PROVENANCE_KEY: "[]",
            "coherent_metadata_attestation_json": "{}",
            "response_role_validation": "stale",
            _ASSEMBLY_BASE_SHA256_KEY: _HASH_A,
            _ASSEMBLY_BASE_RESPONSE_SHA256_KEY: _RESPONSE_HASH_A,
            "body_plus_features_base_response_sha256": [_RESPONSE_HASH_A],
            "feature_delta_base_sha256": [_HASH_A],
            "feature_delta_only": False,
            "rcs_amp_real": np.asarray([1.0]),
            "rcs_amp_imag": np.asarray([0.0]),
            "raw_complex_amplitude_preserved": True,
        }
        source.units.update(stale_dynamic)
        _tree, root, leaves = self._root_with(("response", source))
        _set_node_mode(leaves[0], _MODE_INCOH)

        result, _history = build_assembly_grid(root, axis_mode="strict")

        self.assertEqual(result.extra[_COMBINE_ROLE_KEY], "power")
        for key in stale_dynamic:
            self.assertNotIn(key, result.units)

    def test_build_dialog_has_no_coherent_attestation_gate(self):
        dialog = BuildDialog()

        self.assertFalse(hasattr(dialog, "chk_coherent_attestation"))
        self.assertFalse(hasattr(dialog, "coherent_metadata_attested"))

    def test_unknown_coherent_metadata_is_allowed_and_recorded(self):
        _tree, root, leaves = self._root_with(
            ("left", _grid(1.0)),
            ("right", _grid(2.0)),
        )
        for leaf in leaves:
            _set_node_mode(leaf, _MODE_COH)

        result, history = build_assembly_grid(root, axis_mode="strict")
        self.assertAlmostEqual(float(result.rcs_power.item()), 9.0)
        self.assertIn("coherent_metadata_assumption_json", result.extra)
        assumption = json.loads(
            result.extra["coherent_metadata_assumption_json"]
        )
        self.assertEqual(assumption["operation"], "assembly-field-add")
        self.assertIn("common_vehicle_frame", assumption["assumed_scope"])
        self.assertIn("common_attitude", assumption["assumed_scope"])
        self.assertEqual(
            assumption["missing_metadata_input_indices_1_based"],
            {
                "phase_reference": [1, 2],
                "polarization_basis": [1, 2],
                "time_convention": [1, 2],
            },
        )
        self.assertIn("Field + assumption", history)
        provenance = json.loads(result.extra[_ASSEMBLY_PROVENANCE_KEY])
        self.assertFalse(provenance["coherent_metadata_attested"])
        self.assertTrue(provenance["coherent_registration_assumed"])
        self.assertEqual(
            provenance["coherent_registration_basis"], "Field + role selection"
        )

    def test_matching_coherent_labels_build_without_attestation(self):
        _tree, root, _leaves = self._root_with(
            ("left", _grid(1.0, extra=_coherent_extra())),
            ("right", _grid(2.0, extra=_coherent_extra())),
        )

        result, _history = build_assembly_grid(root, axis_mode="strict")

        self.assertAlmostEqual(float(result.rcs_power.item()), 9.0)
        assumption = json.loads(
            result.extra["coherent_metadata_assumption_json"]
        )
        self.assertEqual(
            assumption["missing_metadata_input_indices_1_based"], {}
        )

    def test_explicit_coherent_convention_conflict_is_advisory(self):
        left_extra = _coherent_extra()
        right_extra = _coherent_extra(time_convention="exp(-jwt)")
        _tree, root, _leaves = self._root_with(
            ("left", _grid(1.0, extra=left_extra)),
            ("right", _grid(2.0, extra=right_extra)),
        )

        result, _history = build_assembly_grid(root, axis_mode="strict")
        self.assertAlmostEqual(float(result.rcs_power.item()), 9.0)
        self.assertIn("time_convention", result.extra["metadata_advisories_json"])
        self.assertNotIn("time_convention", result.extra)

    def test_interpolation_is_refused_when_any_field_input_is_present(self):
        _tree, root, _leaves = self._root_with(
            ("field", _grid(1.0, extra=_coherent_extra())),
            (
                "power",
                _grid(2.0, extra={_COMBINE_ROLE_KEY: "power"}),
            ),
        )

        with self.assertRaisesRegex(ValueError, r"disabled.*Field \+"):
            build_assembly_grid(root, axis_mode="interp")

    def test_intersect_preserves_authoritative_tiny_field_cancellation(self):
        scale = np.sqrt(4.0 * np.pi)

        def solver_grid(field):
            raw = complex(field) / scale
            return RcsGrid(
                [0.0], [0.0], [10.0], ["VV"],
                rcs_power=np.asarray([[[[abs(field) ** 2]]]], dtype=np.float32),
                rcs_phase=np.asarray([[[[np.angle(field)]]]], dtype=np.float32),
                units={
                    "azimuth": "deg",
                    "elevation": "deg",
                    "frequency": "GHz",
                    "rcs_log_unit": "dBsm",
                    "rcs_linear_quantity": "sigma_3d",
                },
                extra=_coherent_extra(
                    rcs_amp_real=np.asarray([[[[raw.real]]]], dtype=np.float64),
                    rcs_amp_imag=np.asarray([[[[raw.imag]]]], dtype=np.float64),
                    raw_complex_amplitude_preserved=True,
                ),
            )

        left = solver_grid(1.0 + 0.0j)
        right = solver_grid(-1.0 + 1.0e-10j)
        _tree, root, _leaves = self._root_with(("left", left), ("right", right))

        result, _history = build_assembly_grid(
            root,
            axis_mode="intersect",
            coherent_metadata_attested=True,
        )

        expected = abs(complex(left.rcs.item()) + complex(right.rcs.item())) ** 2
        self.assertAlmostEqual(
            float(result.rcs_power.item()), expected, delta=max(expected * 1.0e-12, 1.0e-32)
        )

    def test_power_role_cannot_be_overridden_into_field_add(self):
        power_grid = _grid(extra={_COMBINE_ROLE_KEY: "power"})
        _tree, root, leaves = self._root_with(("statistical term", power_grid))
        _set_node_mode(leaves[0], _MODE_COH)

        with self.assertRaisesRegex(ValueError, "combine_role='power'"):
            build_assembly_grid(root, axis_mode="strict")

    def test_leaf_grid_identity_is_not_replaced_or_stripped(self):
        featured = _body_total(2.0)
        _tree, _root, leaves = self._root_with(("featured", featured))
        self.assertIs(leaves[0].data(0, _ROLE_GRID), featured)


if __name__ == "__main__":
    unittest.main()
