"""Fail-closed discovery regressions for the embedded GHOST backend."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import ghost_integration


class GhostBackendDiscoveryTests(unittest.TestCase):
    def test_default_discovery_has_no_silent_sibling_checkout_fallback(self):
        expected = (
            Path(ghost_integration.__file__).resolve().parents[1]
            / "tools"
            / "GHOST"
            / "Backend"
        ).resolve()
        with mock.patch.dict(
            os.environ, {ghost_integration.GHOST_BACKEND_ENV: ""}, clear=False
        ):
            self.assertEqual(
                list(ghost_integration.ghost_backend_candidates()),
                [expected],
            )

    def test_invalid_environment_override_does_not_fall_back_to_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            incomplete = Path(tmp).resolve()
            with mock.patch.dict(
                os.environ,
                {ghost_integration.GHOST_BACKEND_ENV: str(incomplete)},
                clear=False,
            ):
                self.assertEqual(
                    list(ghost_integration.ghost_backend_candidates()),
                    [incomplete],
                )
                self.assertIsNone(ghost_integration.discover_ghost_backend())
                with self.assertRaisesRegex(
                    ImportError, "No complete GHOST backend was found"
                ):
                    ghost_integration.load_ghost_module("ghost_gui")

    def test_missing_backend_never_imports_same_named_sys_path_module(self):
        module_name = "ghost_untrusted_probe"
        with tempfile.TemporaryDirectory() as backend_tmp, tempfile.TemporaryDirectory() as other_tmp:
            backend = Path(backend_tmp)
            other = Path(other_tmp)
            (other / f"{module_name}.py").write_text(
                "VALUE = 'wrong checkout'\n", encoding="utf-8"
            )
            sys.path.insert(0, str(other))
            try:
                with self.assertRaisesRegex(
                    ImportError, "No complete GHOST backend was found"
                ):
                    ghost_integration.load_ghost_module(module_name, backend)
                self.assertNotIn(module_name, sys.modules)
            finally:
                sys.path.remove(str(other))
                sys.modules.pop(module_name, None)

    def test_loaded_flat_module_from_other_checkout_is_rejected(self):
        with mock.patch.dict(
            os.environ, {ghost_integration.GHOST_BACKEND_ENV: ""}, clear=False
        ):
            backend = ghost_integration.discover_ghost_backend()
        self.assertIsNotNone(backend)
        assert backend is not None

        module_name = "geometry_io"
        original = sys.modules.get(module_name)
        stale = ModuleType(module_name)
        stale.__file__ = str(backend.parent / "old_backend" / "geometry_io.py")
        sys.modules[module_name] = stale
        try:
            with self.assertRaisesRegex(
                ImportError, "cannot mix backend modules from different checkouts"
            ):
                ghost_integration.load_ghost_module("feature_workflow", backend)
        finally:
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original


class GhostMaterialDelegateTests(unittest.TestCase):
    def test_material_delegate_forwards_typed_artifact_to_workspace(self):
        attach = mock.Mock(return_value=True)
        host = SimpleNamespace(
            workspace=SimpleNamespace(attach_material_artifact=attach)
        )

        result = ghost_integration.GhostIntegrationWidget.attach_material_artifact(
            host, "ibc", Path("nominal_ibc.csv")
        )

        self.assertTrue(result)
        attach.assert_called_once_with("ibc", os.fspath(Path("nominal_ibc.csv")))

    def test_material_delegate_warns_and_returns_false_without_workspace(self):
        host = SimpleNamespace(workspace=None)
        with mock.patch.object(
            ghost_integration.QMessageBox, "warning"
        ) as warning:
            result = (
                ghost_integration.GhostIntegrationWidget.attach_material_artifact(
                    host, "material", "nominal_material.csv"
                )
            )

        self.assertFalse(result)
        self.assertIn("unavailable", warning.call_args.args[2].lower())


if __name__ == "__main__":
    unittest.main()
