from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile
import unittest

import grim_diagnostics as diagnostics


class GrimDiagnosticsTests(unittest.TestCase):
    def _make_tree(self, root: Path) -> tuple[Path, Path, Path]:
        grim = root / "GRIM_Revised_2"
        ghost = root / "tools" / "GHOST" / "Backend"
        freddy = root / "tools" / "FREDDY"
        for directory in (grim, ghost, freddy / "ibc"):
            directory.mkdir(parents=True, exist_ok=True)
        for relative in diagnostics.GRIM_SENTINELS:
            path = grim / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# sentinel\n", encoding="utf-8")
        (grim / "grim_diagnostics.py").write_text("# sentinel\n", encoding="utf-8")
        for relative in diagnostics.GHOST_SENTINELS:
            (ghost / relative).write_text("# sentinel\n", encoding="utf-8")
        for relative in diagnostics.FREDDY_SENTINELS:
            (freddy / relative).write_text("# sentinel\n", encoding="utf-8")
        return grim, ghost, freddy

    @staticmethod
    def _dependencies(module_name: str, _distribution: str) -> diagnostics.DependencyProbe:
        versions = {
            "numpy": "2.1.0",
            "PySide6.QtWidgets": "6.8.0",
            "matplotlib.backends.backend_qtagg": "3.10.0",
            "scipy": "1.15.0",
        }
        return diagnostics.DependencyProbe(True, versions[module_name])

    @staticmethod
    def _no_native(
        _candidates: list[Path] | tuple[Path, ...],
        _symbols: list[str] | tuple[str, ...],
    ) -> tuple[Path | None, str]:
        return None, "not installed for test platform"

    def _collect(
        self,
        root: Path,
        grim: Path,
        *,
        environ: dict[str, str] | None = None,
    ) -> list[diagnostics.DiagnosticResult]:
        return diagnostics.collect_diagnostics(
            root,
            module_directory=grim,
            environ={} if environ is None else environ,
            dependency_probe=self._dependencies,
            system_name="Linux",
            machine_name="x86_64",
            library_probe=self._no_native,
            powerpoint_probe=lambda: (False, "should not run off Windows"),
        )

    def test_complete_tree_is_ready_despite_optional_capability_notices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grim, _ghost, _freddy = self._make_tree(root)
            results = self._collect(root, grim)

        self.assertEqual(diagnostics.startup_exit_code(results), 0)
        self.assertFalse([result for result in results if result.blocks_startup])
        by_key = {result.key: result for result in results}
        self.assertEqual(by_key["powerpoint"].status, "SKIP")
        self.assertEqual(by_key["native_fmm"].status, "WARN")
        self.assertEqual(by_key["native_bor"].status, "WARN")

        output = io.StringIO()
        diagnostics.write_report(results, stream=output)
        rendered = output.getvalue()
        self.assertIn("RESULT: READY", rendered)
        self.assertIn("[optional] PowerPoint export", rendered)
        self.assertIn("do not prevent GRIM from starting", rendered)

    def test_missing_required_ghost_sentinel_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grim, ghost, _freddy = self._make_tree(root)
            os.unlink(ghost / "rcs_solver.py")
            results = self._collect(root, grim)

        self.assertEqual(diagnostics.startup_exit_code(results), 1)
        workspace = next(result for result in results if result.key == "ghost_workspace")
        self.assertTrue(workspace.blocks_startup)
        self.assertIn("rcs_solver.py", " ".join(workspace.details))
        self.assertEqual(
            next(result for result in results if result.key == "native_fmm").status,
            "SKIP",
        )

    def test_missing_direct_grim_startup_module_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grim, _ghost, _freddy = self._make_tree(root)
            os.unlink(grim / "grim_python.py")
            results = self._collect(root, grim)

        source = next(result for result in results if result.key == "grim_source")
        self.assertTrue(source.blocks_startup)
        self.assertIn("grim_python.py", " ".join(source.details))
        self.assertEqual(diagnostics.startup_exit_code(results), 1)

    def test_incomplete_ghost_override_is_authoritative_and_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grim, bundled_ghost, _freddy = self._make_tree(root)
            override = root / "old-ghost" / "Backend"
            override.mkdir(parents=True)
            (override / "ghost_gui.py").write_text("# incomplete\n", encoding="utf-8")
            results = self._collect(
                root,
                grim,
                environ={"GHOST_BACKEND_PATH": str(override)},
            )

        workspace = next(result for result in results if result.key == "ghost_workspace")
        self.assertEqual(diagnostics.startup_exit_code(results), 1)
        self.assertIn(str(override.resolve()), " ".join(workspace.details))
        self.assertNotIn(str(bundled_ghost.resolve()), workspace.summary)

    def test_loaded_non_sentinel_backend_module_conflict_is_reported(self) -> None:
        import sys
        from types import ModuleType

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grim, ghost, _freddy = self._make_tree(root)
            (ghost / "frame.py").write_text("# backend module\n", encoding="utf-8")
            stale = ModuleType("frame")
            stale.__file__ = str(root / "old-backend" / "frame.py")
            previous = sys.modules.get("frame")
            sys.modules["frame"] = stale
            try:
                results = self._collect(root, grim)
            finally:
                if previous is None:
                    sys.modules.pop("frame", None)
                else:
                    sys.modules["frame"] = previous

        origin = next(result for result in results if result.key == "ghost_origin")
        self.assertEqual(origin.status, "FAIL")
        self.assertIn("frame", " ".join(origin.details))
        self.assertEqual(diagnostics.startup_exit_code(results), 1)

    def test_incomplete_freddy_override_falls_back_with_nonblocking_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grim, _ghost, bundled_freddy = self._make_tree(root)
            override = root / "old-freddy"
            override.mkdir()
            results = self._collect(
                root,
                grim,
                environ={"FREDDY_ROOT_PATH": str(override)},
            )

        workspace = next(result for result in results if result.key == "freddy_workspace")
        self.assertEqual(workspace.status, "WARN")
        self.assertFalse(workspace.blocks_startup)
        self.assertEqual(diagnostics.startup_exit_code(results), 0)
        self.assertIn(str(bundled_freddy.resolve()), " ".join(workspace.details))
        self.assertIn("incomplete", " ".join(workspace.details).lower())

    def test_missing_scipy_or_qt_blocks_integrated_startup(self) -> None:
        def probe(module_name: str, distribution: str) -> diagnostics.DependencyProbe:
            if module_name in {"scipy", "PySide6.QtWidgets"}:
                return diagnostics.DependencyProbe(False, detail=f"{distribution} missing")
            return self._dependencies(module_name, distribution)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grim, _ghost, _freddy = self._make_tree(root)
            results = diagnostics.collect_diagnostics(
                root,
                module_directory=grim,
                environ={},
                dependency_probe=probe,
                system_name="Linux",
                machine_name="x86_64",
                library_probe=self._no_native,
            )

        by_key = {result.key: result for result in results}
        self.assertEqual(by_key["scipy"].status, "FAIL")
        self.assertTrue(by_key["scipy"].blocks_startup)
        self.assertEqual(by_key["pyside6"].status, "FAIL")
        self.assertTrue(by_key["pyside6"].blocks_startup)
        self.assertEqual(diagnostics.startup_exit_code(results), 1)

    def test_powerpoint_probe_is_lightweight_and_optional(self) -> None:
        calls: list[str] = []

        def powerpoint_probe() -> tuple[bool, str]:
            calls.append("registration")
            return False, "PowerPoint.Application is not registered"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grim, _ghost, _freddy = self._make_tree(root)
            results = diagnostics.collect_diagnostics(
                root,
                module_directory=grim,
                environ={},
                dependency_probe=self._dependencies,
                system_name="Windows",
                machine_name="AMD64",
                library_probe=self._no_native,
                powerpoint_probe=powerpoint_probe,
            )

        self.assertEqual(calls, ["registration"])
        powerpoint = next(result for result in results if result.key == "powerpoint")
        self.assertEqual(powerpoint.status, "WARN")
        self.assertFalse(powerpoint.blocks_startup)
        self.assertEqual(diagnostics.startup_exit_code(results), 0)


if __name__ == "__main__":
    unittest.main()
