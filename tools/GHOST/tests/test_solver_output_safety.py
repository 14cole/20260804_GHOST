"""Focused contracts for collision-safe GHOST desktop output planning."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

try:
    import solver_tab
except (ImportError, RuntimeError) as exc:  # GUI dependency is optional in lean CI.
    solver_tab = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(solver_tab is None, f"GHOST GUI dependencies unavailable: {_IMPORT_ERROR}")
class SolverOutputSafetyTests(unittest.TestCase):
    def test_acceleration_status_is_visible_and_names_slow_fallbacks(self) -> None:
        with mock.patch.object(
            solver_tab, "_native_library_available", return_value=False
        ):
            fmm_ready, bor_ready, text = solver_tab._native_acceleration_status(
                Path("missing-backend")
            )
        self.assertFalse(fmm_ready)
        self.assertFalse(bor_ready)
        self.assertIn("100x slower", text)
        self.assertIn("2-8x slower", text)

    def test_windows_acceleration_probe_never_offers_foreign_libraries(self) -> None:
        with (
            mock.patch.object(solver_tab.platform, "system", return_value="Windows"),
            mock.patch.object(solver_tab.platform, "machine", return_value="AMD64"),
            mock.patch.object(
                solver_tab, "_native_library_available", return_value=False
            ) as probe,
        ):
            solver_tab._native_acceleration_status(Path("missing-backend"))

        self.assertEqual(probe.call_count, 2)
        for call in probe.call_args_list:
            candidates = call.args[0]
            self.assertTrue(candidates)
            self.assertTrue(
                all(candidate.suffix.lower() == ".dll" for candidate in candidates),
                candidates,
            )

    def test_monostatic_and_bistatic_paths_are_known_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mono = solver_tab._planned_export_paths(
                {"scattering_mode": "monostatic"},
                str(root / "body"),
            )
            self.assertEqual(mono, [str((root / "body.grim").resolve())])

            bistatic = solver_tab._planned_export_paths(
                {
                    "scattering_mode": "bistatic",
                    "samples": [
                        {"theta_inc_deg": 10.0},
                        {"theta_inc_deg": -2.5},
                        {"theta_inc_deg": 10.0},
                    ],
                },
                str(root / "field.grim"),
            )
            self.assertEqual(
                bistatic,
                [
                    str((root / "field_inc_m2p5.grim").resolve()),
                    str((root / "field_inc_10.grim").resolve()),
                ],
            )

    def test_relative_explicit_output_is_resolved_beside_geometry(self) -> None:
        class Resolver:
            _documents_output_dir = staticmethod(lambda: Path("unused"))

        with tempfile.TemporaryDirectory() as folder:
            geometry = Path(folder) / "geometry" / "body.geo"
            geometry.parent.mkdir()
            geometry.write_text("title: body\n", encoding="utf-8")
            resolved = solver_tab.SolverTab._resolve_output_path(
                Resolver(),
                "results/body_response",
                str(geometry),
            )
            self.assertEqual(
                Path(resolved),
                geometry.parent / "results" / "body_response.grim",
            )


if __name__ == "__main__":
    unittest.main()
