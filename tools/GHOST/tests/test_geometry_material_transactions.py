#!/usr/bin/env python3
"""Material-sidecar grammar and atomic publication regression tests."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

from geometry_io import (  # noqa: E402
    AtomicFileTransaction,
    build_geometry_text,
    parse_geometry,
)


class MaterialFilenameGrammarTests(unittest.TestCase):
    def test_safe_sidecar_rows_round_trip(self) -> None:
        text = build_geometry_text(
            "body",
            [],
            [["1", "coating_ibc.csv"]],
            [["2", "substrate.csv"]],
        )

        _title, _segments, ibcs, dielectrics = parse_geometry(text)

        self.assertEqual(ibcs, [["1", "coating_ibc.csv"]])
        self.assertEqual(dielectrics, [["2", "substrate.csv"]])

    def test_writer_rejects_unquoted_whitespace_in_sidecar_name(self) -> None:
        for unsafe_name in (
            "My IBC.csv",
            "leading.csv ",
            "tab\tname.csv",
            "line\nname.csv",
        ):
            with self.subTest(unsafe_name=unsafe_name):
                with self.assertRaisesRegex(ValueError, "cannot contain whitespace"):
                    build_geometry_text(
                        "body", [], [["1", unsafe_name]], []
                    )


class AtomicFileTransactionTests(unittest.TestCase):
    def test_read_only_source_can_be_staged(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "read_only.csv"
            source.write_text("nominal material\n", encoding="utf-8")
            os.chmod(source, stat.S_IREAD)
            destination = root / "copied.csv"
            try:
                transaction = AtomicFileTransaction()
                transaction.stage_copy(source, destination)
                transaction.publish()
                transaction.commit()

                self.assertEqual(
                    destination.read_text(encoding="utf-8"),
                    "nominal material\n",
                )
            finally:
                # Windows may refuse temporary-directory cleanup while the
                # original source retains its read-only attribute.
                os.chmod(source, stat.S_IWRITE | stat.S_IREAD)

    def test_successful_multi_file_publication_replaces_complete_set(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.csv"
            source.write_text("new sidecar\n", encoding="utf-8")
            sidecar = root / "sidecar.csv"
            sidecar.write_text("old sidecar\n", encoding="utf-8")
            geometry = root / "body.geo"
            geometry.write_text("old geometry\n", encoding="utf-8")

            transaction = AtomicFileTransaction()
            transaction.stage_copy(source, sidecar)
            transaction.stage_text("new geometry\n", geometry)
            transaction.publish()
            transaction.commit()

            self.assertEqual(
                sidecar.read_text(encoding="utf-8"), "new sidecar\n"
            )
            self.assertEqual(
                geometry.read_text(encoding="utf-8"), "new geometry\n"
            )
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["body.geo", "sidecar.csv", "source.csv"],
            )

    def test_later_publish_failure_restores_every_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.csv"
            source.write_text("new sidecar\n", encoding="utf-8")
            sidecar = root / "sidecar.csv"
            sidecar.write_text("old sidecar\n", encoding="utf-8")
            geometry = root / "body.geo"
            geometry.write_text("old geometry\n", encoding="utf-8")

            transaction = AtomicFileTransaction()
            transaction.stage_copy(source, sidecar)
            transaction.stage_text("new geometry\n", geometry)

            real_replace = os.replace
            call_count = 0

            def fail_second_publish(source_path: object, target_path: object) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("simulated geometry publication failure")
                real_replace(source_path, target_path)

            with mock.patch(
                "geometry_io.os.replace", side_effect=fail_second_publish
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    transaction.publish()

            self.assertEqual(
                sidecar.read_text(encoding="utf-8"), "old sidecar\n"
            )
            self.assertEqual(
                geometry.read_text(encoding="utf-8"), "old geometry\n"
            )
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["body.geo", "sidecar.csv", "source.csv"],
            )

    def test_explicit_rollback_restores_file_after_ui_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.csv"
            source.write_text("new\n", encoding="utf-8")
            destination = root / "attached.csv"
            destination.write_text("old\n", encoding="utf-8")

            transaction = AtomicFileTransaction()
            transaction.stage_copy(source, destination)
            transaction.publish()
            self.assertEqual(destination.read_text(encoding="utf-8"), "new\n")

            transaction.rollback()

            self.assertEqual(destination.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["attached.csv", "source.csv"],
            )


if __name__ == "__main__":
    unittest.main()
