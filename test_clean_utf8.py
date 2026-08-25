from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

import clean_utf8


class CleanUtf8Tests(unittest.TestCase):
    def run_main(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = clean_utf8.main(list(arguments))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_preview_then_apply_repairs_text_and_backs_up_originals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            root = temporary / "copied-grim"
            root.mkdir()
            csv_path = root / "measurement.csv"
            csv_original = b'name,comment\r\n1,"\x93quoted\x94"\r\n'
            csv_path.write_bytes(csv_original)
            text_path = root / "notes.txt"
            text_original = "alpha\r\nbeta\r\n".encode("utf-16")
            text_path.write_bytes(text_original)
            binary_path = root / "measurement.ptm"
            binary_original = b"\x00\x81\xff\x10binary"
            binary_path.write_bytes(binary_original)

            status, stdout, stderr = self.run_main(str(root))
            self.assertEqual(status, 0, stderr)
            self.assertIn("[WOULD CONVERT] measurement.csv", stdout)
            self.assertEqual(csv_path.read_bytes(), csv_original)
            self.assertEqual(text_path.read_bytes(), text_original)

            status, stdout, stderr = self.run_main(str(root), "--apply")
            self.assertEqual(status, 0, stderr)
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                csv_text = stream.read()
            with text_path.open("r", encoding="utf-8", newline="") as stream:
                note_text = stream.read()
            self.assertEqual(csv_text, 'name,comment\r\n1,"\u201cquoted\u201d"\r\n')
            self.assertEqual(note_text, "alpha\r\nbeta\r\n")
            self.assertEqual(binary_path.read_bytes(), binary_original)

            backups = list(temporary.glob("copied-grim-utf8-backup-*.zip"))
            self.assertEqual(len(backups), 1)
            with zipfile.ZipFile(backups[0]) as archive:
                self.assertEqual(
                    archive.read("originals/measurement.csv"), csv_original
                )
                self.assertEqual(archive.read("originals/notes.txt"), text_original)
                manifest = json.loads(
                    archive.read("UTF8_CLEANUP_MANIFEST.json").decode("utf-8")
                )
            self.assertEqual(
                {entry["path"] for entry in manifest["files"]},
                {"measurement.csv", "notes.txt"},
            )
            self.assertIn("Recovery ZIP:", stdout)

    def test_python_encoding_cookie_and_executable_mode_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "worker.py"
            path.write_bytes(b"#!/usr/bin/env python3\n# coding: cp1252\nvalue = '\x96'\n")
            os.chmod(path, 0o755)

            status, _, stderr = self.run_main(
                str(path), "--apply", "--no-backup"
            )
            self.assertEqual(status, 0, stderr)
            decoded = path.read_text(encoding="utf-8")
            self.assertIn("# coding: utf-8", decoded)
            self.assertIn("\u2013", decoded)
            if os.name != "nt":
                self.assertTrue(os.stat(path).st_mode & stat.S_IXUSR)

    def test_string_that_mentions_coding_is_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "message.py"
            path.write_bytes(b'print("coding: cp1252") # \x96\n')

            status, _, stderr = self.run_main(
                str(path), "--apply", "--no-backup"
            )
            self.assertEqual(status, 0, stderr)
            decoded = path.read_text(encoding="utf-8")
            self.assertIn('print("coding: cp1252")', decoded)
            self.assertIn("\u2013", decoded)

    def test_valid_utf8_sequences_survive_a_mixed_encoding_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "mixed.py"
            path.write_bytes(
                "# snowman: \u2603; copied dash: ".encode("utf-8") + b"\x97\n"
            )

            status, _, stderr = self.run_main(
                str(path), "--apply", "--no-backup"
            )
            self.assertEqual(status, 0, stderr)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "# snowman: \u2603; copied dash: \u2014\n",
            )

    def test_utf8_bom_is_valid_and_left_byte_for_byte_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "pinned.csv"
            original = b"\xef\xbb\xbffrequency,value\r\n1,2\r\n"
            path.write_bytes(original)

            status, stdout, stderr = self.run_main(str(path), "--apply")
            self.assertEqual(status, 0, stderr)
            self.assertIn("No files changed", stdout)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("pinned.csv-utf8-backup-*.zip")), [])

    def test_undefined_windows_byte_is_reported_and_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "unknown.csv"
            original = b"heading\nvalue=\x81\n"
            path.write_bytes(original)

            status, stdout, stderr = self.run_main(str(path), "--apply")
            self.assertEqual(status, 1)
            self.assertIn("[UNRESOLVED]", stderr)
            self.assertIn("No files changed", stdout)
            self.assertEqual(path.read_bytes(), original)

    def test_unknown_extension_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "input.custom"
            path.write_bytes(b"dash=\x97\n")

            status, _, stderr = self.run_main(str(path), "--apply")
            self.assertEqual(status, 1)
            self.assertIn("unrecognized text extension", stderr)
            self.assertEqual(path.read_bytes(), b"dash=\x97\n")

            status, _, stderr = self.run_main(
                str(path),
                "--include-extension",
                ".custom",
                "--apply",
                "--no-backup",
            )
            self.assertEqual(status, 0, stderr)
            self.assertEqual(path.read_text(encoding="utf-8"), "dash=\u2014\n")

    def test_xml_declaration_is_updated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "sample.xml"
            path.write_bytes(
                b'<?xml version="1.0" encoding="windows-1252"?>\r\n'
                b"<value>\x96</value>\r\n"
            )

            status, _, stderr = self.run_main(
                str(path), "--apply", "--no-backup"
            )
            self.assertEqual(status, 0, stderr)
            with path.open("r", encoding="utf-8", newline="") as stream:
                decoded = stream.read()
            self.assertIn('encoding="utf-8"', decoded)
            self.assertIn("\u2013", decoded)
            self.assertIn("\r\n", decoded)

    def test_concurrent_save_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "changing.py"
            path.write_bytes(b"# copied dash: \x97\n")
            result = clean_utf8.scan_tree(path)
            self.assertEqual(len(result.conversions), 1)
            item = result.conversions[0]
            original_writer = clean_utf8._write_utf8_temporary

            def write_then_change(
                conversion: clean_utf8.Conversion,
            ) -> Path:
                temporary = original_writer(conversion)
                path.write_text("# newer user save\n", encoding="utf-8")
                return temporary

            with mock.patch.object(
                clean_utf8,
                "_write_utf8_temporary",
                side_effect=write_then_change,
            ):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    clean_utf8.apply_conversion(item)

            self.assertEqual(
                path.read_text(encoding="utf-8"), "# newer user save\n"
            )
            self.assertEqual(list(path.parent.glob(".changing.py.utf8-*.tmp")), [])

    def test_symlinked_directory_is_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            root = temporary / "root"
            outside = temporary / "outside"
            root.mkdir()
            outside.mkdir()
            damaged = outside / "outside.py"
            original = b"# dash: \x97\n"
            damaged.write_bytes(original)
            link = root / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")

            result = clean_utf8.scan_tree(root)
            self.assertEqual(result.candidate_count, 0)
            self.assertEqual(result.conversions, ())
            self.assertEqual(damaged.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
