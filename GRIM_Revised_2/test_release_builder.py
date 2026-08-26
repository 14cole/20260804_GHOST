from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import build_release


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        digest, path = line.split("  ", 1)
        entries[path] = digest
    return entries


class ReleaseBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for index, relative_name in enumerate(build_release.REQUIRED_FILES):
            path = self.source.joinpath(*PurePosixPath(relative_name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative_name == "pyproject.toml":
                content = '[project]\nname = "ghost-grim"\nversion = "7.8.9"\n'
            else:
                content = f"sentinel {index}: {relative_name}\n"
            path.write_text(content, encoding="utf-8", newline="\n")

        extra = self.source / "tools" / "FREDDY" / "materials" / "asset.csv"
        extra.write_text("frequency_GHz,epsilon_real\n1,2\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_complete_release_includes_assets_and_valid_hash_manifests(self) -> None:
        (self.source / ".git").mkdir()
        (self.source / ".git" / "config").write_text("private", encoding="utf-8")
        (self.source / ".venv" / "Lib").mkdir(parents=True)
        (self.source / ".venv" / "Lib" / "dependency.py").write_text(
            "ignored", encoding="utf-8"
        )
        codex_scratch = self.source / ".codex-tmp" / "node_modules"
        codex_scratch.mkdir(parents=True)
        (codex_scratch / "external-runtime.js").write_text(
            "ignored", encoding="utf-8"
        )
        cache = self.source / "GRIM_Revised_2" / "__pycache__"
        cache.mkdir()
        (cache / "grim_cut_gui.cpython-312.pyc").write_bytes(b"ignored")
        pytest_cache = self.source / "tools" / "GHOST" / ".pytest_cache"
        pytest_cache.mkdir()
        (pytest_cache / "state").write_text("ignored", encoding="utf-8")
        (self.source / "scratch.tmp").write_text("ignored", encoding="utf-8")

        result = build_release.build_release(self.source, self.root / "output")

        self.assertEqual(result.release_name, "GRIM-7.8.9")
        self.assertTrue(result.release_directory.is_dir())
        self.assertTrue(result.archive_path.is_file())
        self.assertTrue(result.external_manifest_path.is_file())
        for relative_name in build_release.REQUIRED_FILES:
            path = result.release_directory.joinpath(
                *PurePosixPath(relative_name).parts
            )
            self.assertTrue(path.is_file(), relative_name)

        excluded_fragments = (
            "/.git/",
            "/.venv/",
            "/.codex-tmp/",
            "/__pycache__/",
            "/.pytest_cache/",
        )
        with zipfile.ZipFile(result.archive_path) as archive:
            archive_names = archive.namelist()
            self.assertEqual(archive_names, sorted(archive_names))
            for relative_name in build_release.REQUIRED_FILES:
                self.assertIn(f"GRIM-7.8.9/{relative_name}", archive_names)
            self.assertIn(
                "GRIM-7.8.9/tools/FREDDY/materials/asset.csv", archive_names
            )
            self.assertIn("GRIM-7.8.9/tools/GHOST/Backend/ghost_gui.py", archive_names)
            self.assertIn("GRIM-7.8.9/SHA256SUMS.txt", archive_names)
            archived_internal_manifest = archive.read(
                "GRIM-7.8.9/SHA256SUMS.txt"
            )
            self.assertFalse(
                any(fragment in name for name in archive_names for fragment in excluded_fragments)
            )
            self.assertFalse(any(name.endswith(".tmp") for name in archive_names))
            for name in archive_names:
                self.assertEqual(name, PurePosixPath(name).as_posix())
                self.assertFalse(PurePosixPath(name).is_absolute())
                self.assertNotIn("..", PurePosixPath(name).parts)

        internal_text = (result.release_directory / "SHA256SUMS.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(archived_internal_manifest, internal_text.encode("utf-8"))
        internal_entries = _manifest_entries(internal_text)
        self.assertEqual(list(internal_entries), sorted(internal_entries))
        expected_payload_paths = {
            path.relative_to(result.release_directory).as_posix()
            for path in result.release_directory.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.txt"
        }
        self.assertEqual(set(internal_entries), expected_payload_paths)
        for relative_name, expected_digest in internal_entries.items():
            self.assertEqual(
                _sha256(result.release_directory / PurePosixPath(relative_name)),
                expected_digest,
            )

        external_entries = _manifest_entries(
            result.external_manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(external_entries[result.archive_path.name], _sha256(result.archive_path))
        extracted_prefix = f"{result.release_name}/"
        extracted_entries = {
            name.removeprefix(extracted_prefix): digest
            for name, digest in external_entries.items()
            if name.startswith(extracted_prefix)
        }
        all_extracted_paths = {
            path.relative_to(result.release_directory).as_posix()
            for path in result.release_directory.rglob("*")
            if path.is_file()
        }
        self.assertEqual(set(extracted_entries), all_extracted_paths)
        for relative_name, expected_digest in extracted_entries.items():
            self.assertEqual(
                _sha256(result.release_directory / PurePosixPath(relative_name)),
                expected_digest,
            )

    def test_archive_and_relative_paths_are_deterministic(self) -> None:
        first = build_release.build_release(self.source, self.root / "out-one")
        second = build_release.build_release(self.source, self.root / "out-two")

        self.assertEqual(_sha256(first.archive_path), _sha256(second.archive_path))
        self.assertEqual(
            (first.release_directory / "SHA256SUMS.txt").read_bytes(),
            (second.release_directory / "SHA256SUMS.txt").read_bytes(),
        )
        with zipfile.ZipFile(first.archive_path) as first_zip, zipfile.ZipFile(
            second.archive_path
        ) as second_zip:
            self.assertEqual(first_zip.namelist(), second_zip.namelist())

    def test_partial_source_fails_before_creating_output(self) -> None:
        missing = self.source / "tools" / "FREDDY" / "ibc" / "ui.py"
        missing.unlink()
        output = self.root / "must-not-exist"

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "source tree is incomplete"
        ):
            build_release.build_release(self.source, output)

        self.assertFalse(output.exists())

    def test_missing_material_explorer_dependency_fails_before_output(self) -> None:
        missing = self.source / "tools" / "FREDDY" / "ibc" / "material_explorer.py"
        missing.unlink()
        output = self.root / "must-not-exist"

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "material_explorer.py"
        ):
            build_release.build_release(self.source, output)

        self.assertFalse(output.exists())

    def test_missing_direct_grim_startup_module_fails_before_output(self) -> None:
        missing = self.source / "GRIM_Revised_2" / "grim_python.py"
        missing.unlink()
        output = self.root / "must-not-exist"

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "grim_python.py"
        ):
            build_release.build_release(self.source, output)

        self.assertFalse(output.exists())

    def test_missing_standalone_launcher_module_fails_before_output(self) -> None:
        missing = self.source / "GRIM_Revised_2" / "ppt_image_imprinter.py"
        missing.unlink()
        output = self.root / "must-not-exist"

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "ppt_image_imprinter.py"
        ):
            build_release.build_release(self.source, output)

        self.assertFalse(output.exists())

    def test_missing_feature_workflow_module_fails_before_output(self) -> None:
        missing = (
            self.source
            / "tools"
            / "GHOST"
            / "Backend"
            / "feature_workflow.py"
        )
        missing.unlink()
        output = self.root / "must-not-exist"

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "feature_workflow.py"
        ):
            build_release.build_release(self.source, output)

        self.assertFalse(output.exists())

    def test_existing_release_is_not_overwritten(self) -> None:
        output = self.root / "output"
        output.mkdir()
        existing_zip = output / "GRIM-7.8.9.zip"
        existing_zip.write_bytes(b"keep me")

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "will not be overwritten"
        ):
            build_release.build_release(self.source, output)

        self.assertEqual(existing_zip.read_bytes(), b"keep me")
        self.assertFalse((output / "GRIM-7.8.9").exists())


if __name__ == "__main__":
    unittest.main()
