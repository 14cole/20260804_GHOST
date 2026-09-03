from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
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
        self.inventory = tuple(
            path.relative_to(self.source)
            for path in self.source.rglob("*")
            if path.is_file()
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def build(self, output: Path, **kwargs) -> build_release.ReleaseArtifacts:
        return build_release.build_release(
            self.source,
            output,
            source_inventory=self.inventory,
            run_acceptance=False,
            **kwargs,
        )

    def test_complete_release_includes_assets_and_valid_hash_manifests(self) -> None:
        (self.source / ".venv" / "Lib").mkdir(parents=True)
        (self.source / ".venv" / "Lib" / "dependency.py").write_text(
            "ignored", encoding="utf-8"
        )
        hidden_scratch = self.source / ".workspace-scratch" / "node_modules"
        hidden_scratch.mkdir(parents=True)
        (hidden_scratch / "external-runtime.js").write_text(
            "ignored", encoding="utf-8"
        )
        cache = self.source / "GRIM_Revised_2" / "__pycache__"
        cache.mkdir()
        (cache / "grim_cut_gui.cpython-312.pyc").write_bytes(b"ignored")
        pytest_cache = self.source / "tools" / "GHOST" / ".pytest_cache"
        pytest_cache.mkdir()
        (pytest_cache / "state").write_text("ignored", encoding="utf-8")
        (self.source / "scratch.tmp").write_text("ignored", encoding="utf-8")

        result = self.build(self.root / "output")

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
            "/.venv/",
            "/.workspace-scratch/",
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
            self.assertIn("GRIM-7.8.9/BUILD-INFO.json", archive_names)
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

        build_info = json.loads(
            (result.release_directory / "BUILD-INFO.json").read_text(encoding="utf-8")
        )
        self.assertEqual(build_info["version"], "7.8.9")
        self.assertEqual(build_info["source"]["inventory_kind"], "explicit-inventory")
        self.assertTrue(
            result.build_id.endswith(build_info["source"]["tree_sha256"][:12])
        )
        self.assertEqual(build_info["acceptance"]["diagnostics"], "not-run")
        self.assertEqual(build_info["build"]["id"], result.build_id)

    def test_archive_and_relative_paths_are_deterministic(self) -> None:
        first = self.build(self.root / "out-one")
        second = self.build(self.root / "out-two")

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
            self.build(output)

        self.assertFalse(output.exists())

    def test_missing_material_explorer_dependency_fails_before_output(self) -> None:
        missing = self.source / "tools" / "FREDDY" / "ibc" / "material_explorer.py"
        missing.unlink()
        output = self.root / "must-not-exist"

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "material_explorer.py"
        ):
            self.build(output)

        self.assertFalse(output.exists())

    def test_missing_direct_grim_startup_module_fails_before_output(self) -> None:
        missing = self.source / "GRIM_Revised_2" / "grim_python.py"
        missing.unlink()
        output = self.root / "must-not-exist"

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "grim_python.py"
        ):
            self.build(output)

        self.assertFalse(output.exists())

    def test_missing_standalone_launcher_module_fails_before_output(self) -> None:
        missing = self.source / "GRIM_Revised_2" / "ppt_image_imprinter.py"
        missing.unlink()
        output = self.root / "must-not-exist"

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "ppt_image_imprinter.py"
        ):
            self.build(output)

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
            self.build(output)

        self.assertFalse(output.exists())

    def test_missing_assembly_workload_contract_fails_before_output(self) -> None:
        missing = (
            self.source
            / "tools"
            / "GHOST"
            / "Backend"
            / "assembly_workload.py"
        )
        missing.unlink()
        output = self.root / "must-not-exist"

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "assembly_workload.py"
        ):
            self.build(output)

        self.assertFalse(output.exists())

    def test_existing_release_is_not_overwritten(self) -> None:
        output = self.root / "output"
        output.mkdir()
        existing_zip = output / "GRIM-7.8.9.zip"
        existing_zip.write_bytes(b"keep me")

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "will not be overwritten"
        ):
            self.build(output)

        self.assertEqual(existing_zip.read_bytes(), b"keep me")
        self.assertFalse((output / "GRIM-7.8.9").exists())

    def test_explicit_inventory_excludes_unreviewed_file(self) -> None:
        unreviewed = self.source / "customer-secret.csv"
        unreviewed.write_text("do not ship\n", encoding="utf-8")

        result = self.build(self.root / "output")

        self.assertFalse((result.release_directory / unreviewed.name).exists())
        with zipfile.ZipFile(result.archive_path) as archive:
            self.assertNotIn(f"{result.release_name}/{unreviewed.name}", archive.namelist())

    def test_explicit_inventory_cannot_bypass_git_acceptance(self) -> None:
        (self.source / ".git").mkdir()
        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "only for a reviewed source export"
        ):
            self.build(self.root / "output")

    def test_explicit_inventory_cannot_bypass_parent_git_worktree(self) -> None:
        (self.root / ".git").mkdir()
        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "inside a Git worktree"
        ):
            self.build(self.root / "output")

    def test_explicit_inventory_must_include_present_runtime_modules(self) -> None:
        lazy_module = self.source / "tools" / "GHOST" / "Backend" / "lazy_runtime.py"
        lazy_module.write_text("VALUE = 1\n", encoding="utf-8")

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "lazy_runtime.py"
        ):
            self.build(self.root / "output")

    def test_explicit_inventory_must_include_present_acceptance_tests(self) -> None:
        acceptance_test = self.source / "GRIM_Revised_2" / "test_new_gate.py"
        acceptance_test.write_text("import unittest\n", encoding="utf-8")

        with self.assertRaisesRegex(
            build_release.ReleaseBuildError, "test_new_gate.py"
        ):
            self.build(self.root / "output")

    def test_git_dirty_source_is_rejected_before_file_collection(self) -> None:
        responses = {
            ("rev-parse", "--show-toplevel"): (str(self.source) + "\n").encode(),
            (
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored=no",
            ): b"?? customer-secret.csv\0",
        }

        with mock.patch.object(
            build_release,
            "_run_git",
            side_effect=lambda _root, args: responses[tuple(args)],
        ):
            with self.assertRaisesRegex(build_release.ReleaseBuildError, "not exactly clean"):
                build_release.discover_source_inventory(self.source)

    def _clean_git_responses(self, tags: bytes) -> dict[tuple[str, ...], bytes]:
        return {
            ("rev-parse", "--show-toplevel"): (str(self.source) + "\n").encode(),
            (
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored=no",
            ): b"",
            ("ls-files", "--cached", "-z"): b"pyproject.toml\0",
            ("rev-parse", "HEAD"): (b"a" * 40) + b"\n",
            ("tag", "--points-at", "HEAD", "--list"): tags,
            ("show", "-s", "--format=%cI", "HEAD"): b"2026-08-30T10:00:00-07:00\n",
        }

    def test_git_release_requires_matching_version_tag(self) -> None:
        responses = self._clean_git_responses(b"unrelated\n")
        with mock.patch.object(
            build_release,
            "_run_git",
            side_effect=lambda _root, args: responses[tuple(args)],
        ):
            with self.assertRaisesRegex(build_release.ReleaseBuildError, "not tagged"):
                build_release.discover_source_inventory(
                    self.source,
                    expected_version="7.8.9",
                )

    def test_git_release_records_matching_version_tag(self) -> None:
        responses = self._clean_git_responses(b"7.8.9\nv7.8.9\n")
        with mock.patch.object(
            build_release,
            "_run_git",
            side_effect=lambda _root, args: responses[tuple(args)],
        ):
            inventory = build_release.discover_source_inventory(
                self.source,
                expected_version="7.8.9",
            )

        self.assertEqual(inventory.tag, "v7.8.9")
        self.assertEqual(inventory.revision, "a" * 40)
        build_info_text, _build_id = build_release._build_info_text(
            release_name="GRIM-7.8.9",
            version="7.8.9",
            source=inventory,
            source_records=(build_release.FileRecord("pyproject.toml", 1, "b" * 64),),
            acceptance=build_release.AcceptanceReport(
                tests=("suite",),
                diagnostics="pass",
                utf8="pass",
                forbidden_terms="pass",
                dependency_lock="pass",
                native_policy="warn",
                native_status="native_bor=WARN",
            ),
            constraints_sha256="c" * 64,
        )
        self.assertEqual(json.loads(build_info_text)["source"]["tag"], "v7.8.9")

    def test_version_override_must_match_project_metadata(self) -> None:
        with self.assertRaisesRegex(build_release.ReleaseBuildError, "disagrees"):
            self.build(self.root / "output", version="7.8.10")

    def test_forbidden_term_gate_reports_file_and_line(self) -> None:
        path = self.source / "release-note.txt"
        path.write_text("safe\n" + "clau" + "de\n", encoding="utf-8")
        with self.assertRaisesRegex(build_release.ReleaseBuildError, "release-note.txt:2"):
            build_release._validate_forbidden_terms(
                self.source,
                (Path("release-note.txt"),),
            )

    def test_forbidden_term_gate_inspects_powerpoint_xml(self) -> None:
        path = self.source / "slides.pptx"
        with zipfile.ZipFile(path, "w") as package:
            package.writestr(
                "ppt/slides/slide1.xml",
                "<text>" + "co" + "dex" + "</text>",
            )
        with self.assertRaisesRegex(build_release.ReleaseBuildError, "slide1.xml"):
            build_release._validate_forbidden_terms(
                self.source,
                (Path("slides.pptx"),),
            )

    def test_inventory_rejects_parent_traversal(self) -> None:
        with self.assertRaisesRegex(build_release.ReleaseBuildError, "Unsafe"):
            build_release.discover_source_inventory(
                self.source,
                explicit_files=("../outside.txt",),
            )

    def test_native_require_policy_blocks_fallback_release(self) -> None:
        diagnostic = SimpleNamespace(
            key="native_bor",
            status="WARN",
            summary="fallback",
            blocks_startup=False,
        )
        with (
            mock.patch.object(build_release, "_validate_utf8_payload"),
            mock.patch.object(build_release, "_validate_forbidden_terms"),
            mock.patch.object(build_release, "_validate_dependency_lock"),
            mock.patch.object(build_release, "collect_diagnostics", return_value=[diagnostic]),
            mock.patch.object(build_release, "startup_exit_code", return_value=0),
        ):
            with self.assertRaisesRegex(build_release.ReleaseBuildError, "Native acceleration"):
                build_release.run_acceptance_gates(
                    self.source,
                    self.inventory,
                    native_policy="require",
                )

    def test_acceptance_includes_non_unittest_ghost_integration_gates(self) -> None:
        diagnostics = [
            SimpleNamespace(
                key=key,
                status="PASS",
                summary="loaded",
                blocks_startup=False,
            )
            for key in ("native_bor",)
        ]
        with (
            mock.patch.object(build_release, "_validate_utf8_payload"),
            mock.patch.object(build_release, "_validate_forbidden_terms"),
            mock.patch.object(build_release, "_validate_dependency_lock"),
            mock.patch.object(
                build_release,
                "collect_diagnostics",
                return_value=diagnostics,
            ),
            mock.patch.object(
                build_release, "startup_exit_code", return_value=0
            ),
            mock.patch.object(build_release, "_run_test_suite") as run_suite,
        ):
            report = build_release.run_acceptance_gates(
                self.source,
                self.inventory,
                native_policy="require",
            )

        names = [call.args[0] for call in run_suite.call_args_list]
        self.assertIn("GHOST CEM tools tests", names)
        self.assertIn("GHOST HPC scheduling integration", names)
        self.assertIn("GHOST local-driver integration", names)
        self.assertIn("GHOST ASCII-transfer compatibility", names)
        self.assertEqual(tuple(names), report.tests)

    def test_dependency_lock_must_cover_new_project_dependency(self) -> None:
        (self.source / "pyproject.toml").write_text(
            '[project]\nname = "ghost-grim"\nversion = "7.8.9"\n'
            'dependencies = ["numpy>=1", "new-solver>=2"]\n',
            encoding="utf-8",
        )
        lock = self.source.joinpath(*PurePosixPath(build_release.CONSTRAINTS_PATH).parts)
        lock.write_text("numpy==2.5.2\n", encoding="utf-8")

        with self.assertRaisesRegex(build_release.ReleaseBuildError, "new-solver"):
            build_release._read_exact_constraints(self.source)

    def test_dependency_gate_requires_windows_x64_builder(self) -> None:
        with mock.patch.object(build_release.sys, "platform", "linux"):
            with self.assertRaisesRegex(
                build_release.ReleaseBuildError, "built and tested on 64-bit Windows"
            ):
                build_release._validate_dependency_lock(self.source)

        with (
            mock.patch.object(build_release.sys, "platform", "win32"),
            mock.patch.object(build_release.platform, "machine", return_value="ARM64"),
        ):
            with self.assertRaisesRegex(
                build_release.ReleaseBuildError, "64-bit x86 Windows"
            ):
                build_release._validate_dependency_lock(self.source)

        with (
            mock.patch.object(build_release.sys, "platform", "win32"),
            mock.patch.object(build_release.platform, "machine", return_value="AMD64"),
            mock.patch.object(build_release.struct, "calcsize", return_value=4),
        ):
            with self.assertRaisesRegex(
                build_release.ReleaseBuildError, "Python process is 32-bit"
            ):
                build_release._validate_dependency_lock(self.source)

    def test_dependency_gate_rejects_missing_locked_optional_package(self) -> None:
        (self.source / "pyproject.toml").write_text(
            '[project]\nname = "ghost-grim"\nversion = "7.8.9"\n'
            'dependencies = []\n[project.optional-dependencies]\nextra = ["alpha>=1"]\n',
            encoding="utf-8",
        )
        lock = self.source.joinpath(*PurePosixPath(build_release.CONSTRAINTS_PATH).parts)
        lock.write_text("alpha==1.2.3\n", encoding="utf-8")

        with mock.patch.object(
            build_release.importlib_metadata,
            "version",
            side_effect=build_release.importlib_metadata.PackageNotFoundError,
        ):
            with self.assertRaisesRegex(
                build_release.ReleaseBuildError, "alpha: required 1.2.3, not installed"
            ):
                build_release._validate_dependency_lock(self.source)

    def test_atomic_file_publication_refuses_existing_destination(self) -> None:
        source = self.root / "complete.tmp"
        destination = self.root / "release.zip"
        source.write_bytes(b"complete archive")
        destination.write_bytes(b"existing")

        with self.assertRaises(build_release.ReleaseBuildError):
            build_release._publish_file_atomic(source, destination)

        self.assertEqual(destination.read_bytes(), b"existing")
        self.assertEqual(source.read_bytes(), b"complete archive")

    def test_external_manifest_is_published_last_as_completion_marker(self) -> None:
        events: list[str] = []

        def directory_side_effect(_source: Path, destination: Path) -> None:
            events.append(destination.name)

        def file_side_effect(_source: Path, destination: Path) -> None:
            events.append(destination.name)

        with (
            mock.patch.object(
                build_release,
                "_publish_release_directory",
                side_effect=directory_side_effect,
            ),
            mock.patch.object(
                build_release,
                "_publish_file_atomic",
                side_effect=file_side_effect,
            ),
        ):
            self.build(self.root / "output")

        self.assertEqual(
            events,
            ["GRIM-7.8.9", "GRIM-7.8.9.zip", "GRIM-7.8.9-SHA256SUMS.txt"],
        )

    def test_utf8_gate_rejects_nul_contaminated_text(self) -> None:
        path = self.source / "nul.txt"
        path.write_bytes(b"left\x00right\n")
        with self.assertRaisesRegex(build_release.ReleaseBuildError, "NUL"):
            build_release._validate_utf8_payload(self.source, (Path("nul.txt"),))


if __name__ == "__main__":
    unittest.main()
