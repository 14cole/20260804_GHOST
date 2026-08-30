from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

import wheelhouse_manifest


class WheelhouseManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.wheelhouse = self.root / "wheelhouse"
        self.wheelhouse.mkdir()
        self.constraints = self.root / "constraints.txt"
        self.constraints.write_text(
            "alpha-pkg==1.2.3\nbeta_pkg==4.5.6\n",
            encoding="utf-8",
        )
        self._write_wheel("alpha_pkg-1.2.3-py3-none-any.whl", "alpha_pkg-1.2.3")
        self._write_wheel("beta_pkg-4.5.6-py3-none-any.whl", "beta_pkg-4.5.6")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_wheel(self, filename: str, dist_info: str) -> Path:
        path = self.wheelhouse / filename
        python_tag, abi_tag, platform_tag = path.stem.split("-")[-3:]
        tag = f"{python_tag}-{abi_tag}-{platform_tag}"
        identity = dist_info.rsplit("-", 1)
        self.assertEqual(len(identity), 2)
        distribution, version = identity
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"{dist_info}.dist-info/WHEEL",
                f"Wheel-Version: 1.0\nTag: {tag}\n",
            )
            archive.writestr(
                f"{dist_info}.dist-info/METADATA",
                "Metadata-Version: 2.1\n"
                f"Name: {distribution}\n"
                f"Version: {version}\n",
            )
            archive.writestr(f"{dist_info}.dist-info/RECORD", "")
        return path

    def test_generate_and_verify_exact_locked_wheels(self) -> None:
        manifest = wheelhouse_manifest.generate_manifest(
            self.wheelhouse,
            self.constraints,
        )

        self.assertEqual(manifest.name, wheelhouse_manifest.MANIFEST_NAME)
        self.assertEqual(
            wheelhouse_manifest.verify_manifest(self.wheelhouse, self.constraints),
            2,
        )
        first_bytes = manifest.read_bytes()
        self.assertEqual(
            wheelhouse_manifest.generate_manifest(
                self.wheelhouse,
                self.constraints,
            ).read_bytes(),
            first_bytes,
        )
        entries = [
            line for line in manifest.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        self.assertEqual(entries, sorted(entries, key=lambda value: value.split("  ", 1)[1]))

    def test_tampered_wheel_fails_sha256_verification(self) -> None:
        wheelhouse_manifest.generate_manifest(self.wheelhouse, self.constraints)
        wheel = self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl"
        with wheel.open("ab") as stream:
            stream.write(b"tampered")

        with self.assertRaisesRegex(wheelhouse_manifest.WheelhouseError, "SHA-256"):
            wheelhouse_manifest.verify_manifest(self.wheelhouse, self.constraints)

    def test_manifest_is_bound_to_exact_constraints_file(self) -> None:
        wheelhouse_manifest.generate_manifest(self.wheelhouse, self.constraints)
        with self.constraints.open("a", encoding="utf-8") as stream:
            stream.write("# changed review note\n")

        with self.assertRaisesRegex(wheelhouse_manifest.WheelhouseError, "Constraints SHA-256"):
            wheelhouse_manifest.verify_manifest(self.wheelhouse, self.constraints)

    def test_wrong_wheel_version_is_rejected_before_manifest(self) -> None:
        (self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl").unlink()
        self._write_wheel("alpha_pkg-9.9.9-py3-none-any.whl", "alpha_pkg-9.9.9")

        with self.assertRaisesRegex(wheelhouse_manifest.WheelhouseError, "version mismatch"):
            wheelhouse_manifest.generate_manifest(self.wheelhouse, self.constraints)

    def test_unlocked_extra_wheel_is_rejected(self) -> None:
        self._write_wheel("extra_pkg-1.0-py3-none-any.whl", "extra_pkg-1.0")

        with self.assertRaisesRegex(wheelhouse_manifest.WheelhouseError, "unlocked package"):
            wheelhouse_manifest.generate_manifest(self.wheelhouse, self.constraints)

    def test_linux_wheel_is_rejected_for_windows_release(self) -> None:
        (self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl").unlink()
        self._write_wheel(
            "alpha_pkg-1.2.3-cp312-cp312-manylinux_2_28_x86_64.whl",
            "alpha_pkg-1.2.3",
        )

        with self.assertRaisesRegex(
            wheelhouse_manifest.WheelhouseError, "not compatible"
        ):
            wheelhouse_manifest.generate_manifest(
                self.wheelhouse, self.constraints
            )

    def test_newer_cpython_wheel_is_rejected(self) -> None:
        (self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl").unlink()
        self._write_wheel(
            "alpha_pkg-1.2.3-cp313-cp313-win_amd64.whl",
            "alpha_pkg-1.2.3",
        )

        with self.assertRaisesRegex(
            wheelhouse_manifest.WheelhouseError, "not compatible"
        ):
            wheelhouse_manifest.generate_manifest(
                self.wheelhouse, self.constraints
            )

    def test_older_abi3_windows_wheel_is_accepted(self) -> None:
        (self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl").unlink()
        self._write_wheel(
            "alpha_pkg-1.2.3-cp39-abi3-win_amd64.whl",
            "alpha_pkg-1.2.3",
        )

        wheelhouse_manifest.generate_manifest(self.wheelhouse, self.constraints)
        self.assertEqual(
            wheelhouse_manifest.verify_manifest(
                self.wheelhouse, self.constraints
            ),
            2,
        )

    def test_older_pure_python_wheel_is_accepted(self) -> None:
        (self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl").unlink()
        self._write_wheel(
            "alpha_pkg-1.2.3-py39-none-any.whl",
            "alpha_pkg-1.2.3",
        )

        wheelhouse_manifest.generate_manifest(self.wheelhouse, self.constraints)
        self.assertEqual(
            wheelhouse_manifest.verify_manifest(
                self.wheelhouse, self.constraints
            ),
            2,
        )

    def test_extension_abi_with_platform_any_is_rejected(self) -> None:
        (self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl").unlink()
        self._write_wheel(
            "alpha_pkg-1.2.3-cp312-cp312-any.whl",
            "alpha_pkg-1.2.3",
        )

        with self.assertRaisesRegex(
            wheelhouse_manifest.WheelhouseError, "not compatible"
        ):
            wheelhouse_manifest.generate_manifest(
                self.wheelhouse, self.constraints
            )

    def test_dist_info_identity_must_match_filename(self) -> None:
        (self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl").unlink()
        self._write_wheel(
            "alpha_pkg-1.2.3-py3-none-any.whl",
            "other_pkg-1.2.3",
        )

        with self.assertRaisesRegex(
            wheelhouse_manifest.WheelhouseError, "contradictory package identity"
        ):
            wheelhouse_manifest.generate_manifest(
                self.wheelhouse, self.constraints
            )

    def test_metadata_identity_must_match_filename(self) -> None:
        wheel = self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                "alpha_pkg-1.2.3.dist-info/WHEEL",
                "Wheel-Version: 1.0\nTag: py3-none-any\n",
            )
            archive.writestr(
                "alpha_pkg-1.2.3.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: alpha-pkg\nVersion: 9.9.9\n",
            )
            archive.writestr("alpha_pkg-1.2.3.dist-info/RECORD", "")

        with self.assertRaisesRegex(
            wheelhouse_manifest.WheelhouseError, "contradictory package identity"
        ):
            wheelhouse_manifest.generate_manifest(
                self.wheelhouse, self.constraints
            )

    def test_supported_filename_tag_must_also_be_in_wheel_metadata(self) -> None:
        wheel = self.wheelhouse / "alpha_pkg-1.2.3-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                "alpha_pkg-1.2.3.dist-info/WHEEL",
                "Wheel-Version: 1.0\nTag: cp313-cp313-win_amd64\n",
            )
            archive.writestr(
                "alpha_pkg-1.2.3.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: alpha-pkg\nVersion: 1.2.3\n",
            )
            archive.writestr("alpha_pkg-1.2.3.dist-info/RECORD", "")

        with self.assertRaisesRegex(
            wheelhouse_manifest.WheelhouseError, "contradict"
        ):
            wheelhouse_manifest.generate_manifest(
                self.wheelhouse, self.constraints
            )


if __name__ == "__main__":
    unittest.main()
