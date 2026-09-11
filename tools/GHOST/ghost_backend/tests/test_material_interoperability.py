"""Real FREDDY writers -> shared GHOST 2-D/BoR material readers."""
from pathlib import Path
import sys
import tempfile
import unittest

import csv
import math

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
FREDDY_ROOT = Path(__file__).resolve().parents[3] / "FREDDY"
if str(FREDDY_ROOT) not in sys.path:
    sys.path.insert(0, str(FREDDY_ROOT))
from ibc.compute import MaterialTable
from ibc.io import (write_output, write_material_table, read_material_table,
                    write_impedance_batch, write_impedance_bundle,
                    MATERIAL_HEADER, IMPEDANCE_HEADER)
from ghost_backend.twod.solver import MaterialLibrary, _load_impedance_csv, _load_dielectric_csv
from ghost_backend.geometry.io import parse_geometry
from ghost_backend.io.grim import export_result_to_dbke_csv


class FreddyGhostMaterialTests(unittest.TestCase):
    def test_nominal_exports_are_hz_and_preserve_complex_properties(self):
        frequencies = [.125, 1.125, 2.125]
        impedance = [12-3j, 24+5j, 40+9j]
        eps = [2-.1j, 3-.3j, 5-.5j]
        mu = [1-.02j, 1.2-.04j, 1.6-.08j]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ibc_path, material_path = root/"impedance.csv", root/"dielectric.csv"
            write_output(ibc_path, list(zip(frequencies, np.real(impedance), np.imag(impedance))))
            write_material_table(material_path, MaterialTable(frequencies, eps, mu))
            for path, width in ((ibc_path, 3), (material_path, 5)):
                rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
                self.assertEqual(rows.shape, (3, width))
                np.testing.assert_array_equal(rows[:, 0], [125000000, 1125000000, 2125000000])
                self.assertTrue(path.read_text().startswith("frequency_hz,"))
            ztable = _load_impedance_csv(str(ibc_path))
            medium = _load_dielectric_csv(str(material_path))
            np.testing.assert_array_equal(ztable.freqs_ghz, frequencies)
            np.testing.assert_array_equal(medium.freqs_ghz, frequencies)
            np.testing.assert_array_equal(ztable.values, impedance)
            np.testing.assert_array_equal(medium.eps_values, eps)
            np.testing.assert_array_equal(medium.mu_values, mu)
            self.assertEqual(ztable.sample(.625), 18+1j)
            np.testing.assert_allclose(medium.sample(.625), [2.5-.2j, 1.1-.03j], rtol=1e-15)
            # Both solver dispatchers use this same material library.
            library = MaterialLibrary.from_entries([["1", ibc_path.name]], [["2", material_path.name]], str(root))
            self.assertEqual(library.get_impedance(1, .625), 18+1j)
            np.testing.assert_array_equal(library.dielectric_models[2].freqs_ghz, frequencies)
            for table in (ztable, medium):
                with self.assertRaisesRegex(ValueError, "outside"):
                    table.sample(.01)


class CsvContractTests(unittest.TestCase):
    def test_shipped_material_examples_use_explicit_csv_references(self):
        folder = Path(__file__).resolve().parents[1] / "validation/material_examples"
        paths = list(folder.glob("*.geo"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                _title, _segments, ibcs, dielectrics = parse_geometry(path.read_text())
                MaterialLibrary.from_entries(ibcs, dielectrics, str(folder))

    def test_external_csv_parsing_agrees_across_both_material_readers(self):
        contents = (
            "\ufeff# measured properties\r\n\r\n"
            " frequency_hz , eps_real , eps_imag , mu_real , mu_imag \r\n"
            '"2e9",3,-0.2,1,0\r\n'
            "# another sample\r\n1e9,2,-0.1,1,0\r\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.CSV"
            path.write_text(contents, encoding="utf-8")
            freddy = read_material_table(path)
            ghost = _load_dielectric_csv(str(path))
            np.testing.assert_array_equal(ghost.freqs_ghz, [1., 2.])
            np.testing.assert_array_equal(ghost.freqs_ghz, freddy.freq_ghz)
            np.testing.assert_array_equal(ghost.eps_values, freddy.eps_r)
            np.testing.assert_array_equal(ghost.mu_values, freddy.mu_r)

    def test_invalid_material_formats_are_rejected_by_both_tools(self):
        header = MATERIAL_HEADER + "\n"
        cases = {
            "headerless": "1e9,2,-.1,1,0\n",
            "spaces": "frequency_hz eps_real eps_imag mu_real mu_imag\n1e9 2 -.1 1 0\n",
            "tabs": header + "1e9\t2\t-.1\t1\t0\n",
            "ghz_header": header.replace("frequency_hz", "frequency_ghz") + "1,2,-.1,1,0\n",
            "semicolon": header.replace(",", ";") + "1e9;2;-.1;1;0\n",
            "extra_column": header + "1e9,2,-.1,1,0,7\n",
            "blank_cell": header + "1e9,2,,1,0\n",
            "duplicate": header + "1e9,2,-.1,1,0\n1000000000,3,-.1,1,0\n",
            "nan": header + "1e9,nan,-.1,1,0\n",
            "inf_frequency": header + "inf,2,-.1,1,0\n",
            "zero_frequency": header + "0,2,-.1,1,0\n",
            "gain": header + "1e9,2,.1,1,0\n",
            "empty": "# no data\n",
            "only_header": header,
            "inline_comment": header + "1e9,2,-.1,1,0 # lossless mu\n",
            "bad_quote": header + '1e9,"2,-.1,1,0\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, contents in cases.items():
                path = Path(directory) / (name + ".csv")
                path.write_text(contents, encoding="utf-8")
                for reader in (read_material_table, _load_dielectric_csv):
                    with self.subTest(case=name, reader=reader.__name__):
                        with self.assertRaises(ValueError):
                            reader(path)

    def test_ibc_reader_uses_same_header_and_delimiter_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ibc.csv"
            path.write_text("\ufeff# external impedance\n" + IMPEDANCE_HEADER +
                            "\n2e9,30,-4\n\n1e9,20,3\n", encoding="utf-8")
            table = _load_impedance_csv(str(path))
            np.testing.assert_array_equal(table.freqs_ghz, [1., 2.])
            np.testing.assert_array_equal(table.values, [20+3j, 30-4j])
            for contents in (
                "1e9,20,3\n",
                IMPEDANCE_HEADER.replace("frequency_hz", "frequency_ghz") + "\n1,20,3\n",
                IMPEDANCE_HEADER + "\n1e9 20 3\n",
                IMPEDANCE_HEADER + "\n1e9,-20,3\n",
                IMPEDANCE_HEADER + "\n1e9,20,nan\n",
                IMPEDANCE_HEADER + "\n1e9,20,3\n1e9,20,3\n",
            ):
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        _load_impedance_csv(str(path))

    def test_dense_frequencies_and_complex_values_survive_export_and_reload(self):
        # Twelve significant digits used to collapse these into duplicate rows.
        frequencies = [1., 1.000000000001]
        eps = [2.123456789012345-.123456789012345j] * 2
        mu = [1.123456789012345-.012345678901234j] * 2
        impedance = [12.123456789012345-3.123456789012345j] * 2
        rows = [(f, z.real, z.imag) for f, z in zip(frequencies, impedance)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            material = root / "material.csv"
            write_material_table(material, MaterialTable(frequencies, eps, mu))
            loaded = read_material_table(material)
            ghost = _load_dielectric_csv(str(material))
            np.testing.assert_allclose(loaded.freq_ghz, frequencies, rtol=2e-16)
            self.assertGreater(loaded.freq_ghz[1], loaded.freq_ghz[0])
            np.testing.assert_array_equal(loaded.eps_r, eps)
            np.testing.assert_array_equal(loaded.mu_r, mu)
            np.testing.assert_array_equal(ghost.freqs_ghz, loaded.freq_ghz)
            np.testing.assert_array_equal(ghost.eps_values, eps)
            outputs = [root / name for name in ("single.csv", "batch.csv", "bundle.csv")]
            write_output(outputs[0], rows)
            write_impedance_batch([(outputs[1], rows)])
            write_impedance_bundle(outputs[2], rows)
            for path in outputs:
                self.assertEqual(path.read_text().splitlines()[0], IMPEDANCE_HEADER)
                loaded = _load_impedance_csv(str(path))
                self.assertGreater(loaded.freqs_ghz[1], loaded.freqs_ghz[0])
                np.testing.assert_array_equal(loaded.values, impedance)

    def test_non_csv_paths_are_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "material.txt"
            path.write_text(MATERIAL_HEADER + "\n1e9,2,-.1,1,0\n")
            before = path.read_bytes()
            for reader in (read_material_table, _load_dielectric_csv, _load_impedance_csv):
                with self.subTest(reader=reader.__name__):
                    with self.assertRaisesRegex(ValueError, "[.]csv"):
                        reader(path)
            for writer in (
                lambda: write_material_table(path, MaterialTable([1.], [2-.1j], [1+0j])),
                lambda: write_output(path, [(1., 20., 3.)]),
                lambda: write_impedance_batch([(path, [(1., 20., 3.)])]),
                lambda: write_impedance_bundle(path, [(1., 20., 3.)]),
            ):
                with self.assertRaisesRegex(ValueError, "[.]csv"):
                    writer()
                self.assertEqual(path.read_bytes(), before)

    def test_implicit_material_references_are_rejected_for_all_flags(self):
        for section in ("IBCS_Resistances", "Dielectrics"):
            for flag in (1, 50, 61, 999):
                with self.subTest(section=section, flag=flag):
                    with self.assertRaisesRegex(ValueError, "filename[.]csv"):
                        parse_geometry(f"Title: bad input\n{section}:\n{flag}\n")
                    with self.assertRaises(ValueError):
                        MaterialLibrary.from_entries(
                            [[str(flag)]] if section == "IBCS_Resistances" else [],
                            [[str(flag)]] if section == "Dielectrics" else [], ".")

    def test_tabular_rcs_export_labels_and_converts_frequency_to_hz(self):
        result = {"samples": [{"frequency_ghz": 1.25, "rcs_linear": 2.0}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(export_result_to_dbke_csv(result, str(Path(directory)/"result.csv")))
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertNotIn("frequency_ghz", rows[0])
            self.assertEqual(float(rows[0]["frequency_hz"]), 1250000000.)
            self.assertTrue(math.isfinite(float(rows[0]["dbke"])))


if __name__ == "__main__":
    unittest.main()
