"""Real FREDDY writers -> shared GHOST 2-D/BoR material readers."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Backend"))
FREDDY_ROOT = Path(__file__).resolve().parents[2] / "FREDDY"
if str(FREDDY_ROOT) not in sys.path:
    sys.path.insert(0, str(FREDDY_ROOT))
from ibc.compute import MaterialTable
from ibc.io import write_output, write_material_table
from rcs_solver import MaterialLibrary, _load_impedance_csv, _load_dielectric_csv


class FreddyGhostMaterialTests(unittest.TestCase):
    def test_nominal_exports_are_hz_and_preserve_complex_properties(self):
        frequencies = [.125, 1.125, 2.125]
        impedance = [12-3j, 24+5j, 40+9j]
        eps = [2-.1j, 3-.3j, 5-.5j]
        mu = [1-.02j, 1.2-.04j, 1.6-.08j]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for header in (False, True):
                with self.subTest(header=header):
                    ibc_path, material_path = root/"impedance.csv", root/"dielectric.csv"
                    write_output(ibc_path, list(zip(frequencies, np.real(impedance), np.imag(impedance))), header)
                    write_material_table(material_path, MaterialTable(frequencies, eps, mu), header)
                    for path, width in ((ibc_path, 3), (material_path, 5)):
                        rows = np.loadtxt(path, delimiter=",", skiprows=int(header), ndmin=2)
                        self.assertEqual(rows.shape, (3, width))
                        np.testing.assert_array_equal(rows[:, 0], [125000000, 1125000000, 2125000000])
                        if header:
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


if __name__ == "__main__":
    unittest.main()
