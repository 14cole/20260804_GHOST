from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from ibc.compute import (
    C0,
    MIX_RULE_BRUGGEMAN,
    MIX_RULE_HARMONIC,
    MIX_RULE_LINEAR,
    MIX_RULE_LOG,
    MIX_RULE_LOOYENGA,
    MIX_RULE_MG,
    LoadedLayer,
    MaterialTable,
    ambient_wave_impedance,
    combine_mix,
    compute_angle_metrics,
    compute_stack_impedance,
    make_frequency_sweep,
    mix_anisotropic,
    normalize_wave_polarization,
    project_bounded_fractions,
    validate_fraction_bounds,
)
from ibc.io import (
    IMPEDANCE_HEADER,
    IMPEDANCE_UNCERTAINTY_HEADER,
    MATERIAL_HEADER,
    read_material_table,
    uncertainty_report_path,
    write_impedance_uncertainty_report,
    write_material_table,
    write_output,
)


def _bulk(thickness_m: float, eps: complex, mu: complex = 1 + 0j) -> LoadedLayer:
    table = MaterialTable([0.01, 100.0], [eps, eps], [mu, mu])
    return LoadedLayer(thickness_m, False, 0.0, table, None)


def _sheet(resistance_ohm: float) -> LoadedLayer:
    return LoadedLayer(
        0.0, False, 0.0, None, None, True, resistance_ohm
    )


class MaterialIoTests(unittest.TestCase):
    def test_material_round_trip_uses_solver_schema_and_hz(self) -> None:
        table = MaterialTable(
            [1.0, 2.0],
            [2.5 - 0.1j, 2.7 - 0.2j],
            [1.0 + 0j, 1.1 - 0.01j],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "material.csv"
            write_material_table(path, table)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], MATERIAL_HEADER)
            self.assertTrue(lines[1].startswith("1000000000,"))
            loaded = read_material_table(path)
            self.assertEqual(loaded, table)

    def test_one_row_material_is_valid_for_one_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.csv"
            path.write_text(
                MATERIAL_HEADER + "\n1000000000,2.5,-0.1,1,0\n",
                encoding="utf-8",
            )
            table = read_material_table(path)
            self.assertEqual(table.freq_ghz, [1.0])

    def test_bad_material_files_fail_closed(self) -> None:
        cases = {
            "wrong_header": (
                "frequency_ghz,eps_real,eps_imag,mu_real,mu_imag\n"
                "1,2,-.1,1,0\n"
            ),
            "extra_column": (
                MATERIAL_HEADER + ",junk\n1000000000,2,-.1,1,0,9\n"
            ),
            "duplicate": (
                MATERIAL_HEADER
                + "\n1000000000,2,-.1,1,0\n1000000000,3,-.2,1,0\n"
            ),
            "gain": (
                MATERIAL_HEADER + "\n1000000000,2,.1,1,0\n"
            ),
            "nan": (
                MATERIAL_HEADER + "\n1000000000,nan,-.1,1,0\n"
            ),
            "zero_frequency": (
                MATERIAL_HEADER + "\n0,2,-.1,1,0\n"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, contents in cases.items():
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.csv"
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_material_table(path)

    def test_nominal_impedance_stays_three_column_with_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nominal = Path(tmp) / "coating.csv"
            write_output(nominal, [(1.0, 120.0, 15.0)], True)
            self.assertEqual(
                nominal.read_text(encoding="utf-8").splitlines()[0],
                IMPEDANCE_HEADER,
            )

            sidecar = uncertainty_report_path(nominal)
            write_impedance_uncertainty_report(
                sidecar,
                [(1.0, 120.0, 15.0, 110.0, 130.0, 10.0, 20.0)],
            )
            self.assertEqual(
                sidecar.read_text(encoding="utf-8").splitlines()[0],
                IMPEDANCE_UNCERTAINTY_HEADER,
            )
            self.assertEqual(
                nominal.read_text(encoding="utf-8").splitlines()[0],
                IMPEDANCE_HEADER,
            )

    def test_active_or_nonfinite_exports_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            with self.assertRaises(ValueError):
                write_output(path, [(1.0, -1.0, 0.0)], True)
            with self.assertRaises(ValueError):
                write_output(path, [(1.0, math.nan, 0.0)], True)


class PhysicsTests(unittest.TestCase):
    def test_empty_and_quarter_wave_reference_cases(self) -> None:
        empty = compute_angle_metrics(10.0, 30.0, [], "te")
        self.assertEqual(empty["air_loss_db"], -300.0)
        self.assertAlmostEqual(empty["insertion_loss_db"], 0.0)
        self.assertAlmostEqual(empty["metal_loss_db"], 0.0)

        f_hz = 10e9
        eps = 4 + 0j
        thickness = (C0 / f_hz) / (4.0 * math.sqrt(eps.real))
        zin = compute_stack_impedance(
            f_hz / 1e9, [_bulk(thickness, eps)], "pec"
        )
        self.assertGreater(abs(zin), 1e15)

    def test_oblique_resistive_sheet_matches_closed_form(self) -> None:
        resistance = 188.3651567
        for pol in ("te", "tm"):
            for angle in (0.0, 30.0, 60.0, 80.0):
                z0 = ambient_wave_impedance(angle, pol)
                gamma = -z0 / (2.0 * resistance + z0)
                transmission = 2.0 * resistance / (2.0 * resistance + z0)
                result = compute_angle_metrics(
                    10.0, angle, [_sheet(resistance)], pol
                )
                self.assertAlmostEqual(
                    result["air_loss_db"],
                    20.0 * math.log10(abs(gamma)),
                    places=10,
                )
                self.assertAlmostEqual(
                    result["insertion_loss_db"],
                    20.0 * math.log10(abs(transmission)),
                    places=10,
                )

    def test_exact_grazing_and_nonpositive_frequency_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_angle_metrics(10.0, 90.0, [], "te")
        with self.assertRaises(ValueError):
            make_frequency_sweep(0.0, 1.0, 0.1)

    def test_directional_material_is_principal_axis_only(self) -> None:
        e0, m0 = 2 - 0.1j, 1 - 0.01j
        e90, m90 = 4 - 0.2j, 1.5 - 0.02j
        self.assertEqual(
            mix_anisotropic(e0, m0, e90, m90, 0.0), (e0, m0)
        )
        self.assertEqual(
            mix_anisotropic(e0, m0, e90, m90, 90.0), (e90, m90)
        )
        with self.assertRaises(ValueError):
            mix_anisotropic(e0, m0, e90, m90, 45.0)

    def test_polarization_labels_are_unambiguous_with_legacy_aliases(self) -> None:
        self.assertEqual(normalize_wave_polarization("TE"), "te")
        self.assertEqual(normalize_wave_polarization("TM"), "tm")
        self.assertEqual(normalize_wave_polarization("HH"), "te")
        self.assertEqual(normalize_wave_polarization("VV"), "tm")


class EffectiveMediumTests(unittest.TestCase):
    def _mix(self, values: list[complex], fractions: list[float], rule: str) -> complex:
        table = combine_mix(
            [10.0],
            [[value] for value in values],
            [[1.0 + 0j] for _value in values],
            fractions,
            rule,
        )
        return table.eps_r[0]

    def test_models_preserve_identical_constituents_and_endpoints(self) -> None:
        rules = (
            MIX_RULE_LINEAR,
            MIX_RULE_HARMONIC,
            MIX_RULE_LOG,
            MIX_RULE_LOOYENGA,
            MIX_RULE_MG,
            MIX_RULE_BRUGGEMAN,
        )
        material = 3.2 - 0.18j
        for rule in rules:
            with self.subTest(rule=rule):
                self.assertAlmostEqual(
                    self._mix([material, material], [0.37, 0.63], rule),
                    material,
                )
                self.assertAlmostEqual(
                    self._mix([material, 8.0 - 0.4j], [1.0, 0.0], rule),
                    material,
                )

    def test_wiener_looyenga_and_maxwell_garnett_closed_forms(self) -> None:
        first = 2.0 - 0.1j
        second = 8.0 - 0.4j
        fractions = [0.8, 0.2]

        harmonic = 1.0 / (fractions[0] / first + fractions[1] / second)
        self.assertAlmostEqual(
            self._mix([first, second], fractions, MIX_RULE_HARMONIC), harmonic
        )

        looyenga = (
            fractions[0] * first ** (1.0 / 3.0)
            + fractions[1] * second ** (1.0 / 3.0)
        ) ** 3
        self.assertAlmostEqual(
            self._mix([first, second], fractions, MIX_RULE_LOOYENGA), looyenga
        )

        polarizability = fractions[1] * (second - first) / (second + 2.0 * first)
        maxwell_garnett = first * (1.0 + 2.0 * polarizability) / (
            1.0 - polarizability
        )
        self.assertAlmostEqual(
            self._mix([first, second], fractions, MIX_RULE_MG), maxwell_garnett
        )

    def test_bruggeman_solution_satisfies_its_implicit_equation(self) -> None:
        values = [2.0 - 0.1j, 12.0 - 0.8j, 4.0 - 0.25j]
        fractions = [0.25, 0.45, 0.30]
        effective = self._mix(values, fractions, MIX_RULE_BRUGGEMAN)
        residual = sum(
            fraction * (value - effective) / (value + 2.0 * effective)
            for fraction, value in zip(fractions, values)
        )
        self.assertLess(abs(residual), 1e-9)
        self.assertLessEqual(effective.imag, 0.0)

    def test_branch_models_and_passivity_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive-real"):
            self._mix([2.0 - 0.1j, -3.0 - 0.1j], [0.5, 0.5], MIX_RULE_LOG)
        with self.assertRaisesRegex(ValueError, "gain-sign"):
            self._mix([2.0 + 0.1j, 3.0 - 0.1j], [0.5, 0.5], MIX_RULE_LINEAR)
        with self.assertRaisesRegex(ValueError, "host"):
            self._mix([2.0 - 0.1j, 3.0 - 0.1j], [0.0, 1.0], MIX_RULE_MG)

    def test_bounded_volume_fraction_projection(self) -> None:
        lower = [0.10, 0.20, 0.0]
        upper = [0.50, 0.70, 0.40]
        result = project_bounded_fractions([4.0, -2.0, 1.0], lower, upper)
        self.assertAlmostEqual(sum(result), 1.0, places=10)
        for value, lo, hi in zip(result, lower, upper):
            self.assertGreaterEqual(value, lo - 1e-12)
            self.assertLessEqual(value, hi + 1e-12)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            validate_fraction_bounds([0.6, 0.6], [0.8, 0.8])


if __name__ == "__main__":
    unittest.main()
