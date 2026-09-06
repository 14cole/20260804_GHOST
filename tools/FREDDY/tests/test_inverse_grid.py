import unittest
from ibc.compute import LayerConfig
from ibc.inverse_grid import DesignGrid, NumericChoices


class DesignGridTests(unittest.TestCase):
    def test_decimal_steps_do_not_drop_an_aligned_endpoint(self):
        values = NumericChoices.build(.1, .3, .1, 'Layer 1')
        self.assertEqual([values.value(i) for i in range(values.count)], [.1, .2, .3])

    def test_step_starts_at_minimum_and_does_not_add_an_off_step_maximum(self):
        values = NumericChoices.build(.105, .132, .01, 'Layer 1')
        self.assertEqual([values.value(i) for i in range(values.count)], [.105, .115, .125])

    def test_fixed_equal_bounds_and_missing_step(self):
        fixed = LayerConfig(.125, False, '', '', 0.)
        constrained = LayerConfig(0., False, '', '', 0., is_sheet=True, sheet_resistance=100.,
                                  inv_rs_min=200., inv_rs_max=200.)
        grid = DesignGrid([fixed, constrained])
        self.assertEqual(grid.total, 1)
        self.assertEqual(grid.design(0), ([.125, 0.], [0., 200.]))
        constrained.inv_rs_max = 300.
        with self.assertRaisesRegex(ValueError, 'Layer 2: set a step'):
            DesignGrid([fixed, constrained])

    def test_large_product_is_counted_and_indexed_without_allocating_proposals(self):
        layers = [LayerConfig(.1, False, '', '', 0., inv_t_min_in=.1,
                              inv_t_max_in=1000., inv_t_accuracy_in=.1) for _ in range(4)]
        grid = DesignGrid(layers)
        self.assertEqual(grid.total, 10_000 ** 4)
        self.assertEqual(grid.design(0)[0], [.1] * 4)
        self.assertEqual(grid.design(grid.total - 1)[0], [1000.] * 4)
        self.assertEqual(grid.design(1)[0], [.1, .1, .1, .2])

    def test_invalid_and_indistinguishable_values_are_rejected(self):
        for bounds in [(0, 2, 1), (1, float('inf'), 1), (2, 1, 1),
                       (1, 2, float('nan')), (1, 2, 0), (1, 2, 1e-20)]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                NumericChoices.build(*bounds, 'Layer 1')
