"""Reject incomparable measurements and detect resource/field regressions."""
import copy
import unittest
from benchmark_execution import compare, SCHEMA


class ExecutionBenchmarkTests(unittest.TestCase):
    def test_comparison_checks_inputs_resources_and_complex_fields(self):
        case = dict(profile={'factorization': 'compressed'}, workload={'geometry': 'a'},
                    environment={'numpy': 'v'}, median_wall_seconds=10,
                    median_peak_rss_bytes=100, fields={'VV': [[1,2]], 'HH': [[3,4]]})
        old = dict(schema=SCHEMA, cases={'pec/compressed': case})
        new = copy.deepcopy(old)
        self.assertTrue(compare(new, old, .25, .15)['passed'])
        new['cases']['pec/compressed']['median_wall_seconds'] = 13
        self.assertFalse(compare(new, old, .25, .15)['passed'])
        new = copy.deepcopy(old)
        new['cases']['pec/compressed']['fields']['VV'][0][0] = -1
        self.assertFalse(compare(new, old, .25, .15)['passed'])
        for key in ('profile', 'workload', 'environment'):
            new = copy.deepcopy(old)
            new['cases']['pec/compressed'][key] = {}
            with self.assertRaisesRegex(ValueError, 'different ' + key):
                compare(new, old, .25, .15)


if __name__ == '__main__':
    unittest.main()
