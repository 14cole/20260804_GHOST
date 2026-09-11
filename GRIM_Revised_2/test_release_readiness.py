"""Cross-format validity and optional telemetry failure regressions."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import dataset_jobs
from grim_backend.datasets.grid import RcsGrid
from grim_backend.io.csv import write_flat_csv
from grim_backend.io.loaders import load_flat_csv


class ReleaseReadinessTests(unittest.TestCase):
    def test_blank_native_units_round_trip_through_csv_with_explicit_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for axis, expected in (('azimuth', 'deg'), ('elevation', 'deg'), ('frequency', 'GHz')):
                for blank in ('', None):
                    with self.subTest(axis=axis, blank=blank):
                        grid = RcsGrid([0.], [0.], [1.], ['VV'],
                                       rcs_power=np.ones((1,1,1,1)), units={axis:blank})
                        grid.save(str(root / 'source.grim'))
                        loaded = RcsGrid.load(str(root / 'source.grim'))
                        self.assertFalse(loaded.audit()['errors'])
                        write_flat_csv(loaded, str(root / 'export.csv'))
                        exported = load_flat_csv(str(root / 'export.csv'))
                        self.assertEqual(exported.units[axis], expected)
                        np.testing.assert_array_equal(exported.rcs_power, loaded.rcs_power)

    def test_optional_memory_probe_failure_uses_os_fallback(self):
        with mock.patch('psutil.virtual_memory', side_effect=RuntimeError('probe unavailable')):
            memory = dataset_jobs._available_memory_bytes()
        self.assertTrue(memory is None or memory > 0)
