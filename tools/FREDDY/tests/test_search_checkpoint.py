from array import array
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from ibc import search_checkpoint as storage
from ibc.compute import LayerConfig, UncertaintyConfig
from ibc.design_search import InverseSearchRequest, run_inverse_search
from ibc.inverse_grid import DesignGrid


class SearchCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'search.fsearch'
        self.state = dict(identity='a' * 64, score_rows=array('d', [-1.] * 10),
                          next_index=2, total=5, plots_complete=True)

    def test_portable_roundtrip_from_new_interpreter(self):
        storage.save_checkpoint(self.path, self.state)
        loaded = storage.load_checkpoint(self.path)
        self.assertEqual(loaded['score_rows'], self.state['score_rows'])
        self.assertFalse(loaded['plots_complete'])
        script = ('import sys; from ibc.search_checkpoint import load_checkpoint; '
                  'c=load_checkpoint(sys.argv[1]); assert c["next_index"]==2; '
                  'assert len(c["score_rows"])==10')
        result = subprocess.run([sys.executable, '-c', script, str(self.path)],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_publication_preserves_previous_checkpoint(self):
        storage.save_checkpoint(self.path, self.state)
        original = self.path.read_bytes()
        with mock.patch.object(storage.os, 'replace', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                storage.save_checkpoint(self.path, self.state)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_changed_engine_and_corrupted_scores_are_rejected(self):
        storage.save_checkpoint(self.path, self.state)
        with mock.patch.object(storage, 'engine_identity', return_value='b' * 64):
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                storage.load_checkpoint(self.path)
        with zipfile.ZipFile(self.path) as archive:
            metadata = archive.read('checkpoint.json')
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr('checkpoint.json', metadata)
            archive.writestr('scores.f64le', bytes(80))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            storage.load_checkpoint(self.path)

    def test_invalid_counts_and_nonfinite_scores_never_publish(self):
        for change in ({'next_index': True}, {'next_index': 3},
                       {'score_rows': array('d', [float('nan')] * 10)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                storage.save_checkpoint(self.path, dict(self.state, **change))
        self.assertFalse(self.path.exists())

    def test_disk_resume_scores_only_remaining_combinations(self):
        layers = [LayerConfig(0., False, '', '', 0., is_sheet=True, sheet_resistance=100.,
                              inv_rs_min=100., inv_rs_max=500., inv_rs_accuracy=100.)]
        request = InverseSearchRequest(layers, [1.], [0.], 'te', UncertaintyConfig(False, 0., 0., 0.),
                                       'Worst case', None, DesignGrid(layers), 0, 3,
                                       '1 GHz', 0., 0., False)
        calls = []
        def score(_freq, _angle, layers, *_args):
            calls.append(layers[0].sheet_resistance)
            return (-float(len(calls)),) * 5
        def plots(*_args, **_kwargs):
            return {'metal_loss_db': [-1.]}
        run_inverse_search(request, stop_requested=lambda: len(calls) >= 2, progress=lambda *_: None,
                           score_candidate=score, compute_metrics=plots, checkpoint_interval=0,
                           checkpoint_callback=lambda state: storage.save_checkpoint(self.path, state))
        recovered = storage.load_checkpoint(self.path)
        self.assertEqual(recovered['next_index'], 2)
        resumed = replace(request, checkpoint=recovered)
        _, completed = run_inverse_search(resumed, stop_requested=lambda: False,
                                           progress=lambda *_: None, score_candidate=score,
                                           compute_metrics=plots)
        self.assertEqual(calls, [100., 200., 300., 400., 500.])
        self.assertEqual(completed['next_index'], 5)
        for changed in (replace(resumed, target_freqs=[2.]), replace(resumed, skiprows=1)):
            with self.assertRaisesRegex(ValueError, 'Inputs or material files changed'):
                run_inverse_search(changed, stop_requested=lambda: False,
                                   progress=lambda *_: None, score_candidate=score)


if __name__ == '__main__':
    unittest.main()
