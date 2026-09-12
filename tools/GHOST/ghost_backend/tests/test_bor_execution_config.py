"""BOR profiles survive validated JSON and staged driver boundaries."""
from pathlib import Path
import sys,tempfile,json,unittest,subprocess,os
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ghost_backend.hpc.common import configure_driver
from ghost_backend.hpc.bundle import _validate_settings, BundleError
from ghost_backend.bor.options import validate_options


class BorExecutionConfigTests(unittest.TestCase):
    def test_dispatch_rejects_wrong_assembly_or_precision(self):
        from ghost_backend import run_local_bor
        from ghost_backend.bor import dispatch
        backend=Path(__file__).resolve().parents[1]
        snapshot,base=run_local_bor._load_snapshot(str(backend/'geometry/geometries/body.geo'))
        for options,assembly,reported,precision,expected in (
            ({},'streaming','tables','double','assembly'),
            ({'factorization':'compressed'},'auto','tables','double','assembly'),
            ({'factorization':'compressed'},'auto','compressed','single','table_precision'),
        ):
            with mock.patch.object(dispatch,'solve_bor',return_value=dict(assembly=reported,table_precision=precision)):
                with self.assertRaisesRegex(RuntimeError,'did not attest the requested '+expected):
                    dispatch.solve_monostatic_rcs_bor_survey(snapshot,[1.],[0.,90.],geometry_units='meters',
                        material_base_dir=base,workers=1,assembly=assembly,bor_options=options)

    def test_local_and_hpc_staged_profile_loads_in_fresh_process(self):
        backend=Path(__file__).resolve().parents[1]
        options=validate_options(dict(factorization='compressed',angle_batch_size=17,compressed_storage_mib=256))
        with tempfile.TemporaryDirectory() as directory:
            for name in ('run_local_bor.py','run_hpc_bor_monostatic.py'):
                path=configure_driver(backend/name,Path(directory)/name,dict(BOR_EXECUTION_OPTIONS=options))
                payload=json.loads(path.with_suffix('.config.json').read_text())
                self.assertEqual(payload['settings']['BOR_EXECUTION_OPTIONS'],options)
                code="import runpy,json;ns=runpy.run_path({!r},run_name='profile_test');print(json.dumps(ns['BOR_EXECUTION_OPTIONS']))".format(str(path))
                env=dict(os.environ,PYTHONPATH=str(backend.parent))
                result=subprocess.run([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                      universal_newlines=True,env=env,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(json.loads(result.stdout),options)

    def test_portable_bundle_accepts_profile_and_rejects_conflicts(self):
        settings=_validate_settings('bor',dict(BOR_EXECUTION_OPTIONS=dict(factorization='compressed',angle_batch_size=17)))
        self.assertEqual(settings['BOR_EXECUTION_OPTIONS']['angle_batch_size'],17)
        for value in ({'angle_batch_size':0},{'unknown':1}):
            with self.assertRaises(BundleError):_validate_settings('bor',dict(BOR_EXECUTION_OPTIONS=value))
        with self.assertRaises(BundleError):
            _validate_settings('bor',dict(BOR_EXECUTION_OPTIONS=dict(factorization='compressed'),TABLE_PRECISION='single'))

    def test_public_compressed_preview_and_survey_keep_profile(self):
        from ghost_backend import run_local_bor
        from ghost_backend.bor.dispatch import estimate_bor_resources, solve_monostatic_rcs_bor_survey
        backend=Path(__file__).resolve().parents[1]
        snapshot,base=run_local_bor._load_snapshot(str(backend/'geometry/geometries/body.geo'))
        options=dict(factorization='compressed',angle_batch_size=1,compressed_storage_mib=128)
        preview=estimate_bor_resources(snapshot,1.,[0.,90.],geometry_units='meters',
            material_base_dir=base,workers=1,mesh_certification=False,bor_options=options)
        self.assertEqual(preview['assembly_estimate'],'compressed')
        self.assertEqual(preview['angle_batch_size'],1)
        self.assertGreater(preview['estimated_peak_gb'],0.)
        result=solve_monostatic_rcs_bor_survey(snapshot,[1.],[0.,90.],geometry_units='meters',
            material_base_dir=base,workers=1,bor_options=options)
        self.assertEqual(result['metadata']['bor_execution_options']['angle_batch_size'],1)
        self.assertEqual(result['metadata']['assembly_requested'],'compressed')
        self.assertEqual(result['metadata']['per_frequency'][0]['assembly'],'compressed')


if __name__=='__main__':unittest.main()
