"""BOR recipes preserve physics across local and HPC coordinate conventions."""
from pathlib import Path
import copy, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.runs.setup import validate_setup, save_setup, read_setup
from ghost_backend.runs.bor_setup import driver_settings, resource_summary
from ghost_backend.runs.config import configuration_payload, driver_contract
from ghost_backend.assembly.fields import radar_grid_aspects


def recipe():
    return dict(schema='grim.bor-run-setup', version=1, frequencies_ghz=[1., 2.],
        aspects_deg=[0., 31., 90., 180.], units='meters', mesh_certification=False,
        accuracy='tight', cfie_alpha=.65, bor_options=dict(factorization='compressed',
        angle_batch_size=17, compression_tile=24, tile_cache_mib=3))


class BorRunSetupTests(unittest.TestCase):
    def test_recipe_round_trip_and_driver_coordinate_mapping(self):
        value=validate_setup(recipe())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sphere.run.json'
            save_setup(path,value)
            self.assertEqual(read_setup(path),value)
        settings=driver_settings(value)
        self.assertEqual(settings['AZIMUTHS_DEG'],[0.])
        self.assertEqual(settings['ELEVATIONS_DEG'],[90.,59.,0.,-90.])
        np.testing.assert_allclose(radar_grid_aspects(settings['AZIMUTHS_DEG'],settings['ELEVATIONS_DEG'],
            settings['BODY_AXIS_AZ_DEG'],settings['BODY_AXIS_EL_DEG']),value['aspects_deg'],atol=1e-10,rtol=0)
        backend=Path(__file__).resolve().parents[1]
        for name in ('run_local_bor.py','run_hpc_bor_monostatic.py'):
            kind,keys=driver_contract(backend/name)
            result=configuration_payload(kind,{},keys,run_setup=value)
            self.assertEqual(result['settings']['BOR_EXECUTION_OPTIONS'],value['bor_options'])
            self.assertEqual(result['settings']['CFIE_ALPHA'],.65)
            self.assertEqual(result['settings']['BODY_AXIS_EL_DEG'],90.)
            with self.assertRaisesRegex(ValueError,'conflicts'):
                configuration_payload(kind,{'CFIE_ALPHA':.5},keys,run_setup=value)

    def test_radar_recipe_preserves_attitude_and_rejects_mismatch(self):
        value=recipe()
        grid=dict(azimuths_deg=[0.,35.],elevations_deg=[-10.,10.],axis_az_deg=15.,axis_el_deg=-4.,roll_deg=22.5)
        value['radar_grid']=grid
        value['aspects_deg']=radar_grid_aspects(grid['azimuths_deg'],grid['elevations_deg'],15.,-4.).tolist()
        settings=driver_settings(value)
        self.assertEqual(settings['ELEVATIONS_DEG'],[-10.,10.])
        self.assertEqual(settings['BODY_ROLL_DEG'],22.5)
        value['aspects_deg'][0]+=1.
        with self.assertRaisesRegex(ValueError,'do not match'):validate_setup(value)

    def test_invalid_recipe_cannot_change_solver_or_numeric_contract(self):
        for changes in ({'aspects_deg':[-1.]},{'cfie_alpha':0.},{'units':'feet'},
                        {'frequencies_ghz':[float('nan')]},{'bor_options':{'tile_cache_mib':-1}},
                        {'scattering':'bistatic'}):
            with self.assertRaises(ValueError):validate_setup(dict(recipe(),**changes))
        with self.assertRaisesRegex(ValueError,'2-D'):
            configuration_payload('2d',{},[],run_setup=recipe())

    def test_preflight_is_geometry_only_and_cancellable(self):
        from unittest import mock
        from ghost_backend import run_local_bor
        backend=Path(__file__).resolve().parents[1]
        snapshot,base=run_local_bor._load_snapshot(str(backend/'geometry/geometries/body.geo'))
        with mock.patch('ghost_backend.bor.solver.BorPecSolver.prepare_operators',side_effect=AssertionError('assembly')):
            summary=resource_summary(snapshot,base,recipe())
        self.assertIn('estimated peak',summary)
        self.assertIn('17 aspects per batch',summary)
        def cancel():raise InterruptedError('canceled')
        with self.assertRaises(InterruptedError):resource_summary(snapshot,base,recipe(),cancel)


if __name__=='__main__':unittest.main()
