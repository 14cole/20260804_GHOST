"""Real configured local/HPC workers, export and resume for CPU streaming."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parent))
import numpy as np
import ghost_backend.runs.config as driver_config
import ghost_backend.geometry.io as geometry_io
import ghost_backend.hpc.bundle as hpc_bundle
import ghost_backend.hpc.common as hpc_common
import ghost_backend.execution.provenance as provenance
from ghost_backend.runs.setup import DEFAULT_QUALITY, validate_setup
from test_experimental_cpu import fixture


class ExperimentalHeadless(unittest.TestCase):
    def run_process(self, script, args, root):
        # Add the package parent, never its contents: ghost_backend/io shadows
        # the standard-library io module during Python 3.6 process startup.
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(BACKEND.parent))),
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', OMP_NUM_THREADS='2')
        output = subprocess.run([sys.executable, str(script)] + list(args), cwd=str(root),
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=180)
        self.assertEqual(output.returncode, 0, output.stdout)
        return output.stdout

    def test_saved_recipe_and_conflicts(self):
        recipe = dict(schema='grim.2d-run-setup', version=1, frequencies_ghz=[.6],
            angles_deg=[0., 180.], units='meters', mesh_certification=True,
            accuracy='standard', lu_precision='double', scattering='monostatic',
            observation_angles_deg=[], quality=dict(DEFAULT_QUALITY))
        self.assertEqual(validate_setup(recipe)['solver_method'], 'direct')
        recipe['solver_method'] = 'experimental_cpu'
        settings = driver_config.settings_from_run_setup(recipe, '2d')
        self.assertEqual(settings['SOLVER_METHOD'], 'experimental_cpu')
        hpc_bundle._validate_settings('2d', settings)
        for changes in ({'lu_precision': 'mixed'}, {'scattering': 'bistatic', 'observation_angles_deg': [0.]}, {'solver_method': 'typo'}):
            with self.assertRaises(ValueError):
                validate_setup(dict(recipe, **changes))
        with self.assertRaises(hpc_bundle.BundleError):
            hpc_bundle._validate_settings('2d', {'SOLVER_METHOD': 'experimental_cpu', 'LU_PRECISION': 'mixed'})
        with self.assertRaises(hpc_bundle.BundleError):
            hpc_bundle._validate_settings('bor', {'SOLVER_METHOD': 'experimental_cpu'})

    def _run_driver(self, cluster):
        with tempfile.TemporaryDirectory(prefix='ghost-experimental-') as temporary:
            root = Path(temporary)
            geometries = root / "geometry" / "geometries"
            geometries.mkdir(parents=True)
            for kind in ('pec', 'lossy'):
                geometry_io.save_snapshot_geo(fixture(kind, 32), str(geometries / (kind + '.geo')))
            (root / 'sitecustomize.py').write_text(
                "import sys\nclass NoOptionalBackends:\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname.split('.')[0] in ('PySide2', 'PySide6', 'cupy', 'cupyx', 'torch', 'jax'):\n"
                "            raise RuntimeError('Unexpected GUI/GPU import: ' + fullname)\n"
                "sys.meta_path.insert(0, NoOptionalBackends())\n", encoding='utf-8')
            settings = dict(FRD_DIR=str(geometries), OPN_DIR=str(root/'empty'),
                OUTPUT_DIR=str(root/'runs'), FREQUENCIES_GHZ=[.6],
                AZIMUTHS_DEG=np.linspace(0, 180, 519).tolist(), GEOMETRY_UNITS='meters',
                SOLVER_METHOD='experimental_cpu', LU_PRECISION='double',
                MESH_CERTIFICATION=True, BLAS_THREADS_PER_WORKER=2, ASSEMBLY_THREADS=2)
            if cluster:
                bundle_settings = {k: v for k, v in settings.items() if k not in ('FRD_DIR', 'OPN_DIR', 'OUTPUT_DIR')}
                hpc_bundle.create_portable_bundle(root/'bundle', solver='2d', settings=bundle_settings,
                    geometries=[dict(role='FRD', path=str(geometries/'lossy.geo'))])
                request = hpc_bundle.verify_portable_bundle(root/'bundle')
                self.assertEqual(request['settings']['SOLVER_METHOD'], 'experimental_cpu')
                settings.update(SUBMIT=False, N_NODES=1, N_JOBS=1, MAX_WORKERS_PER_NODE=1)
                canonical = BACKEND/'run_hpc_monostatic.py'
            else:
                settings.update(WORKERS=1)
                canonical = BACKEND/'run_local_monostatic.py'
            driver = hpc_common.configure_driver(canonical, root/'driver.py', settings)
            self.run_process(driver, [], root)
            run_dir = hpc_common.latest_run_dir(root/'runs')
            if cluster:
                self.run_process(driver, ['--worker', str(run_dir), '0', '0'], root)
            manifest = json.loads((run_dir/'manifest.json').read_text())
            self.assertEqual(manifest['solver_config']['solver_method'], 'experimental_cpu')
            outputs = list((run_dir/'results').rglob('*.grim'))
            self.assertEqual(len(outputs), 2)
            before = {str(p): p.read_bytes() for p in outputs}
            for path in outputs:
                with np.load(str(path), allow_pickle=False) as archive:
                    metadata = json.loads(str(archive['solver_metadata_json'].reshape(()).item()))['metadata']
                    self.assertEqual(archive['polarizations'].astype(str).tolist(), ['VV', 'HH'])
                self.assertEqual(metadata['solver_method_requested'], 'experimental_cpu')
                self.assertTrue(metadata['mesh_convergence_certified'])
                self.assertEqual(len(metadata['experimental_cpu']['systems']), 4)
                self.assertTrue(all(s['rhs_batches'] == 3 for s in metadata['experimental_cpu']['systems']))
                if os.environ.get('GHOST_CPU_FACTORIZATION') == 'compressed':
                    self.assertEqual(metadata['solver_method'], 'compressed_experimental_cpu')
                    self.assertTrue(all('compressed' in s for s in metadata['experimental_cpu']['systems']))
                attestation = provenance.read_embedded_attestation(str(path))
                self.assertEqual(attestation['solver_config_sha256'], provenance.stable_json_fingerprint(manifest['solver_config']))
            changed = copy.deepcopy(manifest)
            changed['solver_config']['solver_method'] = 'direct'
            self.assertNotEqual(provenance.manifest_solve_spec_fingerprint(manifest), provenance.manifest_solve_spec_fingerprint(changed))
            if cluster:
                self.run_process(driver, ['--worker', str(run_dir), '0', '0'], root)
                self.assertTrue(hpc_common.run_status(run_dir)['attestation_verified'])
                for path in outputs:
                    self.assertEqual(path.read_bytes(), before[str(path)])

    def test_local_configured_worker_and_export(self):
        self._run_driver(False)

    def test_hpc_bundle_worker_export_and_resume(self):
        self._run_driver(True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
