"""Headless request -> configured driver -> worker -> attested material results.

Runs SLURM worker entry points locally with SUBMIT=False and Qt imports blocked.
This tests portability contracts, not a live cluster or material-model accuracy.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
try:
    import psutil
except ImportError:
    psutil = None

BACKEND = Path(__file__).resolve().parents[1] / "Backend"
sys.path.insert(0, str(BACKEND))
import ghost_backend.hpc.bundle as hpc_bundle
import ghost_backend.hpc.common as hpc_common
from ghost_backend.runs.quality import accuracy_target_policy


class HeadlessSolverOptionsTests(unittest.TestCase):
    def test_freddy_and_assembly_apis_need_no_gui(self):
        script = """
import sys
class NoGui:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PySide6', 'PyQt5', 'PyQt6'):
            raise RuntimeError('GUI import in headless calculation: ' + fullname)
sys.meta_path.insert(0, NoGui())
from pathlib import Path
from ibc.compute import LoadedLayer, MaterialTable, compute_stack_impedance_many
from ibc.io import write_output
from ibc.ghost_coating import assess_scalar_coating
from ghost_backend.assembly.inspector import interference_metrics, ContributionInspector
from ghost_backend.assembly.workflow import prepare_feature_assembly, execute_feature_assembly
from feature_family_validation import study_template
layer = LoadedLayer(.000762, False, 0., MaterialTable([1.,18.], [4-.1j]*2, [1.]*2), None)
frequencies = [1., 9.5, 18.]
zs = compute_stack_impedance_many(frequencies, [layer], 'pec')
write_output(Path(sys.argv[1]), [(f,z.real,z.imag) for f,z in zip(frequencies,zs)])
report = assess_scalar_coating(frequencies, [layer])
assert not report['finite_body_accuracy_certified']
assert interference_metrics(1+0j, [-1+0j])['sigma_total'] == 0
assert study_template()['cases']
"""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "coating.csv"
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(
                (str(BACKEND), str(BACKEND.parents[1] / "FREDDY")))}
            result = subprocess.run([sys.executable, "-c", script, str(output)],
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.is_file())

    def _driver_process(self, driver, arguments, env, log):
        # File output avoids inherited pipe handles hanging timeout cleanup.
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                [sys.executable, str(driver), *arguments], env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=(os.name != "nt"),
            )
            try:
                code = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                children = psutil.Process(process.pid).children(recursive=True)
                for child in reversed(children):
                    child.kill()
                process.kill()
                psutil.wait_procs(children, timeout=5)
                process.wait(timeout=10)
                self.fail("Headless test timed out: " + log.read_text(encoding="utf-8"))
        self.assertEqual(code, 0, log.read_text(encoding="utf-8"))

    def test_invalid_options_fail_before_bundle_creation(self):
        for solver, settings in (
            ("2d", {"ACCURACY_TARGET": "typo"}),
            ("bor", {"ACCURACY_TARGET": []}),
            ("2d", {"LU_PRECISION": "single"}),
            ("2d", {"LU_PRECISION": None}),
            ("bor", {"LU_PRECISION": "mixed"}),
        ):
            with self.subTest(solver=solver, settings=settings):
                with self.assertRaises(hpc_bundle.BundleError):
                    hpc_bundle._validate_settings(solver, settings)

    def _run_material_case(self, solver, geometry, *, units="inches", certified=True, execution=None,
                           environment_profile=False):
        if psutil is None:
            self.skipTest("psutil is required to clean up timed-out worker trees")
        with tempfile.TemporaryDirectory(prefix="ghost-headless-") as temporary:
            root = Path(temporary)
            if solver == "bor":
                # A small 8-panel meridian keeps this a handoff test, not a
                # large-body benchmark; retain the actual FREDDY stack CSV.
                source = root / "source"
                source.mkdir()
                shutil.copy2(geometry.parent / "example_coating_30mil.csv", source)
                geometry = source / "coated_sphere.geo"
                angles = np.linspace(0, np.pi, 9)
                points = np.column_stack((.24 * np.sin(angles), .24 * np.cos(angles)))
                geometry.write_text(
                    "Title: Headless coated sphere in inches\nSegment: sphere 2\n"
                    "properties: 2 0 10 0 0\n" +
                    "".join(f"{a[0]} {a[1]} {b[0]} {b[1]}\n" for a, b in zip(points[:-1], points[1:])) +
                    "IBCS_Resistances:\n10 example_coating_30mil.csv\nDielectrics:\n",
                    encoding="utf-8",
                )
            settings = {
                "FREQUENCIES_GHZ": [1.0], "AZIMUTHS_DEG": [0.0, 90.0],
                "GEOMETRY_UNITS": units, "N_NODES": 1, "N_JOBS": 1,
                "MAX_WORKERS_PER_NODE": 1, "BLAS_THREADS_PER_WORKER": 1,
                "MESH_CERTIFICATION": certified, "ACCURACY_TARGET": "tight",
            }
            if solver == "2d":
                settings["LU_PRECISION"] = "mixed"
            else:
                settings.update(WORKERS_PER_UNIT=1, ELEVATIONS_DEG=[0.0], ASSEMBLY="streaming")
            if execution is not None:
                settings.update(EXECUTION_OPTIONS=execution, LU_PRECISION='double',
                                SOLVER_METHOD='experimental_cpu')
                if environment_profile:
                    settings.pop('EXECUTION_OPTIONS')
                    settings['ASSEMBLY_THREADS'] = execution['assembly_threads']
            bundle = root / "request"
            hpc_bundle.create_portable_bundle(
                bundle, solver=solver, settings=settings,
                geometries=[{"role": "FRD" if solver == "2d" else "BOR",
                             "path": str(geometry)}],
            )
            request = hpc_bundle.verify_portable_bundle(bundle)
            staged_geometry = bundle / request["geometries"][0]["path"]
            # Configure the real driver directly: Linux-only stage is covered by
            # the bundle tests; this numerical test also runs on Windows.
            settings.update(SUBMIT=False, OUTPUT_DIR=str(root / "runs"))
            if solver == "2d":
                settings.update(FRD_DIR=str(staged_geometry.parent),
                                OPN_DIR=str(root / "empty_opn"))
                (root / "empty_opn").mkdir()
                canonical = "run_hpc_monostatic.py"
            else:
                settings["GEOMETRY_DIRS"] = [str(staged_geometry.parent)]
                canonical = "run_hpc_bor_monostatic.py"
            driver = hpc_common.configure_driver(BACKEND / canonical, root / "driver.py", settings)
            # sitecustomize runs in both the driver and spawned child processes.
            (root / "sitecustomize.py").write_text(
                "import sys\n"
                "import faulthandler\n"
                "faulthandler.enable()\n"
                "class NoGui:\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname.split('.')[0] in ('PySide6', 'PyQt5', 'PyQt6'):\n"
                "            raise RuntimeError('GUI import in headless run: ' + fullname)\n"
                "sys.meta_path.insert(0, NoGui())\n", encoding="utf-8",
            )
            env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root), str(BACKEND))),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            if environment_profile:
                env.update(GHOST_CPU_FACTORIZATION=execution['factorization'],
                           GHOST_COMPRESSED_STORAGE_MIB=str(execution['compressed_storage_mib']),
                           GHOST_CPU_RHS_COMPRESSION=execution['rhs_compression'],
                           GHOST_CPU_ANGLE_BATCH_SIZE=str(execution['angle_batch_size']))
            self._driver_process(driver, [], env, root / "prepare.log")
            run_dir = hpc_common.latest_run_dir(root / "runs")
            manifest = json.loads((run_dir / "manifest.json").read_text())
            config = manifest["solver_config"]
            self.assertEqual(config["accuracy_target"], "tight")
            self.assertEqual(config["mesh_convergence_policy"], accuracy_target_policy("tight"))
            if solver == "2d":
                self.assertEqual(config["lu_precision"], "double" if execution else "mixed")
            if execution is not None:
                self.assertEqual(config['execution_options'], execution)
                env['GHOST_CPU_FACTORIZATION'] = 'invalid-host-setting'
                env['GHOST_COMPRESSED_STORAGE_MIB'] = 'not-a-number'
            self.assertTrue(list(run_dir.glob("submit_job*.slurm")))
            frozen = Path(manifest["units"][0]["geometry"])
            self.assertEqual(frozen.read_bytes(), staged_geometry.read_bytes())
            for sidecar in request["geometries"][0]["sidecars"]:
                # Sidecar records name bundle paths; each immutable run keeps
                # a private copy next to its geometry.
                source = bundle / sidecar
                self.assertEqual((frozen.parent / source.name).read_bytes(), source.read_bytes())
            self._driver_process(driver, ["--worker", str(run_dir), "0", "0"],
                                 env, root / "worker.log")
            status = hpc_common.run_status(run_dir)
            self.assertTrue(status["complete"], status)
            self.assertTrue(status["attestation_verified"], status)
            unit_outputs = run_dir / "results" / "by_frequency" if solver == "bor" else run_dir / "results"
            output = next(unit_outputs.rglob("*.grim"))
            with np.load(output, allow_pickle=False) as archive:
                audit = json.loads(str(archive["solver_metadata_json"].reshape(()).item()))
            metadata_text = json.dumps(audit)
            self.assertIn('"runtime_profile"', metadata_text)
            if solver == "2d" and execution is None:
                self.assertIn('"mixed_factorization"', metadata_text)
            if execution is not None:
                self.assertIn('"compressed_experimental_cpu"', metadata_text)
                self.assertIn(json.dumps(execution, sort_keys=True), json.dumps(audit, sort_keys=True))
            if certified:
                metadata = audit["metadata"]
                if solver == "2d":
                    for channel in ("VV", "HH"):
                        self.assertEqual(metadata["channel_metadata"][channel]
                                         ["mesh_convergence"]["limits"]["complex_max_normalized"], .01)
                else:
                    self.assertEqual(metadata["mesh_convergence"]["policy"],
                                     accuracy_target_policy("tight"))

    def test_freddy_coating_2d_mixed_tight_worker(self):
        self._run_material_case("2d", BACKEND.parent / "geometry_tests" /
                                "pec_backed_ibc/example/2d_outer_envelope.geo")

    def test_compressed_profile_reaches_fresh_worker_despite_host_environment(self):
        from ghost_backend.execution.options import validate_options
        self._run_material_case('2d', BACKEND.parent / 'geometry_tests' /
            'pec_backed_ibc/example/2d_outer_envelope.geo', certified=False,
            execution=validate_options(dict(factorization='compressed', compressed_storage_mib=64,
                ram_budget_gib=2, assembly_threads='auto', blas_threads=1,
                rhs_compression='on', angle_batch_size=17)))

    def test_manifest_preserves_environment_selection_without_explicit_driver_profile(self):
        from ghost_backend.execution.options import validate_options
        self._run_material_case('2d', BACKEND.parent / 'geometry_tests' /
            'pec_backed_ibc/example/2d_outer_envelope.geo', certified=False, environment_profile=True,
            execution=validate_options(dict(factorization='compressed', compressed_storage_mib=64,
                assembly_threads='auto', blas_threads=1, rhs_compression='on', angle_batch_size=17)))

    def test_freddy_coating_bor_tight_worker(self):
        self._run_material_case("bor", BACKEND.parent / "geometry_tests" /
                                "pec_backed_ibc/example/bor_outer_envelope.geo")

    def test_thin_dielectric_2d_mixed_survey_worker(self):
        self._run_material_case("2d", BACKEND.parent / "geometry_tests" /
                                "thin_dielectric_sheet/thin_strip.geo",
                                units="meters", certified=False)


if __name__ == "__main__":
    unittest.main()
