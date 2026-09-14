"""Real configured SLURM worker entry point, exercised locally without submission."""
from pathlib import Path
import json,os,subprocess,sys
import numpy as np
import pytest

BACKEND=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BACKEND.parent)]
from ghost_backend.hpc.common import configure_driver,latest_run_dir,run_status


@pytest.mark.parametrize('density,legacy_runtime_api', [(24, False), (24, True),
                                                      (-1400, False), (-2800, False)])
def test_default_automatic_request_reaches_fresh_headless_worker(tmp_path, density, legacy_runtime_api):
    geometry=tmp_path/'geometry';geometry.mkdir()
    empty=tmp_path/'empty';empty.mkdir()
    (geometry/'rectangle.geo').write_text(
        f'Title: polygon auto test\nSegment: rectangle 2\nproperties: 2 {density} 0 0 0\n'
        '-.02 -.01 -.02 .01\n-.02 .01 .02 .01\n.02 .01 .02 -.01\n.02 -.01 -.02 -.01\n'
        'IBCS_Resistances:\nDielectrics:\n')
    settings=dict(SOLVE_PRESET='auto',ADVANCED_OVERRIDES=dict(assembly_threads=1,blas_threads=1),
        FREQUENCIES_GHZ=[1.,2.],AZIMUTHS_DEG=[0.,45.,90.],GEOMETRY_UNITS='meters',
        N_NODES=1,N_JOBS=1,MAX_WORKERS_PER_NODE=1,MESH_CERTIFICATION=True,
        OUTPUT_DIR=str(tmp_path/'runs'),SUBMIT=False,FRD_DIR=str(geometry),OPN_DIR=str(empty))
    driver=configure_driver(BACKEND/'run_hpc_monostatic.py',tmp_path/'driver.py',settings)
    (tmp_path/'sitecustomize.py').write_text(
        'import sys\nclass NoGui:\n'
        '    def find_spec(self,fullname,path=None,target=None):\n'
        '        if fullname.split(".")[0] in ("PySide6","PySide2","PyQt5","PyQt6"):\n'
        '            raise RuntimeError("GUI imported into HPC worker")\n'
        'sys.meta_path.insert(0,NoGui())\n' +
        # Scope the legacy API to the planner; current SciPy itself needs prod.
        ('import math\nfrom types import SimpleNamespace\n'
         'import ghost_backend.runs.batch as batch\n'
         'batch.math = SimpleNamespace(isfinite=math.isfinite)\n'
         'from pathlib import Path\n'
         'import ghost_backend.execution.timing_history as history\n'
         'class LegacyPath(type(Path())):\n'
         '    def unlink(self):\n'
         '        return super().unlink()\n'
         'history.Path = LegacyPath\n' if legacy_runtime_api else ''))
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(tmp_path),str(BACKEND.parent))),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             GHOST_TIMING_CACHE_DIR=str(tmp_path/'timings'))
    def run(arguments):
        p=subprocess.run([sys.executable,str(driver),*arguments],env=env,cwd=tmp_path,
                         capture_output=True,text=True,timeout=120)
        assert p.returncode==0,p.stdout+p.stderr
        return p.stdout
    run([])
    directory=latest_run_dir(tmp_path/'runs')
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['solver_config']['execution_options']['factorization']=='adaptive'
    assert manifest['solver_config']['solver_method']=='auto'
    # The saved profile remains authoritative in a fresh compute process.
    env['GHOST_CPU_FACTORIZATION']='invalid-launch-value'
    output=run(['--worker',str(directory),'0','0'])
    assert 'Auto batch:' in output
    assert run_status(directory)['attestation_verified']
    results=list((directory/'results').rglob('*.grim'))
    assert len(results)==2
    degrees=[]
    for path in results:
        with np.load(path,allow_pickle=False) as archive:
            m=json.loads(str(archive['solver_metadata_json'].reshape(()).item()))['metadata']
        assert m['backend_selection']['selected']=='dense'
        assert m['backend_selection']['objective']=='predicted_batch_completion'
        assert m['mesh_convergence_certified']
        degrees.append(m['polynomial_degree'])
        if m['adaptive_mesh']['used']:
            assert m['polynomial_degree'] == 3
            assert all(step['backend_selection']['admission_budget_gib'] <=
                       m['execution_memory_reservation_gib'] for step in m['adaptive_mesh']['steps'])
    assert sorted(degrees)==({24:[1,1],-1400:[1,3],-2800:[3,3]}[density])
    timings=json.loads((tmp_path/'timings'/'timings-v1.json').read_text())
    assert len(timings)==2
    assert all(len(entry['dense'])==1 for entry in timings.values())
    assert not list((tmp_path/'timings').glob('*.tmp'))
