"""Real GUI file loader, preflight/worker and airfoil resource forecast."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '2'
os.environ.update(QT_QPA_PLATFORM='offscreen', GHOST_CPU_FACTORIZATION='compressed',
                  GHOST_DENSE_BACKEND='cpu', GHOST_COMPRESSED_STORAGE_MIB='8192',
                  GHOST_ASSEMBLY_THREADS='4')
from pathlib import Path
import hashlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'tools/GHOST/Backend'))
from PySide6.QtWidgets import QApplication
from solver_tab import SolverTab, _SolveWorker, _2d_panel_limit
from run_setup import DEFAULT_QUALITY
from hpc_scheduler import predict_2d_resources_many

app = QApplication.instance() or QApplication([])
source = ROOT/'airfoil.geo'
snapshot, path, base_dir, digest = SolverTab._load_geometry_file_for_solver(str(source))
record = dict(geometry_sha256=digest, gui_max_panels=_2d_panel_limit(), source_hashes={
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT/'tools/GHOST/Backend').glob('*.py')})
forecasts = {}
for count in (1, 361):
    plan = predict_2d_resources_many(str(source), [10.], ['VV', 'HH'], 'inches',
        _2d_panel_limit(), fine_factor=1.5, n_angles=count, solver_method='experimental_cpu')
    forecasts[str(count)] = {pol: value for (_, pol), value in plan.items()}
    assert all(v['fine_panels'] == 28138 and v['fine_panels'] <= _2d_panel_limit() for v in plan.values())
    print('10 GHz forecast:', count, 'angles', forecasts[str(count)], flush=True)
record['scheduler_forecasts'] = forecasts
setup = dict(units='inches', frequencies_ghz=[1.], angles_deg=list(range(361)),
    scattering='monostatic', mesh_certification=True, accuracy='standard',
    solver_method='experimental_cpu', lu_precision='double')
worker = _SolveWorker(snapshot, path, base_dir, setup['frequencies_ghz'],
    setup['angles_deg'], 'inches', DEFAULT_QUALITY, solver_method='experimental_cpu',
    preflight_setup=setup)
completed, errors, checked, progress = [], [], [], []
worker.finished.connect(lambda result, _: completed.append(result))
worker.error.connect(errors.append)
worker.canceled.connect(errors.append)
worker.setup_checked.connect(checked.append)
worker.progress.connect(lambda percent, message: progress.append((percent, message)))
start = time.perf_counter()
worker.run()
record.update(seconds=time.perf_counter()-start, errors=errors, preflight=checked)
assert not errors, errors
assert len(completed) == 1 and checked
result = completed[0]
assert result['metadata']['solver_method'] == 'compressed_experimental_cpu'
assert result['metadata']['mesh_convergence_certified']
assert result['metadata']['quality_gate']['passed']
assert {pol:len(rows) for pol, rows in result['co_solved_samples'].items()} == {'VV':361, 'HH':361}
assert [pct for pct,_ in progress] == sorted(pct for pct,_ in progress)
assert progress[-1][0] == 100
record.update(passed=True, metadata=result['metadata'], progress=progress)
(HERE/'readiness-probe.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print('GUI loader, preflight and worker passed:', round(record['seconds'], 3), 'seconds', flush=True)
