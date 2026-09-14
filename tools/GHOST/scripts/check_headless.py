"""Run the qualified solver subset without Qt or companion product packages."""
from pathlib import Path
import os,subprocess,sys
import tempfile

root=Path(__file__).resolve().parents[1]
names=('automatic_backend','automatic_hpc','batch_presets','fmm','nystrom','near_separation','direct_solver_methods','execution_options',
       'rcs_physics_regression','2d_capability_acceptance','experimental_cpu','solver_followups',
       'compact_multi_region','compressed_path','memory_safety','thin_sheet')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
command=[sys.executable,'-m','pytest',*[str(root/'ghost_backend/tests'/('test_'+n+'.py')) for n in names],
         '-q','--tb=short',*sys.argv[1:]]
if any(arg.startswith('--basetemp') for arg in sys.argv[1:]):
    raise SystemExit(subprocess.call(command,env=env,cwd=root))
# Never reuse pytest's user-named folder across accounts or sandbox contexts.
with tempfile.TemporaryDirectory(prefix='ghost-qualification-') as directory:
    raise SystemExit(subprocess.call(command+['--basetemp',str(Path(directory)/'pytest')],env=env,cwd=root))
