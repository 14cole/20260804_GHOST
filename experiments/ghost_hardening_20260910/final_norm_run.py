"""Final verification after removing real/imaginary norm work copies."""
import os,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
suite=['test_solver_efficiency','test_solver_compression','test_solver_matrix_pipeline',
       'test_compact_multi_region','test_experimental_cpu','test_thin_sheet',
       'test_2d_co_polarized','test_memory_safety','test_assembly_equivalence',
       'test_assembly_audit_updates','test_hpc_runtime','test_2d_capability_acceptance']
jobs=[('production-tests',root/'tools/GHOST/tests',['-m','unittest']+suite+['-q']),
      ('production-norm',root,[str(here/'benchmark.py'),'--repeat','3']),
      ('materials-dense',root,[str(here/'benchmark.py'),'--materials']),
      ('materials-hierarchical',root,[str(here/'benchmark.py'),'--materials','--factor','hierarchical'])]
for name,cwd,args in jobs:
    start=time.perf_counter()
    with (here/(name+'.log')).open('w') as log:
        completed=subprocess.run([sys.executable]+args,cwd=str(cwd),env=env,stdout=log,stderr=subprocess.STDOUT)
    print(name,completed.returncode,round(time.perf_counter()-start,3),flush=True)
    if completed.returncode:raise RuntimeError(name+' failed')
