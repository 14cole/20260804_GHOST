"""Sequential fresh processes: no benchmark competes with another test/solve."""
import os,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
suite=['test_solver_efficiency','test_solver_compression','test_solver_matrix_pipeline',
       'test_compact_multi_region','test_experimental_cpu','test_thin_sheet',
       'test_2d_co_polarized','test_memory_safety','test_assembly_equivalence',
       'test_assembly_audit_updates','test_hpc_runtime','test_2d_capability_acceptance']
jobs=[('production-tests',root/'tools/GHOST/tests',['-m','unittest']+suite+['-q']),
      ('experimental-tests',root,[str(here/'test_streamed_operator.py'),'-q']),
      ('production-dense',root,[str(here/'benchmark.py')]),
      ('production-hierarchical',root,[str(here/'benchmark.py'),'--factor','hierarchical']),
      ('materials-dense',root,[str(here/'benchmark.py'),'--materials']),
      ('materials-hierarchical',root,[str(here/'benchmark.py'),'--materials','--factor','hierarchical']),
      ('streamed-baseline',root,[str(here/'baseline_probe.py')]),
      ('streamed-final-f1',root,[str(here/'streamed_probe.py'),'--frequency','1','--tile','256','--cut','32']),
      ('streamed-final-f2',root,[str(here/'streamed_probe.py'),'--frequency','2','--tile','512','--cut','32'])]
for name,cwd,args in jobs:
    started=time.perf_counter()
    with (here/(name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable]+args,cwd=str(cwd),env=env,stdout=log,stderr=subprocess.STDOUT)
    print(name,result.returncode,round(time.perf_counter()-started,3),flush=True)
    if result.returncode:raise RuntimeError('Failed {}; see its log'.format(name))
