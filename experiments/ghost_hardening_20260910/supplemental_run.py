"""Check material compression limits and repeat a noisy process-memory pair."""
import os,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
jobs=[('native-matrices',[str(here/'native_matrix_probe.py')],0),
      ('production-baseline-r2',[str(here/'benchmark.py'),'--baseline','--repeat','2'],0),
      ('production-dense-r2',[str(here/'benchmark.py'),'--repeat','2'],0),
      ('experimental-tests-final',[str(here/'test_streamed_operator.py'),'-q'],0),
      ('streamed-final-f1',[str(here/'streamed_probe.py'),'--frequency','1','--tile','256','--cut','32'],0),
      ('streamed-final-f2',[str(here/'streamed_probe.py'),'--frequency','2','--tile','512','--cut','32'],0),
      ('cli-budget-rejection',[str(here/'streamed_probe.py'),'--frequency','1','--pol','TE',
        '--cut','32','--compression','svd','--budget-mib','.001'],1)]
for name,args,expected in jobs:
    started=time.perf_counter()
    with (here/(name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable]+args,cwd=str(root),env=env,stdout=log,stderr=subprocess.STDOUT)
    print(name,result.returncode,round(time.perf_counter()-started,3),flush=True)
    if result.returncode!=expected:raise RuntimeError('Unexpected result for '+name)
