"""Isolate sweep allocation behavior without changing production files."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import sys,types,runpy,json,hashlib
here=Path(__file__).resolve().parent;backend=here.parents[1]/'tools/GHOST/Backend'
variant=sys.argv[1]
sys.path.insert(0,str(backend))
if variant=='previous':source=(here/'baseline_backend/sweep_compression.py').read_text()
elif variant=='restore_validation':
    source=(backend/'sweep_compression.py').read_text()
    source=source.replace('remainder -= q @ extension\n    error = float(np.linalg.norm(remainder)/norm)',
        'reconstructed = state.q @ recovery + q @ extension\n    error = float(np.linalg.norm(reconstructed-scaled)/norm)\n    reconstructed = None')
else:raise ValueError(variant)
module=types.ModuleType('sweep_compression');module.__file__=str(here/('sweep-'+variant+'.py'))
exec(compile(source,module.__file__,'exec'),module.__dict__)
sys.modules['sweep_compression']=module
sys.argv=[str(here/'benchmark.py'),'--repeat','99']
runpy.run_path(str(here/'benchmark.py'),run_name='__main__')
path=here/'airfoil-f2.0-updated-dense-r99.json'
data=json.loads(path.read_text());data[0]['ablation']=variant
data[0]['source_hashes']['sweep_compression.py']=hashlib.sha256(source.encode()).hexdigest()
(here/('ablation-'+variant+'.json')).write_text(json.dumps(data,indent=2))
# Keep the ordinary benchmark inventory restricted to actual production code.
path.unlink()
