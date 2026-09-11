"""Serial experiment: does smaller inner tiling help compact geometry queries?"""
from pathlib import Path
import os,sys,json,time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'ghost_ram_time_20260910'))
from common import prepare,np,ops
from compressed_oracle import PairedOracle
from compressed_operator import StreamedOperator
import multi_region as mr
import cpu_execution as execution

mesh,te,k=prepare('airfoil','TE',frequency=2.)
_,tm,_=prepare('airfoil','TM',frequency=2.)
oracle=PairedOracle(mesh,te,tm,cut=32.)
groups=StreamedOperator(oracle.oracles[0],mr.dof_coordinates(mesh,oracle.oracles[0].layout),tile=512,assemble=False).groups
pairs=[(0,0),(0,1),(0,len(groups)//2),(0,len(groups)-1),
       (len(groups)//2,len(groups)//2),(len(groups)//2,len(groups)-1)]
ops.set_assembly_threads(4)
baseline=[];results=[]
for tile in (0,128,256,512):
    ops._ASSEMBLY_TILE=tile
    state=execution.CPUState();state.reuse_operators=False
    timings=[];error=0.
    with execution._STATE.override(state):
        for repeat in range(2):
            start=time.perf_counter()
            for index,(i,j) in enumerate(pairs):
                values=oracle.get_with_error(groups[i],groups[j])
                if tile==0 and repeat==0:baseline.append(values)
                else:
                    for (a,_),(b,_) in zip(values,baseline[index]):
                        error=max(error,float(np.max(abs(a-b)))/max(float(np.max(abs(b))),1e-300))
            timings.append(time.perf_counter()-start)
    assert error<3e-12,(tile,error)
    results.append(dict(inner_tile=tile,seconds=timings,median_seconds=float(np.median(timings)),max_relative_coefficient_difference=error))
    print(results[-1],flush=True)
(HERE/'tile-probe-results.json').write_text(json.dumps(dict(frequency_ghz=2.,assembly_threads=4,pairs=pairs,results=results),indent=2))
