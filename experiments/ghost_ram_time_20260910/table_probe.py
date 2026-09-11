from common import *
from assembly_variants import table_class
import cpu_kernels
from scipy.interpolate import PPoly

def main():
    rng=np.random.RandomState(991)
    records=[]
    for k in (81.71-14.52j,420.85-204.63j,841.7-409.26j,4208.5-2046.3j,100-1e-8j):
        upper=.31
        distances=np.r_[rng.uniform(1e-12,upper,100000),np.exp(rng.uniform(np.log(5e-13),np.log(upper),10000))]
        reference=cpu_kernels.values(k,distances)
        # Timings use geometrically smooth but unsorted distance tiles.
        x=rng.uniform(0,.3,(300,1));y=rng.uniform(0,.3,(1,300))
        tile=np.sqrt((x-y)**2+.001**2)
        for degree,growth,phase in ((16,1.25,1.),(12,1.2,.6),(10,1.1,.3),(8,1.04,.1)):
            record=dict(k=[k.real,k.imag],degree=degree,growth=growth,phase=phase)
            try:
                cls=cpu_kernels.KernelTable if degree==16 else table_class(degree,growth,phase)
                table=cls(k,upper)
                joint=PPoly(np.stack([p.c for p in table.polys],axis=-1),table.bounds,extrapolate=False)
                got=np.stack([p(distances) for p in table.polys],axis=-1)
                record['independent_relative_error']=float(np.max(abs(got-reference)/np.maximum(abs(reference),1e-280)))
                _,record['two_calls']=timed(lambda: (table.polys[0](tile),table.polys[1](tile)),5)
                _,record['joint_call']=timed(lambda:joint(tile),5)
                record['evidence']=table.evidence
            except Exception as exc:record['rejected']=str(exc)
            records.append(record)
            print(k,degree,record.get('independent_relative_error',record.get('rejected')),flush=True)
    write('table-results.json',records)

if __name__=='__main__':main()
