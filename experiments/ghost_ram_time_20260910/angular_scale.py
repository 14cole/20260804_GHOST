"""10 GHz geometry and incident RHS only: does not assemble or solve dense A."""
from common import *
import multi_region as mr
from algebra import qr_basis,fourier_basis

def main():
    records=[];angles=np.linspace(0,360,361)
    for f in (1,2,10):
        mesh,infos,k=prepare('airfoil',frequency=f)
        layout=mr.build_layout(mesh,infos,'TE')
        xy=mr.dof_coordinates(mesh,layout);center=(xy.min(axis=0)+xy.max(axis=0))/2
        radius=np.linalg.norm(xy-center,axis=1).max()
        b,load=timed(lambda:mr.rhs_many(mesh,layout,k,angles),1)
        record=dict(frequency=f,n=len(b),ka=k*radius,rhs_load=load,rhs_bytes=b.nbytes,fourier=[])
        for padding in (28,40,52):
            order=int(np.ceil(k*radius+padding));m=2*order+1
            sampled,sample_load=timed(lambda:mr.rhs_many(mesh,layout,k,np.arange(m)*360/m),1)
            answer,timing=timed(lambda:fourier_basis(sampled,k,center,angles),1)
            basis,recovery,evidence=answer
            reconstructed=basis@recovery
            scale=np.max(abs(b),axis=1);scale[scale==0]=1
            evidence.update(padding=padding,sampling_columns=m,sample_load=sample_load,basis_time=timing,
                relative_error=difference(reconstructed,b),
                scaled_relative_error=float(np.linalg.norm((reconstructed-b)/scale[:,None])/np.linalg.norm(b/scale[:,None])))
            record['fourier'].append(evidence)
        answer,timing=timed(lambda:qr_basis(b),1)
        record['qr']=dict(answer[2],time=timing)
        records.append(record)
        write('angular-scaling.json',records)
        print(f,record['n'],record['qr'],record['fourier'][-1],flush=True)

if __name__=='__main__':main()
