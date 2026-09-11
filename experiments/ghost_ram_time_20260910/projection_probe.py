"""Project a solved incident basis before recovering all physical densities."""
from common import *
from algebra import qr_basis,fourier_basis
import scipy.linalg as la
from scipy.sparse import load_npz
from cpu_kernels import farfield
import pickle
import argparse

def run(prefix):
    a=np.load(str(prefix)+'-a.npy');b=np.load(str(prefix)+'-b.npy')
    basis,recovery,e=qr_basis(b);xb=la.solve(a,basis,check_finite=False);a=b=None
    meta=json.loads(Path(str(prefix)+'-meta.json').read_text())
    mesh=pickle.loads(Path(str(prefix)+'-mesh.pkl').read_bytes())
    projection=load_npz(str(prefix)+'-projection.npz')
    angles=np.load(str(prefix)+'-angles.npy')
    def field(x,theta,mode):
        f=farfield(mesh,projection@x,meta['k0'],theta,meta['potential'],
            element_mask=meta['element_mask'],projection=mode)
        if meta['second_potential']:
            f+=farfield(mesh,x[len(mesh.nodes):],meta['k0'],theta,meta['second_potential'],projection=mode)
        return f
    def full():return field(xb@recovery,angles,'matched')
    def project_first():return np.sum(field(xb,angles,'grid')*recovery,axis=0)
    def harmonic():
        count=2*int(np.ceil(meta['k0']*meta['radius']+40))+1
        sample=field(xb,np.arange(count)*360/count,'grid')
        coeff,angular,e=fourier_basis(sample,meta['k0'],np.asarray(meta['center']),angles)
        return np.sum((coeff@angular)*recovery,axis=0)
    expected,t=timed(full,3)
    result=dict(rank=e['rank'],full=dict(t,field_error=0),rhs_basis_bytes=xb.nbytes,
        recovered_density_bytes=xb.shape[0]*len(angles)*16)
    for name,fn in (('project_basis',project_first),('harmonic_observation',harmonic)):
        got,t=timed(fn,3);result[name]=dict(t,field_error=difference(got,expected))
    write('projection-'+prefix.parent.name+'-'+prefix.name+'.json',result)
    print(prefix.name,result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args()
    for pol in ('TE','TM'):run(HERE/'systems'/args.folder/pol)
