"""Quantify safe-looking locality in lossy regions against exact captured A.

Dropping is an approximation, not an already-qualified assembly shortcut.
These tests still build dense matrices and do not measure sparse solve speed.
"""
from support import *
import multi_region as mr
import scipy.linalg as la
from algebra import qr_basis,project
import argparse

def geometry(frequency,pol):
    mesh,infos,k0=prepare('airfoil',pol,frequency=frequency)
    layout=mr.build_layout(mesh,infos,pol)
    xy=mr.dof_coordinates(mesh,layout)
    radius=np.zeros(len(mesh.nodes))
    for e in mesh.elements:
        for i in e.node_ids:radius[i]=max(radius[i],e.length)
    support=np.empty(layout['n_dof']);atten=np.empty(layout['n_dof']);regions=np.empty(layout['n_dof'],int)
    for (mi,side),(offset,count) in layout['dof_map'].items():
        iface=layout['ifaces'][mi];rid=iface['r_m' if side=='minus' else 'r_p']
        support[offset:offset+count]=radius[iface['nodes']]
        atten[offset:offset+count]=max(0,-layout['region_props'][rid]['k'].imag)
        regions[offset:offset+count]=rid
    return xy,support,atten,regions

def truncate(a,xy,support,atten,optical):
    candidate=a.copy(order='F');rowerror=np.zeros(len(a));count=0;eligible=0
    by_region={};nonzeros=np.count_nonzero(a)
    for start in range(0,len(a),64):
        stop=min(len(a),start+64)
        # Node-to-node distance minus both entire basis-support radii bounds
        # the separation of any source and testing quadrature points below.
        d=np.linalg.norm(xy[start:stop,None,:]-xy[None,:,:],axis=2)
        d=np.maximum(d-support[start:stop,None]-support[None,:],0)
        mask=d*atten[None,:]>=optical
        block=candidate[start:stop]
        count+=int(np.count_nonzero(mask & (block!=0)))
        eligible+=int(np.count_nonzero((atten[None,:]>0)&(block!=0)))
        rowerror[start:stop]=np.sum(np.where(mask,abs(block),0),axis=1)
        block[mask]=0
    return candidate,rowerror,dict(dropped_nonzeros=count,total_nonzeros=int(nonzeros),lossy_nonzeros=eligible)

def run(frequency):
    records=[]
    for pol in ('TE','TM'):
        prefix=PRIOR/'systems'/('airfoil-n256-f{}'.format(frequency))/pol
        a=np.load(str(prefix)+'-a.npy');b=np.load(str(prefix)+'-b.npy')
        reference=project(prefix,np.load(str(prefix)+'-x.npy'))
        xy,support,atten,regions=geometry(frequency,pol)
        basis,recovery,_=qr_basis(b)
        norm=rcs.matrix_inf_norm(a)
        for optical in (16,24,32,40):
            candidate,rowerror,e=truncate(a,xy,support,atten,optical)
            start=time.perf_counter();xb=la.solve(candidate,basis,check_finite=False)
            solve_time=time.perf_counter()-start;x=xb@recovery
            den=np.maximum(norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
            true_error=np.max(np.max(abs(a@x-b),axis=0)/den)
            bound=np.max((np.max(abs(candidate@x-b),axis=0)+rowerror.max()*np.max(abs(x),axis=0))/den)
            records.append(dict(frequency=frequency,pol=pol,optical_distance=optical,counts=e,
                solve_seconds=solve_time,backward=float(true_error),bound=float(bound),
                coefficient_inf_error=float(rowerror.max()/norm),field_error=difference(project(prefix,x),reference)))
            print(pol,optical,records[-1],flush=True)
            candidate=None
    write('loss-locality-f{}.json'.format(frequency),records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frequency',type=float,default=2.0);args=p.parse_args();run(args.frequency)
