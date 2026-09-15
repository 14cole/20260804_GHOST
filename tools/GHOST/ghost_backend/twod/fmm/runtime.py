"""Explicit FMM construction; unsupported formulations never allocate dense fallbacks."""
import numpy as np
from scipy.sparse import bmat
from ghost_backend.execution.options import environment_value, option, effective_assembly_threads
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.twod.fmm.system import FMMSystem


def enabled():return environment_value('GHOST_CPU_FACTORIZATION','dense').strip().lower()=='fmm'


def frequency_family(mesh,k0):
    from ghost_backend.twod.assembly.session import current_session
    from ghost_backend.twod.assembly.kernels import mesh_key
    session=current_session()
    if session is not None:
        tag=(mesh_key(mesh),complex(k0))
        if getattr(session,'fmm_frequency_family',None)!=tag:
            session.fmm_frequency_family=tag;session.fmm_kernels={}


def kernel(mesh,k,order=8):
    from ghost_backend.twod.fmm.galerkin import GalerkinKernel
    from ghost_backend.twod.assembly.session import current_session
    from ghost_backend.twod.assembly.kernels import mesh_key
    from ghost_backend.compressed.runtime import checkpoint,storage_budget
    session=current_session()
    # Retain only one geometry/frequency family, shared between TE and TM.
    key=mesh_key(mesh)
    if session is not None:
        if getattr(session,'fmm_mesh_key',None)!=key:
            session.fmm_mesh_key=key;session.fmm_kernels={}
        cache=session.fmm_kernels
    else:cache={}
    tag=(complex(k),int(order))
    if tag not in cache:
        budget=storage_budget()-sum(v.storage_bytes for v in cache.values())
        previous=next(iter(cache.values()),None)
        cache[tag]=GalerkinKernel(mesh,k,order,eps=option('fmm_tolerance',1e-10),
            threads=effective_assembly_threads(),budget=budget,checkpoint=checkpoint,
            geometry=previous.geometry if previous else None,pairs=previous.pairs if previous else None,
            quadrature=option('fmm_quadrature_order',0))
    return cache[tag]


@timed_stage('fmm_assembly')
def native(mesh,infos,pol,k0,kind,obs_order=8,src_order=8):
    frequency_family(mesh,k0)
    from ghost_backend.compressed.coefficients import NativeOracle
    oracle=NativeOracle(mesh,infos,pol,k0,kind,obs_order,src_order)
    n=oracle.nn;mass=oracle.mass;ids=np.arange(n);f=kernel(mesh,k0,max(obs_order,src_order))
    if kind=='robin':
        cfie=option('fmm_pec_cfie','auto')
        if pol=='TM' and np.all(oracle.pec_nodes) and not np.any(oracle.alpha) and cfie:
            endpoints=f.geometry.node_ids[:,:2].ravel()
            degree=np.bincount(endpoints,minlength=n)[np.unique(endpoints)]
            closed=np.all(degree==2)
            if not closed and cfie is True:
                raise ValueError('FMM PEC combined field requires closed contours.')
            # Internal GHOST normals point into a PEC body. The exterior DLP
            # trace is therefore +M/2+K for G=i H0^(2)/4.
            # In these inward-normal, H0^(2) conventions the outgoing CFIE
            # coupling is -ik. Preserve the same coupling in field projection.
            if closed:
                a=FMMSystem(n,.5*mass)
                a.add_combined_field(f,-1j*k0)
                a.preferred_recycle_vectors=0
                return a,oracle
        jumps=-.5*mass
        if np.any(oracle.pec_nodes):
            jumps=jumps.multiply((~oracle.pec_nodes)[:,None])
        a=FMMSystem(n,jumps)
        # Closed PEC equations showed a benefit in qualification. Material
        # interfaces, sheets, open contours and CFIE keep the established ILU.
        a.spatial_coarse_eligible=bool(all(complex(i.robin_impedance)==0 for i in infos)
                                      and np.all(np.bincount(f.ids,minlength=n)==2))
        pec=np.where(oracle.pec_nodes,ids,-1);robin=np.where(~oracle.pec_nodes,ids,-1)
        if np.any(pec>=0):a.add(f,'S',rows=pec)
        if np.any(robin>=0):
            a.add(f,'KP',rows=robin)
            if np.any(oracle.alpha):a.add(f,'S',rows=robin,coefficient=oracle.alpha)
        if pol=='TE' and not np.any(oracle.alpha):
            a.preferred_recycle_vectors=0
    elif kind=='sheet':
        a=FMMSystem(n,-oracle.weighted)
        a.add(f,'S' if pol=='TM' else 'W');a.endpoints=oracle.endpoints
    elif kind=='dielectric':
        interior=kernel(mesh,oracle.k1,max(obs_order,src_order))
        a=FMMSystem(2*n,bmat([[.5*mass,None],[None,.5*oracle.factor*mass]],format='csr'))
        a.add(f,'K',rows=ids,cols=ids)
        a.add(interior,'S',rows=ids,cols=n+ids,weight=-1)
        a.add(f,'W',rows=n+ids,cols=ids)
        a.add(interior,'KP',rows=n+ids,cols=n+ids,weight=oracle.factor)
    else:raise ValueError('This formulation is not yet supported by FMM.')
    return a,oracle


@timed_stage('fmm_assembly')
def regional(mesh,infos,pol,obs_order=8,src_order=8):
    from ghost_backend.compressed.regional_coefficients import PreparedOracle
    oracle=PreparedOracle(mesh,infos,pol,None,obs_order,src_order)
    frequency_family(mesh,oracle.layout['region_props'][0]['k'])
    a=FMMSystem(oracle.n,oracle.jumps)
    for k,prepared in oracle.groups:
        f=kernel(mesh,k,max(obs_order,src_order))
        for request,template,source,coefficient,_,_ in prepared:
            for kind,output in zip(('S','KP'),template):
                for rows,cols,weights in output.routes:
                    a.add(f,kind,rows,cols,weights,source,coefficient if kind=='S' else None)
    return a,oracle.layout
