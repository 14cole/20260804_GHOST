"""Explicit compressed CPU path: bounded geometry assembly and strict rejection."""
from ghost_backend.execution.options import temporary_directory
from ghost_backend.execution.options import environment_value
import os,tempfile
import numpy as np
from ghost_backend.execution.metrics import timed_stage


def enabled():
    return environment_value('GHOST_CPU_FACTORIZATION','dense').strip().lower()=='compressed'


def storage_budget():
    text=environment_value('GHOST_COMPRESSED_STORAGE_MIB','2048').strip()
    try:value=int(text)
    except ValueError:raise ValueError('GHOST_COMPRESSED_STORAGE_MIB must be a positive integer.')
    if value<16:raise ValueError('GHOST_COMPRESSED_STORAGE_MIB must be at least 16 MiB.')
    return value*1024**2


def checkpoint():
    from ghost_backend.execution.cpu import current_state
    from ghost_backend.twod.assembly.session import current_session
    owner=current_state() or current_session()
    if owner is not None:owner.checkpoint()


def coordinates(mesh,n):
    xy=np.zeros((len(mesh.nodes),2))
    for e in mesh.elements:xy[list(e.node_ids)]=[e.p0,e.p1]
    return np.tile(xy,(n//len(mesh.nodes),1))


@timed_stage('compressed_assembly')
def build(oracle,xy):
    from ghost_backend.compressed.operator import StreamedOperator
    operator=StreamedOperator(oracle,xy,tile=min(512,getattr(oracle,'maximum_tile',512)),
        budget=storage_budget(),checkpoint=checkpoint)
    if hasattr(oracle,'cached'):oracle.cached=oracle.cached_columns=None
    return operator


def native(mesh,infos,pol,k0,kind,obs_order=8,src_order=8,layer=None):
    from ghost_backend.compressed.coefficients import NativeOracle, PairedNativeOracle
    from ghost_backend.compressed.polarization_cache import build_pair
    from ghost_backend.twod.assembly.session import current_session, system_key
    oracle=NativeOracle(mesh,infos,pol,k0,kind,obs_order,src_order,layer)
    session=current_session()
    key=(system_key(mesh,infos or [],'compressed_'+kind,obs_order,src_order),layer)
    previous=session.take(key,pol) if session is not None else None
    if previous is not None:
        previous.load()
        return previous,oracle
    partner=getattr(session,'compressed_partner',None)
    if pol=='TE' and partner is not None and (partner[0] is mesh or kind=='thin'):
        other=NativeOracle(mesh,None if kind=='thin' else partner[1],'TM',k0,kind,obs_order,src_order,layer)
        if other.n==oracle.n:
            pair=build_pair(PairedNativeOracle(oracle,other),coordinates(mesh,oracle.n),
                tile=min(512,getattr(oracle,'maximum_tile',512)),budget=storage_budget(),
                checkpoint=checkpoint,spool_directory=temporary_directory())
            pair[0].reserved_partner_bytes=pair[1].bytes
            key=(system_key(mesh,partner[1] if kind!='thin' else [],'compressed_'+kind,obs_order,src_order),layer)
            session.save(key,'TE',pair[1]);session.compressed_partner=None
            oracle.cached=oracle.cached_columns=other.cached=other.cached_columns=None
            return pair[0],oracle
    operator=build(oracle,coordinates(mesh,oracle.n))
    return operator,oracle


@timed_stage('compressed_assembly')
def regional(mesh,infos,pol,obs_order=8,src_order=8):
    import ghost_backend.twod.formulations.regions as mr
    from ghost_backend.compressed.regional_coefficients import PreparedOracle, PairedOracle
    from ghost_backend.compressed.operator import StreamedOperator
    from ghost_backend.compressed.polarization_cache import build_pair, SpooledOperator
    from ghost_backend.twod.assembly.session import current_session, system_key
    session=current_session();key=system_key(mesh,infos,'compressed_region',obs_order,src_order)
    previous=session.take(key,pol) if session is not None else None
    if previous is not None:
        operator,layout=previous
        operator.load()
        return operator,layout
    partner=getattr(session,'compressed_partner',None)
    if pol=='TE' and partner is not None and partner[0] is mesh:
        oracle=PairedOracle(mesh,infos,partner[1],cut=32,obs_order=obs_order,src_order=src_order)
        xy=mr.dof_coordinates(mesh,oracle.oracles[0].layout)
        pair=build_pair(oracle,xy,tile=512,budget=storage_budget(),checkpoint=checkpoint,spool_directory=temporary_directory())
        pair[0].reserved_partner_bytes=pair[1].bytes
        key=system_key(mesh,partner[1],'compressed_region',obs_order,src_order)
        session.save(key,'TE',(pair[1],oracle.oracles[1].layout))
        session.compressed_partner=None
        return pair[0],oracle.oracles[0].layout
    oracle=PreparedOracle(mesh,infos,pol,cut=32,obs_order=obs_order,src_order=src_order)
    xy=mr.dof_coordinates(mesh,oracle.layout)
    operator=StreamedOperator(oracle,xy,tile=512,budget=storage_budget(),checkpoint=checkpoint)
    return operator,oracle.layout


def thin(mesh,k0,pol,eps,mu,d,order=8):
    from ghost_backend.twod.assembly.kernels import incident_loads
    operator,oracle=native(mesh,None,pol,k0,'thin',order,order,(eps,mu,d))
    coefficient,B=oracle.coefficient,oracle.B
    if B==0:
        def rhs(batch):
            bu,_=incident_loads(mesh,k0,batch,want_dn=False)
            return -coefficient*bu
    else:
        C,mass_lu,endpoints,n=oracle.C,oracle.mass_lu,oracle.endpoints,oracle.nn
        def rhs(batch):
            bu,bq=incident_loads(mesh,k0,batch)
            result=np.empty((2*n,len(batch)),complex)
            result[:n]=-C@mass_lu.solve(bu);result[n:]=-B*bq;result[n+endpoints]=0
            return result
    return operator,rhs
