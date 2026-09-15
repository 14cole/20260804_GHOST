"""Dense, compressed and FMM pulse-collocation solves on existing polygons."""
from ghost_backend.execution.runtime import replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.sparse import coo_matrix,diags
from ghost_backend.execution.options import option,effective_assembly_threads
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.twod.geometry import LinearMesh,LinearNode
from ghost_backend.twod.pulse.kernel import PulseKernel,PulseSystem


def pulse_mesh(mesh):
    """Panel IDs for shared region routing; geometry and normals are untouched."""
    return LinearMesh([LinearNode(e.center.copy(),(i,i)) for i,e in enumerate(mesh.elements)],
                      [replace(e,node_ids=(i,i)) for i,e in enumerate(mesh.elements)])


def validate_geometry(infos):
    if any(int(i.seg_type)==1 or i.bc_kind=='thin_layer' for i in infos):
        raise ValueError('Pulse collocation currently supports PEC/IBC bodies and bulk dielectric '
                         'interfaces. Thin sheets and thin-layer approximations require Galerkin.')


def resources(mesh,infos,pol):
    from ghost_backend.twod.formulations.regions import build_layout
    from ghost_backend.twod.fmm.memory import geometry_resources
    validate_geometry(infos)
    pmesh=pulse_mesh(mesh);layout=build_layout(pmesh,infos,pol)
    n=len(pmesh.nodes)
    return dict(nodes=n,system_dofs=layout['n_dof'],n_regions=len(layout['region_props']),
        operator_matrices=3*len(layout['region_props']),formulation='pulse_collocation',
        discretization='pulse',fmm_geometry=geometry_resources(mesh,infos))


def incident(g,k,angles):
    radians=np.deg2rad(angles)
    dirs=np.column_stack((np.cos(radians),np.sin(radians)))
    u=np.exp(1j*k*(g.centers@dirs.T))
    return u,1j*k*(g.normals@dirs.T)*u


def build_system(mesh,infos,pol,k0,checkpoint):
    import ghost_backend.twod.solver as s
    from ghost_backend.twod.formulations import regions as mr
    from ghost_backend.twod.assembly.scatter import multi_outputs
    from ghost_backend.compressed.runtime import storage_budget
    validate_geometry(infos)
    from ghost_backend.twod.assembly.session import current_session
    from ghost_backend.twod.assembly.kernels import mesh_key
    n=len(mesh.elements);ids=np.arange(n);kernels={}
    session=current_session()
    if session is not None:
        family=(mesh_key(mesh),complex(k0),option('fmm_quadrature_order',0),option('fmm_tolerance',1e-10))
        if getattr(session,'pulse_family',None)!=family:
            session.pulse_family=family;session.pulse_kernels={}
        kernels=session.pulse_kernels
    def kernel(k):
        if complex(k) not in kernels:
            kernels[complex(k)]=PulseKernel(mesh,k,checkpoint,storage_budget())
        return kernels[complex(k)]
    f=kernel(k0);g=f.geometry
    if s._is_all_robin(infos):
        alpha,pec=s._robin_alpha_elements(mesh,infos,pol)
        pec=pec if pol=='TM' else np.zeros(n,bool)
        degree=np.bincount(np.asarray([e.node_ids for e in mesh.elements]).ravel(),minlength=len(mesh.nodes))
        cfie=pol=='TM' and np.all(pec) and np.all(degree==2) and option('pulse_pec_cfie',True)
        a=PulseSystem(n,diags(np.full(n,.5) if cfie else np.where(pec,0.,-.5)))
        if cfie:
            a.add_combined_field(f,-1j*k0);a.preferred_recycle_vectors=0
            a.pulse_weights={'K':np.ones(n),'S':np.full(n,-1j*k0)}
        else:
            a.pulse_weights={}
            if np.any(pec):a.add(f,'S',rows=np.where(pec,ids,-1))
            if np.any(~pec):
                a.add(f,'KP',rows=np.where(~pec,ids,-1))
                if np.any(alpha):a.add(f,'S',rows=np.where(~pec,ids,-1),coefficient=alpha)
            if pol=='TE' and not np.any(alpha):a.preferred_recycle_vectors=0
            if np.any(pec) or np.any(alpha):a.pulse_weights['S']=np.where(pec,1.,alpha)
            if np.any(~pec):a.pulse_weights['KP']=(~pec).astype(float)
        def rhs(angles):
            u,q=incident(g,k0,angles)
            return -np.where(pec[:,None],u,q+alpha[:,None]*u)
        return a,rhs,lambda x:x,np.ones(n,bool),getattr(a,'combined_field_eta',None),g.centers

    # The existing multi-region single-layer representation uses only S and K'.
    # Its region signs/flux ratios apply equally to point testing, with I jumps.
    pmesh=pulse_mesh(mesh);layout=mr.build_layout(pmesh,infos,pol)
    rr=[];cc=[];vv=[]
    for mi,iface in enumerate(layout['ifaces']):
        rm,rp=iface['r_m'],iface['r_p'];local=np.arange(iface['n'])
        if rm<0 or rp<0:
            offset,_=layout['dof_map'][mi,'plus' if rm<0 else 'minus']
            keep=abs(iface['robin_alpha'])>s.EPS if pol=='TM' else np.ones(len(local),bool)
            rr.extend(offset+local[keep]);cc.extend(offset+local[keep]);vv.extend(np.full(np.sum(keep),.5 if rm<0 else -.5))
        else:
            flux,_=layout['dof_map'][mi,'minus'];trace,_=layout['dof_map'][mi,'plus']
            for column,weight in ((flux,-.5),(trace,-.5*mr._inverse_beta(layout,iface,pol))):
                rr.extend(flux+local);cc.extend(column+local);vv.extend(np.full(len(local),weight))
    a=PulseSystem(layout['n_dof'],coo_matrix((vv,(rr,cc)),shape=(layout['n_dof'],)*2))
    for k,requests in mr.operator_plan(layout):
        f=kernel(k)
        for request,template in zip(requests,multi_outputs(None,pmesh,layout,k,requests)):
            source=layout['ifaces'][request['source']]['mask']
            coefficient=None if request['observer'] is None else layout['ifaces'][request['observer']]['robin_alpha_elements']
            for kind,output in zip(('S','KP'),template):
                for rows,cols,weight in output.routes:
                    a.add(f,kind,rows,cols,weight,source,coefficient if kind=='S' else None)
    def rhs(angles):
        u,q=incident(g,k0,angles);out=np.zeros((len(a),len(angles)),complex)
        regions=layout['region_props']
        for mi,iface in enumerate(layout['ifaces']):
            rm,rp=iface['r_m'],iface['r_p'];nodes=np.asarray(iface['nodes']);count=len(nodes)
            if rm<0 or rp<0:
                rid,side=(rp,'plus') if rm<0 else (rm,'minus')
                if regions[rid]['has_incident']:
                    offset,_=layout['dof_map'][mi,side]
                    pec=abs(iface['robin_alpha'])<=s.EPS if pol=='TM' else np.zeros(count,bool)
                    load=q[nodes]+iface['robin_alpha'][:,None]*u[nodes]
                    out[offset:offset+count]=-np.where(pec[:,None],u[nodes],load)
            else:
                flux,_=layout['dof_map'][mi,'minus'];trace,_=layout['dof_map'][mi,'plus']
                if regions[rm]['has_incident']:
                    out[flux:flux+count]-=q[nodes];out[trace:trace+count]-=u[nodes]
                if regions[rp]['has_incident']:
                    out[flux:flux+count]+=mr._inverse_beta(layout,iface,pol)*q[nodes]
                    out[trace:trace+count]+=u[nodes]
        return out
    mask,density=mr.exterior_projection(pmesh,layout)
    xy=np.empty((len(a),2))
    for (mi,side),(offset,count) in layout['dof_map'].items():
        xy[offset:offset+count]=g.centers[layout['ifaces'][mi]['nodes']]
    return a,rhs,density,mask,None,xy


class PulseOracle:
    def __init__(self,system):
        self.system=system;self.n=len(system)
        self.calls=self.entries=self.dropped_routes=0
        self.diagonal=system.jumps.diagonal()

    def get_with_error(self,rows,cols):
        a=self.system
        if hasattr(a,'pulse_weights'):
            values=a.kernels[0].coefficients(set(a.pulse_weights),rows,cols)
            value=np.where(np.asarray(rows)[:,None]==np.asarray(cols)[None,:],self.diagonal[rows,None],0.)
            for kind,weight in a.pulse_weights.items():value+=weight[rows,None]*values[kind]
            self.calls+=1;self.entries+=value.size
            return value,np.zeros(value.shape)
        value=a.jumps[rows,:][:,cols].toarray()
        groups={}
        for f,kind,R,C,mask,coefficient in a.terms:
            r=R[rows,:];c=C[:,cols]
            ri=np.unique(r.indices);ci=np.unique(c.tocoo().row)
            if not len(ri) or not len(ci):continue
            key=(f,ri.tobytes(),ci.tobytes())
            if key not in groups:groups[key]=[ri,ci,[]]
            groups[key][2].append((kind,r[:,ri],c[ci,:],mask,coefficient))
        for (f,_,_),(ri,ci,routes) in groups.items():
            blocks=f.coefficients({r[0] for r in routes},ri,ci)
            for kind,R,C,mask,coefficient in routes:
                block=blocks[kind]
                if mask is not None:block=block*np.asarray(mask)[ci][None,:]
                if coefficient is not None:block=block*np.asarray(coefficient)[ri][:,None]
                value+=R@block@C
        self.calls+=1;self.entries+=value.size
        return value,np.zeros(value.shape)


def _row_block_rows(oracle,n,threads):
    """Rows per assembly task, so one task spans about one coefficient chunk.

    A whole row span per task keeps each chunk large enough to stay inside long
    ufunc calls; the second term still leaves several tasks per thread so the
    pool balances.
    """
    from ghost_backend.twod.pulse.coefficients import chunk_pairs
    orders=[k.order for k in oracle.system.kernels] or [8]
    rows=max(1,chunk_pairs(max(orders))//max(1,n))
    return max(1,min(rows,n,-(-n//(4*max(1,threads)))))


@timed_stage('operator_assembly')
def dense_matrix(oracle,checkpoint):
    n=oracle.n;matrix=np.empty((n,n),complex,order='F')
    threads=effective_assembly_threads()
    span=_row_block_rows(oracle,n,threads)
    columns=np.arange(n)
    def row_block(start):
        checkpoint();rows=np.arange(start,min(start+span,n))
        matrix[start:start+len(rows)]=oracle.get_with_error(rows,columns)[0]
    if threads>1:
        with ThreadPoolExecutor(max_workers=threads) as pool:list(pool.map(row_block,range(0,n,span)))
    else:
        for i in range(0,n,span):row_block(i)
    return matrix


def solve_fields(mesh,infos,pol,k0,angles,diagnostics=None):
    import ghost_backend.twod.solver as s
    from ghost_backend.twod.assembly.session import current_session
    from ghost_backend.execution.cpu import current_state,configured_batch_size
    from ghost_backend.linalg.hierarchical import factor_mode
    from ghost_backend.linalg.dense import DenseFactor
    from ghost_backend.linalg.sweep import SweepBasis,solve as sweep_solve
    from ghost_backend.twod.fmm.factor import FMMFactor
    from ghost_backend.twod.fmm.memory import rhs_batch_size
    from ghost_backend.compressed.operator import StreamedOperator
    from ghost_backend.compressed.factor import CompressedFactor
    from ghost_backend.compressed.runtime import storage_budget
    session=current_session();state=current_state()
    checkpoint=state.checkpoint if state else session.checkpoint if session else lambda:None
    system,rhs_builder,density_builder,mask,eta,xy=build_system(mesh,infos,pol,k0,checkpoint)
    mode=factor_mode();oracle=PulseOracle(system)
    for kernel in system.kernels:kernel.budget=storage_budget()/max(1,len(system.kernels))
    evidence=state.systems if state is not None else None
    if mode=='fmm':
        factor=FMMFactor(system,diagnostics,'Pulse collocation',evidence,checkpoint)
    elif mode=='compressed':
        matrix=StreamedOperator(oracle,xy,tile=128,budget=storage_budget(),checkpoint=checkpoint)
        factor=CompressedFactor(matrix,diagnostics,'Pulse collocation',evidence=evidence,checkpoint=checkpoint)
    else:
        matrix=dense_matrix(oracle,checkpoint)
        factor=DenseFactor(matrix,diagnostics,'Pulse collocation',evidence,checkpoint,coordinates=xy)
    factor.event['discretization']='pulse_collocation'
    factor.event['quadrature_orders']=[f.order for f in system.kernels]
    batch=rhs_batch_size(len(system),configured_batch_size()) if mode=='fmm' else configured_batch_size()
    basis=None if mode=='fmm' else SweepBasis(batch)
    amplitude=np.empty(len(angles),complex);max_residual=0.
    g=system.kernels[0].geometry
    for start in range(0,len(angles),batch):
        checkpoint();stop=min(start+batch,len(angles));theta=angles[start:stop]
        rhs=rhs_builder(theta)
        x=factor.solve(rhs) if mode=='fmm' else sweep_solve(factor,rhs,basis)
        max_residual=max(max_residual,float(factor.relative_residual.max()))
        density=density_builder(x)
        dirs=np.column_stack((np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))))
        moments=g.lengths[:,None]*np.exp(1j*k0*(g.centers@dirs.T))*np.sinc(k0*(g.segments@dirs.T)/(2*np.pi))
        if eta is not None:moments*=eta+1j*k0*(g.normals@dirs.T)
        amplitude[start:stop]=np.sum(moments[mask]*density[mask],axis=0)
        if state:state.checkpoint(stop,len(angles))
    return s._rcs_sigma_from_amp(amplitude,k0),amplitude,max_residual
