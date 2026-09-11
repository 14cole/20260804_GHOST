"""Experimental regional coefficient queries with prepared plans and loss bounds.

No global dense A. Optional pruning removes whole routed contributions only.
This module is outside the production import path.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ghost_ram_time_20260910'))
from common import *
import multi_region as mr
import system_scatter as ss
from scipy.sparse import coo_matrix
from assembly_geometry import AssemblyGeometry
from compact_operators import CompactOperator


def hankel_envelopes(k, lower_distance):
    """Upper envelopes for |i H0/4| and |i k H1/4|, alpha*r > 1.

    DLMF 10.27.8 and 10.32.9 give |H_nu^(2)(kr)| <= 2 K_nu(alpha*r)/pi.
    cosh(t)>=1+t*t/2 and cosh(t)<=exp(t*t/2) bound K0 and K1 by Gaussians.
    Use a finite-domain bound only; no growing/real kernel is truncated.
    """
    a = -complex(k).imag*np.asarray(lower_distance)
    if complex(k).imag >= 0 or complex(k).real <= 0 or not np.all(np.isfinite(a)) or np.any(a <= 1):
        return None
    # Clamp the exponent upward; this avoids understating subnormal tails.
    decay = np.exp(-np.minimum(a,700))
    g = decay/np.sqrt(8*np.pi*a)
    h = abs(k)*decay/np.sqrt(8*np.pi*(a-1))
    return g*(1+1e-8),h*(1+1e-8)


class PreparedOracle:
    def __init__(self,mesh,infos,pol,cut=None):
        if pol not in ('TE', 'TM'):
            raise ValueError('Polarization must be TE or TM.')
        if cut is not None and (not np.isfinite(cut) or cut <= 1):
            raise ValueError('Attenuation cut must be finite and greater than one.')
        self.mesh,self.pol,self.cut=mesh,pol,cut
        self.geometry=AssemblyGeometry(mesh)
        self.layout=mr.build_layout(mesh,infos,pol)
        self.n=self.layout['n_dof']
        self.entries=self.calls=self.max_entries=self.dropped_routes=0
        nn=len(mesh.nodes)
        self.xy=np.zeros((nn,2));self.radius=np.zeros(nn)
        self.mass=np.zeros(nn);self.normal_mass=np.zeros(nn)
        for e in mesh.elements:
            for node,point in zip(e.node_ids,(e.p0,e.p1)):
                self.xy[node]=point
                self.radius[node]=max(self.radius[node],e.length)
                self.mass[node]+=e.length*.5
                self.normal_mass[node]+=e.length*.5*np.linalg.norm(e.normal)
        self.groups=[]
        for k,requests in mr.operator_plan(self.layout):
            templates=ss.multi_outputs(None,mesh,self.layout,k,requests)
            prepared=[]
            for request,template in zip(requests,templates):
                source=self.layout['ifaces'][request['source']]['mask']
                coefficient=(None if request['observer'] is None else
                    self.layout['ifaces'][request['observer']]['robin_alpha_elements'])
                source_mass=np.zeros(nn);weighted_mass=np.zeros(nn)
                for j,e in enumerate(mesh.elements):
                    if source[j]:np.add.at(source_mass,np.asarray(e.node_ids),e.length*.5)
                    weight=1 if coefficient is None else abs(coefficient[j])
                    np.add.at(weighted_mass,np.asarray(e.node_ids),weight*e.length*.5)
                prepared.append((request,template,source,coefficient,source_mass,weighted_mass))
            self.groups.append((k,prepared))
        # Prepare the sparse jump terms once rather than slicing mass per query.
        mass=mr._sparse_mass(mesh);rr=[];cc=[];vv=[]
        for mi,iface in enumerate(self.layout['ifaces']):
            local=mass[iface['nodes'],:][:,iface['nodes']].tocoo()
            rm,rp=iface['r_m'],iface['r_p']
            if rm<0 or rp<0:
                offset,_=self.layout['dof_map'][mi,'plus' if rm<0 else 'minus']
                keep=np.ones(local.nnz,bool)
                if pol=='TM':keep=abs(iface['robin_alpha'][local.row])>rcs.EPS
                rr.extend(offset+local.row[keep]);cc.extend(offset+local.col[keep])
                vv.extend((.5 if rm<0 else -.5)*local.data[keep])
            else:
                flux,_=self.layout['dof_map'][mi,'minus'];trace,_=self.layout['dof_map'][mi,'plus']
                for column,weight in ((flux,-.5),(trace,-.5*mr._inverse_beta(self.layout,iface,pol))):
                    rr.extend(flux+local.row);cc.extend(column+local.col);vv.extend(weight*local.data)
        self.jumps=coo_matrix((vv,(rr,cc)),shape=(self.n,self.n),dtype=complex).tocsr()

    def get(self,rows,cols):
        return self.get_with_error(rows,cols)[0]

    def get_with_error(self,rows,cols):
        rows,cols=CompactOperator._ids(rows,self.n),CompactOperator._ids(cols,self.n)
        matrix=np.asarray(self.jumps[rows,:][:,cols].toarray(),order='F')
        error=np.zeros(matrix.shape)
        self.entries+=matrix.size;self.calls+=1;self.max_entries=max(self.max_entries,matrix.size)
        rd,cd=np.full(self.n,-1,int),np.full(self.n,-1,int)
        rd[rows]=np.arange(len(rows));cd[cols]=np.arange(len(cols))
        nn=len(self.mesh.nodes)
        for k,prepared in self.groups:
            outputs=[];masks=[];coefficients=[]
            for request,template,source,coefficient,source_mass,weighted_mass in prepared:
                pair=[]
                for kind,output in enumerate(template):
                    routes=[]
                    for old_r,old_c,weight in output.routes:
                        r,c=np.full(nn,-1,int),np.full(nn,-1,int)
                        ok=old_r>=0;r[ok]=rd[old_r[ok]]
                        ok=old_c>=0;c[ok]=cd[old_c[ok]]
                        ri,ci=np.flatnonzero(r>=0),np.flatnonzero(c>=0)
                        if not len(ri) or not len(ci):continue
                        dropped=False
                        if self.cut is not None and complex(k).imag<0:
                            lower=np.linalg.norm(self.xy[ri,None]-self.xy[None,ci],axis=-1)
                            lower=np.maximum(0,lower-self.radius[ri,None]-self.radius[None,ci])
                            if np.all(-complex(k).imag*lower>=self.cut):
                                envelopes=hankel_envelopes(k,lower)
                                if envelopes is not None:
                                    obs=weighted_mass[ri] if kind==0 else self.normal_mass[ri]
                                    bound=envelopes[kind]*obs[:,None]*source_mass[None,ci]*abs(weight[ri,None])
                                    np.add.at(error,(r[ri,None],c[None,ci]),bound)
                                    self.dropped_routes+=1;dropped=True
                        if not dropped:routes.append((r,c,weight))
                    rid=np.flatnonzero(np.any([r>=0 for r,c,w in routes],axis=0)) if routes else np.empty(0,int)
                    cid=np.flatnonzero(np.any([c>=0 for r,c,w in routes],axis=0)) if routes else np.empty(0,int)
                    pair.append(ss.SystemScatter(matrix,nn,rid,cid,routes))
                if any(len(o.row_ids) and len(o.column_ids) for o in pair):
                    outputs.append(pair);masks.append(source);coefficients.append(coefficient)
            if outputs:
                rcs._assemble_linear_operator_matrices_multi(self.mesh,k,True,masks,
                    compute_double_layer_many=[bool(len(o[1].row_ids)) for o in outputs],
                    single_layer_observation_coefficients_many=coefficients,
                    output_node_ids_many=[(o[0].row_ids,o[0].column_ids) for o in outputs],
                    double_layer_output_node_ids_many=[(o[1].row_ids,o[1].column_ids) for o in outputs],
                    operator_outputs=outputs, prepared_geometry=self.geometry)
        return matrix,error
