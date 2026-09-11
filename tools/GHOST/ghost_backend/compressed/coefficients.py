"""Native Robin, dielectric, sheet and thin-layer coefficient queries."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.assembly.compact import CompactOperator
from ghost_backend.twod.assembly.mass import sparse_mass
import ghost_backend.twod.solver as rcs


class NativeOracle:
    def __init__(self,mesh,infos,pol,k0,kind,obs_order=8,src_order=8,layer=None):
        self.mesh,self.infos,self.pol,self.k0,self.kind=mesh,infos,pol,k0,kind
        self.obs_order,self.src_order=obs_order,src_order
        self.geometry=AssemblyGeometry(mesh);self.nn=len(mesh.nodes);self.n=self.nn
        self.entries=self.calls=self.max_entries=self.dropped_routes=0
        self.mass=sparse_mass(mesh);self.endpoints=np.empty(0,int)
        self.cached_columns=self.cached=None
        self.query_cache=None
        if kind=='robin':
            self.alpha,pec=rcs._robin_alpha_elements(mesh,infos,pol)
            self.pec_nodes=np.zeros(self.nn,bool)
            if pol=='TM':
                for e,flag in zip(mesh.elements,pec):
                    if flag:self.pec_nodes[list(e.node_ids)]=True
        elif kind=='dielectric':
            self.n=2*self.nn;info=infos[0]
            num,den=(info.mu_minus,info.mu_plus) if pol=='TM' else (info.eps_minus,info.eps_plus)
            self.factor=complex(num/den) if abs(den)>rcs.EPS else 1.
            media={complex(i.k_plus) for i in infos if i.plus_region>0}
            if not media:media={complex(i.k_minus) for i in infos if i.minus_region>0}
            if len(media)!=1:raise ValueError('Native dielectric requires one interior medium.')
            self.k1=media.pop()
        elif kind=='sheet':
            z=np.asarray([complex(i.robin_impedance) if int(i.seg_type)==1 else 0 for i in infos])
            c=z/(1j*float(k0)*rcs.ETA0) if pol=='TM' else (1j*float(k0)/rcs.ETA0)*z
            self.weighted=sparse_mass(mesh,c)
            if pol=='TE':self.endpoints=rcs._geometric_sheet_endpoint_nodes(mesh,infos)
        elif kind=='thin':
            eps,mu,d=layer
            alpha,beta=(eps,mu) if pol=='TM' else (mu,eps)
            self.B=d*(beta-1.);self.coefficient=k0**2*d*(alpha-1.)
            if self.B!=0:
                self.n=2*self.nn;ids=self.geometry.node_ids
                vals=(d*(1.-1./beta)/self.geometry.lengths[:,None]*np.array([1.,-1.,-1.,1.])).ravel()
                stiffness=coo_matrix((vals,(np.repeat(ids,2,axis=1).ravel(),np.tile(ids,(1,2)).ravel())),shape=(self.nn,self.nn))
                self.C=(self.coefficient*self.mass+stiffness).tocsr()
                self.mass_lu=splu(self.mass.tocsc())
                self.endpoints=rcs._geometric_sheet_endpoint_nodes(mesh)

            self.maximum_tile=max(1,min(512,(8*1024**2)//max(16*self.nn,1)))
        else:raise ValueError('Unknown native compressed formulation.')

    def primitive(self,kind,rows,cols,k=None,weights=None):
        if len(rows)*len(cols)*16>16*1024**2:raise MemoryError('Native coefficient query exceeds 16 MiB.')
        if not len(rows) or not len(cols):return np.zeros((len(rows),len(cols)),complex)
        k=self.k0 if k is None else k
        key=(kind,complex(k),np.asarray(rows).tobytes(),np.asarray(cols).tobytes(),
             None if weights is None else np.asarray(weights).tobytes())
        if self.query_cache is not None and key in self.query_cache:return self.query_cache[key]
        value=self._primitive(kind,rows,cols,k,weights)
        if self.query_cache is not None:
            value.flags.writeable=False;self.query_cache[key]=value
        return value

    def _primitive(self,kind,rows,cols,k,weights):
        if kind=='W':
            value=rcs._assemble_linear_hypersingular_matrix(self.mesh,k,obs_order=self.obs_order,
                src_order=self.src_order,output_node_ids=(rows,cols),prepared_geometry=self.geometry)
            return value.values
        values=rcs._assemble_linear_operator_matrices_multi(self.mesh,k,kind=='KP',[None],
            compute_single_layer=kind=='S',compute_double_layer=kind!='S',
            single_layer_observation_coefficients_many=[weights],
            output_node_ids_many=[(rows if kind=='S' else [],cols)],
            double_layer_output_node_ids_many=[(rows if kind!='S' else [],cols)],
            obs_order=self.obs_order,src_order=self.src_order,prepared_geometry=self.geometry)
        return values[0][0 if kind=='S' else 1].values

    def prepare_columns(self,columns):
        if self.kind!='thin' or self.B==0:return
        self.cached_columns=self.cached=None
        cols=CompactOperator._ids(columns,self.n);all_rows=np.arange(self.nn)
        if len(cols)>self.maximum_tile:raise MemoryError('Thin-layer column workspace cap exceeded.')
        result=np.zeros((self.n,len(cols)),complex)
        for block in (0,1):
            positions=np.flatnonzero(cols//self.nn==block);cc=cols[positions]%self.nn
            if not len(cc):continue
            raw=self.primitive('S' if block==0 else 'K',all_rows,cc)
            result[:self.nn,positions]=self.C@self.mass_lu.solve(raw)
            raw=None
            if block==0:

                result[self.nn:,positions]=self.B*self.primitive('K',cc,all_rows).T
                result[:self.nn,positions]+=self.mass[:,cc].toarray()
            else:
                result[self.nn:,positions]=-self.B*self.primitive('W',all_rows,cc)
                result[self.nn:,positions]+=self.mass[:,cc].toarray()
        result[self.nn+self.endpoints]=0
        for i,node in enumerate(cols):
            if node>=self.nn and node-self.nn in self.endpoints:result[node,i]=1
        self.cached_columns=cols;self.cached=result

    def get_with_error(self,rows,cols):
        rows,cols=CompactOperator._ids(rows,self.n),CompactOperator._ids(cols,self.n)
        if len(rows)*len(cols)*24>16*1024**2:raise MemoryError('Native tile exceeds workspace cap.')
        result=np.zeros((len(rows),len(cols)),complex)
        self.entries+=result.size;self.calls+=1;self.max_entries=max(self.max_entries,result.size)
        if self.kind=='thin' and self.B!=0:
            if self.cached_columns is None or not np.array_equal(cols,self.cached_columns):self.prepare_columns(cols)
            result[:]=self.cached[rows]
        elif self.kind=='thin':
            result=self.coefficient*self.primitive('S',rows,cols)+self.mass[rows,:][:,cols].toarray()
        elif self.kind=='robin':
            pec=self.pec_nodes[rows];robin=~pec
            if np.any(pec):result[pec]=self.primitive('S',rows[pec],cols)
            if np.any(robin):
                rr=rows[robin]
                result[robin]=self.primitive('KP',rr,cols)-.5*self.mass[rr,:][:,cols].toarray()
                if np.any(self.alpha!=0):result[robin]+=self.primitive('S',rr,cols,weights=self.alpha)
        elif self.kind=='sheet':
            result=self.primitive('S' if self.pol=='TM' else 'W',rows,cols)-self.weighted[rows,:][:,cols].toarray()
            for i,node in enumerate(rows):
                if node in self.endpoints:result[i]=cols==node
        else:
            for rb in (0,1):
                ri=np.flatnonzero(rows//self.nn==rb);rr=rows[ri]%self.nn
                for cb in (0,1):
                    ci=np.flatnonzero(cols//self.nn==cb);cc=cols[ci]%self.nn
                    if not len(ri) or not len(ci):continue
                    if rb==0 and cb==0:value=self.primitive('K',rr,cc)+.5*self.mass[rr,:][:,cc].toarray()
                    elif rb==0:value=-self.primitive('S',rr,cc,k=self.k1)
                    elif cb==0:value=self.primitive('W',rr,cc)
                    else:value=self.factor*(self.primitive('KP',rr,cc,k=self.k1)+.5*self.mass[rr,:][:,cc].toarray())
                    result[np.ix_(ri,ci)]=value
        return result,np.zeros(result.shape)

    def get(self,rows,cols):return self.get_with_error(rows,cols)[0]


class PairedNativeOracle:
    """Cache shared primitive tiles only until both material laws consume them."""
    def __init__(self,first,second):
        if first.n!=second.n:raise ValueError('Paired native layouts must match.')
        self.oracles=(first,second);self.n=first.n
        second.geometry=first.geometry

    def _shared(self,operation):
        cache={}
        for oracle in self.oracles:oracle.query_cache=cache
        try:return [operation(oracle) for oracle in self.oracles]
        finally:
            for oracle in self.oracles:oracle.query_cache=None

    def prepare_columns(self,columns):self._shared(lambda o:o.prepare_columns(columns))
    def get_with_error(self,rows,cols):return self._shared(lambda o:o.get_with_error(rows,cols))
