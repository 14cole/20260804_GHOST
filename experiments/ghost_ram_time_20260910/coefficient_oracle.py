"""Requested blocks assembled from geometry, with no global dense allocation.

Prototype is restricted to the multi-region formulation and uses existing
quadrature/scatter contracts. Rebuilding geometry plans per query is deliberately
measured, not represented as a fast matrix-free backend.
"""
from common import *
import argparse
import multi_region as mr
import system_scatter as ss
import cpu_execution as ce

class GeometryOracle:
    def __init__(self,mesh,infos,pol):
        self.mesh=mesh;self.pol=pol;self.layout=mr.build_layout(mesh,infos,pol)
        self.n=self.layout['n_dof'];self.mass=mr._sparse_mass(mesh)
        self.entries=0;self.calls=0;self.max_entries=0
    def get(self,rows,cols):
        rows=np.asarray(rows,int);cols=np.asarray(cols,int)
        matrix=np.zeros((len(rows),len(cols)),complex,order='F')
        self.entries+=matrix.size;self.calls+=1;self.max_entries=max(self.max_entries,matrix.size)
        rowdest=np.full(self.n,-1,int);coldest=np.full(self.n,-1,int)
        rowdest[rows]=np.arange(len(rows));coldest[cols]=np.arange(len(cols))
        def add(rr,cc,value):
            r,c=rowdest[rr],coldest[cc];keep=(r>=0)&(c>=0)
            np.add.at(matrix,(r[keep],c[keep]),np.broadcast_to(value,r.shape)[keep])
        layout=self.layout
        for mi,interface in enumerate(layout['ifaces']):
            nodes=interface['nodes'];local=self.mass[nodes,:][:,nodes].tocoo()
            rm,rp=interface['r_m'],interface['r_p']
            if rm<0 or rp<0:
                offset,_=layout['dof_map'][mi,'plus' if rm<0 else 'minus']
                keep=np.ones(local.nnz,bool)
                if self.pol=='TM':keep=np.abs(interface['robin_alpha'][local.row])>rcs.EPS
                add(offset+local.row[keep],offset+local.col[keep],(.5 if rm<0 else -.5)*local.data[keep])
            else:
                flux,_=layout['dof_map'][mi,'minus'];trace,_=layout['dof_map'][mi,'plus']
                add(flux+local.row,flux+local.col,-.5*local.data)
                add(flux+local.row,trace+local.col,-.5*mr._inverse_beta(layout,interface,self.pol)*local.data)
        for k,requests in mr.operator_plan(layout):
            original=ss.multi_outputs(matrix,self.mesh,layout,k,requests)
            outputs=[];selected=[]
            for request,pair in zip(requests,original):
                converted=[]
                for output in pair:
                    routes=[]
                    for rr,cc,ww in output.routes:
                        r,c=np.full(len(rr),-1,int),np.full(len(cc),-1,int)
                        ok=rr>=0;r[ok]=rowdest[rr[ok]]
                        ok=cc>=0;c[ok]=coldest[cc[ok]]
                        if np.any(r>=0) and np.any(c>=0):routes.append((r,c,ww))
                    rid=np.flatnonzero(np.any([r>=0 for r,c,w in routes],axis=0)) if routes else np.empty(0,int)
                    cid=np.flatnonzero(np.any([c>=0 for r,c,w in routes],axis=0)) if routes else np.empty(0,int)
                    converted.append(ss.SystemScatter(matrix,len(self.mesh.nodes),rid,cid,routes))
                if any(len(o.row_ids) and len(o.column_ids) for o in converted):
                    selected.append(request);outputs.append(tuple(converted))
            if not outputs:continue
            rcs._assemble_linear_operator_matrices_multi(self.mesh,k,True,
                [layout['ifaces'][r['source']]['mask'] for r in selected],
                compute_double_layer_many=[len(o[1].row_ids)>0 for o in outputs],
                single_layer_observation_coefficients_many=[None if r['observer'] is None else
                    layout['ifaces'][r['observer']]['robin_alpha_elements'] for r in selected],
                output_node_ids_many=[(o[0].row_ids,o[0].column_ids) for o in outputs],
                double_layer_output_node_ids_many=[(o[1].row_ids,o[1].column_ids) for o in outputs],
                operator_outputs=outputs)
        return matrix

def probe():
    p=argparse.ArgumentParser();p.add_argument('--frequency',type=float,default=1.0);args=p.parse_args()
    records=[]
    for pol in ('TE','TM'):
        mesh,infos,k=prepare('airfoil',pol,frequency=args.frequency)
        oracle=GeometryOracle(mesh,infos,pol)
        a=np.load(HERE/'systems'/('airfoil-n256-f{}'.format(args.frequency))/(pol+'-a.npy'),mmap_mode='r')
        rng=np.random.RandomState(398)
        queries=[(np.arange(32),np.arange(128)),
            (rng.choice(oracle.n,32,False),rng.choice(oracle.n,128,False)),
            (np.arange(oracle.n//2,oracle.n//2+1),np.arange(oracle.n))]
        state=ce.CPUState();state.reuse_operators=False
        with ce._STATE.override(state):
            for rows,cols in queries:
                got,timing=timed(lambda:oracle.get(rows,cols),2)
                ref=np.array(a[np.ix_(rows,cols)])
                records.append(dict(pol=pol,n=oracle.n,shape=list(got.shape),timing=timing,
                    relative_error=difference(got,ref),max_absolute_error=float(abs(got-ref).max()),
                    returned_bytes=got.nbytes))
                print(pol,got.shape,timing['median'],difference(got,ref),flush=True)
    write('coefficient-oracle-results.json',records)

if __name__=='__main__':probe()
