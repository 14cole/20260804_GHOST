"""Scoped prototypes for ownership/scatter, preserving all original quadrature."""
from support import *
import inspect
import linecache
from contextlib import contextmanager
from unittest.mock import patch
import robin_system,dielectric_system,sheet_system
from system_scatter import SystemScatter

def clone(function,replacements,extra=None):
    source=inspect.getsource(inspect.unwrap(function));source=source[source.index('def '):]
    for old,new in replacements:
        if source.count(old)!=1:raise RuntimeError('Prototype source drift: '+old[:60])
        source=source.replace(old,new)
    namespace=dict(inspect.unwrap(function).__globals__)
    namespace.update(extra or {})
    filename='<audit '+function.__name__+' '+hashlib.sha256(source.encode()).hexdigest()[:12]+'>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    exec(compile(source,filename,'exec'),namespace)
    return namespace[function.__name__]

class WholeSink(SystemScatter):
    def __init__(self,matrix,n,enabled):
        ids=np.arange(n,dtype=int) if enabled else np.empty(0,int)
        super().__init__(matrix,n,ids,ids,[]);self.enabled=enabled
    def scatter_add(self,rows,columns,values):
        if self.enabled:np.add.at(self.matrix,(rows,columns),values)

def sk(mesh,k0,obs_normal_deriv,obs_order=8,src_order=8,far_ratio=3.0,source_element_mask=None,
       compute_single_layer=True,compute_double_layer=True,target_s=None,target_k=None,**kwargs):
    n=len(mesh.nodes);ids=np.arange(n,dtype=int);empty=np.empty(0,int)
    zero=np.broadcast_to(np.zeros((),complex),(n,n))
    s=(np.zeros((n,n),complex,order='F') if target_s is None else target_s) if compute_single_layer else zero
    k=(np.zeros((n,n),complex,order='F') if target_k is None else target_k) if compute_double_layer else zero
    rcs._assemble_linear_operator_matrices_multi(mesh,k0,obs_normal_deriv,[source_element_mask],
        obs_order=obs_order,src_order=src_order,far_ratio=far_ratio,compute_single_layer=compute_single_layer,
        compute_double_layer=compute_double_layer,
        output_node_ids_many=[(ids if compute_single_layer else empty,ids)],
        double_layer_output_node_ids_many=[(ids if compute_double_layer else empty,ids)],
        operator_outputs=[(WholeSink(s,n,compute_single_layer),WholeSink(k,n,compute_double_layer))],**kwargs)
    return s,k

def robin_outputs(matrix,mesh,requests):
    n=len(mesh.nodes);ids=np.arange(n,dtype=int);ones=np.ones(n,complex)
    outputs=[]
    for srows,krows,weights in requests:
        if weights is None:matrix[np.asarray(srows,int)]=0
        pair=[]
        for rows in (srows,krows):
            r=np.full(n,-1,int);r[rows]=rows
            routes=[(r,ids,ones)] if len(rows) else []
            pair.append(SystemScatter(matrix,n,rows,ids,routes))
        outputs.append(tuple(pair))
    return outputs

@contextmanager
def owned():
    w=clone(rcs._assemble_linear_hypersingular_matrix,[
        ("    source_element_mask: 'Optional[np.ndarray]' = None,\n)","    source_element_mask: 'Optional[np.ndarray]' = None,\n    destination=None,\n)"),
        ('d_mat = np.zeros((nnodes, nnodes), dtype=np.complex128)',
         "d_mat = np.zeros((nnodes, nnodes), dtype=np.complex128, order='F') if destination is None else destination")])
    ro_source=inspect.getsource(robin_system.assemble_system)
    start=ro_source.index('    for (srows, krows, weights), (s, k) in zip(requests, outputs):')
    stop=ro_source.index("    if session is not None and pol == 'TE':",start)
    robin=clone(robin_system.assemble_system,[
        ("matrix = np.array(k, complex, order='F', copy=True)",'matrix = k'),
        ('    outputs = rcs._assemble_linear_operator_matrices_multi(',
         '    destinations = robin_outputs(matrix,mesh,requests)\n    outputs = rcs._assemble_linear_operator_matrices_multi('),
        ('double_layer_output_node_ids_many=[(rows, np.arange(n)) for _, rows, _ in requests]) if requests else []',
         'double_layer_output_node_ids_many=[(rows, np.arange(n)) for _, rows, _ in requests],\n        operator_outputs=destinations) if requests else []'),
        (ro_source[start:stop],'')],dict(robin_outputs=robin_outputs))
    dielectric=clone(dielectric_system.assemble_system,[
        ('obs_order=obs_order, src_order=src_order, compute_single_layer=False)',
         'obs_order=obs_order, src_order=src_order, compute_single_layer=False, target_k=matrix[:n,:n])'),
        ('    matrix[:n, :n] = k\n',''),
        ('obs_order=obs_order, src_order=src_order)\n    matrix[:n, n:] = s',
         'obs_order=obs_order, src_order=src_order, target_s=matrix[:n,n:], target_k=matrix[n:,n:])'),
        ('    matrix[n:, n:] = k\n',''),
        ('w = rcs._assemble_linear_hypersingular_matrix(mesh, k0, obs_order=obs_order, src_order=src_order)\n    matrix[n:, :n] = w',
         'w = rcs._assemble_linear_hypersingular_matrix(mesh, k0, obs_order=obs_order, src_order=src_order, destination=matrix[n:,:n])')])
    sheet=clone(sheet_system.assemble_system,[
        ("matrix = np.array(operator, dtype=np.complex128, order='F', copy=True)",'matrix = operator')])
    with patch.object(rcs,'_assemble_linear_operator_matrices',sk), \
         patch.object(rcs,'_assemble_linear_hypersingular_matrix',w), \
         patch.object(robin_system,'assemble_system',robin), \
         patch.object(dielectric_system,'assemble_system',dielectric), \
         patch.object(sheet_system,'assemble_system',sheet):
        yield
