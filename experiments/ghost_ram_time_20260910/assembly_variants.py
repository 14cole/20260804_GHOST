"""Scoped assembly counterfactuals, never imported by production."""
from common import *
from contextlib import ExitStack,contextmanager
import inspect
import textwrap
from unittest.mock import patch
from scipy.interpolate import PPoly
import cpu_kernels
import system_scatter

ORIGINAL_SCATTER=system_scatter.SystemScatter.scatter_add
SCATTER_STATS=dict(unique=0,repeated=0)

def fast_scatter(self,rows,columns,values):
    if np.asarray(rows).ndim!=2 or np.asarray(columns).ndim!=2 or rows.shape[1]!=1 or columns.shape[0]!=1:
        return ORIGINAL_SCATTER(self,rows,columns,values)
    for row_map,col_map,weights in self.routes:
        rr,cc=row_map[rows].ravel(),col_map[columns].ravel()
        ri,ci=np.flatnonzero(rr>=0),np.flatnonzero(cc>=0)
        if not len(ri) or not len(ci):continue
        r,c=rr[ri],cc[ci]
        value=np.broadcast_to(values,(len(rr),len(cc)))[np.ix_(ri,ci)]*weights[rows].ravel()[ri,None]
        if len(np.unique(r))==len(r) and len(np.unique(c))==len(c):
            self.matrix[np.ix_(r,c)]+=value
            SCATTER_STATS['unique']+=1
        else:
            np.add.at(self.matrix,(r[:,None],c[None,:]),value)
            SCATTER_STATS['repeated']+=1

ORIGINAL_SELECTOR=cpu_kernels.select_far_kernels
def paired_selector(mesh,k,green,hankel):
    fg,fh=ORIGINAL_SELECTOR(mesh,k,green,hankel)
    table=inspect.getclosurevars(fg).nonlocals.get('table')
    if table is not None:
        joint=getattr(table,'audit_joint',None)
        if joint is None:
            joint=PPoly(np.stack([p.c for p in table.polys],axis=-1),table.bounds,extrapolate=False)
            table.audit_joint=joint
        def pair(k0,real_k,dist,kr,scratch,g,h):
            values=joint(dist)
            g[:],h[:]=values[...,0],values[...,1]
            if not np.all(np.isfinite(values)):
                fg(k0,real_k,dist,kr,scratch,g);fh(k0,real_k,dist,kr,scratch,h)
        fg.pair=pair
    return fg,fh

def clone(function,replacements):
    source=inspect.getsource(inspect.unwrap(function))
    source=source[source.index('def '):]
    for old,new in replacements:
        if source.count(old)!=1:raise RuntimeError('Source drift in experiment: '+old[:80])
        source=source.replace(old,new)
    namespace=dict(ops.__dict__)
    exec(compile(source,'<experiment '+function.__name__+'>','exec'),namespace)
    return namespace[function.__name__]

def paired_assembly():
    old='''                    if want_s:
                        far_green(k0, real_k, dist, krbuf, work, g_buf)
                    if want_k:
                        far_hankel(k0, real_k, dist, krbuf, work, h1_buf)'''
    new='''                    if want_s and want_k and hasattr(far_green, 'pair'):
                        far_green.pair(k0, real_k, dist, krbuf, work, g_buf, h1_buf)
                    else:
                        if want_s:
                            far_green(k0, real_k, dist, krbuf, work, g_buf)
                        if want_k:
                            far_hankel(k0, real_k, dist, krbuf, work, h1_buf)'''
    return clone(ops._assemble_linear_operator_matrices_multi,[(old,new)])

def graded_w(unsafe=False):
    if unsafe:
        return clone(ops._assemble_linear_hypersingular_matrix,[('box_order = max(int(obs_order), int(src_order), 16)',
            'box_order = max(int(obs_order), int(src_order), 8)')])
    old='''            src_pts = quad_pts[src_slice]
            acc ='''
    new='''            active = batch_ij | (batch_ji if mirrored else False)
            ratio_min = float(np.min(np.where(active, centre_dist / scale, np.inf)))
            cap = max(int(obs_order), int(src_order))
            order = max(cap, 16) if ratio_min < 3.0 else _graded_far_order(
                abs(complex(k0))*float(max(obs_len.max(), src_len.max())), ratio_min, cap)
            qt, qw = _get_quadrature(order)
            t_f = np.asarray(qt)
            phi_arr = np.array([_linear_shape_values(float(t)) for t in qt])
            obs_pts = p0_arr[obs_slice,None,:] + t_f[None,:,None]*seg_arr[obs_slice,None,:]
            src_pts = p0_arr[src_slice,None,:] + t_f[None,:,None]*seg_arr[src_slice,None,:]
            acc ='''
    return clone(ops._assemble_linear_hypersingular_matrix,[(old,new)])

def table_class(degree,growth,phase):
    source=textwrap.dedent(inspect.getsource(cpu_kernels.KernelTable.__init__))
    source=source.replace('degree=16','degree='+str(degree))
    old='bounds[-1]*1.25,bounds[-1]+1./abs(k)'
    if source.count(old)!=1:raise RuntimeError('Kernel table source drift')
    source=source.replace(old,'bounds[-1]*{},bounds[-1]+{}/abs(k)'.format(growth,phase))
    namespace=dict(cpu_kernels.__dict__)
    exec(compile(source,'<experimental screened polynomial>','exec'),namespace)
    return type('ExperimentalKernelTable',(cpu_kernels.KernelTable,),{'__init__':namespace['__init__']})

@contextmanager
def variant(name):
    with ExitStack() as stack:
        if name=='partial':
            from partial_table import partial_table
            stack.enter_context(partial_table())
        if name=='wpairs':
            from w_pair_groups import candidate
            fn=candidate()
            stack.enter_context(patch.object(rcs,'_assemble_linear_hypersingular_matrix',fn))
            stack.enter_context(patch.object(ops,'_assemble_linear_hypersingular_matrix',fn))
        if name.startswith('table'):
            degree=int(name.split('_')[0][5:])
            growth,phase={12:(1.2,.6),10:(1.1,.3),8:(1.04,.1)}[degree]
            stack.enter_context(patch.object(cpu_kernels,'KernelTable',table_class(degree,growth,phase)))
        if name in ('scatter','combined'):
            stack.enter_context(patch.object(system_scatter.SystemScatter,'scatter_add',fast_scatter))
        if name in ('paired','combined') or name.endswith('_paired'):
            candidate=paired_assembly()
            stack.enter_context(patch.object(ops,'_assemble_linear_operator_matrices_multi',candidate))
            stack.enter_context(patch.object(rcs,'_assemble_linear_operator_matrices_multi',candidate))
            stack.enter_context(patch.object(cpu_kernels,'select_far_kernels',paired_selector))
        if name in ('wgraded','w8','combined'):
            candidate=graded_w(name=='w8')
            stack.enter_context(patch.object(rcs,'_assemble_linear_hypersingular_matrix',candidate))
            stack.enter_context(patch.object(ops,'_assemble_linear_hypersingular_matrix',candidate))
        yield
