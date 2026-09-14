"""Component-wise FMM allowances shared by the solver and execution-node scheduler."""
from ghost_backend.execution.options import option

MIB=1024**2


def forecast(nodes,dofs,regions,batch,storage_cap,resources):
    n,d=int(nodes),int(dofs)
    orders=resources.get('fmm_kernel_orders') or [64]*max(1,int(regions))
    panels=int(resources.get('panels',n))
    # With no geometric count, retain the near-assembly ceiling. Each directed
    # panel pair can feed four endpoint pairs and four interface-side blocks.
    directed=resources.get('geometric_near_pairs',2*storage_cap//2304)
    near_nnz=min(d*d,16*int(directed)*len(orders))
    near_bytes=20*near_nnz+8*(d+1)
    ilu_bytes=20*min(d*d,10*near_nnz)+16*(d+1)
    # Includes scaled CSC copies, SuperLU setup/fill, and sparse routing.
    sparse_peak=4*near_bytes+2*ilu_bytes
    krylov=16*d*(4*min(batch,32)+option('fmm_restart',80)+
        2*option('fmm_recycle_vectors',12)+20)
    # Coarse Z, AZ, Q and projected workspace coexist during setup.
    coarse=16*d*16*4+16*16*16*4
    # Persistent charge/gradient/translation data plus internal Fortran scratch.
    # Reserve the maximum native density group (8), all distinct material kernels,
    # and high-order quadrature points, rather than an allowance per boundary node.
    quadrature_points=panels*sum(orders)
    native=4096*quadrature_points
    parts=dict(near_operator_bytes=int(storage_cap),sparse_preconditioner_peak_bytes=sparse_peak,
        krylov_bytes=krylov,coarse_workspace_bytes=coarse,native_workspace_bytes=native,
        process_allowance_bytes=128*MIB)
    return dict(method='fmm_workspace_allowance',version=2,peak_bytes=sum(parts.values()),
        components=parts,unknowns=d,dense_matrix_bytes=0,
        sparse_near_nnz_allowance=near_nnz,ilu_fill_factor=10,
        quadrature_points=quadrature_points,native_density_group=8,
        estimate_semantics='Conservative allowances; native internal allocations are not a hard allocator cap.')
