"""Bound node-space sweep work independently of point-space FMM batches."""
from ghost_backend.execution.options import option, allocated_memory_budget


def sweep_workspace_bytes():
    budgets=[b for b in (option('ram_budget_gib'),allocated_memory_budget()) if b is not None]
    return int(min(64*1024**2, .05*min(budgets)*1024**3)) if budgets else 64*1024**2


def rhs_batch_size(unknowns,requested):
    # RHS, row scaling, QR proposal/recovery, solution, residual and temporaries.
    return max(1,min(int(requested),256,sweep_workspace_bytes()//(16*max(1,unknowns)*16)))


def rhs_basis_capacity(unknowns):
    return max(0,min(256,sweep_workspace_bytes()//(4*16*max(1,unknowns))))


def geometry_resources(mesh,infos):
    """Count actual close pairs without constructing sparse matrices or a plan."""
    import math
    from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
    from ghost_backend.twod.fmm.galerkin import near_pairs
    from ghost_backend.compressed.runtime import storage_budget
    g=AssemblyGeometry(mesh)
    cap=storage_budget()
    try:
        count=near_pairs(g,cap,count_only=True)
        exceeded=False
    except MemoryError:
        count=cap//(576*g.node_ids.shape[1]**2)+1
        exceeded=True
    wave_numbers={complex(getattr(i,'k_'+side)) for i in infos for side in ('minus','plus')
                  if getattr(i,side+'_region')>=0}
    electrical=max((abs(k)*float(g.lengths.max()) for k in wave_numbers),default=0.)
    return dict(panels=len(g.lengths),near_pairs=count,near_count_capped=exceeded,
                kernels=max(1,len(wave_numbers)),max_electrical_panel_length=electrical
                if math.isfinite(electrical) else 128.)


MIB=1024**2


def forecast(nodes,dofs,regions,batch,storage_cap,resources):
    n,d=int(nodes),int(dofs)
    import math
    g=resources.get('fmm_geometry',{})
    kernels=g.get('kernels',max(1,int(regions)))
    order=max(8,option('fmm_quadrature_order',0),math.ceil(g.get('max_electrical_panel_length',0)/2)+4)
    orders=resources.get('fmm_kernel_orders') or [order if g else 64]*kernels
    orders=[max(q,option('fmm_quadrature_order',0)) for q in orders]
    panels=int(g.get('panels',resources.get('panels',n)))
    # With no geometric count, retain the near-assembly ceiling. Each directed
    # panel pair can feed four endpoint pairs and four interface-side blocks.
    directed=resources.get('geometric_near_pairs')
    if directed is None:
        # A capped prefix may not yet include every self panel. Keep the known
        # diagonal floor instead of deriving a negative directed-pair count.
        directed=max(panels,2*g['near_pairs']-panels) if g else max(panels,2*storage_cap//2304)
    width=int(resources.get('basis_width',2))
    near_nnz=min(d*d,4*width*width*int(directed)*len(orders))
    near_bytes=20*near_nnz+8*(d+1)
    ilu_bytes=20*min(d*d,10*near_nnz)+16*(d+1)
    # Includes scaled CSC copies, SuperLU setup/fill, and sparse routing.
    sparse_peak=4*near_bytes+2*ilu_bytes
    batch=rhs_batch_size(d,batch)
    recycle=option('fmm_recycle_vectors','auto')
    recycle=12 if recycle=='auto' else recycle
    krylov=16*d*(option('fmm_restart',80)+2*recycle+20)
    sweep=16*d*(16*batch+2*rhs_basis_capacity(d))
    # Coarse Z, AZ, Q and projected workspace coexist during setup.
    coarse=16*d*16*4+16*16*16*4
    # Persistent charge/gradient/translation data plus internal Fortran scratch.
    # Reserve the maximum native density group (8), all distinct material kernels,
    # and high-order quadrature points, rather than an allowance per boundary node.
    quadrature_points=panels*(sum(orders)+(len(orders) if resources.get('discretization')=='pulse' else 0))
    native=4096*quadrature_points
    operator_bytes=576*width**2*((directed+panels)//2)*len(orders)
    parts=dict(near_operator_bytes=operator_bytes,sparse_preconditioner_peak_bytes=sparse_peak,
        krylov_bytes=krylov,rhs_workspace_bytes=sweep,coarse_workspace_bytes=coarse,native_workspace_bytes=native,
        process_allowance_bytes=128*MIB)
    return dict(method='fmm_workspace_allowance',version=3,geometry_preflight=bool(g) or 'geometric_near_pairs' in resources,
        near_count_capped=g.get('near_count_capped',False),angle_batch_size=batch,near_storage_cap_bytes=storage_cap,peak_bytes=sum(parts.values()),
        components=parts,unknowns=d,dense_matrix_bytes=0,
        sparse_near_nnz_allowance=near_nnz,ilu_fill_factor=10,
        quadrature_points=quadrature_points,native_density_group=8,
        estimate_semantics='Conservative allowances; native internal allocations are not a hard allocator cap.')
