"""Bounded operator-storage sampling experiment; no full assembly or solve."""
import os,sys,json,time,math
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='2'
os.environ['GHOST_ASSEMBLY_THREADS']='4'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/GHOST/Backend'))
import numpy as np
import rcs_solver as rcs
from solver_quality import scale_snapshot_panel_density
from compressed_oracle import PreparedOracle
from compressed_operator import StreamedOperator,tile_payload
from multi_region import dof_coordinates
from sweep_compression import _qr_basis
from geometry_io import parse_geometry,build_geometry_snapshot

title,segs,ibcs,dielectrics=parse_geometry((ROOT/'airfoil.geo').read_text())
s=build_geometry_snapshot(title,segs,ibcs,dielectrics)
freq=float(sys.argv[1]) if len(sys.argv)>1 else 10.
fine=scale_snapshot_panel_density(s,1.5)
fine['_2d_certification_refinement_factor']=1.5
fine['_2d_certification_base_segment_n']=[v['properties'][1] for v in s['segments']]
materials=rcs.MaterialLibrary.from_entries(s['ibcs'],s['dielectrics'],str(ROOT))
wave,_,_=rcs._mesh_wavelength_for_snapshot(s,materials,freq)
k0=2*math.pi*freq*1e9/rcs.C0
panels=rcs._build_panels(fine,.0254,wave,max_panels=100000)
infos=rcs._build_coupled_panel_info(panels,materials,freq,'TE',k0)
mesh,_=rcs._build_linear_mesh_interface_aware(panels,infos)
oracle=PreparedOracle(mesh,infos,'TE',cut=32)
op=StreamedOperator(oracle,dof_coordinates(mesh,oracle.layout),tile=512,assemble=False,budget=8*1024**3)
g=len(op.groups);rng=np.random.RandomState(1904);rows=[];start=time.perf_counter()
estimate=op.bytes+sum(16*len(ids)**2 for ids in op.groups)
reserved=estimate
for low in [2**i for i in range(int(math.ceil(math.log(g,2))))]:
    high=min(2*low,g);gaps=np.arange(low,high);cum=np.cumsum(g-gaps)
    if not len(cum):continue
    ratios=[];ranks=[]
    for picked in rng.choice(int(cum[-1])*2,min(6,int(cum[-1])*2),replace=False):
        direction=int(picked>=cum[-1]);q=int(picked%cum[-1]);index=int(np.searchsorted(cum,q,side='right'))
        gap=int(gaps[index]);i=q-int(cum[index-1] if index else 0);j=i+gap
        if direction:i,j=j,i
        raw,tail=oracle.get_with_error(op.groups[i],op.groups[j])
        size=raw.nbytes
        payload,_,_,_=tile_payload(raw,tail,1e-14,'qr')
        ratios.append(sum(v.nbytes for v in payload if v is not None)/size)
        q,r=_qr_basis(raw,1e-6*max(np.linalg.norm(raw),1e-300));ranks.append(q.shape[1])
    # Balanced leaf sizes differ by at most one. Weight each band by exact
    # entry count from group lengths, with no global coefficient allocation.
    lengths=np.array([len(ids) for ids in op.groups])
    entries=sum(int(np.dot(lengths[:-gap],lengths[gap:])) for gap in gaps)*2
    mean=float(np.mean(ratios));sem=float(np.std(ratios,ddof=1)/np.sqrt(len(ratios))) if len(ratios)>1 else 0.
    estimate+=entries*16*mean
    reserved+=entries*16*min(1.,mean+2*sem)
    rows.append(dict(low=low,high=high,ratios=ratios,ranks=ranks,entries=entries))
    print(rows[-1],flush=True)
result=dict(frequency=freq,dofs=op.n,groups=g,seconds=time.perf_counter()-start,
    estimated_operator_bytes=estimate,reserved_operator_bytes=reserved,bands=rows)
(Path(__file__).parent/('memory-sample-'+str(freq)+'.json')).write_text(json.dumps(result,indent=2))
print(result['seconds'],estimate/1024**3,reserved/1024**3,flush=True)
