"""Interface-local wavelength counterfactual. This changes the mesh, not kernels."""
from common import *
from contextlib import contextmanager
from unittest.mock import patch
from collections import Counter
import argparse
import copy
import multi_region

@contextmanager
def local_wavelength(value,frequency):
    original=rcs._build_panels
    materials=rcs.MaterialLibrary.from_entries(value['ibcs'],value['dielectrics'],str(ROOT))
    global_wave=rcs._mesh_wavelength_for_snapshot(value,materials,frequency)[0]
    def build(snapshot,meters_scale,min_wavelength,max_panels=100000):
        all_panels=[]
        for index,segment in enumerate(snapshot['segments']):
            local=dict(snapshot,segments=[segment])
            if '_2d_certification_base_segment_n' in snapshot:
                local['_2d_certification_base_segment_n']=[snapshot['_2d_certification_base_segment_n'][index]]
            local_wave=rcs._mesh_wavelength_for_snapshot(local,materials,frequency)[0]
            all_panels.extend(original(local,meters_scale,min_wavelength*local_wave/global_wave,max_panels))
            if len(all_panels)>max_panels:raise ValueError('Local mesh exceeded panel limit')
        return all_panels
    with patch.object(rcs,'_build_panels',build):yield

def counts(frequency):
    value=snapshot('airfoil')
    records={}
    for mode in ('global','local'):
        from contextlib import nullcontext
        with local_wavelength(value,frequency) if mode=='local' else nullcontext():
            mesh,infos,k=prepare('airfoil',frequency=frequency)
        layout=multi_region.build_layout(mesh,infos,'TE')
        records[mode]=dict(panels=len(mesh.elements),nodes=len(mesh.nodes),dofs=layout['n_dof'],
            matrix_gib=16*layout['n_dof']**2/1024**3,
            panels_by_segment=dict(Counter(e.name for e in mesh.elements)))
    return records

def compare(frequency):
    value=snapshot('airfoil');results={};fields_by_mode={}
    from contextlib import nullcontext
    for mode in ('global','local'):
        with local_wavelength(value,frequency) if mode=='local' else nullcontext():
            t=time.perf_counter();result=public_solve('airfoil',frequency=frequency)
        fields_by_mode[mode]=np.asarray([complex(v['rcs_amp_real'],v['rcs_amp_imag'])
            for pol in ('VV','HH') for v in result['co_solved_samples'][pol]])
        results[mode]=dict(seconds=time.perf_counter()-t,metadata=result['metadata'])
        print(frequency,mode,round(results[mode]['seconds'],3),flush=True)
    results['field_peak_difference']=difference(fields_by_mode['local'],fields_by_mode['global'])
    # Independently refine the local mesh, rather than treating its small
    # linear-system residual as a certificate of discretization accuracy.
    write('mesh-fields-f{}.json'.format(frequency),results)
    try:
        with local_wavelength(value,frequency):
            certified=rcs.solve_monostatic_rcs_2d_certified(value,[frequency],np.linspace(0,360,37).tolist(),
                geometry_units='inches',solver_method='experimental_cpu',max_panels=100000,
                material_base_dir=str(ROOT))
        results['local_certification']=certified['metadata'].get('mesh_convergence',{})
        results['local_certified']=certified['metadata'].get('mesh_convergence_certified')
    except Exception as exc:
        results['local_certified']=False
        results['certification_rejection']=str(exc)
    write('mesh-fields-f{}.json'.format(frequency),results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frequency',type=float);a=p.parse_args()
    if a.frequency is not None:compare(a.frequency)
    else:write('mesh-counts.json',{str(f):counts(f) for f in (.5,1.,2.,10.)})
