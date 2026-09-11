"""Compare revised forecasts with saved independent RSS/operator measurements."""
import os,sys,json,time,math,hashlib
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='2'
os.environ.update(GHOST_CPU_FACTORIZATION='compressed',GHOST_DENSE_BACKEND='cpu',
                  GHOST_COMPRESSED_STORAGE_MIB='8192',GHOST_ASSEMBLY_THREADS='4')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tools/GHOST/Backend'))
import rcs_solver as rcs
from hpc_scheduler import _resource_records_for_frequency, predict_2d_resources_many
from solver_quality import scale_snapshot_panel_density
from compressed_memory import forecast,GIB
inputs=json.loads((ROOT/'experiments/ghost_final_20260910/inputs.json').read_text())
references=json.loads((HERE/'materials-compressed-certified-a361-r2.json').read_text())['runs']
references+=json.loads((HERE/'airfoil-f2.0-compressed-certified-a361-r1.json').read_text())['runs']
records=[]
for ref in references:
    kind=ref['kind'];frequency=ref['frequency'];snapshot=inputs[kind]
    started=time.perf_counter()
    fine=scale_snapshot_panel_density(snapshot,1.5)
    fine['_2d_certification_refinement_factor']=1.5
    fine['_2d_certification_base_segment_n']=[s['properties'][1] for s in snapshot['segments']]
    materials=rcs.MaterialLibrary.from_entries(snapshot.get('ibcs',[]),snapshot.get('dielectrics',[]),str(ROOT))
    raw=_resource_records_for_frequency(rcs,fine,materials,frequency,[('VV','TE'),('HH','TM')],
        .0254 if kind=='airfoil' else 1.,100000)
    measured=ref['metadata']['runtime_profile']['sampled_peak_process_rss_bytes']
    systems=ref['metadata'].get('experimental_cpu',{}).get('systems',[])[-2:]
    plans={}
    for pol,resources in raw.items():
        if resources.get('analytic_zero'):
            plan=dict(peak_bytes=int(.6*GIB),operator_bytes=0,operator_allowance_bytes=0,
                      temporary_disk_bytes=0,samples=0)
        else:
            plan=forecast(resources['nodes'],resources['system_dofs'],361,256,4,8*GIB,
                          resources,safety=1.35,floor_gb=.6)
        if systems:
            actual=systems[0 if pol=='VV' else 1]['compressed']['retained_bytes']
            plan['measured_operator_bytes']=actual
            assert plan['operator_allowance_bytes']>=actual,(kind,pol,plan['operator_allowance_bytes'],actual)
        plans[pol]=plan
    peak=max(v['peak_bytes'] for v in plans.values())
    assert peak>=measured,(kind,peak,measured)
    records.append(dict(kind=kind,frequency=frequency,planning_seconds=time.perf_counter()-started,
        measured_peak_bytes=measured,scheduled_peak_bytes=peak,plans=plans))
    print(kind,frequency,'scheduled GiB',round(peak/GIB,4),'measured',round(measured/GIB,4),
          'planning s',round(records[-1]['planning_seconds'],3),flush=True)
started=time.perf_counter()
plans=predict_2d_resources_many(str(ROOT/'airfoil.geo'),[10.],['VV','HH'],'inches',100000,
    fine_factor=1.5,n_angles=361,solver_method='experimental_cpu')
airfoil=dict(seconds=time.perf_counter()-started,plans={p:v for (_,p),v in plans.items()})
(HERE/'memory-new-airfoil10.json').write_text(json.dumps(airfoil,indent=2))
ref=json.loads((HERE/'airfoil-f10.0-compressed-certified-a361-r1.json').read_text())['runs'][0]
for index,pol in enumerate(('VV','HH')):
    plan=airfoil['plans'][pol]['memory_estimate']
    measured=ref['metadata']['experimental_cpu']['systems'][-2+index]['compressed']['retained_bytes']
    assert plan['operator_allowance_bytes']>=measured
    plan['measured_operator_bytes']=measured
record=dict(kind='airfoil',frequency=10.,planning_seconds=airfoil['seconds'],
    measured_peak_bytes=ref['metadata']['runtime_profile']['sampled_peak_process_rss_bytes'],
    scheduled_peak_bytes=max(v['memory_estimate']['peak_bytes'] for v in airfoil['plans'].values()),
    plans={p:v['memory_estimate'] for p,v in airfoil['plans'].items()})
assert record['scheduled_peak_bytes']>=record['measured_peak_bytes']
records.append(record)
print('airfoil 10 GHz scheduled GiB',record['scheduled_peak_bytes']/GIB,
      'planning seconds',record['planning_seconds'],flush=True)
out=dict(passed=True,runs=records,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT/'tools/GHOST/Backend').glob('*.py')})
(HERE/'memory-forecast-validation.json').write_text(json.dumps(out,indent=2))
print('All 16 workloads and both polarized operator allowances cover saved measurements.',flush=True)
