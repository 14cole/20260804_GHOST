"""Independent qualification and this turn's source inventory."""
from pathlib import Path
import json,hashlib,ast,difflib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PRIOR=HERE.parent/'ghost_ram_time_20260910'
def read(path):return json.loads(path.read_text())
def fields(record):
    if 'fields' in record:
        return {p:np.asarray(v)[:,0]+1j*np.asarray(v)[:,1] for p,v in record['fields'].items()}
    return {p:np.asarray([complex(v['rcs_amp_real'],v['rcs_amp_imag']) for v in rows])
            for p,rows in record['co_solved_samples'].items()}
def compare(actual,reference):
    a,b=fields(actual),fields(reference)
    assert actual['metadata']['quality_gate']['passed']
    result={p:float(np.max(abs(a[p]-b[p]))/max(np.max(abs(b[p])),1e-300)) for p in b}
    assert all(np.isfinite(v) and v<1e-10 for v in result.values()),result
    return result
result=dict(materials={},production={},paired={})
for mode in ('dense','hierarchical'):
    rows=read(HERE/('materials-updated-{}-r{}.json'.format(mode,2 if mode=='hierarchical' else 1)))
    assert len(rows)==14
    result['materials'][mode]={r['kind']:compare(r,read(PRIOR/'systems'/(r['kind']+'-n384-f0.6')/'public-result.json')) for r in rows}
for path in HERE.glob('airfoil-f*-*-r*.json'):
    r=read(path)[0];frequency=path.name.split('-')[1][1:]
    reference=read(PRIOR/'systems'/('airfoil-n256-f'+frequency)/'public-result.json')
    profile=r['metadata']['runtime_profile']
    result['production'][path.name]=dict(seconds=r['seconds'],errors=compare(r,reference),
        peak_mib=profile['sampled_peak_process_rss_bytes']/2**20,
        sampled_20ms_peak_mib=r.get('peak_rss_bytes',0)/2**20,
        stage_seconds=profile['stage_seconds'],
        factors={p:v.get('hierarchical_factors',[]) for p,v in r['metadata']['channel_metadata'].items()})
for frequency,repeat in ((1.,1),(2.,2)):
    name='paired-f{}-tile512-tol1e-06-spool-r{}.json'.format(frequency,repeat)
    r=read(HERE/name)
    assert r['accepted'] and r['oracle_released'],r
    for v in r['channels'].values():
        assert v['operator_released'] and v['field_error']<1e-10 and v['independent_backward']<1e-12
        assert v['solver']['original_residual_bound']<1e-12
        assert v['solver']['combined_bytes']<=r['budget']
        assert v['operator']['geometry_coefficients']==v['operator']['unknowns']**2
    r['core_seconds']=r['preparation_seconds']+r['paired_assembly_seconds']+sum(
        v['load_seconds']+v['factor_seconds']+v['rhs_solve_projection_seconds'] for v in r['channels'].values())
    r['core_peak_mib']=r['sampled_core_rss_bytes']/2**20
    result['paired'][name]=r
native=read(HERE/'native-results.json')
assert len(native)==28 and sum(r.get('accepted',False) for r in native)==26
assert all(r.get('accepted') or (r['kind']=='transparent' and 'skipped' in r) for r in native)
result['native']=dict(accepted=26,transparent=2,max_field_error=max(r.get('field_error',0) for r in native),
    max_backward=max(r.get('backward',0) for r in native),
    payload_ratios={r['kind']+'-'+r['pol']:r['evidence']['combined_bytes']/r['dense_pair_bytes'] for r in native if r.get('accepted')})
hashes={};diff=[];changed=[]
for path in sorted((ROOT/'tools/GHOST/Backend').glob('*.py')):
    hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    before=HERE/'baseline_backend'/path.name
    old=before.read_text(encoding='utf-8-sig') if before.exists() else ''
    new=path.read_text(encoding='utf-8-sig')
    if old!=new:
        changed.append(path.name);ast.parse(new,feature_version=(3,6))
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='baseline/'+path.name,tofile='updated/'+path.name))
for path in HERE.glob('*.py'):ast.parse(path.read_text())
assert not list(HERE.glob('ghost-tm-*.bin'))
result['changed_backend_files']=changed
(HERE/'final-source-hashes.json').write_text(json.dumps(hashes,indent=2))
(HERE/'final.diff').write_text(''.join(diff))
(HERE/'validated-results.json').write_text(json.dumps(result,indent=2))
print('PASS: 14 material families under both public factors; 26 native compressed systems; 1/2 GHz paired sweeps; fields, bounds, budgets, lifetimes and Python 3.6 Backend syntax.')
print('Changed Backend:',changed)
