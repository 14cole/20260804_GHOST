"""Independent field checks, construction budgets and source/diff inventory."""
from pathlib import Path
import json,hashlib,ast,difflib
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
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
result=dict(materials={},production={},streamed={})
native=read(HERE/'native-matrix-results.json')
assert sum(v.get('accepted',False) for v in native)==22
assert {(v['kind'],v['pol']) for v in native if 'rejected' in v}=={
    ('mixed_sheet','TE'),('thin','TE'),('thin_magnetic','TE'),('thin_magnetic','TM')}
assert sum('skipped' in v for v in native)==2
result['native_matrix_accepted']=22
for mode in ('dense','hierarchical'):
    rows=read(HERE/('materials-updated-{}-r1.json'.format(mode)))
    assert len(rows)==14
    result['materials'][mode]={r['kind']:compare(r,read(PRIOR/'systems'/(r['kind']+'-n384-f0.6')/'public-result.json')) for r in rows}
baseline=read(HERE/'airfoil-f2.0-baseline-dense-r1.json')[0]
for path in HERE.glob('airfoil-f2.0-*-r*.json'):
    r=read(path)[0]
    result['production'][path.name]=dict(seconds=r['seconds'],errors=compare(r,baseline),
        peak_mib=r['metadata']['runtime_profile']['sampled_peak_process_rss_bytes']/2**20,
        sweep_sha256=r['source_hashes']['sweep_compression.py'])
for frequency,tile in ((1.,256),(2.,512)):
    for pol in ('TE','TM'):
        name='streamed-f{}-{}-tile{}-cut32.0-qr.json'.format(frequency,pol,tile)
        r=read(HERE/name)
        assert r['accepted'] and r['construction_inputs_released'],r
        assert r['original_residual_bound']<=1e-12 and r['independent_backward']<=1e-12 and r['field_error']<=1e-10
        assert r['operator']['geometry_coefficients']==r['operator']['unknowns']**2
        assert r['combined_retained_bytes']<=r['combined_payload_budget']
        result['streamed'][name]=r
        if frequency==2:
            b=read(HERE/'baseline_experiment'/name.replace('-qr',''))
            result['streamed'][name]['baseline']=b
hashes={};diff=[];changed=[]
for path in sorted((ROOT/'tools/GHOST/Backend').glob('*.py')):
    hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    before=HERE/'baseline_backend'/path.name
    old=before.read_text(encoding='utf-8-sig') if before.exists() else ''
    new=path.read_text(encoding='utf-8-sig')
    if old!=new:
        changed.append(path.name)
        ast.parse(new,feature_version=(3,6))
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='baseline/'+path.name,tofile='updated/'+path.name))
for path in HERE.glob('*.py'):ast.parse(path.read_text())
result['changed_backend_files']=changed
(HERE/'final-source-hashes.json').write_text(json.dumps(hashes,indent=2))
(HERE/'hardening.diff').write_text(''.join(diff))
(HERE/'validated-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result['production'],indent=2))
print('PASS: 14 material families, four streamed cases, budgets and changed-source Python 3.6 syntax.')
