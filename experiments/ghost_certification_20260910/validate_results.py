"""Check saved public results and produce a compact, reproducible evidence index."""
from pathlib import Path
import json,hashlib,difflib,sys,argparse
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BACKEND=ROOT/'tools/GHOST/Backend'
PRIOR=HERE.parent/'ghost_ram_time_20260910/systems'


def field_error(actual,expected):
    a=np.asarray(actual);b=np.asarray(expected)
    assert a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    peak=max(float(np.max(np.linalg.norm(b,axis=1))),1e-300)
    return float(np.max(np.linalg.norm(a-b,axis=1))/peak)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete',action='store_true',help='Summarize partial evidence without declaring qualification complete.')
    args=parser.parse_args()
    sys.path.insert(0,str(BACKEND))
    import geometry_io
    current=geometry_io.build_geometry_snapshot(*geometry_io.parse_geometry((ROOT/'airfoil.geo').read_text()))
    recorded=json.loads((HERE.parent/'ghost_final_20260910/inputs.json').read_text())['airfoil']
    assert json.dumps(current,sort_keys=True)==json.dumps(recorded,sort_keys=True),'airfoil.geo differs from the measured snapshot'
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BACKEND.glob('*.py'))}
    summaries=[];comparisons=[];runs={}
    for path in sorted(HERE.glob('*-a361-r*.json')):
        data=json.loads(path.read_text())
        for run in data['runs']:
            entry=dict(file=path.name,kind=run['kind'],factor=run['factor'],frequency=run['frequency'],
                       certified=run['certified'],accepted=run.get('accepted',False),seconds=run['seconds'])
            entry['source_changes_since_run']=[k for k,v in hashes.items() if data['source_hashes'].get(k)!=v]
            if 'metadata' in run:
                m=run['metadata'];profile=m['runtime_profile']
                assert run['accepted'] and m['quality_gate']['passed']
                assert not run['certified'] or m['mesh_convergence_certified']
                assert set(run['fields'])=={'VV','HH'}
                assert all(len(v)==361 for v in run['fields'].values())
                entry.update(peak_rss_bytes=profile['sampled_peak_process_rss_bytes'],
                             stages=profile['stage_seconds'],method=m['solver_method'],
                             residual=m['residual_norm_max'],condition=m['condition_est_max'],
                             mesh_certified=m['mesh_convergence_certified'],
                             phase_counters=m.get('certification_phase_counters'))
                entry['factors']=[dict(unknowns=e['unknowns'],factorizations=e['factorizations'],
                     backward_error=e['max_backward_error'],gmres_columns=e['gmres_columns'],
                     operator=e['compressed'],inverse=e['preconditioners'],sweep=e.get('sweep_compression'))
                     for e in m.get('experimental_cpu',{}).get('systems',[]) if 'compressed' in e]
                if run['certified']:
                    entry['mesh']={c:{k:v[k] for k in ('complex_rms_normalized','complex_max_normalized',
                        'rms_db','max_abs_db','phase_rms_deg','phase_max_deg','base_panel_count','fine_panel_count')}
                        for c,v in m['mesh_convergence']['channels'].items()}
                runs[path.name,run['kind']]=run
            else:entry['rejected']=run.get('rejected')
            summaries.append(entry)

    for (name,kind),actual in runs.items():
        if actual['factor']!='compressed':continue
        reference_name=name.replace('-compressed-','-dense-')
        # Multiple qualification repeats share the same first dense reference.
        reference_name=reference_name.rsplit('-r',1)[0]+'-r1.json'
        expected=runs.get((reference_name,kind))
        if expected is not None:
            values={c:field_error(actual['fields'][c],expected['fields'][c]) for c in ('VV','HH')}
            assert max(values.values())<1e-10,(name,kind,values)
            comparisons.append(dict(actual=name,kind=kind,reference=reference_name,field_error=values))
        elif not actual['certified'] and kind!='airfoil':
            path=PRIOR/(kind+'-n384-f0.6')/'public-result.json'
            reference=json.loads(path.read_text())
            values={c:field_error(actual['fields'][c],[[v['rcs_amp_real'],v['rcs_amp_imag']]
                        for v in reference['co_solved_samples'][c]]) for c in ('VV','HH')}
            assert max(values.values())<1e-10,(name,kind,values)
            comparisons.append(dict(actual=name,kind=kind,reference=str(path.relative_to(ROOT)),field_error=values))

    required=('airfoil-f1.0-compressed-certified-a361-r1.json',
              'airfoil-f2.0-compressed-certified-a361-r1.json',
              'airfoil-f2.0-dense-certified-a361-r1.json',
              'airfoil-f10.0-compressed-certified-a361-r1.json',
              'materials-compressed-certified-a361-r2.json',
              'materials-dense-certified-a361-r1.json')
    missing=[name for name in required if not (HERE/name).exists()]
    expected_kinds={'pec','ibc','pec_ibc','lossless','lossy','magnetic','coated','layered',
                    'mixed','sheet','mixed_sheet','thin','thin_magnetic','transparent'}
    for name in required[-2:]:
        if name not in missing:assert {kind for file,kind in runs if file==name}==expected_kinds
    (HERE/'validated-results.json').write_text(json.dumps(dict(runs=summaries,comparisons=comparisons,
        required_results_missing=missing,qualification_complete=not missing and all(v['accepted'] for v in summaries),
        airfoil_snapshot_matches_current_geometry=True,
        field_limit=1e-10,airfoil_sha256=hashlib.sha256((ROOT/'airfoil.geo').read_bytes()).hexdigest()),indent=2))
    (HERE/'final-source-hashes.json').write_text(json.dumps(hashes,indent=2))
    diff=[]
    for path in sorted(BACKEND.glob('*.py')):
        baseline=HERE/'baseline_backend'/path.name
        before=baseline.read_text().splitlines(True) if baseline.exists() else []
        after=path.read_text().splitlines(True)
        diff.extend(difflib.unified_diff(before,after,fromfile='baseline/'+path.name,tofile='Backend/'+path.name))
    (HERE/'certification.diff').write_text(''.join(diff))
    print(json.dumps(dict(saved_runs=len(summaries),accepted=sum(v['accepted'] for v in summaries),
        comparisons=len(comparisons),max_complex_field_error=max([max(v['field_error'].values()) for v in comparisons] or [0.])),indent=2))
    if missing:print('Required results missing: '+', '.join(missing))
    return 0 if all(v['accepted'] for v in summaries) and (not missing or args.allow_incomplete) else 1


if __name__=='__main__':sys.exit(main())
