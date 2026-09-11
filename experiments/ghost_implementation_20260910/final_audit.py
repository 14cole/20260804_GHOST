"""Record this turn's production diff and verify reproducible evidence."""
from pathlib import Path
import ast,difflib,hashlib,json,re
HERE=Path(__file__).resolve().parent
BACKEND=HERE.parents[1]/'tools/GHOST/Backend'
diff=[];changed=[];hashes={}
for path in sorted(BACKEND.glob('*.py')):
    hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    baseline=HERE/'baseline_backend'/path.name
    if baseline.exists() and baseline.read_bytes()!=path.read_bytes():
        changed.append(path.name)
        source=path.read_text(encoding='utf-8-sig')
        ast.parse(source,filename=str(path),feature_version=(3,6))
        diff.extend(difflib.unified_diff(baseline.read_text(encoding='utf-8-sig').splitlines(True),
            source.splitlines(True),fromfile='before/'+path.name,tofile='after/'+path.name))
(HERE/'implementation.diff').write_text(''.join(diff))
(HERE/'final-source-hashes.json').write_text(json.dumps(hashes,indent=2))
streams=[]
for freq,tile in ((1.,256),(2.,512)):
    for pol in ('TE','TM'):
        record=json.loads((HERE/'streamed-f{}-{}-tile{}-cut32.0.json'.format(freq,pol,tile)).read_text())
        assert record['accepted'] and record['construction_inputs_released'],record
        for field in ('original_residual_bound','independent_backward'):
            assert record[field]<=1e-12
        assert record['field_error']<=1e-10
        assert record['operator']['geometry_coefficients']==record['operator']['unknowns']**2
        streams.append(record)
(HERE/'validated-streamed-results.json').write_text(json.dumps(streams,indent=2))
report=(BACKEND.parent/'EFFICIENCY_IMPLEMENTATION.md').read_text()
for link in re.findall(r'\]\((C:/[^)]*)\)',report):
    assert Path(re.sub(r':\d+$','',link)).exists(),link
print('PASS: source syntax parses as Python 3.6; all four compressed airfoil cases pass; report links exist.')
print('Changed Backend files:',', '.join(changed))
