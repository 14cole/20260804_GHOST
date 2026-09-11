"""Validate production results against this turn's baseline and earlier fixtures."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'ghost_ram_time_20260910'


def read(name):return json.loads((HERE/name).read_text())
def amplitudes(record):
    if 'fields' in record:
        return {p:np.asarray(v)[:,0]+1j*np.asarray(v)[:,1] for p,v in record['fields'].items()}
    return {p:np.asarray([complex(v['rcs_amp_real'],v['rcs_amp_imag']) for v in rows])
            for p,rows in record['co_solved_samples'].items()}
def compare(actual,reference):
    a,b=amplitudes(actual),amplitudes(reference)
    assert actual['metadata']['quality_gate']['passed']
    errors={p:float(np.max(abs(a[p]-b[p]))/max(np.max(abs(b[p])),1e-300)) for p in b}
    assert all(np.isfinite(e) and e<1e-10 for e in errors.values()),errors
    return errors


def main():
    result={}
    materials=read('materials-updated-dense-r1.json')
    assert len(materials)==14
    result['materials']={}
    for record in materials:
        reference=json.loads((PRIOR/'systems'/(record['kind']+'-n384-f0.6')/'public-result.json').read_text())
        result['materials'][record['kind']]=compare(record,reference)
    hierarchical=HERE/'materials-updated-hierarchical-r1.json'
    if hierarchical.exists():
        result['hierarchical_materials']={}
        records=read(hierarchical.name)
        assert len(records)==14
        for record in records:
            reference=json.loads((PRIOR/'systems'/(record['kind']+'-n384-f0.6')/'public-result.json').read_text())
            result['hierarchical_materials'][record['kind']]=compare(record,reference)
    result['benchmarks']={}
    for kind,freq in (('airfoil',2.),('ibc4096',.6)):
        before=read('{}-f{}-baseline-dense-r1.json'.format(kind,freq))[0]
        for path in HERE.glob('{}-f{}-*.json'.format(kind,freq)):
            record=read(path.name)[0]
            errors=compare(record,before)
            result['benchmarks'][path.name]=dict(seconds=record['seconds'],field_errors=errors,
                peak_mib=record['metadata']['runtime_profile']['sampled_peak_process_rss_bytes']/2**20,
                systems=record['metadata']['experimental_cpu']['systems'])
    (HERE/'validated-results.json').write_text(json.dumps(result,indent=2))
    print('PASS: all 14 material fixtures and completed full-run benchmarks match independent references.')
    print(json.dumps({p:{k:v for k,v in r.items() if k!='systems'} for p,r in result['benchmarks'].items()},indent=2))


if __name__=='__main__':main()
