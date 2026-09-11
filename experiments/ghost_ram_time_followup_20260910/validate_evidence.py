"""Check saved independent comparisons and summarize follow-up measurements.

This validates experimental evidence, not production or mesh certification.
Run the generating scripts to repeat the numerical experiments themselves.
"""
from support import *


def read(name):
    return json.loads((HERE/name).read_text())


def below(value, limit):
    assert np.isfinite(value) and 0 <= value < limit, (value, limit)


def find_key(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for nested in value.values():
            found = find_key(nested, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for nested in value:
            found = find_key(nested, key)
            if found is not None:
                return found


def field_differences(candidate, reference):
    assert set(candidate['fields']) == set(reference['fields']) == {'VV', 'HH'}
    errors = {}
    for pol in ('VV', 'HH'):
        x, y = (np.asarray(r['fields'][pol]) for r in (candidate, reference))
        assert x.shape == y.shape == (361, 2)
        errors[pol] = difference(x[:, 0]+1j*x[:, 1], y[:, 0]+1j*y[:, 1])
        below(errors[pol], 1e-10)
    return errors


def main():
    assert source_hashes() == read('source-manifest.json'), 'Backend changed during experiments'
    summary = {'production_sources_unchanged': True}
    materials = read('ownership-materials.json')
    assert len(materials) == 14
    for result in materials:
        assert result['gate']
        for error in result['field_error'].values():
            below(error, 1e-12)
        for backward in result['backward']:
            below(backward, 1e-12)
    summary['ownership_materials'] = dict(cases=len(materials), max_field_error=max(
        v for r in materials for v in r['field_error'].values()))
    reference = read('ownership-ibc-baseline-n4096-t4-r1.json')
    summary['ownership_ibc'] = {}
    for mode in ('baseline', 'owned'):
        records = [read('ownership-ibc-{}-n4096-t4-r{}.json'.format(mode, i)) for i in (1, 2)]
        for r in records:
            assert r['metadata']['quality_gate']['passed']
        summary['ownership_ibc'][mode] = dict(
            median_seconds=float(np.median([r['seconds'] for r in records])),
            median_peak_mib=float(np.median([find_key(r, 'sampled_peak_process_rss_bytes')/2**20 for r in records])),
            field_errors=[field_differences(r, reference) for r in records])
    inverses = []
    for path in HERE.glob('inverse-*.json'):
        records = read(path.name)
        assert {r['mode'] for r in records} == {'periodic', 'deferred'}
        for r in records:
            assert 'rejected' not in r, (path.name, r)
            below(r['backward'], 1e-12)
            below(r['adjoint_backward'], 1e-12)
            below(r['field_error'], 1e-10)
        inverses.extend(r for r in records if r['mode'] == 'deferred')
    assert len(inverses) == 28
    summary['deferred_inverse'] = dict(cases=len(inverses),
        max_field_error=max(r['field_error'] for r in inverses),
        max_backward=max(r['backward'] for r in inverses),
        max_adjoint_backward=max(r['adjoint_backward'] for r in inverses),
        airfoil={p:read('inverse-airfoil-n256-f2.0-'+p+'.json') for p in ('TE', 'TM')})
    summary['compression'] = {}
    for pol in ('TE', 'TM'):
        record = read('schedule-airfoil-n256-f2.0-'+pol+'.json')
        assert len(record['methods']) == 4
        for r in record['methods']:
            assert 'rejected' not in r
            below(r['backward'], 1e-12)
            below(r['bound'], 1e-12)
            below(r['field_error'], 1e-10)
        summary['compression'][pol] = [r for r in record['methods'] if r['tolerance'] == 1e-13]
    reference = read('loss-assembly-cut0-r1.json')
    summary['loss_assembly'] = {}
    for cut in (0, 32):
        records = [read('loss-assembly-cut{}-r{}.json'.format(cut, i)) for i in (1, 2)]
        for r in records:
            assert r['metadata']['quality_gate']['passed']
        summary['loss_assembly'][str(cut)] = dict(
            median_seconds=float(np.median([r['seconds'] for r in records])),
            stats=records[0]['stats'],
            field_errors=[field_differences(r, reference) for r in records])
    audit = read('loss-assembly-cut32-r1-check.json')
    assert len(audit['checks']) == 2
    for r in audit['checks']:
        below(r['original_backward'], 1e-12)
        below(r['residual_bound'], 1e-12)
    summary['loss_assembly']['independent_checks'] = audit['checks']
    summary['loss_assembly']['audit_field_errors'] = field_differences(audit, reference)
    combined = []
    for path in sorted(HERE.glob('loss-assembly-*-table12_paired*.json')):
        record = read(path.name)
        assert record['metadata']['quality_gate']['passed']
        for check in record['checks']:
            below(check['original_backward'], 1e-12)
            below(check['residual_bound'], 1e-12)
        combined.append(dict(file=path.name,seconds=record['seconds'],
            field_errors=field_differences(record, reference),checks=record['checks'],
            stages=record['metadata']['runtime_profile']['stage_seconds']))
    summary['combination_trials'] = combined
    late_reference = read('loss-assembly-cut0-r3.json')
    assert late_reference['metadata']['quality_gate']['passed']
    summary['late_baseline'] = dict(seconds=late_reference['seconds'],
        field_errors=field_differences(late_reference, reference),
        stages=late_reference['metadata']['runtime_profile']['stage_seconds'])
    locality = read('loss-locality-f2.0.json')
    assert len(locality) == 8
    for r in locality:
        if r['optical_distance'] >= 32:
            below(r['backward'], 1e-12)
            below(r['bound'], 1e-12)
            below(r['field_error'], 1e-10)
        if r['optical_distance'] == 16:
            assert r['backward'] > 1e-12
    assert any(r['optical_distance'] == 24 and r['bound'] > 1e-12 for r in locality)
    summary['loss_locality'] = locality
    summary['residual_reuse'] = {}
    for pol in ('TE', 'TM'):
        records = read('residual-airfoil-n256-f2.0-'+pol+'.json')
        assert len(records) == 4
        for r in records:
            assert r['solution_difference'] == 0
            below(r['backward'], 1e-12)
            if r['reuse']:
                assert r['cache_stats'] == dict(hits=4, misses=0)
        summary['residual_reuse'][pol] = records
    write('validated-summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k in (
        'production_sources_unchanged', 'ownership_materials', 'ownership_ibc', 'loss_assembly')}, indent=2))
    print('PASS: 14 ownership fixtures, 28 inverse systems, compressed airfoil checks, '
          'loss tests with retained failures, exact residual reuse, and unchanged Backend hashes.')


if __name__ == '__main__':
    main()
