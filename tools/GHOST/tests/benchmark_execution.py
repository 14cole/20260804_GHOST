"""Measure isolated PEC/mixed sweeps and compare matching runtime baselines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

SCHEMA = 'ghost.execution-benchmark.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def worker(args):
    sys.path.insert(0, str(args.backend.resolve()))
    from ghost_backend.twod import solver
    if Path(solver.__file__).resolve().parents[2] != args.backend.resolve():
        raise RuntimeError('Imported solver does not belong to the requested backend.')
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_limits, threadpool_info
    from test_experimental_cpu import fixture, fields
    from ghost_backend.execution.options import validate_options
    profile = validate_options(dict(factorization=args.mode, blas_threads=args.blas_threads,
        assembly_threads=args.assembly_threads, compressed_storage_mib=args.storage_mib))
    snapshot = fixture(args.material, args.panels)
    frequencies, angles = [.6], list(range(361))
    function = solver.solve_monostatic_rcs_2d_certified if args.certified else solver.solve_monostatic_rcs_2d_survey
    import inspect
    kwargs = dict(geometry_units='meters', solver_method='experimental_cpu')
    if 'execution_options' in inspect.signature(function).parameters:
        kwargs['execution_options'] = profile
    else:
        os.environ.update(GHOST_CPU_FACTORIZATION=args.mode, GHOST_CPU_RHS_COMPRESSION='auto',
            GHOST_COMPRESSED_STORAGE_MIB=str(args.storage_mib), GHOST_CPU_ANGLE_BATCH_SIZE='256')
        solver.set_assembly_threads(args.assembly_threads)
    with threadpool_limits(limits=args.blas_threads, user_api='blas'):
        result = function(snapshot, frequencies, angles, **kwargs)
        native = [{k: row.get(k) for k in ('internal_api', 'version', 'architecture', 'num_threads')}
                  for row in threadpool_info() if row['user_api'] == 'blas']
    meta = result['metadata']
    field_values = {pol: [[float(x.real), float(x.imag)] for x in fields(result, pol)] for pol in ('VV', 'HH')}
    sources = {p.relative_to(args.backend).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((args.backend / 'ghost_backend').rglob('*.py'))}
    report = dict(schema=SCHEMA, material=args.material, profile=profile,
        workload=dict(snapshot_sha256=digest(snapshot), frequencies_ghz=frequencies,
                      angles_deg=angles, certified=args.certified),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         system=platform.platform(), processor=platform.processor(),
                         cpu_count=os.cpu_count(), native=native),
        source_sha256=digest(sources), runtime=meta['runtime_profile'], fields=field_values,
        mesh_certified=meta.get('mesh_convergence_certified', False),
        solver_method=meta.get('solver_method'))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


def field_error(actual, reference):
    maximum = 0.0
    for pol in ('VV', 'HH'):
        a, b = actual[pol], reference[pol]
        if len(a) != len(b):
            raise ValueError('Field sample counts differ.')
        denominator = max(max(abs(complex(*x)) for x in b), 1e-280)
        maximum = max(maximum, max(abs(complex(*x)-complex(*y)) for x,y in zip(a,b))/denominator)
    return maximum


def summarize(samples):
    first = samples[0]
    for sample in samples[1:]:
        for key in ('profile', 'workload', 'environment', 'source_sha256'):
            if sample[key] != first[key]:
                raise ValueError('Repeat changed {}.'.format(key))
        if field_error(sample['fields'], first['fields']) > 1e-8:
            raise ValueError('Repeated solve fields changed.')
    runtime = [sample['runtime'] for sample in samples]
    peaks = [row['sampled_peak_process_rss_bytes'] for row in runtime]
    return dict(first, repeats=len(samples),
        median_wall_seconds=statistics.median(row['wall_seconds'] for row in runtime),
        median_peak_rss_bytes=statistics.median(peaks) if all(x is not None for x in peaks) else None,
        measurements=runtime)


def compare(current, baseline, time_tolerance, ram_tolerance):
    if current['schema'] != SCHEMA or baseline.get('schema') != SCHEMA:
        raise ValueError('Benchmark schemas differ.')
    if set(current['cases']) != set(baseline['cases']):
        raise ValueError('Benchmark case sets differ.')
    comparisons = {}
    passed = True
    for name, case in current['cases'].items():
        old = baseline['cases'][name]
        for key in ('profile', 'workload', 'environment'):
            if case[key] != old[key]:
                raise ValueError('{} has a different {}; remeasure a matching baseline.'.format(name, key))
        ratios = {}
        for key, tolerance in (('median_wall_seconds', time_tolerance), ('median_peak_rss_bytes', ram_tolerance)):
            before, after = old[key], case[key]
            if before is None or after is None or before <= 0:
                raise ValueError('{} lacks usable {} measurements.'.format(name, key))
            ratios[key] = after / before
            passed = passed and ratios[key] <= 1+tolerance
        error = field_error(case['fields'], old['fields'])
        passed = passed and error <= 1e-8
        comparisons[name] = dict(ratios=ratios, complex_field_error=error)
    return dict(passed=passed, cases=comparisons)


def suite(args):
    cases = {}
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith('GHOST_'):
            env.pop(key)
    env.update(OPENBLAS_NUM_THREADS=str(args.blas_threads), MKL_NUM_THREADS=str(args.blas_threads),
               OMP_NUM_THREADS='1', GHOST_ASSEMBLY_THREADS=str(args.assembly_threads))
    with tempfile.TemporaryDirectory(prefix='ghost-benchmark-') as directory:
        for material in ('pec', 'mixed'):
            for mode in ('dense', 'compressed'):
                samples = []
                for repeat in range(args.repeats):
                    output = Path(directory) / 'sample.json'
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                        '--backend', str(args.backend.resolve()), '--output', str(output),
                        '--material', material, '--mode', mode, '--panels', str(args.panels),
                        '--blas-threads', str(args.blas_threads), '--assembly-threads', str(args.assembly_threads),
                        '--storage-mib', str(args.storage_mib)]
                    if args.certified:
                        command.append('--certified')
                    process = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             universal_newlines=True, timeout=args.timeout)
                    if process.returncode:
                        raise RuntimeError('{} {} failed:\n{}'.format(material, mode, process.stdout))
                    sample = json.loads(output.read_text(encoding='utf-8'))
                    if args.certified and not sample['mesh_certified']:
                        raise ValueError('Mesh certification failed for {} {}.'.format(material, mode))
                    samples.append(sample)
                    print('{} {} {}/{}: {:.3f}s'.format(material, mode, repeat+1, args.repeats,
                                                       sample['runtime']['wall_seconds']), flush=True)
                cases[material + '/' + mode] = summarize(samples)
            error = field_error(cases[material+'/compressed']['fields'], cases[material+'/dense']['fields'])
            cases[material+'/compressed']['dense_complex_field_error'] = error
            if error > 1e-8:
                raise ValueError('{} compressed fields differ from dense: {}'.format(material, error))
    result = dict(schema=SCHEMA, cases=cases,
        measurement_notes='Fresh process per sample; solver wall time excludes imports. RSS sampled every 50 ms, includes runtime. Nested stage times overlap.')
    if args.baseline:
        result['comparison'] = compare(result, json.loads(args.baseline.read_text(encoding='utf-8')),
                                       args.time_tolerance, args.ram_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return 0 if result.get('comparison', {}).get('passed', True) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', type=Path, default=Path(__file__).resolve().parents[1]/'Backend')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--panels', type=int, default=256, help='Panels per circular boundary')
    parser.add_argument('--blas-threads', type=int, default=1)
    parser.add_argument('--assembly-threads', type=int, default=1)
    parser.add_argument('--storage-mib', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--certified', action='store_true')
    parser.add_argument('--time-tolerance', type=float, default=.25)
    parser.add_argument('--ram-tolerance', type=float, default=.15)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--material', choices=('pec', 'mixed'), default='pec', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=('dense', 'compressed'), default='dense', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (args.backend / 'ghost_backend/twod/solver.py').is_file():
        parser.error('Backend must contain ghost_backend/twod/solver.py.')
    if min(args.repeats, args.panels, args.blas_threads, args.assembly_threads) < 1:
        parser.error('Repeat, panel, and thread counts must be positive.')
    if min(args.time_tolerance, args.ram_tolerance) < 0 or args.timeout <= 0:
        parser.error('Tolerances must be nonnegative and timeout positive.')
    if args.worker:
        worker(args)
        return 0
    return suite(args)


if __name__ == '__main__':
    sys.exit(main())
