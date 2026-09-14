"""Reproduce certified reference/adaptive comparisons on supplied polygons."""
import argparse
import math
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from ghost_backend.tests.general_fixtures import CASES, fixture
    from ghost_backend.geometry.io import snapshot_to_geometry_text
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', choices=CASES, nargs='+', default=list(CASES))
    parser.add_argument('--scales', type=float, nargs='+', default=[1., 5.])
    parser.add_argument('--frequency', type=float, default=3.)
    parser.add_argument('--density', type=int, default=40)
    parser.add_argument('--angles', type=int, default=19)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--ram-gib', type=float, default=24.)
    parser.add_argument('--storage-mib', type=int, default=256)
    args = parser.parse_args()
    if (any(not math.isfinite(x) or x <= 0 for x in [args.frequency,args.ram_gib,*args.scales])
            or min(args.density,args.angles,args.threads,args.repeats,args.storage_mib) < 1):
        parser.error('Frequencies, scales, budgets and counts must be positive and finite.')
    args.output.mkdir(parents=True, exist_ok=True)
    vertices = dict(rectangle=4,reentrant=8,acute=4,gap=8,dielectric=4,mixed=8,sheet=2)
    failed = False
    for scale in args.scales:
        for case in args.cases:
            snapshot = fixture(case, vertices[case])
            for segment in snapshot['segments']:
                segment['properties'][1] = str(-args.density)
                segment['name'] = segment['name'].replace(' ', '_')
                for pair in segment['point_pairs']:
                    for key in ('x1','y1','x2','y2'): pair[key] *= scale
            label = '{}-scale-{:g}'.format(case, scale)
            geometry = args.output/(label+'.geo')
            geometry.write_text(snapshot_to_geometry_text(snapshot), encoding='utf-8')
            for strategy in ('global','adaptive'):
                command = [sys.executable,str(root/'scripts/benchmark_solver.py'),str(geometry.resolve()),
                    '--frequencies',str(args.frequency),'--units','meters','--certified',
                    '--mesh-strategy',strategy,'--angles',str(args.angles),'--threads',str(args.threads),
                    '--repeats',str(args.repeats),'--ram-gib',str(args.ram_gib),
                    '--storage-mib',str(args.storage_mib),'--output',
                    str((args.output/(label+'-'+strategy+'.json')).resolve())]
                status = subprocess.call(command)
                failed = failed or status != 0
                print(label, strategy, 'passed' if status == 0 else 'failed', flush=True)
    return int(failed)


if __name__ == '__main__': raise SystemExit(main())
