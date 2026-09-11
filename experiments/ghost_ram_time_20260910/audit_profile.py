from common import *
import argparse
import cProfile
import pstats
from unittest.mock import patch
import solver_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind')
    parser.add_argument('--count', type=int, default=384)
    parser.add_argument('--frequency', type=float, default=.6)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    ops.set_assembly_threads(args.threads)
    label = 'profile-{}-n{}-f{}-t{}'.format(args.kind,args.count,args.frequency,args.threads)
    # cProfile only measures the main Python thread; disable background sampler.
    def start(metrics):
        metrics.started = time.perf_counter(); metrics.sample_memory()
    profile = cProfile.Profile()
    with patch.object(solver_metrics.SolveMetrics,'start',start):
        profile.enable()
        result = public_solve(args.kind,args.count,args.frequency)
        profile.disable()
    profile.dump_stats(str(HERE/(label+'.prof')))
    with (HERE/(label+'.txt')).open('w') as stream:
        pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(65)
        pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('tottime').print_stats(40)
    write(label+'.json',result['metadata'])
    print(label,result['metadata']['runtime_profile']['wall_seconds'],flush=True)

if __name__ == '__main__': main()
