"""Bounded batch throughput planning from reusable per-frequency forecasts."""
import math
from itertools import product
from ghost_backend.execution.policy import BACKENDS, MODEL, available_candidates

# Initial relative timing prior from the paired airfoil benchmark in RUN_PROFILES.
# This is a scheduling estimate, not seconds or a hardware-independent guarantee.
COMPRESSED_COST_RATIO = 1.4


def combine_channels(plans, n_angles, fine_factor):
    from ghost_backend.hpc.scheduler import unit_cost
    cost = sum(unit_cost(p['nodes'], n_angles, fine_factor, fine_nodes=p['fine_nodes'],
        system_dofs=p['base_system_dofs'], fine_system_dofs=p['fine_system_dofs'],
        operator_matrices=p['base_operator_matrices'], fine_operator_matrices=p['fine_operator_matrices'])
        for p in plans)
    result = dict(cost=cost, peak_gb=max(p['peak_gb'] for p in plans))
    if all('backend_candidates' in p for p in plans):
        modes=[m for m in BACKENDS if all(m in p['backend_candidates'] for p in plans)]
        result['backend_candidates'] = {
            mode: dict(cost=sum(p['backend_candidates'][mode]['cost'] for p in plans)
                       if all('cost' in p['backend_candidates'][mode] for p in plans) else
                       cost*(COMPRESSED_COST_RATIO if mode == 'compressed' else 1.),
                       peak_gb=max(p['backend_candidates'][mode]['peak_gb'] for p in plans))
            for mode in modes}
    return result


def _simulate(records, choices, cores, workers, budget, options):
    """Match the driver's expensive-first, CPU/memory backfill dispatch."""
    from ghost_backend.hpc.scheduler import assembly_threads_for_unit
    pending = sorted(records, key=lambda r: (-r['backend_candidates'][choices[r['unit']]]['cost'], r['unit']))
    running = []
    clock = 0.
    while pending or running:
        remaining = []
        used_ram = sum(r[1] for r in running)
        used_cpu = sum(r[2] for r in running)
        for record in pending:
            candidate = record['backend_candidates'][choices[record['unit']]]
            ram = candidate['peak_gb']
            threads = max(options['blas_threads'], assembly_threads_for_unit(
                cores, workers, budget, ram, options['assembly_threads']))
            if len(running) < workers and (not running or
                    (used_ram+ram <= budget and used_cpu+threads <= cores)):
                running.append((clock+candidate['cost'], ram, threads))
                used_ram += ram
                used_cpu += threads
            else:
                remaining.append(record)
        pending = remaining
        clock = min(r[0] for r in running)
        running = [r for r in running if r[0] > clock]
    return clock


def select_batch_backends(records, cores, workers, budget_gb, options):
    """Choose a bounded set of whole-batch schedules on the executing node.

    Evaluate fastest-unit-first, compressed-first and mixed schedules. Admit candidates
    against both node and per-solve RAM; optimize predicted completion of the
    complete share instead of maximizing concurrency or minimizing one solve.
    No mesh builds, coefficient samples, or trial factorizations occur here.
    """
    records = [dict(r,backend_candidates=available_candidates(r['backend_candidates']))
               for r in records if 'backend_candidates' in r]
    if not records:
        return {}, {}
    cores, workers = max(1, int(cores)), max(1, min(int(workers), int(cores)))
    cap = min(float(budget_gb), float(options['ram_budget_gib'] or budget_gb))
    allowed = {}
    for r in records:
        candidates = r['backend_candidates']
        if not candidates:
            raise RuntimeError('No compatible backend is available for {} on this execution node.'.format(r['unit']))
        for mode in candidates:
            c = candidates[mode]
            if any(not math.isfinite(float(c[key])) or c[key] <= 0 for key in ('cost', 'peak_gb')):
                raise ValueError('Invalid batch resource forecast for {}.'.format(r['unit']))
        fitting = sorted([mode for mode in candidates if candidates[mode]['peak_gb'] <= cap],
                         key=lambda m:(candidates[m]['cost'],BACKENDS.index(m)))
        # Retain fail-loud progress for one oversized unit. The solver's real
        # admission gate still decides whether it can execute.
        allowed[r['unit']] = fitting or [min(candidates, key=lambda m: candidates[m]['peak_gb'])]
    dense_first = {r['unit']: allowed[r['unit']][0] for r in records}
    compressed_first = {r['unit']: 'compressed' if 'compressed' in allowed[r['unit']] else allowed[r['unit']][-1] for r in records}
    memory_first={r['unit']:min(allowed[r['unit']],key=lambda m:r['backend_candidates'][m]['peak_gb']) for r in records}
    proposals = [dense_first, compressed_first,memory_first]
    flexible = [r for r in records if len(allowed[r['unit']]) > 1]
    flexible.sort(key=lambda r: (-(max(c['peak_gb'] for c in r['backend_candidates'].values())-
                                  min(c['peak_gb'] for c in r['backend_candidates'].values())), r['unit']))
    # Eight thresholds keep search cost bounded for large sweeps.
    for step in range(1, 8):
        choice = dict(dense_first)
        for r in flexible[:len(flexible)*step//8]:
            choice[r['unit']] = memory_first[r['unit']]
        proposals.append(choice)
    # Older HPC interpreters lack math.prod. Only count up to the search cap,
    # so large sweeps also avoid constructing an unnecessary huge integer.
    combinations=1
    for r in flexible:
        combinations*=len(allowed[r['unit']])
        if combinations > 4096:
            break
    exhaustive=combinations <= 4096
    if exhaustive:
        # Compare every combination only while the three-backend search fits
        # the explicit cap; larger sweeps use bounded candidate schedules.
        names = [r['unit'] for r in flexible]
        for modes in product(*(allowed[name] for name in names)):
            choice = dict(dense_first)
            choice.update(zip(names, modes))
            proposals.append(choice)
    scores = [_simulate(records, c, cores, workers, budget_gb, options) for c in proposals]
    best_index = min(range(len(scores)), key=lambda i: scores[i])
    best, best_score = proposals[best_index], scores[best_index]
    # Refine small batches (the common 1-geometry frequency sweep) per unit.
    if not exhaustive and len(records) <= 32:
        for r in flexible:
            name = r['unit']
            for mode in allowed[name]:
                choice=dict(best);choice[name]=mode
                score = _simulate(records, choice, cores, workers, budget_gb, options)
                if score < best_score:
                    best, best_score = choice, score
    summary = dict(objective='predicted_batch_completion', model=MODEL,
        search='all_backend_combinations' if exhaustive else 'bounded_mixed_schedules',
        cores=cores, workers=workers, memory_budget_gib=budget_gb,
        compressed_cost_ratio=COMPRESSED_COST_RATIO,
        dense_first_cost=scores[0], fastest_unit_first_cost=scores[0],
        compressed_first_cost=scores[1], selected_cost=best_score,
        dense_units=sum(v == 'dense' for v in best.values()),
        compressed_units=sum(v == 'compressed' for v in best.values()),
        fmm_units=sum(v == 'fmm' for v in best.values()),optimality_guaranteed=False)
    selections = {}
    for r in records:
        name = r['unit']
        selections[name] = dict(requested='adaptive', selected=best[name],
            reason='Selected from whole-batch CPU/memory schedules on the execution node.',
            objective=summary['objective'], model=summary['model'],
            search=summary['search'], compressed_cost_ratio=COMPRESSED_COST_RATIO,
            allocation=dict(cores=cores, workers=workers, memory_budget_gib=budget_gb),
            optimality_guaranteed=False,
            # Reserve no more RAM on retry than the scheduler assigned this unit.
            retry_order=[m for m in allowed[name] if m != best[name] and
                         r['backend_candidates'][m]['peak_gb'] <= r['backend_candidates'][best[name]]['peak_gb']],
            candidates=r['backend_candidates'], selected_batch_cost=best_score)
    return selections, summary


def apply_batch_choices(records, cores, workers, budget, options):
    selections, summary = select_batch_backends(records, cores, workers, budget, options)
    resolved = []
    for original in records:
        r = dict(original)
        if r['unit'] in selections:
            selection = selections[r['unit']]
            r.update(r['backend_candidates'][selection['selected']])
        resolved.append(r)
    if summary:
        print('  Auto batch: {} dense, {} compressed, {} FMM; predicted completion cost {:.3g} '
              '(fastest-unit-first {:.3g}, compressed-first {:.3g}); {} CPUs, {:.1f} GiB budget'.format(
                  summary['dense_units'], summary['compressed_units'], summary['fmm_units'], summary['selected_cost'],
                  summary['dense_first_cost'], summary['compressed_first_cost'], cores, budget), flush=True)
    return resolved, selections, summary
