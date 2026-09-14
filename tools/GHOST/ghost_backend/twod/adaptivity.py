"""Automatic h/p candidates with modal marking and independent field checks.

Modal tails guide where to spend work; they are not rigorous error bounds.
Acceptance uses every requested complex far field and the existing residual,
condition and mesh-convergence gates. Geometry and material values are unchanged.
"""
import copy
import time
import numpy as np
from ghost_backend.execution.runtime import ScopedValue
from ghost_backend.execution.options import current_options, execution_scope, automatic_backend_requested
from ghost_backend.twod.basis import abscissae

_INDICATORS = ScopedValue('ghost_hp_indicators', None)


class Indicators:
    def __init__(self):
        self.scores = {}

    def observe(self, mesh, density):
        degree = len(mesh.elements[0].node_ids)-1
        if degree < 2: return
        ids = np.asarray([element.node_ids for element in mesh.elements])
        # Coefficients in a shifted Legendre basis give a mesh-local smoothness
        # indicator. Angle batches are discarded; only one scalar per primitive
        # survives, including contributions from both sides of material interfaces.
        transform = np.linalg.inv(np.polynomial.legendre.legvander(2*abscissae(degree)-1, degree))
        magnitude = np.max(abs(density), axis=0)
        scale = np.maximum(magnitude, np.finfo(float).tiny)
        for first in range(0, len(ids), 256):
            last = min(first+256, len(ids))
            local = density[ids[first:last]] / scale[None, None, :]
            tail = np.einsum('a,ear->er', transform[-1], local)
            scores = np.max(abs(tail)**2, axis=1)
            for element, score in zip(mesh.elements[first:last], scores):
                key = getattr(element, 'primitive_key', '')
                if key:
                    self.scores[key] = self.scores.get(key, 0.) + float(score)*element.length

    def marked(self):
        ordered = sorted(self.scores.items(), key=lambda item: (-item[1], item[0]))
        target = .6 * sum(value for _, value in ordered)
        result, total = [], 0.
        for key, value in ordered:
            if value <= 0: break
            result.append(key); total += value
            if total >= target: break
        return result


def observe(mesh, solution, density, routes=None):
    indicators = _INDICATORS.get()
    if indicators is None: return
    n = len(mesh.nodes)
    if routes is not None:
        for nodes, offset in routes:
            component = np.zeros((n, solution.shape[1]), complex)
            component[nodes] = solution[offset:offset+len(nodes)]
            indicators.observe(mesh, component)
    elif solution.shape[0] % n == 0:
        for start in range(0, solution.shape[0], n):
            indicators.observe(mesh, solution[start:start+n])
    else:
        indicators.observe(mesh, density)


def run_certified(low_level_solver, geometry_snapshot, solver_kwargs,
                  mesh_convergence_policy, progress_callback, shared_discretization_caches=None):
    from ghost_backend.twod import solver as s
    from ghost_backend.twod.preparation import prepare_geometry
    from ghost_backend.twod.adaptive_geometry import eligible_snapshot
    from ghost_backend.runs.quality import validate_mesh_convergence_policy
    options = current_options()
    _, _, materials, scale = prepare_geometry(geometry_snapshot, solver_kwargs.get('material_base_dir'),
                                         solver_kwargs.get('geometry_units', 'inches'))
    eligible, reason = eligible_snapshot(geometry_snapshot, materials, solver_kwargs['frequencies_ghz'],
                                         scale, solver_kwargs.get('mesh_reference_ghz'))
    policy = validate_mesh_convergence_policy(mesh_convergence_policy)
    if not eligible:
        with execution_scope(dict(options, mesh_strategy='global', basis_order=1)):
            result = s._run_certified_2d_pair_impl(low_level_solver, geometry_snapshot, solver_kwargs,
                policy, progress_callback, shared_discretization_caches)
        result['metadata']['adaptive_mesh'] = dict(used=False, reason=reason)
        return result
    # Coarsening must not consume the permissive legacy default error budget.
    # Honor any stricter requested limits and retain the user's reference policy
    # if a candidate fails. These are convergence tolerances, not error bounds.
    candidate_policy = dict(policy)
    for key, ceiling in dict(rms_limit_db=.05, max_abs_limit_db=.25,
                             complex_rms_limit=.001, complex_max_limit=.002,
                             phase_rms_limit_deg=.5, phase_max_limit_deg=2.).items():
        candidate_policy[key] = min(candidate_policy[key], ceiling)
    common = dict(solver_kwargs, strict_quality_gate=True, compute_condition_number=True)
    steps = []
    start = time.perf_counter()

    def solve(snapshot, degree, label):
        if progress_callback is not None: progress_callback(0, 1, label)
        indicators = Indicators()
        kwargs = dict(common, geometry_snapshot=snapshot, progress_callback=progress_callback)
        kwargs.pop('_shared_discretization_cache', None)
        before = time.perf_counter()
        selected_options = dict(options, mesh_strategy='local', basis_order=degree)
        selection = None
        if automatic_backend_requested():
            from ghost_backend.execution.selection import select_backend
            selection = select_backend(kwargs, selected_options, certified=False)
            admitted = [selection['selected']] + selection.get('retry_order', [])
            preferred = options['factorization']
            if len(steps) < 2 and preferred in admitted:
                # Retain whole-request timing evidence and the scheduler's
                # concurrency choice, after admission on this actual mesh.
                selection = dict(selection, selected=preferred,
                    retry_order=[mode for mode in admitted if mode != preferred],
                    reason='Retained request or batch preference after admission on the actual adaptive mesh.')
            selected_options['factorization'] = selection['selected']
        from ghost_backend.execution.errors import BackendNumericalError
        modes = [selected_options['factorization']] + (selection.get('retry_order', []) if selection else [])
        failures = []
        for index, mode in enumerate(modes):
            selected_options['factorization'] = mode
            indicators = Indicators()
            try:
                with execution_scope(selected_options), _INDICATORS.override(indicators):
                    result = low_level_solver(**kwargs)
                break
            except (MemoryError, BackendNumericalError, np.linalg.LinAlgError) as exc:
                if index == len(modes)-1: raise
                failures.append(dict(backend=mode, reason=str(exc)))
            import gc
            gc.collect()
        metadata = result['metadata']
        if failures:
            selection = dict(selection, initial_selection=selection['selected'], selected=mode,
                failed_attempts=failures, retry_order=modes[index+1:],
                reason='Selected after an admitted backend retry on the adaptive mesh.')
        # Co-polarized metadata preserves the per-channel discretization counts.
        steps.append(dict(polynomial_degree=degree, panels=metadata.get('panel_count'),
                          nodes=metadata.get('linear_node_count'), seconds=time.perf_counter()-before,
                          backend=selected_options['factorization'], backend_selection=selection,
                          failed_backends=failures, marked_primitives=len(indicators.marked())))
        return result, indicators

    snapshot = copy.deepcopy(geometry_snapshot)
    snapshot['_2d_hp_coarsening'] = 4.
    snapshot['_2d_hp_refinements'] = {}
    try:
        base, indicators = solve(snapshot, 2, 'Adaptive mesh: quadratic candidate')
        # A global increase of polynomial degree tests every primitive, including
        # ones whose modal indicator happened to be small.
        fine, indicators = solve(snapshot, 3, 'Adaptive mesh: cubic accuracy check')
        for attempt in range(3):
            try:
                with execution_scope(dict(options, mesh_strategy='adaptive', basis_order=3)):
                    result = s._finish_certified_2d_pair(base, fine, candidate_policy)
                result['metadata']['adaptive_mesh'] = dict(used=True, strategy='hp', steps=steps,
                    elapsed_seconds=time.perf_counter()-start, accepted_degree=3,
                    final_backend=steps[-1]['backend'],
                    acceptance_policy=candidate_policy,
                    estimator='normalized Legendre tail marking; full-field convergence acceptance',
                    geometry_preserved=True, fallback=False)
                return result
            except ValueError as exc:
                if not str(exc).startswith('Certified 2-D mesh convergence failed:'): raise
                reason = str(exc)
            if attempt == 2: break
            base = fine
            snapshot = copy.deepcopy(snapshot)
            # Global h enrichment supplies an independent comparison; marked
            # primitives receive an additional local refinement.
            snapshot['_2d_hp_coarsening'] = max(1., snapshot['_2d_hp_coarsening'] / policy['fine_factor'])
            refinements = snapshot['_2d_hp_refinements']
            for key in indicators.marked(): refinements[key] = min(4., 1.5*refinements.get(key, 1.))
            fine, indicators = solve(snapshot, 3, 'Adaptive mesh: local refinement and global accuracy check')
    except MemoryError as exc:
        reason = 'Adaptive refinement exceeded the execution reservation: ' + str(exc)
    except ValueError as exc:
        if not str(exc).startswith(('Polynomial near quadrature did not converge',
                                    'Certified 2-D mesh refinement failed:', 'Quality gate failed:')):
            raise
        reason = str(exc)
    if progress_callback is not None: progress_callback(0, 1, 'Adaptive candidate rejected; checking the reference mesh')
    fallback_options = dict(options, mesh_strategy='global', basis_order=1)
    if automatic_backend_requested():
        from ghost_backend.execution.selection import select_backend
        arguments = dict(solver_kwargs, geometry_snapshot=geometry_snapshot, mesh_convergence_policy=policy)
        fallback_selection = select_backend(arguments, fallback_options, certified=True)
        fallback_options['factorization'] = fallback_selection['selected']
    with execution_scope(fallback_options):
        result = s._run_certified_2d_pair_impl(low_level_solver, geometry_snapshot, solver_kwargs,
            policy, progress_callback, shared_discretization_caches)
    result['metadata']['adaptive_mesh'] = dict(used=False, strategy='hp', steps=steps, fallback=True, reason=reason,
        final_backend=fallback_options['factorization'],
        elapsed_seconds=time.perf_counter()-start)
    return result
