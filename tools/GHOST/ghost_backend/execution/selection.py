"""Shared, capability-aware planning before allocating solver operators."""
import math
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.execution.runtime import ScopedValue
from ghost_backend.execution.policy import MODEL, BACKENDS, relative_cost, fmm_eligibility, rank_candidates, native_fmm_available

_BATCH_SELECTION = ScopedValue('ghost_batch_backend_selection', default=None)


def batch_selection_scope(value):
    if value is not None and (value.get('requested') != 'adaptive' or
                              value.get('selected') not in BACKENDS):
        raise ValueError('Invalid batch backend selection.')
    return _BATCH_SELECTION.override(value)


def current_batch_selection():
    value = _BATCH_SELECTION.get()
    return dict(value) if value is not None else None


def select_backend(arguments, options, certified=False, checkpoint=None):
    """Forecast both polarizations and certification meshes before allocating A.

    Rank compatible backends using mesh-specific work and memory forecasts.
    This is a deterministic resource heuristic, not a promise of minimum time.
    Explicit backend choices never call this function.
    """
    from ghost_backend.twod import solver as s
    from ghost_backend.twod.preparation import prepare_geometry
    from ghost_backend.runs.quality import validate_mesh_convergence_policy, scale_snapshot_panel_density
    snapshot = arguments['geometry_snapshot']
    frequencies = list(arguments['frequencies_ghz'])
    if not frequencies or any(not math.isfinite(f) or f <= 0 for f in frequencies):
        raise ValueError('Frequencies must be a nonempty list of finite positive GHz values.')
    if len(set(frequencies)) != len(frequencies):
        raise ValueError('Duplicate frequencies are not supported.')
    if not arguments['elevations_deg'] or any(not math.isfinite(a) for a in arguments['elevations_deg']):
        raise ValueError('Angles must be a nonempty list of finite values.')
    units = arguments.get('geometry_units', 'inches')
    _, _, materials, scale = prepare_geometry(snapshot, arguments.get('material_base_dir'), units)
    from ghost_backend.twod.adaptive_geometry import candidate_meshes
    factor = validate_mesh_convergence_policy(arguments.get('mesh_convergence_policy'))['fine_factor'] if certified else 1.
    records = []
    candidates={m:dict(cost=0.,peak_gb=0.) for m in BACKENDS}
    exclusions={}
    dense = dict(validate_options(options), factorization='dense')
    with execution_scope(dense):
        budget = s._solve_memory_limit_gb()
        for freq in arguments['frequencies_ghz']:
            if checkpoint:
                checkpoint()
            event = arguments.get('abort_event')
            if event is not None and event.is_set():
                raise InterruptedError('Backend planning canceled.')
            ref = arguments.get('mesh_reference_ghz') or freq
            # The co-polarized API finishes each frequency independently. The
            # explicit single-channel API can still solve one joint mesh pair.
            mesh_frequencies = frequencies if 'polarization' in arguments else [freq]
            geometries = candidate_meshes(snapshot, materials, factor, options['mesh_strategy']=='adaptive',
                                         mesh_frequencies, scale, arguments.get('mesh_reference_ghz'))
            if options['mesh_strategy']!='adaptive':
                geometries = [(phase, geometry, options['basis_order']) for phase, geometry, _ in geometries]
            for phase, geometry, degree in geometries:
                mesh_options=dict(dense,basis_order=degree,mesh_strategy='local' if '_2d_hp_coarsening' in geometry else dense['mesh_strategy'])
                with execution_scope(mesh_options):
                    wavelength, _, _ = s._conservative_mesh_wavelength_for_frequencies(
                        geometry, materials, set(mesh_frequencies) | {ref}) if arguments.get('mesh_reference_ghz') else s._mesh_wavelength_for_snapshot(geometry, materials, ref)
                    # Conservative global mesh bounds local-material candidate sizes.
                    panels = s._build_panels(geometry, scale, wavelength, max_panels=arguments.get('max_panels', s.MAX_PANELS_DEFAULT),
                        segment_wavelengths=s.segment_wavelengths(geometry,materials,set(mesh_frequencies) | {ref} if arguments.get('mesh_reference_ghz') else [ref],scale,wavelength))
                    k0 = 2 * math.pi * freq * 1e9 / s.C0
                    pols=(s._normalize_polarization(arguments['polarization']),) if 'polarization' in arguments else ('TE','TM')
                    for pol in pols:
                        infos = s._build_coupled_panel_info(panels, materials, freq, pol, k0)
                        mesh, _ = s._build_linear_mesh_interface_aware(panels, infos)
                        coupled = s._build_linear_coupled_infos(mesh, materials, freq, pol, k0)
                        layer = s.layer_for_mesh(mesh, materials, freq) if any(i.bc_kind == 'thin_layer' for i in coupled) else None
                        resources = s._dense_formulation_resources(mesh, coupled, pol, layer, sample_compression=False)
                        eligible,why=fmm_eligibility(resources,mesh,coupled)
                        if not eligible:
                            exclusions['fmm']=why
                        peaks={}
                        for mode in BACKENDS:
                            if mode == 'fmm' and not eligible:
                                continue
                            with execution_scope(dict(dense,factorization=mode)):
                                measured=dict(resources)
                                peaks[mode]=s._estimate_memory_gb(resources['nodes'], False,
                                    n_regions=resources['n_regions'], system_dofs=resources['system_dofs'],
                                    operator_matrices=resources['operator_matrices'], dense_resources=measured,
                                    n_rhs=len(arguments['elevations_deg']), solver_method='experimental_cpu')
                            candidates[mode]['cost']+=relative_cost(resources,len(arguments['elevations_deg']),mode)
                            candidates[mode]['peak_gb']=max(candidates[mode]['peak_gb'],peaks[mode])
                        records.append(dict(frequency_ghz=float(freq), phase=phase, polarization=pol,
                            polynomial_degree=degree, panels=len(panels), unknowns=resources['system_dofs'], dense_peak_gib=peaks['dense'],
                            formulation=resources['formulation'],backend_peak_gib=peaks))
    peak = max(r['dense_peak_gib'] for r in records)
    if not native_fmm_available():
        exclusions['fmm']='The native FMM library is unavailable on this execution host.'
    candidates={m:c for m,c in candidates.items() if m not in exclusions}
    try:
        ranked=rank_candidates(candidates,budget)
    except MemoryError as exc:
        kinds=', '.join(sorted({r['formulation'] for r in records}))
        raise MemoryError('{} Formulations: {}.'.format(exc,kinds)) from exc
    selected=ranked[0]
    return dict(requested='adaptive', selected=selected, dense_peak_gib=peak,
        model=MODEL,objective='predicted_solve_completion',candidates=candidates,
        retry_order=ranked[1:],exclusions=exclusions,optimality_guaranteed=False,
        admission_budget_gib=budget, dense_margin_fraction=.2,
        reason='Lowest predicted cost among compatible backends admitted against available RAM.',
        meshes=records)
