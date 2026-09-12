"""Conservative RAM-aware choice between existing dense and compressed paths."""
import math
from ghost_backend.execution.options import execution_scope, validate_options


def select_backend(arguments, options, certified=False, checkpoint=None):
    """Forecast both polarizations and certification meshes before allocating A.

    Dense is preferred when its forecast fits with an additional 20% margin.
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
    geometries = [('base', snapshot)]
    if certified:
        factor = validate_mesh_convergence_policy(arguments.get('mesh_convergence_policy'))['fine_factor']
        fine = scale_snapshot_panel_density(snapshot, factor)
        fine['_2d_certification_refinement_factor'] = factor
        fine['_2d_certification_base_segment_n'] = [(list(seg.get('properties', [])) + [0, 0])[1] for seg in snapshot['segments']]
        geometries.append(('fine', fine))
    records = []
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
            for phase, geometry in geometries:
                wavelength, _, _ = s._conservative_mesh_wavelength_for_frequencies(
                    geometry, materials, set(arguments['frequencies_ghz']) | {ref}) if arguments.get('mesh_reference_ghz') else s._mesh_wavelength_for_snapshot(geometry, materials, ref)
                # Conservative global mesh bounds local-material candidate sizes.
                panels = s._build_panels(geometry, scale, wavelength, max_panels=arguments.get('max_panels', s.MAX_PANELS_DEFAULT))
                k0 = 2 * math.pi * freq * 1e9 / s.C0
                for pol in ('TE', 'TM'):
                    infos = s._build_coupled_panel_info(panels, materials, freq, pol, k0)
                    mesh, _ = s._build_linear_mesh_interface_aware(panels, infos)
                    coupled = s._build_linear_coupled_infos(mesh, materials, freq, pol, k0)
                    layer = s.layer_for_mesh(mesh, materials, freq) if any(i.bc_kind == 'thin_layer' for i in coupled) else None
                    resources = s._dense_formulation_resources(mesh, coupled, pol, layer)
                    peak = s._estimate_memory_gb(resources['nodes'], False,
                        n_regions=resources['n_regions'], system_dofs=resources['system_dofs'],
                        operator_matrices=resources['operator_matrices'], dense_resources=resources,
                        n_rhs=len(arguments['elevations_deg']), solver_method=arguments['solver_method'])
                    records.append(dict(frequency_ghz=float(freq), phase=phase, polarization=pol,
                        panels=len(panels), unknowns=resources['system_dofs'], dense_peak_gib=peak))
    peak = max(r['dense_peak_gib'] for r in records)
    selected = 'dense' if peak <= .8 * budget else 'compressed'
    return dict(requested='adaptive', selected=selected, dense_peak_gib=peak,
        admission_budget_gib=budget, dense_margin_fraction=.2,
        reason='Dense forecast fits with 20% additional headroom.' if selected == 'dense' else
               'Dense forecast exceeds the selection margin; compressed admission is checked before assembly.',
        meshes=records)
