"""Shared bounded RHS solve and field projection for every 2-D formulation."""
import numpy as np
from ghost_backend.execution.cpu import current_state, configured_batch_size
from ghost_backend.linalg.dense import DenseFactor


def solve_fields(mesh, matrix, k0, angles, rhs_builder, diagnostics, label,
                 potential='SLP', density_builder=None, observation_angles=None,
                 element_mask=None, order=8, return_density=False, project=True,
                 second_potential=None, coordinates=None):
    import ghost_backend.twod.solver as rcs
    angles = np.asarray(angles, dtype=float).reshape(-1)
    observations = angles if observation_angles is None else np.asarray(observation_angles, float).reshape(-1)
    state = current_state()
    from ghost_backend.twod.assembly.session import current_session
    session = current_session()
    checkpoint = state.checkpoint if state is not None else session.checkpoint if session is not None else None
    requested, threshold = rcs._requested_dense_backend()
    from ghost_backend.linalg.hierarchical import factor_mode
    hierarchical = factor_mode() != 'dense'
    from ghost_backend.compressed.operator import StreamedOperator
    compressed=isinstance(matrix,StreamedOperator)
    if compressed:coordinates=matrix.coordinates
    if requested == 'gpu' and hierarchical:
        raise ValueError('Hierarchical factorization requires the CPU dense backend.')
    if requested == 'gpu' and rcs.requested_precision() == 'mixed':
        raise ValueError('Mixed-precision LU is a CPU method; select the CPU dense backend.')


    gpu = (not hierarchical and state is None and diagnostics is None and rcs.requested_precision() != 'mixed'
           and (requested == 'gpu' or requested == 'auto' and len(matrix) >= threshold))
    batch_size = len(angles) if gpu else configured_batch_size()
    if coordinates is None and len(matrix) in (len(mesh.nodes), 2*len(mesh.nodes)):
        xy = np.zeros((len(mesh.nodes), 2))
        for element in mesh.elements:
            xy[element.node_ids[0]], xy[element.node_ids[1]] = element.p0, element.p1
        coordinates = np.tile(xy, (len(matrix)//len(mesh.nodes), 1))
    from ghost_backend.compressed.factor import CompressedFactor
    factor_class=CompressedFactor if compressed else DenseFactor
    factor = None if gpu else factor_class(matrix, diagnostics, label,
        evidence=state.systems if state is not None else None, checkpoint=checkpoint,
        force_double=state is not None, coordinates=coordinates)
    amplitude = np.zeros(len(angles) if observation_angles is None else (len(angles), len(observations)), complex)
    densities = np.empty((len(mesh.nodes), len(angles)), complex) if return_density else None
    max_residual = 0.0
    from ghost_backend.linalg.sweep import (
        SweepBasis,
        solve as solve_sweep,
        mode as compression_mode,
    )
    sweep_basis = (SweepBasis(batch_size) if factor is not None and len(angles) >= 32
                   and compression_mode() != 'off' else None)
    for start in range(0, len(angles), batch_size):
        if checkpoint is not None:
            checkpoint()
        stop = min(len(angles), start + batch_size)
        rhs = rhs_builder(angles[start:stop])
        if factor is None:
            evidence = {}
            solution = rcs._solve_dense_system(matrix, rhs, diagnostics, label, residual_diagnostics=evidence)
            relative = evidence['relative_residual']
        else:
            solution = solve_sweep(factor, rhs, sweep_basis)
            relative = factor.relative_residual
        max_residual = max(max_residual, float(np.max(relative)))
        density = solution[:len(mesh.nodes)] if density_builder is None else density_builder(solution)
        if densities is not None:
            densities[:, start:stop] = density
        if project:
            obs = angles[start:stop] if observation_angles is None else observations
            projection = 'matched' if observation_angles is None else 'grid'
            field = rcs._farfield_linear_density_many(mesh, density, k0, obs, potential,
                element_mask=element_mask, projection=projection, order=order)
            if second_potential is not None:
                field += rcs._farfield_linear_density_many(mesh, solution[len(mesh.nodes):], k0, obs,
                    second_potential, projection=projection, order=order)
            amplitude[start:stop] = field
        if state is not None:
            state.checkpoint(stop, len(angles))


        solution = density = rhs = None
    return rcs._rcs_sigma_from_amp(amplitude, k0), amplitude, max_residual, densities
