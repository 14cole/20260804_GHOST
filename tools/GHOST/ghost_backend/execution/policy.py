"""Shared CPU backend eligibility and relative runtime prior.

The prior ranks work; it is not a wall-time or optimality guarantee. It retains
the measured dense preference on small/medium systems and accounts for FMM's
per-illumination iterations. Both desktop and execution-node scheduling use it.
"""
import math

MODEL = 'geometry_work_v3_pulse'
BACKENDS = ('dense', 'compressed', 'fmm')


def native_fmm_available():
    try:
        from ghost_backend.twod.fmm.kernel import library
        library()
        return True
    except (OSError, RuntimeError, AttributeError):
        return False


def fmm_eligibility(resources, mesh=None, infos=None):
    if resources.get('formulation') == 'thin_dielectric_layer' and not resources.get('analytic_zero'):
        return False, 'Nonzero thin-layer approximations require a qualified dense/compressed formulation.'
    if mesh is not None and infos is not None:
        import numpy as np
        lengths=np.asarray([e.length for e in mesh.elements])
        longest=float(lengths.max()) if len(lengths) else 0.
        for info in infos:
            for side in ('minus','plus'):
                if getattr(info,side+'_region') < 0:
                    continue
                k=complex(getattr(info,'k_'+side))
                if not math.isfinite(abs(k)) or k.real <= 0 or k.imag > 0:
                    return False, 'The material wavenumber is outside the qualified native FMM range.'
                if max(8,math.ceil(abs(k)*longest/2)+4) > 64:
                    return False, 'The supplied mesh requires more than 64 FMM quadrature nodes per panel.'
    return True, 'Supported 2-D material equations.'


def relative_cost(resources, n_angles, mode):
    n=max(1,int(resources['nodes'])); d=max(1,int(resources['system_dofs']))
    angles=max(1,int(n_angles)); kernels=max(1.,resources.get('operator_matrices',3)/3.)
    if resources.get('analytic_zero'):
        return .001
    assembly=7e-7*n*n*kernels
    # Integrated P0 timings include accurate near integrals and routing; kernel
    # sample-count ratios alone substantially overpredict its assembly gain.
    if resources.get('discretization')=='pulse':assembly*=.75
    dense=.025 + assembly + 4e-12*d**3 + 2e-10*d*d*angles
    if mode == 'dense':
        return dense
    if mode == 'compressed':
        return 1.4*dense
    if mode != 'fmm':
        raise ValueError('Unknown automatic backend.')
    # Conservative prior from isolated Galerkin/FMM measurements. Recycled
    # solves can beat this prior; difficult cavities/materials can exceed it.
    return .025 + .00045*n*kernels + .0012*n*max(1.,math.log2(n))*kernels*angles/3.


def available_candidates(candidates):
    """Check native availability on the host that will actually execute."""
    return {mode:dict(c) for mode,c in candidates.items()
            if mode != 'fmm' or native_fmm_available()}


def rank_candidates(candidates, budget_gib, margin=.2):
    available=available_candidates(candidates)
    if not available:
        raise RuntimeError('No compatible solver backend is available.')
    for mode,c in available.items():
        if mode not in BACKENDS or any(not math.isfinite(float(c[k])) or c[k] <= 0 for k in ('cost','peak_gb')):
            raise ValueError('Invalid automatic backend forecast.')
    fitting=[m for m,c in available.items() if c['peak_gb'] <= (1-margin)*budget_gib]
    if not fitting:
        fitting=[m for m,c in available.items() if c['peak_gb'] <= budget_gib]
    if not fitting:
        description=', '.join('{} {:.2f} GiB'.format(m,c['peak_gb']) for m,c in available.items())
        raise MemoryError('No compatible backend fits the {:.2f} GiB solve budget ({}).'.format(budget_gib,description))
    return sorted(fitting,key=lambda m:(available[m]['cost'],BACKENDS.index(m)))
