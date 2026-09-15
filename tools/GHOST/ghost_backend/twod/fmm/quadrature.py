"""Conservative point-quadrature policy, separate from native FMM tolerance."""
import math


def quadrature_order(k, longest, requested=0, integration_order=8, eps=1e-10):
    """Use the qualified six-point range; retain eight points outside it.

    All pairs within three maximum panel lengths still receive accurate near
    corrections. The reduced far rule is restricted to |k|h <= 1 and native
    tolerances >= 1e-10. This is an empirically qualified rule, not a rigorous
    global quadrature error bound. Explicit orders are lower bounds, with the
    existing electrical-length safeguard always applied.
    """
    electrical = abs(k)*longest
    if requested:
        base = max(6, int(requested))
        policy = 'explicit_minimum'
    elif electrical <= 1. and eps >= 1e-10 and integration_order <= 8:
        base = 6
        policy = 'qualified_six_point'
    else:
        base = max(8, int(integration_order))
        policy = 'conservative_eight_point'
    order = max(base, math.ceil(electrical/2)+4)
    if order > 64:
        raise ValueError('FMM quadrature needs a finer mesh (order exceeds 64).')
    return order, policy
