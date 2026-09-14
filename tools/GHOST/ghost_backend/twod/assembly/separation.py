"""Geometric separation predicates shared by all boundary operators."""
import numpy as np


def segment_distance(p0, p1, q0, q1):
    """Euclidean distance between closed 2-D segments; supports broadcasting.

    Endpoint projections also handle parallel and collinear segments. A proper
    crossing has zero distance even when no endpoint projects onto the crossing.
    """
    p0, p1, q0, q1 = (np.asarray(v, dtype=float) for v in (p0, p1, q0, q1))
    def point_distance(x, a, b):
        edge = b-a
        denominator = np.sum(edge*edge, axis=-1)
        t = np.divide(np.sum((x-a)*edge, axis=-1), denominator,
                      out=np.zeros_like(denominator), where=denominator > 0)
        delta = x-a-np.clip(t, 0., 1.)[..., None]*edge
        return np.sum(delta*delta, axis=-1)
    squared = np.minimum.reduce((point_distance(p0,q0,q1), point_distance(p1,q0,q1),
                                 point_distance(q0,p0,p1), point_distance(q1,p0,p1)))
    u, v, w = p1-p0, q1-q0, q0-p0
    def cross(a,b):
        return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    denominator = cross(u,v)
    safe = np.where(denominator != 0, denominator, 1.)
    t, s = cross(w,v)/safe, cross(w,u)/safe
    crossing = (denominator != 0) & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
    return np.sqrt(np.where(crossing, 0., np.maximum(squared,0.)))


def requires_adaptive(obs, src):
    scale = max(obs.length, src.length)
    distance = float(np.linalg.norm(obs.center-src.center))
    if distance < .75*scale:
        return True
    # The triangle inequality excludes distant pairs without projections.
    if distance >= .5*(obs.length+src.length)+.25*scale:
        return False
    return bool(segment_distance(obs.p0,obs.p1,src.p0,src.p1) < .25*scale)


def close_pairs(p0, p1, q0, q1, center_distance, scale):
    """Tile mask, computing exact distances only in a conservative near band."""
    result = np.zeros(center_distance.shape, dtype=bool)
    i,j = np.nonzero(center_distance < 1.25*scale)
    if len(i):
        result[i,j] = segment_distance(p0[i],p1[i],q0[j],q1[j]) < .25*scale[i,j]
    return result
