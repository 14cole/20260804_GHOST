"""Geometry-based mesh guidance; these indicators are not error estimates."""
import math
import numpy as np


def refined_density(value, factor=2.):
    n = float(value or 0)
    if not math.isfinite(n) or not n.is_integer() or not math.isfinite(factor) or factor <= 1:
        raise ValueError("Mesh N must be an integer and refinement factor greater than one.")
    if n > 0:
        return str(int(math.ceil(n * factor)))
    return str(-int(math.ceil((abs(n) or 20) * factor)))


def geometry_refinement_candidates(segments, corner_angle_deg=25.):
    """Identify snapshot segments touching open ends, bends, or material junctions.

    Coordinates are only used for direction and connectivity; no unit-dependent
    proximity threshold or guessed material wavelength enters this selection.
    """
    nodes = {}
    for index, segment in enumerate(segments):
        pairs = segment.get("point_pairs", [])
        props = segment.get("properties", [])
        signature = tuple(str(v) for i, v in enumerate(props) if i != 1)
        for pair in pairs:
            a = np.array([pair["x1"], pair["y1"]], float)
            b = np.array([pair["x2"], pair["y2"]], float)
            length = np.linalg.norm(b-a)
            if length <= 0:
                continue
            nodes.setdefault(tuple(a), []).append((index, (b-a)/length, signature))
            nodes.setdefault(tuple(b), []).append((index, (a-b)/length, signature))
    reasons = {}
    for neighbors in nodes.values():
        reason = None
        if len(neighbors) == 1:
            reason = "open end"
        elif len(neighbors) > 2 or len({n[2] for n in neighbors}) > 1:
            reason = "junction"
        elif math.degrees(math.acos(float(np.clip(-np.dot(neighbors[0][1], neighbors[1][1]), -1., 1.)))) >= corner_angle_deg:
            reason = "corner"
        if reason:
            for index, _, _ in neighbors:
                reasons.setdefault(index, set()).add(reason)
    return {index: sorted(value) for index, value in reasons.items()}
