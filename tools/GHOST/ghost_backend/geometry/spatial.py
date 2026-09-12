"""Streaming broad-phase queries shared by editor and solver validation."""


def overlapping_pairs(bounds, tolerance=0.0, checkpoint=None):
    """Yield candidate index pairs once, without retaining a quadratic pair list.

    Bounds are (xmin, xmax, ymin, ymax). Exact geometric predicates remain the
    caller's responsibility, including endpoint and collinearity semantics.
    """
    if not bounds:
        return
    span_x = max(b[1] for b in bounds) - min(b[0] for b in bounds)
    span_y = max(b[3] for b in bounds) - min(b[2] for b in bounds)
    # Prefer the axis with less interval coverage. Extent alone degenerates to
    # all-pairs work on a stack of long parallel lines along its widest axis.
    coverage_x = sum(b[1]-b[0] for b in bounds)/span_x if span_x > 0 else float('inf')
    coverage_y = sum(b[3]-b[2] for b in bounds)/span_y if span_y > 0 else float('inf')
    lo, hi, other_lo, other_hi = (0, 1, 2, 3) if coverage_x <= coverage_y else (2, 3, 0, 1)
    order = sorted(range(len(bounds)), key=lambda i: (bounds[i][lo], i))
    checks = 0
    for pos, i in enumerate(order):
        if checkpoint is not None:
            checkpoint()
        a = bounds[i]
        for later in range(pos + 1, len(order)):
            j = order[later]
            b = bounds[j]
            if b[lo] > a[hi] + tolerance:
                break
            checks += 1
            if checkpoint is not None and checks % 1024 == 0:
                checkpoint()
            if b[other_lo] <= a[other_hi] + tolerance and a[other_lo] <= b[other_hi] + tolerance:
                yield (i, j) if i < j else (j, i)


def primitive_bounds(primitives):
    return [(min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
            for x1, y1, x2, y2 in primitives]
