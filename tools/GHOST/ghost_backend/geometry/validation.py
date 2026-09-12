"""Captured geometry checks, independent of Qt widgets and plotting."""
import math
from typing import Any, Dict, List, Set, Tuple
from ghost_backend.geometry.io import ChainSpec, Segment, check_orientation_consistency
from ghost_backend.geometry.spatial import overlapping_pairs, primitive_bounds

def _parse_mesh_n_token(token: 'Any') -> 'int':
    """Parse the geometry N field without changing its solver semantics.

    Blank or zero selects automatic 20-panels-per-wavelength meshing, a
    positive integer is an explicit panel count per primitive, and a negative
    integer selects its absolute value in panels per wavelength.
    """

    text = str(token or "").strip()
    if not text:
        return 0
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("N must be an integer.") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError("N must be an integer.")
    return int(value)



class GeometryAudit:
    def __init__(self, segments, ibcs, dielectrics, material_dir, checkpoint=None):
        self.segments = segments
        self.ibcs = ibcs
        self.dielectrics = dielectrics
        self.material_dir = material_dir
        self.checkpoint = checkpoint or (lambda: None)

    def _parse_int_token(self, token: 'str', default: 'int' = 0) -> 'int':
        text = (token or "").strip().lower()
        if not text:
            return default
        if text.startswith("mat."):
            text = text.split("mat.", 1)[1]
        try:
            return int(float(text))
        except ValueError:
            return default


    def _parse_float_token(self, token: 'str', default: 'float' = 0.0) -> 'float':
        text = (token or "").strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default


    def _point_key(self, x: 'float', y: 'float', tol: 'float') -> 'Tuple[int, int]':
        inv = 1.0 / max(tol, 1e-12)
        return int(round(float(x) * inv)), int(round(float(y) * inv))


    def _segments_intersect(
        self,
        a1: 'Tuple[float, float]',
        a2: 'Tuple[float, float]',
        b1: 'Tuple[float, float]',
        b2: 'Tuple[float, float]',
        tol: 'float',
    ) -> 'bool':
        ax1, ay1 = a1
        ax2, ay2 = a2
        bx1, by1 = b1
        bx2, by2 = b2

        min_ax, max_ax = min(ax1, ax2), max(ax1, ax2)
        min_ay, max_ay = min(ay1, ay2), max(ay1, ay2)
        min_bx, max_bx = min(bx1, bx2), max(bx1, bx2)
        min_by, max_by = min(by1, by2), max(by1, by2)
        if max_ax < min_bx - tol or max_bx < min_ax - tol:
            return False
        if max_ay < min_by - tol or max_by < min_ay - tol:
            return False

        def orient(px: 'float', py: 'float', qx: 'float', qy: 'float', rx: 'float', ry: 'float') -> 'float':
            return (qx - px) * (ry - py) - (qy - py) * (rx - px)

        def on_seg(px: 'float', py: 'float', qx: 'float', qy: 'float', rx: 'float', ry: 'float') -> 'bool':
            return (
                min(px, qx) - tol <= rx <= max(px, qx) + tol
                and min(py, qy) - tol <= ry <= max(py, qy) + tol
            )

        o1 = orient(ax1, ay1, ax2, ay2, bx1, by1)
        o2 = orient(ax1, ay1, ax2, ay2, bx2, by2)
        o3 = orient(bx1, by1, bx2, by2, ax1, ay1)
        o4 = orient(bx1, by1, bx2, by2, ax2, ay2)

        if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (
            o3 > tol and o4 < -tol or o3 < -tol and o4 > tol
        ):
            return True

        if abs(o1) <= tol and on_seg(ax1, ay1, ax2, ay2, bx1, by1):
            return True
        if abs(o2) <= tol and on_seg(ax1, ay1, ax2, ay2, bx2, by2):
            return True
        if abs(o3) <= tol and on_seg(bx1, by1, bx2, by2, ax1, ay1):
            return True
        if abs(o4) <= tol and on_seg(bx1, by1, bx2, by2, ax2, ay2):
            return True
        return False


    def _ensure_prop_len(self, props: 'List[str]', n: 'int') -> 'List[str]':
        if len(props) < n:
            props.extend([""] * (n - len(props)))
        return props


    def _segment_primitives(self, seg: 'Segment') -> 'List[Tuple[float, float, float, float]]':
        count = min(len(seg.x), len(seg.y))
        n_pairs = count // 2
        out: 'List[Tuple[float, float, float, float]]' = []
        for i in range(n_pairs):
            idx = 2 * i
            out.append((seg.x[idx], seg.y[idx], seg.x[idx + 1], seg.y[idx + 1]))
        return out


    def _segment_plot_xy(self, seg: 'Segment') -> 'Tuple[List[float], List[float]]':
        primitives = self._segment_primitives(seg)
        if not primitives:
            return list(seg.x), list(seg.y)

        xs: 'List[float]' = []
        ys: 'List[float]' = []
        for i, (x1, y1, x2, y2) in enumerate(primitives):
            if i == 0:
                xs.append(x1)
                ys.append(y1)
            xs.append(x2)
            ys.append(y2)

        if not xs or not ys:
            return list(seg.x), list(seg.y)
        return xs, ys


    def run(self):
        ibcs_rows = self.ibcs
        dielectric_rows = self.dielectrics
        diel_flags = {
            self._parse_int_token(row[0], 0)
            for row in dielectric_rows if row
        }
        ibc_flags = {
            self._parse_int_token(row[0], 0)
            for row in ibcs_rows if row
        }

        findings: 'List[Tuple[str, int, str]]' = []
        issue_rows: 'Set[int]' = set()

        material_dir = self.material_dir
        try:


            from ghost_backend.twod.solver import MaterialLibrary
            MaterialLibrary.from_entries(
                ibcs_rows, dielectric_rows, material_dir
            )
        except Exception as exc:
            findings.append(("ERROR", -1, f"Material definition error: {exc}"))

        for ibc_idx, row in enumerate(ibcs_rows):
            if (
                len(row) == 6
                and str(row[1]).strip().lower() == "exp"
            ):
                z_parts = [
                    self._parse_float_token(row[i], float("nan"))
                    for i in (2, 3, 4, 5)
                ]
                if all(math.isfinite(value) for value in z_parts):
                    if (
                        z_parts[0] ** 2 + z_parts[1] ** 2 == 0.0
                        or z_parts[2] ** 2 + z_parts[3] ** 2 == 0.0
                    ):
                        findings.append((
                            "WARN", -1,
                            f"IBCS row {ibc_idx + 1}: exp taper endpoints "
                            "should be nonzero; prefer linear or cosine for "
                            "PEC-limit transitions.",
                        ))

        all_points = [(x, y) for seg in self.segments for x, y in zip(seg.x, seg.y)]
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            diag = max(((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5, 1.0)
        else:
            diag = 1.0
        tol = max(1e-8, 1e-6 * diag)

        primitives_by_row = [self._segment_primitives(seg) for seg in self.segments]
        chain_end_rows = {}
        for row, primitives in enumerate(primitives_by_row):
            self.checkpoint()
            if primitives:
                for x, y in (primitives[0][:2], primitives[-1][2:]):
                    chain_end_rows.setdefault(self._point_key(x, y, tol), set()).add(row)

        for row, seg in enumerate(self.segments):
            self.checkpoint()
            props = self._ensure_prop_len(seg.properties, 5)
            seg_type = self._parse_int_token(props[0], -1)
            ibc = self._parse_int_token(props[2], 0)
            pos_mat = self._parse_int_token(props[3], 0)
            neg_mat = self._parse_int_token(props[4], 0)
            primitives = primitives_by_row[row]
            label = f"Row {row + 1} '{seg.name}'"

            if seg_type < 1 or seg_type > 5:
                findings.append(("ERROR", row, f"{label}: invalid TYPE '{props[0]}', expected 1..5."))
                issue_rows.add(row)

            try:
                _parse_mesh_n_token(props[1])
            except ValueError:
                findings.append((
                    "ERROR",
                    row,
                    f"{label}: N must be an integer; use 0 or blank for "
                    "automatic 20-panels-per-wavelength meshing, a positive "
                    "integer for an explicit panel count per primitive, or a "
                    "negative integer for panels per wavelength. Current "
                    f"value is '{props[1]}'.",
                ))
                issue_rows.add(row)

            if not primitives:
                findings.append(("ERROR", row, f"{label}: no line primitives found."))
                issue_rows.add(row)
                continue

            for i, (x1, y1, x2, y2) in enumerate(primitives):
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length <= tol:
                    findings.append(("ERROR", row, f"{label}: primitive {i + 1} has near-zero length."))
                    issue_rows.add(row)

            for i in range(len(primitives) - 1):
                _, _, ex, ey = primitives[i]
                nx1, ny1, nx2, ny2 = primitives[i + 1]
                d_start = ((ex - nx1) ** 2 + (ey - ny1) ** 2) ** 0.5
                d_end = ((ex - nx2) ** 2 + (ey - ny2) ** 2) ** 0.5
                if d_start > tol:
                    if d_end <= tol:
                        findings.append(
                            ("WARN", row, f"{label}: primitive {i + 2} appears reversed relative to previous one.")
                        )
                    else:
                        findings.append(("WARN", row, f"{label}: primitive {i + 1} and {i + 2} are not connected."))
                    issue_rows.add(row)

            sx, sy, _, _ = primitives[0]
            _, _, ex, ey = primitives[-1]
            closed = (((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5) <= tol

            if closed:
                points = [(sx, sy)] + [(x2, y2) for _, _, x2, y2 in primitives]
                area2 = 0.0
                for i in range(len(points) - 1):
                    x0, y0 = points[i]
                    x1, y1 = points[i + 1]
                    area2 += x0 * y1 - x1 * y0
                orient = "CCW" if area2 > 0 else "CW"
                findings.append(("INFO", row, f"{label}: closed chain, orientation {orient}."))


            else:


                start_connected = bool(chain_end_rows.get(self._point_key(sx, sy, tol), set()) - {row})
                end_connected = bool(chain_end_rows.get(self._point_key(ex, ey, tol), set()) - {row})
                if start_connected and end_connected:
                    findings.append((
                        "INFO", row,
                        f"{label}: open chain; both ends continue into other segments.",
                    ))
                else:
                    findings.append(("WARN", row, f"{label}: open chain (start/end do not close)."))
                    if seg_type in {3, 4, 5}:
                        issue_rows.add(row)

            if ibc > 0 and ibc not in ibc_flags:
                findings.append(("ERROR", row, f"{label}: IBC flag {ibc} is referenced but not defined in IBCS."))
                issue_rows.add(row)

            if seg_type in {3, 4, 5} and pos_mat <= 0:
                findings.append(("ERROR", row, f"{label}: TYPE {seg_type} requires pos_mat > 0."))
                issue_rows.add(row)
            if pos_mat > 0 and pos_mat not in diel_flags:
                findings.append(
                    ("ERROR", row, f"{label}: dielectric flag pos_mat={pos_mat} is referenced but not defined.")
                )
                issue_rows.add(row)
            if seg_type == 5 and neg_mat <= 0:
                findings.append(("ERROR", row, f"{label}: TYPE 5 requires neg_mat > 0."))
                issue_rows.add(row)
            if neg_mat > 0 and neg_mat not in diel_flags:
                findings.append(
                    ("ERROR", row, f"{label}: dielectric flag neg_mat={neg_mat} is referenced but not defined.")
                )
                issue_rows.add(row)
            if seg_type in {1, 2, 3, 4} and neg_mat != 0:
                findings.append(("WARN", row, f"{label}: TYPE {seg_type} typically uses neg_mat=0."))
                issue_rows.add(row)


        global_primitives: 'List[Tuple[int, int, Tuple[float, float, float, float], str]]' = []
        row_type: 'Dict[int, int]' = {}
        for row, seg in enumerate(self.segments):
            self.checkpoint()
            props = self._ensure_prop_len(seg.properties, 5)
            seg_type = self._parse_int_token(props[0], -1)
            row_type[row] = seg_type
            for pidx, prim in enumerate(primitives_by_row[row]):
                global_primitives.append((row, pidx, prim, seg.name))

        endpoint_hits: 'Dict[Tuple[int, int], List[Tuple[int, int, int]]]' = {}
        for row, pidx, (x1, y1, x2, y2), _name in global_primitives:
            k1 = self._point_key(x1, y1, tol)
            k2 = self._point_key(x2, y2, tol)
            endpoint_hits.setdefault(k1, []).append((row, pidx, 0))
            endpoint_hits.setdefault(k2, []).append((row, pidx, 1))

        for _key, hits in endpoint_hits.items():
            incident_rows = sorted({h[0] for h in hits})
            if len(hits) == 1:
                row = hits[0][0]
                if row_type.get(row, -1) in {2, 3, 4, 5}:
                    findings.append(
                        ("WARN", row, f"Row {row + 1}: dangling endpoint not connected to any other primitive.")
                    )
                    issue_rows.add(row)
            if len(hits) > 6:
                row = incident_rows[0]
                findings.append(
                    (
                        "WARN",
                        row,
                        f"Row {row + 1}: high-degree node with {len(hits)} incident primitive endpoints "
                        "(possible non-manifold junction).",
                    )
                )
                issue_rows.add(row)

        max_intersections = 30
        found_intersections = 0
        bounds = primitive_bounds([entry[2] for entry in global_primitives])
        for i, j in overlapping_pairs(bounds, tol, self.checkpoint):
            row_i, pidx_i, prim_i, name_i = global_primitives[i]
            x1, y1, x2, y2 = prim_i
            k_i0 = self._point_key(x1, y1, tol)
            k_i1 = self._point_key(x2, y2, tol)
            row_j, pidx_j, prim_j, name_j = global_primitives[j]
            u1, v1, u2, v2 = prim_j
            k_j0 = self._point_key(u1, v1, tol)
            k_j1 = self._point_key(u2, v2, tol)

            shared_endpoint = k_i0 in {k_j0, k_j1} or k_i1 in {k_j0, k_j1}
            if shared_endpoint:
                continue
            if row_i == row_j and abs(pidx_i - pidx_j) <= 1:
                continue

            if not self._segments_intersect((x1, y1), (x2, y2), (u1, v1), (u2, v2), tol):
                continue

            findings.append(
                (
                    "ERROR",
                    row_i,
                    (
                        f"Rows {row_i + 1} ('{name_i}') and {row_j + 1} ('{name_j}') have a non-endpoint "
                        "primitive intersection."
                    ),
                )
            )
            issue_rows.add(row_i)
            issue_rows.add(row_j)
            found_intersections += 1
            if found_intersections >= max_intersections:
                findings.append(
                    (
                        "WARN",
                        row_i,
                        f"Intersection reporting truncated after {max_intersections} findings.",
                    )
                )
                break


        chain_specs: 'List[ChainSpec]' = []
        for row, seg in enumerate(self.segments):
            self.checkpoint()
            props = self._ensure_prop_len(seg.properties, 5)
            xs, ys = self._segment_plot_xy(seg)
            chain_specs.append(ChainSpec(
                name=seg.name or f"segment_{row + 1}",
                seg_type=self._parse_int_token(props[0], 2),
                pos_mat=self._parse_int_token(props[3], 0),
                points=list(zip(xs, ys)),
            ))
        for severity, chain_idx, message in check_orientation_consistency(chain_specs):
            findings.append((severity, chain_idx, message))
            if severity == "ERROR" and 0 <= chain_idx < len(self.segments):
                issue_rows.add(chain_idx)

        self.checkpoint()
        return findings, issue_rows

