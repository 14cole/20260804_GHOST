import math
import os
from pathlib import Path
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# IBCS row shapes are documented in GEOMETRY_INPUT_CHEATSHEET.md.
IBC_KINDS = ("constant", "linear", "cosine", "exp")
# Legacy compatibility only: a one-token flag above this threshold resolves
# to the historical whitespace/GHz ``mat.<flag>`` sidecar.  New file-backed
# rows use ``flag filename.csv`` for every positive flag value.
TABULATED_FLAG_THRESHOLD = 50


def is_ibc_inline_row(row: 'List[str]') -> 'bool':
    """True if the row is in the 6-token inline form (flag kind R_s X_s R_e X_e)."""
    return (
        len(row) == 6
        and str(row[1]).strip().lower() in IBC_KINDS
    )


def is_file_material_row(row: 'List[str]') -> 'bool':
    """True for the explicit ``flag filename.csv`` material-row shape."""
    return (
        len(row) == 2
        and str(row[1]).strip().lower().endswith(".csv")
    )


def is_legacy_tabulated_row(row: 'List[str]') -> 'bool':
    """True for the deprecated one-token ``flag -> mat.<flag>`` form."""
    if len(row) != 1:
        return False
    try:
        return int(row[0]) > TABULATED_FLAG_THRESHOLD
    except (ValueError, TypeError):
        return False


def is_tabulated_row(row: 'List[str]') -> 'bool':
    """True if the row references an explicit CSV or legacy material table."""
    return is_file_material_row(row) or is_legacy_tabulated_row(row)


def _validate_material_filename(filename: 'str', context: 'str') -> 'str':
    """Validate a portable same-directory CSV material sidecar name."""
    raw_name = str(filename)
    name = raw_name.strip()
    if not name or not name.lower().endswith(".csv"):
        raise ValueError(
            f"{context} must name a .csv file; got {filename!r}."
        )
    # Material rows in the established .geo format are deliberately simple,
    # unquoted whitespace-delimited fields.  Accepting whitespace here would
    # let the GUI create ``1 My Material.csv`` even though the parser must read
    # that as three tokens.  Reject it at the shared boundary so handoffs,
    # direct table edits, snapshot exports, and solver preflight all agree.
    if any(character.isspace() for character in raw_name):
        raise ValueError(
            f"{context} cannot contain whitespace because .geo material rows "
            f"use unquoted fields; got {filename!r}. Rename the CSV and try "
            "again."
        )
    if (
        name in (".", "..")
        or os.path.isabs(name)
        or os.path.basename(name) != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(
            f"{context} must be a filename in the same directory as the "
            f"geometry file (no directory components); got {filename!r}."
        )
    return name


class AtomicFileTransaction:
    """Stage and publish a small related set of files with rollback support.

    ``os.replace`` is atomic for each file but a geometry and its material
    sidecars form a set.  This helper stages every replacement first, keeps a
    same-directory backup of each existing destination, and restores already
    published destinations if a later publication fails.  Call ``commit``
    only after any associated in-memory/UI update has also succeeded; callers
    may call ``rollback`` until then.
    """

    def __init__(self) -> 'None':
        self._staged: 'List[Tuple[Path, Path]]' = []
        self._backups: 'Dict[Path, Optional[Path]]' = {}
        self._published: 'List[Path]' = []
        self._committed = False

    @staticmethod
    def _temporary_path(destination: 'Path', suffix: 'str') -> 'Path':
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=suffix,
            dir=str(destination.parent),
        )
        os.close(fd)
        return Path(temporary_name)

    def _check_destination(self, destination: 'Path') -> 'Path':
        target = Path(destination).expanduser().resolve(strict=False)
        key = os.path.normcase(str(target))
        if any(os.path.normcase(str(existing)) == key for existing, _ in self._staged):
            raise ValueError(
                f"The file transaction contains the destination more than once: {target}"
            )
        if target.exists() and not target.is_file():
            raise OSError(f"save target is not a regular file: {target}")
        return target

    def stage_copy(self, source: 'Path', destination: 'Path') -> 'None':
        """Copy ``source`` to a temporary file beside ``destination``."""

        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"source file does not exist: {source_path}")
        target = self._check_destination(destination)
        temporary = self._temporary_path(target, ".stage")
        try:
            # ``copy2`` propagates the Windows read-only attribute to the
            # staging file, after which opening it for fsync fails even though
            # the source itself was perfectly readable.  Transaction stages
            # are new writable artifacts; copy content without source mode
            # bits or filesystem flags.
            shutil.copyfile(source_path, temporary)
            # Windows requires a writable descriptor for fsync.
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        self._staged.append((target, temporary))

    def stage_text(self, text: 'str', destination: 'Path') -> 'None':
        """Write UTF-8 ``text`` to a temporary file beside ``destination``."""

        target = self._check_destination(destination)
        temporary = self._temporary_path(target, ".stage")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        self._staged.append((target, temporary))

    def publish(self) -> 'None':
        """Publish all staged files, restoring the old set on any failure."""

        if self._committed:
            raise RuntimeError("The file transaction has already been committed.")
        if self._published:
            raise RuntimeError("The file transaction has already been published.")
        try:
            # Back up the complete old set before changing any destination.
            for destination, _temporary in self._staged:
                if destination.exists():
                    backup = self._temporary_path(destination, ".backup")
                    try:
                        # Keep the backup writable while it participates in
                        # publication/rollback; only its bytes are authoritative.
                        shutil.copyfile(destination, backup)
                        with backup.open("r+b") as stream:
                            os.fsync(stream.fileno())
                    except Exception:
                        backup.unlink(missing_ok=True)
                        raise
                    self._backups[destination] = backup
                else:
                    self._backups[destination] = None

            for destination, temporary in self._staged:
                os.replace(temporary, destination)
                self._published.append(destination)
        except Exception as publish_error:
            try:
                self.rollback()
            except Exception as rollback_error:
                raise OSError(
                    f"File publication failed ({publish_error}); rollback also "
                    f"failed ({rollback_error})."
                ) from publish_error
            raise

    def rollback(self) -> 'None':
        """Restore all destinations changed by ``publish`` and remove stages."""

        errors: 'List[str]' = []
        preserved_backups: 'Set[Path]' = set()
        for destination in reversed(self._published):
            backup = self._backups.get(destination)
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except Exception as exc:
                if backup is not None and backup.exists():
                    preserved_backups.add(backup)
                errors.append(f"{destination}: {exc}")

        self._published.clear()
        self._remove_temporary_files(preserved_backups)
        self._backups.clear()
        if errors:
            backup_note = ""
            if preserved_backups:
                backup_note = " Recovery backup(s): " + ", ".join(
                    str(path) for path in sorted(preserved_backups, key=str)
                )
            raise OSError("Could not restore " + "; ".join(errors) + backup_note)

    def commit(self) -> 'None':
        """Accept the published set and remove transaction backups."""

        if self._committed:
            return
        self._committed = True
        self._published.clear()
        self._remove_temporary_files(set())
        self._backups.clear()

    def abort(self) -> 'None':
        """Rollback a published set or discard an unpublished staged set."""

        if self._committed:
            return
        if self._published:
            self.rollback()
        else:
            self._remove_temporary_files(set())
            self._backups.clear()

    def _remove_temporary_files(self, preserve: 'Set[Path]') -> 'None':
        for _destination, temporary in self._staged:
            if temporary not in preserve:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        self._staged.clear()
        for backup in self._backups.values():
            if backup is not None and backup not in preserve:
                try:
                    backup.unlink(missing_ok=True)
                except OSError:
                    pass


def material_filename_from_row(row: 'List[str]') -> 'Optional[str]':
    """Return the sidecar filename referenced by a material row, if any."""
    if is_file_material_row(row):
        return _validate_material_filename(
            row[1], f"Material flag {row[0]} CSV reference"
        )
    if is_legacy_tabulated_row(row):
        return f"mat.{int(row[0])}"
    return None


def _validate_ibc_row(tokens: 'List[str]', lineno_for_err: 'str') -> 'None':
    """Raise ValueError if the row is not a supported IBC shape.

    Supported shapes (the `flag R X` and `flag taper kind ...` forms are not accepted):
      * CSV: `flag filename.csv`, for any positive integer flag.
      * Legacy tabulated: one token `flag` with flag > TABULATED_FLAG_THRESHOLD.
      * Inline: `flag kind R_start X_start R_end X_end` with kind in IBC_KINDS.
        For ``kind == "constant"`` only R_start/X_start matter; the end values
        are placeholders (write 0) and are ignored on read.
    """
    if not tokens:
        return
    try:
        flag = int(tokens[0])
    except (ValueError, TypeError):
        raise ValueError(
            f"IBC row must start with a positive integer flag: "
            f"{lineno_for_err}"
        )
    if flag <= 0:
        raise ValueError(
            f"IBC row must start with a positive integer flag: "
            f"{lineno_for_err}"
        )
    if len(tokens) > 1 and tokens[1].lower() == "thin_dielectric":
        from thin_sheet import ThinLayerDefinition
        ThinLayerDefinition.from_row(tokens)
        return
    if len(tokens) == 2:
        _validate_material_filename(
            tokens[1], f"IBC flag {flag} CSV reference"
        )
        return
    if len(tokens) == 1:
        if flag <= TABULATED_FLAG_THRESHOLD:
            raise ValueError(
                f"IBC flag {flag} has no definition. Use either "
                "'flag filename.csv' or "
                "'flag kind R_start X_start R_end X_end'."
            )
        return
    if len(tokens) != 6:
        raise ValueError(
            f"Inline IBC row must have 6 tokens (flag kind R_start X_start R_end X_end); "
            f"got {len(tokens)}: {lineno_for_err}"
        )
    kind = tokens[1].strip().lower()
    if kind not in IBC_KINDS:
        raise ValueError(
            f"IBC kind must be one of {IBC_KINDS}; got {tokens[1]!r}: {lineno_for_err}"
        )
    try:
        values = [float(token) for token in tokens[2:]]
    except ValueError as exc:
        raise ValueError(
            f"Inline IBC resistance/reactance values must be numeric: "
            f"{lineno_for_err}"
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"Inline IBC resistance/reactance values must be finite: "
            f"{lineno_for_err}"
        )


def _validate_dielectric_row(tokens: 'List[str]', lineno_for_err: 'str') -> 'None':
    """Validate one strict inline, CSV, or legacy dielectric definition."""
    if not tokens:
        return
    try:
        flag = int(tokens[0])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Dielectric row must start with a positive integer flag: "
            f"{lineno_for_err}") from exc
    if flag <= 0:
        raise ValueError(
            f"Dielectric row must start with a positive integer flag: "
            f"{lineno_for_err}"
        )
    if len(tokens) == 2:
        _validate_material_filename(
            tokens[1], f"Dielectric flag {flag} CSV reference"
        )
        return
    if len(tokens) == 1:
        if flag <= TABULATED_FLAG_THRESHOLD:
            raise ValueError(
                f"Dielectric flag {flag} has no definition. Use either "
                "'flag filename.csv' or "
                "'flag eps_real eps_imag mu_real mu_imag'."
            )
        return
    if len(tokens) != 5:
        raise ValueError(
            "Inline dielectric row must have 5 tokens "
            "(flag eps_real eps_imag mu_real mu_imag); "
            f"got {len(tokens)}: {lineno_for_err}")
    try:
        values = [float(token) for token in tokens[1:]]
    except ValueError as exc:
        raise ValueError(
            f"Inline dielectric values must be numeric: "
            f"{lineno_for_err}") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"Inline dielectric values must be finite: {lineno_for_err}")


@dataclass
class Segment:
    name: 'str'
    seg_type: 'Optional[str]'
    properties: 'List[str]'
    x: 'List[float]'
    y: 'List[float]'


def parse_geometry(text: 'str') -> 'Tuple[str, List[Segment], List[List[str]], List[List[str]]]':
    lines = [ln.strip() for ln in text.splitlines()]
    title = "Geometry"
    segments: 'List[Segment]' = []
    ibcs_entries: 'List[List[str]]' = []
    dielectric_entries: 'List[List[str]]' = []

    state = "segments"
    current_name: 'Optional[str]' = None
    current_type: 'Optional[str]' = None
    current_props: 'List[str]' = []
    current_props_seen = False
    cur_x: 'List[float]' = []
    cur_y: 'List[float]' = []

    def flush_segment() -> 'None':
        if current_name is not None:
            if not current_props_seen:
                raise ValueError(
                    f"Segment '{current_name}' is missing its required "
                    "properties line (type n ibc pos_mat neg_mat)."
                )
            segments.append(
                Segment(
                    name=current_name,
                    seg_type=current_type,
                    properties=current_props[:],
                    x=cur_x[:],
                    y=cur_y[:],
                )
            )

    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        low = ln.lower()
        if low.startswith("title"):
            if ":" not in ln:
                raise ValueError(f"Title line must contain ':': {ln}")
            title = ln.split(":", 1)[1].strip() or title
            continue
        if low.startswith("ibcs_resistances:") or low.startswith("ibcs:"):
            if state == "segments":
                flush_segment()
            state = "ibcs"
            continue
        if low.startswith("dielectrics:"):
            if state == "segments":
                flush_segment()
            state = "dielectrics"
            continue

        if state == "segments":
            if low.startswith("segment:"):
                flush_segment()
                parts = ln.split(":", 1)[1].strip().split()
                if not parts:
                    current_name, current_type = "Unnamed", None
                elif len(parts) == 1:
                    current_name, current_type = parts[0], None
                else:
                    current_name, current_type = parts[0], parts[1]
                current_props = []
                current_props_seen = False
                cur_x.clear()
                cur_y.clear()
                continue
            if low.startswith("properties:"):
                if current_name is None:
                    raise ValueError(
                        "properties line appears before the first "
                        f"'Segment:' header: {ln}"
                    )
                if current_props_seen:
                    raise ValueError(
                        f"Segment '{current_name}' has more than one "
                        "properties line; exactly one is required."
                    )
                current_props = ln.split(":", 1)[1].strip().split()
                if len(current_props) != 5:
                    raise ValueError(
                        f"properties line must have exactly 5 fields "
                        f"(type n ibc pos_mat neg_mat); got {len(current_props)}: {ln}"
                    )
                current_props_seen = True
                continue

            tokens = ln.split()
            if len(tokens) != 4:
                raise ValueError(f"Geometry line must have 4 numbers, got {len(tokens)} {ln}")
            if current_name is None:
                # Without this, coordinates before the first "Segment:" header
                # were accumulated and then silently dropped by flush_segment.
                raise ValueError(
                    f"Geometry data line appears before the first 'Segment:' header: {ln}"
                )
            try:
                x1, y1, x2, y2 = map(float, tokens)
            except ValueError:
                raise ValueError(f"Geometry line must contain valid numbers: {ln}")
            cur_x.extend([x1, x2])
            cur_y.extend([y1, y2])
        elif state == "ibcs":
            tokens = ln.split()
            if tokens:
                _validate_ibc_row(tokens, ln)
                ibcs_entries.append(list(tokens))
        elif state == "dielectrics":
            tokens = ln.split()
            if tokens:
                _validate_dielectric_row(tokens, ln)
                dielectric_entries.append(tokens)

    if state == "segments":
        flush_segment()

    return title, segments, ibcs_entries, dielectric_entries


def build_geometry_text(
    title: 'str',
    segments: 'List[Segment]',
    ibcs_entries: 'List[List[str]]',
    dielectric_entries: 'List[List[str]]',
) -> 'str':
    lines: 'List[str]' = [f"Title: {title}"]
    for seg in segments:
        segment_name = str(seg.name).strip()
        if not segment_name or any(char.isspace() for char in segment_name):
            raise ValueError(
                f"Segment name {seg.name!r} is not representable in .geo files; "
                "use a non-empty name without whitespace."
            )
        if ":" in segment_name or "#" in segment_name:
            raise ValueError(
                f"Segment name {seg.name!r} contains reserved .geo punctuation."
            )
        props = list(seg.properties)
        # Prefer the segment's declared type; fall back to properties[0] only if
        # seg_type is missing. This keeps load -> save idempotent when the Segment:
        # header and properties[0] disagree.
        effective_type = seg.seg_type
        if not effective_type and props and str(props[0]).strip():
            effective_type = props[0]
        if effective_type:
            lines.append(f"Segment: {segment_name} {effective_type}")
        else:
            lines.append(f"Segment: {segment_name}")

        if len(props) < 5:
            props.extend([""] * (5 - len(props)))
        elif len(props) > 5:
            props = props[:5]
        # Blank fields must be written as concrete tokens: space-joined blanks
        # vanish on split(), producing a file parse_geometry rejects ("must
        # have exactly 5 fields").  Blank TYPE falls back to the effective
        # segment type; blank N means auto density (0); blank flags are 0.
        type_token = str(props[0]).strip() if props[0] is not None and str(props[0]).strip() else str(effective_type or "2")
        out_props = [type_token]
        for p in props[1:]:
            token = str(p).strip() if p is not None else ""
            out_props.append(token if token else "0")
        lines.append("properties: " + " ".join(out_props))

        if len(seg.x) != len(seg.y) or len(seg.x) % 2 != 0:
            raise ValueError(
                f"Segment {seg.name} has mismatched or odd number of coordinates."
            )
        for i in range(0, len(seg.x), 2):
            x1, y1, x2, y2 = seg.x[i], seg.y[i], seg.x[i + 1], seg.y[i + 1]
            # repr() of a Python float is the shortest string that round-trips
            # exactly: clean values stay clean ("0.05"), everything else keeps
            # full precision.  The old "%.4f" silently truncated coordinates
            # to 4 decimals (0.1 mm resolution in meters mode), corrupting
            # fine features and breaking closed contours on reload.
            lines.append(f"{float(x1)!r} {float(y1)!r} {float(x2)!r} {float(y2)!r}")

    lines.append("IBCS_Resistances:")
    for raw_row in ibcs_entries:
        raw_tokens = [str(token) for token in raw_row]
        _validate_ibc_row(raw_tokens, " ".join(raw_tokens))
        row = [token.strip() for token in raw_tokens]
        if row:
            lines.append(" ".join(row))
    lines.append("Dielectrics:")
    for raw_row in dielectric_entries:
        raw_tokens = [str(token) for token in raw_row]
        _validate_dielectric_row(raw_tokens, " ".join(raw_tokens))
        row = [token.strip() for token in raw_tokens]
        if row:
            lines.append(" ".join(row))
    return "\n".join(lines) + "\n"


def snapshot_to_geometry_text(snapshot: 'Dict[str, Any]') -> 'str':
    """Serialize a geometry SNAPSHOT dict (the {title, segments:[{name, seg_type,
    properties, point_pairs}], ibcs, dielectrics} form used by the solvers and
    the feature pipeline) to .geo text.  Bridges the dict form to
    build_geometry_text, which wants Segment objects."""
    segs: 'List[Segment]' = []
    for s in snapshot.get("segments", []):
        x: 'List[float]' = []
        y: 'List[float]' = []
        for pp in s.get("point_pairs", []):
            x.extend([float(pp["x1"]), float(pp["x2"])])
            y.extend([float(pp["y1"]), float(pp["y2"])])
        segs.append(Segment(str(s.get("name", "seg")),
                            (str(s["seg_type"]) if s.get("seg_type") is not None else None),
                            [str(p) for p in s.get("properties", [])], x, y))
    return build_geometry_text(str(snapshot.get("title", "geometry")), segs,
                               list(snapshot.get("ibcs", [])),
                               list(snapshot.get("dielectrics", [])))


def save_snapshot_geo(snapshot: 'Dict[str, Any]', path: 'str') -> 'str':
    """Write a geometry snapshot dict to a .geo file; returns the path."""
    out = path if path.lower().endswith(".geo") else path + ".geo"
    with open(out, "w") as fh:
        fh.write(snapshot_to_geometry_text(snapshot))
    return out


def material_sidecar_paths(geometry_path: 'str') -> 'List[str]':
    """Return the exact material files referenced by a saved geometry.

    Missing sidecars raise here so cache fingerprints and HPC staging cannot
    silently omit a physical input.
    """

    geo_path = os.path.abspath(str(geometry_path))
    with open(geo_path, "r") as geo_file:
        _title, _segments, ibcs, dielectrics = parse_geometry(geo_file.read())
    folder = os.path.dirname(geo_path)
    names: 'Set[str]' = set()
    for row in list(ibcs) + list(dielectrics):
        filename = material_filename_from_row(row)
        if filename:
            names.add(filename)
    paths: 'List[str]' = []
    for name in sorted(names):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Could not locate referenced material file {name} beside "
                f"geometry file {geo_path}."
            )
        paths.append(path)
    return paths


# -----------------------------------------------------------------------------
# Orientation / material-side consistency checks (shared by the GUI validator
# and the solver preflight).
#
# Drawing convention (see rcs_solver._apply_user_convention_flip):
#   TYPE 1 sheet:        both sides air; orientation irrelevant.
#   TYPE 2 (air|PEC):    normal (left of travel) points INTO AIR.
#   TYPE 3 (air|diel):   normal points INTO AIR; pos_mat behind.
#   TYPE 4 (diel|PEC):   normal points INTO the pos_mat dielectric; PEC behind.
#   TYPE 5 (diel|diel):  normal points INTO pos_mat; neg_mat behind.
#
# The TM formulations are winding-insensitive, but the TE rows carry a +/-1/2
# mass jump tied to the normal direction, so a wrong winding or an
# inconsistent air side silently corrupts TE results.  These checks make
# that a loud, named error instead.
# -----------------------------------------------------------------------------


@dataclass
class ChainSpec:
    """Neutral view of one segment's primitive chain for orientation checks."""

    name: 'str'
    seg_type: 'int'
    pos_mat: 'int'
    points: 'List[Tuple[float, float]]' = field(default_factory=list)


def _chain_is_closed(points: 'List[Tuple[float, float]]', tol: 'float') -> 'bool':
    if len(points) < 4:
        return False
    return math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= tol


def _chain_area2(points: 'List[Tuple[float, float]]') -> 'float':
    area2 = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        area2 += x0 * y1 - x1 * y0
    return area2


def _point_in_polygon(px: 'float', py: 'float', poly: 'List[Tuple[float, float]]') -> 'bool':
    """Even-odd ray casting; poly is a closed vertex chain (first ~= last)."""

    inside = False
    for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
        if (y1 > py) != (y2 > py):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_cross:
                inside = not inside
    return inside


def _rep_point(points: 'List[Tuple[float, float]]') -> 'Tuple[float, float]':
    """A point strictly on the chain: midpoint of the first primitive."""

    return (0.5 * (points[0][0] + points[1][0]), 0.5 * (points[0][1] + points[1][1]))


def _geometry_tolerance(chains: 'List["ChainSpec"]') -> 'float':
    xs = [p[0] for c in chains for p in c.points]
    ys = [p[1] for c in chains for p in c.points]
    if not xs:
        return 1e-9
    diag = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
    return max(1e-12, 1e-9 * diag)


def check_orientation_consistency(
    chains: 'List[ChainSpec]',
    tol: 'Optional[float]' = None,
) -> 'List[Tuple[str, int, str]]':
    """
    Validate winding and air-side consistency of a set of segment chains.

    Returns findings as (severity, chain_index, message) with severity in
    {"ERROR", "INFO"}.  ERROR findings mean the geometry, solved as drawn,
    would produce silently wrong TE results:

    1. A closed TYPE 2/3 chain whose winding puts the air side on the wrong
       side for its nesting depth (a top-level body must be CW = normals out
       into unbounded air; a void nested inside a body must be CCW = normals
       into the enclosed air; parity alternates with depth).
    2. A closed TYPE 4 chain enclosed by a matching-pos_mat TYPE 3/5 contour
       (standard coated-body layout) wound so its normal points away from
       the coating.
    3. Two open TYPE 2/3 chains meeting end-to-end or start-to-start at a
       degree-2 junction: their air sides disagree there (e.g. an
       air|dielectric wall chained to an air|PEC wall the wrong way round).
    4. A closed loop stitched head-to-tail from open TYPE 2/3 chains whose
       overall winding is inside-out for its nesting depth (consistent with
       each other, but all reversed together).

    TYPE 5 chains and TYPE 1 sheets are never flagged: for TYPE 5 the
    winding IS the user's pos/neg labeling choice, and TYPE 1 is symmetric.
    """

    findings: 'List[Tuple[str, int, str]]' = []
    if not chains:
        return findings
    if tol is None:
        tol = _geometry_tolerance(chains)

    closed_polys: 'List[Tuple[int, List[Tuple[float, float]]]]' = []  # (chain idx, poly)
    open_air_chains: 'List[int]' = []
    for idx, ch in enumerate(chains):
        if len(ch.points) < 2:
            continue
        if _chain_is_closed(ch.points, tol):
            if abs(_chain_area2(ch.points)) > 0.0:
                closed_polys.append((idx, ch.points))
        elif ch.seg_type in (2, 3):
            open_air_chains.append(idx)

    def _depth(rep: 'Tuple[float, float]', exclude_idx: 'int') -> 'int':
        return sum(
            1 for cidx, poly in closed_polys
            if cidx != exclude_idx and _point_in_polygon(rep[0], rep[1], poly)
        )

    # -- 1 & 2: closed-chain winding --------------------------------------
    for idx, poly in closed_polys:
        ch = chains[idx]
        drawn_ccw = _chain_area2(poly) > 0.0
        if ch.seg_type in (2, 3):
            depth = _depth(_rep_point(poly), idx)
            expected_ccw = (depth % 2) == 1
            if drawn_ccw != expected_ccw:
                where = (
                    "the air side is the unbounded exterior" if depth % 2 == 0
                    else "the air side is the enclosed interior (nested void)"
                )
                findings.append((
                    "ERROR", idx,
                    f"Segment '{ch.name}' (TYPE {ch.seg_type}) is a closed contour drawn "
                    f"{'CCW' if drawn_ccw else 'CW'}, but {where}, so the drawing convention "
                    f"(normal into air) requires {'CCW' if expected_ccw else 'CW'} winding. "
                    "Reverse the segment's endpoint order.",
                ))
        elif ch.seg_type == 4:
            rep = _rep_point(poly)
            has_matching_parent = any(
                cidx != idx
                and chains[cidx].seg_type in (3, 5)
                and chains[cidx].pos_mat == ch.pos_mat
                and _point_in_polygon(rep[0], rep[1], p)
                for cidx, p in closed_polys
            )
            if has_matching_parent and drawn_ccw:
                findings.append((
                    "ERROR", idx,
                    f"Segment '{ch.name}' (TYPE 4) is a closed contour drawn CCW, but the "
                    f"pos_mat={ch.pos_mat} coating lies outside it, so the drawing convention "
                    "(normal into the dielectric) requires CW winding. "
                    "Reverse the segment's endpoint order.",
                ))

    # -- 3: air-side continuity at degree-2 junctions of open T2/T3 chains -
    def _key(p: 'Tuple[float, float]') -> 'Tuple[int, int]':
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))

    # endpoint registry: key -> list of (chain idx, 'start'|'end')
    ends: 'Dict[Tuple[int, int], List[Tuple[int, str]]]' = {}
    for idx in open_air_chains:
        pts = chains[idx].points
        ends.setdefault(_key(pts[0]), []).append((idx, "start"))
        ends.setdefault(_key(pts[-1]), []).append((idx, "end"))

    adjacency: 'Dict[int, List[Tuple[str, int, str]]]' = {i: [] for i in open_air_chains}
    for key, members in ends.items():
        if len(members) != 2:
            continue  # free end, or junction of degree > 2 -- no pairing implied
        (ia, ea), (ib, eb) = members
        if ia == ib:
            continue
        adjacency[ia].append((ea, ib, eb))
        adjacency[ib].append((eb, ia, ea))
        if ea == eb:  # end-to-end or start-to-start: air sides disagree
            ca, cb = chains[ia], chains[ib]
            findings.append((
                "ERROR", ia,
                f"Segments '{ca.name}' (TYPE {ca.seg_type}) and '{cb.name}' "
                f"(TYPE {cb.seg_type}) meet {ea}-to-{eb} at "
                f"({chains[ia].points[0 if ea == 'start' else -1][0]:.6g}, "
                f"{chains[ia].points[0 if ea == 'start' else -1][1]:.6g}): "
                "their air sides point to opposite sides of the boundary there. "
                "Reverse one segment's endpoint order so the chains run head-to-tail.",
            ))

    # -- 4: winding of loops stitched from consistently-oriented open chains -
    visited: 'set' = set()
    for start_idx in open_air_chains:
        if start_idx in visited or len(adjacency[start_idx]) != 2:
            continue
        # Walk forward from this chain's end until we return or hit a break.
        loop = [start_idx]
        cur, arrived_via = start_idx, "end"
        ok = True
        while True:
            nxt = next(
                ((jb, eb) for (ea, jb, eb) in adjacency[cur] if ea == arrived_via),
                None,
            )
            if nxt is None:
                ok = False
                break
            jdx, joint_end = nxt
            if jdx == start_idx:
                break
            if jdx in loop or joint_end != "start":
                # revisited chain mid-walk, or an inconsistent junction
                # (already reported by check 3) -- skip loop-level check.
                ok = False
                break
            loop.append(jdx)
            cur, arrived_via = jdx, "end"
        visited.update(loop)
        if not ok or len(loop) < 2:
            continue
        stitched: 'List[Tuple[float, float]]' = []
        for idx in loop:
            pts = chains[idx].points
            stitched.extend(pts if not stitched else pts[1:])
        if not _chain_is_closed(stitched, tol) or abs(_chain_area2(stitched)) <= 0.0:
            continue
        drawn_ccw = _chain_area2(stitched) > 0.0
        depth = sum(
            1 for cidx, poly in closed_polys
            if cidx not in loop and _point_in_polygon(*_rep_point(stitched), poly)
        )
        expected_ccw = (depth % 2) == 1
        if drawn_ccw != expected_ccw:
            names = ", ".join(f"'{chains[i].name}'" for i in loop)
            findings.append((
                "ERROR", loop[0],
                f"Segments {names} form a closed boundary drawn "
                f"{'CCW' if drawn_ccw else 'CW'}, but the drawing convention (normal "
                f"into air) requires {'CCW' if expected_ccw else 'CW'} winding for this "
                "loop. Reverse every segment in the loop.",
            ))

    return findings


def chains_from_snapshot_segments(segments: 'List[Dict[str, Any]]') -> 'List[ChainSpec]':
    """Build ChainSpecs from solver-snapshot segment dicts (point_pairs form)."""

    chains: 'List[ChainSpec]' = []
    for seg_idx, seg in enumerate(segments):
        props = list(seg.get("properties", []) or [])

        def _flag(tok: 'Any', default: 'int' = 0) -> 'int':
            try:
                text = str(tok).strip().lower()
                if text.startswith("mat."):
                    text = text[4:]
                return int(float(text))
            except (ValueError, TypeError):
                return default

        seg_type = _flag(props[0], 2) if len(props) > 0 and str(props[0]).strip() else _flag(seg.get("seg_type", 2), 2)
        pos_mat = _flag(props[3]) if len(props) > 3 else 0
        pts: 'List[Tuple[float, float]]' = []
        for i, pair in enumerate(list(seg.get("point_pairs", []) or [])):
            try:
                x1 = float(pair.get("x1", 0.0)); y1 = float(pair.get("y1", 0.0))
                x2 = float(pair.get("x2", 0.0)); y2 = float(pair.get("y2", 0.0))
            except (TypeError, ValueError):
                continue
            if i == 0:
                pts.append((x1, y1))
            pts.append((x2, y2))
        chains.append(ChainSpec(
            name=str(seg.get("name", f"segment_{seg_idx + 1}")),
            seg_type=seg_type,
            pos_mat=pos_mat,
            points=pts,
        ))
    return chains


def build_geometry_snapshot(
    title: 'str',
    segments: 'List[Segment]',
    ibcs_entries: 'List[List[str]]',
    dielectric_entries: 'List[List[str]]',
) -> 'Dict[str, Any]':
    segments_payload = []
    for seg in segments:
        if len(seg.x) != len(seg.y) or len(seg.x) % 2 != 0:
            raise ValueError(
                f"Segment {seg.name} has mismatched or odd number of coordinates."
            )
        point_pairs = []
        for i in range(0, len(seg.x), 2):
            point_pairs.append(
                {
                    "x1": seg.x[i],
                    "y1": seg.y[i],
                    "x2": seg.x[i + 1],
                    "y2": seg.y[i + 1],
                }
            )
        props = list(seg.properties)
        effective_type = seg.seg_type
        if not effective_type and props and str(props[0]).strip():
            effective_type = props[0]
        segments_payload.append(
            {
                "name": seg.name,
                "seg_type": effective_type,
                "properties": props,
                "point_pairs": point_pairs,
            }
        )

    return {
        "title": title,
        "segment_count": len(segments),
        "segments": segments_payload,
        "ibcs": [list(row) for row in ibcs_entries],
        "dielectrics": [list(row) for row in dielectric_entries],
    }
