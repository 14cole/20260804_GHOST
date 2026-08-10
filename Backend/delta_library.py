#!/usr/bin/env python3
"""
delta_library.py -- a FILENAME-INDEXED library of seam delta .grim files.

The parameters of a delta live in its FILENAME and nowhere else:

    0.020bmag_0.060gap.grim        bmag = 0.020 m, gap = 0.060 m

    <value><key>_<value><key>_... .grim

Rules that make the filesystem itself the parameter key:

  * tokens are sorted by key, so one parameter point has exactly ONE legal
    spelling -- the OS's "no two files with the same name" then means "no two
    files at the same parameter point";
  * every value of a given key uses the SAME number of decimals across the
    library (the scan INFERS that width from the folder and rejects
    disagreement) -- otherwise 0.02gap.grim and 0.020gap.grim are two files at
    one point and nothing notices;
  * the seam TYPE is the DIRECTORY, never a token -- different seam types are
    not one numeric family and must never be interpolated across:

        library/seal/0.020bmag_0.060gap.grim
        library/panel_gap/0.020bmag_0.060gap.grim     (no conflict)

  * ``rev`` is a reserved key (``2rev``) for a RE-SOLVE of the same parameter
    point (finer mesh, solver fix, measured vs simulated).  It is excluded from
    the axes and from sweeps, and is used only to tie-break, loudly.
  * units are the library's convention -- metres, like everything else
    post-solve in this toolchain -- and are NOT encoded in the name.

CONSEQUENCE OF FILENAME-ONLY, stated once: a hand-edited filename is
undetectable.  Nothing in the file cross-checks it, so a rename silently
changes what a delta claims to be.  Treat the names as data, not decoration.

Typical use -- pin the configuration, spread a tolerance around a door:

    lib = DeltaLibrary.from_dir("library/seal")
    lib.summary()                                   # axes, counts, unindexed
    fam = lib.select(bmag=0.020)                    # config -> 1-D gap family
    res = fam.resolve(gap=Range(0.030, 0.080), n=6) # tolerance -> 6 entries
    print(res.report)                               # every snap, named
    pl  = tolerance_placements(perimeter, res.entries)
    out = sum_features(body, pl, dirs, 6.0, generatrix=gen, mode="hybrid")

Gated by tests/validate_delta_library.py.
"""

import glob
import math
import os
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

REV_KEY = "rev"                      # reserved: a re-solve discriminator, not an axis
STUDY_KEY = "study"                  # reserved: the categorical study id, not an axis
_TOKEN = re.compile(r"^(-?[0-9]+(?:\.[0-9]+)?)([A-Za-z][A-Za-z0-9]*)$")


class Range(NamedTuple):
    """Inclusive selection range.  A plain 2-tuple is accepted as one too, but
    a LIST always means "these explicit values" -- keep the two apart."""
    lo: 'float'
    hi: 'float'


class DeltaEntry(NamedTuple):
    path: 'str'
    params: 'Dict[str, float]'         # physics keys only (no rev, no study)
    rev: 'float'                       # 0.0 when the name carries no rev token
    study: 'Optional[str]' = None      # categorical: seam type / study / version

    def label(self) -> 'str':
        return os.path.basename(self.path)


class Resolution(NamedTuple):
    entries: 'List[DeltaEntry]'
    report: 'List[str]'                # one line per substitution -- NEVER dropped
    requested: 'List[float]'

    @property
    def paths(self) -> 'List[str]':
        return [e.path for e in self.entries]


# -----------------------------------------------------------------------------
# Names
# -----------------------------------------------------------------------------

def parse_name(path: 'str') -> 'Tuple[Dict[str, float], Dict[str, int]]':
    """Parse a library filename -> (values, decimals) keyed by variable name.

    Accepts an optional leading STUDY ID -- ``SEAL-00-01_0.010gap.grim`` -- which
    is returned under the key ``study`` in a parallel call to ``parse_study``.
    The id is CATEGORICAL (a seam type / study / version), never an axis, so it
    is excluded from the numeric parameters here.

    Raises ValueError with the offending token; callers collect the failures
    into the scan report rather than skipping the file silently."""
    base = os.path.basename(str(path))
    if not base.lower().endswith(".grim"):
        raise ValueError(f"{base}: not a .grim file.")
    stem = base[: -len(".grim")]
    if not stem:
        raise ValueError(f"{base}: empty name.")
    toks = stem.split("_")
    vals: 'Dict[str, float]' = {}
    decs: 'Dict[str, int]' = {}
    for i, tok in enumerate(toks):
        m = _TOKEN.match(tok)
        if not m:
            if i == 0 and len(toks) > 1:
                continue                      # the study id (see parse_study)
            raise ValueError(f"{base}: token {tok!r} is not <value><key> "
                             f"(e.g. 0.060gap).")
        num, key = m.group(1), m.group(2)
        if key in vals:
            raise ValueError(f"{base}: key {key!r} appears twice.")
        vals[key] = float(num)
        decs[key] = len(num.split(".")[1]) if "." in num else 0
    if not vals:
        raise ValueError(f"{base}: no <value><key> parameter token.")
    return vals, decs


def parse_study(path: 'str') -> 'Optional[str]':
    """The leading study id of a library filename, or None if it has none.
    ``SEAL-00-01_0.010gap.grim`` -> 'SEAL-00-01';  ``0.010gap.grim`` -> None."""
    base = os.path.basename(str(path))
    stem = base[: -len(".grim")] if base.lower().endswith(".grim") else base
    toks = stem.split("_")
    if len(toks) > 1 and not _TOKEN.match(toks[0]):
        return toks[0]
    return None


def format_name(params: 'Dict[str, float]', decimals: 'Dict[str, int]',
                rev: 'float' = 0.0, study: 'Optional[str]' = None) -> 'str':
    """Canonical filename for a parameter point: the study id first (if any),
    then tokens sorted by key, each value at that key's declared decimal width."""
    p = dict(params)
    if rev:
        p[REV_KEY] = rev
    missing = [k for k in p if k not in decimals]
    if missing:
        raise ValueError(f"no decimal width declared for {missing!r}.")
    toks = [f"{p[k]:.{decimals[k]}f}{k}" for k in sorted(p)]
    if study:
        if "_" in str(study):
            raise ValueError(f"study id {study!r} must not contain '_' "
                             f"(it is the token separator).")
        toks = [str(study)] + toks
    return "_".join(toks) + ".grim"


# -----------------------------------------------------------------------------
# The library
# -----------------------------------------------------------------------------

class DeltaLibrary:
    """One seam family = one directory of filename-indexed delta .grim files."""

    def __init__(self, entries: 'Sequence[DeltaEntry]', decimals: 'Dict[str, int]',
                 root: 'str' = "", unindexed: 'Sequence[Tuple[str, str]]' = ()):
        self.entries: 'List[DeltaEntry]' = list(entries)
        self.decimals: 'Dict[str, int]' = dict(decimals)
        self.root = str(root)
        self.unindexed: 'List[Tuple[str, str]]' = list(unindexed)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_dir(cls, directory: 'str', decimals: 'Optional[Dict[str, int]]' = None
                 ) -> "DeltaLibrary":
        """Scan ONE seam-family directory (not recursive -- the seam type IS the
        directory; use ``families()`` for a whole library root).

        Raises on anything that would make the filename an unreliable key:
        mixed decimal widths for a key, a file missing a key the others carry,
        a non-canonical spelling, or two files at the same (params, rev).
        Files whose names do not parse are NOT silently skipped -- they land in
        ``self.unindexed`` and are shown by ``summary()``."""
        root = str(directory)
        if not os.path.isdir(root):
            raise ValueError(f"{root}: not a directory.")
        paths = sorted(glob.glob(os.path.join(root, "*.grim")))
        if not paths:
            raise ValueError(f"{root}: no .grim files found.")

        parsed: 'List[Tuple[str, Dict[str, float], Dict[str, int]]]' = []
        unindexed: 'List[Tuple[str, str]]' = []
        for p in paths:
            try:
                v, d = parse_name(p)
            except ValueError as exc:
                unindexed.append((p, str(exc)))
                continue
            parsed.append((p, v, d))
        if not parsed:
            raise ValueError(f"{root}: no filename parsed as a parameter point "
                             f"({len(unindexed)} unindexed).")

        # one decimal width per key, library-wide
        widths: 'Dict[str, Dict[int, List[str]]]' = {}
        for p, _v, d in parsed:
            for k, w in d.items():
                widths.setdefault(k, {}).setdefault(w, []).append(os.path.basename(p))
        if decimals is None:
            decimals = {}
            for k, seen in widths.items():
                if len(seen) > 1:
                    detail = "; ".join(f"{w} dp: {', '.join(sorted(f)[:3])}"
                                       for w, f in sorted(seen.items()))
                    raise ValueError(
                        f"{root}: key {k!r} is written with several decimal widths "
                        f"({detail}).  One point would then have two legal names -- "
                        f"pick one width and rename.")
                decimals[k] = next(iter(seen))
        decimals = dict(decimals)

        # every file carries the same physics keys (rev is optional)
        keysets = {frozenset(k for k in v if k != REV_KEY) for _p, v, _d in parsed}
        if len(keysets) > 1:
            want = max(keysets, key=len)
            bad = [os.path.basename(p) for p, v, _d in parsed
                   if frozenset(k for k in v if k != REV_KEY) != want]
            raise ValueError(f"{root}: files disagree on which variables they carry "
                             f"(most carry {sorted(want)}); odd ones out: {bad[:4]}")

        entries: 'List[DeltaEntry]' = []
        for p, v, _d in parsed:
            rev = float(v.get(REV_KEY, 0.0))
            phys = {k: x for k, x in v.items() if k != REV_KEY}
            study = parse_study(p)
            canon = format_name(phys, decimals, rev=rev, study=study)
            if canon != os.path.basename(p):
                raise ValueError(f"{p}: non-canonical name -- should be {canon!r} "
                                 f"(study id first, then tokens sorted by key at "
                                 f"a fixed decimal width).")
            entries.append(DeltaEntry(p, phys, rev, study))

        seen: 'Dict[Tuple, str]' = {}
        for e in entries:
            key = (e.study, tuple(sorted(e.params.items())), e.rev)
            if key in seen:
                raise ValueError(f"{root}: two files at the same parameter point "
                                 f"{e.params} rev={e.rev:g}: {seen[key]} and {e.path}.")
            seen[key] = e.path
        return cls(entries, decimals, root, unindexed)

    # -- description ---------------------------------------------------------
    def __len__(self) -> 'int':
        return len(self.entries)

    def __repr__(self) -> 'str':
        return (f"DeltaLibrary({os.path.basename(self.root) or '?'}, "
                f"{len(self.entries)} entries, axes={list(self.axes())})")

    def keys(self) -> 'List[str]':
        return sorted({k for e in self.entries for k in e.params})

    def axes(self) -> 'Dict[str, List[float]]':
        """Available values per physics variable (rev excluded)."""
        out: 'Dict[str, List[float]]' = {}
        for k in self.keys():
            out[k] = sorted({e.params[k] for e in self.entries})
        return out

    def revs(self) -> 'List[float]':
        return sorted({e.rev for e in self.entries})

    def studies(self) -> 'List[str]':
        """The study ids present.  CATEGORICAL: different seam types or versions,
        so you select one -- resolve() refuses to span several."""
        return sorted({e.study for e in self.entries if e.study is not None})

    def missing_points(self) -> 'List[Dict[str, float]]':
        """Parameter points the full rectangular grid would have but the library
        lacks.  A ragged library silently biases any sweep across it."""
        ax = self.axes()
        if not ax:
            return []
        have = {tuple(sorted(e.params.items())) for e in self.entries}
        grid = [{}]
        for k, vals in ax.items():
            grid = [dict(g, **{k: v}) for g in grid for v in vals]
        return [g for g in grid if tuple(sorted(g.items())) not in have]

    def is_rectangular(self) -> 'bool':
        return not self.missing_points()

    def summary(self, show: 'int' = 6) -> 'str':
        ax = self.axes()
        lines = [f"delta library: {self.root}  ({len(self.entries)} entries)"]
        for k, vals in ax.items():
            shown = ", ".join(f"{v:g}" for v in vals[:show])
            more = "" if len(vals) <= show else f", ... (+{len(vals)-show})"
            lines.append(f"  {k:<8} {len(vals):>3} values: {shown}{more}")
        if self.studies():
            lines.append(f"  {'study':<8} {len(self.studies()):>3} value(s): "
                         f"{', '.join(self.studies())}   (categorical)")
        if self.revs() != [0.0]:
            lines.append(f"  {'rev':<8} {len(self.revs()):>3} values: "
                         f"{', '.join(f'{r:g}' for r in self.revs())}")
        miss = self.missing_points()
        lines.append(f"  grid: {'rectangular' if not miss else f'RAGGED -- {len(miss)} points missing'}")
        if self.unindexed:
            lines.append(f"  UNINDEXED ({len(self.unindexed)} file(s) present but not in the library):")
            for p, why in self.unindexed[:show]:
                lines.append(f"    {os.path.basename(p)}: {why}")
        text = "\n".join(lines)
        print(text)
        return text

    # -- selection (on-grid, no snapping) ------------------------------------
    def select(self, **constraints) -> "DeltaLibrary":
        """Subset by exact value, explicit list, ``Range``/2-tuple, or predicate.

            lib.select(bmag=0.020)                     exact
            lib.select(gap=[0.03, 0.05])               these values
            lib.select(gap=Range(0.03, 0.08))          inclusive range
            lib.select(gap=lambda g: g > 0.05)         predicate
            lib.select(rev=2)                          pick a re-solve

        Unknown variable names raise (typo protection).  An empty result raises
        -- a silently empty selection is the worst outcome of all."""
        known = set(self.keys()) | {REV_KEY, STUDY_KEY}
        bad = [k for k in constraints if k not in known]
        if bad:
            raise ValueError(f"unknown variable(s) {bad!r}; library has {sorted(known)}.")
        kept = self.entries
        for k, spec in constraints.items():
            tol = self._tol(k)
            kept = [e for e in kept if self._match(self._value(e, k), spec, tol)]
        if not kept:
            raise ValueError(f"no delta matches {constraints!r}; axes are {self.axes()}.")
        return DeltaLibrary(kept, self.decimals, self.root, self.unindexed)

    def _value(self, e: 'DeltaEntry', k: 'str'):
        if k == REV_KEY:
            return e.rev
        if k == STUDY_KEY:
            return e.study
        return e.params[k]

    def _tol(self, k: 'str') -> 'float':
        return 0.5 * 10.0 ** (-int(self.decimals.get(k, 6)))

    @staticmethod
    def _match(v, spec: 'Any', tol: 'float') -> 'bool':
        if isinstance(v, str) or isinstance(spec, str):      # study: categorical
            if callable(spec):
                return bool(spec(v))
            if isinstance(spec, (list, set, frozenset, tuple)):
                return str(v) in {str(x) for x in spec}
            return str(v) == str(spec)
        if callable(spec):
            return bool(spec(v))
        if isinstance(spec, Range) or (isinstance(spec, tuple) and len(spec) == 2):
            return float(spec[0]) - tol <= v <= float(spec[1]) + tol
        if isinstance(spec, (list, set, frozenset, np.ndarray)):
            return any(abs(v - float(x)) <= tol for x in spec)
        return abs(v - float(spec)) <= tol

    # -- resolution (may snap OFF-grid requests) -----------------------------
    def resolve(self, n: 'Optional[int]' = None, off_grid: 'str' = "snap",
                prefer_rev: 'str' = "highest", **spec) -> 'Resolution':
        """Turn a TOLERANCE request into an ordered list of library entries.

        Exactly one variable may be requested, and every other variable must
        already be pinned to a single value (``select`` first) -- resolution is
        a 1-D operation by design.

            fam.resolve(gap=[0.035, 0.045])            explicit widths
            fam.resolve(gap=Range(0.03, 0.08), n=6)    min/max over 6 arcs
            fam.resolve(gap=Range(0.03, 0.08))         every node in range

        ``off_grid``: ``"snap"`` (nearest node, reported), or ``"error"``.
        A request OUTSIDE the axis is always an error -- the library is never
        extrapolated: below ~0.17 lambda a seam is negligible and above ~1
        lambda the calibrated psi drifts, so the curve changes character at
        both ends (see tests/validate_line_expansion_size.py).

        Interpolation is deliberately NOT offered here: it needs a magnitude
        and unwrapped-phase check between the bracketing nodes, and when the
        grid is too coarse the honest fix is to solve the intermediate coupon.

        The returned ``report`` lists every substitution.  Do not drop it."""
        if off_grid not in ("snap", "error"):
            raise ValueError(f"off_grid must be 'snap' or 'error', got {off_grid!r}.")
        if len(spec) != 1:
            raise ValueError(f"resolve() takes exactly one variable, got {list(spec)}.")
        key, req = next(iter(spec.items()))
        if key not in self.keys():
            raise ValueError(f"unknown variable {key!r}; library has {self.keys()}.")
        if len(self.studies()) > 1:
            raise ValueError(f"this library holds several studies "
                             f"{self.studies()} -- pin one with "
                             f"select(study='...') before resolving; studies are "
                             f"different seams/versions and are never interpolated.")
        loose = [k for k, v in self.axes().items() if k != key and len(v) > 1]
        if loose:
            raise ValueError(f"pin {loose!r} with select() before resolving {key!r} "
                             f"-- resolution is 1-D.")

        nodes = self.axes()[key]
        tol = self._tol(key)
        report: 'List[str]' = []

        if isinstance(req, Range) or (isinstance(req, tuple) and len(req) == 2):
            lo, hi = float(req[0]), float(req[1])
            if hi < lo:
                raise ValueError(f"{key}: range ({lo}, {hi}) is inverted.")
            inside = [v for v in nodes if lo - tol <= v <= hi + tol]
            if n is None:
                if not inside:
                    raise ValueError(f"{key}: no library value in [{lo:g}, {hi:g}]; "
                                     f"available {nodes}.")
                requested = list(inside)
            else:
                requested = list(np.linspace(lo, hi, int(n)))
        elif isinstance(req, (list, set, frozenset, np.ndarray)):
            requested = [float(x) for x in req]
        else:
            requested = [float(req)]
        if n is not None and len(requested) != int(n) and not (
                isinstance(req, Range) or (isinstance(req, tuple) and len(req) == 2)):
            raise ValueError(f"{key}: asked for n={n} values but gave {len(requested)}.")

        entries: 'List[DeltaEntry]' = []
        for want in requested:
            if want < nodes[0] - tol or want > nodes[-1] + tol:
                raise ValueError(f"{key}={want:g} is outside the library "
                                 f"[{nodes[0]:g}, {nodes[-1]:g}] -- the library is "
                                 f"never extrapolated; solve that coupon.")
            hit = min(nodes, key=lambda v: abs(v - want))
            if abs(hit - want) > tol:
                if off_grid == "error":
                    raise ValueError(f"{key}={want:g} is not in the library "
                                     f"({nodes}); pass off_grid='snap' to accept "
                                     f"the nearest, or solve that coupon.")
                report.append(f"{key}: requested {want:g} -> used {hit:g} "
                              f"(nearest node, off by {abs(hit-want):g})")
            entries.append(self._pick(key, hit, prefer_rev, report))
        return Resolution(entries, report, [float(x) for x in requested])

    def _pick(self, key: 'str', value: 'float', prefer_rev: 'str',
              report: 'List[str]') -> 'DeltaEntry':
        tol = self._tol(key)
        cands = [e for e in self.entries if abs(self._value(e, key) - value) <= tol]
        if not cands:
            raise ValueError(f"{key}={value:g} not in the library.")
        if len(cands) > 1:
            if prefer_rev != "highest":
                raise ValueError(f"{key}={value:g} has {len(cands)} revisions "
                                 f"{[c.rev for c in cands]}; select(rev=...) to choose.")
            cands = sorted(cands, key=lambda e: e.rev)
            report.append(f"{key}={value:g}: {len(cands)} revisions "
                          f"{[c.rev for c in cands]} -- using rev={cands[-1].rev:g}")
            cands = [cands[-1]]
        return cands[0]

    def nearest(self, **params) -> 'Tuple[DeltaEntry, List[str]]':
        """Closest single entry to a (possibly off-grid) point, with a report."""
        report: 'List[str]' = []
        for k, want in params.items():
            if k not in self.keys():
                raise ValueError(f"unknown variable {k!r}; library has {self.keys()}.")
            nodes = self.axes()[k]
            hit = min(nodes, key=lambda v: abs(v - float(want)))
            if abs(hit - float(want)) > self._tol(k):
                report.append(f"{k}: requested {float(want):g} -> used {hit:g} "
                              f"(nearest node, off by {abs(hit-float(want)):g})")
            params[k] = hit
        sub = self.select(**params)
        e = sorted(sub.entries, key=lambda x: x.rev)[-1]
        return e, report

    def paths(self) -> 'List[str]':
        return [e.path for e in self.entries]

    # -- payload sanity (reads the files, NOT for metadata) -------------------
    def validate(self, frequencies_ghz: 'Sequence[float]' = (), require: 'bool' = True
                 ) -> 'List[str]':
        """Open every entry and check it is USABLE: readable, and (if given) that
        it covers ``frequencies_ghz``.  Parameters still come only from the
        filename -- this reads the payload to catch a file that cannot serve, not
        to learn what it is.

        Production libraries require the same strict, dimensioned
        featured-minus-clean schema used by placement. A folder name is not
        enough to turn a whole-object field into a physical delta."""
        from feature_sum import _load_grim, load_seam_from_grim
        problems: 'List[str]' = []
        for e in self.entries:
            try:
                g = _load_grim(e.path)
            except Exception as exc:                      # noqa: BLE001
                problems.append(f"{e.label()}: unreadable ({exc}).")
                continue
            have = np.asarray(g.get("frequencies", []), float).ravel()
            if have.size == 0:
                problems.append(
                    f"{e.label()}: has no frequency samples.")
                continue
            requested_present = [
                float(f) for f in frequencies_ghz
                if np.any(np.abs(have - float(f)) <= 1e-6)
            ]
            probes = requested_present or [float(have[0])]
            for frequency in probes:
                try:
                    load_seam_from_grim(e.path, frequency)
                except Exception as exc:  # noqa: BLE001
                    problems.append(
                        f"{e.label()}: invalid physical delta ({exc}).")
                    break
            for f in frequencies_ghz:
                if not np.any(np.abs(have - float(f)) <= 1e-6):
                    problems.append(f"{e.label()}: no {float(f):g} GHz "
                                    f"(has {np.round(have, 3).tolist()}).")
        if problems and require:
            raise ValueError("delta library validation failed:\n  "
                             + "\n  ".join(problems))
        return problems


def families(root: 'str') -> 'Dict[str, DeltaLibrary]':
    """Every seam-type subdirectory of a library root -> its DeltaLibrary."""
    out: 'Dict[str, DeltaLibrary]' = {}
    for name in sorted(os.listdir(str(root))):
        d = os.path.join(str(root), name)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.grim")):
            out[name] = DeltaLibrary.from_dir(d)
    return out


# -----------------------------------------------------------------------------
# Spreading a tolerance around a perimeter
# -----------------------------------------------------------------------------

def arc_slices(perimeter, n_arcs: 'int') -> 'List[np.ndarray]':
    """Split a perimeter into ``n_arcs`` contiguous, non-overlapping arcs of
    roughly equal ARC LENGTH, cutting only at segment boundaries (so the
    closed-form per-segment phase integral is untouched and the split is exact:
    the same delta on every arc reproduces the single-placement result)."""
    per = np.asarray(perimeter, dtype=float)
    if per.ndim != 3 or per.shape[1:] != (2, 3):
        raise ValueError("perimeter must be (n_seg, 2, 3).")
    n = int(n_arcs)
    if n < 1 or n > len(per):
        raise ValueError(f"n_arcs={n} must be in 1..{len(per)} (one arc per segment "
                         f"at most -- subdivide the perimeter for finer arcs).")
    L = np.linalg.norm(per[:, 1] - per[:, 0], axis=1)
    s = np.concatenate([[0.0], np.cumsum(L)])
    cut = np.searchsorted(s, np.linspace(0.0, s[-1], n + 1))
    cut[0], cut[-1] = 0, len(per)
    cut = np.maximum.accumulate(np.clip(cut, 0, len(per)))
    for i in range(1, n + 1):                     # keep every arc non-empty
        cut[i] = max(cut[i], cut[i - 1] + 1)
        cut[i] = min(cut[i], len(per) - (n - i))
    arcs = [per[a:b] for a, b in zip(cut[:-1], cut[1:])]
    if sum(len(a) for a in arcs) != len(per):
        raise ValueError("arc split lost or duplicated segments.")
    return arcs


def smooth_cycle(values: 'Sequence[float]') -> 'List[int]':
    """Order indices so neighbours differ as little as possible AROUND A CLOSED
    LOOP: up the odd ranks, back down the even ranks.

    Every neighbour is then two ranks apart, so the largest adjacent jump is
    ``max(a[i+2] - a[i])`` -- the best any cyclic arrangement can do -- and the
    widest gap sits beside the 2nd/3rd widest, never beside the tightest.
    Physically it reads as one side of the door tight and the other loose (a
    ramp up, a ramp down), which is what a hinge/latch stack-up looks like, and
    it keeps the line expansion inside its own validity: the expansion assumes
    the cross-section is locally invariant along the line, so an abrupt
    coefficient step is a discontinuity it does not model."""
    r = list(np.argsort(np.asarray(values, dtype=float), kind="stable"))
    return [int(i) for i in (r[0::2] + r[1::2][::-1])]


def tolerance_placements(perimeter, entries: 'Sequence[DeltaEntry]',
                         order: 'str' = "smooth", order_by: 'Optional[str]' = None,
                         rng=None, **extra) -> 'List[Dict[str, Any]]':
    """One ``sum_features`` placement per arc, one library entry per arc.

    ``order``    ``"smooth"`` (default, see smooth_cycle), ``"as_given"``, or
                 ``"random"`` (pass ``rng`` for a reproducible draw -- useful
                 for the arrangement ENSEMBLE, since the true per-unit
                 arrangement is unknown and one arrangement is spuriously
                 precise).
    ``order_by`` which variable to order on; default is the one that varies.
    ``extra``    merged into every placement (e.g. ``scale=``, ``normal=``).
    """
    ents = list(entries)
    arcs = arc_slices(perimeter, len(ents))
    if order == "as_given":
        idx = list(range(len(ents)))
    elif order == "random":
        r = rng if rng is not None else np.random.default_rng()
        idx = [int(i) for i in r.permutation(len(ents))]
    elif order == "smooth":
        if order_by is None:
            varying = [k for k in ents[0].params
                       if len({e.params[k] for e in ents}) > 1]
            order_by = varying[0] if varying else next(iter(ents[0].params))
        idx = smooth_cycle([e.params[order_by] for e in ents])
    else:
        raise ValueError(f"unknown order {order!r} "
                         f"(smooth | as_given | random).")
    return [dict({"delta": ents[j].path, "perimeter": arcs[i]}, **extra)
            for i, j in enumerate(idx)]
