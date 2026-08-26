"""Compact, non-blocking controls for coherent feature assembly.

The physics and placement validation deliberately do not live here.
``FeatureWorkflowAdapter`` accepts the authoritative GHOST
``feature_workflow`` module (or a compatible injected service), while
``FeatureAssemblyFormModel`` keeps request construction testable without Qt or
GHOST on the import path.  The fixed CSV headers are mirrored here only so the
GUI can explain the contract and write blank templates before a backend is
connected; GHOST remains the authoritative parser.

Preview visibility is intentionally absent from the request model.  Hiding a
point or line group in the Assembly 3-D view must never remove that response
from the coherent physical assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable


UNIT_CHOICES = (
    ("inches (in)", "inches"),
    ("millimeters (mm)", "millimeters"),
    ("meters (m)", "meters"),
    ("feet (ft)", "feet"),
)

# Keep the always-visible trade-study status readable for large fastener sets.
# The complete disabled-ID list remains available through the explicit copy
# action next to the summary.
FEATURE_SELECTION_DISPLAY_ID_LIMIT = 8


# Display/template mirrors of GHOST's versioned point- and line-placement v1
# contracts.  These are intentionally strict: the GUI, local scripts, and HPC
# workflow all accept the same files without column inference or conversion.
POINT_PLACEMENT_COLUMNS = (
    "placement_id",
    "dataset_id",
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "roll_x",
    "roll_y",
    "roll_z",
)
LINE_PLACEMENT_COLUMNS = (
    "line_id",
    "dataset_id",
    "segment_index",
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
    "n1x",
    "n1y",
    "n1z",
    "n2x",
    "n2y",
    "n2z",
)
POINT_PLACEMENT_EXAMPLE = (
    "fastener_001,fastener,1.2,8.4,0.5,0,0,1,1,0,0"
)
LINE_PLACEMENT_EXAMPLE = (
    "gap_001,panel_gap,1,-2,6,0,-2,10,0,0,0,1,0,0,1"
)


def placement_csv_template_text(kind: str) -> str:
    """Return the exact blank v1 placement template for ``kind``."""

    normalized = str(kind).strip().lower()
    if normalized == "point":
        columns = POINT_PLACEMENT_COLUMNS
    elif normalized == "line":
        columns = LINE_PLACEMENT_COLUMNS
    else:
        raise ValueError("Placement template kind must be 'point' or 'line'.")
    return ",".join(columns) + "\n"


def write_placement_csv_template(kind: str, path: str | Path) -> Path:
    """Write a blank strict placement CSV and return its final path."""

    raw = _clean_path(path)
    if not raw:
        raise ValueError("Choose where to save the placement CSV template.")
    target = Path(raw)
    if not target.suffix:
        target = target.with_suffix(".csv")
    target.write_text(placement_csv_template_text(kind), encoding="utf-8")
    return target


@runtime_checkable
class _FeatureWorkflowModule(Protocol):
    FeatureAssemblyRequest: Callable[..., Any]

    def discover_feature_dataset_ids(self, **kwargs: Any) -> Any: ...

    def prepare_feature_assembly(self, request: Any) -> Any: ...

    def execute_feature_assembly(self, plan: Any) -> Any: ...


@dataclass(frozen=True)
class FeatureWorkflowAdapter:
    """Neutral adapter around the authoritative feature-workflow API.

    Pass ``FeatureWorkflowAdapter.from_module(feature_workflow)`` to the panel.
    Keeping these four callables explicit avoids importing GHOST from GRIM and
    makes dependency injection straightforward in tests and packaged builds.
    """

    request_factory: Callable[..., Any]
    discover: Callable[..., Any]
    prepare: Callable[[Any], Any]
    execute: Callable[[Any], Any]
    preview_inputs: Callable[..., Any] | None = None

    @classmethod
    def from_module(cls, module: _FeatureWorkflowModule) -> "FeatureWorkflowAdapter":
        missing = [
            name
            for name in (
                "FeatureAssemblyRequest",
                "discover_feature_dataset_ids",
                "prepare_feature_assembly",
                "execute_feature_assembly",
            )
            if not callable(getattr(module, name, None))
        ]
        if missing:
            raise TypeError(
                "Feature workflow service is missing callable(s): "
                + ", ".join(missing)
            )
        mirrored_contracts = (
            ("POINT_CSV_COLUMNS", POINT_PLACEMENT_COLUMNS),
            ("LINE_CSV_COLUMNS", LINE_PLACEMENT_COLUMNS),
        )
        for name, expected in mirrored_contracts:
            backend_columns = getattr(module, name, None)
            if backend_columns is not None and tuple(backend_columns) != expected:
                raise RuntimeError(
                    f"GHOST {name} no longer matches the placement format "
                    "shown by GRIM. Update the GUI template before assembly."
                )
        return cls(
            request_factory=module.FeatureAssemblyRequest,
            discover=module.discover_feature_dataset_ids,
            prepare=module.prepare_feature_assembly,
            execute=module.execute_feature_assembly,
            preview_inputs=getattr(module, "prepare_feature_input_preview", None),
        )

    @classmethod
    def from_service(cls, service: Any) -> "FeatureWorkflowAdapter":
        """Adapt GRIM's small integration service contract.

        The integration-facing names are ``make_request``,
        ``discover_dataset_ids(point_csv=None, line_csv=None)``, ``prepare``,
        and ``execute``.  ``from_module`` remains available when the GHOST
        module itself is injected directly.
        """

        missing = [
            name
            for name in ("make_request", "discover_dataset_ids", "prepare", "execute")
            if not callable(getattr(service, name, None))
        ]
        if missing:
            raise TypeError(
                "Feature assembly service is missing callable(s): "
                + ", ".join(missing)
            )

        def discover(**kwargs: Any) -> Any:
            return service.discover_dataset_ids(
                point_csv=kwargs.get("point_locations_csv"),
                line_csv=kwargs.get("line_locations_csv"),
            )

        return cls(
            request_factory=service.make_request,
            discover=discover,
            prepare=service.prepare,
            execute=service.execute,
            preview_inputs=getattr(service, "prepare_input_preview", None),
        )


def coerce_feature_workflow(service: Any) -> FeatureWorkflowAdapter:
    """Return an adapter for an adapter or feature-workflow-like module."""

    if isinstance(service, FeatureWorkflowAdapter):
        return service
    if service is None:
        raise RuntimeError(
            "Feature assembly is unavailable because no GHOST feature service "
            "has been connected."
        )
    if callable(getattr(service, "make_request", None)):
        return FeatureWorkflowAdapter.from_service(service)
    return FeatureWorkflowAdapter.from_module(service)


@dataclass
class FeatureAssemblyValues:
    """User-editable values, independent of any GUI toolkit."""

    base_grim: str = ""
    output_grim: str = ""
    coordinate_units: str = "inches"
    surface_mesh: str = ""
    surface_units: str = "inches"
    flip_surface_normals: bool = False
    shadow: bool = False
    shadow_bias_m: float | None = None
    point_locations_csv: str = ""
    line_locations_csv: str = ""
    skin_tol_m: float = 1.0e-3
    skin_phase_tol_deg: float = 15.0
    normal_tol_deg: float = 15.0
    base_dir: str | None = None
    point_datasets: dict[str, str] = field(default_factory=dict)
    line_datasets: dict[str, str] = field(default_factory=dict)
    # Spatial feature-definition state. These stable CSV IDs are independent
    # from both preview visibility and whole-response dataset arithmetic.
    excluded_point_placement_ids: set[str] = field(default_factory=set)
    excluded_line_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class FeatureBuildDispatch:
    """Result returned by the combined prepare/execute operation."""

    plan: Any
    output_path: str
    reused_validated_plan: bool = False


@dataclass(frozen=True)
class _FileFingerprint:
    """Stable-enough identity for detecting in-place input edits.

    Placement CSVs are small, so hashing them avoids accepting a rewrite whose
    size and timestamp happen to be unchanged.  Larger GRIM/STL inputs use the
    same helper only when a validated plan is being considered for reuse.
    """

    resolved_path: str
    exists: bool
    size: int | None = None
    mtime_ns: int | None = None
    ctime_ns: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class _DatasetDiscovery:
    """Parser result paired with the exact CSV bytes that were parsed."""

    requirements: Any
    point_fingerprint: _FileFingerprint | None = None
    line_fingerprint: _FileFingerprint | None = None


@dataclass(frozen=True)
class _VerifiedInputPreview:
    """Input preview paired with the exact placement CSVs that produced it."""

    preview: Any
    discovery: _DatasetDiscovery | None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.preview, name)


@dataclass(frozen=True)
class _PreparedPlanCache:
    """A physically validated plan that is safe to reuse while inputs match."""

    plan: Any
    semantic_signature: tuple[Any, ...]
    source_fingerprints: tuple[tuple[str, _FileFingerprint], ...]
    service_key: tuple[Any, ...]


@dataclass(frozen=True)
class LoadedDatasetEntry:
    """Small, file-oriented dataset reference accepted by the Assembly UI.

    Feature assembly is intentionally a file-backed workflow: the GHOST
    backend consumes response paths rather than live ``RcsGrid`` objects.
    ``dirty`` therefore prevents an inherited source path from being offered
    as though it represented the current in-memory data.
    """

    dataset_id: str
    name: str
    path: str = ""
    dirty: bool = False
    _usable_path: str = field(init=False, repr=False)
    _unavailable_reason: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        name = str(self.name).strip()
        path = str(self.path or "").strip()
        dirty = bool(self.dirty)
        if not dataset_id:
            raise ValueError(
                "An Assembly loaded-dataset entry requires a stable dataset_id."
            )
        if not name:
            raise ValueError(
                "An Assembly loaded-dataset entry requires a display name."
            )
        if dirty or not path:
            usable_path = ""
            reason = "save unsaved derived dataset first"
        else:
            candidate = Path(path)
            if candidate.suffix.casefold() != ".grim":
                usable_path = ""
                reason = "not a .grim file"
            elif not candidate.is_file():
                usable_path = ""
                reason = "saved file is missing"
            else:
                usable_path = str(candidate)
                reason = ""
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "dirty", dirty)
        object.__setattr__(self, "_usable_path", usable_path)
        object.__setattr__(self, "_unavailable_reason", reason)

    @property
    def usable_path(self) -> str:
        """Return the backend-safe path, or an empty string when unavailable."""

        return self._usable_path

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason


def _entry_value(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _coerce_loaded_dataset_entry(value: Any) -> LoadedDatasetEntry:
    """Accept shell-facing mappings, tuples, or objects with familiar names."""

    if isinstance(value, LoadedDatasetEntry):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            dataset_id, name, path = value
            return LoadedDatasetEntry(
                str(dataset_id or ""), str(name or ""), _clean_path(path)
            )
        if len(value) == 4:
            dataset_id, name, path, dirty = value
            return LoadedDatasetEntry(
                str(dataset_id or ""),
                str(name or ""),
                _clean_path(path),
                bool(dirty),
            )
        raise TypeError(
            "Assembly loaded-dataset tuples must contain "
            "(dataset_id, name, path[, dirty])."
        )

    dataset_id = _entry_value(value, "dataset_id", "stable_id", "id", "key")
    name = _entry_value(value, "name", "display_name", "label")
    path = _entry_value(
        value,
        "path",
        "source",
        "source_path",
        "file_path",
        "output_path",
        default="",
    )
    dirty = _entry_value(
        value,
        "dirty",
        "is_dirty",
        "unsaved",
        "is_unsaved",
        default=False,
    )
    if dataset_id is None and name is None and path is None:
        raise TypeError(
            "Assembly loaded-dataset entries must be mappings, "
            "(dataset_id, name, path[, dirty]) tuples, or objects with "
            "matching attributes."
        )
    return LoadedDatasetEntry(
        str(dataset_id or ""),
        str(name or ""),
        _clean_path(path),
        bool(dirty),
    )


def _coerce_loaded_dataset_catalog(
    entries: Iterable[Any],
) -> tuple[LoadedDatasetEntry, ...]:
    catalog = tuple(_coerce_loaded_dataset_entry(entry) for entry in entries)
    identifiers = [entry.dataset_id for entry in catalog]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Assembly loaded dataset_id values must be unique.")
    return catalog


def _clean_path(value: Any) -> str:
    return str(value or "").strip()


def _resolved_user_path(value: Any, *, base_dir: Any = None) -> Path:
    """Resolve a user path with the same base-directory rule as GHOST."""

    path = Path(_clean_path(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = Path.cwd() if not _clean_path(base_dir) else Path(base_dir).expanduser()
    return (root.resolve() / path).resolve()


def _normalized_grim_output_path(value: Any, *, base_dir: Any = None) -> Path:
    """Return the destination the writer will actually use."""

    resolved = _resolved_user_path(value, base_dir=base_dir)
    if str(resolved).casefold().endswith(".grim"):
        return resolved
    return Path(str(resolved) + ".grim")


def _path_key(path: Path) -> str:
    """Comparable canonical path key (case-insensitive on Windows)."""

    return os.path.normcase(str(path.resolve()))


def _paths_alias(first: Path, second: Path) -> bool:
    """Return whether two paths name the same target, including hard links."""

    if _path_key(first) == _path_key(second):
        return True
    try:
        return first.samefile(second)
    except (FileNotFoundError, OSError):
        return False


def _fingerprint_file(
    value: Any,
    *,
    base_dir: Any = None,
    include_hash: bool = True,
) -> _FileFingerprint:
    """Fingerprint one input and reject a file that changes while hashing."""

    resolved = _resolved_user_path(value, base_dir=base_dir)
    key = _path_key(resolved)
    if not resolved.is_file():
        return _FileFingerprint(resolved_path=key, exists=False)

    if not include_hash:
        stat = resolved.stat()
        return _FileFingerprint(
            resolved_path=key,
            exists=True,
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ctime_ns=int(stat.st_ctime_ns),
        )

    for _attempt in range(2):
        before = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = resolved.stat()
        if (
            before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
            and before.st_ctime_ns == after.st_ctime_ns
        ):
            return _FileFingerprint(
                resolved_path=key,
                exists=True,
                size=int(after.st_size),
                mtime_ns=int(after.st_mtime_ns),
                ctime_ns=int(after.st_ctime_ns),
                sha256=digest.hexdigest(),
            )
    raise RuntimeError(f"Input changed while it was being read: {resolved}")


def _callable_key(value: Callable[..., Any]) -> tuple[int, int]:
    """Identify a bound function without depending on transient method objects."""

    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", value)
    return (id(owner), id(function))


def _require_finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return number


def _requirements_ids(requirements: Any, attribute: str) -> tuple[str, ...]:
    if isinstance(requirements, Mapping):
        values = requirements.get(attribute, ())
    else:
        values = getattr(requirements, attribute, ())
    ordered = tuple(dict.fromkeys(str(value).strip() for value in values))
    if any(not value for value in ordered):
        raise ValueError("Placement CSV returned an empty dataset_id.")
    return ordered


def _requirements_count(requirements: Any, attribute: str) -> int:
    if isinstance(requirements, Mapping):
        value = requirements.get(attribute, 0)
    else:
        value = getattr(requirements, attribute, 0)
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, count)


def _requirements_point_instances(
    requirements: Any,
) -> tuple[tuple[str, str], ...]:
    raw = (
        requirements.get("point_instances", ())
        if isinstance(requirements, Mapping)
        else getattr(requirements, "point_instances", ())
    )
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in raw or ():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                "Placement parser returned an invalid point instance descriptor."
            )
        placement_id, dataset_id = (str(part).strip() for part in value)
        if not placement_id or not dataset_id or placement_id in seen:
            raise ValueError(
                "Placement parser returned an empty or duplicate point placement_id."
            )
        seen.add(placement_id)
        result.append((placement_id, dataset_id))
    return tuple(result)


def _requirements_line_instances(
    requirements: Any,
) -> tuple[tuple[str, str, int], ...]:
    raw = (
        requirements.get("line_instances", ())
        if isinstance(requirements, Mapping)
        else getattr(requirements, "line_instances", ())
    )
    result: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for value in raw or ():
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError(
                "Placement parser returned an invalid line instance descriptor."
            )
        line_id = str(value[0]).strip()
        dataset_id = str(value[1]).strip()
        try:
            segment_count = int(value[2])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Placement parser returned a non-integer line segment count."
            ) from exc
        if (
            not line_id
            or not dataset_id
            or line_id in seen
            or segment_count <= 0
        ):
            raise ValueError(
                "Placement parser returned an empty, duplicate, or invalid line_id."
            )
        seen.add(line_id)
        result.append((line_id, dataset_id, segment_count))
    return tuple(result)


class FeatureAssemblyFormModel:
    """Headless state, validation, discovery, and service dispatch."""

    def __init__(self, values: FeatureAssemblyValues | None = None) -> None:
        self.values = values if values is not None else FeatureAssemblyValues()
        self._point_dataset_ids: tuple[str, ...] = ()
        self._line_dataset_ids: tuple[str, ...] = ()
        self._point_requirements_csv = ""
        self._line_requirements_csv = ""
        self._point_requirements_fingerprint: _FileFingerprint | None = None
        self._line_requirements_fingerprint: _FileFingerprint | None = None
        self._point_placement_count = 0
        self._line_path_count = 0
        self._line_segment_count = 0
        self._point_instances: tuple[tuple[str, str], ...] = ()
        self._line_instances: tuple[tuple[str, str, int], ...] = ()
        self._prepared_plan_cache: _PreparedPlanCache | None = None

    @property
    def point_dataset_ids(self) -> tuple[str, ...]:
        return self._point_dataset_ids

    @property
    def line_dataset_ids(self) -> tuple[str, ...]:
        return self._line_dataset_ids

    @property
    def point_placement_count(self) -> int:
        return self._point_placement_count

    @property
    def line_path_count(self) -> int:
        return self._line_path_count

    @property
    def line_segment_count(self) -> int:
        return self._line_segment_count

    @property
    def point_instances(self) -> tuple[tuple[str, str], ...]:
        return self._point_instances

    @property
    def line_instances(self) -> tuple[tuple[str, str, int], ...]:
        return self._line_instances

    @property
    def enabled_point_placement_ids(self) -> tuple[str, ...] | None:
        if not self._point_instances:
            return None
        excluded = self.values.excluded_point_placement_ids
        return tuple(
            placement_id
            for placement_id, _dataset_id in self._point_instances
            if placement_id not in excluded
        )

    @property
    def enabled_line_ids(self) -> tuple[str, ...] | None:
        if not self._line_instances:
            return None
        excluded = self.values.excluded_line_ids
        return tuple(
            line_id
            for line_id, _dataset_id, _segments in self._line_instances
            if line_id not in excluded
        )

    def active_point_dataset_ids(self) -> tuple[str, ...]:
        enabled = self.enabled_point_placement_ids
        if enabled is None:
            return self._point_dataset_ids
        selected = set(enabled)
        return tuple(dict.fromkeys(
            dataset_id
            for placement_id, dataset_id in self._point_instances
            if placement_id in selected
        ))

    def active_line_dataset_ids(self) -> tuple[str, ...]:
        enabled = self.enabled_line_ids
        if enabled is None:
            return self._line_dataset_ids
        selected = set(enabled)
        return tuple(dict.fromkeys(
            dataset_id
            for line_id, dataset_id, _segments in self._line_instances
            if line_id in selected
        ))

    def set_feature_instance_enabled(
        self, kind: str, instance_id: str, enabled: bool
    ) -> None:
        normalized = str(kind).strip().lower()
        key = str(instance_id).strip()
        if normalized == "point":
            known = {value[0] for value in self._point_instances}
            excluded = self.values.excluded_point_placement_ids
        elif normalized == "line":
            known = {value[0] for value in self._line_instances}
            excluded = self.values.excluded_line_ids
        else:
            raise ValueError("Feature instance kind must be point or line.")
        if key not in known:
            raise KeyError(f"Unknown {normalized} feature instance {key!r}.")
        if enabled:
            excluded.discard(key)
        else:
            excluded.add(key)
        self.invalidate_prepared_plan()

    def set_excluded_feature_instances(
        self,
        *,
        point_ids: Iterable[str],
        line_ids: Iterable[str],
    ) -> None:
        points = {str(value).strip() for value in point_ids}
        lines = {str(value).strip() for value in line_ids}
        known_points = {value[0] for value in self._point_instances}
        known_lines = {value[0] for value in self._line_instances}
        unknown_points = sorted(points - known_points)
        unknown_lines = sorted(lines - known_lines)
        if "" in points or "" in lines or unknown_points or unknown_lines:
            raise ValueError(
                "Feature selection contains blank or stale IDs: "
                f"point={unknown_points}, line={unknown_lines}. Refresh the CSVs."
            )
        self.values.excluded_point_placement_ids = points
        self.values.excluded_line_ids = lines
        self.invalidate_prepared_plan()

    def feature_selection_summary(
        self, *, max_disabled_ids_per_kind: int | None = None
    ) -> str:
        """Describe exact membership, optionally shortening displayed ID lists.

        The default is deliberately lossless for logs, the clipboard action,
        and headless callers. The GUI passes a small limit only to its
        always-visible label so a large fastener trade study does not become a
        wall of text.
        """

        if max_disabled_ids_per_kind is not None:
            max_disabled_ids_per_kind = int(max_disabled_ids_per_kind)
            if max_disabled_ids_per_kind < 0:
                raise ValueError("Disabled-ID summary limit must be non-negative.")

        def disabled_text(kind: str, identifiers: Iterable[str]) -> str:
            ordered = sorted(str(value) for value in identifiers)
            if (
                max_disabled_ids_per_kind is None
                or len(ordered) <= max_disabled_ids_per_kind
            ):
                return f"{kind}=[" + ", ".join(ordered) + "]"
            visible = ordered[:max_disabled_ids_per_kind]
            omitted = len(ordered) - len(visible)
            prefix = ", ".join(visible)
            if prefix:
                prefix += ", "
            return (
                f"{kind}=[{prefix}… +{omitted} more]"
                " (use Copy full selection)"
            )

        point_total = len(self._point_instances)
        line_total = len(self._line_instances)
        point_enabled = self.enabled_point_placement_ids
        line_enabled = self.enabled_line_ids
        summary = (
            f"Enabled spatial features: {len(point_enabled or ())}/{point_total} "
            f"point placement(s), {len(line_enabled or ())}/{line_total} line "
            "path(s). Disabled features remain in the parsed configuration but "
            "are omitted from preview, validation, response loading, and build."
        )
        disabled_parts = []
        if self.values.excluded_point_placement_ids:
            disabled_parts.append(disabled_text(
                "point", self.values.excluded_point_placement_ids
            ))
        if self.values.excluded_line_ids:
            disabled_parts.append(disabled_text(
                "line", self.values.excluded_line_ids
            ))
        if disabled_parts:
            return summary + " Disabled IDs: " + "; ".join(disabled_parts) + "."
        if point_total or line_total:
            return summary + " All parsed spatial features are enabled."
        return summary + " Refresh a placement CSV to populate the hierarchy."

    def clear_feature_selection(self, kind: str | None = None) -> None:
        normalized = None if kind is None else str(kind).strip().lower()
        if normalized not in (None, "point", "line"):
            raise ValueError("Feature selection kind must be point, line, or None.")
        if normalized in (None, "point"):
            self.values.excluded_point_placement_ids.clear()
        if normalized in (None, "line"):
            self.values.excluded_line_ids.clear()
        self.invalidate_prepared_plan()

    def feature_selection_source_changed(self, kind: str, path: Any) -> bool:
        """Return whether ``path`` differs from the CSV that defined the IDs.

        Exclusions are a live trade-study choice tied to one parsed CSV. They
        survive an in-place rescan of that file, but must not silently migrate
        to another file merely because it reuses the same placement IDs.
        """

        normalized = str(kind).strip().lower()
        if normalized not in {"point", "line"}:
            raise ValueError("Feature selection kind must be point or line.")
        recorded = (
            self._point_requirements_fingerprint
            if normalized == "point"
            else self._line_requirements_fingerprint
        )
        if recorded is None:
            return False
        cleaned = _clean_path(path)
        if not cleaned:
            return True
        candidate = _path_key(
            _resolved_user_path(cleaned, base_dir=self.values.base_dir)
        )
        return candidate != recorded.resolved_path

    def _selected_csv_fingerprint(self, kind: str) -> _FileFingerprint | None:
        path = (
            self.values.point_locations_csv
            if kind == "point"
            else self.values.line_locations_csv
        )
        if not _clean_path(path):
            return None
        return _fingerprint_file(path, base_dir=self.values.base_dir)

    def requirements_are_current(self, kind: str) -> bool:
        """Return whether discovered IDs still describe the selected CSV bytes."""

        normalized = str(kind).strip().lower()
        if normalized not in {"point", "line"}:
            raise ValueError("Dataset requirement kind must be point or line.")
        ids = self._point_dataset_ids if normalized == "point" else self._line_dataset_ids
        recorded = (
            self._point_requirements_fingerprint
            if normalized == "point"
            else self._line_requirements_fingerprint
        )
        if not ids or recorded is None:
            return False
        try:
            return self._selected_csv_fingerprint(normalized) == recorded
        except OSError:
            return False

    def requirements_look_current(self, kind: str) -> bool:
        """Cheap UI hint using path/stat identity; validation still hashes bytes."""

        normalized = str(kind).strip().lower()
        if normalized not in {"point", "line"}:
            raise ValueError("Dataset requirement kind must be point or line.")
        ids = self._point_dataset_ids if normalized == "point" else self._line_dataset_ids
        recorded = (
            self._point_requirements_fingerprint
            if normalized == "point"
            else self._line_requirements_fingerprint
        )
        if not ids or recorded is None:
            return False
        path = (
            self.values.point_locations_csv
            if normalized == "point"
            else self.values.line_locations_csv
        )
        try:
            current = _fingerprint_file(
                path,
                base_dir=self.values.base_dir,
                include_hash=False,
            )
        except OSError:
            return False
        return (
            current.resolved_path == recorded.resolved_path
            and current.exists == recorded.exists
            and current.size == recorded.size
            and current.mtime_ns == recorded.mtime_ns
            and current.ctime_ns == recorded.ctime_ns
        )

    def invalidate_prepared_plan(self) -> None:
        """Release cached validated geometry after any semantic input edit."""

        self._prepared_plan_cache = None

    def update_dataset_requirements(self, requirements: Any) -> None:
        """Apply discovered IDs while preserving paths for surviving IDs."""

        discovery = requirements if isinstance(requirements, _DatasetDiscovery) else None
        payload = discovery.requirements if discovery is not None else requirements
        point_ids = _requirements_ids(payload, "point_dataset_ids")
        line_ids = _requirements_ids(payload, "line_dataset_ids")
        self._point_dataset_ids = point_ids
        self._line_dataset_ids = line_ids
        self._point_placement_count = _requirements_count(
            payload, "point_placement_count"
        )
        self._line_path_count = _requirements_count(payload, "line_path_count")
        self._line_segment_count = _requirements_count(
            payload, "line_segment_count"
        )
        self._point_instances = _requirements_point_instances(payload)
        self._line_instances = _requirements_line_instances(payload)
        unknown_point_datasets = sorted(
            {dataset_id for _placement_id, dataset_id in self._point_instances}
            - set(point_ids)
        )
        unknown_line_datasets = sorted(
            {
                dataset_id
                for _line_id, dataset_id, _segments in self._line_instances
            }
            - set(line_ids)
        )
        if unknown_point_datasets or unknown_line_datasets:
            raise ValueError(
                "Placement parser returned instance descriptors for unknown "
                f"dataset IDs: point={unknown_point_datasets}, "
                f"line={unknown_line_datasets}."
            )
        # Stable exclusions survive a re-scan only while the same explicit
        # CSV IDs survive. Newly parsed instances default enabled.
        self.values.excluded_point_placement_ids.intersection_update(
            placement_id for placement_id, _dataset_id in self._point_instances
        )
        self.values.excluded_line_ids.intersection_update(
            line_id for line_id, _dataset_id, _segments in self._line_instances
        )
        self._point_requirements_csv = _clean_path(
            self.values.point_locations_csv
        )
        self._line_requirements_csv = _clean_path(
            self.values.line_locations_csv
        )
        self._point_requirements_fingerprint = (
            discovery.point_fingerprint
            if discovery is not None
            else self._selected_csv_fingerprint("point")
        )
        self._line_requirements_fingerprint = (
            discovery.line_fingerprint
            if discovery is not None
            else self._selected_csv_fingerprint("line")
        )
        self.values.point_datasets = {
            dataset_id: _clean_path(self.values.point_datasets.get(dataset_id))
            for dataset_id in point_ids
        }
        self.values.line_datasets = {
            dataset_id: _clean_path(self.values.line_datasets.get(dataset_id))
            for dataset_id in line_ids
        }
        self.invalidate_prepared_plan()

    def invalidate_dataset_requirements(self, kind: str | None = None) -> None:
        """Discard IDs that no longer describe the selected/on-disk CSV."""

        normalized = None if kind is None else str(kind).strip().lower()
        if normalized not in (None, "point", "line"):
            raise ValueError("Dataset requirement kind must be point, line, or None.")
        if normalized in (None, "point"):
            self._point_dataset_ids = ()
            self._point_placement_count = 0
            self._point_instances = ()
            self._point_requirements_csv = ""
            self._point_requirements_fingerprint = None
            self.values.point_datasets = {}
        if normalized in (None, "line"):
            self._line_dataset_ids = ()
            self._line_path_count = 0
            self._line_segment_count = 0
            self._line_instances = ()
            self._line_requirements_csv = ""
            self._line_requirements_fingerprint = None
            self.values.line_datasets = {}
        self.invalidate_prepared_plan()

    def query_dataset_ids(self, service: Any) -> Any:
        """Validate CSVs without applying IDs; stale prior IDs are invalidated."""

        adapter = coerce_feature_workflow(service)
        point_csv = _clean_path(self.values.point_locations_csv)
        line_csv = _clean_path(self.values.line_locations_csv)
        if not point_csv and not line_csv:
            raise ValueError("Select a point or line placement CSV first.")
        point_before = self._selected_csv_fingerprint("point")
        line_before = self._selected_csv_fingerprint("line")
        requirements = adapter.discover(
            point_locations_csv=point_csv or None,
            line_locations_csv=line_csv or None,
            base_dir=self.values.base_dir,
        )
        point_after = self._selected_csv_fingerprint("point")
        line_after = self._selected_csv_fingerprint("line")
        if point_before != point_after or line_before != line_after:
            if point_before != point_after:
                self.invalidate_dataset_requirements("point")
            if line_before != line_after:
                self.invalidate_dataset_requirements("line")
            raise RuntimeError(
                "A placement CSV changed while it was being read. Save it, then refresh."
            )
        return _DatasetDiscovery(
            requirements=requirements,
            point_fingerprint=point_after,
            line_fingerprint=line_after,
        )

    def discover_dataset_ids(self, service: Any) -> Any:
        """Ask the authoritative parser to validate CSVs and apply their IDs."""

        discovery = self.query_dataset_ids(service)
        self.update_dataset_requirements(discovery)
        return discovery.requirements

    def set_point_dataset(self, dataset_id: str, path: str) -> None:
        self._set_dataset("point", dataset_id, path)

    def set_line_dataset(self, dataset_id: str, path: str) -> None:
        self._set_dataset("line", dataset_id, path)

    def _set_dataset(self, kind: str, dataset_id: str, path: str) -> None:
        key = str(dataset_id).strip()
        ids = self._point_dataset_ids if kind == "point" else self._line_dataset_ids
        if key not in ids:
            raise KeyError(f"Unknown {kind} dataset_id {key!r}.")
        mapping = (
            self.values.point_datasets
            if kind == "point"
            else self.values.line_datasets
        )
        mapping[key] = _clean_path(path)
        self.invalidate_prepared_plan()

    def missing_dataset_mappings(self) -> tuple[str, ...]:
        missing = [
            f"point:{dataset_id}"
            for dataset_id in self.active_point_dataset_ids()
            if not _clean_path(self.values.point_datasets.get(dataset_id))
        ]
        missing.extend(
            f"line:{dataset_id}"
            for dataset_id in self.active_line_dataset_ids()
            if not _clean_path(self.values.line_datasets.get(dataset_id))
        )
        return tuple(missing)

    def _validate_output_target(self) -> None:
        values = self.values
        output = _normalized_grim_output_path(
            values.output_grim, base_dir=values.base_dir
        )
        protected: list[tuple[str, str]] = [("clean-body response", values.base_grim)]
        protected.extend(
            (f"point response {dataset_id!r}", path)
            for dataset_id, path in values.point_datasets.items()
        )
        protected.extend(
            (f"line response {dataset_id!r}", path)
            for dataset_id, path in values.line_datasets.items()
        )
        for label, path in protected:
            if _clean_path(path) and _paths_alias(
                output,
                _resolved_user_path(path, base_dir=values.base_dir),
            ):
                raise ValueError(
                    f"Output must not overwrite the {label}. Choose a new file name."
                )

    def validate(self) -> None:
        values = self.values
        if not _clean_path(values.base_grim):
            raise ValueError("Select the clean-body/base GRIM file.")
        if not _clean_path(values.output_grim):
            raise ValueError("Choose an output GRIM file.")

        point_csv = _clean_path(values.point_locations_csv)
        line_csv = _clean_path(values.line_locations_csv)
        if not point_csv and not line_csv:
            raise ValueError("Select a point or line placement CSV.")
        if point_csv and not self.requirements_are_current("point"):
            raise ValueError(
                "The point CSV changed after its last successful scan. "
                "Re-scan it before continuing."
            )
        if line_csv and not self.requirements_are_current("line"):
            raise ValueError(
                "The line CSV changed after its last successful scan. "
                "Re-scan it before continuing."
            )
        if point_csv and not self._point_dataset_ids:
            raise ValueError(
                "Point dataset IDs have not been discovered. Re-scan the "
                "point CSV before continuing."
            )
        if line_csv and not self._line_dataset_ids:
            raise ValueError(
                "Line dataset IDs have not been discovered. Re-scan the line "
                "CSV before continuing."
            )
        if (
            (self._point_instances or self._line_instances)
            and not (self.enabled_point_placement_ids or self.enabled_line_ids)
        ):
            raise ValueError(
                "No enabled spatial features remain. Enable at least one point "
                "placement or line path before validating or building."
            )
        missing = self.missing_dataset_mappings()
        if missing:
            raise ValueError(
                "Choose an OPN-FRD GRIM response for: " + ", ".join(missing)
            )
        if values.shadow and not _clean_path(values.surface_mesh):
            raise ValueError(
                "Geometric shadowing requires an STL or facet surface mesh."
            )
        if values.coordinate_units not in {value for _, value in UNIT_CHOICES}:
            raise ValueError(f"Unsupported coordinate units: {values.coordinate_units!r}.")
        if values.surface_units not in {value for _, value in UNIT_CHOICES}:
            raise ValueError(f"Unsupported surface units: {values.surface_units!r}.")
        _require_finite_nonnegative(values.skin_tol_m, "Skin distance tolerance")
        _require_finite_nonnegative(
            values.skin_phase_tol_deg, "Skin phase tolerance"
        )
        normal = _require_finite_nonnegative(
            values.normal_tol_deg, "Normal tolerance"
        )
        if normal >= 90.0:
            raise ValueError(
                "Normal tolerance must be less than 90 degrees so inward-facing "
                "feature frames cannot pass validation."
            )
        if values.shadow_bias_m is not None:
            _require_finite_nonnegative(values.shadow_bias_m, "Shadow bias")
        self._validate_output_target()

    def build_request(self, service: Any) -> Any:
        """Create the backend request only after local completeness checks."""

        adapter = coerce_feature_workflow(service)
        self.validate()
        values = self.values
        return adapter.request_factory(
            base_grim=_clean_path(values.base_grim),
            output_grim=_clean_path(values.output_grim),
            coordinate_units=values.coordinate_units,
            surface_mesh=_clean_path(values.surface_mesh) or None,
            surface_units=values.surface_units,
            flip_surface_normals=bool(values.flip_surface_normals),
            shadow=bool(values.shadow),
            shadow_bias_m=(
                None
                if values.shadow_bias_m is None
                else float(values.shadow_bias_m)
            ),
            point_locations_csv=_clean_path(values.point_locations_csv) or None,
            point_datasets={
                key: _clean_path(values.point_datasets[key])
                for key in self.active_point_dataset_ids()
            },
            enabled_point_placement_ids=self.enabled_point_placement_ids,
            line_locations_csv=_clean_path(values.line_locations_csv) or None,
            line_datasets={
                key: _clean_path(values.line_datasets[key])
                for key in self.active_line_dataset_ids()
            },
            enabled_line_ids=self.enabled_line_ids,
            skin_tol_m=float(values.skin_tol_m),
            skin_phase_tol_deg=float(values.skin_phase_tol_deg),
            normal_tol_deg=float(values.normal_tol_deg),
            base_dir=values.base_dir,
        )

    def _semantic_signature(self) -> tuple[Any, ...]:
        values = self.values

        def path_signature(path: Any, *, output: bool = False) -> str:
            if not _clean_path(path):
                return ""
            resolved = (
                _normalized_grim_output_path(path, base_dir=values.base_dir)
                if output
                else _resolved_user_path(path, base_dir=values.base_dir)
            )
            return _path_key(resolved)

        return (
            path_signature(values.base_grim),
            path_signature(values.output_grim, output=True),
            values.coordinate_units,
            path_signature(values.surface_mesh),
            values.surface_units,
            bool(values.flip_surface_normals),
            bool(values.shadow),
            values.shadow_bias_m,
            path_signature(values.point_locations_csv),
            self.enabled_point_placement_ids,
            tuple(
                (dataset_id, path_signature(values.point_datasets.get(dataset_id)))
                for dataset_id in self.active_point_dataset_ids()
            ),
            path_signature(values.line_locations_csv),
            self.enabled_line_ids,
            tuple(
                (dataset_id, path_signature(values.line_datasets.get(dataset_id)))
                for dataset_id in self.active_line_dataset_ids()
            ),
            float(values.skin_tol_m),
            float(values.skin_phase_tol_deg),
            float(values.normal_tol_deg),
            (
                _path_key(Path(values.base_dir).expanduser().resolve())
                if _clean_path(values.base_dir)
                else ""
            ),
        )

    def _source_fingerprints(
        self,
    ) -> tuple[tuple[str, _FileFingerprint], ...]:
        values = self.values
        sources: list[tuple[str, str, bool]] = [
            ("base", _clean_path(values.base_grim), True),
            ("surface", _clean_path(values.surface_mesh), True),
            ("point CSV", _clean_path(values.point_locations_csv), True),
            ("line CSV", _clean_path(values.line_locations_csv), True),
        ]
        sources.extend(
            (
                f"point:{dataset_id}",
                _clean_path(values.point_datasets.get(dataset_id)),
                True,
            )
            for dataset_id in self.active_point_dataset_ids()
        )
        sources.extend(
            (
                f"line:{dataset_id}",
                _clean_path(values.line_datasets.get(dataset_id)),
                True,
            )
            for dataset_id in self.active_line_dataset_ids()
        )
        return tuple(
            (
                label,
                _fingerprint_file(
                    path,
                    base_dir=values.base_dir,
                    include_hash=include_hash,
                ),
            )
            for label, path, include_hash in sources
            if path
        )

    @staticmethod
    def _service_key(adapter: FeatureWorkflowAdapter) -> tuple[Any, ...]:
        return (
            _callable_key(adapter.request_factory),
            _callable_key(adapter.prepare),
            _callable_key(adapter.execute),
        )

    def _prepare_and_cache(
        self,
        adapter: FeatureWorkflowAdapter,
        request: Any,
        *,
        before: tuple[tuple[str, _FileFingerprint], ...] | None = None,
    ) -> Any:
        # ``assemble`` may already have captured this exact full-content
        # snapshot while deciding whether a validated plan can be reused.  Pass
        # it through so a cache miss does not immediately reread every large
        # GRIM/STL response before the authoritative prepare operation.
        semantic_before = self._semantic_signature()
        if before is None:
            before = self._source_fingerprints()
        plan = adapter.prepare(request)
        after = self._source_fingerprints()
        semantic_after = self._semantic_signature()
        if semantic_before != semantic_after:
            self.invalidate_prepared_plan()
            raise RuntimeError(
                "The spatial feature configuration changed during validation. "
                "Review the enabled features and validate again."
            )
        if before != after:
            before_by_label = dict(before)
            after_by_label = dict(after)
            if before_by_label.get("point CSV") != after_by_label.get("point CSV"):
                self.invalidate_dataset_requirements("point")
            if before_by_label.get("line CSV") != after_by_label.get("line CSV"):
                self.invalidate_dataset_requirements("line")
            raise RuntimeError(
                "An Assembly input changed during validation. Save the input, "
                "refresh the placement CSVs, and validate again."
            )
        self._prepared_plan_cache = _PreparedPlanCache(
            plan=plan,
            semantic_signature=semantic_before,
            source_fingerprints=after,
            service_key=self._service_key(adapter),
        )
        return plan

    def prepare_preview(self, service: Any) -> Any:
        adapter = coerce_feature_workflow(service)
        request = self.build_request(adapter)
        return self._prepare_and_cache(adapter, request)

    def prepare_input_preview(self, service: Any) -> Any:
        """Preview selected geometry/locations before response mapping.

        This is a deliberately non-physical staging preview.  The optional
        backend callable owns CSV parsing and geometry loading; the GUI only
        passes paths and units through unchanged.
        """

        adapter = coerce_feature_workflow(service)
        if not callable(adapter.preview_inputs):
            raise RuntimeError(
                "This GHOST backend does not support staged input preview. "
                "Use Validate placements after mapping responses."
            )
        values = self.values
        base_grim = _clean_path(values.base_grim)
        surface_mesh = _clean_path(values.surface_mesh)
        point_csv = _clean_path(values.point_locations_csv)
        line_csv = _clean_path(values.line_locations_csv)
        if not any((base_grim, surface_mesh, point_csv, line_csv)):
            raise ValueError(
                "Choose a clean-body GRIM, body mesh, or placement CSV to preview."
            )
        if values.coordinate_units not in {value for _, value in UNIT_CHOICES}:
            raise ValueError(f"Unsupported coordinate units: {values.coordinate_units!r}.")
        if values.surface_units not in {value for _, value in UNIT_CHOICES}:
            raise ValueError(f"Unsupported surface units: {values.surface_units!r}.")
        source_values = (
            ("base", base_grim, False),
            ("surface", surface_mesh, False),
            ("point CSV", point_csv, True),
            ("line CSV", line_csv, True),
        )
        enabled_point_ids = self.enabled_point_placement_ids
        enabled_line_ids = self.enabled_line_ids
        def snapshot() -> tuple[tuple[str, _FileFingerprint], ...]:
            return tuple(
                (
                    label,
                    _fingerprint_file(
                        path,
                        base_dir=values.base_dir,
                        include_hash=include_hash,
                    ),
                )
                for label, path, include_hash in source_values
                if path
            )

        before = snapshot()
        preview = adapter.preview_inputs(
            base_grim=base_grim or None,
            surface_mesh=surface_mesh or None,
            coordinate_units=values.coordinate_units,
            surface_units=values.surface_units,
            point_locations_csv=point_csv or None,
            line_locations_csv=line_csv or None,
            enabled_point_placement_ids=enabled_point_ids,
            enabled_line_ids=enabled_line_ids,
            base_dir=values.base_dir,
        )
        after = snapshot()
        if (
            enabled_point_ids != self.enabled_point_placement_ids
            or enabled_line_ids != self.enabled_line_ids
        ):
            raise RuntimeError(
                "The spatial feature configuration changed while the input "
                "preview was loading. Preview again to use the current selection."
            )
        if before != after:
            before_by_label = dict(before)
            after_by_label = dict(after)
            if before_by_label.get("point CSV") != after_by_label.get("point CSV"):
                self.invalidate_dataset_requirements("point")
            if before_by_label.get("line CSV") != after_by_label.get("line CSV"):
                self.invalidate_dataset_requirements("line")
            raise RuntimeError(
                "An Assembly input changed while the input preview was loading. "
                "Save it, then preview again."
            )
        requirements = getattr(preview, "dataset_requirements", None)
        after_by_label = dict(after)
        discovery = (
            None
            if requirements is None
            else _DatasetDiscovery(
                requirements=requirements,
                point_fingerprint=after_by_label.get("point CSV"),
                line_fingerprint=after_by_label.get("line CSV"),
            )
        )
        return _VerifiedInputPreview(preview=preview, discovery=discovery)

    def assemble(self, service: Any) -> FeatureBuildDispatch:
        adapter = coerce_feature_workflow(service)
        request = self.build_request(adapter)
        signature = self._semantic_signature()
        service_key = self._service_key(adapter)
        cache = self._prepared_plan_cache
        reused = False
        if (
            cache is not None
            and cache.semantic_signature == signature
            and cache.service_key == service_key
        ):
            fingerprints = self._source_fingerprints()
            if cache.source_fingerprints == fingerprints:
                plan = cache.plan
                reused = True
            else:
                plan = self._prepare_and_cache(
                    adapter,
                    request,
                    before=fingerprints,
                )
        else:
            plan = self._prepare_and_cache(adapter, request)
        # Recheck filesystem aliases immediately before publication. A link may
        # have appeared after the initial request validation while a cached plan
        # was being accepted.
        self._validate_output_target()
        output = adapter.execute(plan)
        return FeatureBuildDispatch(
            plan=plan,
            output_path=str(output),
            reused_validated_plan=reused,
        )


_GUI_IMPORT_ERROR: Exception | None = None
try:  # Keep the model importable on headless/minimal installations.
    from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMenu,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QToolButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment-specific
    _GUI_IMPORT_ERROR = exc


GUI_AVAILABLE = _GUI_IMPORT_ERROR is None


if GUI_AVAILABLE:

    class _OperationWorker(QObject):
        succeeded = Signal(object)
        failed = Signal(str)

        def __init__(self, operation: Callable[[], Any]) -> None:
            super().__init__()
            self._operation = operation

        @Slot()
        def run(self) -> None:
            try:
                result = self._operation()
            except Exception as exc:  # The UI reports authoritative validation errors.
                self.failed.emit(str(exc) or type(exc).__name__)
            else:
                self.succeeded.emit(result)


    class _DisclosureSection(QWidget):
        """Small local disclosure; avoids importing the shell and a Qt cycle."""

        def __init__(
            self,
            title: str,
            parent: QWidget | None = None,
            *,
            expanded: bool = False,
        ) -> None:
            super().__init__(parent)
            self._title = str(title)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self.header = QToolButton(self)
            self.header.setObjectName("sectionHeader")
            self.header.setCheckable(True)
            self.header.setChecked(bool(expanded))
            self.header.setCursor(Qt.CursorShape.PointingHandCursor)
            self.header.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self.body = QWidget(self)
            self.body.setObjectName("sectionBody")
            self.body_layout = QVBoxLayout(self.body)
            self.body_layout.setContentsMargins(8, 8, 8, 8)
            self.body_layout.setSpacing(6)
            layout.addWidget(self.header)
            layout.addWidget(self.body)
            self.header.toggled.connect(self._sync)
            self._sync(bool(expanded))

        def _sync(self, expanded: bool) -> None:
            self.header.setText(("▾  " if expanded else "▸  ") + self._title)
            self.body.setVisible(bool(expanded))

        def addWidget(self, widget: QWidget, stretch: int = 0) -> None:
            self.body_layout.addWidget(widget, stretch)

        def addLayout(self, layout: Any, stretch: int = 0) -> None:
            self.body_layout.addLayout(layout, stretch)


    class _LoadedDatasetButton(QPushButton):
        """Menu button that exposes only backend-usable loaded artifacts."""

        path_selected = Signal(str)
        notice = Signal(str)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__("Use loaded…", parent)
            self.setAutoDefault(False)
            self._catalog: tuple[LoadedDatasetEntry, ...] = ()
            self._menu = QMenu(self)
            self.setMenu(self._menu)
            self._menu.aboutToShow.connect(self._announce_constraints)
            self.set_catalog(())

        def catalog_menu(self) -> QMenu:
            """Return the owned menu for focused UI tests and shell tooling."""

            return self._menu

        def set_catalog(
            self, entries: tuple[LoadedDatasetEntry, ...]
        ) -> None:
            self._catalog = tuple(entries)
            self._menu.clear()
            usable = [entry for entry in self._catalog if entry.usable_path]
            unavailable = [entry for entry in self._catalog if not entry.usable_path]

            if usable:
                for entry in usable:
                    file_name = Path(entry.usable_path).name
                    action = self._menu.addAction(f"{entry.name} — {file_name}")
                    action.setData(entry.dataset_id)
                    action.setToolTip(entry.usable_path)
                    action.setStatusTip(entry.usable_path)
                    action.triggered.connect(
                        lambda _checked=False, path=entry.usable_path: (
                            self.path_selected.emit(path)
                        )
                    )
            else:
                action = self._menu.addAction("No saved .grim datasets available")
                action.setEnabled(False)

            if unavailable:
                self._menu.addSeparator()
                heading = self._menu.addAction(
                    "Save unsaved derived datasets first"
                )
                heading.setEnabled(False)
                for entry in unavailable:
                    reason = entry.unavailable_reason
                    action = self._menu.addAction(f"{entry.name} — {reason}")
                    action.setData(entry.dataset_id)
                    action.setToolTip(_clean_path(entry.path) or reason)
                    action.setEnabled(False)

            usable_count = len(usable)
            unavailable_count = len(unavailable)
            if usable_count:
                tooltip = (
                    f"Choose one of {usable_count} loaded, saved .grim "
                    "dataset(s)."
                )
                if unavailable_count:
                    tooltip += (
                        f" {unavailable_count} unsaved or unavailable "
                        "dataset(s) are disabled; save unsaved derived "
                        "datasets first."
                    )
            else:
                tooltip = (
                    "No usable saved .grim dataset is loaded. Save unsaved "
                    "derived datasets first, or use Browse…."
                )
            self.setToolTip(tooltip)

        def _announce_constraints(self) -> None:
            usable_count = sum(bool(entry.usable_path) for entry in self._catalog)
            unavailable_count = len(self._catalog) - usable_count
            if unavailable_count:
                self.notice.emit(
                    "Assembly requires an existing .grim file. Save unsaved "
                    "derived datasets first; unavailable entries are disabled."
                )
            elif not usable_count:
                self.notice.emit(
                    "No saved .grim dataset is currently loaded. Save the "
                    "required dataset first, or use Browse…."
                )


    class _PathPicker(QWidget):
        editing_finished = Signal()
        catalog_notice = Signal(str)

        def __init__(
            self,
            *,
            caption: str,
            file_filter: str,
            save: bool = False,
            allow_loaded_dataset: bool = False,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.caption = caption
            self.file_filter = file_filter
            self.save = bool(save)
            self.edit = QLineEdit(self)
            self.loaded_button: _LoadedDatasetButton | None = None
            self.button = QPushButton("Browse…", self)
            self.button.setAutoDefault(False)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.edit, 1)
            if allow_loaded_dataset:
                self.loaded_button = _LoadedDatasetButton(self)
                self.loaded_button.path_selected.connect(self._use_loaded_path)
                self.loaded_button.notice.connect(self.catalog_notice.emit)
                layout.addWidget(self.loaded_button)
            layout.addWidget(self.button)
            self.button.clicked.connect(self._browse)
            self.edit.editingFinished.connect(self.editing_finished.emit)

        def path(self) -> str:
            return self.edit.text().strip()

        def set_path(self, path: str) -> None:
            self.edit.setText(_clean_path(path))

        def set_loaded_dataset_catalog(
            self, entries: tuple[LoadedDatasetEntry, ...]
        ) -> None:
            if self.loaded_button is not None:
                self.loaded_button.set_catalog(entries)

        @Slot(str)
        def _use_loaded_path(self, path: str) -> None:
            self.set_path(path)
            self.editing_finished.emit()

        def _browse(self) -> None:
            start = self.path()
            if self.save:
                path, _ = QFileDialog.getSaveFileName(
                    self, self.caption, start, self.file_filter
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, self.caption, start, self.file_filter
                )
            if not path:
                return
            if self.save and Path(path).suffix == "":
                path += ".grim"
            self.set_path(path)
            self.editing_finished.emit()


    class _DatasetMappingEditor(QWidget):
        mapping_changed = Signal()
        catalog_notice = Signal(str)

        def __init__(self, empty_text: str, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._empty_text = empty_text
            self._ids: tuple[str, ...] = ()
            self._required_ids: tuple[str, ...] = ()
            self._catalog: tuple[LoadedDatasetEntry, ...] = ()
            self._loaded_buttons: dict[str, _LoadedDatasetButton] = {}
            self.table = QTableWidget(0, 4, self)
            self.table.setHorizontalHeaderLabels(
                [
                    "dataset_id",
                    "OPN − FRD response (.grim)",
                    "Loaded",
                    "",
                ]
            )
            self.table.setToolTip(
                "Every dataset_id used by the placement CSV must map to the "
                "matching coherent OPN-FRD .grim response."
            )
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setAlternatingRowColors(True)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            self.empty_label = QLabel(empty_text, self)
            self.empty_label.setWordWrap(True)
            self.completeness_label = QLabel(self)
            self.completeness_label.setWordWrap(True)
            self.completeness_label.setObjectName("featureMappingStatus")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.empty_label)
            layout.addWidget(self.table)
            layout.addWidget(self.completeness_label)
            self.table.cellChanged.connect(self._table_changed)
            self.set_dataset_ids(())

        @property
        def dataset_ids(self) -> tuple[str, ...]:
            return self._ids

        def mapping(self) -> dict[str, str]:
            result: dict[str, str] = {}
            for row, dataset_id in enumerate(self._ids):
                item = self.table.item(row, 1)
                result[dataset_id] = "" if item is None else item.text().strip()
            return result

        def missing_ids(self) -> tuple[str, ...]:
            current = self.mapping()
            return tuple(
                dataset_id
                for dataset_id in self._required_ids
                if not _clean_path(current.get(dataset_id))
            )

        def _update_completeness(self) -> None:
            missing = self.missing_ids()
            if not self._ids:
                self.completeness_label.setText("")
                self.completeness_label.setVisible(False)
            elif missing:
                self.completeness_label.setText(
                    f"○ {len(missing)} response file(s) still required: "
                    + ", ".join(missing)
                )
                self.completeness_label.setVisible(True)
            else:
                self.completeness_label.setText(
                    f"✓ All {len(self._required_ids)} enabled response(s) mapped."
                )
                self.completeness_label.setVisible(True)

        def _table_changed(self, *_args: Any) -> None:
            self._update_completeness()
            self.mapping_changed.emit()

        def set_dataset_ids(
            self,
            dataset_ids: tuple[str, ...] | list[str],
            mapping: Mapping[str, str] | None = None,
        ) -> None:
            existing = self.mapping() if self._ids else {}
            if mapping is not None:
                existing.update({str(key): _clean_path(value) for key, value in mapping.items()})
            self._ids = tuple(str(value) for value in dataset_ids)
            self._required_ids = self._ids
            self._loaded_buttons = {}
            self.table.blockSignals(True)
            self.table.setRowCount(len(self._ids))
            for row, dataset_id in enumerate(self._ids):
                id_item = QTableWidgetItem(dataset_id)
                id_item.setFlags(
                    id_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                self.table.setItem(row, 0, id_item)
                self.table.setItem(
                    row, 1, QTableWidgetItem(existing.get(dataset_id, ""))
                )
                loaded_button = _LoadedDatasetButton(self.table)
                loaded_button.set_catalog(self._catalog)
                loaded_button.path_selected.connect(
                    lambda path, key=dataset_id: self.set_path(key, path)
                )
                loaded_button.notice.connect(self.catalog_notice.emit)
                self._loaded_buttons[dataset_id] = loaded_button
                self.table.setCellWidget(row, 2, loaded_button)
                button = QPushButton("Browse…", self.table)
                button.clicked.connect(
                    lambda _checked=False, key=dataset_id: self._browse(key)
                )
                self.table.setCellWidget(row, 3, button)
            self.table.blockSignals(False)
            has_rows = bool(self._ids)
            self.empty_label.setVisible(not has_rows)
            self.table.setVisible(has_rows)
            self.table.setMinimumHeight(112 if has_rows else 0)
            self._update_completeness()

        def set_required_dataset_ids(self, dataset_ids: Iterable[str]) -> None:
            required = tuple(dict.fromkeys(str(value) for value in dataset_ids))
            unknown = sorted(set(required) - set(self._ids))
            if unknown:
                raise ValueError(
                    f"Enabled spatial features reference unknown dataset IDs {unknown}."
                )
            self._required_ids = required
            self._update_completeness()

        def set_loaded_dataset_catalog(
            self, entries: tuple[LoadedDatasetEntry, ...]
        ) -> None:
            self._catalog = tuple(entries)
            for button in self._loaded_buttons.values():
                button.set_catalog(self._catalog)

        def loaded_dataset_button(
            self, dataset_id: str
        ) -> _LoadedDatasetButton:
            """Return the row's chooser without exposing table-column details."""

            try:
                return self._loaded_buttons[str(dataset_id)]
            except KeyError as exc:
                raise KeyError(f"Unknown dataset_id {dataset_id!r}.") from exc

        def set_path(self, dataset_id: str, path: str) -> None:
            try:
                row = self._ids.index(str(dataset_id))
            except ValueError as exc:
                raise KeyError(f"Unknown dataset_id {dataset_id!r}.") from exc
            self.table.item(row, 1).setText(_clean_path(path))

        def _browse(self, dataset_id: str) -> None:
            current = self.mapping().get(dataset_id, "")
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Choose OPN-FRD response for {dataset_id}",
                current,
                "GRIM response (*.grim);;All files (*)",
            )
            if path:
                self.set_path(dataset_id, path)


    class _SpatialFeatureTree(QTreeWidget):
        """Checkable spatial definition tree, separate from response math."""

        selection_changed = Signal()
        _ROLE_KIND = Qt.ItemDataRole.UserRole
        _ROLE_INSTANCE_ID = Qt.ItemDataRole.UserRole + 1
        _USE_COLUMN = 2

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setColumnCount(3)
            self.setHeaderLabels(["Body / spatial features", "Response", "Use"])
            self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.header().setSectionResizeMode(
                1, QHeaderView.ResizeMode.ResizeToContents
            )
            self.header().setSectionResizeMode(
                2, QHeaderView.ResizeMode.ResizeToContents
            )
            self.headerItem().setToolTip(
                self._USE_COLUMN,
                "Include this spatial feature in preview, physical validation, "
                "response loading, and assembly.",
            )
            self.setAlternatingRowColors(True)
            self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.setMinimumHeight(190)
            self._syncing = False
            self._filter_text = ""
            self.itemChanged.connect(self._on_item_changed)

        @staticmethod
        def _search_text(item: QTreeWidgetItem) -> str:
            return " ".join(item.text(column) for column in (0, 1)).casefold()

        def _set_default_expansion(self) -> None:
            """Expand structural roots while leaving response groups compact."""

            for top_index in range(self.topLevelItemCount()):
                body = self.topLevelItem(top_index)
                body.setExpanded(True)
                for kind_index in range(body.childCount()):
                    kind_root = body.child(kind_index)
                    kind_root.setExpanded(True)
                    for dataset_index in range(kind_root.childCount()):
                        kind_root.child(dataset_index).setExpanded(False)

        def set_filter_text(self, text: str) -> None:
            """Filter by instance, dataset ID, or mapped response text.

            Filtering is display-only. Hidden leaves remain part of recursive
            ``Use`` operations and :meth:`excluded_ids`, so searching can never
            silently change assembly membership.
            """

            query = str(text or "").strip().casefold()
            self._filter_text = query

            def visit(item: QTreeWidgetItem, ancestor_matches: bool = False) -> bool:
                own_match = bool(query) and query in self._search_text(item)
                reveal_subtree = ancestor_matches or own_match
                child_visible = False
                for index in range(item.childCount()):
                    child_visible = (
                        visit(item.child(index), reveal_subtree) or child_visible
                    )
                visible = not query or reveal_subtree or child_visible
                item.setHidden(not visible)
                if query and visible and item.childCount():
                    item.setExpanded(True)
                return visible

            for top_index in range(self.topLevelItemCount()):
                visit(self.topLevelItem(top_index))
            if not query:
                self._set_default_expansion()

        @staticmethod
        def _checkable_flags(item: QTreeWidgetItem) -> None:
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )

        def _set_checked(self, item: QTreeWidgetItem, enabled: bool) -> None:
            item.setCheckState(
                self._USE_COLUMN,
                Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked,
            )

        def _set_subtree(self, item: QTreeWidgetItem, enabled: bool) -> None:
            if item.data(0, self._ROLE_KIND) != "body":
                self._set_checked(item, enabled)
            for index in range(item.childCount()):
                self._set_subtree(item.child(index), enabled)

        def _sync_ancestors(self, item: QTreeWidgetItem | None) -> None:
            current = item
            while current is not None:
                # The body is the required host response, not a switchable
                # feature group. Keep its Use cell labelled ``Required`` while
                # its point/line children summarize their own selections.
                if current.data(0, self._ROLE_KIND) == "body":
                    break
                states = [
                    current.child(index).checkState(self._USE_COLUMN)
                    for index in range(current.childCount())
                ]
                if states:
                    if all(state == Qt.CheckState.Checked for state in states):
                        state = Qt.CheckState.Checked
                    elif all(state == Qt.CheckState.Unchecked for state in states):
                        state = Qt.CheckState.Unchecked
                    else:
                        state = Qt.CheckState.PartiallyChecked
                    current.setCheckState(self._USE_COLUMN, state)
                current = current.parent()

        @Slot(QTreeWidgetItem, int)
        def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
            if self._syncing or column != self._USE_COLUMN:
                return
            if item.data(0, self._ROLE_KIND) == "body":
                return
            state = item.checkState(self._USE_COLUMN)
            self._syncing = True
            try:
                if state in (Qt.CheckState.Checked, Qt.CheckState.Unchecked):
                    enabled = state == Qt.CheckState.Checked
                    for index in range(item.childCount()):
                        self._set_subtree(item.child(index), enabled)
                self._sync_ancestors(item.parent())
            finally:
                self._syncing = False
            self.selection_changed.emit()

        def set_configuration(self, model: FeatureAssemblyFormModel) -> None:
            """Rebuild from parsed descriptors while honoring model exclusions."""
            self._syncing = True
            try:
                self.clear()
                body_name = Path(_clean_path(model.values.base_grim)).name
                body = QTreeWidgetItem(
                    ["Body", body_name or "clean-body response not selected", "Required"]
                )
                body.setData(0, self._ROLE_KIND, "body")
                body.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                body.setToolTip(
                    0,
                    "The clean-body response is always required. Feature checkboxes "
                    "control installed-minus-clean deltas added to it.\n"
                    f"Response: {_clean_path(model.values.base_grim) or 'not selected'}\n"
                    f"Surface: {_clean_path(model.values.surface_mesh) or 'embedded/not selected'}\n"
                    f"Surface units: {model.values.surface_units}; "
                    f"flip normals: {bool(model.values.flip_surface_normals)}",
                )
                self.addTopLevelItem(body)

                self._add_kind(
                    model,
                    parent=body,
                    kind="point",
                    label="Point features",
                    descriptors=model.point_instances,
                    mappings=model.values.point_datasets,
                    excluded=model.values.excluded_point_placement_ids,
                )
                self._add_kind(
                    model,
                    parent=body,
                    kind="line",
                    label="Line features",
                    descriptors=model.line_instances,
                    mappings=model.values.line_datasets,
                    excluded=model.values.excluded_line_ids,
                )
                self.set_filter_text(self._filter_text)
            finally:
                self._syncing = False

        def _add_kind(
            self,
            model: FeatureAssemblyFormModel,
            *,
            parent: QTreeWidgetItem,
            kind: str,
            label: str,
            descriptors: tuple,
            mappings: Mapping[str, str],
            excluded: set[str],
        ) -> None:
            root = QTreeWidgetItem([f"{label} ({len(descriptors)})", "", ""])
            root.setData(0, self._ROLE_KIND, f"{kind}_root")
            self._checkable_flags(root)
            parent.addChild(root)
            by_dataset: dict[str, list[tuple]] = {}
            for descriptor in descriptors:
                by_dataset.setdefault(str(descriptor[1]), []).append(descriptor)
            for dataset_id, instances in by_dataset.items():
                mapped_path = _clean_path(mappings.get(dataset_id))
                mapped = Path(mapped_path).name or "not mapped"
                group = QTreeWidgetItem(
                    [f"dataset_id: {dataset_id} ({len(instances)})", mapped, ""]
                )
                group.setData(0, self._ROLE_KIND, f"{kind}_dataset")
                group.setToolTip(
                    1,
                    mapped_path or "No OPN-FRD response is mapped for this dataset ID.",
                )
                self._checkable_flags(group)
                root.addChild(group)
                for descriptor in instances:
                    instance_id = str(descriptor[0])
                    suffix = (
                        ""
                        if kind == "point"
                        else f" ({int(descriptor[2])} segment(s))"
                    )
                    leaf = QTreeWidgetItem([instance_id + suffix, "", ""])
                    leaf.setData(0, self._ROLE_KIND, kind)
                    leaf.setData(0, self._ROLE_INSTANCE_ID, instance_id)
                    self._checkable_flags(leaf)
                    self._set_checked(leaf, instance_id not in excluded)
                    group.addChild(leaf)
                self._sync_ancestors(group)
            self._sync_ancestors(root)

        def excluded_ids(self) -> tuple[set[str], set[str]]:
            point_ids: set[str] = set()
            line_ids: set[str] = set()
            iterator = self.invisibleRootItem()
            pending = [iterator.child(index) for index in range(iterator.childCount())]
            while pending:
                item = pending.pop()
                pending.extend(
                    item.child(index) for index in range(item.childCount())
                )
                kind = item.data(0, self._ROLE_KIND)
                instance_id = item.data(0, self._ROLE_INSTANCE_ID)
                if (
                    kind in {"point", "line"}
                    and instance_id
                    and item.checkState(self._USE_COLUMN) == Qt.CheckState.Unchecked
                ):
                    (point_ids if kind == "point" else line_ids).add(
                        str(instance_id)
                    )
            return point_ids, line_ids


    class FeatureAssemblyPanel(QWidget):
        """New-user-facing feature assembly form with background execution."""

        preview_ready = Signal(object)
        preview_stale = Signal(str)
        feature_built = Signal(str)
        build_failed = Signal(str)
        status_changed = Signal(str)

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            service: Any = None,
        ) -> None:
            super().__init__(parent)
            self.model = FeatureAssemblyFormModel()
            self._service: Any = service
            self._thread: QThread | None = None
            self._worker: _OperationWorker | None = None
            self._active_kind = ""
            self._discovery_paths: tuple[str, str] | None = None
            self._preview_is_current = False
            self._loaded_dataset_catalog: tuple[LoadedDatasetEntry, ...] = ()
            self._build_ui()

        def _build_ui(self) -> None:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(6, 6, 6, 6)
            outer.setSpacing(6)

            intro = QLabel(
                "Place point scatterers and expanded line sources on a clean body, "
                "check them in 3-D, then save one coherent response.",
                self,
            )
            intro.setWordWrap(True)
            intro.setObjectName("featurePanelIntro")
            outer.addWidget(intro)

            self.workflow_steps_label = QLabel(
                "1  Body   ›   2  Place   ›   3  Map   ›   4  Review",
                self,
            )
            self.workflow_steps_label.setWordWrap(False)
            self.workflow_steps_label.setObjectName("featureWorkflowSteps")
            outer.addWidget(self.workflow_steps_label)
            self.next_step_label = QLabel(self)
            self.next_step_label.setObjectName("featureNextStep")
            self.next_step_label.setWordWrap(True)
            outer.addWidget(self.next_step_label)

            scroll = QScrollArea(self)
            scroll.setObjectName("featureAssemblyScroll")
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setAutoFillBackground(False)
            scroll.viewport().setAutoFillBackground(False)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            content = QWidget(scroll)
            content.setObjectName("featureAssemblyContent")
            content.setAutoFillBackground(False)
            self.form_content = content
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(7)

            body_group = QGroupBox("1  Choose the body", content)
            body_group.setObjectName("featureStepCard")
            body_form = QFormLayout(body_group)
            body_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.base_picker = _PathPicker(
                caption="Choose clean-body/base GRIM",
                file_filter="GRIM response (*.grim);;All files (*)",
                allow_loaded_dataset=True,
            )
            self.surface_picker = _PathPicker(
                caption="Choose body surface mesh",
                file_filter="Surface mesh (*.stl *.facet);;All files (*)",
            )
            self.output_picker = _PathPicker(
                caption="Save assembled GRIM",
                file_filter="GRIM response (*.grim);;All files (*)",
                save=True,
            )
            self.coordinate_units = QComboBox(body_group)
            self.surface_units = QComboBox(body_group)
            for label, value in UNIT_CHOICES:
                self.coordinate_units.addItem(label, value)
                self.surface_units.addItem(label, value)
            self.flip_normals = QCheckBox("Flip mesh normals", body_group)
            self.shadow = QCheckBox(
                "Apply geometric body shadowing (requires mesh)", body_group
            )
            mesh_options = QWidget(body_group)
            mesh_layout = QVBoxLayout(mesh_options)
            mesh_layout.setContentsMargins(0, 0, 0, 0)
            mesh_layout.addWidget(self.flip_normals)
            mesh_layout.addWidget(self.shadow)
            self.base_picker.setToolTip(
                "Clean-body response to which the point and/or line feature "
                "responses will be coherently added."
            )
            self.surface_picker.setToolTip(
                "Choose the matching STL/facet surface for a 3-D body. Leave "
                "blank when the base GRIM contains an embedded BoR profile."
            )
            self.coordinate_units.setToolTip(
                "Units used by every x/y/z coordinate in both placement CSVs."
            )
            self.surface_units.setToolTip(
                "Units of the selected STL/facet surface, independent of the CSV units."
            )
            body_form.addRow("Clean-body response:", self.base_picker)
            body_form.addRow("Surface mesh (optional):", self.surface_picker)
            body_form.addRow("Surface mesh units:", self.surface_units)
            body_form.addRow("Mesh options:", mesh_options)
            self.body_preview_help = QLabel(
                "Body preview: selected mesh, or the base file's embedded BoR. "
                "A 3-D base without embedded geometry needs its matching mesh.",
                body_group,
            )
            self.body_preview_help.setWordWrap(True)
            self.body_preview_help.setObjectName("featureHint")
            body_form.addRow("", self.body_preview_help)
            content_layout.addWidget(body_group)

            feature_group = QGroupBox("2–3  Add placements and map responses", content)
            feature_group.setObjectName("featureStepCard")
            feature_layout = QVBoxLayout(feature_group)
            units_row = QHBoxLayout()
            units_label = QLabel("All CSV coordinates:", feature_group)
            units_label.setBuddy(self.coordinate_units)
            units_row.addWidget(units_label)
            units_row.addWidget(self.coordinate_units, 1)
            feature_layout.addLayout(units_row)
            self.shared_units_label = QLabel(
                "One point CSV can contain every point family; one line CSV can "
                "contain every ordered line chain. Both selected CSVs share this unit.",
                feature_group,
            )
            self.shared_units_label.setObjectName("featureHint")
            self.shared_units_label.setWordWrap(True)
            feature_layout.addWidget(self.shared_units_label)
            self.feature_summary_label = QLabel(feature_group)
            self.feature_summary_label.setObjectName("featureSummary")
            self.feature_summary_label.setWordWrap(True)
            feature_layout.addWidget(self.feature_summary_label)
            self.feature_tabs = QTabWidget(feature_group)

            point_page = QWidget(self.feature_tabs)
            point_layout = QVBoxLayout(point_page)
            self.point_csv_picker = _PathPicker(
                caption="Choose point placement CSV",
                file_filter="CSV placement file (*.csv);;All files (*)",
            )
            self.point_csv_picker.setToolTip(
                "Strict GHOST point-placement CSV. This is the same file used "
                "by local scripts and the HPC workflow."
            )
            point_layout.addWidget(QLabel("Point location/orientation CSV:"))
            point_layout.addWidget(self.point_csv_picker)
            self.point_csv_summary = QLabel("No point CSV selected.", point_page)
            self.point_csv_summary.setObjectName("featureCsvSummary")
            self.point_csv_summary.setWordWrap(True)
            point_layout.addWidget(self.point_csv_summary)
            self.point_help_label = QLabel(
                "This is the same strict GHOST CSV used locally and on HPC. The "
                "header is followed directly by data rows—no units row or comments. "
                "Normal is local +z; projected roll is local +x / azimuth zero. "
                "placement_id values must be unique.",
                point_page,
            )
            self.point_help_label.setWordWrap(True)
            self.point_help_label.setVisible(False)
            point_layout.addWidget(self.point_help_label)
            point_format_row = QHBoxLayout()
            self.point_format_button = QPushButton(
                "CSV guide", point_page
            )
            self.point_format_button.setCheckable(True)
            point_format_row.addWidget(self.point_format_button)
            self.point_schema_label = QLabel(
                "Exact header (column order is fixed):\n"
                + ",".join(POINT_PLACEMENT_COLUMNS)
                + "\nExample row:\n"
                + POINT_PLACEMENT_EXAMPLE,
                point_page,
            )
            self.point_schema_label.setWordWrap(True)
            self.point_schema_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.point_schema_label.setObjectName("featureCsvSchema")
            self.point_template_button = QPushButton(
                "Save template…", point_page
            )
            self.point_template_button.setToolTip(
                "Write the exact required point header to a new .csv file."
            )
            point_format_row.addWidget(self.point_template_button)
            self.point_clear_button = QPushButton("Remove", point_page)
            self.point_clear_button.setToolTip(
                "Remove the point CSV and its response mappings from this build."
            )
            point_format_row.addWidget(self.point_clear_button)
            point_format_row.addStretch(1)
            point_layout.addLayout(point_format_row)
            self.point_schema_label.setVisible(False)
            point_layout.addWidget(self.point_schema_label)
            point_response_help = QLabel(
                "Response contract: each dataset_id maps to a coherent OPN − FRD "
                "delta (installed/featured minus clean skin), with VV, HH, and "
                "reciprocal cross-polar response.",
                point_page,
            )
            point_response_help.setWordWrap(True)
            point_response_help.setObjectName("featureContract")
            point_layout.addWidget(point_response_help)
            self.point_mapping = _DatasetMappingEditor(
                "Choose a point CSV; one response row will appear per dataset_id.",
                point_page,
            )
            point_layout.addWidget(self.point_mapping)
            self.feature_tabs.addTab(point_page, "Point features")

            line_page = QWidget(self.feature_tabs)
            line_layout = QVBoxLayout(line_page)
            self.line_csv_picker = _PathPicker(
                caption="Choose line placement CSV",
                file_filter="CSV placement file (*.csv);;All files (*)",
            )
            self.line_csv_picker.setToolTip(
                "Strict GHOST ordered-segment line-placement CSV. This is the "
                "same file used by local scripts and the HPC workflow."
            )
            line_layout.addWidget(QLabel("Line path/orientation CSV:"))
            line_layout.addWidget(self.line_csv_picker)
            self.line_csv_summary = QLabel("No line CSV selected.", line_page)
            self.line_csv_summary.setObjectName("featureCsvSummary")
            self.line_csv_summary.setWordWrap(True)
            line_layout.addWidget(self.line_csv_summary)
            self.line_help_label = QLabel(
                "This is the same strict GHOST CSV used locally and on HPC. The "
                "header is followed directly by data rows—no units row or comments. "
                "Rows for each line_id stay together, segment_index starts at 1, "
                "segments meet head-to-tail, and endpoint normals point outward.",
                line_page,
            )
            self.line_help_label.setWordWrap(True)
            self.line_help_label.setVisible(False)
            line_layout.addWidget(self.line_help_label)
            line_format_row = QHBoxLayout()
            self.line_format_button = QPushButton(
                "CSV guide", line_page
            )
            self.line_format_button.setCheckable(True)
            line_format_row.addWidget(self.line_format_button)
            self.line_schema_label = QLabel(
                "Exact header (column order is fixed):\n"
                + ",".join(LINE_PLACEMENT_COLUMNS)
                + "\nExample row:\n"
                + LINE_PLACEMENT_EXAMPLE,
                line_page,
            )
            self.line_schema_label.setWordWrap(True)
            self.line_schema_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.line_schema_label.setObjectName("featureCsvSchema")
            self.line_template_button = QPushButton(
                "Save template…", line_page
            )
            self.line_template_button.setToolTip(
                "Write the exact required line header to a new .csv file."
            )
            line_format_row.addWidget(self.line_template_button)
            self.line_clear_button = QPushButton("Remove", line_page)
            self.line_clear_button.setToolTip(
                "Remove the line CSV and its response mappings from this build."
            )
            line_format_row.addWidget(self.line_clear_button)
            line_format_row.addStretch(1)
            line_layout.addLayout(line_format_row)
            self.line_schema_label.setVisible(False)
            line_layout.addWidget(self.line_schema_label)
            line_response_help = QLabel(
                "Response contract: each dataset_id maps to a coherent OPN − FRD "
                "delta (installed/featured minus clean skin) containing TE and TM.",
                line_page,
            )
            line_response_help.setWordWrap(True)
            line_response_help.setObjectName("featureContract")
            line_layout.addWidget(line_response_help)
            self.line_mapping = _DatasetMappingEditor(
                "Choose a line CSV; one response row will appear per dataset_id.",
                line_page,
            )
            line_layout.addWidget(self.line_mapping)
            self.feature_tabs.addTab(line_page, "Line features")
            feature_layout.addWidget(self.feature_tabs)
            scan_row = QHBoxLayout()
            self.scan_button = QPushButton("Refresh selected CSVs", feature_group)
            self.scan_button.setToolTip(
                "Parse the selected CSVs with the authoritative GHOST parser "
                "and list every response dataset that must be supplied."
            )
            scan_row.addWidget(self.scan_button)
            scan_hint = QLabel(
                "CSV files are read automatically after Browse.", feature_group
            )
            scan_hint.setObjectName("featureHint")
            scan_hint.setWordWrap(True)
            scan_row.addWidget(scan_hint, 1)
            feature_layout.addLayout(scan_row)
            hierarchy_help = QLabel(
                "Spatial configuration — separate from whole-response dataset "
                "arithmetic. Uncheck a dataset family or individual placement "
                "to omit it from preview, physical validation, response loading, "
                "and assembly. The CSV itself is never rewritten.",
                feature_group,
            )
            hierarchy_help.setWordWrap(True)
            hierarchy_help.setObjectName("featureHint")
            feature_layout.addWidget(hierarchy_help)
            filter_row = QHBoxLayout()
            filter_label = QLabel("Find feature:", feature_group)
            self.spatial_feature_filter = QLineEdit(feature_group)
            self.spatial_feature_filter.setPlaceholderText(
                "Instance ID, dataset ID, or response file"
            )
            self.spatial_feature_filter.setClearButtonEnabled(True)
            self.spatial_feature_filter.setToolTip(
                "Filters the displayed hierarchy only. A parent Use checkbox still "
                "applies recursively to its complete subtree, including hidden items."
            )
            filter_row.addWidget(filter_label)
            filter_row.addWidget(self.spatial_feature_filter, 1)
            feature_layout.addLayout(filter_row)
            self.spatial_feature_tree = _SpatialFeatureTree(feature_group)
            feature_layout.addWidget(self.spatial_feature_tree)
            self.spatial_feature_filter.textChanged.connect(
                self.spatial_feature_tree.set_filter_text
            )
            self.spatial_selection_summary = QLabel(feature_group)
            self.spatial_selection_summary.setWordWrap(True)
            self.spatial_selection_summary.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.spatial_selection_summary.setToolTip(
                "Large disabled-ID lists are shortened here. Use Copy full selection "
                "to record exact trade-study membership."
            )
            self.spatial_selection_summary.setObjectName("featureSummary")
            summary_row = QHBoxLayout()
            summary_row.addWidget(self.spatial_selection_summary, 1)
            self.copy_spatial_selection_button = QPushButton(
                "Copy full selection", feature_group
            )
            self.copy_spatial_selection_button.setToolTip(
                "Copy the complete unshortened enabled/disabled membership summary."
            )
            self.copy_spatial_selection_button.clicked.connect(
                self._copy_full_spatial_selection_summary
            )
            summary_row.addWidget(self.copy_spatial_selection_button)
            feature_layout.addLayout(summary_row)
            content_layout.addWidget(feature_group)

            advanced = QWidget(content)
            advanced_form = QFormLayout(advanced)
            advanced_form.setContentsMargins(8, 8, 8, 8)
            advanced_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.skin_tol = QDoubleSpinBox(advanced)
            self.skin_tol.setDecimals(6)
            self.skin_tol.setRange(0.0, 1.0e3)
            self.skin_tol.setValue(1.0e-3)
            self.skin_tol.setSuffix(" m")
            self.phase_tol = QDoubleSpinBox(advanced)
            self.phase_tol.setDecimals(2)
            self.phase_tol.setRange(0.0, 1.0e6)
            self.phase_tol.setValue(15.0)
            self.phase_tol.setSuffix("°")
            self.normal_tol = QDoubleSpinBox(advanced)
            self.normal_tol.setDecimals(2)
            self.normal_tol.setRange(0.0, 89.99)
            self.normal_tol.setValue(15.0)
            self.normal_tol.setSuffix("°")
            self.shadow_bias = QLineEdit(advanced)
            self.shadow_bias.setPlaceholderText("Auto (recommended)")
            advanced_form.addRow("Maximum skin distance:", self.skin_tol)
            advanced_form.addRow("Maximum two-way phase error:", self.phase_tol)
            advanced_form.addRow("Maximum normal mismatch:", self.normal_tol)
            advanced_form.addRow("Shadow ray bias (m):", self.shadow_bias)
            self.advanced_section = _DisclosureSection(
                "Advanced placement checks · defaults active",
                content,
                expanded=False,
            )
            self.advanced_section.addWidget(advanced)
            self.advanced_section.header.setToolTip(
                "The displayed defaults remain active while this section is collapsed."
            )
            content_layout.addWidget(self.advanced_section)

            self.preview_help_label = QLabel(
                "Preview Geometry is visual QA only. Validate Placements additionally "
                "checks skin distance, outward normals, frame validity, and response "
                "mappings. Magenta arrows are normals; lavender arrows are point-roll "
                "references. Preview Layers → Show changes only the display. Spatial "
                "Feature Configuration → Use controls which parsed instances enter "
                "preview, validation, response loading, and build.",
                content,
            )
            self.preview_help_label.setWordWrap(True)
            self.preview_guide = _DisclosureSection(
                "How to read the 3-D preview", content, expanded=False
            )
            self.preview_guide.addWidget(self.preview_help_label)
            content_layout.addWidget(self.preview_guide)

            review_group = QGroupBox("4  Review and save", content)
            review_group.setObjectName("featureStepCard")
            review_form = QFormLayout(review_group)
            review_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            review_form.addRow("Output response:", self.output_picker)
            self.readiness_label = QLabel(review_group)
            self.readiness_label.setObjectName("featureReadiness")
            self.readiness_label.setWordWrap(True)
            review_form.addRow("Ready check:", self.readiness_label)
            self.build_summary_label = QLabel(review_group)
            self.build_summary_label.setObjectName("featureBuildSummary")
            self.build_summary_label.setWordWrap(True)
            review_form.addRow("This build:", self.build_summary_label)
            content_layout.addWidget(review_group)
            content_layout.addStretch(1)
            scroll.setWidget(content)
            outer.addWidget(scroll, 1)

            self.status_label = QLabel(
                "No Assembly operation is running.",
                self,
            )
            self.status_label.setObjectName("featureAssemblyStatus")
            self.status_label.setWordWrap(True)
            self.status_label.setFrameShape(QFrame.Shape.StyledPanel)
            self.status_label.setMargin(6)
            outer.addWidget(self.status_label)

            action_row = QHBoxLayout()
            self.input_preview_button = QPushButton("Preview geometry", self)
            self.input_preview_button.setToolTip(
                "Show available body geometry and enabled CSV locations without "
                "requiring response mappings or an output path. This is visual QA only."
            )
            self.preview_button = QPushButton("Validate placements", self)
            self.preview_button.setToolTip(
                "Validate body skin, normals, and response mapping completeness, then "
                "show the prepared body and features in the 3-D Assembly view."
            )
            self.build_button = QPushButton("Assemble && save", self)
            self.build_button.setToolTip(
                "Run the same validation, coherently add every enabled mapped feature, "
                "and save the selected output .grim file."
            )
            self.build_button.setDefault(True)
            action_row.addWidget(self.input_preview_button)
            action_row.addWidget(self.preview_button)
            action_row.addWidget(self.build_button)
            outer.addLayout(action_row)

            self.status_changed.connect(self.status_label.setText)
            self.base_picker.editing_finished.connect(self._base_path_changed)
            self.surface_picker.editing_finished.connect(self._input_setting_changed)
            self.output_picker.editing_finished.connect(self._output_path_changed)
            self.point_csv_picker.editing_finished.connect(
                lambda: self._placement_csv_changed("point")
            )
            self.line_csv_picker.editing_finished.connect(
                lambda: self._placement_csv_changed("line")
            )
            self.coordinate_units.currentIndexChanged.connect(
                self._mark_preview_stale
            )
            self.surface_units.currentIndexChanged.connect(self._mark_preview_stale)
            self.flip_normals.toggled.connect(self._mark_preview_stale)
            self.shadow.toggled.connect(self._mark_preview_stale)
            self.skin_tol.valueChanged.connect(self._mark_preview_stale)
            self.phase_tol.valueChanged.connect(self._mark_preview_stale)
            self.normal_tol.valueChanged.connect(self._mark_preview_stale)
            self.shadow_bias.editingFinished.connect(self._mark_preview_stale)
            self.point_mapping.mapping_changed.connect(self._mapping_changed)
            self.line_mapping.mapping_changed.connect(self._mapping_changed)
            self.spatial_feature_tree.selection_changed.connect(
                self._spatial_selection_changed
            )
            self.base_picker.catalog_notice.connect(
                self._loaded_dataset_notice
            )
            self.point_mapping.catalog_notice.connect(
                self._loaded_dataset_notice
            )
            self.line_mapping.catalog_notice.connect(
                self._loaded_dataset_notice
            )
            self.point_format_button.toggled.connect(
                lambda checked: self._toggle_schema_help("point", checked)
            )
            self.line_format_button.toggled.connect(
                lambda checked: self._toggle_schema_help("line", checked)
            )
            self.point_template_button.clicked.connect(
                lambda _checked=False: self._save_template("point")
            )
            self.line_template_button.clicked.connect(
                lambda _checked=False: self._save_template("line")
            )
            self.point_clear_button.clicked.connect(
                lambda _checked=False: self._clear_placement_csv("point")
            )
            self.line_clear_button.clicked.connect(
                lambda _checked=False: self._clear_placement_csv("line")
            )
            self.scan_button.clicked.connect(self.refresh_dataset_ids)
            self.input_preview_button.clicked.connect(self.preview_inputs)
            self.preview_button.clicked.connect(self.validate_and_preview)
            self.build_button.clicked.connect(self.assemble_and_save)
            self._refresh_spatial_feature_tree()
            self._update_workflow_readiness()

        def set_service(self, service: Any) -> None:
            coerce_feature_workflow(service)  # Fail early with an actionable API error.
            self._service = service
            self._update_workflow_readiness()

        def service(self) -> Any:
            return self._service

        def set_loaded_dataset_catalog(self, entries: Iterable[Any]) -> None:
            """Offer saved, file-backed GRIM rows without replacing Browse.

            The combined shell may pass its existing stable dataset catalog.
            Entries can be :class:`LoadedDatasetEntry` instances, mappings,
            ``(dataset_id, name, path[, dirty])`` tuples, or objects exposing
            equivalent attributes. Dirty/in-memory/missing entries remain
            visible as disabled explanations so users know to save first.
            """

            catalog = _coerce_loaded_dataset_catalog(entries)
            self._loaded_dataset_catalog = catalog
            self.base_picker.set_loaded_dataset_catalog(catalog)
            self.point_mapping.set_loaded_dataset_catalog(catalog)
            self.line_mapping.set_loaded_dataset_catalog(catalog)

        def loaded_dataset_catalog(self) -> tuple[LoadedDatasetEntry, ...]:
            """Return the last normalized catalog snapshot."""

            return self._loaded_dataset_catalog

        def set_base_grim(self, path: str) -> None:
            self.base_picker.set_path(path)
            self._base_path_changed()

        def set_surface_mesh(self, path: str) -> None:
            self.surface_picker.set_path(path)
            self._mark_preview_stale()

        def set_point_csv(self, path: str, *, discover: bool = True) -> None:
            self.point_csv_picker.set_path(path)
            if discover:
                self._placement_csv_changed("point")
            else:
                if self.model.feature_selection_source_changed("point", path):
                    self.model.clear_feature_selection("point")
                self.model.values.point_locations_csv = _clean_path(path)
                self.model.invalidate_dataset_requirements("point")
                self.point_mapping.set_dataset_ids(())
                self._refresh_spatial_feature_tree()
                self._mark_preview_stale()

        def set_line_csv(self, path: str, *, discover: bool = True) -> None:
            self.line_csv_picker.set_path(path)
            if discover:
                self._placement_csv_changed("line")
            else:
                if self.model.feature_selection_source_changed("line", path):
                    self.model.clear_feature_selection("line")
                self.model.values.line_locations_csv = _clean_path(path)
                self.model.invalidate_dataset_requirements("line")
                self.line_mapping.set_dataset_ids(())
                self._refresh_spatial_feature_tree()
                self._mark_preview_stale()

        def set_output_grim(self, path: str) -> None:
            self.output_picker.set_path(path)
            self._output_path_changed()

        @Slot(str)
        def _loaded_dataset_notice(self, message: str) -> None:
            self.status_changed.emit(message)

        def _update_workflow_readiness(self) -> None:
            """Keep the compact step summary and actions honest and actionable."""

            values = self.model.values
            values.base_grim = self.base_picker.path()
            values.output_grim = self.output_picker.path()
            values.surface_mesh = self.surface_picker.path()
            values.point_locations_csv = self.point_csv_picker.path()
            values.line_locations_csv = self.line_csv_picker.path()
            values.point_datasets = self.point_mapping.mapping()
            values.line_datasets = self.line_mapping.mapping()
            values.coordinate_units = str(self.coordinate_units.currentData())

            point_selected = bool(values.point_locations_csv)
            line_selected = bool(values.line_locations_csv)
            try:
                point_current = (
                    point_selected and self.model.requirements_look_current("point")
                )
                line_current = (
                    line_selected and self.model.requirements_look_current("line")
                )
            except Exception:
                point_current = False
                line_current = False

            point_ids = len(self.model.point_dataset_ids)
            line_ids = len(self.model.line_dataset_ids)
            point_count = self.model.point_placement_count
            line_count = self.model.line_path_count
            segment_count = self.model.line_segment_count
            missing_mappings = self.model.missing_dataset_mappings()
            point_missing = tuple(
                value.split(":", 1)[1]
                for value in missing_mappings
                if value.startswith("point:")
            )
            line_missing = tuple(
                value.split(":", 1)[1]
                for value in missing_mappings
                if value.startswith("line:")
            )
            active_point_ids = self.model.active_point_dataset_ids()
            active_line_ids = self.model.active_line_dataset_ids()
            enabled_point_count = len(
                self.model.enabled_point_placement_ids or ()
            )
            enabled_line_count = len(self.model.enabled_line_ids or ())
            try:
                adapter = coerce_feature_workflow(self._service)
                service_ready = True
            except (RuntimeError, TypeError):
                adapter = None
                service_ready = False

            def existing_file(path: str) -> bool:
                if not _clean_path(path):
                    return False
                try:
                    return _resolved_user_path(
                        path, base_dir=values.base_dir
                    ).is_file()
                except OSError:
                    return False

            def existing_grim_file(path: str) -> bool:
                if not _clean_path(path):
                    return False
                try:
                    resolved = _resolved_user_path(path, base_dir=values.base_dir)
                    return (
                        resolved.is_file()
                        and resolved.suffix.casefold() == ".grim"
                    )
                except OSError:
                    return False

            def existing_surface_file(path: str) -> bool:
                if not _clean_path(path):
                    return False
                try:
                    resolved = _resolved_user_path(path, base_dir=values.base_dir)
                    return (
                        resolved.is_file()
                        and resolved.suffix.casefold() in {".stl", ".facet"}
                    )
                except OSError:
                    return False

            point_unusable = tuple(
                dataset_id
                for dataset_id in active_point_ids
                if _clean_path(values.point_datasets.get(dataset_id))
                and not existing_grim_file(values.point_datasets.get(dataset_id, ""))
            )
            line_unusable = tuple(
                dataset_id
                for dataset_id in active_line_ids
                if _clean_path(values.line_datasets.get(dataset_id))
                and not existing_grim_file(values.line_datasets.get(dataset_id, ""))
            )

            if not point_selected:
                point_text = "No point CSV selected."
                point_tab = "Point features"
            elif not point_current:
                point_text = "Point CSV needs refresh."
                point_tab = "Point features · refresh"
            else:
                count_label = (
                    f"{point_count} placement(s)"
                    if point_count
                    else f"{point_ids} response type(s)"
                )
                mapping_label = (
                    f"{len(point_missing)} response(s) missing"
                    if point_missing
                    else (
                        f"{len(point_unusable)} response file(s) not found"
                        if point_unusable
                        else "response files ready"
                    )
                )
                point_text = f"Point CSV ready — {count_label}; {mapping_label}."
                point_tab = f"Point features · {point_count or point_ids}"

            if not line_selected:
                line_text = "No line CSV selected."
                line_tab = "Line features"
            elif not line_current:
                line_text = "Line CSV needs refresh."
                line_tab = "Line features · refresh"
            else:
                count_label = (
                    f"{line_count} path(s), {segment_count} segment(s)"
                    if line_count or segment_count
                    else f"{line_ids} response type(s)"
                )
                mapping_label = (
                    f"{len(line_missing)} response(s) missing"
                    if line_missing
                    else (
                        f"{len(line_unusable)} response file(s) not found"
                        if line_unusable
                        else "response files ready"
                    )
                )
                line_text = f"Line CSV ready — {count_label}; {mapping_label}."
                line_tab = f"Line features · {line_count or line_ids}"

            self.point_csv_summary.setText(point_text)
            self.line_csv_summary.setText(line_text)
            self.feature_tabs.setTabText(0, point_tab)
            self.feature_tabs.setTabText(1, line_tab)
            unit_text = self.coordinate_units.currentText()
            self.shared_units_label.setText(
                f"Shared units: {unit_text}. One point CSV may contain every point "
                "family; one line CSV may contain every ordered line chain."
            )

            selected_parts = []
            if point_selected:
                selected_parts.append(
                    f"{enabled_point_count}/{point_count or '?'} point placement(s) enabled"
                )
            if line_selected:
                selected_parts.append(
                    f"{enabled_line_count}/{line_count or '?'} line path(s) enabled / "
                    f"{segment_count or '?'} parsed segment(s)"
                )
            self.feature_summary_label.setText(
                "Selected: " + ("; ".join(selected_parts) if selected_parts else "none yet")
            )
            self.feature_summary_label.setVisible(point_selected and line_selected)
            self.build_summary_label.setText(
                "; ".join(selected_parts)
                if selected_parts
                else "Choose a point or line placement CSV."
            )

            has_body = bool(values.base_grim)
            body_ready = existing_grim_file(values.base_grim)
            has_placements = point_selected or line_selected
            has_enabled_features = bool(
                self.model.enabled_point_placement_ids
                or self.model.enabled_line_ids
                or (
                    not self.model.point_instances
                    and not self.model.line_instances
                    and has_placements
                )
            )
            placement_files_ready = (
                (not point_selected or existing_file(values.point_locations_csv))
                and (not line_selected or existing_file(values.line_locations_csv))
            )
            scans_current = (
                placement_files_ready
                and (not point_selected or point_current)
                and (not line_selected or line_current)
            )
            mappings_complete = not point_missing and not line_missing
            response_files_ready = (
                mappings_complete and not point_unusable and not line_unusable
            )
            surface_selected = bool(values.surface_mesh)
            surface_file_ready = existing_surface_file(values.surface_mesh)
            surface_ready = (
                surface_file_ready
                if self.shadow.isChecked()
                else not surface_selected or surface_file_ready
            )
            has_output = bool(values.output_grim)
            output_ready = has_output
            if has_output:
                try:
                    self.model._validate_output_target()
                except (OSError, ValueError):
                    output_ready = False
            bias_text = self.shadow_bias.text().strip()
            settings_ready = True
            if bias_text:
                try:
                    bias_value = float(bias_text)
                    settings_ready = math.isfinite(bias_value) and bias_value >= 0.0
                except ValueError:
                    settings_ready = False
            self.build_summary_label.setVisible(
                has_placements and (point_current or line_current)
            )
            full_ready = all(
                (
                    service_ready,
                    body_ready,
                    has_placements,
                    has_enabled_features,
                    scans_current,
                    response_files_ready,
                    output_ready,
                    surface_ready,
                    settings_ready,
                )
            )

            checks = [
                (service_ready, "GHOST backend"),
                (body_ready, "body file"),
                (has_placements, "placements"),
                (has_enabled_features, "features enabled"),
                (scans_current and has_placements, "CSV read"),
                (
                    scans_current and response_files_ready and has_placements,
                    "response files",
                ),
            ]
            if surface_selected or self.shadow.isChecked():
                checks.append((surface_ready, "surface mesh"))
            if not settings_ready:
                checks.append((False, "advanced settings"))
            checks.append((output_ready, "output"))
            self.readiness_label.setText(
                "   ".join(("✓" if ok else "○") + " " + label for ok, label in checks)
            )

            if not service_ready:
                next_step = (
                    "GHOST feature backend unavailable; repair the integration "
                    "to continue."
                )
            elif not has_body:
                next_step = "Next: choose the clean-body .grim response."
            elif not body_ready:
                next_step = "Next: choose an existing clean-body .grim file."
            elif not has_placements:
                next_step = "Next: choose a point or line placement CSV."
            elif not scans_current:
                next_step = "Next: refresh the selected CSV and correct any format error."
            elif not has_enabled_features:
                next_step = (
                    "Next: enable at least one item in Spatial Feature Configuration."
                )
            elif not mappings_complete:
                next_step = "Next: map every dataset_id to its OPN − FRD response."
            elif not response_files_ready:
                next_step = "Next: map each response to an existing .grim file."
            elif not surface_ready:
                next_step = "Next: choose an existing .stl or .facet surface mesh."
            elif not settings_ready:
                next_step = "Next: enter a finite, non-negative shadow ray bias or leave it blank."
            elif not has_output:
                next_step = "Next: choose the assembled output file."
            elif not output_ready:
                next_step = "Next: choose an output that does not alias an Assembly input."
            else:
                next_step = (
                    "Ready: validate in 3-D, or assemble directly (validation runs "
                    "automatically)."
                )
            self.next_step_label.setText(next_step)

            busy = self.job_is_running()
            input_preview_supported = bool(
                service_ready
                and adapter is not None
                and callable(adapter.preview_inputs)
            )
            preview_possible = any(
                (
                    body_ready,
                    surface_file_ready,
                    point_selected and existing_file(values.point_locations_csv),
                    line_selected and existing_file(values.line_locations_csv),
                )
            )
            self.scan_button.setEnabled(not busy and service_ready and has_placements)
            self.input_preview_button.setEnabled(
                not busy
                and input_preview_supported
                and preview_possible
            )
            self.preview_button.setEnabled(not busy and full_ready)
            self.build_button.setEnabled(not busy and full_ready)
            self.point_clear_button.setEnabled(not busy and point_selected)
            self.line_clear_button.setEnabled(not busy and line_selected)

        def job_is_running(self) -> bool:
            return bool(self._thread is not None and self._thread.isRunning())

        def is_busy(self) -> bool:
            """Return whether discovery, validation, or assembly is active."""

            return self.job_is_running()

        def can_close(self) -> bool:
            """Closing is safe only after the non-cancellable physics job ends."""

            return not self.is_busy()

        def closeEvent(self, event: Any) -> None:
            if self.is_busy():
                self.status_changed.emit(
                    "Feature validation/assembly is still running; wait before closing."
                )
                event.ignore()
                return
            super().closeEvent(event)

        def _base_path_changed(self) -> None:
            self._mark_preview_stale()
            base = self.base_picker.path()
            if base and not self.output_picker.path():
                source = Path(base)
                suggestion = source.with_name(source.stem + "_features.grim")
                self.output_picker.set_path(str(suggestion))
            self.model.values.base_grim = base
            self._refresh_spatial_feature_tree()
            self._update_workflow_readiness()

        def _input_setting_changed(self, *_args: Any) -> None:
            self._mark_preview_stale()

        def _output_path_changed(self) -> None:
            self.model.invalidate_prepared_plan()
            self._update_workflow_readiness()

        @Slot()
        def _mark_preview_stale(self, *_args: Any) -> None:
            self.model.invalidate_prepared_plan()
            self._update_workflow_readiness()
            if not self._preview_is_current:
                return
            self._preview_is_current = False
            message = (
                "Inputs changed — the 3-D preview is out of date. Preview "
                "inputs again, or validate placements for an authoritative preview."
            )
            self.status_changed.emit(message)
            self.preview_stale.emit(message)

        def _placement_csv_changed(self, kind: str) -> None:
            selected_path = (
                self.point_csv_picker.path()
                if kind == "point"
                else self.line_csv_picker.path()
            )
            if self.model.feature_selection_source_changed(kind, selected_path):
                self.model.clear_feature_selection(kind)
            if kind == "point":
                self.model.values.point_locations_csv = selected_path
            else:
                self.model.values.line_locations_csv = selected_path

            if self.model.requirements_look_current(kind):
                self._update_workflow_readiness()
                return

            self._mark_preview_stale()
            self.model.invalidate_dataset_requirements(kind)
            if kind == "point":
                self.point_mapping.set_dataset_ids(())
            else:
                self.line_mapping.set_dataset_ids(())
            self._refresh_spatial_feature_tree()
            picker = self.point_csv_picker if kind == "point" else self.line_csv_picker
            if picker.path():
                self.refresh_dataset_ids()
            else:
                self.status_changed.emit(
                    f"{kind.capitalize()} placements removed from this build."
                )
                self._update_workflow_readiness()

        def _clear_placement_csv(self, kind: str) -> None:
            picker = self.point_csv_picker if kind == "point" else self.line_csv_picker
            picker.set_path("")
            self.model.clear_feature_selection(kind)
            self._placement_csv_changed(kind)

        def _mapping_changed(self) -> None:
            self.model.values.point_datasets = self.point_mapping.mapping()
            self.model.values.line_datasets = self.line_mapping.mapping()
            self._refresh_spatial_feature_tree()
            self._mark_preview_stale()
            missing = [
                f"point:{dataset_id}" for dataset_id in self.point_mapping.missing_ids()
            ]
            missing.extend(
                f"line:{dataset_id}" for dataset_id in self.line_mapping.missing_ids()
            )
            if missing:
                self.status_changed.emit(
                    "Response mapping incomplete — choose an OPN-FRD .grim for: "
                    + ", ".join(missing)
                )
            elif self.point_mapping.dataset_ids or self.line_mapping.dataset_ids:
                self.status_changed.emit(
                    "All discovered dataset IDs are mapped. Next, validate "
                    "placements and inspect them in the 3-D Assembly view."
                )
            self._update_workflow_readiness()

        def _toggle_schema_help(self, kind: str, checked: bool) -> None:
            if kind == "point":
                button = self.point_format_button
                label = self.point_schema_label
                help_label = self.point_help_label
            else:
                button = self.line_format_button
                label = self.line_schema_label
                help_label = self.line_help_label
            label.setVisible(bool(checked))
            help_label.setVisible(bool(checked))
            button.setText("Hide guide" if checked else "CSV guide")

        def _save_template(self, kind: str) -> None:
            default_name = (
                "point_features_template.csv"
                if kind == "point"
                else "line_features_template.csv"
            )
            path, _ = QFileDialog.getSaveFileName(
                self,
                f"Save blank {kind} placement CSV template",
                default_name,
                "CSV placement file (*.csv);;All files (*)",
            )
            if not path:
                return
            try:
                saved = write_placement_csv_template(kind, path)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self.status_changed.emit(
                f"Saved blank {kind} template: {saved}. Add placement rows, "
                "then choose that CSV above to validate it."
            )

        def _pull_values(self) -> None:
            values = self.model.values
            values.base_grim = self.base_picker.path()
            values.output_grim = self.output_picker.path()
            values.coordinate_units = str(self.coordinate_units.currentData())
            values.surface_mesh = self.surface_picker.path()
            values.surface_units = str(self.surface_units.currentData())
            values.flip_surface_normals = self.flip_normals.isChecked()
            values.shadow = self.shadow.isChecked()
            bias = self.shadow_bias.text().strip()
            try:
                values.shadow_bias_m = None if not bias else float(bias)
            except ValueError as exc:
                raise ValueError("Shadow bias must be a number in meters or blank.") from exc
            values.point_locations_csv = self.point_csv_picker.path()
            values.line_locations_csv = self.line_csv_picker.path()
            values.point_datasets = self.point_mapping.mapping()
            values.line_datasets = self.line_mapping.mapping()
            values.skin_tol_m = self.skin_tol.value()
            values.skin_phase_tol_deg = self.phase_tol.value()
            values.normal_tol_deg = self.normal_tol.value()

        def _show_error(self, text: str) -> None:
            message = str(text).strip() or "Feature assembly failed."
            self.status_changed.emit(message)
            self.build_failed.emit(message)
            self._update_workflow_readiness()

        @Slot()
        def refresh_dataset_ids(self) -> None:
            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                if not (
                    self.model.values.point_locations_csv
                    or self.model.values.line_locations_csv
                ):
                    self.model.update_dataset_requirements(
                        {"point_dataset_ids": (), "line_dataset_ids": ()}
                    )
                    self._apply_requirements_to_tables()
                    self.status_changed.emit("Select a placement CSV to discover IDs.")
                    return
                adapter = coerce_feature_workflow(self._service)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._discovery_paths = (
                self.model.values.point_locations_csv,
                self.model.values.line_locations_csv,
            )
            self._start_operation(
                "discover",
                lambda: self.model.query_dataset_ids(adapter),
            )

        def _apply_requirements_to_tables(self) -> None:
            self.point_mapping.set_dataset_ids(
                self.model.point_dataset_ids, self.model.values.point_datasets
            )
            self.line_mapping.set_dataset_ids(
                self.model.line_dataset_ids, self.model.values.line_datasets
            )
            self._refresh_spatial_feature_tree()

        def _refresh_spatial_feature_tree(self) -> None:
            self.spatial_feature_tree.set_configuration(self.model)
            self.point_mapping.set_required_dataset_ids(
                self.model.active_point_dataset_ids()
            )
            self.line_mapping.set_required_dataset_ids(
                self.model.active_line_dataset_ids()
            )
            self._update_spatial_selection_summary()

        def _update_spatial_selection_summary(self) -> None:
            self.spatial_selection_summary.setText(
                self.model.feature_selection_summary(
                    max_disabled_ids_per_kind=FEATURE_SELECTION_DISPLAY_ID_LIMIT
                )
            )
            self.copy_spatial_selection_button.setEnabled(
                bool(self.model.point_instances or self.model.line_instances)
            )

        @Slot()
        def _copy_full_spatial_selection_summary(self) -> None:
            summary = self.model.feature_selection_summary()
            QApplication.clipboard().setText(summary)
            self.status_changed.emit(
                "Copied the full spatial feature selection summary to the clipboard."
            )

        @Slot()
        def _spatial_selection_changed(self) -> None:
            point_ids, line_ids = self.spatial_feature_tree.excluded_ids()
            try:
                self.model.set_excluded_feature_instances(
                    point_ids=point_ids,
                    line_ids=line_ids,
                )
            except Exception as exc:
                self._show_error(str(exc))
                self._refresh_spatial_feature_tree()
                return
            self._update_spatial_selection_summary()
            self.point_mapping.set_required_dataset_ids(
                self.model.active_point_dataset_ids()
            )
            self.line_mapping.set_required_dataset_ids(
                self.model.active_line_dataset_ids()
            )
            self._mark_preview_stale()
            if not (
                self.model.enabled_point_placement_ids
                or self.model.enabled_line_ids
            ):
                self.status_changed.emit(
                    "No spatial features are enabled. Enable at least one point "
                    "placement or line path before validation or assembly."
                )

        @Slot()
        def preview_inputs(self) -> None:
            """Show geometry/locations without requiring response mappings."""

            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                adapter = coerce_feature_workflow(self._service)
                if not callable(adapter.preview_inputs):
                    raise RuntimeError(
                        "This GHOST backend does not support staged input preview. "
                        "Use Validate placements after mapping responses."
                    )
                if not any(
                    (
                        self.model.values.base_grim,
                        self.model.values.surface_mesh,
                        self.model.values.point_locations_csv,
                        self.model.values.line_locations_csv,
                    )
                ):
                    raise ValueError(
                        "Choose a clean-body GRIM, body mesh, or placement CSV "
                        "to preview."
                    )
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._start_operation(
                "input_preview", lambda: self.model.prepare_input_preview(adapter)
            )

        @Slot()
        def validate_and_preview(self) -> None:
            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                adapter = coerce_feature_workflow(self._service)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._start_operation(
                "preview", lambda: self.model.prepare_preview(adapter)
            )

        @Slot()
        def assemble_and_save(self) -> None:
            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                adapter = coerce_feature_workflow(self._service)
            except Exception as exc:
                self._show_error(str(exc))
                return
            output = _normalized_grim_output_path(
                self.model.values.output_grim,
                base_dir=self.model.values.base_dir,
            )
            if output.exists():
                answer = QMessageBox.question(
                    self,
                    "Replace assembled response?",
                    f"{output.name} already exists. Replace it with this assembly?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    self.status_changed.emit("Assembly cancelled; existing output kept.")
                    return
            self._start_operation("build", lambda: self.model.assemble(adapter))

        def _set_busy(self, busy: bool) -> None:
            self.form_content.setEnabled(not busy)
            if busy:
                self.scan_button.setEnabled(False)
                self.input_preview_button.setEnabled(False)
                self.preview_button.setEnabled(False)
                self.build_button.setEnabled(False)
            else:
                self._update_workflow_readiness()

        def _start_operation(
            self, kind: str, operation: Callable[[], Any]
        ) -> None:
            if self.job_is_running():
                raise RuntimeError("A feature operation is already running.")
            thread = QThread(self)
            worker = _OperationWorker(operation)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.succeeded.connect(self._operation_succeeded)
            worker.failed.connect(self._operation_failed)
            worker.succeeded.connect(thread.quit)
            worker.failed.connect(thread.quit)
            worker.succeeded.connect(worker.deleteLater)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._operation_thread_finished)
            self._thread = thread
            self._worker = worker
            self._active_kind = kind
            self._set_busy(True)
            status = {
                "discover": "Reading placement CSV schemas…",
                "input_preview": (
                    "Loading body geometry and placement locations for visual preview…"
                ),
                "preview": "Validating placements and preparing preview…",
                "build": "Assembling coherent feature responses…",
            }[kind]
            self.status_changed.emit(status)
            thread.start()

        @Slot(object)
        def _operation_succeeded(self, result: Any) -> None:
            kind = self._active_kind
            if kind == "discover":
                current_paths = (
                    self.point_csv_picker.path(),
                    self.line_csv_picker.path(),
                )
                if current_paths != self._discovery_paths:
                    self.model.invalidate_dataset_requirements()
                    self._apply_requirements_to_tables()
                    self.status_changed.emit(
                        "CSV paths changed during discovery; re-scan them."
                    )
                    return
                self.model.update_dataset_requirements(result)
                self._apply_requirements_to_tables()
                point_count = len(self.model.point_dataset_ids)
                line_count = len(self.model.line_dataset_ids)
                missing = self.model.missing_dataset_mappings()
                if missing:
                    self.status_changed.emit(
                        f"✓ CSV schema valid: found {point_count} point and "
                        f"{line_count} line dataset ID(s). Next, choose an "
                        "OPN-FRD .grim response for: " + ", ".join(missing)
                    )
                else:
                    self.status_changed.emit(
                        f"✓ CSV schema valid: found {point_count} point and "
                        f"{line_count} line dataset ID(s); every response is mapped. "
                        "Next, validate placements and preview in 3-D."
                    )
                self._update_workflow_readiness()
            elif kind == "input_preview":
                preview_result = (
                    result.preview
                    if isinstance(result, _VerifiedInputPreview)
                    else result
                )
                requirements = (
                    result.discovery
                    if isinstance(result, _VerifiedInputPreview)
                    else getattr(preview_result, "dataset_requirements", None)
                )
                if requirements is not None:
                    self.model.update_dataset_requirements(requirements)
                    self._apply_requirements_to_tables()
                point_groups = getattr(preview_result, "point_locations_cad_m", {})
                line_groups = getattr(preview_result, "line_paths_cad_m", {})
                try:
                    point_total = sum(len(group) for group in point_groups.values())
                    line_total = sum(len(group) for group in line_groups.values())
                    count_text = (
                        f" ({point_total} point placement(s), "
                        f"{line_total} line path(s))"
                    )
                except (AttributeError, TypeError):
                    count_text = ""
                self._preview_is_current = True
                self.status_changed.emit(
                    "Geometry preview prepared"
                    + count_text
                    + ". Visual QA only: physical placement and response checks "
                    "have not run. Preview Layers → Show only displays or hides "
                    "artists; Spatial Feature Configuration → Use controls "
                    "preview, validation, response loading, and build membership."
                )
                self.preview_ready.emit(preview_result)
                self._update_workflow_readiness()
            elif kind == "preview":
                self._preview_is_current = True
                skin_limit = getattr(result, "skin_limit_m", None)
                skin_text = (
                    f" Effective skin limit: {float(skin_limit) * 1e3:.3f} mm."
                    if skin_limit is not None
                    else ""
                )
                self.status_changed.emit(
                    "Enabled placements validated and every enabled dataset ID is mapped. "
                    "Showing the body and feature groups in the 3-D Assembly "
                    "view."
                    + skin_text
                    + " Build will reuse this validation while inputs remain unchanged."
                )
                self.preview_ready.emit(result)
                self._update_workflow_readiness()
            elif kind == "build":
                dispatch = result
                self._preview_is_current = True
                self.preview_ready.emit(dispatch.plan)
                self.feature_built.emit(str(dispatch.output_path))
                reuse_text = (
                    " Reused the unchanged validated preview."
                    if dispatch.reused_validated_plan
                    else ""
                )
                self.status_changed.emit(
                    f"Saved assembled response: {dispatch.output_path}."
                    + reuse_text
                    + " The result is ready in GRIM for plotting or further "
                    "dataset assembly."
                )
                self._update_workflow_readiness()

        @Slot(str)
        def _operation_failed(self, text: str) -> None:
            if self._active_kind == "discover":
                changed = False
                for kind, picker in (
                    ("point", self.point_csv_picker),
                    ("line", self.line_csv_picker),
                ):
                    if picker.path() and not self.model.requirements_look_current(kind):
                        self.model.invalidate_dataset_requirements(kind)
                        changed = True
                if changed:
                    self._apply_requirements_to_tables()
            elif self._active_kind in {"input_preview", "preview", "build"}:
                # A worker invalidates only the CSV rows whose bytes changed
                # during its operation. Mirror that safe model state into the
                # mapping tables before readiness is recomputed.
                self._apply_requirements_to_tables()
            self._show_error(text)

        @Slot()
        def _operation_thread_finished(self) -> None:
            self._thread = None
            self._worker = None
            self._active_kind = ""
            self._set_busy(False)


else:

    class FeatureAssemblyPanel:  # pragma: no cover - exercised only without Qt
        """Placeholder that preserves an actionable import-time API."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "FeatureAssemblyPanel requires PySide6. "
                f"Original import error: {_GUI_IMPORT_ERROR}"
            )


__all__ = [
    "GUI_AVAILABLE",
    "LINE_PLACEMENT_COLUMNS",
    "LINE_PLACEMENT_EXAMPLE",
    "POINT_PLACEMENT_COLUMNS",
    "POINT_PLACEMENT_EXAMPLE",
    "UNIT_CHOICES",
    "FeatureAssemblyFormModel",
    "FeatureAssemblyPanel",
    "FeatureAssemblyValues",
    "FeatureBuildDispatch",
    "FeatureWorkflowAdapter",
    "LoadedDatasetEntry",
    "coerce_feature_workflow",
    "placement_csv_template_text",
    "write_placement_csv_template",
]
