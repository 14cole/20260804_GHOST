"""Scalar conventions, compatibility checks, and derived-grid metadata."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import copy
import json
import re

import numpy as np

from GRIM_Backend.datasets.constants import (
    _ACQUISITION_METADATA_FAMILIES,
    _ANGLE_UNITS,
    _FREQUENCY_UNITS,
)


ADVISORY_METADATA_KEYS = frozenset({
    "amplitude_version", "phase_reference", "time_convention",
    "polarization_basis", "amplitude_convention", "complex_field_domain",
})


def canonical_time_convention(value) -> str:
    compact = (
        str(value or "").strip().casefold()
        .replace("ω", "omega").replace("*", "").replace(" ", "")
    )
    if re.search(r"exp\(\+?j(?:omega|w)t\)", compact):
        return "+jwt"
    if re.search(r"exp\(-j(?:omega|w)t\)", compact):
        return "-jwt"
    return compact


@dataclass(frozen=True)
class ScalarMetadata:
    key: str
    declarations: tuple[str, ...]
    sources: tuple[str, ...]
    malformed_sources: tuple[str, ...]
    conflicting: bool

    @property
    def status(self) -> str:
        if self.malformed_sources:
            return "malformed"
        if self.conflicting:
            return "conflicting"
        return "consistent" if self.declarations else "missing"

    def scalar(self, *, advisory: bool) -> str:
        """Return one scalar declaration.

        For advisory keys, conflicting values return an empty string. For other
        keys, malformed or conflicting declarations raise ValueError.
        """
        if self.malformed_sources and not advisory:
            raise ValueError(f"metadata {self.key!r} must be scalar")
        if self.conflicting:
            if advisory:
                return ""
            raise ValueError(f"dataset contains contradictory {self.key} metadata")
        return self.declarations[0] if self.declarations else ""


def inspect_scalar_metadata(
    key: str,
    units: Mapping | None,
    extra: Mapping | None,
    *,
    canonicalizer: Callable[[str], str] | None = None,
) -> ScalarMetadata:
    declarations, sources, malformed = [], [], []
    for name, container in (("units", units or {}), ("extra", extra or {})):
        if key not in container:
            continue
        raw = np.asarray(container[key])
        if raw.size != 1:
            malformed.append(name)
            continue
        value = raw.reshape(-1)[0]
        if isinstance(value, np.generic):
            value = value.item()

        text = "" if value is None else str(value).strip()
        if text:
            declarations.append(text)
            sources.append(name)
    normalize = canonicalizer or (
        canonical_time_convention if key == "time_convention"
        else lambda value: " ".join(value.split()).casefold()
    )
    return ScalarMetadata(
        key, tuple(declarations), tuple(sources), tuple(malformed),
        len({normalize(value) for value in declarations}) > 1,
    )


class GridMetadataMixin:
    """Scalar conventions, compatibility checks, and derived-grid metadata."""

    def inspect_scalar_metadata(self, key):
        """Expose metadata evidence without changing numerical eligibility."""
        return inspect_scalar_metadata(
            key, self.units, self.extra,
            canonicalizer=(
                self._canonical_time_convention if key == "time_convention" else None
            ),
        )

    def _declared_scalar_metadata(self, key):
        return self.inspect_scalar_metadata(key).scalar(
            advisory=key in ADVISORY_METADATA_KEYS
        )

    _canonical_time_convention = staticmethod(canonical_time_convention)

    def linear_quantity(self):
        """Physical meaning of ``rcs_power`` (sigma_2d, sigma_3d, or ratio)."""
        raw = str((self.units or {}).get("rcs_linear_quantity", "")).strip().lower()
        if raw:
            return raw
        return "sigma_2d" if self.default_log_unit().lower() == "dbke" else "sigma_3d"

    def _phase_reference(self):
        return self._declared_scalar_metadata("phase_reference")

    def _assert_axis_metadata_compatible(self, other):
        """Require coordinates to share units and the same angular frame.

        This is the compatibility contract for operations that only align or
        crop coordinates and keep each dataset's response values separate.
        Response quantity and logarithmic display metadata are intentionally
        irrelevant to those operations.
        """
        from GRIM_Backend.datasets.grid import RcsGrid

        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        for key, aliases, default in (
            ("azimuth", _ANGLE_UNITS, "deg"),
            ("elevation", _ANGLE_UNITS, "deg"),
            ("frequency", _FREQUENCY_UNITS, "GHz"),
        ):
            left = self._supported_unit(key, aliases, default)
            right = other._supported_unit(key, aliases, default)
            if left != right:
                raise ValueError(f"{key} unit mismatch: {left} != {right}")
        left_angles = self.angular_coordinate_system()
        right_angles = other.angular_coordinate_system()
        if left_angles != right_angles:
            raise ValueError(
                "angular coordinate system mismatch: "
                f"{left_angles} != {right_angles}"
            )
        if left_angles == "great_circle":
            left_convention = self.great_circle_coordinate_convention()
            right_convention = other.great_circle_coordinate_convention()
            if left_convention != right_convention:
                raise ValueError(
                    "great-circle coordinate convention mismatch: "
                    f"{left_convention} != {right_convention}"
                )
            left_orientation = self.angular_frame_orientation_deg()
            right_orientation = other.angular_frame_orientation_deg()
            if not np.allclose(
                left_orientation, right_orientation, rtol=0.0, atol=1.0e-7
            ):
                raise ValueError(
                    "great-circle frame orientation mismatch: "
                    f"roll/tilt {left_orientation} != {right_orientation} deg"
                )

    def _assert_physical_metadata_compatible(self, other):
        self._assert_axis_metadata_compatible(other)
        if self.linear_quantity() != other.linear_quantity():
            raise ValueError(
                "RCS linear quantity mismatch: "
                f"{self.linear_quantity()} != {other.linear_quantity()}"
            )
        if self.default_log_unit().lower() != other.default_log_unit().lower():
            raise ValueError(
                f"RCS log unit mismatch: {self.default_log_unit()} != "
                f"{other.default_log_unit()}"
            )

    @staticmethod
    def _metadata_placeholder(value):
        words = set(
            re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split()
        )
        return bool(
            words.intersection(
                {"unknown", "unspecified", "undetermined", "unverified", "arbitrary"}
            )
        )

    def _coherent_source_convention_values(self, key):
        """Return explicit source declarations without promoting one-sided data."""

        values = []
        direct = self._declared_scalar_metadata(key)
        if direct and not self._metadata_placeholder(direct):
            values.append(direct)
        raw = (self.extra or {}).get("coherent_source_conventions_json")
        if raw is not None:
            try:
                if isinstance(raw, np.ndarray):
                    raw = raw.reshape(()).item()
                record = json.loads(str(raw))
                stored = record.get("declared_values", {}).get(key, [])
                if isinstance(stored, list):
                    values.extend(
                        str(value).strip()
                        for value in stored
                        if str(value).strip()
                        and not self._metadata_placeholder(value)
                    )
            except (TypeError, ValueError, json.JSONDecodeError, AttributeError):


                pass
        return values

    def _assert_coherent_metadata_compatible(
        self, other, *, metadata_attested=False
    ):
        """Return advisory convention differences; field operations stay usable."""

        if not isinstance(metadata_attested, (bool, np.bool_)):
            raise TypeError("metadata_attested must be True or False")
        issues = []
        if self.linear_quantity() == "sigma_2d":
            versions = [grid._declared_scalar_metadata("amplitude_version") for grid in (self, other)]
            if any(versions) and versions != ["2", "2"]:
                issues.append("2-D amplitude_version annotations differ or are unverified; using supplied complex samples.")
        fields = (
            (
                "phase_reference",
                "phase references",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "time_convention",
                "time conventions",
                self._canonical_time_convention,
            ),
            (
                "polarization_basis",
                "polarization bases",
                lambda value: " ".join(value.split()).casefold(),
            ),
            ("amplitude_convention", "amplitude conventions", lambda value: " ".join(value.split()).casefold()),
            ("complex_field_domain", "complex field domains", lambda value: " ".join(value.split()).casefold()),
        )
        for key, label, canonicalize in fields:
            left_values = self._coherent_source_convention_values(key)
            right_values = other._coherent_source_convention_values(key)
            normalized = {
                canonicalize(value) for value in (*left_values, *right_values)
            }
            if len(normalized) > 1:
                issues.append(
                    f"coherent operation uses supplied samples despite different {label} ({key}): "
                    f"{left_values or ['<unspecified>']!r} and "
                    f"{right_values or ['<unspecified>']!r}"
                )
        return issues

    def _assert_acquisition_metadata_compatible(
        self,
        other,
        *,
        operation_label,
        left_role,
        right_role,
        schema,
        excluded_families=(),
    ):
        """Compare acquisition declarations using canonical metadata alias families.

        Reject contradictions within either dataset and between the two inputs.
        """
        from GRIM_Backend.datasets.grid import RcsGrid

        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        operation = str(operation_label or "").strip()
        left_name = str(left_role or "").strip()
        right_name = str(right_role or "").strip()
        contract_schema = str(schema or "").strip()
        if not operation or not left_name or not right_name or not contract_schema:
            raise ValueError("acquisition metadata contract labels cannot be blank")
        if left_name == right_name:
            raise ValueError("acquisition metadata contract roles must be distinct")
        excluded = {str(value).strip() for value in tuple(excluded_families)}
        known_families = {
            family for family, _label, _kind, _keys in _ACQUISITION_METADATA_FAMILIES
        }
        unknown_exclusions = sorted(excluded - known_families)
        if unknown_exclusions:
            raise ValueError(
                "unknown acquisition metadata families excluded from contract: "
                + ", ".join(unknown_exclusions)
            )

        def normalized(value: str) -> str:
            return re.sub(
                r"[^a-z0-9]+", " ", str(value or "").strip().casefold()
            ).strip()

        def identity(value: str) -> str:

            return " ".join(str(value or "").strip().casefold().split())

        def canonical(key: str, value: str, kind: str, role: str) -> dict[str, str]:
            semantic = normalized(value)
            words = set(semantic.split())
            if words.intersection(
                {
                    "unknown",
                    "unspecified",
                    "undetermined",
                    "unverified",
                    "arbitrary",
                }
            ):


                return {}

            if kind == "identity":
                return {"identity": identity(value)}
            if kind == "text":
                return {"value": semantic}
            if kind == "range_phase":
                compact = str(value).casefold()
                compact = compact.replace("−", "-").replace("–", "-")
                compact = re.sub(r"[\s*·^{}()\[\]_=~]+", "", compact)
                matches = re.findall(
                    r"(?:exp|e)([+-])j2(?:\.0+)?kr", compact
                )
                if "negativetwowayrangephase" in compact:
                    matches.append("-")
                if "positivetwowayrangephase" in compact:
                    matches.append("+")
                signs = {"negative" if match == "-" else "positive" for match in matches}
                if len(signs) > 1:
                    raise ValueError(
                        f"{role} contains a contradictory two-way range-phase "
                        f"declaration: {key}={value!r}"
                    )
                if signs:
                    return {"two_way_sign": next(iter(signs))}
                if key == "range_phase_convention":
                    return {}


                return {}
            if kind == "geometry":
                dimensions: dict[str, str] = {}
                topologies = set()
                if "multistatic" in words:
                    topologies.add("multistatic")
                if "bistatic" in words:
                    topologies.add("bistatic")
                if "quasi monostatic" in semantic or "quasimonostatic" in words:
                    topologies.add("quasi_monostatic")
                elif "not monostatic" in semantic:
                    topologies.add("non_monostatic")
                elif "monostatic" in words:
                    topologies.add("monostatic")
                if len(topologies) > 1:
                    raise ValueError(
                        f"{role} contains a contradictory measurement geometry "
                        f"declaration: {key}={value!r}"
                    )
                if topologies:
                    dimensions["scattering_configuration"] = next(iter(topologies))

                far_field = bool(
                    "far field" in semantic
                    or "farfield" in words
                    or "far zone" in semantic
                    or "farzone" in words
                    or "fraunhofer" in words
                    or "plane wave" in semantic
                    or "radiation zone" in semantic
                    or semantic in {"far", "ff"}
                )
                near_field = bool(
                    "near field" in semantic
                    or "nearfield" in words
                    or "near zone" in semantic
                    or "nearzone" in words
                    or "fresnel" in words
                    or "reactive near" in semantic
                    or semantic in {"near", "nf"}
                )
                if far_field and near_field:
                    raise ValueError(
                        f"{role} contains a contradictory measurement geometry "
                        f"declaration: {key}={value!r}"
                    )
                if far_field or near_field:
                    dimensions["propagation_regime"] = (
                        "far_field" if far_field else "near_field"
                    )
                return dimensions

            if kind == "motion":
                positive_state_key = key != "phase_center_motion"
                if semantic in {"1", "true", "yes"}:
                    return {"state": "stable" if positive_state_key else "unsafe"}
                if semantic in {"0", "false", "no", "none", "n a", "na"}:
                    return {"state": "unsafe" if positive_state_key else "stable"}
                safe = bool(
                    "no motion" in semantic
                    or "without motion" in semantic
                    or "no drift" in semantic
                    or words.intersection(
                        {
                            "compensated",
                            "stable",
                            "static",
                            "fixed",
                            "aligned",
                            "corrected",
                        }
                    )
                )
                unsafe = bool(
                    words.intersection(
                        {
                            "uncompensated",
                            "unstable",
                            "moving",
                            "varying",
                            "variable",
                            "misaligned",
                        }
                    )
                    or (
                        any(word.startswith("drift") for word in words)
                        and "no drift" not in semantic
                        and "without drift" not in semantic
                    )
                    or "not compensated" in semantic
                    or "not stable" in semantic
                    or "not static" in semantic
                    or "not fixed" in semantic
                    or "not aligned" in semantic
                    or "motion present" in semantic
                )
                if safe and unsafe:
                    raise ValueError(
                        f"{role} contains a contradictory motion-state declaration: "
                        f"{key}={value!r}"
                    )
                if safe or unsafe:
                    return {"state": "stable" if safe else "unsafe"}
                return {}

            if kind == "setup_state":
                if semantic in {"1", "true", "yes", "same", "unchanged"}:
                    return {"state": "stable"}
                if semantic in {"0", "false", "no", "different", "changed"}:
                    return {"state": "unsafe"}
                safe = bool(words.intersection({"static", "fixed", "unchanged"}))
                unsafe = bool(
                    words.intersection({"changed", "different", "reconfigured"})
                    or "not static" in semantic
                    or "not fixed" in semantic
                )
                if safe and unsafe:
                    raise ValueError(
                        f"{role} contains a contradictory static-setup declaration: "
                        f"{key}={value!r}"
                    )
                if safe or unsafe:
                    return {"state": "stable" if safe else "unsafe"}
                return {}
            raise RuntimeError(f"unsupported acquisition metadata kind {kind!r}")

        def collect(grid, family, label, kind, keys, role):
            raw_by_key: dict[str, str] = {}
            canonical_by_key: dict[str, dict[str, str]] = {}
            dimensions: dict[str, str] = {}
            dimension_sources: dict[str, str] = {}
            for key in keys:
                raw = grid._declared_scalar_metadata(key)
                if not raw:
                    continue
                values = canonical(key, raw, kind, role)
                raw_by_key[key] = raw
                canonical_by_key[key] = values
                for dimension, value in values.items():
                    prior = dimensions.get(dimension)
                    if prior is not None and prior != value:
                        prior_key = dimension_sources[dimension]
                        raise ValueError(
                            f"{role} contains contradictory {label} declarations: "
                            f"{prior_key}={raw_by_key[prior_key]!r} conflicts with "
                            f"{key}={raw!r}"
                        )
                    dimensions[dimension] = value
                    dimension_sources[dimension] = key
            if kind == "motion" and dimensions.get("state") == "unsafe":
                raise ValueError(
                    f"{operation} requires stable/aligned acquisitions; "
                    f"{role} declares {label}: {raw_by_key!r}"
                )
            if kind == "setup_state" and dimensions.get("state") == "unsafe":
                raise ValueError(
                    f"{operation} requires an unchanged static setup; "
                    f"{role} declares {label}: {raw_by_key!r}"
                )
            return {
                "declared_by_key": raw_by_key,
                "canonical_by_key": canonical_by_key,
                "canonical_dimensions": dimensions,
            }

        matching: dict[str, str] = {}
        missing: dict[str, list[str]] = {}
        semantic_families = {}
        for family, label, kind, keys in _ACQUISITION_METADATA_FAMILIES:
            if family in excluded:
                continue
            left = collect(self, family, label, kind, keys, left_name)
            right = collect(other, family, label, kind, keys, right_name)
            left_dimensions = left["canonical_dimensions"]
            right_dimensions = right["canonical_dimensions"]
            for dimension in sorted(set(left_dimensions).intersection(right_dimensions)):
                if left_dimensions[dimension] != right_dimensions[dimension]:
                    raise ValueError(
                        f"{operation} requires matching explicit {label}; "
                        f"{left_name} declares {left['declared_by_key']!r}, while "
                        f"{right_name} declares {right['declared_by_key']!r} "
                        f"({dimension} mismatch)"
                    )

            all_dimensions_set = set(left_dimensions).union(right_dimensions)
            if kind == "range_phase":


                all_dimensions_set.add("two_way_sign")
            all_dimensions = sorted(all_dimensions_set)
            if not all_dimensions:
                missing[family] = [left_name, right_name]
            else:
                for dimension in all_dimensions:
                    absent = []
                    if dimension not in left_dimensions:
                        absent.append(left_name)
                    if dimension not in right_dimensions:
                        absent.append(right_name)
                    if absent:
                        missing[f"{family}.{dimension}"] = absent


            for key, raw in left["declared_by_key"].items():
                values = left["canonical_by_key"][key]
                if values and all(
                    right_dimensions.get(dimension) == value
                    for dimension, value in values.items()
                ):
                    matching[key] = raw

            semantic_families[family] = {
                "label": label,
                "aliases": list(keys),
                "declarations_by_role": {
                    left_name: left["declared_by_key"],
                    right_name: right["declared_by_key"],
                },
                "canonical_dimensions_by_role": {
                    left_name: left_dimensions,
                    right_name: right_dimensions,
                },
            }

        return {
            "schema": contract_schema,
            "checked_fields": [
                key
                for family, _label, _kind, keys in _ACQUISITION_METADATA_FAMILIES
                if family not in excluded
                for key in keys
            ],
            "excluded_families": sorted(excluded),
            "semantic_families": semantic_families,
            "matching_explicit_declarations": matching,
            "missing_declarations_by_role": missing,
            "missing_declarations_covered_by_operation_assumption": bool(missing),
            "missing_declarations_covered_by_user_attestation": False,
            "explicit_contradictions_allowed": False,
        }

    def _assert_support_reference_metadata_compatible(self, other):
        """Reject explicit acquisition/setup contradictions for support subtraction.

        Missing declarations are recorded as assumptions. Explicit declarations
        are compared across their semantic alias families and can never be
        waived by an operation choice.
        """

        return self._assert_acquisition_metadata_compatible(
            other,
            operation_label="support-referenced subtraction",
            left_role="target_plus_support",
            right_role="support_only_reference",
            schema="grim.support-reference-metadata-contract.v3",
        )

    def _coherent_attestation_provenance(
        self,
        others,
        *,
        operation,
        metadata_attested,
    ):
        """Return metadata describing coherent-operation assumptions.

        Missing convention values remain unspecified. ``metadata_attested=True``
        records an explicit attestation.
        """

        if not isinstance(metadata_attested, (bool, np.bool_)):
            raise TypeError("metadata_attested must be True or False")
        inputs = (self, *tuple(others))
        fields = (
            "phase_reference",
            "time_convention",
            "polarization_basis",
            "amplitude_version",
            "amplitude_convention",
            "complex_field_domain",
        )
        advisories = [issue for grid in inputs[1:] for issue in self._assert_coherent_metadata_compatible(grid)]
        missing = {}
        for key in fields:
            missing_indices = [
                index
                for index, grid in enumerate(inputs, start=1)
                if (
                    not grid._declared_scalar_metadata(key)
                    or grid._metadata_placeholder(
                        grid._declared_scalar_metadata(key)
                    )
                )
            ]
            if missing_indices:
                missing[key] = missing_indices

        if not missing and not metadata_attested and not advisories:
            return None, None

        operation_name = str(operation).strip().lower().replace("_", "-")
        user_attested = bool(metadata_attested)
        record = {
            "metadata_policy": "advisory; no phase or amplitude conversion applied",
            "advisories": advisories,
            "schema": (
                "grim.coherent-metadata-attestation.v1"
                if user_attested
                else "grim.coherent-metadata-assumption.v1"
            ),
            "operation": operation_name,
            "input_count": len(inputs),
            "user_attested": user_attested,
            "operation_requested_with_unspecified_metadata": bool(missing),
            "assumed_scope": [
                "phase_reference_or_center",
                "phasor_time_convention",
                "polarization_basis",
            ],
            "missing_declarations_by_input": missing,
            "declarations_inferred": False,
        }
        if user_attested:
            history_entry = (
                f"User-attested coherent metadata compatibility ({operation_name}, "
                f"{len(inputs)} inputs): compatible phase reference/center, phasor "
                "time convention, and polarization basis where declarations were "
                "unspecified; no convention values inferred"
            )
        else:
            history_entry = (
                f"Coherent operation used available complex samples ({operation_name}, "
                f"{len(inputs)} inputs): missing convention metadata recorded as "
                "unspecified; no convention values inferred"
            )
        prior_history = str(self.history or "").strip()
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )

        extra = {}
        source_declarations = {}
        for key in fields:
            declared_values = []
            direct_values = []
            for grid in inputs:
                declared_values.extend(
                    grid._coherent_source_convention_values(key)
                )
                direct = grid._declared_scalar_metadata(key)
                if direct and not grid._metadata_placeholder(direct):
                    direct_values.append(direct)
            if key == "time_convention":
                normalized = {
                    self._canonical_time_convention(value)
                    for value in declared_values
                }
            else:
                normalized = {
                    " ".join(value.split()).casefold()
                    for value in declared_values
                }
            if declared_values:
                unique_values = []
                seen = set()
                for value in declared_values:
                    canonical = (
                        self._canonical_time_convention(value)
                        if key == "time_convention"
                        else " ".join(value.split()).casefold()
                    )
                    if canonical not in seen:
                        unique_values.append(value)
                        seen.add(canonical)
                source_declarations[key] = unique_values
            if len(direct_values) == len(inputs) and len(normalized) == 1:


                extra[key] = direct_values[0]
        if source_declarations:
            extra["coherent_source_conventions_json"] = json.dumps(
                {
                    "schema": "grim.coherent-source-conventions.v1",
                    "declared_values": source_declarations,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        record_key = (
            "coherent_metadata_attestation_json"
            if user_attested
            else "coherent_metadata_assumption_json"
        )
        extra[record_key] = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return history, extra

    def _assert_compatible(
        self,
        other,
        *,
        coherent=False,
        coherent_metadata_attested=False,
        _scan_phase_samples=True,
    ):
        """Validate another grid for element-wise operations.

        Use before coherent/incoherent add/subtract operations.

        Args:
            other: Another RcsGrid instance.

        Raises:
            TypeError: if other is not an RcsGrid.
            ValueError: if axes or shapes differ.
        """
        from GRIM_Backend.datasets.grid import RcsGrid
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        if self.rcs_power.shape != other.rcs_power.shape:
            raise ValueError(f"rcs shape {other.rcs_power.shape} != {self.rcs_power.shape}")
        if not np.array_equal(self.azimuths, other.azimuths):
            raise ValueError("azimuth axis mismatch")
        if not np.array_equal(self.elevations, other.elevations):
            raise ValueError("elevation axis mismatch")
        if not np.array_equal(self.frequencies, other.frequencies):
            raise ValueError("frequency axis mismatch")
        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch")
        self._assert_physical_metadata_compatible(other)
        if coherent:
            if not isinstance(_scan_phase_samples, (bool, np.bool_)):
                raise TypeError("_scan_phase_samples must be True or False")
            if _scan_phase_samples:
                for label, grid in (("left", self), ("right", other)):
                    missing = np.isfinite(grid.rcs_power) & ~np.isfinite(grid.rcs_phase)
                    if np.any(missing):
                        raise ValueError(
                            f"coherent operation requires phase; {label} grid has "
                            f"{int(np.count_nonzero(missing))} finite-power sample(s) "
                            "with unknown phase"
                        )
            self._assert_coherent_metadata_compatible(
                other, metadata_attested=coherent_metadata_attested
            )

    _FIELD_CONVENTION_EXTRA_KEYS = frozenset({
        "amplitude_version",
        "phase_reference",
        "time_convention",
        "polarization_basis",
        "amplitude_convention",
        "complex_field_domain",
    })

    _COORDINATE_LINEAGE_EXTRA_KEYS = frozenset({
        "source_format",
        "angular_coordinate_declaration_json",
        "angular_coordinate_system",
        "great_circle_coordinate_convention",
        "elevation_coordinate_convention",
        "ptm_cut_type",
        "ptm_roll",
        "ptm_tilt",
        "assembly_angular_coordinate_contract",
    })

    _ASSEMBLY_LINEAGE_EXTRA_KEYS = frozenset({
        "combine_role",
        "combine_role_note",
        "assembly_response_role",
        "assembly_base_sha256",
        "assembly_source_base_sha256_json",
        "assembly_base_response_sha256",
        "assembly_source_base_response_sha256",
        "assembly_base_response_transform",
        "source_monostatic_sha256",
        "feature_provenance_json",
        "assembly_provenance_json",
        "coherent_metadata_attestation_json",
        "coherent_metadata_assumption_json",
        "coherent_source_conventions_json",
    })

    _DERIVED_PROVENANCE_EXTRA_KEYS = frozenset({


        "support_reference_difference_json",
        "coherent_sample_qa_json",
        "coherent_metadata_assumption_json",
        "merge_metadata_assumption_json",
        "coherent_source_conventions_json",
        "elevation_pair_to_azimuth_json",
        "decimation_json",
        "statistics_reduction_json",
    })

    _RAW_AMPLITUDE_EXTRA_KEYS = ("rcs_amp_real", "rcs_amp_imag")

    def _safe_derived_scalar_extra(self, *, include_field_conventions=True):
        """Copy only scalar metadata with durable derived-grid semantics.

        Every ``sentri_*`` scalar is retained conservatively.  That prefix is
        vendor provenance as well as UI information, and losing it could make
        a native polar-theta table look like canonical signed elevation after
        an otherwise unrelated dataset operation.
        """

        allowed = set(self._COORDINATE_LINEAGE_EXTRA_KEYS)
        allowed.update(self._ASSEMBLY_LINEAGE_EXTRA_KEYS)
        allowed.update(self._DERIVED_PROVENANCE_EXTRA_KEYS)
        if include_field_conventions:
            allowed.update(self._FIELD_CONVENTION_EXTRA_KEYS)
        result = {}
        for key, value in self.extra.items():
            if key not in allowed and not str(key).startswith("sentri_"):
                continue
            array = np.asarray(value)
            if array.size != 1:
                continue
            result[key] = copy.deepcopy(value)
        return result

    def _has_native_sentri_coordinate_hazard(self):
        """Return whether this grid still uses vendor polar theta as elevation."""

        convention_values = []
        for container in (self.units or {}, self.extra or {}):
            value = container.get("elevation_coordinate_convention")
            if value is not None:
                convention_values.append(str(value).strip().casefold())
            value = container.get("sentri_elevation_convention")
            if value is not None:
                convention_values.append(str(value).strip().casefold())
        if "sentri_theta_top_zero" in convention_values:
            return True
        mapping = str(
            (self.extra or {}).get("sentri_coordinate_mapping", "") or ""
        ).casefold().replace(" ", "")
        return "elevation=theta" in mapping and "elevation=90-theta" not in mapping

    @classmethod
    def _carry_native_sentri_hazard(cls, extra, sources):
        """Make a native-SENTRi source impossible to hide in a derived grid."""

        if not any(grid._has_native_sentri_coordinate_hazard() for grid in sources):
            return extra


        extra.pop("assembly_angular_coordinate_contract", None)
        extra["source_format"] = "derived response includes native SENTRi coordinates"
        extra["sentri_elevation_convention"] = "sentri_theta_top_zero"
        extra["sentri_coordinate_mapping"] = (
            "elevation=theta; native SENTRi polar coordinates retained"
        )
        return extra

    @staticmethod
    def _invalidate_assembly_sampling_hash(extra, operation):
        """Retain source lineage while invalidating a sampling-bound base hash."""

        digest = extra.pop("assembly_base_response_sha256", None)
        has_assembly_lineage = digest is not None or any(
            key in extra
            for key in (
                "assembly_response_role",
                "assembly_base_sha256",
                "source_monostatic_sha256",
                "feature_provenance_json",
                "assembly_provenance_json",
            )
        )
        if digest is not None and str(np.asarray(digest).reshape(-1)[0]).strip():
            prior = extra.get("assembly_source_base_response_sha256")
            if prior is None:
                extra["assembly_source_base_response_sha256"] = copy.deepcopy(digest)
            elif str(np.asarray(prior).reshape(-1)[0]).strip().casefold() != str(
                np.asarray(digest).reshape(-1)[0]
            ).strip().casefold():


                extra["assembly_source_base_response_sha256"] = json.dumps(
                    sorted({
                        str(np.asarray(prior).reshape(-1)[0]).strip().lower(),
                        str(np.asarray(digest).reshape(-1)[0]).strip().lower(),
                    }),
                    separators=(",", ":"),
                )
        if has_assembly_lineage:
            extra["assembly_base_response_transform"] = str(operation)
        return extra

    def _exact_transform_extra(
        self,
        array_transform=None,
        *,
        preserve_all=False,
        preserve_raw=True,
        coordinate_change=None,
        preserve_angular_contract=True,
    ):
        """Metadata policy for an exact sample-preserving transform.

        ``array_transform`` is applied only to the authoritative raw solver
        field.  Other grid-shaped producer arrays are deliberately not guessed
        at.  ``preserve_all`` is reserved for representation-only operations
        such as phase wrapping where neither axes nor samples change.
        """

        if preserve_all:
            extra = {
                key: copy.deepcopy(value)
                for key, value in self._extra_to_write().items()
            }
            if self._complete_authoritative_raw_arrays() is None:
                for key in (
                    *self._RAW_AMPLITUDE_EXTRA_KEYS,
                    "raw_complex_amplitude_preserved",
                ):
                    extra.pop(key, None)
        else:
            extra = self._safe_derived_scalar_extra(
                include_field_conventions=True
            )
            pair = self._complete_authoritative_raw_arrays()
            if preserve_raw and pair is not None:
                real_array, imag_array = pair
                transform = array_transform or (
                    lambda value: np.array(value, copy=True)
                )
                transformed_real = np.asarray(transform(real_array))
                transformed_imag = np.asarray(transform(imag_array))
                if np.shares_memory(transformed_real, real_array):
                    transformed_real = np.array(transformed_real, copy=True)
                if np.shares_memory(transformed_imag, imag_array):
                    transformed_imag = np.array(transformed_imag, copy=True)
                extra["rcs_amp_real"] = transformed_real
                extra["rcs_amp_imag"] = transformed_imag
                extra["raw_complex_amplitude_preserved"] = True
        if not preserve_all:
            self._carry_native_sentri_hazard(extra, (self,))
        if coordinate_change:
            self._invalidate_assembly_sampling_hash(extra, coordinate_change)
        if not preserve_angular_contract:
            extra.pop("assembly_angular_coordinate_contract", None)
        return extra

    def _derived_response_extra(
        self,
        others=(),
        *,
        operation,
        coherent,
        attestation_extra=None,
    ):
        """Metadata policy for arithmetic/interpolated/statistical responses."""

        sources = (self, *tuple(others))
        extra = self._safe_derived_scalar_extra(
            include_field_conventions=bool(coherent)
        )
        self._carry_native_sentri_hazard(extra, sources)

        def declared_values(key):
            values = []
            for grid in sources:
                value = grid._declared_scalar_metadata(key)
                if value:
                    values.append(value)
            return values

        source_roles = [
            grid._declared_scalar_metadata(
                "assembly_response_role"
            ).strip().casefold()
            or None
            for grid in sources
        ]
        roles = [role for role in source_roles if role is not None]
        base_hashes = {
            value.strip().casefold()
            for value in declared_values("assembly_base_sha256")
        }
        response_hashes = {
            value.strip().casefold()
            for value in declared_values("assembly_base_response_sha256")
        }
        if len(base_hashes) == 1:
            extra["assembly_base_sha256"] = next(iter(base_hashes))
        elif len(base_hashes) > 1:
            extra.pop("assembly_base_sha256", None)
            extra["assembly_source_base_sha256_json"] = json.dumps(
                sorted(base_hashes), separators=(",", ":")
            )
        if len(response_hashes) == 1:
            extra["assembly_base_response_sha256"] = next(iter(response_hashes))
        elif len(response_hashes) > 1:
            extra.pop("assembly_base_response_sha256", None)
            extra["assembly_source_base_response_sha256"] = json.dumps(
                sorted(response_hashes), separators=(",", ":")
            )

        if coherent:


            for key in (
                "phase_reference",
                "time_convention",
                "polarization_basis",
                "amplitude_version",
                "amplitude_convention",
                "complex_field_domain",
            ):
                values = [
                    grid._declared_scalar_metadata(key) for grid in sources
                ]
                if any(not value for value in values):
                    extra.pop(key, None)
                    continue
                if key == "time_convention":
                    normalized = {
                        self._canonical_time_convention(value)
                        for value in values
                    }
                else:
                    normalized = {
                        " ".join(value.split()).casefold() for value in values
                    }
                if len(normalized) == 1:
                    extra[key] = values[0]
                else:


                    extra.pop(key, None)
            if "body_plus_features" in roles:
                extra["assembly_response_role"] = "body_plus_features"
            elif source_roles and all(
                role == "features_only_delta" for role in source_roles
            ) and len(base_hashes) == 1 and len(response_hashes) == 1:
                extra["assembly_response_role"] = "features_only_delta"
            elif roles:
                extra["assembly_response_role"] = "coherent_field_sum"
            if roles or any(
                grid._declared_scalar_metadata("combine_role") for grid in sources
            ):
                extra["combine_role"] = "coherent"
        else:


            if "body_plus_features" in roles:
                extra["assembly_response_role"] = "body_plus_features"
            elif roles:
                extra["assembly_response_role"] = "incoherent_power_sum"
            extra["combine_role"] = "power"
            for key in (
                "phase_reference",
                "time_convention",
                "amplitude_convention",
                "complex_field_domain",
            ):
                extra.pop(key, None)
            self._invalidate_assembly_sampling_hash(extra, operation)

        if attestation_extra:
            extra.update(copy.deepcopy(attestation_extra))
        return extra
