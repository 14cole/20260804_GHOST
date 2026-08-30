"""Repeat-acquisition contract for future transient/bird ISAR screening.

``RcsGrid`` intentionally remains a four-dimensional static-scene product.
Repeat identity and time are represented here instead of overloading azimuth as
slow time.  The utilities align only a reviewed global complex gain/phase and
return diagnostic masks; they never delete samples or claim bird removal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
from typing import Iterable

import numpy as np

from grim_dataset import RcsGrid


_REQUIRED_UNIT_KEYS = (
    "azimuth",
    "elevation",
    "frequency",
    "rcs_linear_quantity",
    "angular_coordinate_system",
)

_REQUIRED_COHERENT_FIELDS = (
    "phase_reference",
    "time_convention",
    "polarization_basis",
)

_RANGE_PHASE_METADATA_KEYS = (
    "range_phase_convention",
    "phase_law",
    "amplitude_convention",
    "phase_reference",
)

_OPTIONAL_EXACT_FIELDS = (
    "elevation_coordinate_convention",
    "sentri_elevation_convention",
)

# A calibration run commonly identifies an individual acquisition, so unlike
# a shared calibration chain/version it is not required to be equal across
# repeated sweeps.
_REPEAT_EXCLUDED_ACQUISITION_FAMILIES = ("calibration_run_id",)
_DEFAULT_REPEAT_WORKING_SET_MB = 2048
_REPEAT_STATISTICS_SCRATCH_BYTES = 64 * 1024**2


def _repeat_working_set_limit_bytes(maximum_working_bytes) -> int:
    if maximum_working_bytes is None:
        raw = os.environ.get(
            "GRIM_REPEAT_WORKING_SET_MB",
            str(_DEFAULT_REPEAT_WORKING_SET_MB),
        )
        try:
            limit = int(raw) * 1024**2
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "GRIM_REPEAT_WORKING_SET_MB must be a positive integer"
            ) from exc
    else:
        if isinstance(maximum_working_bytes, (bool, np.bool_)):
            raise TypeError("maximum_working_bytes must be a positive integer")
        try:
            limit = int(maximum_working_bytes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                "maximum_working_bytes must be a positive integer"
            ) from exc
    if limit <= 0:
        raise ValueError("repeat working-set limit must be positive")
    return limit


def _repeat_stack_working_set_bytes(
    repeat_count: int, azimuth_count: int, frequency_count: int
) -> int:
    cells = int(azimuth_count) * int(frequency_count)
    # Retained complex128 stack plus one complex128 source slice.
    return 16 * (int(repeat_count) + 1) * cells


def _repeat_screen_working_set_bytes(
    repeat_count: int, azimuth_count: int, frequency_count: int
) -> int:
    repeats = int(repeat_count)
    cells = int(azimuth_count) * int(frequency_count)
    # Returned arrays: registered complex128, center complex128, residual and
    # z-score float64, and Boolean mask. Add one complex source/delta slice and
    # the explicitly bounded robust-statistics scratch.
    retained = (33 * repeats + 16) * cells
    scratch = max(
        16 * cells,
        min(_REPEAT_STATISTICS_SCRATCH_BYTES, 16 * repeats * cells),
    )
    return retained + scratch


def _validate_repeat_working_set(required: int, limit: int, operation: str) -> None:
    if int(required) <= int(limit):
        return
    raise ValueError(
        f"{operation} needs an estimated {required / 1024**3:.2f} GiB working "
        f"set, above the {limit / 1024**3:.2f} GiB limit. Crop azimuth/frequency, "
        "screen smaller contiguous blocks, or deliberately raise "
        "maximum_working_bytes / GRIM_REPEAT_WORKING_SET_MB on a machine with "
        "verified headroom."
    )


def _utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("acquisition timestamp must be a datetime")
    if value.tzinfo is None:
        raise ValueError("acquisition timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def _repeat_range_phase_profile(grid: RcsGrid) -> dict:
    """Return explicit two-way range-phase sign and its alias evidence."""

    signs = set()
    declarations = {}
    for key in _RANGE_PHASE_METADATA_KEYS:
        declared = grid._declared_scalar_metadata(key)
        if not declared:
            continue
        declarations[key] = declared
        compact = str(declared).casefold()
        compact = compact.replace("−", "-").replace("–", "-")
        compact = re.sub(r"[\s*·^{}()\[\]_=~]+", "", compact)
        matches = re.findall(r"(?:exp|e)([+-])j2(?:\.0+)?kr", compact)
        if "negativetwowayrangephase" in compact:
            matches.append("-")
        if "positivetwowayrangephase" in compact:
            matches.append("+")
        signs.update(-1 if match == "-" else 1 for match in matches)
        if key == "range_phase_convention" and not matches:
            raise ValueError(
                "range_phase_convention is explicit but does not declare "
                "S~exp(-j*2*k*R) or S~exp(+j*2*k*R)"
            )
    if len(signs) > 1:
        raise ValueError("contradictory two-way range-phase declarations")
    return {
        "sign": next(iter(signs), None),
        "declared_by_key": declarations,
    }


@dataclass(frozen=True)
class IsarSweep:
    acquisition_id: str
    timestamp: datetime
    grid: RcsGrid
    notes: str = ""

    def __post_init__(self):
        identifier = str(self.acquisition_id).strip()
        if not identifier:
            raise ValueError("acquisition_id cannot be empty")
        if not isinstance(self.grid, RcsGrid):
            raise TypeError("repeat acquisition grid must be an RcsGrid")
        object.__setattr__(self, "acquisition_id", identifier)
        object.__setattr__(self, "timestamp", _utc_datetime(self.timestamp))
        object.__setattr__(self, "notes", str(self.notes))


@dataclass(frozen=True)
class ComplexRegistration:
    acquisition_id: str
    multiplier_to_reference: complex
    coherence: float
    reviewed_sample_count: int


@dataclass(frozen=True)
class RepeatScreenResult:
    registered_stack: np.ndarray
    registrations: tuple[ComplexRegistration, ...]
    robust_center: np.ndarray
    residual_magnitude: np.ndarray
    robust_z_score: np.ndarray
    candidate_outlier_mask: np.ndarray
    threshold: float


class RepeatAcquisitionStack:
    """Validated repeated static-scene sweeps with explicit time identity."""

    def __init__(
        self,
        sweeps: Iterable[IsarSweep],
        *,
        axis_tolerance: float = 1.0e-9,
        legacy_metadata_attested: bool = False,
    ):
        if not isinstance(legacy_metadata_attested, (bool, np.bool_)):
            raise TypeError("legacy_metadata_attested must be True or False")
        values = tuple(sweeps)
        if len(values) < 2:
            raise ValueError("repeat-acquisition analysis needs at least two sweeps")
        if any(not isinstance(sweep, IsarSweep) for sweep in values):
            raise TypeError("repeat acquisitions must be IsarSweep objects")
        if len({sweep.acquisition_id for sweep in values}) != len(values):
            raise ValueError("repeat acquisition IDs must be unique")
        timestamps = [sweep.timestamp for sweep in values]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("repeat acquisitions must be strictly time ordered")
        tolerance = float(axis_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("axis_tolerance must be finite and nonnegative")
        reference = values[0].grid
        for sweep in values[1:]:
            candidate = sweep.grid
            try:
                reference._assert_physical_metadata_compatible(candidate)
            except ValueError as exc:
                message = str(exc)
                if "RCS linear quantity mismatch" in message:
                    raise ValueError(
                        "repeat acquisitions disagree on response quantity metadata: "
                        + message
                    ) from exc
                for axis_name in ("azimuth", "elevation", "frequency"):
                    if message.startswith(f"{axis_name} unit mismatch"):
                        raise ValueError(
                            f"repeat acquisitions disagree on {axis_name} metadata: "
                            + message
                        ) from exc
                raise ValueError(
                    "repeat acquisitions have incompatible physical metadata: "
                    + message
                ) from exc
            for name in ("azimuths", "elevations", "frequencies"):
                left = np.asarray(getattr(reference, name), dtype=float)
                right = np.asarray(getattr(candidate, name), dtype=float)
                if left.shape != right.shape or not np.allclose(
                    left, right, rtol=0.0, atol=tolerance, equal_nan=False
                ):
                    raise ValueError(
                        f"repeat acquisition {sweep.acquisition_id!r} has a "
                        f"different {name} axis"
                    )
            if not np.array_equal(
                np.asarray(reference.polarizations),
                np.asarray(candidate.polarizations),
            ):
                raise ValueError("repeat acquisitions have different polarization axes")
            if reference.linear_quantity() != candidate.linear_quantity():
                raise ValueError(
                    "repeat acquisitions disagree on response quantity metadata"
                )

        # Validate every pair, not just each sweep against the first.  If the
        # reference is legacy/blank, comparing only to it would let later
        # acquisitions contradict one another while attestation is enabled.
        profiles: dict[str, dict] = {
            sweep.acquisition_id: {
                "declared_units": {},
                "coherent_fields": {},
                "range_phase": {},
                "optional_exact_fields": {},
                "semantic_families": {},
            }
            for sweep in values
        }
        missing_by_acquisition: dict[str, set[str]] = {
            sweep.acquisition_id: set() for sweep in values
        }

        for sweep in values:
            profile = profiles[sweep.acquisition_id]
            for key in _REQUIRED_UNIT_KEYS:
                if key == "angular_coordinate_system":
                    text = sweep.grid._declared_scalar_metadata(key)
                else:
                    raw = (sweep.grid.units or {}).get(key)
                    text = "" if raw is None else str(raw).strip()
                if text:
                    profile["declared_units"][key] = text
                else:
                    missing_by_acquisition[sweep.acquisition_id].add(key)
            for key in _REQUIRED_COHERENT_FIELDS:
                declared = sweep.grid._declared_scalar_metadata(key)
                if declared:
                    normalized = " ".join(declared.casefold().split())
                    if set(normalized.split()).intersection(
                        {
                            "unknown",
                            "unspecified",
                            "undetermined",
                            "unverified",
                            "arbitrary",
                        }
                    ):
                        raise ValueError(
                            f"repeat acquisition {sweep.acquisition_id!r} declares "
                            f"explicitly unverified {key} metadata: {declared!r}"
                        )
                    profile["coherent_fields"][key] = declared
                else:
                    missing_by_acquisition[sweep.acquisition_id].add(key)
            try:
                range_phase = _repeat_range_phase_profile(sweep.grid)
            except ValueError as exc:
                raise ValueError(
                    f"repeat acquisition {sweep.acquisition_id!r} has invalid "
                    f"two-way range-phase metadata: {exc}"
                ) from exc
            profile["range_phase"] = range_phase
            if range_phase["sign"] is None:
                missing_by_acquisition[sweep.acquisition_id].add(
                    "range_phase_convention"
                )
            for key in _OPTIONAL_EXACT_FIELDS:
                declared = sweep.grid._declared_scalar_metadata(key)
                if declared:
                    profile["optional_exact_fields"][key] = declared

        for left_index, left_sweep in enumerate(values[:-1]):
            for right_sweep in values[left_index + 1 :]:
                left_grid = left_sweep.grid
                right_grid = right_sweep.grid
                try:
                    # True here covers only missing coherent declarations; this
                    # method still rejects every explicit contradiction.  The
                    # stack's own legacy gate below decides whether omissions
                    # are permitted and records that decision durably.
                    left_grid._assert_coherent_metadata_compatible(
                        right_grid, metadata_attested=True
                    )
                except ValueError as exc:
                    detail = str(exc)
                    if "phase references" in detail:
                        detail = "phase_reference metadata mismatch: " + detail
                    elif "time conventions" in detail:
                        detail = "time_convention metadata mismatch: " + detail
                    elif "polarization bases" in detail:
                        detail = "polarization_basis metadata mismatch: " + detail
                    raise ValueError(
                        f"repeat acquisitions {left_sweep.acquisition_id!r} and "
                        f"{right_sweep.acquisition_id!r} disagree: {detail}"
                    ) from exc

                left_range_phase = profiles[left_sweep.acquisition_id][
                    "range_phase"
                ]["sign"]
                right_range_phase = profiles[right_sweep.acquisition_id][
                    "range_phase"
                ]["sign"]
                if (
                    left_range_phase is not None
                    and right_range_phase is not None
                    and left_range_phase != right_range_phase
                ):
                    raise ValueError(
                        f"repeat acquisitions {left_sweep.acquisition_id!r} and "
                        f"{right_sweep.acquisition_id!r} disagree on the two-way "
                        "range-phase convention"
                    )

                left_role = f"acquisition:{left_sweep.acquisition_id}"
                right_role = f"acquisition:{right_sweep.acquisition_id}"
                try:
                    acquisition_contract = (
                        left_grid._assert_acquisition_metadata_compatible(
                            right_grid,
                            operation_label="repeat-acquisition analysis",
                            left_role=left_role,
                            right_role=right_role,
                            schema="grim.repeat-acquisition-pair-metadata.v2",
                            excluded_families=(
                                _REPEAT_EXCLUDED_ACQUISITION_FAMILIES
                            ),
                        )
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"repeat acquisitions {left_sweep.acquisition_id!r} and "
                        f"{right_sweep.acquisition_id!r} disagree: {exc}"
                    ) from exc

                for family, family_contract in acquisition_contract[
                    "semantic_families"
                ].items():
                    for sweep, role in (
                        (left_sweep, left_role),
                        (right_sweep, right_role),
                    ):
                        semantic_profile = {
                            "declared_by_key": family_contract[
                                "declarations_by_role"
                            ][role],
                            "canonical_dimensions": family_contract[
                                "canonical_dimensions_by_role"
                            ][role],
                        }
                        prior = profiles[sweep.acquisition_id][
                            "semantic_families"
                        ].get(family)
                        if prior is not None and prior != semantic_profile:
                            raise RuntimeError(
                                "inconsistent internal acquisition metadata profile"
                            )
                        profiles[sweep.acquisition_id]["semantic_families"][
                            family
                        ] = semantic_profile

                # Optional declarations are allowed to be absent from every
                # repeat.  A one-sided declaration, however, needs the same
                # explicit legacy attestation as other incomplete evidence.
                for fact, absent_roles in acquisition_contract[
                    "missing_declarations_by_role"
                ].items():
                    if len(absent_roles) != 1:
                        continue
                    absent_role = absent_roles[0]
                    absent_id = (
                        left_sweep.acquisition_id
                        if absent_role == left_role
                        else right_sweep.acquisition_id
                    )
                    missing_by_acquisition[absent_id].add(fact)

                for key in _OPTIONAL_EXACT_FIELDS:
                    left = left_grid._declared_scalar_metadata(key)
                    right = right_grid._declared_scalar_metadata(key)
                    if left and right:
                        if " ".join(left.casefold().split()) != " ".join(
                            right.casefold().split()
                        ):
                            raise ValueError(
                                f"repeat acquisitions {left_sweep.acquisition_id!r} "
                                f"and {right_sweep.acquisition_id!r} disagree on "
                                f"{key} metadata"
                            )
                    elif left or right:
                        missing_id = (
                            right_sweep.acquisition_id
                            if left
                            else left_sweep.acquisition_id
                        )
                        missing_by_acquisition[missing_id].add(key)

        # Repeat screening itself is valid for consistently declared near- or
        # far-field and mono- or bistatic data.  It nevertheless needs a known
        # topology/zone and stable phase center; absolute ISAR restrictions
        # (+jwt, monostatic, far field, two-way phase sign) remain the renderer's
        # responsibility.
        for sweep in values:
            geometry = profiles[sweep.acquisition_id]["semantic_families"].get(
                "acquisition_geometry", {}
            ).get("canonical_dimensions", {})
            if "scattering_configuration" not in geometry:
                missing_by_acquisition[sweep.acquisition_id].add(
                    "acquisition_geometry.scattering_configuration"
                )
            if "propagation_regime" not in geometry:
                missing_by_acquisition[sweep.acquisition_id].add(
                    "acquisition_geometry.propagation_regime"
                )
            motion = profiles[sweep.acquisition_id]["semantic_families"].get(
                "motion_state", {}
            ).get("canonical_dimensions", {})
            if motion.get("state") != "stable":
                missing_by_acquisition[sweep.acquisition_id].add(
                    "motion_state.stable"
                )

        durable_missing = {
            acquisition_id: sorted(facts)
            for acquisition_id, facts in missing_by_acquisition.items()
            if facts
        }
        if durable_missing and not bool(legacy_metadata_attested):
            details = "; ".join(
                f"{acquisition_id}: {', '.join(facts)}"
                for acquisition_id, facts in durable_missing.items()
            )
            raise ValueError(
                "repeat-acquisition metadata is incomplete ("
                + details
                + "). Declare the missing facts, or set "
                "legacy_metadata_attested=True only after verifying that every "
                "repeat used compatible units, geometry, phase reference/time "
                "convention, two-way range-phase law, polarization basis, and "
                "stable phase center"
            )

        self.sweeps = values
        self.axis_tolerance = tolerance
        self.metadata_contract = {
            "schema": "grim.repeat-acquisition-metadata-contract.v2",
            "legacy_metadata_attested": bool(legacy_metadata_attested),
            "acquisition_ids": [sweep.acquisition_id for sweep in values],
            "required_unit_fields": list(_REQUIRED_UNIT_KEYS),
            "required_coherent_fields": list(_REQUIRED_COHERENT_FIELDS),
            "range_phase_metadata_aliases": list(_RANGE_PHASE_METADATA_KEYS),
            "metadata_profiles": profiles,
            "missing_declarations_by_acquisition": durable_missing,
            "missing_declarations_covered_by_user_attestation": bool(
                durable_missing
            ),
            "declarations_inferred": False,
            "explicit_contradictions_allowed": False,
        }

    @property
    def reference(self) -> IsarSweep:
        return self.sweeps[0]

    def complex_stack(
        self,
        *,
        elevation_index: int,
        polarization_index: int,
        maximum_working_bytes: int | None = None,
    ) -> np.ndarray:
        reference = self.reference.grid
        elevation_index = int(elevation_index)
        polarization_index = int(polarization_index)
        if not 0 <= elevation_index < len(reference.elevations):
            raise IndexError("elevation_index is out of range")
        if not 0 <= polarization_index < len(reference.polarizations):
            raise IndexError("polarization_index is out of range")
        required = _repeat_stack_working_set_bytes(
            len(self.sweeps),
            len(reference.azimuths),
            len(reference.frequencies),
        )
        limit = _repeat_working_set_limit_bytes(maximum_working_bytes)
        _validate_repeat_working_set(required, limit, "Repeat complex stack")
        selection = np.ix_(
            np.arange(len(reference.azimuths)),
            [elevation_index],
            np.arange(len(reference.frequencies)),
            [polarization_index],
        )
        stack = np.empty(
            (
                len(self.sweeps),
                len(reference.azimuths),
                len(reference.frequencies),
            ),
            dtype=np.complex128,
        )
        for index, sweep in enumerate(self.sweeps):
            values = np.asarray(
                sweep.grid.rcs_slice(selection)[:, 0, :, 0],
                dtype=np.complex128,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "repeat screening requires finite complex phase histories; "
                    "crop or explicitly mask missing samples first"
                )
            stack[index] = values
        return stack

    def screen_transients(
        self,
        *,
        elevation_index: int,
        polarization_index: int,
        registration_guard: np.ndarray | None = None,
        threshold: float = 6.0,
        minimum_coherence: float = 0.8,
        maximum_working_bytes: int | None = None,
    ) -> RepeatScreenResult:
        """Register global complex drift and flag repeat-domain outlier candidates.

        The returned Boolean mask has shape ``(repeat, azimuth, frequency)``.
        No RCS samples are changed.  Review the guard region and every candidate
        before using the mask in a later physical reconstruction.
        """

        reference_grid = self.reference.grid
        elevation_index = int(elevation_index)
        polarization_index = int(polarization_index)
        if not 0 <= elevation_index < len(reference_grid.elevations):
            raise IndexError("elevation_index is out of range")
        if not 0 <= polarization_index < len(reference_grid.polarizations):
            raise IndexError("polarization_index is out of range")
        repeat_count = len(self.sweeps)
        azimuth_count = len(reference_grid.azimuths)
        frequency_count = len(reference_grid.frequencies)
        if repeat_count < 3:
            raise ValueError("transient screening requires at least three repeats")
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("outlier threshold must be finite and positive")
        minimum_coherence = float(minimum_coherence)
        if not 0.0 <= minimum_coherence <= 1.0:
            raise ValueError("minimum_coherence must be between zero and one")
        required = _repeat_screen_working_set_bytes(
            repeat_count, azimuth_count, frequency_count
        )
        limit = _repeat_working_set_limit_bytes(maximum_working_bytes)
        _validate_repeat_working_set(required, limit, "Repeat transient screening")
        slice_selection = np.ix_(
            np.arange(azimuth_count),
            [elevation_index],
            np.arange(frequency_count),
            [polarization_index],
        )
        if registration_guard is None:
            guard = np.ones((azimuth_count, frequency_count), dtype=bool)
        else:
            guard = np.asarray(registration_guard, dtype=bool)
            if guard.shape != (azimuth_count, frequency_count):
                raise ValueError("registration guard must match azimuth/frequency shape")
        if np.count_nonzero(guard) < 2:
            raise ValueError("registration guard must contain at least two samples")

        reference = np.asarray(
            self.reference.grid.rcs_slice(slice_selection)[:, 0, :, 0],
            dtype=np.complex128,
        )
        if not np.all(np.isfinite(reference)):
            raise ValueError(
                "repeat screening requires finite complex phase histories; "
                "crop or explicitly mask missing samples first"
            )
        registered = np.empty(
            (repeat_count, azimuth_count, frequency_count),
            dtype=np.complex128,
        )
        registered[0] = reference
        registrations = [
            ComplexRegistration(
                self.sweeps[0].acquisition_id, 1.0 + 0.0j, 1.0, int(np.sum(guard))
            )
        ]
        ref_guard = reference[guard]
        ref_norm = float(np.linalg.norm(ref_guard))
        if ref_norm <= 1.0e-30:
            raise ValueError("registration guard has zero reference energy")
        for index in range(1, repeat_count):
            candidate = np.asarray(
                self.sweeps[index].grid.rcs_slice(slice_selection)[:, 0, :, 0],
                dtype=np.complex128,
            )
            if not np.all(np.isfinite(candidate)):
                raise ValueError(
                    "repeat screening requires finite complex phase histories; "
                    "crop or explicitly mask missing samples first"
                )
            candidate_guard = candidate[guard]
            denominator = float(np.vdot(candidate_guard, candidate_guard).real)
            if denominator <= 1.0e-30:
                raise ValueError("registration guard has zero candidate energy")
            multiplier = np.vdot(candidate_guard, ref_guard) / denominator
            coherence = float(
                np.abs(np.vdot(candidate_guard, ref_guard))
                / max(float(np.linalg.norm(candidate_guard)) * ref_norm, 1.0e-30)
            )
            if coherence < minimum_coherence:
                raise ValueError(
                    f"repeat {self.sweeps[index].acquisition_id!r} registration "
                    f"coherence {coherence:.3f} is below {minimum_coherence:.3f}"
                )
            np.multiply(candidate, multiplier, out=registered[index])
            registrations.append(
                ComplexRegistration(
                    self.sweeps[index].acquisition_id,
                    complex(multiplier),
                    coherence,
                    int(np.sum(guard)),
                )
            )

        cell_count = azimuth_count * frequency_count
        registered_flat = registered.reshape(repeat_count, cell_count)
        center = np.empty(cell_count, dtype=np.complex128)
        chunk_cells = max(
            1,
            min(
                cell_count,
                _REPEAT_STATISTICS_SCRATCH_BYTES
                // max(16 * repeat_count, 1),
            ),
        )
        for start in range(0, cell_count, chunk_cells):
            stop = min(cell_count, start + chunk_cells)
            block = registered_flat[:, start:stop]
            center.real[start:stop] = np.median(block.real, axis=0)
            center.imag[start:stop] = np.median(block.imag, axis=0)

        residual = np.empty((repeat_count, cell_count), dtype=np.float64)
        delta = np.empty(cell_count, dtype=np.complex128)
        for index in range(repeat_count):
            np.subtract(registered_flat[index], center, out=delta)
            np.absolute(delta, out=residual[index])

        z_score = np.empty_like(residual)
        candidate_mask = np.empty(residual.shape, dtype=bool)
        for start in range(0, cell_count, chunk_cells):
            stop = min(cell_count, start + chunk_cells)
            block = residual[:, start:stop]
            median_residual = np.median(block, axis=0)
            deviation = np.abs(block - median_residual[None, :])
            robust_scale = 1.4826 * np.median(deviation, axis=0)
            floor = np.maximum(np.abs(center[start:stop]) * 1.0e-12, 1.0e-15)
            denominator = np.maximum(robust_scale, floor)
            np.subtract(
                block,
                median_residual[None, :],
                out=z_score[:, start:stop],
            )
            np.divide(
                z_score[:, start:stop],
                denominator[None, :],
                out=z_score[:, start:stop],
            )
            np.greater(
                z_score[:, start:stop],
                threshold,
                out=candidate_mask[:, start:stop],
            )
        return RepeatScreenResult(
            registered_stack=registered,
            registrations=tuple(registrations),
            robust_center=center.reshape(azimuth_count, frequency_count),
            residual_magnitude=residual.reshape(
                repeat_count, azimuth_count, frequency_count
            ),
            robust_z_score=z_score.reshape(
                repeat_count, azimuth_count, frequency_count
            ),
            candidate_outlier_mask=candidate_mask.reshape(
                repeat_count, azimuth_count, frequency_count
            ),
            threshold=threshold,
        )


__all__ = [
    "ComplexRegistration",
    "IsarSweep",
    "RepeatAcquisitionStack",
    "RepeatScreenResult",
]
