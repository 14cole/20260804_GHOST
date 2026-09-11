"""Dataset validation reports, content hashes, and support-reference checks."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from GRIM_Backend.datasets.constants import _COHERENT_OPERATION_BLOCK_CELLS
from GRIM_Backend.datasets.memory import _bounded_grid_selections


def _physical_grid_content_sha256(grid, *, namespace):
    """Hash axes, physical conventions, and complex samples in bounded blocks.

    Descriptive labels and history are excluded from the content hash.
    """

    digest = hashlib.sha256()
    digest.update(str(namespace).encode("utf-8") + b"\0")

    def _update_array(label, values):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(label).encode("ascii") + b"\0")
        digest.update(str(contiguous.shape).encode("ascii") + b"\0")
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(contiguous.tobytes(order="C"))

    for label, values in (
        ("azimuth", np.asarray(grid.azimuths, dtype=np.float64)),
        ("elevation", np.asarray(grid.elevations, dtype=np.float64)),
        ("frequency", np.asarray(grid.frequencies, dtype=np.float64)),
    ):
        _update_array(label, values)

    digest.update(b"authoritative-complex-field\0")
    digest.update(str(grid.rcs_power.shape).encode("ascii") + b"\0")
    read_complex, _real_dtype = grid._bounded_complex_slice_reader()
    for selection in _bounded_grid_selections(
        grid.rcs_power.shape, _COHERENT_OPERATION_BLOCK_CELLS
    ):
        field = np.asarray(
            read_complex(selection),
            dtype=np.complex128,
        )
        _update_array("field-real", field.real)
        _update_array("field-imag", field.imag)

    digest.update(
        json.dumps(
            [str(value) for value in grid.polarizations.tolist()],
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        json.dumps(
            dict(grid.units or {}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    convention_metadata = {
        key: grid._declared_scalar_metadata(key)
        for key in (
            "phase_reference",
            "time_convention",
            "polarization_basis",
            "amplitude_convention",
        )
    }
    digest.update(
        json.dumps(
            convention_metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _support_reference_qa(target_with_support, support_reference, difference):
    """Return unweighted sample diagnostics for an exact complex difference.

    The sums use common finite samples; complex coherence measures acquisition
    similarity.
    """

    energy_terms = {
        "pre_target_plus_support": [],
        "subtracted_support_reference": [],
        "post_support_referenced_difference": [],
        "algebraic_closure_residual": [],
    }
    cross_real_terms = []
    cross_imag_terms = []
    common_count = 0
    read_target, _target_dtype = target_with_support._bounded_complex_slice_reader()
    read_support, _support_dtype = support_reference._bounded_complex_slice_reader()
    read_difference, _difference_dtype = difference._bounded_complex_slice_reader()
    for selection in _bounded_grid_selections(
        target_with_support.rcs_power.shape, _COHERENT_OPERATION_BLOCK_CELLS
    ):
        target = np.asarray(read_target(selection), dtype=np.complex128)
        support = np.asarray(read_support(selection), dtype=np.complex128)
        post = np.asarray(read_difference(selection), dtype=np.complex128)
        common = np.isfinite(target) & np.isfinite(support) & np.isfinite(post)
        if not np.any(common):
            continue
        target = target[common]
        support = support[common]
        post = post[common]
        closure = target - support - post
        common_count += int(target.size)
        energy_terms["pre_target_plus_support"].append(
            float(np.vdot(target, target).real)
        )
        energy_terms["subtracted_support_reference"].append(
            float(np.vdot(support, support).real)
        )
        energy_terms["post_support_referenced_difference"].append(
            float(np.vdot(post, post).real)
        )
        energy_terms["algebraic_closure_residual"].append(
            float(np.vdot(closure, closure).real)
        )
        cross = np.vdot(support, target)
        cross_real_terms.append(float(cross.real))
        cross_imag_terms.append(float(cross.imag))

    def _bounded_fsum(values):
        try:
            total = float(math.fsum(values))
        except (OverflowError, ValueError):
            return float("nan")
        return total if np.isfinite(total) else float("nan")

    energies = {
        key: _bounded_fsum(values) for key, values in energy_terms.items()
    }
    cross = complex(
        _bounded_fsum(cross_real_terms), _bounded_fsum(cross_imag_terms)
    )
    pre_energy = energies["pre_target_plus_support"]
    support_energy = energies["subtracted_support_reference"]
    post_energy = energies["post_support_referenced_difference"]

    def _safe_db_ratio(numerator, denominator):
        if not (
            np.isfinite(numerator)
            and np.isfinite(denominator)
            and numerator > 0.0
            and denominator > 0.0
        ):
            return None
        return float(10.0 * np.log10(numerator / denominator))

    coherence = None
    coherence_phase_deg = None
    coherence_meaningful = bool(
        common_count >= 2
        and np.isfinite(pre_energy)
        and np.isfinite(support_energy)
        and pre_energy > 0.0
        and support_energy > 0.0
        and np.isfinite(cross)
    )
    if coherence_meaningful:
        normalization = np.sqrt(pre_energy) * np.sqrt(support_energy)
        coherence = float(abs(cross) / normalization)

        coherence = min(1.0, max(0.0, coherence))
        coherence_phase_deg = float(np.rad2deg(np.angle(cross)))

    def _finite_or_none(value):
        return float(value) if np.isfinite(value) else None

    return {
        "energy_metric": (
            "unweighted_sum_of_squared_complex_sample_magnitudes_on_common_"
            "finite_support"
        ),
        "total_sample_count": int(target_with_support.rcs_power.size),
        "common_finite_sample_count": int(common_count),
        "excluded_sample_count": int(
            target_with_support.rcs_power.size - common_count
        ),
        "energy_sum_linear": {
            key: _finite_or_none(value) for key, value in energies.items()
        },
        "post_to_pre_energy_db": _safe_db_ratio(post_energy, pre_energy),
        "reference_to_pre_energy_db": _safe_db_ratio(
            support_energy, pre_energy
        ),
        "complex_coherence": coherence,
        "complex_coherence_phase_deg": coherence_phase_deg,
        "complex_coherence_meaningful": coherence_meaningful,
        "complex_coherence_definition": (
            "abs(sum(conj(support_reference)*target_plus_support)) / "
            "sqrt(sum(abs(support_reference)^2)*"
            "sum(abs(target_plus_support)^2))"
        ),
    }


def audit_dataset(self):
    """Return a non-mutating, JSON-serializable dataset health report.

    The report contains status, errors, warnings, info, and metrics. Samples are
    checked in bounded blocks; malformed data is reported without repair.
    """
    from GRIM_Backend.datasets.constants import _ANGLE_UNITS, _FREQUENCY_UNITS, _JOIN_MERGE_BLOCK_CELLS

    errors = []
    warnings_out = []
    info = []
    metrics = {
        "axes": {},
        "grid": {},
        "metadata": {},
        "phase": {},
        "raw_complex": {},
        "seam": {},
        "frequency_uniformity": {},
        "readiness": {},
    }

    def add_issue(target, code, message, **details):
        issue = {"code": str(code), "message": str(message)}
        for key, value in details.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            issue[str(key)] = value
        target.append(issue)

    def finite_number(value):
        value = float(value)
        return value if np.isfinite(value) else None

    def iter_blocks(*arrays):
        iterator = np.nditer(
            tuple(np.asarray(array) for array in arrays),
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=[["readonly"] for _ in arrays],
            order="K",
            buffersize=_JOIN_MERGE_BLOCK_CELLS,
        )
        for block in iterator:
            if len(arrays) == 1:
                yield (np.asarray(block),)
            else:
                yield tuple(np.asarray(value) for value in block)

    numeric_axes = {}
    axes_well_formed = True
    axes_strictly_increasing = True
    for axis_name, raw_axis in (
        ("azimuth", self.azimuths),
        ("elevation", self.elevations),
        ("frequency", self.frequencies),
    ):
        axis = np.asarray(raw_axis)
        axis_metric = {
            "count": int(axis.size),
            "shape": [int(value) for value in axis.shape],
            "dtype": str(axis.dtype),
            "finite_count": 0,
            "nonfinite_count": 0,
            "duplicate_count": 0,
            "strictly_increasing": False,
            "minimum": None,
            "maximum": None,
        }
        metrics["axes"][axis_name] = axis_metric
        if axis.ndim != 1 or axis.size == 0 or axis.dtype.kind not in "iuf":
            axes_well_formed = False
            axes_strictly_increasing = False
            add_issue(
                errors,
                f"invalid_{axis_name}_axis",
                f"{axis_name} must be a nonempty one-dimensional real numeric axis",
            )
            continue
        numeric = axis.astype(np.float64, copy=False)
        numeric_axes[axis_name] = numeric
        finite_mask = np.isfinite(numeric)
        finite_count = int(np.count_nonzero(finite_mask))
        axis_metric["finite_count"] = finite_count
        axis_metric["nonfinite_count"] = int(numeric.size - finite_count)
        if finite_count:
            axis_metric["minimum"] = finite_number(np.min(numeric[finite_mask]))
            axis_metric["maximum"] = finite_number(np.max(numeric[finite_mask]))
        if finite_count != numeric.size:
            axes_well_formed = False
            axes_strictly_increasing = False
            add_issue(
                errors,
                f"nonfinite_{axis_name}_coordinate",
                f"{axis_name} contains nonfinite coordinates",
                count=int(numeric.size - finite_count),
            )
            continue
        unique_count = int(np.unique(numeric).size)
        axis_metric["duplicate_count"] = int(numeric.size - unique_count)
        if unique_count != numeric.size:
            axes_well_formed = False
            axes_strictly_increasing = False
            add_issue(
                errors,
                f"duplicate_{axis_name}_coordinate",
                f"{axis_name} contains duplicate coordinates",
                count=int(numeric.size - unique_count),
            )
        increasing = bool(
            numeric.size <= 1 or np.all(np.diff(numeric) > 0.0)
        )
        axis_metric["strictly_increasing"] = increasing
        axes_strictly_increasing &= increasing
        if not increasing and unique_count == numeric.size:
            add_issue(
                warnings_out,
                f"unsorted_{axis_name}_axis",
                f"{axis_name} is not strictly increasing; interpolation is not ready",
            )
        if axis_name == "frequency" and np.any(numeric <= 0.0):
            axes_well_formed = False
            add_issue(
                errors,
                "nonpositive_frequency",
                "frequency contains nonpositive coordinates",
                count=int(np.count_nonzero(numeric <= 0.0)),
            )

    polarizations = np.asarray(self.polarizations)
    pol_metric = {
        "count": int(polarizations.size),
        "shape": [int(value) for value in polarizations.shape],
        "dtype": str(polarizations.dtype),
        "blank_count": 0,
        "duplicate_count": 0,
    }
    metrics["axes"]["polarization"] = pol_metric
    if polarizations.ndim != 1 or polarizations.size == 0:
        axes_well_formed = False
        add_issue(
            errors,
            "invalid_polarization_axis",
            "polarization must be a nonempty one-dimensional string axis",
        )
    else:
        labels = [str(value).strip() for value in polarizations.tolist()]
        blank_count = sum(not label for label in labels)
        folded = [label.casefold() for label in labels]
        duplicate_count = len(folded) - len(set(folded))
        pol_metric["blank_count"] = int(blank_count)
        pol_metric["duplicate_count"] = int(duplicate_count)
        if blank_count:
            axes_well_formed = False
            add_issue(
                errors,
                "blank_polarization",
                "polarization contains blank labels",
                count=int(blank_count),
            )
        if duplicate_count:
            axes_well_formed = False
            add_issue(
                errors,
                "duplicate_polarization",
                "polarization contains duplicate labels after normalization",
                count=int(duplicate_count),
            )

    expected_shape = (
        int(np.asarray(self.azimuths).size),
        int(np.asarray(self.elevations).size),
        int(np.asarray(self.frequencies).size),
        int(np.asarray(self.polarizations).size),
    )
    power = np.asarray(self.rcs_power)
    phase = np.asarray(self.rcs_phase)
    grid_metric = metrics["grid"]
    grid_metric.update(
        {
            "expected_shape": list(expected_shape),
            "power_shape": [int(value) for value in power.shape],
            "phase_shape": [int(value) for value in phase.shape],
            "power_dtype": str(power.dtype),
            "phase_dtype": str(phase.dtype),
            "cell_count": int(np.prod(expected_shape, dtype=np.int64)),
            "finite_power_count": 0,
            "missing_power_count": 0,
            "infinite_power_count": 0,
            "negative_power_count": 0,
            "zero_power_count": 0,
            "minimum_finite_power": None,
            "maximum_finite_power": None,
        }
    )
    shapes_valid = power.shape == expected_shape and phase.shape == expected_shape
    if power.shape != expected_shape:
        add_issue(
            errors,
            "power_shape_mismatch",
            f"rcs_power shape {power.shape} does not match axes {expected_shape}",
        )
    if phase.shape != expected_shape:
        add_issue(
            errors,
            "phase_shape_mismatch",
            f"rcs_phase shape {phase.shape} does not match axes {expected_shape}",
        )

    power_numeric = power.dtype.kind in "iuf"
    phase_numeric = phase.dtype.kind in "iuf"
    if not power_numeric:
        add_issue(errors, "non_numeric_power", "rcs_power must be real numeric")
    if not phase_numeric:
        add_issue(errors, "non_numeric_phase", "rcs_phase must be real numeric")

    if power_numeric:
        finite_power_count = 0
        missing_power_count = 0
        infinite_power_count = 0
        negative_power_count = 0
        zero_power_count = 0
        minimum_power = None
        maximum_power = None
        for (power_block,) in iter_blocks(power):
            finite = np.isfinite(power_block)
            finite_values = power_block[finite]
            finite_power_count += int(finite_values.size)
            missing_power_count += int(np.count_nonzero(np.isnan(power_block)))
            infinite_power_count += int(np.count_nonzero(np.isinf(power_block)))
            negative_power_count += int(np.count_nonzero(finite_values < 0.0))
            zero_power_count += int(np.count_nonzero(finite_values == 0.0))
            if finite_values.size:
                block_min = float(np.min(finite_values))
                block_max = float(np.max(finite_values))
                minimum_power = block_min if minimum_power is None else min(minimum_power, block_min)
                maximum_power = block_max if maximum_power is None else max(maximum_power, block_max)
        grid_metric.update(
            {
                "finite_power_count": finite_power_count,
                "missing_power_count": missing_power_count,
                "infinite_power_count": infinite_power_count,
                "negative_power_count": negative_power_count,
                "zero_power_count": zero_power_count,
                "minimum_finite_power": finite_number(minimum_power) if minimum_power is not None else None,
                "maximum_finite_power": finite_number(maximum_power) if maximum_power is not None else None,
                "sparsity_fraction": (
                    float(missing_power_count / power.size) if power.size else None
                ),
            }
        )
        if infinite_power_count:
            add_issue(
                errors,
                "infinite_power",
                "rcs_power contains infinite samples",
                count=infinite_power_count,
            )
        if negative_power_count:
            add_issue(
                errors,
                "negative_power",
                "rcs_power contains negative finite samples",
                count=negative_power_count,
                minimum=grid_metric["minimum_finite_power"],
            )

    phase_metric = metrics["phase"]
    raw_phase_wrap = str((self.units or {}).get("phase_wrap", "")).strip()
    declared_phase_wrap = raw_phase_wrap or None
    valid_phase_wrap = declared_phase_wrap in {None, "0_360", "-180_180"}
    if not valid_phase_wrap:
        add_issue(
            errors,
            "unsupported_phase_wrap",
            "phase_wrap must be '0_360' or '-180_180' when declared",
            value=declared_phase_wrap,
        )
    phase_metric.update(
        {
            "declared_wrap": declared_phase_wrap,
            "finite_phase_count": 0,
            "missing_phase_count": 0,
            "infinite_phase_count": 0,
            "power_without_phase_count": 0,
            "phase_without_power_count": 0,
            "finite_complex_count": 0,
            "outside_minus_pi_pi_count": 0,
            "outside_declared_wrap_count": 0,
        }
    )
    if phase_numeric:
        for (phase_block,) in iter_blocks(phase):
            finite_phase = np.isfinite(phase_block)
            phase_metric["finite_phase_count"] += int(np.count_nonzero(finite_phase))
            phase_metric["missing_phase_count"] += int(np.count_nonzero(np.isnan(phase_block)))
            phase_metric["infinite_phase_count"] += int(np.count_nonzero(np.isinf(phase_block)))
            phase_metric["outside_minus_pi_pi_count"] += int(
                np.count_nonzero(
                    finite_phase & ((phase_block < -np.pi) | (phase_block >= np.pi))
                )
            )
            if declared_phase_wrap == "0_360":
                phase_metric["outside_declared_wrap_count"] += int(
                    np.count_nonzero(
                        finite_phase
                        & ((phase_block < 0.0) | (phase_block >= 2.0 * np.pi))
                    )
                )
            elif declared_phase_wrap == "-180_180":
                phase_metric["outside_declared_wrap_count"] += int(
                    np.count_nonzero(
                        finite_phase
                        & ((phase_block < -np.pi) | (phase_block >= np.pi))
                    )
                )
        if phase_metric["infinite_phase_count"]:
            add_issue(
                errors,
                "infinite_phase",
                "rcs_phase contains infinite samples",
                count=phase_metric["infinite_phase_count"],
            )
        if valid_phase_wrap and phase_metric["outside_declared_wrap_count"]:
            add_issue(
                errors,
                "phase_outside_declared_wrap",
                "finite phase samples fall outside the declared phase_wrap interval",
                count=phase_metric["outside_declared_wrap_count"],
                phase_wrap=declared_phase_wrap,
            )

    if power_numeric and phase_numeric and power.shape == phase.shape:
        for power_block, phase_block in iter_blocks(power, phase):
            finite_power = np.isfinite(power_block)
            finite_phase = np.isfinite(phase_block)
            phase_metric["power_without_phase_count"] += int(
                np.count_nonzero(finite_power & ~finite_phase)
            )
            phase_metric["phase_without_power_count"] += int(
                np.count_nonzero(~finite_power & finite_phase)
            )
            phase_metric["finite_complex_count"] += int(
                np.count_nonzero(finite_power & finite_phase)
            )
        finite_power_count = grid_metric["finite_power_count"]
        phase_metric["finite_power_phase_coverage_fraction"] = (
            float(phase_metric["finite_complex_count"] / finite_power_count)
            if finite_power_count
            else None
        )
        if phase_metric["power_without_phase_count"]:
            add_issue(
                warnings_out,
                "missing_coherent_phase",
                "finite power samples with missing phase are masked by coherent operations",
                count=phase_metric["power_without_phase_count"],
            )
        if phase_metric["phase_without_power_count"]:
            add_issue(
                warnings_out,
                "orphan_phase",
                "phase is finite where power is missing",
                count=phase_metric["phase_without_power_count"],
            )

    metadata_metric = metrics["metadata"]
    supported_units = True
    physical_metadata_valid = True
    for key, aliases, default in (
        ("azimuth", _ANGLE_UNITS, "deg"),
        ("elevation", _ANGLE_UNITS, "deg"),
        ("frequency", _FREQUENCY_UNITS, "GHz"),
    ):
        try:
            metadata_metric[f"{key}_unit"] = self._supported_unit(
                key, aliases, default
            )
        except (TypeError, ValueError) as exc:
            supported_units = False
            metadata_metric[f"{key}_unit"] = None
            add_issue(errors, f"unsupported_{key}_unit", str(exc))

    for key in ("phase_reference", "time_convention", "polarization_basis"):
        try:
            value = self._declared_scalar_metadata(key)
        except (TypeError, ValueError) as exc:
            value = ""
            add_issue(errors, f"invalid_{key}_metadata", str(exc))
        declared = bool(value)
        metadata_metric[key] = value or None
        metadata_metric[f"{key}_declared"] = declared
        if not declared:
            add_issue(
                info,
                f"unspecified_{key}",
                f"{key.replace('_', ' ')} is not declared; coherent operations "
                "record this as an assumption",
            )
    try:
        metadata_metric["linear_quantity"] = self.linear_quantity()
        metadata_metric["log_unit"] = self.default_log_unit()
        metadata_metric["angular_coordinate_system"] = self.angular_coordinate_system()
        if metadata_metric["linear_quantity"] not in {
            "sigma_3d", "sigma_2d", "power_ratio"
        }:
            physical_metadata_valid = False
            add_issue(
                errors,
                "unsupported_linear_quantity",
                "rcs_linear_quantity is not sigma_3d, sigma_2d, or power_ratio",
                value=metadata_metric["linear_quantity"],
            )
        raw_log_unit = str(
            (self.units or {}).get("rcs_log_unit", "dBsm")
        ).strip().casefold()
        if raw_log_unit not in {"dbsm", "dbke", "db"}:
            physical_metadata_valid = False
            add_issue(
                errors,
                "unsupported_log_unit",
                "rcs_log_unit is not dBsm, dBke, or dB",
                value=str((self.units or {}).get("rcs_log_unit")),
            )
        expected_log_unit = {
            "sigma_3d": "dBsm",
            "sigma_2d": "dBke",
            "power_ratio": "dB",
        }.get(metadata_metric["linear_quantity"])
        if (
            expected_log_unit is not None
            and metadata_metric["log_unit"] != expected_log_unit
        ):
            physical_metadata_valid = False
            add_issue(
                errors,
                "quantity_log_unit_mismatch",
                "rcs_linear_quantity and rcs_log_unit describe different physical quantities",
                linear_quantity=metadata_metric["linear_quantity"],
                log_unit=metadata_metric["log_unit"],
            )
    except (TypeError, ValueError) as exc:
        physical_metadata_valid = False
        add_issue(errors, "invalid_physical_metadata", str(exc))

    raw_report = self._raw_complex_consistency_report(
        expected_shape=expected_shape,
        frequencies=self.frequencies,
        rcs_power=power,
        rcs_phase=phase,
        units=self.units,
        extra=self.extra,
    )
    raw_issues = list(raw_report.pop("issues"))
    metrics["raw_complex"] = raw_report
    for raw_issue in raw_issues:
        details = {
            key: value
            for key, value in raw_issue.items()
            if key not in {"code", "message"}
        }
        add_issue(
            errors,
            raw_issue["code"],
            raw_issue["message"],
            **details,
        )

    frequency_metric = metrics["frequency_uniformity"]
    frequency = numeric_axes.get("frequency")
    frequency_uniform = None
    if frequency is None or frequency.size < 2 or np.any(~np.isfinite(frequency)):
        frequency_metric.update(
            {
                "applicable": False,
                "uniform": None,
                "nominal_step": None,
                "maximum_absolute_step_error": None,
                "maximum_relative_step_error": None,
            }
        )
    else:
        differences = np.diff(frequency)
        nominal_step = float(np.median(differences))
        absolute_error = np.abs(differences - nominal_step)
        max_absolute_error = float(np.max(absolute_error))
        scale = max(abs(nominal_step), np.finfo(np.float64).tiny)
        max_relative_error = float(max_absolute_error / scale)
        frequency_uniform = bool(
            nominal_step > 0.0
            and np.allclose(
                differences,
                nominal_step,
                rtol=1.0e-6,
                atol=np.finfo(np.float64).eps * max(abs(nominal_step), 1.0),
            )
        )
        frequency_metric.update(
            {
                "applicable": True,
                "uniform": frequency_uniform,
                "nominal_step": finite_number(nominal_step),
                "maximum_absolute_step_error": finite_number(max_absolute_error),
                "maximum_relative_step_error": finite_number(max_relative_error),
                "unit": metadata_metric.get("frequency_unit"),
            }
        )
        if not frequency_uniform:
            add_issue(
                warnings_out,
                "nonuniform_frequency",
                "frequency samples are not uniformly spaced",
                maximum_relative_step_error=frequency_metric[
                    "maximum_relative_step_error"
                ],
            )

    seam_metric = metrics["seam"]
    seam_metric.update(
        {
            "applicable": False,
            "equivalent_endpoint_pair": False,
            "equal_cell_count": 0,
            "conflict_cell_count": 0,
            "complementary_cell_count": 0,
        }
    )
    azimuth = numeric_axes.get("azimuth")
    if (
        azimuth is not None
        and azimuth.size >= 2
        and np.all(np.isfinite(azimuth))
        and shapes_valid
        and power_numeric
        and phase_numeric
    ):
        azimuth_unit = metadata_metric.get("azimuth_unit")
        period = 2.0 * np.pi if azimuth_unit == "rad" else 360.0
        seam_tolerance = (
            float(np.deg2rad(1.0e-9)) if azimuth_unit == "rad" else 1.0e-9
        )
        endpoint_pair = bool(
            np.isclose(
                abs(float(azimuth[-1] - azimuth[0])),
                period,
                rtol=0.0,
                atol=seam_tolerance,
            )
        )
        seam_metric["applicable"] = True
        seam_metric["equivalent_endpoint_pair"] = endpoint_pair
        if endpoint_pair:
            first_power = power[0, ...]
            last_power = power[-1, ...]
            first_phase = phase[0, ...]
            last_phase = phase[-1, ...]
            for p_left, p_right, ph_left, ph_right in iter_blocks(
                first_power, last_power, first_phase, last_phase
            ):
                left_finite = np.isfinite(p_left)
                right_finite = np.isfinite(p_right)
                both = left_finite & right_finite
                power_equal = both & np.isclose(
                    p_left, p_right, rtol=1.0e-6, atol=1.0e-12
                )
                both_phase = (
                    power_equal & np.isfinite(ph_left) & np.isfinite(ph_right)
                )
                zero_power = power_equal & (p_left == 0.0) & (p_right == 0.0)
                phase_conflict = (
                    both_phase
                    & ~zero_power
                    & (
                        np.abs(
                            np.angle(np.exp(1j * (ph_left - ph_right)))
                        )
                        > 1.0e-5
                    )
                )
                conflict = (both & ~power_equal) | phase_conflict
                equal = both & power_equal & ~phase_conflict
                complementary = left_finite ^ right_finite
                seam_metric["conflict_cell_count"] += int(
                    np.count_nonzero(conflict)
                )
                seam_metric["equal_cell_count"] += int(np.count_nonzero(equal))
                seam_metric["complementary_cell_count"] += int(
                    np.count_nonzero(complementary)
                )
            if seam_metric["conflict_cell_count"]:
                add_issue(
                    warnings_out,
                    "conflicting_azimuth_seam",
                    "equivalent azimuth endpoints contain conflicting finite samples",
                    count=seam_metric["conflict_cell_count"],
                )
            else:
                add_issue(
                    info,
                    "closed_azimuth_seam",
                    "azimuth contains equivalent closed-sweep endpoints",
                )

    structural_ready = bool(
        axes_well_formed
        and shapes_valid
        and power_numeric
        and phase_numeric
        and supported_units
        and physical_metadata_valid
        and not raw_issues
        and not grid_metric.get("infinite_power_count", 0)
        and not grid_metric.get("negative_power_count", 0)
        and not phase_metric.get("infinite_phase_count", 0)
        and valid_phase_wrap
        and not phase_metric.get("outside_declared_wrap_count", 0)
    )
    coherent_phase_ready = bool(
        structural_ready and phase_metric.get("finite_complex_count", 0) > 0
    )
    metrics["readiness"].update(
        {
            "incoherent_arithmetic": structural_ready,
            "strict_join": structural_ready,
            "coherent_arithmetic": coherent_phase_ready,
            "interpolation": bool(
                structural_ready and axes_strictly_increasing
            ),
            "frequency_transform": bool(
                coherent_phase_ready
                and frequency_uniform is True
                and frequency is not None
                and frequency.size >= 2
            ),
        }
    )
    add_issue(
        info,
        "audit_summary",
        "dataset audit completed without modifying samples",
        cell_count=grid_metric["cell_count"],
    )

    status = "error" if errors else ("warning" if warnings_out else "ok")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings_out,
        "info": info,
        "metrics": metrics,
    }
