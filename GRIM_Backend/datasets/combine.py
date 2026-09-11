"""Grid joins, stitching, overlap, and multi-dataset dispatch."""
from __future__ import annotations

import json

import numpy as np

from GRIM_Backend.datasets.constants import _ADOPT_CLEAN_ARRAYS_TOKEN, _JOIN_MERGE_BLOCK_CELLS


def combine_datasets(
    grids,
    operation: str,
    *,
    overlap="error",
    max_output_bytes=None,
    coherent_metadata_attested=False,
    stitch_policy="priority-first",
    tol=1.0e-6,
):
    from GRIM_Backend.datasets.grid import RcsGrid
    grids = list(grids)
    if not grids:
        raise ValueError("at least one dataset is required")
    if not isinstance(coherent_metadata_attested, (bool, np.bool_)):
        raise TypeError("coherent_metadata_attested must be True or False")
    operation = str(operation).strip().lower().replace("_", "-")
    if operation == "join":
        return RcsGrid.join_many(
            *grids,
            tol=float(tol),
            overlap=overlap,
            max_output_bytes=max_output_bytes,
        )
    if operation == "stitch":
        return RcsGrid.stitch_many(
            *grids,
            policy=stitch_policy,
            tol=float(tol),
            metadata_attested=coherent_metadata_attested,
            max_output_bytes=max_output_bytes,
            return_report=False,
        )
    result = grids[0]
    for grid in grids[1:]:
        if operation == "coherent-add":
            result = result.coherent_add(
                grid, metadata_attested=coherent_metadata_attested
            )
        elif operation == "incoherent-add":
            result = result.incoherent_add(grid)
        else:
            raise ValueError(
                "operation must be join, stitch, coherent-add, or incoherent-add"
            )
    return result


class GridCombineMixin:
    """Grid joins, stitching, overlap, and multi-dataset dispatch."""

    @classmethod
    def stitch_many(
        cls,
        *grids,
        policy="priority-first",
        tol=1e-6,
        metadata_attested=False,
        max_output_bytes=None,
        return_report=False,
    ):
        """Stitch union-grid samples using one explicit overlap policy.

        Policies are ``"priority-first"``, ``"priority-last"``,
        ``"power-mean"``, and ``"coherent-mean"``.  Priority policies choose
        an entire power/phase sample atomically in input order. ``power-mean``
        averages finite linear power and makes phase unknown only at cells
        with multiple contributors. ``coherent-mean`` averages complex fields
        and therefore requires finite phase plus compatible coherent metadata.

        Overlaps follow the selected policy and are recorded in the report.
        Processing uses bounded blocks; ``max_output_bytes`` caps the estimated
        retained arrays and workspace.

        When ``return_report`` is true, return ``(grid, report)``; otherwise
        return only the stitched grid.  Report counts are union-grid cell
        counts except ``contributing_count``, which counts all finite input
        contributions.
        """

        policies = {
            "priority-first",
            "priority-last",
            "power-mean",
            "coherent-mean",
        }
        policy = str(policy).strip().lower().replace("_", "-")
        if policy not in policies:
            raise ValueError(
                "policy must be 'priority-first', 'priority-last', "
                "'power-mean', or 'coherent-mean'"
            )
        if not isinstance(metadata_attested, (bool, np.bool_)):
            raise TypeError("metadata_attested must be True or False")
        if not isinstance(return_report, (bool, np.bool_)):
            raise TypeError("return_report must be True or False")
        try:
            tolerance = float(tol)
        except (TypeError, ValueError) as exc:
            raise TypeError("tol must be a finite nonnegative number") from exc
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tol must be a finite nonnegative number")
        if max_output_bytes is not None:
            try:
                memory_limit = int(max_output_bytes)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("max_output_bytes must be a nonnegative integer") from exc
            if memory_limit < 0:
                raise ValueError("max_output_bytes must be nonnegative")
        else:
            memory_limit = None

        grids = cls._ensure_grids(grids)
        if len(grids) > np.iinfo(np.uint32).max:
            raise ValueError("too many input grids for stitch contributor counts")
        ref = grids[0]

        convention_fields = (
            (
                "phase_reference",
                "phase references",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "time_convention",
                "time conventions",
                ref._canonical_time_convention,
            ),
            (
                "polarization_basis",
                "polarization bases",
                lambda value: " ".join(value.split()).casefold(),
            ),
        )

        metadata_assumptions = {}
        preserved_conventions = {}
        for key, label, canonicalize in convention_fields:
            values = [grid._declared_scalar_metadata(key) for grid in grids]
            values = [
                "" if grid._metadata_placeholder(value) else value
                for grid, value in zip(grids, values)
            ]
            nonblank = [value for value in values if value]
            normalized = {canonicalize(value) for value in nonblank}
            if len(normalized) > 1:
                rendered = ", ".join(
                    f"input {index}={value or '<unspecified>'!r}"
                    for index, value in enumerate(values, start=1)
                )
                metadata_assumptions[f"{key} (conflicting annotations; samples unchanged)"] = rendered
            if nonblank and len(nonblank) == len(grids) and len(normalized) == 1:
                preserved_conventions[key] = nonblank[0]
            elif len(nonblank) != len(grids):
                metadata_assumptions[key] = [
                    index
                    for index, value in enumerate(values, start=1)
                    if not value
                ]

        def scalar_convention_extra():
            return dict(preserved_conventions)


        for grid in grids[1:]:
            ref._assert_physical_metadata_compatible(grid)

        if policy == "coherent-mean":
            for grid in grids[1:]:
                ref._assert_coherent_metadata_compatible(
                    grid, metadata_attested=bool(metadata_attested)
                )

        expected_shapes = []
        coherent_missing_phase_count = 0
        for input_index, grid in enumerate(grids, start=1):
            input_phase_wrap = str(
                (grid.units or {}).get("phase_wrap", "")
            ).strip()
            if input_phase_wrap not in {"", "0_360", "-180_180"}:
                raise ValueError(
                    f"stitch input {input_index} has unsupported phase_wrap "
                    f"{input_phase_wrap!r}"
                )
            numeric_axes = (
                ("azimuth", np.asarray(grid.azimuths)),
                ("elevation", np.asarray(grid.elevations)),
                ("frequency", np.asarray(grid.frequencies)),
            )
            for axis_name, axis in numeric_axes:
                if (
                    axis.ndim != 1
                    or axis.size == 0
                    or axis.dtype.kind not in "iuf"
                    or np.any(~np.isfinite(axis))
                    or np.unique(axis).size != axis.size
                ):
                    raise ValueError(
                        f"stitch input {input_index} has an invalid {axis_name} axis"
                    )
                if axis_name == "frequency" and np.any(axis <= 0.0):
                    raise ValueError(
                        f"stitch input {input_index} has nonpositive frequencies"
                    )
            polarizations = np.asarray(grid.polarizations)
            labels = [str(value).strip() for value in polarizations.tolist()]
            if (
                polarizations.ndim != 1
                or polarizations.size == 0
                or any(not label for label in labels)
                or len({label.casefold() for label in labels}) != len(labels)
            ):
                raise ValueError(
                    f"stitch input {input_index} has an invalid polarization axis"
                )
            expected = tuple(len(axis) for _name, axis in numeric_axes) + (
                len(polarizations),
            )
            expected_shapes.append(expected)
            power = np.asarray(grid.rcs_power)
            phase = np.asarray(grid.rcs_phase)
            if power.shape != expected or phase.shape != expected:
                raise ValueError(
                    f"stitch input {input_index} sample shape does not match its axes"
                )
            if power.dtype.kind not in "iuf" or phase.dtype.kind not in "iuf":
                raise ValueError(
                    f"stitch input {input_index} power and phase must be real numeric"
                )

            infinite_power_count = 0
            negative_power_count = 0
            minimum_negative = None
            infinite_phase_count = 0
            missing_coherent_phase_count = 0
            iterator = np.nditer(
                (power, phase),
                flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"], ["readonly"]],
                order="K",
                buffersize=_JOIN_MERGE_BLOCK_CELLS,
            )
            for power_block, phase_block in iterator:
                power_block = np.asarray(power_block)
                phase_block = np.asarray(phase_block)
                infinite_power_count += int(np.count_nonzero(np.isinf(power_block)))
                finite_negative = np.isfinite(power_block) & (power_block < 0.0)
                negative_power_count += int(np.count_nonzero(finite_negative))
                if np.any(finite_negative):
                    block_minimum = float(np.min(power_block[finite_negative]))
                    minimum_negative = (
                        block_minimum
                        if minimum_negative is None
                        else min(minimum_negative, block_minimum)
                    )
                infinite_phase_count += int(np.count_nonzero(np.isinf(phase_block)))
                if policy == "coherent-mean":
                    missing_coherent_phase_count += int(
                        np.count_nonzero(
                            np.isfinite(power_block) & ~np.isfinite(phase_block)
                        )
                    )
            if infinite_power_count or infinite_phase_count:
                raise ValueError(
                    f"stitch input {input_index} contains infinite samples "
                    f"(power={infinite_power_count}, phase={infinite_phase_count})"
                )
            if negative_power_count:
                raise ValueError(
                    f"stitch input {input_index} contains {negative_power_count} "
                    "negative power sample(s); minimum is "
                    f"{minimum_negative:.17g}"
                )
            coherent_missing_phase_count += missing_coherent_phase_count

        if len(grids) == 1:
            az_union = np.array(ref.azimuths, copy=True)
            el_union = np.array(ref.elevations, copy=True)
            f_union = np.array(ref.frequencies, copy=True)
            p_union = np.array(ref.polarizations, copy=True)
        else:
            az_union = cls._axis_union(
                [grid.azimuths for grid in grids], tol=tolerance
            )
            el_union = cls._axis_union(
                [grid.elevations for grid in grids], tol=tolerance
            )
            f_union = cls._axis_union(
                [grid.frequencies for grid in grids], tol=tolerance
            )
            p_union = cls._axis_union(
                [grid.polarizations for grid in grids], tol=0.0
            )

        shape = (len(az_union), len(el_union), len(f_union), len(p_union))
        cell_count = 1
        for dimension in shape:
            cell_count *= int(dimension)
        if policy in {"power-mean", "coherent-mean"}:
            output_dtype = np.result_type(
                *[grid.rcs_power.dtype for grid in grids], np.float64
            )
        else:
            output_dtype = np.result_type(
                *[grid.rcs_power.dtype for grid in grids]
            )
        itemsize = np.dtype(output_dtype).itemsize
        output_bytes = cell_count * itemsize * 2
        state_bytes = cell_count * (
            np.dtype(np.uint32).itemsize + np.dtype(np.bool_).itemsize
        )
        merge_block_cells = min(cell_count, _JOIN_MERGE_BLOCK_CELLS)
        merge_scratch_bytes = merge_block_cells * (12 * itemsize + 64)
        estimated_peak_bytes = output_bytes + state_bytes + merge_scratch_bytes
        if memory_limit is not None and estimated_peak_bytes > memory_limit:
            raise MemoryError(
                "dense stitched grid needs about "
                f"{estimated_peak_bytes / (1024**3):.2f} GiB peak "
                f"({(output_bytes + state_bytes) / (1024**3):.2f} GiB retained "
                "during construction), above the configured limit of "
                f"{memory_limit / (1024**3):.2f} GiB"
            )

        stitched_power = np.full(shape, np.nan, dtype=output_dtype)
        stitched_phase = np.full(shape, np.nan, dtype=output_dtype)
        contributor_counts = np.zeros(shape, dtype=np.uint32)
        conflict_flags = np.zeros(shape, dtype=bool)

        mapped_indices = []
        for grid in grids:
            indices = (
                cls._indices_for_axis_values(
                    az_union, grid.azimuths, tol=tolerance
                ),
                cls._indices_for_axis_values(
                    el_union, grid.elevations, tol=tolerance
                ),
                cls._indices_for_axis_values(
                    f_union, grid.frequencies, tol=tolerance
                ),
                cls._indices_for_axis_values(
                    p_union, grid.polarizations, tol=0.0
                ),
            )
            if any(value is None for value in indices):
                raise ValueError("failed to align a dataset during stitch")
            for axis_name, axis_indices, source_axis, axis_tol in (
                ("azimuth", indices[0], grid.azimuths, tolerance),
                ("elevation", indices[1], grid.elevations, tolerance),
                ("frequency", indices[2], grid.frequencies, tolerance),
                ("polarization", indices[3], grid.polarizations, 0.0),
            ):
                if len(axis_indices) != np.asarray(source_axis).size:
                    raise ValueError(
                        f"cannot stitch: an input {axis_name} axis contains "
                        "coordinates that collapse within the matching "
                        f"tolerance ({axis_tol:g}); deduplicate that axis or "
                        "use a smaller tolerance"
                    )
            mapped_indices.append(indices)

        for grid, indices in zip(grids, mapped_indices):
            az_idx, el_idx, f_idx, p_idx = indices
            incoming_power = np.asarray(grid.rcs_power)
            incoming_phase = np.asarray(grid.rcs_phase)
            pol_block = max(1, min(len(p_idx), _JOIN_MERGE_BLOCK_CELLS))
            freq_block = max(
                1, min(len(f_idx), _JOIN_MERGE_BLOCK_CELLS // pol_block)
            )
            remaining = max(
                1, _JOIN_MERGE_BLOCK_CELLS // (pol_block * freq_block)
            )
            elev_block = max(1, min(len(el_idx), remaining))
            remaining = max(
                1,
                _JOIN_MERGE_BLOCK_CELLS
                // (pol_block * freq_block * elev_block),
            )
            az_block = max(1, min(len(az_idx), remaining))
            for a_start in range(0, len(az_idx), az_block):
                a_stop = min(a_start + az_block, len(az_idx))
                union_a = az_idx[a_start:a_stop]
                for e_start in range(0, len(el_idx), elev_block):
                    e_stop = min(e_start + elev_block, len(el_idx))
                    union_e = el_idx[e_start:e_stop]
                    for f_start in range(0, len(f_idx), freq_block):
                        f_stop = min(f_start + freq_block, len(f_idx))
                        union_f = f_idx[f_start:f_stop]
                        for p_start in range(0, len(p_idx), pol_block):
                            p_stop = min(p_start + pol_block, len(p_idx))
                            union_p = p_idx[p_start:p_stop]
                            target = np.ix_(union_a, union_e, union_f, union_p)
                            source = (
                                slice(a_start, a_stop),
                                slice(e_start, e_stop),
                                slice(f_start, f_stop),
                                slice(p_start, p_stop),
                            )
                            block_power = incoming_power[source]
                            block_phase = incoming_phase[source]
                            valid = np.isfinite(block_power)
                            if policy == "coherent-mean":
                                valid &= np.isfinite(block_phase)
                            if not np.any(valid):
                                continue

                            existing_power = stitched_power[target]
                            existing_phase = stitched_phase[target]
                            count_block = contributor_counts[target]
                            conflict_block = conflict_flags[target]
                            overlap = valid & (count_block > 0)

                            incoming_field = None
                            if policy == "coherent-mean":
                                incoming_field = np.asarray(grid.rcs_slice(source))
                                invalid_field = valid & (
                                    ~np.isfinite(incoming_field.real)
                                    | ~np.isfinite(incoming_field.imag)
                                )
                                if np.any(invalid_field):
                                    valid &= ~invalid_field
                                    if not np.any(valid):
                                        continue

                            if np.any(overlap):
                                if policy == "power-mean":
                                    reference_power = np.divide(
                                        existing_power,
                                        count_block,
                                        out=np.full_like(existing_power, np.nan),
                                        where=count_block > 0,
                                    )
                                    power_equal = overlap & np.isclose(
                                        reference_power,
                                        block_power,
                                        rtol=1.0e-6,
                                        atol=1.0e-12,
                                    )
                                    conflict_block |= overlap & ~power_equal
                                elif policy == "coherent-mean":
                                    reference_field = np.divide(
                                        existing_power + 1j * existing_phase,
                                        count_block,
                                        out=np.full(
                                            existing_power.shape,
                                            np.nan + 1j * np.nan,
                                            dtype=np.complex128,
                                        ),
                                        where=count_block > 0,
                                    )
                                    reference_power = np.abs(reference_field) ** 2
                                    incoming_field_power = np.abs(incoming_field) ** 2
                                    power_equal = overlap & np.isclose(
                                        reference_power,
                                        incoming_field_power,
                                        rtol=1.0e-6,
                                        atol=1.0e-12,
                                    )
                                    both_zero = (
                                        power_equal
                                        & (reference_power == 0.0)
                                        & (incoming_field_power == 0.0)
                                    )
                                    phase_delta = np.abs(
                                        np.angle(reference_field / incoming_field)
                                    )
                                    phase_conflict = (
                                        overlap
                                        & power_equal
                                        & ~both_zero
                                        & (phase_delta > 1.0e-5)
                                    )
                                    conflict_block |= (
                                        (overlap & ~power_equal) | phase_conflict
                                    )
                                else:
                                    power_equal = overlap & np.isclose(
                                        existing_power,
                                        block_power,
                                        rtol=1.0e-6,
                                        atol=1.0e-12,
                                    )
                                    both_phase = (
                                        power_equal
                                        & np.isfinite(existing_phase)
                                        & np.isfinite(block_phase)
                                    )
                                    both_zero = (
                                        power_equal
                                        & (existing_power == 0.0)
                                        & (block_power == 0.0)
                                    )
                                    phase_delta = np.abs(
                                        np.angle(
                                            np.exp(1j * (existing_phase - block_phase))
                                        )
                                    )
                                    phase_conflict = (
                                        both_phase
                                        & ~both_zero
                                        & (phase_delta > 1.0e-5)
                                    )
                                    conflict_block |= (
                                        (overlap & ~power_equal) | phase_conflict
                                    )

                            first_contribution = valid & (count_block == 0)
                            repeated_contribution = valid & (count_block > 0)
                            if policy == "priority-first":
                                existing_power[first_contribution] = block_power[
                                    first_contribution
                                ]
                                existing_phase[first_contribution] = block_phase[
                                    first_contribution
                                ]
                            elif policy == "priority-last":
                                existing_power[valid] = block_power[valid]
                                existing_phase[valid] = block_phase[valid]
                            elif policy == "power-mean":
                                existing_power[first_contribution] = block_power[
                                    first_contribution
                                ]
                                existing_phase[first_contribution] = block_phase[
                                    first_contribution
                                ]
                                existing_power[repeated_contribution] += block_power[
                                    repeated_contribution
                                ]
                                existing_phase[repeated_contribution] = np.nan
                            else:
                                field_real = incoming_field.real
                                field_imag = incoming_field.imag
                                existing_power[first_contribution] = field_real[
                                    first_contribution
                                ]
                                existing_phase[first_contribution] = field_imag[
                                    first_contribution
                                ]
                                existing_power[repeated_contribution] += field_real[
                                    repeated_contribution
                                ]
                                existing_phase[repeated_contribution] += field_imag[
                                    repeated_contribution
                                ]

                            count_block[valid] += np.uint32(1)
                            stitched_power[target] = existing_power
                            stitched_phase[target] = existing_phase
                            contributor_counts[target] = count_block
                            conflict_flags[target] = conflict_block

        flat_power = stitched_power.reshape(-1)
        flat_phase = stitched_phase.reshape(-1)
        flat_counts = contributor_counts.reshape(-1)
        if policy in {"power-mean", "coherent-mean"}:
            for start in range(0, cell_count, _JOIN_MERGE_BLOCK_CELLS):
                stop = min(start + _JOIN_MERGE_BLOCK_CELLS, cell_count)
                counts_block = flat_counts[start:stop]
                valid = counts_block > 0
                if policy == "power-mean":
                    power_block = flat_power[start:stop]
                    power_block[valid] /= counts_block[valid]
                    if np.any(~np.isfinite(power_block[valid])):
                        raise ValueError(
                            "power-mean stitch produced nonfinite output power"
                        )
                else:
                    real_block = flat_power[start:stop]
                    imag_block = flat_phase[start:stop]
                    mean_real = real_block[valid] / counts_block[valid]
                    mean_imag = imag_block[valid] / counts_block[valid]
                    result_power = mean_real * mean_real + mean_imag * mean_imag
                    if np.any(~np.isfinite(result_power)):
                        raise ValueError(
                            "coherent-mean stitch produced nonfinite output power"
                        )
                    real_block[valid] = result_power
                    result_phase = np.arctan2(mean_imag, mean_real)
                    result_phase[result_power == 0.0] = np.nan
                    imag_block[valid] = result_phase
                flat_power[start:stop][~valid] = np.nan
                flat_phase[start:stop][~valid] = np.nan

        output_phase_wrap = str(
            (ref.units or {}).get("phase_wrap", "")
        ).strip() or "-180_180"
        for start in range(0, cell_count, _JOIN_MERGE_BLOCK_CELLS):
            stop = min(start + _JOIN_MERGE_BLOCK_CELLS, cell_count)
            phase_block = flat_phase[start:stop]
            finite_phase = np.isfinite(phase_block)
            if output_phase_wrap == "0_360":
                phase_block[finite_phase] = np.mod(
                    phase_block[finite_phase], 2.0 * np.pi
                )
            else:
                phase_block[finite_phase] = (
                    np.mod(phase_block[finite_phase] + np.pi, 2.0 * np.pi)
                    - np.pi
                )

        contributing_count = 0
        output_finite_count = 0
        overlap_count = 0
        conflict_count = 0
        max_contributors = 0
        flat_conflicts = conflict_flags.reshape(-1)
        for start in range(0, cell_count, _JOIN_MERGE_BLOCK_CELLS):
            stop = min(start + _JOIN_MERGE_BLOCK_CELLS, cell_count)
            counts_block = flat_counts[start:stop]
            conflicts_block = flat_conflicts[start:stop]
            contributing_count += int(np.sum(counts_block, dtype=np.uint64))
            output_finite_count += int(np.count_nonzero(counts_block > 0))
            overlap_count += int(np.count_nonzero(counts_block > 1))
            conflict_count += int(
                np.count_nonzero((counts_block > 1) & conflicts_block)
            )
            if counts_block.size:
                max_contributors = max(
                    max_contributors, int(np.max(counts_block))
                )
        equal_count = int(overlap_count - conflict_count)
        report = {
            "schema": "grim.stitch-report.v1",
            "policy": policy,
            "selected_policy": policy,
            "input_count": int(len(grids)),
            "output_cell_count": int(cell_count),
            "output_finite_count": int(output_finite_count),
            "missing_count": int(cell_count - output_finite_count),
            "single_source_count": int(output_finite_count - overlap_count),
            "contributing_count": int(contributing_count),
            "overlap_count": int(overlap_count),
            "equal_count": int(equal_count),
            "conflict_count": int(conflict_count),
            "max_contributors": int(max_contributors),
            "masked_missing_phase_sample_count": int(
                coherent_missing_phase_count
            ),
            "metadata_assumptions": metadata_assumptions,
            "tolerance": float(tolerance),
            "estimated_peak_bytes": int(estimated_peak_bytes),
            "count_semantics": (
                "union-grid cells; contributing_count is finite input samples"
            ),
        }

        if policy == "coherent-mean" and output_finite_count == 0:
            raise ValueError(
                "coherent-mean stitch has no usable complex samples"
            )

        if policy == "coherent-mean":
            attested_history, attested_extra = ref._coherent_attestation_provenance(
                grids[1:],
                operation="coherent-mean-stitch",
                metadata_attested=bool(metadata_attested),
            )
            base_history = (
                attested_history
                if attested_history is not None
                else str(ref.history or "").strip()
            )
            output_extra = (
                dict(attested_extra)
                if attested_extra is not None
                else scalar_convention_extra()
            )
        else:
            base_history = str(ref.history or "").strip()
            output_extra = scalar_convention_extra()
        if metadata_assumptions:
            output_extra["merge_metadata_assumption_json"] = json.dumps(
                {
                    "schema": "grim.merge-metadata-assumption.v1",
                    "operation": "overlap-merge",
                    "policy": policy,
                    "input_count": len(grids),
                    "unspecified_declarations_by_input": metadata_assumptions,
                    "declarations_inferred": False,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        history_entry = (
            f"Stitch ({policy}, inputs={len(grids)}): "
            f"overlap={overlap_count}, equal={equal_count}, "
            f"conflict={conflict_count}, contributors={contributing_count}"
        )
        history = (
            f"{base_history}\n{history_entry}" if base_history else history_entry
        )
        if metadata_assumptions:
            assumption_entry = (
                "Overlap merge retained unspecified metadata: "
                + ", ".join(sorted(metadata_assumptions))
            )
            history = f"{history}\n{assumption_entry}"
        provenance = dict(report)
        provenance.update(
            {
                "schema": "grim.stitch-provenance.v1",
                "metadata_attested": bool(metadata_attested),
                "input_sources": [str(grid.source_path or "") for grid in grids],
            }
        )
        output_extra["stitch_provenance_json"] = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        output_units = dict(ref.units)
        output_units["phase_wrap"] = output_phase_wrap
        stitched = cls(
            az_union,
            el_union,
            f_union,
            p_union,
            rcs_power=stitched_power,
            rcs_phase=stitched_phase,
            rcs_domain="power_phase",
            source_path=ref.source_path,
            history=history,
            units=output_units,
            extra=output_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )
        if bool(return_report):
            return stitched, report
        return stitched

    @classmethod
    def join_many(cls, *grids, tol=1e-6, overlap="error", max_output_bytes=None):
        """Join datasets on union axes without silently replacing finite data.

        ``overlap`` may be ``"error"`` (default), ``"first"``, or ``"last"``.
        Equal finite samples are accepted in all modes. ``max_output_bytes`` can
        cap the estimated peak allocation for memory-aware folder workflows.
        """
        grids = cls._ensure_grids(grids)
        if overlap not in {"error", "first", "last"}:
            raise ValueError("overlap must be 'error', 'first', or 'last'")
        ref = grids[0]

        def canonical_json_scalar(value):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return value.strip()
            return json.dumps(
                decoded,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )

        scalar_metadata_fields = (
            (
                "phase_reference",
                "phase references",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "time_convention",
                "time conventions",
                ref._canonical_time_convention,
            ),
            (
                "polarization_basis",
                "polarization bases",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "amplitude_convention",
                "amplitude conventions",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "complex_field_domain",
                "complex-field domains",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "combine_role",
                "combination roles",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "assembly_response_role",
                "Assembly response roles",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "assembly_base_sha256",
                "Assembly base identities",
                lambda value: value.strip().casefold(),
            ),
            (
                "assembly_base_response_sha256",
                "Assembly base response identities",
                lambda value: value.strip().casefold(),
            ),
            (
                "assembly_angular_coordinate_contract",
                "Assembly angular-coordinate contracts",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "elevation_coordinate_convention",
                "elevation-coordinate conventions",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "sentri_elevation_convention",
                "SENTRi elevation conventions",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "sentri_coordinate_mapping",
                "SENTRi coordinate mappings",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "feature_provenance_json",
                "feature provenance records",
                canonical_json_scalar,
            ),
        )
        for grid in grids[1:]:
            ref._assert_physical_metadata_compatible(grid)

        metadata_assumptions = {}
        for key, label, canonicalize in scalar_metadata_fields:
            declared = [grid._declared_scalar_metadata(key) for grid in grids]
            declared = [
                "" if grid._metadata_placeholder(value) else value
                for grid, value in zip(grids, declared)
            ]
            nonblank = [value for value in declared if value]
            normalized = {canonicalize(value) for value in nonblank}
            if len(normalized) > 1:
                rendered = ", ".join(
                    f"input {index}={value or '<unspecified>'!r}"
                    for index, value in enumerate(declared, start=1)
                )
                if key in {"phase_reference", "time_convention", "polarization_basis", "amplitude_convention", "complex_field_domain", "feature_provenance_json", "amplitude_version"}:
                    metadata_assumptions[f"{key} (conflicting annotations; samples unchanged)"] = rendered
                else:
                    raise ValueError(
                        f"cannot join grids with different explicit {label}: {rendered}"
                    )
            if nonblank and len(nonblank) != len(grids):
                metadata_assumptions[key] = {
                    "declared_input_indices": [
                        index
                        for index, value in enumerate(declared, start=1)
                        if value
                    ],
                    "unspecified_input_indices": [
                        index
                        for index, value in enumerate(declared, start=1)
                        if not value
                    ],
                }

        preserved_scalar_extra = {}
        for key, _label, canonicalize in scalar_metadata_fields:
            declared = [grid._declared_scalar_metadata(key) for grid in grids]
            declared = [
                "" if grid._metadata_placeholder(value) else value
                for grid, value in zip(grids, declared)
            ]
            nonblank = [value for value in declared if value]
            if not nonblank:
                continue
            if any(not value for value in declared):


                continue
            normalized = {canonicalize(value) for value in nonblank}
            if len(normalized) == 1:
                preserved_scalar_extra[key] = nonblank[0]

        raw_inputs = []
        preserve_raw_amplitude = True
        for grid in grids:
            raw_real = grid.extra.get("rcs_amp_real")
            raw_imag = grid.extra.get("rcs_amp_imag")
            if raw_real is None or raw_imag is None:
                preserve_raw_amplitude = False
                break
            try:
                raw_real = np.asarray(raw_real, dtype=np.float64)
                raw_imag = np.asarray(raw_imag, dtype=np.float64)
            except (TypeError, ValueError):
                preserve_raw_amplitude = False
                break
            if (
                raw_real.shape != grid.rcs_power.shape
                or raw_imag.shape != grid.rcs_power.shape
            ):
                preserve_raw_amplitude = False
                break


            modeled = np.isfinite(grid.rcs_power)
            raw_finite = np.isfinite(raw_real) & np.isfinite(raw_imag)
            if np.any(modeled & ~raw_finite):
                preserve_raw_amplitude = False
                break
            raw_inputs.append((raw_real, raw_imag))
        if not preserve_raw_amplitude:
            raw_inputs = []
        if len(grids) == 1:


            az_union = np.array(ref.azimuths, copy=True)
            el_union = np.array(ref.elevations, copy=True)
            f_union = np.array(ref.frequencies, copy=True)
            p_union = np.array(ref.polarizations, copy=True)
        else:
            az_union = cls._axis_union([grid.azimuths for grid in grids], tol=tol)
            el_union = cls._axis_union([grid.elevations for grid in grids], tol=tol)
            f_union = cls._axis_union([grid.frequencies for grid in grids], tol=tol)
            p_union = cls._axis_union([grid.polarizations for grid in grids], tol=0.0)

        shape = (len(az_union), len(el_union), len(f_union), len(p_union))
        out_dtype = np.result_type(*[g.rcs_power.dtype for g in grids])
        cell_count = 1
        for dimension in shape:
            cell_count *= int(dimension)
        itemsize = np.dtype(out_dtype).itemsize
        raw_output_bytes = (
            cell_count * 2 * np.dtype(np.float64).itemsize
            if preserve_raw_amplitude else 0
        )
        output_bytes = cell_count * itemsize * 2 + raw_output_bytes


        merge_block_cells = min(cell_count, _JOIN_MERGE_BLOCK_CELLS)
        merge_scratch_bytes = merge_block_cells * (
            8 * itemsize + (48 if preserve_raw_amplitude else 32)
        )
        estimated_peak_bytes = output_bytes + cell_count + merge_scratch_bytes
        if max_output_bytes is not None and estimated_peak_bytes > int(max_output_bytes):
            raise MemoryError(
                f"dense joined grid needs about {estimated_peak_bytes / (1024**3):.2f} GiB peak "
                f"({output_bytes / (1024**3):.2f} GiB retained), "
                f"above the configured limit of {int(max_output_bytes) / (1024**3):.2f} GiB"
            )
        joined_power = np.full(shape, np.nan, dtype=out_dtype)
        joined_phase = np.full(shape, np.nan, dtype=out_dtype)
        joined_raw_real = (
            np.full(shape, np.nan, dtype=np.float64)
            if preserve_raw_amplitude else None
        )
        joined_raw_imag = (
            np.full(shape, np.nan, dtype=np.float64)
            if preserve_raw_amplitude else None
        )

        for grid_index, grid in enumerate(grids):
            az_idx = cls._indices_for_axis_values(az_union, grid.azimuths, tol=tol)
            el_idx = cls._indices_for_axis_values(el_union, grid.elevations, tol=tol)
            f_idx = cls._indices_for_axis_values(f_union, grid.frequencies, tol=tol)
            p_idx = cls._indices_for_axis_values(p_union, grid.polarizations, tol=0.0)
            if az_idx is None or el_idx is None or f_idx is None or p_idx is None:
                raise ValueError("failed to align a dataset during join")
            for axis_name, indices, source_axis, axis_tol in (
                ("azimuth", az_idx, grid.azimuths, tol),
                ("elevation", el_idx, grid.elevations, tol),
                ("frequency", f_idx, grid.frequencies, tol),
                ("polarization", p_idx, grid.polarizations, 0.0),
            ):
                if len(indices) != np.asarray(source_axis).size:
                    raise ValueError(
                        f"cannot join: an input {axis_name} axis contains "
                        "coordinates that collapse within the matching "
                        f"tolerance ({axis_tol:g}); deduplicate that axis or "
                        "use a smaller tolerance"
                    )


            incoming_power = np.asarray(grid.rcs_power)
            incoming_phase = np.asarray(grid.rcs_phase)


            pol_block = max(1, min(len(p_idx), _JOIN_MERGE_BLOCK_CELLS))
            freq_block = max(
                1,
                min(len(f_idx), _JOIN_MERGE_BLOCK_CELLS // pol_block),
            )
            remaining = max(
                1,
                _JOIN_MERGE_BLOCK_CELLS // (pol_block * freq_block),
            )
            elev_block = max(1, min(len(el_idx), remaining))
            remaining = max(
                1,
                _JOIN_MERGE_BLOCK_CELLS
                // (pol_block * freq_block * elev_block),
            )
            az_block = max(1, min(len(az_idx), remaining))
            for a_start in range(0, len(az_idx), az_block):
                a_stop = min(a_start + az_block, len(az_idx))
                union_a = az_idx[a_start:a_stop]
                for e_start in range(0, len(el_idx), elev_block):
                    e_stop = min(e_start + elev_block, len(el_idx))
                    union_e = el_idx[e_start:e_stop]
                    for f_start in range(0, len(f_idx), freq_block):
                        f_stop = min(f_start + freq_block, len(f_idx))
                        union_f = f_idx[f_start:f_stop]
                        for p_start in range(0, len(p_idx), pol_block):
                            p_stop = min(p_start + pol_block, len(p_idx))
                            union_p = p_idx[p_start:p_stop]
                            target = np.ix_(union_a, union_e, union_f, union_p)
                            existing_power = joined_power[target]
                            existing_phase = joined_phase[target]
                            block_selection = (
                                slice(a_start, a_stop),
                                slice(e_start, e_stop),
                                slice(f_start, f_stop),
                                slice(p_start, p_stop),
                            )
                            block_power = incoming_power[block_selection]
                            block_phase = incoming_phase[block_selection]
                            if preserve_raw_amplitude:
                                incoming_raw_real, incoming_raw_imag = raw_inputs[
                                    grid_index
                                ]
                                existing_raw_real = joined_raw_real[target]
                                existing_raw_imag = joined_raw_imag[target]
                                block_raw_real = incoming_raw_real[block_selection]
                                block_raw_imag = incoming_raw_imag[block_selection]

                            both = np.isfinite(existing_power) & np.isfinite(block_power)
                            power_conflict = both & ~np.isclose(
                                existing_power, block_power, rtol=1e-6, atol=1e-12
                            )
                            both_phase = (
                                both
                                & np.isfinite(existing_phase)
                                & np.isfinite(block_phase)
                            )
                            phase_delta = np.abs(
                                np.angle(np.exp(1j * (existing_phase - block_phase)))
                            )
                            both_zero = (
                                both
                                & (existing_power == 0.0)
                                & (block_power == 0.0)
                            )


                            phase_conflict = (
                                both_phase
                                & ~both_zero
                                & (phase_delta > 1e-5)
                            )
                            raw_conflict = np.zeros_like(both)
                            if preserve_raw_amplitude:
                                both_raw = (
                                    both
                                    & np.isfinite(existing_raw_real)
                                    & np.isfinite(existing_raw_imag)
                                    & np.isfinite(block_raw_real)
                                    & np.isfinite(block_raw_imag)
                                )
                                raw_equal = (
                                    np.isclose(
                                        existing_raw_real,
                                        block_raw_real,
                                        rtol=1.0e-12,
                                        atol=1.0e-15,
                                    )
                                    & np.isclose(
                                        existing_raw_imag,
                                        block_raw_imag,
                                        rtol=1.0e-12,
                                        atol=1.0e-15,
                                    )
                                )
                                raw_conflict = both_raw & ~raw_equal
                            if overlap == "error" and (
                                np.any(power_conflict)
                                or np.any(phase_conflict)
                                or np.any(raw_conflict)
                            ):
                                raise ValueError(
                                    "conflicting finite samples overlap during join"
                                )

                            if overlap == "last":
                                take_power = np.isfinite(block_power)
                                take_phase = take_power
                            else:
                                take_power = (
                                    ~np.isfinite(existing_power)
                                    & np.isfinite(block_power)
                                )


                                fill_phase = (
                                    both
                                    & ~power_conflict
                                    & ~np.isfinite(existing_phase)
                                    & np.isfinite(block_phase)
                                )
                                take_phase = take_power | fill_phase
                            existing_power[take_power] = block_power[take_power]
                            existing_phase[take_phase] = block_phase[take_phase]
                            joined_power[target] = existing_power
                            joined_phase[target] = existing_phase
                            if preserve_raw_amplitude:
                                existing_raw_real[take_power] = block_raw_real[
                                    take_power
                                ]
                                existing_raw_imag[take_power] = block_raw_imag[
                                    take_power
                                ]
                                joined_raw_real[target] = existing_raw_real
                                joined_raw_imag[target] = existing_raw_imag


        finite = np.empty(shape, dtype=bool)
        np.isfinite(joined_power, out=finite)
        np.maximum(joined_power, 0.0, out=joined_power, where=finite)
        np.logical_not(finite, out=finite)
        joined_power[finite] = np.nan
        np.isfinite(joined_phase, out=finite)
        np.logical_not(finite, out=finite)
        joined_phase[finite] = np.nan
        np.isfinite(joined_power, out=finite)
        np.logical_not(finite, out=finite)
        joined_phase[finite] = np.nan
        if preserve_raw_amplitude:
            joined_raw_real[finite] = np.nan
            joined_raw_imag[finite] = np.nan
        del finite

        output_extra = dict(preserved_scalar_extra)
        if metadata_assumptions:
            output_extra["merge_metadata_assumption_json"] = json.dumps(
                {
                    "schema": "grim.merge-metadata-assumption.v1",
                    "operation": "strict-merge",
                    "input_count": len(grids),
                    "one_sided_declarations": metadata_assumptions,
                    "declarations_inferred": False,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        if preserve_raw_amplitude:
            output_extra["rcs_amp_real"] = joined_raw_real
            output_extra["rcs_amp_imag"] = joined_raw_imag
            output_extra["raw_complex_amplitude_preserved"] = True

        history = str(ref.history or "").strip()
        if metadata_assumptions:
            note = (
                "Join allowed one-sided metadata as unspecified: "
                + ", ".join(sorted(metadata_assumptions))
            )
            history = f"{history}\n{note}" if history else note

        return cls(
            az_union,
            el_union,
            f_union,
            p_union,
            rcs_power=joined_power,
            rcs_phase=joined_phase,
            rcs_domain="power_phase",
            source_path=ref.source_path,
            history=history,
            units=dict(ref.units),
            extra=output_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @classmethod
    def overlap_many(cls, *grids, tol=1e-6):
        """Return one cropped dataset per input, all on common overlap axes.

        Every input participates equally in one all-selected intersection; no
        input is treated as a reference grid.  Numeric matching and the output
        axis coordinates are therefore independent of input selection order.

        Overlap is enforced cell-wise: if any input is missing data (NaN) at a
        given (az, el, freq, pol) cell, that cell is set to NaN in every output.
        Axis values whose entire slice becomes NaN after this intersection are
        dropped — so e.g. a frequency that one dataset lacks for HH but all
        datasets have for VV will stay on the axis, with HH masked to NaN.
        """
        grids = cls._ensure_grids(grids)
        if len(grids) == 1:
            return [grids[0]]
        for grid in grids[1:]:
            grids[0]._assert_axis_metadata_compatible(grid)

        az_common, az_indices = cls._common_axis_alignment(
            [grid.azimuths for grid in grids], tol=tol
        )
        el_common, el_indices = cls._common_axis_alignment(
            [grid.elevations for grid in grids], tol=tol
        )
        f_common, f_indices = cls._common_axis_alignment(
            [grid.frequencies for grid in grids], tol=tol
        )
        p_common, p_indices = cls._common_axis_alignment(
            [grid.polarizations for grid in grids], tol=0.0
        )

        if (
            az_common.size == 0
            or el_common.size == 0
            or f_common.size == 0
            or p_common.size == 0
        ):
            raise ValueError("no overlap across one or more axes")

        aligned_power = []
        aligned_phase = []
        for grid_idx, grid in enumerate(grids):
            az_idx = az_indices[grid_idx]
            el_idx = el_indices[grid_idx]
            f_idx = f_indices[grid_idx]
            p_idx = p_indices[grid_idx]
            aligned_power.append(grid.rcs_power[np.ix_(az_idx, el_idx, f_idx, p_idx)].copy())
            aligned_phase.append(grid.rcs_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)].copy())

        missing_any = np.zeros(aligned_power[0].shape, dtype=bool)
        for power in aligned_power:
            missing_any |= ~np.isfinite(power)
        for power, phase in zip(aligned_power, aligned_phase):
            power[missing_any] = np.nan
            phase[missing_any] = np.nan

        finite = ~missing_any
        az_keep = finite.any(axis=(1, 2, 3))
        el_keep = finite.any(axis=(0, 2, 3))
        f_keep = finite.any(axis=(0, 1, 3))
        p_keep = finite.any(axis=(0, 1, 2))

        if not (az_keep.any() and el_keep.any() and f_keep.any() and p_keep.any()):
            raise ValueError("no overlap across one or more axes")

        az_sel = np.where(az_keep)[0]
        el_sel = np.where(el_keep)[0]
        f_sel = np.where(f_keep)[0]
        p_sel = np.where(p_keep)[0]
        az_common = az_common[az_sel]
        el_common = el_common[el_sel]
        f_common = f_common[f_sel]
        p_common = p_common[p_sel]

        overlap_grids = []
        final_common_selection = np.ix_(az_sel, el_sel, f_sel, p_sel)
        final_missing = missing_any[final_common_selection]
        for grid_index, (grid, power, phase) in enumerate(
            zip(grids, aligned_power, aligned_phase)
        ):
            source_indices = (
                np.asarray(az_indices[grid_index], dtype=int)[az_sel],
                np.asarray(el_indices[grid_index], dtype=int)[el_sel],
                np.asarray(f_indices[grid_index], dtype=int)[f_sel],
                np.asarray(p_indices[grid_index], dtype=int)[p_sel],
            )
            source_selection = np.ix_(*source_indices)
            source_axes = (
                np.asarray(grid.azimuths)[source_indices[0]],
                np.asarray(grid.elevations)[source_indices[1]],
                np.asarray(grid.frequencies)[source_indices[2]],
                np.asarray(grid.polarizations)[source_indices[3]],
            )
            relabeled = not all(
                np.array_equal(source, target)
                for source, target in zip(
                    source_axes, (az_common, el_common, f_common, p_common)
                )
            )
            overlap_extra = grid._exact_transform_extra(
                lambda value, selection=source_selection: value[selection],
                coordinate_change=("overlap-axis-relabel" if relabeled else None),
            )
            for raw_key in grid._RAW_AMPLITUDE_EXTRA_KEYS:
                if raw_key in overlap_extra:
                    raw = np.asarray(overlap_extra[raw_key]).copy()
                    raw[final_missing] = np.nan
                    overlap_extra[raw_key] = raw
            overlap_grids.append(
                cls(
                    az_common,
                    el_common,
                    f_common,
                    p_common,
                    rcs_power=power[final_common_selection],
                    rcs_phase=phase[final_common_selection],
                    rcs_domain="power_phase",
                    source_path=grid.source_path,
                    history=grid.history,
                    units=dict(grid.units),
                    extra=overlap_extra,
                )
            )

        return overlap_grids
