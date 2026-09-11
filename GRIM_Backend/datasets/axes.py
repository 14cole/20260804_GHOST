"""Axis selection, alignment, interpolation, cropping, and wrapping."""
from __future__ import annotations

import copy

import numpy as np

from GRIM_Backend.datasets.constants import (
    _ADOPT_CLEAN_ARRAYS_TOKEN,
    _ANGLE_UNITS,
    _FREQUENCY_UNITS,
)


class GridAxesMixin:
    """Axis selection, alignment, interpolation, cropping, and wrapping."""

    def edit_axis_value(self, name, index, value):
        """Return a grid with one safely edited axis value.

        Numeric coordinate edits are kept finite and unique, then the edited
        axis is stable-sorted.  Every sample array follows the same permutation,
        including passthrough arrays whose leading four dimensions match the
        RCS grid.  Polarization edits are label-only: surrounding whitespace is
        removed, blank labels and case-insensitive duplicates are rejected, and
        channel order is preserved.

        The operation is transactional: validation and all reordered arrays are
        prepared before a new :class:`RcsGrid` is returned, so ``self`` is never
        partially mutated when an edit is invalid.
        """
        from GRIM_Backend.datasets.grid import RcsGrid

        axis_specs = {
            "azimuth": ("azimuths", 0),
            "elevation": ("elevations", 1),
            "frequency": ("frequencies", 2),
            "polarization": ("polarizations", 3),
        }
        try:
            attribute, axis_index = axis_specs[str(name).strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unknown axis name: {name}") from exc

        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("axis index must be an integer")
        item_index = int(index)
        source_axis = np.asarray(getattr(self, attribute))
        if item_index < 0 or item_index >= source_axis.size:
            raise IndexError(
                f"{name} axis index {item_index} is outside 0..{source_axis.size - 1}"
            )

        axes = [
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
        ]
        order = np.arange(source_axis.size, dtype=int)
        reordered = False

        if axis_index == 3:
            old_value = str(source_axis[item_index])
            new_value = str(value).strip().upper()
            if not new_value:
                raise ValueError("polarization label must not be blank")
            duplicate_key = new_value.casefold()
            if any(
                str(label).strip().casefold() == duplicate_key
                for candidate_index, label in enumerate(source_axis)
                if candidate_index != item_index
            ):
                raise ValueError(
                    f"polarization label {new_value!r} duplicates another channel"
                )
            if new_value == old_value:
                return self


            labels = [str(label) for label in source_axis.tolist()]
            labels[item_index] = new_value
            axes[axis_index] = np.asarray(labels, dtype=str)
            old_text = repr(old_value)
            new_text = repr(new_value)
        else:
            old_value = float(source_axis[item_index])
            try:
                new_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} axis value must be numeric") from exc
            if not np.isfinite(new_value):
                raise ValueError(f"{name} axis value must be finite")
            if axis_index == 2 and new_value <= 0.0:
                raise ValueError("frequency axis value must be greater than zero")
            if new_value == old_value:
                return self

            edited_axis = np.asarray(source_axis, dtype=float).copy()
            edited_axis[item_index] = new_value
            if np.unique(edited_axis).size != edited_axis.size:
                raise ValueError(
                    f"{name} axis value {new_value:g} duplicates another coordinate"
                )
            order = np.argsort(edited_axis, kind="stable")
            reordered = not np.array_equal(
                order, np.arange(edited_axis.size, dtype=int)
            )
            axes[axis_index] = edited_axis[order]
            old_text = f"{old_value:g}"
            new_text = f"{new_value:g}"

        if reordered:
            power = np.take(self.rcs_power, order, axis=axis_index)
            phase = np.take(self.rcs_phase, order, axis=axis_index)
        else:


            power = np.array(self.rcs_power, copy=True)
            phase = np.array(self.rcs_phase, copy=True)

        original_shape = tuple(self.rcs_power.shape)
        stale_grid_metadata = {
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
            "requested_radar_grid_json",
        }
        if axis_index == 3:
            stale_grid_metadata.update(
                {"polarization_alias_primary", "polarization_aliases_json"}
            )

        edited_extra = {}
        for key, extra_value in self.extra.items():
            if key in stale_grid_metadata:
                continue
            if reordered:
                extra_array = np.asarray(extra_value)
                if (
                    extra_array.ndim >= 4
                    and tuple(extra_array.shape[:4]) == original_shape
                ):
                    edited_extra[key] = np.take(
                        extra_array, order, axis=axis_index
                    )
                    continue


            edited_extra[key] = extra_value
        self._drop_malformed_raw_metadata(edited_extra)
        self._invalidate_assembly_sampling_hash(
            edited_extra, f"edit-{str(name).strip().lower()}-axis"
        )
        if axis_index in (0, 1):
            edited_extra.pop("assembly_angular_coordinate_contract", None)

        history_entry = (
            f"Edit {name} axis[{item_index}]: {old_text} -> {new_text}"
        )
        if reordered:
            history_entry += "; stable-sorted axis and sample arrays"
        prior_history = str(self.history or "").strip()
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )
        return RcsGrid(
            axes[0],
            axes[1],
            axes[2],
            axes[3],
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=copy.deepcopy(self.units),
            extra=edited_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @staticmethod
    def _canonical_unit(value, aliases, default):
        text = str(value or default).strip().lower()
        return aliases.get(text, text)

    def _supported_unit(self, axis_name, aliases, default):
        """Return a canonical unit; use the supplied default for a missing value.

        Reject explicit unknown units.
        """

        raw = (self.units or {}).get(axis_name)
        canonical = self._canonical_unit(raw, aliases, default)
        supported = set(aliases.values())
        if canonical not in supported:
            raise ValueError(
                f"unsupported {axis_name} unit {raw!r}; expected one of "
                + ", ".join(sorted(supported))
            )
        return canonical

    def _angle_value_from_degrees(self, value, axis_name):
        """Convert a degree-valued operation argument to an axis's unit."""

        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{axis_name} angle must be finite")
        unit = self._supported_unit(axis_name, _ANGLE_UNITS, "deg")
        return float(np.deg2rad(numeric)) if unit == "rad" else numeric

    def align_to(self, other, mode="exact"):
        """Align this grid to another grid's axes.

        Modes:
            exact: require identical axes (returns self on success).
            intersect: keep only axis values present in both grids.
            interp: interpolate numeric axes to match other (no extrapolation).

        Args:
            other: Another RcsGrid instance.
            mode: "exact", "intersect", or "interp".

        Returns:
            New RcsGrid aligned to other's axes.
        """
        from GRIM_Backend.datasets.grid import RcsGrid
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        self._assert_axis_metadata_compatible(other)

        if mode == "exact":
            if self.rcs_power.shape != other.rcs_power.shape:
                raise ValueError(
                    f"rcs shape {other.rcs_power.shape} != {self.rcs_power.shape}"
                )
            for name, left, right in (
                ("azimuth", self.azimuths, other.azimuths),
                ("elevation", self.elevations, other.elevations),
                ("frequency", self.frequencies, other.frequencies),
                ("polarization", self.polarizations, other.polarizations),
            ):
                if not np.array_equal(left, right):
                    raise ValueError(f"{name} axis mismatch")
            return self
        if mode not in ("intersect", "interp"):
            raise ValueError("mode must be 'exact', 'intersect', or 'interp'")

        if mode == "intersect":
            def _match_axis(axis_self, axis_other, tol=1e-6):
                axis_self = np.asarray(axis_self).ravel()
                axis_other = np.asarray(axis_other).ravel()
                _common, matched = self._common_axis_alignment(
                    (axis_self, axis_other), tol=tol
                )
                indices_self = [int(value) for value in matched[0]]
                indices_other = [int(value) for value in matched[1]]
                if not indices_self:
                    raise ValueError("no overlapping axis values for intersect")


                target_source_pairs = sorted(
                    zip(indices_other, indices_self), key=lambda pair: pair[0]
                )
                target_indices = [pair[0] for pair in target_source_pairs]
                source_indices = [pair[1] for pair in target_source_pairs]
                return axis_other[target_indices], source_indices

            az_unit = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
            el_unit = self._supported_unit("elevation", _ANGLE_UNITS, "deg")
            frequency_unit = self._supported_unit(
                "frequency", _FREQUENCY_UNITS, "GHz"
            )
            az_tol = float(np.deg2rad(1.0e-6)) if az_unit == "rad" else 1.0e-6
            el_tol = float(np.deg2rad(1.0e-6)) if el_unit == "rad" else 1.0e-6


            f_tol = {
                "Hz": 1.0e3,
                "kHz": 1.0,
                "MHz": 1.0e-3,
                "GHz": 1.0e-6,
            }[frequency_unit]
            az_new, az_idx = _match_axis(
                self.azimuths, other.azimuths, tol=az_tol
            )
            el_new, el_idx = _match_axis(
                self.elevations, other.elevations, tol=el_tol
            )
            f_new, f_idx = _match_axis(
                self.frequencies, other.frequencies, tol=f_tol
            )
            pol_new, pol_idx = _match_axis(self.polarizations, other.polarizations, tol=0.0)
            pwr_new = self.rcs_power[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            phs_new = self.rcs_phase[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            selection = np.ix_(az_idx, el_idx, f_idx, pol_idx)
            source_axes = (
                np.asarray(self.azimuths)[az_idx],
                np.asarray(self.elevations)[el_idx],
                np.asarray(self.frequencies)[f_idx],
                np.asarray(self.polarizations)[pol_idx],
            )
            relabeled = not all(
                np.array_equal(source, target)
                for source, target in zip(
                    source_axes, (az_new, el_new, f_new, pol_new)
                )
            )
            return self._new_grid(
                az_new,
                el_new,
                f_new,
                pol_new,
                rcs_power=pwr_new,
                rcs_phase=phs_new,
                rcs_domain="power_phase",
                extra=self._exact_transform_extra(
                    lambda value: value[selection],
                    coordinate_change=("align-intersect-relabel" if relabeled else None),
                ),
            )


        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch for interp")

        self._check_axis_sorted(self.azimuths, "azimuth")
        self._check_axis_sorted(self.elevations, "elevation")
        self._check_axis_sorted(self.frequencies, "frequency")
        self._check_axis_sorted(other.azimuths, "azimuth")
        self._check_axis_sorted(other.elevations, "elevation")
        self._check_axis_sorted(other.frequencies, "frequency")


        power_interp = np.asarray(self.rcs_power)
        complex_interp = np.asarray(self.rcs, dtype=np.complex128)
        for axis, old, new in (
            (0, self.azimuths, other.azimuths),
            (1, self.elevations, other.elevations),
            (2, self.frequencies, other.frequencies),
        ):
            power_interp = self._interp_real_axis(
                power_interp, old, new, axis
            )
            complex_interp = self._interp_complex_axis(
                complex_interp, old, new, axis
            )
        complex_valid = np.isfinite(complex_interp.real) & np.isfinite(
            complex_interp.imag
        )
        phase_interp = np.full(power_interp.shape, np.nan, dtype=np.float64)
        power_interp = np.asarray(power_interp, dtype=np.float64)
        power_interp[complex_valid] = np.abs(complex_interp[complex_valid]) ** 2
        phase_interp[complex_valid] = np.angle(complex_interp[complex_valid])
        interp_extra = self._derived_response_extra(
            operation="align-interpolate", coherent=True
        )
        self._invalidate_assembly_sampling_hash(
            interp_extra, "align-interpolate"
        )
        return self._new_grid(
            other.azimuths,
            other.elevations,
            other.frequencies,
            other.polarizations,
            rcs_power=power_interp,
            rcs_phase=phase_interp,
            rcs_domain="power_phase",
            extra=interp_extra,
        )

    @staticmethod
    def _check_axis_sorted(axis, name):
        axis = np.asarray(axis)
        if axis.size < 2:
            return
        if not np.all(np.diff(axis) > 0):
            raise ValueError(f"{name} axis must be strictly increasing for interp")

    @staticmethod
    def _interp_complex_axis(data, x_old, x_new, axis):
        from GRIM_Backend.datasets.grid import RcsGrid
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.min() < x_old.min() or x_new.max() > x_old.max():
            raise ValueError("interp would require extrapolation")
        return RcsGrid._interp_linear_axis(data, x_old, x_new, axis)

    @staticmethod
    def _interp_real_axis(data, x_old, x_new, axis):
        from GRIM_Backend.datasets.grid import RcsGrid
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.min() < x_old.min() or x_new.max() > x_old.max():
            raise ValueError("interp would require extrapolation")
        return RcsGrid._interp_linear_axis(data, x_old, x_new, axis)

    @staticmethod
    def _interp_linear_axis(data, x_old, x_new, axis):
        """Vectorized adjacent-bin interpolation; NaNs remain local."""
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.min() < x_old.min() or x_new.max() > x_old.max():
            raise ValueError("interp would require extrapolation")
        moved = np.moveaxis(np.asarray(data), axis, 0)
        right = np.searchsorted(x_old, x_new, side="left")
        right = np.clip(right, 0, len(x_old) - 1)
        exact = x_old[right] == x_new
        left = np.where(exact, right, np.maximum(right - 1, 0))
        denom = x_old[right] - x_old[left]
        weight = np.divide(
            x_new - x_old[left],
            denom,
            out=np.zeros_like(x_new, dtype=float),
            where=denom != 0.0,
        )
        reshape = (len(x_new),) + (1,) * (moved.ndim - 1)
        w = weight.reshape(reshape)
        out = moved[left] * (1.0 - w) + moved[right] * w
        return np.moveaxis(out.astype(moved.dtype, copy=False), 0, axis)

    def interpolate_axis(self, axis_name, new_values):
        """Linearly interpolate the grid onto new values along one numeric axis.

        Other axes are left unchanged. Raises if `new_values` extends beyond
        the existing axis range (no extrapolation).
        """
        axis_map = {"azimuth": 0, "elevation": 1, "frequency": 2}
        key = str(axis_name).strip().lower()
        if key not in axis_map:
            raise ValueError(f"axis must be one of {list(axis_map)}")
        axis_idx = axis_map[key]
        new_arr = np.asarray(new_values, dtype=float).ravel()
        if new_arr.size == 0:
            raise ValueError("new axis must have at least one value")
        if new_arr.size > 1 and not np.all(np.diff(new_arr) > 0):
            raise ValueError("new axis must be strictly increasing")

        old_axes = [self.azimuths, self.elevations, self.frequencies]
        self._check_axis_sorted(old_axes[axis_idx], key)

        new_axes = list(old_axes)
        new_axes[axis_idx] = new_arr

        power_interp = self._interp_real_axis(
            self.rcs_power, old_axes[axis_idx], new_arr, axis_idx
        )
        complex_interp = self._interp_complex_axis(
            np.asarray(self.rcs, dtype=np.complex128),
            old_axes[axis_idx],
            new_arr,
            axis_idx,
        )
        complex_valid = np.isfinite(complex_interp.real) & np.isfinite(
            complex_interp.imag
        )
        power_interp = np.asarray(power_interp, dtype=np.float64)
        phase_interp = np.full(power_interp.shape, np.nan, dtype=np.float64)
        power_interp[complex_valid] = np.abs(complex_interp[complex_valid]) ** 2
        phase_interp[complex_valid] = np.angle(complex_interp[complex_valid])
        interp_extra = self._derived_response_extra(
            operation=f"interpolate-{key}", coherent=True
        )
        self._invalidate_assembly_sampling_hash(
            interp_extra, f"interpolate-{key}"
        )
        return self._new_grid(
            new_axes[0],
            new_axes[1],
            new_axes[2],
            self.polarizations,
            rcs_power=power_interp,
            rcs_phase=phase_interp,
            rcs_domain="power_phase",
            extra=interp_extra,
        )

    @staticmethod
    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def _clean_axis(axis):
        """Normalize an axis to float64 (numeric) or keep dtype (non-numeric).

        For float32 input, round-trips each value through its shortest-decimal
        repr so that user-intended values like 0.1 stay as 0.1 in float64
        instead of inheriting the float32 quantization noise (0.10000000149...).
        That way later ops like `shift_azimuth(180)` produce clean values
        (180.1 instead of 180.10000001).
        """
        arr = np.asarray(axis)
        if not np.issubdtype(arr.dtype, np.number):
            return arr
        if arr.dtype == np.float32:
            return arr.astype(str).astype(np.float64)
        return arr.astype(np.float64, copy=False)

    @staticmethod
    def _canonical_polarization_axis(polarizations):
        """Return stripped uppercase polarization labels.

        Reject blank labels and channels that share a case-insensitive identity.
        """

        values = np.asarray(polarizations)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "polarizations must be a nonempty one-dimensional string axis"
            )
        labels = []
        for raw_value in values.tolist():
            if isinstance(raw_value, bytes):
                try:
                    label = raw_value.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("polarization labels must be UTF-8 strings") from exc
            elif isinstance(raw_value, (str, np.str_)):
                label = str(raw_value)
            else:
                raise ValueError(
                    f"polarization labels must be strings; got {raw_value!r}"
                )
            label = label.strip().upper()
            if not label:
                raise ValueError("polarization labels must not be blank")
            labels.append(label)
        if len(set(labels)) != len(labels):
            raise ValueError(
                "polarization labels must be unique after case normalization"
            )

        return np.asarray(labels, dtype=str)

    @staticmethod
    def _axis_value_match(axis_arr, value, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        if np.issubdtype(axis_arr.dtype, np.number) and isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return np.where(np.isclose(axis_arr, float(value), atol=tol, rtol=0.0))[0]
        return np.where(axis_arr == value)[0]

    @staticmethod
    def _indices_for_axis_values(axis_arr, values, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        values_arr = np.asarray(values)
        if values_arr.size == 0:
            return []
        if axis_arr.size == 0:
            return None
        if np.issubdtype(axis_arr.dtype, np.number) and np.issubdtype(
            values_arr.dtype, np.number
        ):
            axis_f = axis_arr.astype(float, copy=False).ravel()
            values_f = values_arr.astype(float, copy=False).ravel()
            order = np.argsort(axis_f, kind="stable")
            sorted_axis = axis_f[order]
            pos = np.searchsorted(sorted_axis, values_f)
            n = sorted_axis.size
            left = np.clip(pos - 1, 0, n - 1)
            right = np.clip(pos, 0, n - 1)
            d_left = np.abs(sorted_axis[left] - values_f)
            d_right = np.abs(sorted_axis[right] - values_f)
            use_right = d_right <= d_left
            sorted_idx = np.where(use_right, right, left)
            dist = np.where(use_right, d_right, d_left)
            if np.any(dist > tol):
                return None
            orig_idx = order[sorted_idx]
            seen = set()
            out = []
            for i in orig_idx.tolist():
                if i not in seen:
                    seen.add(i)
                    out.append(i)
            return out
        idx_map = {}
        for i in range(axis_arr.size):
            v = axis_arr[i]
            key = v.item() if isinstance(v, np.generic) else v
            if key not in idx_map:
                idx_map[key] = i
        seen = set()
        out = []
        for value in values_arr:
            key = value.item() if isinstance(value, np.generic) else value
            if key not in idx_map:
                return None
            idx = idx_map[key]
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out

    @staticmethod
    def _axis_union(axis_arrays, tol=1e-6):
        if not axis_arrays:
            return np.asarray([])
        first_dtype = np.asarray(axis_arrays[0]).dtype
        numeric_axis = np.issubdtype(first_dtype, np.number)
        if not numeric_axis:
            seen = {}
            for axis_arr in axis_arrays:
                for value in np.asarray(axis_arr):
                    key = value.item() if isinstance(value, np.generic) else value
                    if key not in seen:
                        seen[key] = None
            return np.asarray(list(seen))
        parts = [np.asarray(a, dtype=float).ravel() for a in axis_arrays]
        combined = np.concatenate(parts) if parts else np.asarray([], dtype=float)
        if combined.size == 0:
            return np.asarray([])
        combined.sort(kind="mergesort")
        keep = np.ones(combined.size, dtype=bool)
        if tol <= 0:
            keep[1:] = combined[1:] != combined[:-1]
        else:
            last_kept = combined[0]
            for i in range(1, combined.size):
                if combined[i] - last_kept > tol:
                    last_kept = combined[i]
                else:
                    keep[i] = False
        return combined[keep]

    @staticmethod
    def _common_axis_alignment(axis_arrays, tol=1e-6):
        """Return a symmetric axis intersection and indices into every input.

        Numeric values match only when one value from every axis fits in a
        window no wider than ``tol``.  The lowest value in that window is the
        canonical output coordinate.  Sorting the values before matching makes
        both the coordinates and the chosen samples independent of which
        dataset happened to be selected first.
        """
        arrays = [np.asarray(axis).ravel() for axis in axis_arrays]
        if not arrays:
            return np.asarray([]), []

        try:
            tol = float(tol)
        except (TypeError, ValueError) as exc:
            raise ValueError("tol must be a finite nonnegative number") from exc
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be a finite nonnegative number")

        numeric_flags = [np.issubdtype(array.dtype, np.number) for array in arrays]
        if any(numeric_flags) and not all(numeric_flags):
            raise TypeError("axis inputs must all be numeric or all be nonnumeric")

        if all(numeric_flags):
            sorted_values = []
            sorted_indices = []
            for array in arrays:
                values = array.astype(float, copy=False)
                finite_indices = np.flatnonzero(np.isfinite(values))
                order = np.argsort(values[finite_indices], kind="stable")
                original_indices = finite_indices[order]
                sorted_values.append(values[original_indices])
                sorted_indices.append(original_indices)

            if any(values.size == 0 for values in sorted_values):
                return np.asarray([], dtype=float), [[] for _ in arrays]

            pointers = np.zeros(len(arrays), dtype=np.int64)
            common_values = []
            matched_indices = [[] for _ in arrays]
            while all(
                pointer < values.size
                for pointer, values in zip(pointers, sorted_values)
            ):
                current = np.asarray(
                    [values[pointer] for values, pointer in zip(sorted_values, pointers)],
                    dtype=float,
                )
                low = float(np.min(current))
                high = float(np.max(current))
                if high - low <= tol:
                    common_values.append(low)
                    for axis_idx in range(len(arrays)):
                        matched_indices[axis_idx].append(
                            int(sorted_indices[axis_idx][pointers[axis_idx]])
                        )
                        pointers[axis_idx] += 1
                    continue


                for axis_idx, value in enumerate(current):
                    if value == low:
                        pointers[axis_idx] += 1

            return np.asarray(common_values, dtype=float), matched_indices

        indices_by_value = []
        for array in arrays:
            mapping = {}
            for index, raw_value in enumerate(array):
                value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
                mapping.setdefault(value, []).append(index)
            indices_by_value.append(mapping)

        common_values = set(indices_by_value[0])
        for mapping in indices_by_value[1:]:
            common_values.intersection_update(mapping)

        def _canonical_key(value):
            value_type = type(value)
            return (value_type.__module__, value_type.__qualname__, repr(value))

        output_values = []
        matched_indices = [[] for _ in arrays]
        for value in sorted(common_values, key=_canonical_key):
            occurrences = min(len(mapping[value]) for mapping in indices_by_value)
            output_values.extend([value] * occurrences)
            for axis_idx, mapping in enumerate(indices_by_value):
                matched_indices[axis_idx].extend(mapping[value][:occurrences])
        return np.asarray(output_values), matched_indices

    @classmethod
    def _axis_intersection(cls, axis_arrays, tol=1e-6):
        common, _indices = cls._common_axis_alignment(axis_arrays, tol=tol)
        return common

    @classmethod
    def _ensure_grids(cls, grids):
        checked = []
        for grid in grids:
            if not isinstance(grid, cls):
                raise TypeError("all inputs must be RcsGrid instances")
            checked.append(grid)
        if not checked:
            raise ValueError("at least one grid is required")
        return checked

    def axis_crop(
        self,
        *,
        azimuths=None,
        elevations=None,
        frequencies=None,
        polarizations=None,
        azimuth_range=None,
        elevation_range=None,
        frequency_range=None,
        azimuth_min=None,
        azimuth_max=None,
        elevation_min=None,
        elevation_max=None,
        frequency_min=None,
        frequency_max=None,
        tol=1e-6,
    ):
        """Return a grid cropped by explicit axis values and/or numeric ranges."""

        def _resolve_range(raw_range, vmin, vmax):
            if raw_range is not None:
                if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                    raise ValueError("axis range must be a 2-item [min, max] sequence")
                return raw_range[0], raw_range[1]
            if vmin is None and vmax is None:
                return None
            return vmin, vmax

        azimuth_range = _resolve_range(azimuth_range, azimuth_min, azimuth_max)
        elevation_range = _resolve_range(elevation_range, elevation_min, elevation_max)
        frequency_range = _resolve_range(frequency_range, frequency_min, frequency_max)

        def _axis_indices(axis_arr, axis_values, axis_range, axis_name, axis_tol):
            all_indices = list(range(len(axis_arr)))
            values = self._as_list(axis_values)
            if values is not None:
                if axis_name == "polarization":
                    values = [str(value).strip().upper() for value in values]
                selected = self._indices_for_axis_values(axis_arr, values, tol=axis_tol)
                if selected is None:
                    raise ValueError(f"{axis_name} contains value(s) not present in dataset")
                indices = selected
            else:
                indices = all_indices

            if axis_range is not None:
                lo, hi = axis_range
                if lo is not None:
                    lo = float(lo)
                if hi is not None:
                    hi = float(hi)
                if lo is not None and hi is not None and lo > hi:
                    lo, hi = hi, lo

                axis_num = np.asarray(axis_arr, dtype=float)
                range_mask = np.ones(axis_num.shape[0], dtype=bool)
                if lo is not None:
                    range_mask &= axis_num >= (lo - axis_tol)
                if hi is not None:
                    range_mask &= axis_num <= (hi + axis_tol)
                range_idx = set(np.where(range_mask)[0].tolist())
                indices = [idx for idx in indices if idx in range_idx]

            if not indices:
                raise ValueError(f"{axis_name} crop produced no samples")
            return indices

        az_idx = _axis_indices(self.azimuths, azimuths, azimuth_range, "azimuth", tol)
        el_idx = _axis_indices(self.elevations, elevations, elevation_range, "elevation", tol)
        f_idx = _axis_indices(self.frequencies, frequencies, frequency_range, "frequency", tol)
        p_idx = _axis_indices(self.polarizations, polarizations, None, "polarization", 0.0)
        selection = np.ix_(az_idx, el_idx, f_idx, p_idx)

        return self._new_grid(
            self.azimuths[az_idx],
            self.elevations[el_idx],
            self.frequencies[f_idx],
            self.polarizations[p_idx],
            rcs_power=self.rcs_power[np.ix_(az_idx, el_idx, f_idx, p_idx)],
            rcs_phase=self.rcs_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)],
            extra=self._exact_transform_extra(
                lambda value: value[selection]
            ),
        )

    @staticmethod
    def _merge_equivalent_sample_blocks(
        existing_power,
        existing_phase,
        incoming_power,
        incoming_phase,
        *,
        context,
    ):
        """Merge complementary/equivalent samples or reject a seam conflict."""

        existing_power = np.asarray(existing_power)
        existing_phase = np.asarray(existing_phase)
        incoming_power = np.asarray(incoming_power)
        incoming_phase = np.asarray(incoming_phase)
        existing_finite = np.isfinite(existing_power)
        incoming_finite = np.isfinite(incoming_power)
        both = existing_finite & incoming_finite
        power_equal = both & np.isclose(
            existing_power, incoming_power, rtol=1.0e-6, atol=1.0e-12
        )
        power_conflict = both & ~power_equal

        both_phase = (
            power_equal
            & np.isfinite(existing_phase)
            & np.isfinite(incoming_phase)
        )
        both_zero = both & (existing_power == 0.0) & (incoming_power == 0.0)
        phase_delta = np.abs(
            np.angle(np.exp(1j * (existing_phase - incoming_phase)))
        )
        phase_conflict = both_phase & ~both_zero & (phase_delta > 1.0e-5)
        if np.any(power_conflict) or np.any(phase_conflict):
            raise ValueError(
                f"{context}: conflicting finite seam samples would overlap"
            )

        take_power = ~existing_finite & incoming_finite
        if np.any(take_power):
            existing_power[take_power] = incoming_power[take_power]
            existing_phase[take_power] = np.where(
                np.isfinite(incoming_phase[take_power]),
                incoming_phase[take_power],
                np.nan,
            )
        fill_phase = (
            power_equal
            & ~np.isfinite(existing_phase)
            & np.isfinite(incoming_phase)
        )
        existing_phase[fill_phase] = incoming_phase[fill_phase]

    @staticmethod
    def _merge_equivalent_raw_blocks(
        existing_real,
        existing_imag,
        incoming_real,
        incoming_imag,
        *,
        context,
    ):
        """Merge authoritative float64 seam fields or reject hidden conflicts."""

        existing_real = np.asarray(existing_real)
        existing_imag = np.asarray(existing_imag)
        incoming_real = np.asarray(incoming_real)
        incoming_imag = np.asarray(incoming_imag)
        existing_finite = np.isfinite(existing_real) & np.isfinite(existing_imag)
        incoming_finite = np.isfinite(incoming_real) & np.isfinite(incoming_imag)
        both = existing_finite & incoming_finite
        equivalent = (
            np.isclose(
                existing_real,
                incoming_real,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            & np.isclose(
                existing_imag,
                incoming_imag,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        )
        if np.any(both & ~equivalent):
            raise ValueError(
                f"{context}: conflicting authoritative raw seam samples would overlap"
            )
        take = ~existing_finite & incoming_finite
        existing_real[take] = incoming_real[take]
        existing_imag[take] = incoming_imag[take]

    def mirror_about_azimuth(self, azimuth_deg: float):
        """Mirror azimuth axis about a reference angle and return a new grid.

        The transformed axis is `az' = 2*azimuth_deg - az`. Output azimuths are
        sorted ascending, with samples reordered to match.
        """
        about = self._angle_value_from_degrees(azimuth_deg, "azimuth")

        az = np.asarray(self.azimuths, dtype=float)
        mirrored_az = (2.0 * about) - az
        order = np.argsort(mirrored_az, kind="stable")

        return self._new_grid(
            mirrored_az[order],
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=self.rcs_power[order, :, :, :],
            rcs_phase=self.rcs_phase[order, :, :, :],
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                lambda value: np.take(value, order, axis=0),
                coordinate_change="mirror-azimuth",
                preserve_angular_contract=False,
            ),
        )

    def swap_elevation_azimuth(self):
        """Swap the elevation and azimuth axes and return a new grid."""
        swapped_units = copy.deepcopy(self.units)
        azimuth_unit = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
        elevation_unit = self._supported_unit("elevation", _ANGLE_UNITS, "deg")
        swapped_units["azimuth"] = elevation_unit
        swapped_units["elevation"] = azimuth_unit
        return self._new_grid(
            np.array(self.elevations, copy=True),
            np.array(self.azimuths, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.swapaxes(self.rcs_power, 0, 1).copy(),
            rcs_phase=np.swapaxes(self.rcs_phase, 0, 1).copy(),
            rcs_domain="power_phase",
            units=swapped_units,
            extra=self._exact_transform_extra(
                lambda value: np.swapaxes(value, 0, 1),
                coordinate_change="swap-elevation-azimuth",
                preserve_angular_contract=False,
            ),
        )

    def convert_axis_units(
        self,
        *,
        azimuth="deg",
        elevation="deg",
        frequency="GHz",
    ):
        """Convert numeric-axis storage units without changing physical samples."""

        target_az = self._canonical_unit(azimuth, _ANGLE_UNITS, "deg")
        target_el = self._canonical_unit(elevation, _ANGLE_UNITS, "deg")
        target_frequency = self._canonical_unit(
            frequency, _FREQUENCY_UNITS, "GHz"
        )
        source_az = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
        source_el = self._supported_unit("elevation", _ANGLE_UNITS, "deg")
        source_frequency = self._supported_unit(
            "frequency", _FREQUENCY_UNITS, "GHz"
        )

        def convert_angle(values, source, target):
            values = np.asarray(values, dtype=float)
            degrees = np.rad2deg(values) if source == "rad" else values
            return np.deg2rad(degrees) if target == "rad" else degrees.copy()

        frequency_to_hz = {
            "Hz": 1.0,
            "kHz": 1.0e3,
            "MHz": 1.0e6,
            "GHz": 1.0e9,
        }
        frequency_hz = (
            np.asarray(self.frequencies, dtype=float)
            * frequency_to_hz[source_frequency]
        )
        converted_frequency = frequency_hz / frequency_to_hz[target_frequency]
        converted_units = copy.deepcopy(self.units or {})
        converted_units.update(
            azimuth=target_az,
            elevation=target_el,
            frequency=target_frequency,
        )
        history_entry = (
            "Convert axis storage units without interpolation: "
            f"azimuth {source_az}->{target_az}, elevation {source_el}->{target_el}, "
            f"frequency {source_frequency}->{target_frequency}"
        )
        history = (
            f"{self.history}\n{history_entry}" if self.history else history_entry
        )
        return self._new_grid(
            convert_angle(self.azimuths, source_az, target_az),
            convert_angle(self.elevations, source_el, target_el),
            converted_frequency,
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            units=converted_units,
            history=history,
            extra=self._exact_transform_extra(
                coordinate_change="convert-axis-storage-units"
            ),
        )

    def shift_azimuth(self, delta_deg: float):
        """Shift azimuth axis by a constant offset and return a new grid."""
        delta = self._angle_value_from_degrees(delta_deg, "azimuth")
        shifted_az = np.asarray(self.azimuths, dtype=float) + delta
        return self._new_grid(
            shifted_az,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="shift-azimuth",
                preserve_angular_contract=False,
            ),
        )

    def wrap_azimuth(self, mode: str):
        """Wrap azimuth axis into the given range and return a new grid.

        ``mode`` is ``"0_360"`` for [0, 360) or ``"-180_180"`` for [-180, 180).
        Output azimuths are sorted ascending; samples are reordered to match.
        Degree axes use 360/180 and radian axes use 2*pi/pi.  If wrapping
        collapses distinct inputs onto one seam coordinate, complementary or
        equivalent samples are merged; conflicting finite samples are rejected
        instead of silently discarding one.
        """
        az = np.asarray(self.azimuths, dtype=float)
        unit = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
        period = (2.0 * np.pi) if unit == "rad" else 360.0
        half_period = 0.5 * period
        seam_tol = float(np.deg2rad(1.0e-9)) if unit == "rad" else 1.0e-9
        if mode == "0_360":
            wrapped = np.mod(az, period)
            wrapped[np.isclose(wrapped, period, atol=seam_tol, rtol=0.0)] = 0.0
            wrapped[np.isclose(wrapped, 0.0, atol=seam_tol, rtol=0.0)] = 0.0
        elif mode == "-180_180":
            wrapped = np.mod(az + half_period, period) - half_period
            wrapped[
                np.isclose(wrapped, half_period, atol=seam_tol, rtol=0.0)
                | np.isclose(wrapped, -half_period, atol=seam_tol, rtol=0.0)
            ] = -half_period
        else:
            raise ValueError(f"unknown wrap mode: {mode!r}")

        order = np.argsort(wrapped, kind="stable")
        groups = []
        for source_index in order.tolist():
            if not groups or (
                wrapped[source_index] - wrapped[groups[-1][0]] > seam_tol
            ):
                groups.append([source_index])
            else:
                groups[-1].append(source_index)
        unique_vals = np.asarray(
            [wrapped[group[0]] for group in groups], dtype=float
        )
        output_shape = (len(groups),) + self.rcs_power.shape[1:]
        output_power = np.full(output_shape, np.nan, dtype=self.rcs_power.dtype)
        output_phase = np.full(output_shape, np.nan, dtype=self.rcs_phase.dtype)
        raw_pair = self._complete_authoritative_raw_arrays()
        preserve_raw = raw_pair is not None
        if preserve_raw:
            raw_real = np.asarray(raw_pair[0], dtype=np.float64)
            raw_imag = np.asarray(raw_pair[1], dtype=np.float64)
            output_raw_real = np.full(output_shape, np.nan, dtype=np.float64)
            output_raw_imag = np.full(output_shape, np.nan, dtype=np.float64)
        for output_index, group in enumerate(groups):
            for source_index in group:
                context = (
                    f"azimuth wrap at {unique_vals[output_index]:.12g} {unit}"
                )
                self._merge_equivalent_sample_blocks(
                    output_power[output_index],
                    output_phase[output_index],
                    self.rcs_power[source_index],
                    self.rcs_phase[source_index],
                    context=context,
                )
                if preserve_raw:
                    self._merge_equivalent_raw_blocks(
                        output_raw_real[output_index],
                        output_raw_imag[output_index],
                        raw_real[source_index],
                        raw_imag[source_index],
                        context=context,
                    )
        if preserve_raw:
            unmodeled = ~np.isfinite(output_power)
            output_raw_real[unmodeled] = np.nan
            output_raw_imag[unmodeled] = np.nan
        wrapped_extra = self._exact_transform_extra(
            coordinate_change="wrap-azimuth", preserve_raw=False
        )
        if preserve_raw:
            wrapped_extra["rcs_amp_real"] = output_raw_real
            wrapped_extra["rcs_amp_imag"] = output_raw_imag
            wrapped_extra["raw_complex_amplitude_preserved"] = True
        return self._new_grid(
            unique_vals,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=output_power,
            rcs_phase=output_phase,
            rcs_domain="power_phase",
            extra=wrapped_extra,
        )

    def wrap_phase(self, mode: str):
        """Wrap stored phase while preserving power and the complex field.

        ``mode`` is ``"0_360"`` for [0, 360) degrees or ``"-180_180"``
        for [-180, 180) degrees.  Phase is stored in radians, so only a
        modulo-2*pi representation change is made.  Missing phase remains
        missing and power samples are copied without modification.
        """

        if mode not in {"0_360", "-180_180"}:
            raise ValueError(
                "phase wrap mode must be '0_360' or '-180_180'"
            )

        wrapped_phase = np.array(self.rcs_phase, copy=True)
        period = 2.0 * np.pi
        with np.errstate(invalid="ignore"):
            if mode == "0_360":
                np.remainder(wrapped_phase, period, out=wrapped_phase)
                range_label = "[0, 360) deg"
            else:
                np.add(wrapped_phase, np.pi, out=wrapped_phase)
                np.remainder(wrapped_phase, period, out=wrapped_phase)
                np.subtract(wrapped_phase, np.pi, out=wrapped_phase)
                range_label = "[-180, 180) deg"

        history_entry = f"Wrap phase to {range_label}; complex field unchanged"
        prior_history = str(self.history or "").strip()
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )
        wrapped_units = dict(self.units)
        wrapped_units["phase_wrap"] = mode
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=wrapped_phase,
            history=history,
            units=wrapped_units,
            extra=self._exact_transform_extra(preserve_all=True),
        )

    def round_azimuths(self, decimals: int):
        """Round azimuth axis values to ``decimals`` decimal places (no resampling).

        Use to clean up floating-point noise like 180.0001 -> 180.0.
        Raises if rounding collapses two distinct azimuths into the same value.
        """
        decimals = int(decimals)
        rounded = np.round(np.asarray(self.azimuths, dtype=float), decimals)
        if rounded.size != np.unique(rounded).size:
            raise ValueError(
                f"Rounding azimuths to {decimals} decimal(s) would create duplicate "
                "values. Use a higher decimal count."
            )
        return self._new_grid(
            rounded,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="round-azimuth"
            ),
        )

    def round_elevations(self, decimals: int):
        """Round elevation axis values to ``decimals`` decimal places (no resampling)."""
        decimals = int(decimals)
        rounded = np.round(np.asarray(self.elevations, dtype=float), decimals)
        if rounded.size != np.unique(rounded).size:
            raise ValueError(
                f"Rounding elevations to {decimals} decimal(s) would create duplicate "
                "values. Use a higher decimal count."
            )
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            rounded,
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="round-elevation"
            ),
        )

    def round_frequencies(self, decimals: int):
        """Round frequency axis values to ``decimals`` decimal places (no resampling)."""
        decimals = int(decimals)
        rounded = np.round(np.asarray(self.frequencies, dtype=float), decimals)
        if rounded.size != np.unique(rounded).size:
            raise ValueError(
                f"Rounding frequencies to {decimals} decimal(s) would create duplicate "
                "values. Use a higher decimal count."
            )
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            rounded,
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="round-frequency"
            ),
        )

    def shift_elevation(self, delta_deg: float):
        """Shift elevation axis by a constant offset and return a new grid."""
        delta = self._angle_value_from_degrees(delta_deg, "elevation")
        shifted_el = np.asarray(self.elevations, dtype=float) + delta
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            shifted_el,
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="shift-elevation",
                preserve_angular_contract=False,
            ),
        )
