"""Physics-explicit basis-pursuit dictionary editing foundations for ISAR.

This module intentionally has no GUI "remove" button.  It provides the pieces
that must be validated first: named component dictionaries, an implicit
concatenated operator, a true residual-constrained BPDN solver, component phase
histories, residuals, and an identifiability check.  A component name is not a
classifier; target/support/cavity separation is only meaningful when their
forward dictionaries are physically justified and distinguishable.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Iterable, Sequence

import numpy as np


_C0 = 299_792_458.0


def _finite_complex_vector(values, expected_size: int, label: str) -> np.ndarray:
    """Return one finite complex vector or fail at the operator boundary."""

    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    if vector.size != int(expected_size):
        raise ValueError(
            f"{label} size {vector.size} does not match {int(expected_size)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite complex values")
    return vector


def _finite_real_weights(values, expected_size: int) -> np.ndarray:
    weights = np.asarray(values, dtype=float).reshape(-1)
    if weights.size != int(expected_size):
        raise ValueError("noise-whitening weights must match measurements")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("noise-whitening weights must be finite and nonnegative")
    if not np.any(weights > 0.0):
        raise ValueError("at least one noise-whitening weight must be positive")
    return weights


def _stable_vector_norm(values: np.ndarray) -> float:
    """Euclidean norm without avoidable underflow from arbitrary scaling."""

    magnitudes = np.abs(np.asarray(values, dtype=np.complex128).reshape(-1))
    if not np.all(np.isfinite(magnitudes)):
        return float("nan")
    scale = float(np.max(magnitudes, initial=0.0))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * np.sqrt(np.sum((magnitudes / scale) ** 2)))


def _stable_column_norms(values: np.ndarray) -> np.ndarray:
    """Column-wise Euclidean norms stable across representable scales."""

    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("dictionary columns must be a finite two-dimensional array")
    magnitudes = np.abs(matrix)
    scale = np.max(magnitudes, axis=0, initial=0.0)
    norms = np.zeros(matrix.shape[1], dtype=float)
    nonzero = scale > 0.0
    if np.any(nonzero):
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = magnitudes[:, nonzero] / scale[None, nonzero]
            norms[nonzero] = scale[nonzero] * np.sqrt(
                np.sum(normalized * normalized, axis=0)
            )
    return norms


def _complex_soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError("BPDN threshold input contains nonfinite iterate values")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("BPDN threshold must be finite and nonnegative")
    magnitude = np.abs(values)
    scale = np.maximum(magnitude - float(threshold), 0.0) / np.maximum(
        magnitude, 1.0e-30
    )
    return values * scale


class ComponentDictionary:
    """Named complex linear operator mapping coefficients to measurements."""

    def __init__(self, name: str, coefficient_count: int, measurement_count: int):
        self.name = str(name).strip()
        self.coefficient_count = int(coefficient_count)
        self.measurement_count = int(measurement_count)
        if not self.name:
            raise ValueError("component dictionary name cannot be empty")
        if self.coefficient_count < 1 or self.measurement_count < 1:
            raise ValueError("component dictionary dimensions must be positive")

    def forward(self, coefficients: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def adjoint(self, measurements: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sampled_columns(self, maximum_columns: int = 256) -> np.ndarray:
        """Return representative finite columns for a conservative screen."""

        count = min(self.coefficient_count, max(1, int(maximum_columns)))
        indices = np.unique(
            np.linspace(0, self.coefficient_count - 1, count).astype(np.int64)
        )
        columns = np.empty(
            (self.measurement_count, indices.size), dtype=np.complex128
        )
        basis = np.zeros(self.coefficient_count, dtype=np.complex128)
        for column, index in enumerate(indices):
            basis.fill(0.0)
            basis[int(index)] = 1.0
            columns[:, column] = _finite_complex_vector(
                self.forward(basis),
                self.measurement_count,
                f"{self.name} dictionary column {int(index)}",
            )
        return columns

    def weighted_column_norms(self, weights: np.ndarray) -> np.ndarray:
        """Return exact ``||W g_j||_2`` values for every dictionary column.

        The generic implementation is intentionally exhaustive. BPDE remains
        a gated validation foundation, and solving a scale-invariant weighted
        L1 problem requires every physical column scale—not a sampled estimate.
        Concrete dictionaries override this method with block/vectorized forms.
        """

        observed_weights = _finite_real_weights(weights, self.measurement_count)
        norms = np.empty(self.coefficient_count, dtype=float)
        basis = np.zeros(self.coefficient_count, dtype=np.complex128)
        for index in range(self.coefficient_count):
            basis.fill(0.0)
            basis[index] = 1.0
            column = _finite_complex_vector(
                self.forward(basis),
                self.measurement_count,
                f"{self.name} dictionary column {index}",
            )
            norms[index] = _stable_vector_norm(observed_weights * column)
        if not np.all(np.isfinite(norms)):
            raise ValueError(
                f"{self.name} weighted dictionary-column norms must be finite"
            )
        return norms


class DenseComponentDictionary(ComponentDictionary):
    """Dense reference/dictionary matrix, mainly for calibrated small bases."""

    def __init__(self, name: str, matrix: np.ndarray):
        values = np.asarray(matrix)
        if values.ndim != 2:
            raise ValueError("dense component matrix must be two-dimensional")
        self.matrix = np.ascontiguousarray(values, dtype=np.complex128)
        if not np.all(np.isfinite(self.matrix)):
            raise ValueError("dense component matrix must contain only finite values")
        super().__init__(name, self.matrix.shape[1], self.matrix.shape[0])

    def forward(self, coefficients: np.ndarray) -> np.ndarray:
        values = _finite_complex_vector(
            coefficients, self.coefficient_count, f"{self.name} coefficients"
        )
        return _finite_complex_vector(
            self.matrix @ values,
            self.measurement_count,
            f"{self.name} forward output",
        )

    def adjoint(self, measurements: np.ndarray) -> np.ndarray:
        values = _finite_complex_vector(
            measurements, self.measurement_count, f"{self.name} measurements"
        )
        return _finite_complex_vector(
            self.matrix.conj().T @ values,
            self.coefficient_count,
            f"{self.name} adjoint output",
        )

    def sampled_columns(self, maximum_columns: int = 256) -> np.ndarray:
        count = min(self.coefficient_count, max(1, int(maximum_columns)))
        indices = np.unique(
            np.linspace(0, self.coefficient_count - 1, count).astype(np.int64)
        )
        return np.asarray(self.matrix[:, indices], dtype=np.complex128)

    def weighted_column_norms(self, weights: np.ndarray) -> np.ndarray:
        observed_weights = _finite_real_weights(weights, self.measurement_count)
        with np.errstate(over="ignore", invalid="ignore"):
            norms = _stable_column_norms(
                observed_weights[:, None] * self.matrix
            )
        if not np.all(np.isfinite(norms)):
            raise ValueError(
                f"{self.name} weighted dictionary-column norms must be finite"
            )
        return np.asarray(norms, dtype=float)


class PointScattererDictionary(ComponentDictionary):
    """Bounded direct far-field monostatic point-scatterer dictionary.

    The measurement ordering is angle-major, then frequency.  Positions are
    ``(x_cross_range, y_down_range)`` in meters in the same fixed body frame as
    the azimuth angles.  ``visibility`` may be an ``(angle, point)`` real array;
    it is useful for physically reviewed support/cavity masks but must not be
    inferred merely from the desired answer.  This is a small/reference direct
    operator, not a production-scale NUFFT implementation.  Its phase cache is
    explicitly bounded by both a complex-cell budget and a byte budget.
    """

    def __init__(
        self,
        name: str,
        azimuth_radians: Sequence[float],
        frequency_hz: Sequence[float],
        positions_m: np.ndarray,
        *,
        visibility: np.ndarray | None = None,
        maximum_temporary_cells: int = 1_000_000,
        maximum_cached_phase_cells: int = 1_000_000,
        maximum_cached_phase_bytes: int = 64 * 1024 * 1024,
        maximum_cached_phase_blocks: int = 4096,
        maximum_uncached_iteration_cells: int = 250_000_000,
    ):
        self.azimuth_radians = np.array(
            azimuth_radians, dtype=float, copy=True
        ).reshape(-1)
        self.frequency_hz = np.array(
            frequency_hz, dtype=float, copy=True
        ).reshape(-1)
        self.positions_m = np.array(positions_m, dtype=float, copy=True)
        if self.positions_m.ndim != 2 or self.positions_m.shape[1] != 2:
            raise ValueError("point positions must have shape (points, 2)")
        if self.azimuth_radians.size < 1 or self.frequency_hz.size < 1:
            raise ValueError("point dictionary needs angle and frequency samples")
        if not np.all(np.isfinite(self.azimuth_radians)):
            raise ValueError("point-dictionary azimuths must be finite")
        if not np.all(np.isfinite(self.frequency_hz)) or np.any(
            self.frequency_hz <= 0.0
        ):
            raise ValueError("point-dictionary frequencies must be finite and positive")
        if not np.all(np.isfinite(self.positions_m)):
            raise ValueError("point positions must be finite")
        self.visibility = None
        if visibility is not None:
            visible = np.array(visibility, dtype=float, copy=True)
            expected = (self.azimuth_radians.size, self.positions_m.shape[0])
            if visible.shape != expected:
                raise ValueError(f"visibility shape {visible.shape} != {expected}")
            if not np.all(np.isfinite(visible)) or np.any(visible < 0.0):
                raise ValueError("visibility weights must be finite and nonnegative")
            visible.setflags(write=False)
            self.visibility = visible
        self._maximum_temporary_cells = int(maximum_temporary_cells)
        self._maximum_cached_phase_cells = int(maximum_cached_phase_cells)
        self._maximum_cached_phase_bytes = int(maximum_cached_phase_bytes)
        self._maximum_cached_phase_blocks = int(maximum_cached_phase_blocks)
        self._maximum_uncached_iteration_cells = int(
            maximum_uncached_iteration_cells
        )
        if self.maximum_temporary_cells < 1:
            raise ValueError("maximum_temporary_cells must be positive")
        minimum_phase_cells = (
            self.azimuth_radians.size * self.frequency_hz.size
        )
        if self.maximum_temporary_cells < minimum_phase_cells:
            raise ValueError(
                "maximum_temporary_cells must fit at least one complete "
                f"point column ({minimum_phase_cells} complex cells)"
            )
        if self.maximum_cached_phase_cells < 0:
            raise ValueError("maximum_cached_phase_cells cannot be negative")
        if self.maximum_cached_phase_bytes < 0:
            raise ValueError("maximum_cached_phase_bytes cannot be negative")
        if self.maximum_cached_phase_blocks < 0:
            raise ValueError("maximum_cached_phase_blocks cannot be negative")
        if self.maximum_uncached_iteration_cells < 1:
            raise ValueError(
                "maximum_uncached_iteration_cells must be positive"
            )
        complex_bytes = np.dtype(np.complex128).itemsize
        self._phase_cache_capacity_cells = min(
            self.maximum_cached_phase_cells,
            self.maximum_cached_phase_bytes // complex_bytes,
        )
        self._phase_cache: OrderedDict[tuple[int, int], np.ndarray] = (
            OrderedDict()
        )
        self._phase_cache_lock = RLock()
        self._cached_phase_cells = 0
        self._phase_cache_hits = 0
        self._phase_cache_misses = 0
        self._phase_block_computations = 0
        self._sampled_column_computations = 0
        self._directions = np.column_stack(
            (np.sin(self.azimuth_radians), np.cos(self.azimuth_radians))
        )
        self.azimuth_radians.setflags(write=False)
        self.frequency_hz.setflags(write=False)
        self.positions_m.setflags(write=False)
        self._directions.setflags(write=False)
        super().__init__(
            name,
            self.positions_m.shape[0],
            self.azimuth_radians.size * self.frequency_hz.size,
        )

    @property
    def maximum_temporary_cells(self) -> int:
        return self._maximum_temporary_cells

    @property
    def maximum_cached_phase_cells(self) -> int:
        return self._maximum_cached_phase_cells

    @property
    def maximum_cached_phase_bytes(self) -> int:
        return self._maximum_cached_phase_bytes

    @property
    def maximum_cached_phase_blocks(self) -> int:
        return self._maximum_cached_phase_blocks

    @property
    def maximum_uncached_iteration_cells(self) -> int:
        return self._maximum_uncached_iteration_cells

    def _points_per_block(self) -> int:
        cells_per_point = self.measurement_count
        block = max(1, self.maximum_temporary_cells // cells_per_point)
        if self._phase_cache_capacity_cells >= cells_per_point:
            block = min(
                block,
                max(1, self._phase_cache_capacity_cells // cells_per_point),
            )
        return block

    def _point_blocks(self, *, reverse: bool = False):
        block = self._points_per_block()
        if reverse:
            final_start = ((self.coefficient_count - 1) // block) * block
            starts = range(final_start, -1, -block)
        else:
            starts = range(0, self.coefficient_count, block)
        for start in starts:
            yield start, min(start + block, self.coefficient_count)

    def _compute_phase_columns(self, indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if selected.size < 1:
            return np.empty((self.measurement_count, 0), dtype=np.complex128)
        if np.any(selected < 0) or np.any(selected >= self.coefficient_count):
            raise IndexError("point-dictionary column index is out of range")
        positions = self.positions_m[selected]
        projected_range = self._directions @ positions.T
        phase = np.exp(
            -1j
            * (4.0 * np.pi / _C0)
            * self.frequency_hz[None, :, None]
            * projected_range[:, None, :]
        )
        if self.visibility is not None:
            phase *= np.take(self.visibility, selected, axis=1)[:, None, :]
        columns = np.ascontiguousarray(
            phase.reshape(self.measurement_count, selected.size),
            dtype=np.complex128,
        )
        if not np.all(np.isfinite(columns)):
            raise ValueError(
                f"{self.name} point-dictionary columns must remain finite"
            )
        return columns

    def _phase_block(self, start: int, stop: int) -> np.ndarray:
        key = (int(start), int(stop))
        with self._phase_cache_lock:
            cached = self._phase_cache.get(key)
            if cached is not None:
                self._phase_cache.move_to_end(key)
                self._phase_cache_hits += 1
                return cached

            self._phase_cache_misses += 1
            self._phase_block_computations += 1
            columns = self._compute_phase_columns(
                np.arange(start, stop, dtype=np.int64)
            )
            block_cells = int(columns.size)
            if (
                self.maximum_cached_phase_blocks > 0
                and block_cells <= self._phase_cache_capacity_cells
            ):
                while self._phase_cache and (
                    self._cached_phase_cells + block_cells
                    > self._phase_cache_capacity_cells
                    or len(self._phase_cache)
                    >= self.maximum_cached_phase_blocks
                ):
                    _, evicted = self._phase_cache.popitem(last=False)
                    self._cached_phase_cells -= int(evicted.size)
                columns.setflags(write=False)
                self._phase_cache[key] = columns
                self._cached_phase_cells += block_cells
            return columns

    def clear_phase_cache(self, *, reset_statistics: bool = False) -> None:
        """Release cached phase blocks without changing the configured budget."""

        with self._phase_cache_lock:
            self._phase_cache.clear()
            self._cached_phase_cells = 0
            if reset_statistics:
                self._phase_cache_hits = 0
                self._phase_cache_misses = 0
                self._phase_block_computations = 0
                self._sampled_column_computations = 0

    def _canonical_cache_capacity(self) -> tuple[int, int]:
        """Return maximum whole canonical blocks/cells that can be resident."""

        block_points = self._points_per_block()
        full_blocks, remainder = divmod(
            self.coefficient_count, block_points
        )
        full_cells = block_points * self.measurement_count
        remainder_cells = remainder * self.measurement_count
        block_limit = self.maximum_cached_phase_blocks
        cell_limit = self._phase_cache_capacity_cells
        without_remainder_blocks = min(
            full_blocks,
            block_limit,
            cell_limit // full_cells,
        )
        best_blocks = without_remainder_blocks
        best_cells = without_remainder_blocks * full_cells
        if remainder_cells and block_limit >= 1 and cell_limit >= remainder_cells:
            with_remainder_blocks = min(
                full_blocks,
                block_limit - 1,
                (cell_limit - remainder_cells) // full_cells,
            )
            candidate_cells = (
                remainder_cells + with_remainder_blocks * full_cells
            )
            if candidate_cells > best_cells:
                best_blocks = with_remainder_blocks + 1
                best_cells = candidate_cells
        return int(best_blocks), int(best_cells)

    def cache_status(self) -> dict[str, object]:
        """Report bounded-cache state and lifetime computation counters."""

        with self._phase_cache_lock:
            complex_bytes = np.dtype(np.complex128).itemsize
            total_cells = self.measurement_count * self.coefficient_count
            canonical_blocks = (
                self.coefficient_count + self._points_per_block() - 1
            ) // self._points_per_block()
            usable_blocks, usable_cells = self._canonical_cache_capacity()
            full_cacheable = (
                total_cells <= usable_cells
                and canonical_blocks <= usable_blocks
            )
            return {
                "operator_kind": "bounded_direct_point_scatterer_reference",
                "phase_matrix_cells": total_cells,
                "phase_matrix_bytes": total_cells * complex_bytes,
                "maximum_temporary_cells": self.maximum_temporary_cells,
                "cache_budget_cells": self._phase_cache_capacity_cells,
                "cache_payload_budget_bytes": (
                    self._phase_cache_capacity_cells * complex_bytes
                ),
                "cache_budget_bytes": (
                    self._phase_cache_capacity_cells * complex_bytes
                ),
                "configured_cache_cell_limit": (
                    self.maximum_cached_phase_cells
                ),
                "configured_cache_byte_limit": (
                    self.maximum_cached_phase_bytes
                ),
                "configured_cache_block_limit": (
                    self.maximum_cached_phase_blocks
                ),
                "usable_canonical_cache_blocks": usable_blocks,
                "usable_canonical_cache_cells": usable_cells,
                "cached_phase_cells": self._cached_phase_cells,
                "cached_phase_payload_bytes": (
                    self._cached_phase_cells * complex_bytes
                ),
                "cached_phase_bytes": (
                    self._cached_phase_cells * complex_bytes
                ),
                "cached_phase_blocks": len(self._phase_cache),
                "phase_cache_hits": self._phase_cache_hits,
                "phase_cache_misses": self._phase_cache_misses,
                "phase_block_computations": self._phase_block_computations,
                "sampled_column_computations": (
                    self._sampled_column_computations
                ),
                "statistics_scope": "dictionary lifetime since counter reset",
                "cache_byte_scope": (
                    "complex phase-array payload only; Python cache metadata "
                    "is separately bounded by configured_cache_block_limit, "
                    "and phase-construction temporaries use the independent "
                    "maximum_temporary_cells limit"
                ),
                "full_phase_matrix_cacheable": full_cacheable,
                "full_phase_matrix_cached": (
                    full_cacheable
                    and self._cached_phase_cells == total_cells
                    and len(self._phase_cache) == canonical_blocks
                ),
            }

    def iterative_readiness(
        self,
        max_iterations: int,
        *,
        norm_iterations: int = 20,
        history_stride: int = 25,
        maximum_identifiability_columns: int = 128,
    ) -> dict[str, object]:
        """Report whether a planned direct iterative solve needs opt-in."""

        planned_iterations = int(max_iterations)
        planned_norm_iterations = int(norm_iterations)
        planned_history_stride = int(history_stride)
        planned_identifiability_columns = int(maximum_identifiability_columns)
        if planned_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if planned_norm_iterations < 2:
            raise ValueError("norm_iterations must be at least two")
        if planned_history_stride < 1:
            raise ValueError("history_stride must be positive")
        if planned_identifiability_columns < 1:
            raise ValueError(
                "maximum_identifiability_columns must be positive"
            )
        status = self.cache_status()
        total_cells = int(status["phase_matrix_cells"])
        usable_capacity = int(status["usable_canonical_cache_cells"])
        uncached_per_application = max(total_cells - usable_capacity, 0)
        # Each solver iteration has at least one forward and one adjoint.  The
        # norm estimate has two of each.  Preserve this useful lower bound for
        # compatibility, but do not mistake it for the solve-wide upper bound.
        planned_applications_lower = (
            2 * planned_iterations + 4 * planned_norm_iterations
        )
        projected_uncached_lower = (
            uncached_per_application * planned_applications_lower
        )

        # A complete solve also traverses every column once to normalize W*G,
        # renders history at iteration 1 and every requested stride, and
        # performs one final component reconstruction.  Identifiability
        # sampling deliberately bypasses the canonical phase cache, so account
        # for those directly-computed cells separately.
        planned_history_applications = (
            planned_iterations
            if planned_history_stride == 1
            else 1 + planned_iterations // planned_history_stride
        )
        planned_normalization_applications = 1
        planned_norm_applications = 4 * planned_norm_iterations
        planned_solver_applications = 2 * planned_iterations
        planned_final_reconstruction_applications = 1
        planned_phase_applications_upper = (
            planned_normalization_applications
            + planned_norm_applications
            + planned_solver_applications
            + planned_history_applications
            + planned_final_reconstruction_applications
        )
        requested_samples = min(
            self.coefficient_count, planned_identifiability_columns
        )
        temporary_sample_limit = max(
            1, self.maximum_temporary_cells // self.measurement_count
        )
        planned_sampled_columns = min(
            requested_samples, temporary_sample_limit
        )
        planned_sampled_cells = (
            planned_sampled_columns * self.measurement_count
        )
        uncached_per_application_upper = (
            0 if bool(status["full_phase_matrix_cached"]) else total_cells
        )
        if bool(status["full_phase_matrix_cached"]):
            projected_uncached_phase_upper = 0
        elif bool(status["full_phase_matrix_cacheable"]):
            # The exhaustive column-normalization traversal warms every
            # canonical block; subsequent canonical applications are cached.
            projected_uncached_phase_upper = total_cells
        else:
            projected_uncached_phase_upper = (
                total_cells * planned_phase_applications_upper
            )
        projected_uncached_upper = (
            projected_uncached_phase_upper + planned_sampled_cells
        )
        oversized = (
            projected_uncached_upper
            > self.maximum_uncached_iteration_cells
        )
        if oversized:
            readiness = "uncached_oversized_opt_in_required"
        elif bool(status["full_phase_matrix_cached"]):
            readiness = "full_phase_matrix_cached"
        elif bool(status["full_phase_matrix_cacheable"]):
            readiness = "full_phase_matrix_will_warm"
        else:
            readiness = "bounded_block_cache_reference"
        status.update(
            {
                "planned_solver_iterations": planned_iterations,
                "planned_norm_iterations": planned_norm_iterations,
                "planned_history_stride": planned_history_stride,
                "planned_operator_applications_lower_bound": (
                    planned_applications_lower
                ),
                "planned_column_normalization_applications": (
                    planned_normalization_applications
                ),
                "planned_norm_operator_applications": (
                    planned_norm_applications
                ),
                "planned_solver_operator_applications": (
                    planned_solver_applications
                ),
                "planned_history_operator_applications": (
                    planned_history_applications
                ),
                "planned_final_reconstruction_applications": (
                    planned_final_reconstruction_applications
                ),
                "planned_operator_applications_upper_bound": (
                    planned_phase_applications_upper
                ),
                "uncached_cells_per_application_lower_bound": (
                    uncached_per_application
                ),
                "projected_uncached_cells_lower_bound": (
                    projected_uncached_lower
                ),
                "uncached_cells_per_application_upper_bound": (
                    uncached_per_application_upper
                ),
                "planned_identifiability_sampled_columns": (
                    planned_sampled_columns
                ),
                "projected_identifiability_sample_cells_upper_bound": (
                    planned_sampled_cells
                ),
                "projected_uncached_phase_cells_upper_bound": (
                    projected_uncached_phase_upper
                ),
                "projected_uncached_cells_upper_bound": (
                    projected_uncached_upper
                ),
                "maximum_uncached_iteration_cells": (
                    self.maximum_uncached_iteration_cells
                ),
                "uncached_cell_accounting_scope": (
                    "column normalization, identifiability sampling, operator-"
                    "norm estimation, solver iterations, history diagnostics, "
                    "and final component reconstruction"
                ),
                "uncached_oversized": oversized,
                "readiness": readiness,
                "readiness_note": (
                    "Bounded direct/reference point operator; use a reviewed "
                    "small dictionary or a future NUFFT implementation for "
                    "production-scale iterative reconstruction."
                ),
            }
        )
        return status

    def sampled_columns(self, maximum_columns: int = 256) -> np.ndarray:
        """Compute only representative point columns for screening."""

        requested = min(self.coefficient_count, max(1, int(maximum_columns)))
        temporary_limit = max(
            1, self.maximum_temporary_cells // self.measurement_count
        )
        count = min(requested, temporary_limit)
        indices = np.unique(
            np.linspace(0, self.coefficient_count - 1, count).astype(np.int64)
        )
        with self._phase_cache_lock:
            self._sampled_column_computations += int(indices.size)
        return self._compute_phase_columns(indices)

    def forward(self, coefficients: np.ndarray) -> np.ndarray:
        values = _finite_complex_vector(
            coefficients, self.coefficient_count, f"{self.name} coefficients"
        )
        output = np.zeros(self.measurement_count, dtype=np.complex128)
        for start, stop in self._point_blocks():
            output += self._phase_block(start, stop) @ values[start:stop]
            if not np.all(np.isfinite(output)):
                raise ValueError(f"{self.name} forward output became nonfinite")
        return output

    def adjoint(self, measurements: np.ndarray) -> np.ndarray:
        values = _finite_complex_vector(
            measurements, self.measurement_count, f"{self.name} measurements"
        )
        output = np.empty(self.coefficient_count, dtype=np.complex128)
        for start, stop in self._point_blocks(reverse=True):
            output[start:stop] = self._phase_block(start, stop).conj().T @ values
            if not np.all(np.isfinite(output[start:stop])):
                raise ValueError(f"{self.name} adjoint output became nonfinite")
        return output

    def weighted_column_norms(self, weights: np.ndarray) -> np.ndarray:
        observed_weights = _finite_real_weights(weights, self.measurement_count)
        norms = np.empty(self.coefficient_count, dtype=float)
        for start, stop in self._point_blocks():
            with np.errstate(over="ignore", invalid="ignore"):
                weighted = observed_weights[:, None] * self._phase_block(start, stop)
                norms[start:stop] = _stable_column_norms(weighted)
        if not np.all(np.isfinite(norms)):
            raise ValueError(
                f"{self.name} weighted dictionary-column norms must be finite"
            )
        return norms


class ConcatenatedDictionary:
    """Implicit ``[G_target, G_contamination, ...]`` operator."""

    def __init__(self, components: Iterable[ComponentDictionary]):
        self.components = tuple(components)
        if not self.components:
            raise ValueError("BPDE needs at least one component dictionary")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError("BPDE component names must be unique")
        counts = {component.measurement_count for component in self.components}
        if len(counts) != 1:
            raise ValueError("BPDE component dictionaries must share measurement size")
        self.measurement_count = counts.pop()
        self.coefficient_count = sum(
            component.coefficient_count for component in self.components
        )
        self.slices: dict[str, slice] = {}
        start = 0
        for component in self.components:
            stop = start + component.coefficient_count
            self.slices[component.name] = slice(start, stop)
            start = stop

    def forward(self, coefficients: np.ndarray) -> np.ndarray:
        values = _finite_complex_vector(
            coefficients,
            self.coefficient_count,
            "concatenated BPDE coefficients",
        )
        output = np.zeros(self.measurement_count, dtype=np.complex128)
        for component in self.components:
            contribution = _finite_complex_vector(
                component.forward(values[self.slices[component.name]]),
                self.measurement_count,
                f"{component.name} forward output",
            )
            output += contribution
            if not np.all(np.isfinite(output)):
                raise ValueError("concatenated BPDE forward output became nonfinite")
        return output

    def adjoint(self, measurements: np.ndarray) -> np.ndarray:
        values = _finite_complex_vector(
            measurements,
            self.measurement_count,
            "concatenated BPDE measurements",
        )
        parts = [
            _finite_complex_vector(
                component.adjoint(values),
                component.coefficient_count,
                f"{component.name} adjoint output",
            )
            for component in self.components
        ]
        return _finite_complex_vector(
            np.concatenate(parts),
            self.coefficient_count,
            "concatenated BPDE adjoint output",
        )


class _ColumnNormalizedDictionary:
    """``G D^-1`` with ``D[j,j] = ||W g_j||_2``.

    The solver variable is ``alpha = D gamma``. Thus its ordinary L1 norm is
    the scale-invariant weighted physical objective ``sum_j D_j |gamma_j|``.
    Returned coefficients are unscaled back to physical dictionary units.
    """

    def __init__(
        self,
        operator: ConcatenatedDictionary,
        component_column_norms: dict[str, np.ndarray],
    ):
        self.base = operator
        self.components = operator.components
        self.measurement_count = operator.measurement_count
        self.coefficient_count = operator.coefficient_count
        self.slices = operator.slices
        ordered = []
        for component in operator.components:
            values = np.asarray(
                component_column_norms[component.name], dtype=float
            ).reshape(-1)
            if values.size != component.coefficient_count:
                raise ValueError(
                    f"{component.name} dictionary-column norm count mismatch"
                )
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(
                    f"{component.name} observed dictionary-column norms must "
                    "be finite and positive"
                )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                inverse = 1.0 / values
            if not np.all(np.isfinite(inverse)) or np.any(inverse <= 0.0):
                raise ValueError(
                    f"{component.name} observed dictionary columns are too "
                    "small to normalize safely"
                )
            ordered.append(values)
        self.column_norms = np.concatenate(ordered)
        self.component_column_norms = {
            name: np.array(values, copy=True)
            for name, values in component_column_norms.items()
        }

    def forward(self, normalized_coefficients: np.ndarray) -> np.ndarray:
        values = _finite_complex_vector(
            normalized_coefficients,
            self.coefficient_count,
            "normalized BPDE coefficients",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            physical = values / self.column_norms
        physical = _finite_complex_vector(
            physical,
            self.coefficient_count,
            "physical BPDE coefficients",
        )
        return self.base.forward(physical)

    def adjoint(self, measurements: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            values = self.base.adjoint(measurements) / self.column_norms
        return _finite_complex_vector(
            values,
            self.coefficient_count,
            "normalized BPDE adjoint output",
        )

    def physical_coefficients(self, normalized_coefficients: np.ndarray) -> np.ndarray:
        values = _finite_complex_vector(
            normalized_coefficients,
            self.coefficient_count,
            "normalized BPDE coefficients",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            physical = values / self.column_norms
        return _finite_complex_vector(
            physical,
            self.coefficient_count,
            "physical BPDE coefficients",
        )


@dataclass(frozen=True)
class IdentifiabilityReport:
    maximum_cross_coherence: float
    component_pair: tuple[str, str] | None
    sampled_column_counts: dict[str, int]
    threshold: float
    zero_observed_column_counts: dict[str, int] = field(default_factory=dict)
    screen_note: str = (
        "Conservative sampled screen on W*G: failure identifies a sampled "
        "ambiguity; passing is not proof that the full component model is "
        "identifiable."
    )

    @property
    def identifiable(self) -> bool:
        """Backward-compatible name for whether the conservative screen passed."""

        return self.screen_passed

    @property
    def screen_passed(self) -> bool:
        return (
            self.maximum_cross_coherence < self.threshold
            and not any(self.zero_observed_column_counts.values())
        )


def assess_component_identifiability(
    components: Sequence[ComponentDictionary],
    *,
    noise_whitening_weights: np.ndarray | None = None,
    maximum_columns_per_component: int = 128,
    rejection_threshold: float = 0.999,
) -> IdentifiabilityReport:
    """Conservatively screen sampled columns of the observed operator ``W*G``.

    A failed screen demonstrates that at least one sampled cross-component pair
    is effectively aliased (or unobservable) under the actual measurement
    weights. A pass is *not* proof of global identifiability: unsampled columns,
    multi-column dependencies, and model error can remain ambiguous.
    """

    threshold = float(rejection_threshold)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("identifiability threshold must be in (0, 1]")
    if int(maximum_columns_per_component) < 1:
        raise ValueError("maximum sampled columns per component must be positive")
    components = tuple(components)
    operator = ConcatenatedDictionary(components)
    if noise_whitening_weights is None:
        weights = np.ones(operator.measurement_count, dtype=float)
    else:
        weights = _finite_real_weights(
            noise_whitening_weights, operator.measurement_count
        )
    sampled: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    zero_counts: dict[str, int] = {}
    for component in components:
        columns = np.asarray(
            component.sampled_columns(maximum_columns_per_component),
            dtype=np.complex128,
        )
        if columns.ndim != 2 or columns.shape[0] != component.measurement_count:
            raise ValueError(
                f"{component.name} sampled dictionary columns have invalid shape"
            )
        if not np.all(np.isfinite(columns)):
            raise ValueError(
                f"{component.name} sampled dictionary columns must be finite"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            columns = weights[:, None] * columns
        if not np.all(np.isfinite(columns)):
            raise ValueError(
                f"{component.name} observed sampled columns must be finite"
            )
        norms = _stable_column_norms(columns)
        if not np.all(np.isfinite(norms)):
            raise ValueError(
                f"{component.name} observed sampled-column norms must be finite"
            )
        keep = norms > 0.0
        zero_counts[component.name] = int(np.count_nonzero(~keep))
        columns = columns[:, keep]
        norms = norms[keep]
        if columns.shape[1]:
            columns = columns / norms[None, :]
            if not np.all(np.isfinite(columns)):
                raise ValueError(
                    f"{component.name} normalized sampled columns must be finite"
                )
        sampled[component.name] = columns
        counts[component.name] = int(columns.shape[1])
    maximum = 0.0
    pair = None
    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            if not sampled[left.name].size or not sampled[right.name].size:
                continue
            coherence = float(
                np.max(np.abs(sampled[left.name].conj().T @ sampled[right.name]))
            )
            if not np.isfinite(coherence):
                raise ValueError("sampled observed coherence became nonfinite")
            coherence = min(1.0, max(0.0, coherence))
            if coherence > maximum:
                maximum = coherence
                pair = (left.name, right.name)
    return IdentifiabilityReport(
        maximum,
        pair,
        counts,
        threshold,
        zero_observed_column_counts=zero_counts,
    )


@dataclass
class BpdnResult:
    coefficients: dict[str, np.ndarray]
    reconstructed_components: dict[str, np.ndarray]
    reconstructed_measurements: np.ndarray
    residual: np.ndarray
    sigma: float
    residual_norm: float
    l1_norm: float
    iterations: int
    converged: bool
    stopping_reason: str
    operator_norm: float
    relative_change: float
    identifiability: IdentifiabilityReport | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    column_norms: dict[str, np.ndarray] = field(default_factory=dict)
    operator_readiness: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    dual_relative_change: float = 0.0
    fixed_point_residual: float = 0.0
    internal_problem_scale: float = 1.0
    feasibility_tolerance: float = 0.0
    l1_norm_definition: str = (
        "sum_j ||W*g_j||_2 * |physical_coefficient_j|"
    )
    operator_norm_estimate: float = 0.0
    operator_norm_bound_kind: str = ""


def _estimate_operator_norm(
    operator: _ColumnNormalizedDictionary,
    weights: np.ndarray,
    iterations: int,
) -> float:
    if int(iterations) < 2:
        raise ValueError("BPDE norm-estimation iterations must be at least two")
    weights = _finite_real_weights(weights, operator.measurement_count)
    rng = np.random.default_rng(1729)
    vector = rng.standard_normal(operator.coefficient_count) + 1j * rng.standard_normal(
        operator.coefficient_count
    )
    vector_norm = _stable_vector_norm(vector)
    if not np.isfinite(vector_norm) or vector_norm <= 0.0:
        raise ValueError("BPDE norm-estimation seed is invalid")
    vector /= vector_norm
    eigenvalue = 0.0
    for _ in range(max(2, int(iterations))):
        with np.errstate(over="ignore", invalid="ignore"):
            forward = weights * operator.forward(vector)
        forward = _finite_complex_vector(
            forward, operator.measurement_count, "weighted BPDE forward output"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_forward = weights * forward
        weighted_forward = _finite_complex_vector(
            weighted_forward,
            operator.measurement_count,
            "twice-weighted BPDE forward output",
        )
        adjoint = operator.adjoint(weighted_forward)
        norm = _stable_vector_norm(adjoint)
        if not np.isfinite(norm) or norm <= 1.0e-30:
            raise ValueError("BPDE dictionary has zero operator norm")
        vector = adjoint / norm
        with np.errstate(over="ignore", invalid="ignore"):
            rayleigh_forward = weights * operator.forward(vector)
            rayleigh_adjoint_input = weights * rayleigh_forward
        rayleigh_adjoint_input = _finite_complex_vector(
            rayleigh_adjoint_input,
            operator.measurement_count,
            "BPDE norm-estimation weighted forward output",
        )
        rayleigh_adjoint = operator.adjoint(rayleigh_adjoint_input)
        eigenvalue = float(np.vdot(vector, rayleigh_adjoint).real)
        if not np.isfinite(eigenvalue) or eigenvalue <= 0.0:
            raise ValueError("BPDE operator-norm estimate became nonfinite")
    operator_norm = float(np.sqrt(eigenvalue))
    if not np.isfinite(operator_norm) or operator_norm <= 0.0:
        raise ValueError("BPDE operator norm must be finite and positive")
    return operator_norm


def _certified_operator_norm_bound(
    operator: _ColumnNormalizedDictionary,
    weights: np.ndarray,
) -> tuple[float, str]:
    """Return a conservative upper bound for ``||W G D^-1||_2``.

    Every observed column was normalized with its exact Euclidean norm, so the
    Frobenius bound is analytically ``sqrt(number_of_columns)``.  When all
    components are dense, the induced ``sqrt(||K||_1 ||K||_inf)`` bound is also
    inexpensive and can be substantially tighter.  The smaller of two valid
    upper bounds remains valid.  A dimension-scaled floating-point margin is
    included before the bound is used for PDHG steps.
    """

    observed_weights = _finite_real_weights(
        weights, operator.measurement_count
    )
    coefficient_count = int(operator.coefficient_count)
    if coefficient_count < 1:
        raise ValueError("BPDE operator must have at least one column")
    frobenius_bound = float(np.sqrt(float(coefficient_count)))
    bound = frobenius_bound
    kind = "normalized-column Frobenius bound"

    if all(
        isinstance(component, DenseComponentDictionary)
        for component in operator.components
    ):
        row_absolute_sums = np.zeros(operator.measurement_count, dtype=float)
        maximum_column_absolute_sum = 0.0
        for component in operator.components:
            norms = np.asarray(
                operator.component_column_norms[component.name], dtype=float
            ).reshape(-1)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                observed_absolute = (
                    observed_weights[:, None] * np.abs(component.matrix)
                ) / norms[None, :]
            if not np.all(np.isfinite(observed_absolute)):
                raise ValueError(
                    "normalized dense BPDE operator must remain finite"
                )
            row_absolute_sums += np.sum(observed_absolute, axis=1)
            column_absolute_sums = np.sum(observed_absolute, axis=0)
            maximum_column_absolute_sum = max(
                maximum_column_absolute_sum,
                float(np.max(column_absolute_sums, initial=0.0)),
            )
        maximum_row_absolute_sum = float(
            np.max(row_absolute_sums, initial=0.0)
        )
        induced_bound = float(
            np.sqrt(
                maximum_column_absolute_sum * maximum_row_absolute_sum
            )
        )
        if not np.isfinite(induced_bound) or induced_bound <= 0.0:
            raise ValueError(
                "dense BPDE operator-norm bound must be finite and positive"
            )
        if induced_bound < bound:
            bound = induced_bound
            kind = "dense induced 1/infinity-norm bound"

    rounding_factor = 1.0 + (
        128.0
        * np.finfo(np.float64).eps
        * float(
            max(
                1,
                operator.measurement_count,
                operator.coefficient_count,
            )
        )
    )
    if not np.isfinite(rounding_factor):
        raise ValueError("BPDE operator-norm rounding margin became nonfinite")
    bound = float(np.nextafter(bound * rounding_factor, np.inf))
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError(
            "certified BPDE operator-norm bound must be finite and positive"
        )
    return bound, kind


def solve_bpdn_components(
    measurements: np.ndarray,
    components: Sequence[ComponentDictionary],
    *,
    sigma: float,
    noise_whitening_weights: np.ndarray | None = None,
    max_iterations: int = 4000,
    tolerance: float = 1.0e-5,
    norm_iterations: int = 20,
    history_stride: int = 25,
    cancel_check: Callable[[], bool] | None = None,
    require_identifiable: bool = True,
    identifiability_threshold: float = 0.999,
    allow_uncached_oversized_point_operator: bool = False,
    maximum_total_uncached_iteration_cells: int = 250_000_000,
) -> BpdnResult:
    r"""Solve scale-invariant weighted BPDN on the observed operator.

    In physical coefficient units the objective is

    ``min sum_j ||W g_j||_2 |gamma_j|`` subject to
    ``||W(G gamma-s)||_2 <= sigma``.

    Equivalently, every observed column is normalized and ordinary L1 is
    solved for ``alpha_j = ||W g_j||_2 gamma_j``. Returned coefficients are
    unscaled back to the caller's physical dictionary convention. Therefore
    multiplying a dictionary column by any representable nonzero scalar cannot
    change component attribution or reconstruction.

    A primal-dual hybrid-gradient method is used with a certified operator-norm
    upper bound; the finite power estimate is retained only as a diagnostic and
    never used to relax the convergence-safe steps. This is residual-constrained
    BPDN, not the fixed-lambda LASSO used by GRIM's visualization-oriented sparse
    image former. Oversized direct point dictionaries fail before preflight or
    iteration unless the caller accepts their projected uncached runtime with
    the explicit opt-in flag.
    """

    physical_operator = ConcatenatedDictionary(components)
    observed = _finite_complex_vector(
        measurements,
        physical_operator.measurement_count,
        "BPDE measurements",
    )
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("BPDN sigma must be finite and nonnegative")
    if int(max_iterations) < 1:
        raise ValueError("BPDN max_iterations must be positive")
    if int(norm_iterations) < 2:
        raise ValueError("BPDN norm_iterations must be at least two")
    if int(history_stride) < 1:
        raise ValueError("BPDN history_stride must be positive")
    if not isinstance(
        allow_uncached_oversized_point_operator, (bool, np.bool_)
    ):
        raise ValueError(
            "allow_uncached_oversized_point_operator must be boolean"
        )
    maximum_total_uncached_iteration_cells = int(
        maximum_total_uncached_iteration_cells
    )
    if maximum_total_uncached_iteration_cells < 1:
        raise ValueError(
            "maximum_total_uncached_iteration_cells must be positive"
        )
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("BPDN tolerance must be finite and in (0, 1)")
    if noise_whitening_weights is None:
        weights = np.ones(observed.size, dtype=float)
    else:
        weights = _finite_real_weights(noise_whitening_weights, observed.size)
    with np.errstate(over="ignore", invalid="ignore"):
        whitened_observed = weights * observed
    whitened_observed = _finite_complex_vector(
        whitened_observed, observed.size, "whitened BPDE measurements"
    )
    observed_norm = _stable_vector_norm(whitened_observed)
    if not np.isfinite(observed_norm):
        raise ValueError("whitened BPDE measurement norm must be finite")
    problem_scale = max(observed_norm, sigma)
    if problem_scale == 0.0:
        problem_scale = 1.0
    if not np.isfinite(problem_scale) or problem_scale <= 0.0:
        raise ValueError("BPDE internal problem scale must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scaled_whitened_observed = whitened_observed / problem_scale
        scaled_sigma = sigma / problem_scale
    scaled_whitened_observed = _finite_complex_vector(
        scaled_whitened_observed,
        observed.size,
        "scaled whitened BPDE measurements",
    )
    if not np.isfinite(scaled_sigma) or scaled_sigma < 0.0:
        raise ValueError("scaled BPDN sigma must be finite and nonnegative")
    scaled_reference = max(
        _stable_vector_norm(scaled_whitened_observed),
        scaled_sigma,
        np.finfo(np.float64).tiny,
    )
    effective_relative_tolerance = max(
        tolerance, 64.0 * np.finfo(np.float64).eps
    )
    scaled_feasibility_tolerance = (
        effective_relative_tolerance * scaled_reference
    )
    feasibility_tolerance = problem_scale * scaled_feasibility_tolerance
    if not np.isfinite(feasibility_tolerance):
        raise ValueError("BPDN feasibility tolerance became nonfinite")

    point_readiness_before: dict[str, dict[str, object]] = {}
    individually_oversized: list[str] = []
    for component in physical_operator.components:
        if not isinstance(component, PointScattererDictionary):
            continue
        status = component.iterative_readiness(
            int(max_iterations),
            norm_iterations=int(norm_iterations),
            history_stride=int(history_stride),
        )
        oversized = bool(status["uncached_oversized"])
        if oversized:
            individually_oversized.append(component.name)
        point_readiness_before[component.name] = dict(status)
    total_projected_uncached = sum(
        int(status["projected_uncached_cells_upper_bound"])
        for status in point_readiness_before.values()
    )
    aggregate_oversized = (
        total_projected_uncached
        > maximum_total_uncached_iteration_cells
    )
    if (
        individually_oversized or aggregate_oversized
    ) and not allow_uncached_oversized_point_operator:
        if individually_oversized:
            name = individually_oversized[0]
            status = point_readiness_before[name]
            cells = int(status["phase_matrix_cells"])
            phase_bytes = int(status["phase_matrix_bytes"])
            capacity = int(status["usable_canonical_cache_cells"])
            projected = int(status["projected_uncached_cells_upper_bound"])
            raise ValueError(
                f"Point dictionary {name!r} would use a direct "
                f"{cells:,}-cell ({phase_bytes:,}-byte) phase operator with "
                f"only {capacity:,} usable cached cells and up to "
                f"{projected:,} uncached cell evaluations across preflight, "
                "the requested iterations, diagnostics, and final "
                "reconstruction. Increase maximum_cached_phase_cells and "
                "maximum_cached_phase_bytes enough to cache the matrix, "
                "reduce the dictionary or iteration limits, or use a future "
                "NUFFT operator. To accept this bounded-reference direct "
                "runtime explicitly, pass "
                "allow_uncached_oversized_point_operator=True."
            )
        names = ", ".join(repr(name) for name in point_readiness_before)
        raise ValueError(
            f"Point dictionaries {names} project up to "
            f"{total_projected_uncached:,} combined uncached cell "
            "evaluations, exceeding the solve-wide "
            f"maximum_total_uncached_iteration_cells="
            f"{maximum_total_uncached_iteration_cells:,} budget. Increase "
            "the phase caches or solve-wide budget, reduce the dictionaries "
            "or iteration limits, or use a future NUFFT operator. To accept "
            "this bounded-reference direct runtime explicitly, pass "
            "allow_uncached_oversized_point_operator=True."
        )

    # Solve in normalized coefficient coordinates alpha_j = ||W*g_j||*gamma_j.
    # This makes the L1 objective invariant when any physical dictionary column
    # is multiplied by an arbitrary nonzero scalar; returned gamma coefficients
    # are converted back to each caller's original dictionary units.
    component_column_norms: dict[str, np.ndarray] = {}
    for component in physical_operator.components:
        norms = np.asarray(
            component.weighted_column_norms(weights), dtype=float
        ).reshape(-1)
        if norms.size != component.coefficient_count:
            raise ValueError(
                f"{component.name} dictionary-column norm count mismatch"
            )
        if not np.all(np.isfinite(norms)):
            raise ValueError(
                f"{component.name} observed dictionary-column norms must be finite"
            )
        unobservable = norms <= 0.0
        if np.any(unobservable):
            raise ValueError(
                f"{component.name!r} has {int(np.count_nonzero(unobservable))} "
                "dictionary column(s) that are unobservable under the supplied "
                "measurement weights W"
            )
        component_column_norms[component.name] = norms
    operator = _ColumnNormalizedDictionary(
        physical_operator, component_column_norms
    )

    identifiability = assess_component_identifiability(
        physical_operator.components,
        noise_whitening_weights=weights,
        rejection_threshold=identifiability_threshold,
    )
    if require_identifiable and not identifiability.identifiable:
        pair = identifiability.component_pair or ("unknown", "unknown")
        raise ValueError(
            f"BPDE component dictionaries {pair[0]!r} and {pair[1]!r} are "
            "not identifiable on the observed weighted samples (failed the "
            "conservative sampled W*G coherence screen at "
            f"{identifiability.maximum_cross_coherence:.6f}); passing this "
            "screen would still not prove full-model identifiability"
        )

    operator_norm_estimate = _estimate_operator_norm(
        operator, weights, norm_iterations
    )
    certified_operator_norm, operator_norm_bound_kind = (
        _certified_operator_norm_bound(operator, weights)
    )
    # The analytic/induced bound is convergence-safe. Retaining the maximum
    # also protects against unexpected floating-point disagreement in either
    # diagnostic calculation.
    operator_norm = max(operator_norm_estimate, certified_operator_norm)
    primal_step = 0.99 / operator_norm
    dual_step = 0.99 / operator_norm
    if not (
        np.isfinite(primal_step)
        and np.isfinite(dual_step)
        and primal_step > 0.0
        and dual_step > 0.0
    ):
        raise ValueError("BPDE primal/dual steps must be finite and positive")
    coefficients = np.zeros(operator.coefficient_count, dtype=np.complex128)
    extrapolated = coefficients.copy()
    dual = np.zeros(operator.measurement_count, dtype=np.complex128)
    relative_change = 0.0
    dual_relative_change = 0.0
    fixed_point_residual = 0.0
    history: list[dict[str, float]] = []
    stopping_reason = "maximum iterations reached"
    converged = False
    iterations_used = 0

    for iteration in range(1, int(max_iterations) + 1):
        if cancel_check is not None and cancel_check():
            stopping_reason = "cancelled"
            iterations_used = iteration - 1
            break
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_forward = weights * operator.forward(extrapolated)
            dual_trial = dual + dual_step * weighted_forward
        weighted_forward = _finite_complex_vector(
            weighted_forward,
            operator.measurement_count,
            "weighted BPDE forward iterate",
        )
        dual_trial = _finite_complex_vector(
            dual_trial, operator.measurement_count, "BPDE dual trial"
        )
        scaled = dual_trial / dual_step
        scaled = _finite_complex_vector(
            scaled, operator.measurement_count, "scaled BPDE dual trial"
        )
        displacement = scaled - scaled_whitened_observed
        displacement = _finite_complex_vector(
            displacement, operator.measurement_count, "BPDE dual displacement"
        )
        displacement_norm = _stable_vector_norm(displacement)
        if not np.isfinite(displacement_norm):
            raise ValueError("BPDE dual displacement norm became nonfinite")
        if displacement_norm <= scaled_sigma:
            projected = scaled
        else:
            projected = scaled_whitened_observed + displacement * (
                scaled_sigma / max(displacement_norm, np.finfo(float).tiny)
            )
        projected = _finite_complex_vector(
            projected, operator.measurement_count, "BPDE dual projection"
        )
        dual_new = dual_trial - dual_step * projected
        dual_new = _finite_complex_vector(
            dual_new, operator.measurement_count, "BPDE dual iterate"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_dual = weights * dual_new
        weighted_dual = _finite_complex_vector(
            weighted_dual,
            operator.measurement_count,
            "weighted BPDE dual iterate",
        )
        gradient = operator.adjoint(weighted_dual)
        gradient = _finite_complex_vector(
            gradient, operator.coefficient_count, "BPDE primal gradient"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            primal_trial = coefficients - primal_step * gradient
        coefficients_new = _complex_soft_threshold(primal_trial, primal_step)
        coefficients_new = _finite_complex_vector(
            coefficients_new,
            operator.coefficient_count,
            "normalized BPDE coefficient iterate",
        )
        change_norm = _stable_vector_norm(coefficients_new - coefficients)
        coefficient_norm = _stable_vector_norm(coefficients_new)
        previous_coefficient_norm = _stable_vector_norm(coefficients)
        dual_change_norm = _stable_vector_norm(dual_new - dual)
        dual_norm = _stable_vector_norm(dual_new)
        previous_dual_norm = _stable_vector_norm(dual)
        if not all(
            np.isfinite(value)
            for value in (
                change_norm,
                coefficient_norm,
                previous_coefficient_norm,
                dual_change_norm,
                dual_norm,
                previous_dual_norm,
            )
        ):
            raise ValueError("BPDE coefficient-iterate norm became nonfinite")
        coefficient_reference = max(
            coefficient_norm, previous_coefficient_norm
        )
        dual_reference = max(dual_norm, previous_dual_norm)
        relative_change = (
            0.0
            if coefficient_reference == 0.0
            else float(change_norm / coefficient_reference)
        )
        dual_relative_change = (
            0.0
            if dual_reference == 0.0
            else float(dual_change_norm / dual_reference)
        )
        fixed_point_residual = float(
            max(relative_change, dual_relative_change)
        )
        if not all(
            np.isfinite(value)
            for value in (
                relative_change,
                dual_relative_change,
                fixed_point_residual,
            )
        ):
            raise ValueError("BPDE relative iterate change became nonfinite")
        extrapolated = coefficients_new + (coefficients_new - coefficients)
        extrapolated = _finite_complex_vector(
            extrapolated,
            operator.coefficient_count,
            "extrapolated BPDE coefficient iterate",
        )
        coefficients = coefficients_new
        dual = dual_new
        iterations_used = iteration

        if iteration == 1 or iteration % max(1, int(history_stride)) == 0:
            with np.errstate(over="ignore", invalid="ignore"):
                scaled_weighted_prediction = weights * operator.forward(
                    coefficients
                )
                scaled_weighted_residual = (
                    scaled_weighted_prediction - scaled_whitened_observed
                )
            scaled_weighted_residual = _finite_complex_vector(
                scaled_weighted_residual,
                operator.measurement_count,
                "scaled weighted BPDE residual iterate",
            )
            scaled_residual_norm = _stable_vector_norm(
                scaled_weighted_residual
            )
            residual_norm = problem_scale * scaled_residual_norm
            normalized_l1 = problem_scale * float(
                np.sum(np.abs(coefficients))
            )
            if not all(
                np.isfinite(value)
                for value in (
                    scaled_residual_norm,
                    residual_norm,
                    normalized_l1,
                )
            ):
                raise ValueError("BPDE convergence diagnostics became nonfinite")
            history.append(
                {
                    "iteration": float(iteration),
                    "residual_norm": residual_norm,
                    "l1_norm": normalized_l1,
                    "relative_change": relative_change,
                    "dual_relative_change": dual_relative_change,
                    "fixed_point_residual": fixed_point_residual,
                }
            )
            if (
                scaled_residual_norm
                <= scaled_sigma + scaled_feasibility_tolerance
                and fixed_point_residual <= effective_relative_tolerance
            ):
                converged = True
                stopping_reason = "feasible primal-dual fixed point"
                break

    reconstructed_components: dict[str, np.ndarray] = {}
    coefficient_components: dict[str, np.ndarray] = {}
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_coefficients = coefficients * problem_scale
    normalized_coefficients = _finite_complex_vector(
        normalized_coefficients,
        operator.coefficient_count,
        "unscaled normalized BPDE coefficients",
    )
    physical_coefficients = operator.physical_coefficients(
        normalized_coefficients
    )
    reconstructed = np.zeros(
        operator.measurement_count, dtype=np.complex128
    )
    for component in physical_operator.components:
        part = np.array(
            physical_coefficients[physical_operator.slices[component.name]],
            copy=True,
        )
        part = _finite_complex_vector(
            part,
            component.coefficient_count,
            f"{component.name} physical BPDE coefficients",
        )
        contribution = _finite_complex_vector(
            component.forward(part),
            operator.measurement_count,
            f"{component.name} reconstructed BPDE measurements",
        )
        coefficient_components[component.name] = part
        reconstructed_components[component.name] = contribution
        with np.errstate(over="ignore", invalid="ignore"):
            reconstructed += contribution
        if not np.all(np.isfinite(reconstructed)):
            raise ValueError("BPDE reconstructed measurements became nonfinite")
    residual = observed - reconstructed
    residual = _finite_complex_vector(
        residual, operator.measurement_count, "BPDE output residual"
    )
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_residual = weights * residual
    weighted_residual = _finite_complex_vector(
        weighted_residual,
        operator.measurement_count,
        "weighted BPDE output residual",
    )
    residual_norm = _stable_vector_norm(weighted_residual)
    normalized_l1 = float(np.sum(np.abs(normalized_coefficients)))
    if not np.isfinite(residual_norm) or not np.isfinite(normalized_l1):
        raise ValueError("BPDE output diagnostics must be finite")
    if not history or history[-1].get("iteration") != float(iterations_used):
        history.append(
            {
                "iteration": float(iterations_used),
                "residual_norm": residual_norm,
                "l1_norm": normalized_l1,
                "relative_change": relative_change,
                "dual_relative_change": dual_relative_change,
                "fixed_point_residual": fixed_point_residual,
            }
        )
    for entry in history:
        if not all(np.isfinite(float(value)) for value in entry.values()):
            raise ValueError("BPDE iteration history contains nonfinite values")
    final_readiness: dict[str, dict[str, object]] = {}
    for component in physical_operator.components:
        if not isinstance(component, PointScattererDictionary):
            continue
        status = component.iterative_readiness(
            int(max_iterations),
            norm_iterations=int(norm_iterations),
            history_stride=int(history_stride),
        )
        contributes_uncached_work = (
            int(status["projected_uncached_cells_upper_bound"]) > 0
        )
        accepted_oversized = bool(
            allow_uncached_oversized_point_operator
            and (
                status["uncached_oversized"]
                or (aggregate_oversized and contributes_uncached_work)
            )
        )
        status["uncached_oversized_opt_in_used"] = bool(
            accepted_oversized
        )
        status["aggregate_uncached_oversized"] = aggregate_oversized
        status["total_projected_uncached_cells_upper_bound"] = (
            total_projected_uncached
        )
        status["maximum_total_uncached_iteration_cells"] = (
            maximum_total_uncached_iteration_cells
        )
        initial = point_readiness_before[component.name]
        for counter in (
            "phase_cache_hits",
            "phase_cache_misses",
            "phase_block_computations",
            "sampled_column_computations",
        ):
            status[f"solve_{counter}"] = int(status[counter]) - int(
                initial[counter]
            )
        if accepted_oversized:
            status["guard_readiness"] = status["readiness"]
            status["readiness"] = "uncached_oversized_opt_in_accepted"
        final_readiness[component.name] = status
    return BpdnResult(
        coefficients=coefficient_components,
        reconstructed_components=reconstructed_components,
        reconstructed_measurements=reconstructed,
        residual=residual,
        sigma=sigma,
        residual_norm=residual_norm,
        l1_norm=normalized_l1,
        iterations=iterations_used,
        converged=converged,
        stopping_reason=stopping_reason,
        operator_norm=operator_norm,
        relative_change=relative_change,
        operator_norm_estimate=operator_norm_estimate,
        operator_norm_bound_kind=operator_norm_bound_kind,
        identifiability=identifiability,
        history=history,
        column_norms={
            name: np.array(values, copy=True)
            for name, values in component_column_norms.items()
        },
        operator_readiness=final_readiness,
        dual_relative_change=dual_relative_change,
        fixed_point_residual=fixed_point_residual,
        internal_problem_scale=problem_scale,
        feasibility_tolerance=feasibility_tolerance,
    )


__all__ = [
    "BpdnResult",
    "ComponentDictionary",
    "ConcatenatedDictionary",
    "DenseComponentDictionary",
    "IdentifiabilityReport",
    "PointScattererDictionary",
    "assess_component_identifiability",
    "solve_bpdn_components",
]
