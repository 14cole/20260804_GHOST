from __future__ import annotations

import csv
import copy
import json
import math
import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterator, TextIO

from .compute import LayerConfig, MaterialTable

MATERIAL_HEADER = "frequency_hz,eps_real,eps_imag,mu_real,mu_imag"
IMPEDANCE_HEADER = "frequency_hz,resistance_ohm,reactance_ohm"
IMPEDANCE_UNCERTAINTY_HEADER = (
    "frequency_hz,resistance_ohm,reactance_ohm,"
    "resistance_ohm_min,resistance_ohm_max,"
    "reactance_ohm_min,reactance_ohm_max"
)
MATERIAL_SINGULAR_TOL = 1e-12

PROJECT_SCHEMA_VERSION = 1
PROJECT_PATH_POLICY_VERSION = 1
PROJECT_PATH_PARENT_HOPS = 1

_PROJECT_CONTROL_PATH_KEYS = (
    "output",
    "ibc_batch_output_dir",
    "angle_output",
    "thk_output",
    "mix_prop_file",
)

# Property and result files are CSV with frequency in Hz. Frequencies are
# converted to GHz on read and back to Hz on write, since every computation and
# UI field in this project works in GHz.
HZ_PER_GHZ = 1e9


@contextmanager
def _atomic_text_file(path: Path) -> Iterator[TextIO]:
    """Yield a same-directory temporary text file and atomically publish it.

    Keeping the temporary file beside the destination makes ``os.replace`` an
    atomic same-filesystem operation.  A failed write or replace removes only
    the temporary file and leaves an existing destination untouched.
    """

    destination = Path(path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    fd_needs_close = True
    try:
        stream = os.fdopen(fd, "w", encoding="utf-8")
        fd_needs_close = False
        with stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if fd_needs_close:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _stage_text_file(
    path: Path, writer: Callable[[TextIO], None]
) -> Path:
    """Write and fsync one same-directory text stage without publishing it."""

    destination = Path(path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".stage",
        dir=str(destination.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    fd_needs_close = True
    try:
        stream = os.fdopen(fd, "w", encoding="utf-8")
        fd_needs_close = False
        with stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary_path
    except BaseException:
        if fd_needs_close:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _atomic_text_batch(
    entries: list[tuple[Path, Callable[[TextIO], None]]],
) -> None:
    """Publish related text artifacts together, restoring all on failure."""

    normalized: set[str] = set()
    planned: list[tuple[Path, Callable[[TextIO], None]]] = []
    for raw_path, writer in entries:
        destination = Path(raw_path)
        # Treat case-only path differences as collisions even when this batch
        # is prepared on a case-sensitive filesystem for later Windows use.
        identity = os.path.normcase(
            str(destination.resolve(strict=False))
        ).casefold()
        if identity in normalized:
            raise ValueError(
                f"Output batch contains duplicate path {destination}."
            )
        if destination.exists() and not destination.is_file():
            raise ValueError(
                f"Output target {destination} exists but is not a regular file."
            )
        normalized.add(identity)
        planned.append((destination, writer))

    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path | None] = {}
    publication_complete = False
    try:
        for destination, writer in planned:
            staged.append((_stage_text_file(destination, writer), destination))

        for stage, destination in staged:
            backup: Path | None = None
            if destination.exists():
                backup_fd, backup_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".backup",
                    dir=str(destination.parent),
                )
                os.close(backup_fd)
                backup = Path(backup_name)
                try:
                    os.replace(destination, backup)
                except BaseException:
                    try:
                        backup.unlink()
                    except OSError:
                        pass
                    raise
            backups[destination] = backup
            os.replace(stage, destination)
        publication_complete = True
    except BaseException as original_error:
        rollback_errors: list[str] = []
        for destination, backup in reversed(list(backups.items())):
            try:
                if backup is None:
                    try:
                        destination.unlink()
                    except FileNotFoundError:
                        pass
                elif backup.exists():
                    os.replace(backup, destination)
                    backups[destination] = None
            except OSError as exc:
                rollback_errors.append(f"{destination}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "Output publication failed and rollback could not restore every "
                "prior artifact. Retained .backup file(s): "
                + "; ".join(rollback_errors)
            ) from original_error
        raise
    finally:
        for stage, _destination in staged:
            try:
                stage.unlink()
            except FileNotFoundError:
                pass
        if publication_complete:
            for backup in backups.values():
                if backup is not None:
                    try:
                        backup.unlink()
                    except OSError:
                        pass


def _passivity_tolerance(value: complex) -> float:
    return 64.0 * math.ulp(1.0) * max(1.0, abs(value))


def _validate_medium(eps: complex, mu: complex, context: str) -> None:
    for label, value in (("epsilon", eps), ("mu", mu)):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError(f"{context} {label} contains a non-finite value.")
        if abs(value) <= MATERIAL_SINGULAR_TOL:
            raise ValueError(
                f"{context} has unsupported singular/near-zero {label} "
                f"{value!r} (magnitude <= {MATERIAL_SINGULAR_TOL:g})."
            )
        if value.imag > _passivity_tolerance(value):
            raise ValueError(
                f"{context} {label} has gain-sign imaginary part "
                f"{value.imag:g}. FREDDY and the RCS solvers use e^(+jωt), "
                "so passive media require signed imaginary parts <= 0."
            )


def _validate_surface_impedance(z: complex, context: str) -> None:
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise ValueError(f"{context} contains a non-finite impedance.")
    if z.real < -_passivity_tolerance(z):
        raise ValueError(
            f"{context} has negative resistance Re(Zs)={z.real:g} ohm. "
            "Active/gain surface impedances are not supported."
        )


def _validate_frequency_rows(
    frequencies_ghz: list[float], context: str, *, minimum_rows: int = 1
) -> None:
    if len(frequencies_ghz) < minimum_rows:
        row_word = "row" if minimum_rows == 1 else "rows"
        raise ValueError(f"{context} requires at least {minimum_rows} data {row_word}.")
    previous: float | None = None
    for index, freq_ghz in enumerate(frequencies_ghz, start=1):
        if not math.isfinite(freq_ghz) or freq_ghz <= 0:
            raise ValueError(
                f"{context} row {index} has invalid frequency {freq_ghz!r} GHz; "
                "frequencies must be finite and > 0."
            )
        if previous is not None and freq_ghz <= previous:
            relation = "duplicate" if freq_ghz == previous else "out-of-order"
            raise ValueError(
                f"{context} row {index} has {relation} frequency "
                f"{freq_ghz:g} GHz; frequencies must be strictly increasing."
            )
        previous = freq_ghz


def read_material_table(path: Path, skiprows: int = 0) -> MaterialTable:
    """Read the exact five-column material CSV accepted by the RCS solvers.

    Frequencies are stored in Hz in the file and converted to GHz internally.
    Blank lines and ``#`` comments may precede the header or appear between
    data rows. ``skiprows`` is retained for old project files, but the first
    remaining non-comment row must still be the exact schema header.
    """
    rows: list[tuple[float, complex, complex]] = []
    header_found = False
    with path.open("r", encoding="utf-8", newline="") as f:
        for idx, raw_line in enumerate(f, start=1):
            if idx <= skiprows:
                continue
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = next(csv.reader([raw_line]))
            parts = [part.strip() for part in parts]
            if not header_found:
                found = ",".join(parts)
                if found != MATERIAL_HEADER:
                    raise ValueError(
                        f"{path}: line {idx} must have header {MATERIAL_HEADER}; "
                        f"found {found}."
                    )
                header_found = True
                continue
            if len(parts) != 5:
                raise ValueError(
                    f"{path}: line {idx} must contain exactly 5 comma-separated "
                    f"columns; found {len(parts)}."
                )
            try:
                values = [float(part) for part in parts]
            except ValueError:
                raise ValueError(
                    f"{path}: line {idx} contains a non-numeric value."
                ) from None
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: line {idx} contains a non-finite value.")
            if values[0] <= 0:
                raise ValueError(
                    f"{path}: line {idx} has non-positive frequency "
                    f"{values[0]:g} Hz."
                )
            # e^{+jwt} convention: loss is a negative imaginary part. The file
            # supplies eps''/mu'' already signed (negative for lossy), so the
            # column values are used directly as the imaginary parts.
            freq_ghz = values[0] / HZ_PER_GHZ
            eps = complex(values[1], values[2])
            mu = complex(values[3], values[4])
            _validate_medium(eps, mu, f"{path}: line {idx}")
            rows.append((freq_ghz, eps, mu))

    if not header_found:
        raise ValueError(f"{path}: missing required header {MATERIAL_HEADER}.")
    if not rows:
        raise ValueError(f"{path}: at least one valid data row is required.")

    rows.sort(key=lambda r: r[0])
    _validate_frequency_rows(
        [row[0] for row in rows], f"Material file {path}"
    )
    return MaterialTable(
        freq_ghz=[r[0] for r in rows],
        eps_r=[r[1] for r in rows],
        mu_r=[r[2] for r in rows],
    )


def write_output(
    path: Path,
    rows: list[tuple[float, float, float]],
    include_header: bool,
) -> None:
    """Write a solver-compatible frequency/impedance CSV in Hz and ohms.

    ``rows`` carries frequency in GHz (the internal unit)."""
    _validate_impedance_rows(rows)
    with _atomic_text_file(path) as f:
        _write_impedance_rows(f, rows, include_header)


def _validate_impedance_rows(
    rows: list[tuple[float, float, float]],
    context: str = "Impedance export",
) -> None:
    frequencies = [float(row[0]) for row in rows]
    _validate_frequency_rows(frequencies, context)
    for index, (_freq_ghz, zr, zi) in enumerate(rows, start=1):
        _validate_surface_impedance(complex(zr, zi), f"{context} row {index}")


def _write_impedance_rows(
    stream: TextIO,
    rows: list[tuple[float, float, float]],
    include_header: bool,
) -> None:
    if include_header:
        stream.write(IMPEDANCE_HEADER + "\n")
    for freq_ghz, zr, zi in rows:
        stream.write(
            f"{freq_ghz * HZ_PER_GHZ:.12g},{zr:.12g},{zi:.12g}\n"
        )


def write_impedance_batch(
    outputs: list[tuple[Path, list[tuple[float, float, float]]]],
    include_header: bool = True,
) -> None:
    """Atomically publish a set of nominal solver-compatible IBC CSVs."""

    if not outputs:
        raise ValueError("Impedance output batch is empty.")
    entries: list[tuple[Path, Callable[[TextIO], None]]] = []
    for output_index, (path, rows) in enumerate(outputs, start=1):
        _validate_impedance_rows(rows, f"Impedance export {output_index}")

        def write_nominal(
            stream: TextIO,
            batch_rows: list[tuple[float, float, float]] = rows,
        ) -> None:
            _write_impedance_rows(stream, batch_rows, include_header)

        entries.append((Path(path), write_nominal))
    _atomic_text_batch(entries)


def uncertainty_report_path(nominal_path: Path) -> Path:
    """Return the non-solver uncertainty sidecar path for a nominal IBC CSV."""
    suffix = nominal_path.suffix or ".csv"
    return nominal_path.with_name(f"{nominal_path.stem}_uncertainty{suffix}")


def write_impedance_uncertainty_report(
    path: Path,
    rows: list[tuple[float, float, float, float, float, float, float]],
) -> None:
    """Write nominal/min/max impedance values to an analysis-only sidecar."""
    frequencies = [float(row[0]) for row in rows]
    _validate_frequency_rows(frequencies, "Impedance uncertainty report")
    for index, row in enumerate(rows, start=1):
        _, zr, zi, zr_min, zr_max, zi_min, zi_max = row
        values = (zr, zi, zr_min, zr_max, zi_min, zi_max)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(
                f"Impedance uncertainty report row {index} contains "
                "a non-finite value."
            )
        _validate_surface_impedance(
            complex(zr, zi), f"Impedance uncertainty report row {index}"
        )
        if zr_min > zr_max or zi_min > zi_max:
            raise ValueError(
                f"Impedance uncertainty report row {index} has inverted bounds."
            )
        if zr_min < -_passivity_tolerance(complex(zr_min, zi)):
            raise ValueError(
                f"Impedance uncertainty report row {index} contains negative "
                f"minimum resistance {zr_min:g} ohm."
            )
    with _atomic_text_file(path) as f:
        f.write(IMPEDANCE_UNCERTAINTY_HEADER + "\n")
        for row in rows:
            freq_hz = row[0] * HZ_PER_GHZ
            values = (freq_hz, *row[1:])
            f.write(",".join(f"{value:.12g}" for value in values) + "\n")


def write_impedance_bundle(
    nominal_path: Path,
    nominal_rows: list[tuple[float, float, float]],
    include_header: bool,
    uncertainty_path: Path | None = None,
    uncertainty_rows: (
        list[tuple[float, float, float, float, float, float, float]] | None
    ) = None,
) -> None:
    """Atomically publish a nominal IBC and its optional uncertainty sidecar."""

    _validate_impedance_rows(nominal_rows)
    frequencies = [float(row[0]) for row in nominal_rows]

    if (uncertainty_path is None) != (uncertainty_rows is None):
        raise ValueError(
            "Uncertainty output path and rows must either both be supplied or both omitted."
        )
    if uncertainty_rows is not None:
        uncertainty_frequencies = [float(row[0]) for row in uncertainty_rows]
        _validate_frequency_rows(
            uncertainty_frequencies, "Impedance uncertainty report"
        )
        if uncertainty_frequencies != frequencies:
            raise ValueError(
                "Nominal and uncertainty impedance outputs must use the same frequency rows."
            )
        for index, row in enumerate(uncertainty_rows, start=1):
            _, zr, zi, zr_min, zr_max, zi_min, zi_max = row
            values = (zr, zi, zr_min, zr_max, zi_min, zi_max)
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(
                    f"Impedance uncertainty report row {index} contains a non-finite value."
                )
            _validate_surface_impedance(
                complex(zr, zi), f"Impedance uncertainty report row {index}"
            )
            if zr_min > zr_max or zi_min > zi_max:
                raise ValueError(
                    f"Impedance uncertainty report row {index} has inverted bounds."
                )
            if zr_min < -_passivity_tolerance(complex(zr_min, zi)):
                raise ValueError(
                    f"Impedance uncertainty report row {index} contains negative "
                    f"minimum resistance {zr_min:g} ohm."
                )

    def write_nominal(stream: TextIO) -> None:
        _write_impedance_rows(stream, nominal_rows, include_header)

    entries: list[tuple[Path, Callable[[TextIO], None]]] = [
        (Path(nominal_path), write_nominal)
    ]
    if uncertainty_path is not None and uncertainty_rows is not None:
        def write_uncertainty(stream: TextIO) -> None:
            stream.write(IMPEDANCE_UNCERTAINTY_HEADER + "\n")
            for row in uncertainty_rows:
                values = (row[0] * HZ_PER_GHZ, *row[1:])
                stream.write(
                    ",".join(f"{value:.12g}" for value in values) + "\n"
                )

        entries.append((Path(uncertainty_path), write_uncertainty))
    _atomic_text_batch(entries)


def write_material_table(path: Path, table: MaterialTable, include_header: bool = True) -> None:
    """Write a MaterialTable in the 5-column property-file format
    (frequency_hz,eps_real,eps_imag,mu_real,mu_imag) so a blended material can
    be reloaded as a normal layer material. The imaginary parts are written as
    stored (already signed for the e^{+jwt} convention used by
    ``read_material_table``)."""
    n = len(table.freq_ghz)
    if not (len(table.eps_r) == len(table.mu_r) == n):
        raise ValueError("MaterialTable columns must have matching lengths.")
    _validate_frequency_rows(table.freq_ghz, "Material export")
    for index, (eps, mu) in enumerate(
        zip(table.eps_r, table.mu_r), start=1
    ):
        _validate_medium(eps, mu, f"Material export row {index}")
    with _atomic_text_file(path) as f:
        if include_header:
            f.write(MATERIAL_HEADER + "\n")
        for freq_ghz, eps, mu in zip(table.freq_ghz, table.eps_r, table.mu_r):
            f.write(
                f"{freq_ghz * HZ_PER_GHZ:.12g},{eps.real:.12g},{eps.imag:.12g},"
                f"{mu.real:.12g},{mu.imag:.12g}\n"
            )


def layer_config_to_dict(layer: LayerConfig) -> dict[str, Any]:
    d: dict[str, Any] = {
        "thickness_in": layer.thickness_in,
        "anisotropic": layer.anisotropic,
        "file_0deg": layer.file_0deg,
        "file_90deg": layer.file_90deg,
        "polarization_deg": layer.polarization_deg,
    }
    if layer.is_sheet:
        d["is_sheet"] = True
        d["sheet_resistance"] = layer.sheet_resistance
        if layer.inv_rs_min is not None:
            d["inv_rs_min"] = layer.inv_rs_min
        if layer.inv_rs_max is not None:
            d["inv_rs_max"] = layer.inv_rs_max
        if layer.inv_rs_accuracy is not None:
            d["inv_rs_accuracy"] = layer.inv_rs_accuracy
    if layer.inv_t_min_in is not None:
        d["inv_t_min_in"] = layer.inv_t_min_in
    if layer.inv_t_max_in is not None:
        d["inv_t_max_in"] = layer.inv_t_max_in
    if layer.inv_t_accuracy_in is not None:
        d["inv_t_accuracy_in"] = layer.inv_t_accuracy_in
    return d


def _validate_search_bounds(
    lo: float | None,
    hi: float | None,
    acc: float | None,
    label: str,
    prefix: str,
) -> None:
    """Shared validation for an inverse-design search variable: bounds come as a
    pair (both or neither), are positive, and ordered; the accuracy increment is
    positive when set."""
    if (lo is None) != (hi is None):
        raise ValueError(f"{label}: set both {prefix}_min and {prefix}_max, or neither.")
    if lo is not None and lo <= 0:
        raise ValueError(f"{label}: {prefix}_min must be > 0.")
    if hi is not None and hi <= 0:
        raise ValueError(f"{label}: {prefix}_max must be > 0.")
    if lo is not None and hi is not None and hi < lo:
        raise ValueError(f"{label}: {prefix}_max must be >= {prefix}_min.")
    if acc is not None and acc <= 0:
        raise ValueError(f"{label}: {prefix}_accuracy must be > 0.")


def layer_config_from_dict(data: dict[str, Any], index: int = 0) -> LayerConfig:
    label = f"Layer {index}" if index > 0 else "Layer"

    is_sheet = bool(data.get("is_sheet", False))
    if is_sheet:
        try:
            sheet_resistance = float(data.get("sheet_resistance", 0.0))
        except Exception as exc:
            raise ValueError(f"{label}: invalid sheet_resistance.") from exc
        if not math.isfinite(sheet_resistance) or sheet_resistance <= 0:
            raise ValueError(f"{label}: sheet_resistance must be > 0.")
        inv_rs_min = _parse_optional_float(data.get("inv_rs_min"), label, "inv_rs_min")
        inv_rs_max = _parse_optional_float(data.get("inv_rs_max"), label, "inv_rs_max")
        inv_rs_accuracy = _parse_optional_float(
            data.get("inv_rs_accuracy"), label, "inv_rs_accuracy"
        )
        _validate_search_bounds(inv_rs_min, inv_rs_max, inv_rs_accuracy, label, "inv_rs")
        return LayerConfig(
            thickness_in=0.0,
            anisotropic=False,
            file_0deg="",
            file_90deg="",
            polarization_deg=0.0,
            is_sheet=True,
            sheet_resistance=sheet_resistance,
            inv_rs_min=inv_rs_min,
            inv_rs_max=inv_rs_max,
            inv_rs_accuracy=inv_rs_accuracy,
        )

    try:
        thickness_in = float(data.get("thickness_in", 0.0))
    except Exception as exc:
        raise ValueError(f"{label}: invalid thickness_in.") from exc
    if not math.isfinite(thickness_in) or thickness_in <= 0:
        raise ValueError(f"{label}: thickness_in must be > 0.")

    anisotropic = bool(data.get("anisotropic", False))
    file_0deg = str(data.get("file_0deg", "")).strip()
    file_90deg = str(data.get("file_90deg", "")).strip()
    try:
        polarization_deg = float(data.get("polarization_deg", 0.0))
    except Exception as exc:
        raise ValueError(f"{label}: invalid polarization_deg.") from exc
    if not math.isfinite(polarization_deg):
        raise ValueError(f"{label}: polarization_deg must be finite.")
    if anisotropic:
        axis = polarization_deg % 180.0
        if not (
            min(abs(axis), abs(axis - 180.0)) <= 1e-9
            or abs(axis - 90.0) <= 1e-9
        ):
            raise ValueError(
                f"{label}: directional layers require a principal-axis "
                "polarization_deg of 0 or 90."
            )

    if not file_0deg:
        raise ValueError(f"{label}: file_0deg is required.")
    if anisotropic and not file_90deg:
        raise ValueError(f"{label}: file_90deg is required for anisotropic layers.")

    inv_t_min_in = _parse_optional_float(data.get("inv_t_min_in"), label, "inv_t_min_in")
    inv_t_max_in = _parse_optional_float(data.get("inv_t_max_in"), label, "inv_t_max_in")
    inv_t_accuracy_in = _parse_optional_float(
        data.get("inv_t_accuracy_in"), label, "inv_t_accuracy_in"
    )
    _validate_search_bounds(inv_t_min_in, inv_t_max_in, inv_t_accuracy_in, label, "inv_t")

    return LayerConfig(
        thickness_in=thickness_in,
        anisotropic=anisotropic,
        file_0deg=file_0deg,
        file_90deg=file_90deg,
        polarization_deg=polarization_deg,
        inv_t_min_in=inv_t_min_in,
        inv_t_max_in=inv_t_max_in,
        inv_t_accuracy_in=inv_t_accuracy_in,
    )


def _parse_optional_float(value: Any, label: str, field: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{label}: invalid {field}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label}: {field} must be finite.")
    return parsed


def _project_path_slots(
    state: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any], str]]:
    """Yield the known FREDDY file-path fields without guessing by suffix."""

    layers = state.get("layers", [])
    if isinstance(layers, list):
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                continue
            for key in ("file_0deg", "file_90deg"):
                if key in layer:
                    yield f"layers[{index}].{key}", layer, key

    controls = state.get("controls", {})
    if isinstance(controls, dict):
        for key in _PROJECT_CONTROL_PATH_KEYS:
            if key in controls:
                yield f"controls.{key}", controls, key

    mixes = state.get("mixes", {})
    if isinstance(mixes, dict):
        components = mixes.get("components", [])
        if isinstance(components, list):
            for index, component in enumerate(components):
                if isinstance(component, dict) and "file" in component:
                    yield f"mixes.components[{index}].file", component, "file"


def _is_absolute_on_any_platform(value: str) -> bool:
    """Recognize native paths plus absolute Windows/POSIX paths in moved JSON."""

    return (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    )


def _native_project_path(value: str, project_directory: Path) -> Path | None:
    """Return a native absolute candidate, or ``None`` for a foreign absolute."""

    native = Path(value).expanduser()
    if native.is_absolute():
        return native.resolve(strict=False)
    if _is_absolute_on_any_platform(value):
        return None

    # New project files always use forward slashes for relative paths.  Accept
    # backslashes as separators too so an older Windows project can move to a
    # Mac without treating the complete relative path as one filename.
    relative_value = value.replace("\\", "/")
    return (project_directory / Path(relative_value)).resolve(strict=False)


def _leading_parent_hops(path: Path) -> int:
    hops = 0
    for part in path.parts:
        if part != "..":
            break
        hops += 1
    return hops


def _portable_project_path(
    value: object,
    project_directory: Path,
) -> tuple[str, bool]:
    """Encode one path relative to a project when it stays within/near it.

    One leading ``..`` is allowed so a project directory and a neighboring
    ``materials`` or ``outputs`` directory can move together.  More distant or
    cross-volume paths remain absolute and are identified in project metadata.
    """

    text = str(value).strip()
    if not text:
        return "", False

    native = Path(text).expanduser()
    if native.is_absolute():
        candidate = native.resolve(strict=False)
    elif _is_absolute_on_any_platform(text):
        return text, True
    else:
        # Live FREDDY controls historically interpret relative paths against
        # the process working directory.  Resolve that exact runtime meaning
        # before encoding it relative to the project; interpreting the same
        # spelling against the new project folder can silently select a
        # different same-named material and change the physics.
        candidate = (
            Path.cwd() / Path(text.replace("\\", "/"))
        ).resolve(strict=False)

    try:
        relative = Path(os.path.relpath(candidate, project_directory))
    except ValueError:
        return str(candidate), True

    if _leading_parent_hops(relative) <= PROJECT_PATH_PARENT_HOPS:
        return relative.as_posix(), False
    return str(candidate), True


def _portable_project_state(
    state: dict[str, Any], project_directory: Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    portable_state = copy.deepcopy(state)
    external_paths: list[dict[str, str]] = []
    for field, container, key in _project_path_slots(portable_state):
        encoded, external = _portable_project_path(
            container.get(key, ""), project_directory
        )
        container[key] = encoded
        if external:
            external_paths.append({"field": field, "path": encoded})
    return portable_state, external_paths


def _resolved_project_state(
    state: dict[str, Any], project_directory: Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    resolved_state = copy.deepcopy(state)
    external_paths: list[dict[str, str]] = []
    for field, container, key in _project_path_slots(resolved_state):
        value = str(container.get(key, "")).strip()
        if not value:
            container[key] = ""
            continue
        if _is_absolute_on_any_platform(value):
            # Preserve external absolute paths verbatim, including paths from a
            # different operating system.  The user can repair one explicitly
            # without the loader silently rebasing it to an unrelated file.
            container[key] = value
            external_paths.append({"field": field, "path": value})
            continue
        candidate = _native_project_path(value, project_directory)
        assert candidate is not None
        container[key] = str(candidate)
    return resolved_state, external_paths


def save_project_file(path: Path, state: dict[str, Any]) -> None:
    destination = Path(path)
    project_directory = destination.expanduser().resolve(strict=False).parent
    portable_state, external_paths = _portable_project_state(
        state, project_directory
    )
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "path_portability": {
            "version": PROJECT_PATH_POLICY_VERSION,
            "base": "project_file_directory",
            "nearby_parent_hops": PROJECT_PATH_PARENT_HOPS,
            "external_absolute_paths": external_paths,
        },
        "state": portable_state,
    }
    with _atomic_text_file(destination) as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_project_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Project file must be a JSON object.")

    schema_version = payload.get("schema_version")
    if schema_version != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema version: {schema_version}. "
            f"Expected {PROJECT_SCHEMA_VERSION}."
        )

    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("Project file is missing a valid 'state' object.")

    portability = payload.get("path_portability")
    if not isinstance(portability, dict):
        warnings.warn(
            "Legacy FREDDY project has no path-portability metadata; relative "
            "paths retain their historical working-directory meaning. Save "
            "the project again to migrate them explicitly.",
            RuntimeWarning,
            stacklevel=2,
        )
        return copy.deepcopy(state)
    if (
        portability.get("version") != PROJECT_PATH_POLICY_VERSION
        or portability.get("base") != "project_file_directory"
    ):
        raise ValueError(
            "Unsupported FREDDY project path-portability policy."
        )

    project_directory = Path(path).expanduser().resolve(strict=False).parent
    resolved_state, external_paths = _resolved_project_state(
        state, project_directory
    )
    if external_paths:
        fields = ", ".join(item["field"] for item in external_paths)
        warnings.warn(
            "FREDDY project contains external absolute path(s) that cannot "
            f"move with the project folder: {fields}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return resolved_state
