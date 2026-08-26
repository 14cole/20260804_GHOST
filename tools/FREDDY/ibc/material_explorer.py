"""Read-only comparison model and Qt workspace for FREDDY material tables.

The explorer deliberately reuses :func:`ibc.io.read_material_table` so it has
exactly the same units, sign convention, and validation rules as every FREDDY
solver workflow.  Loaded sources are session-only and never mutate a layer
stack or project file.
"""

from __future__ import annotations

import bisect
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .compute import MaterialTable, interp_complex
from .io import read_material_table


PathLike = str | os.PathLike[str]
SourceRequest = PathLike | tuple[PathLike, str]
FileFingerprint = tuple[int, int]


def _path_identity(path: Path) -> str:
    """Return a resolved identity with the host filesystem's case semantics."""

    # normcase folds case on Windows and preserves it on POSIX. An additional
    # casefold would incorrectly merge distinct A.csv/a.csv files on a
    # case-sensitive macOS or Linux filesystem.
    return os.path.normcase(str(path.resolve(strict=False)))


def _file_fingerprint(path: Path) -> FileFingerprint | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def loss_tangent(value: complex) -> float | None:
    """Return passive loss tangent ``-imag/real`` when it is well-defined."""

    if value.real <= 0.0:
        return None
    result = -value.imag / value.real
    return 0.0 if result == 0.0 else result


def _numeric_range(values: Iterable[float]) -> tuple[float, float]:
    iterator = iter(values)
    first = next(iterator)
    low = high = first
    for value in iterator:
        low = min(low, value)
        high = max(high, value)
    return low, high


def _optional_numeric_range(
    values: Iterable[float | None],
) -> tuple[float, float] | None:
    low: float | None = None
    high: float | None = None
    for value in values:
        if value is None:
            continue
        low = value if low is None else min(low, value)
        high = value if high is None else max(high, value)
    return None if low is None or high is None else (low, high)


def extrema_preserving_indices(
    values: Sequence[float],
    limit: int = 5000,
) -> list[int]:
    """Select ordered plot indices while retaining each bucket's extrema.

    Uniform stride sampling can entirely omit a narrow resonance. This method
    always keeps the endpoints plus the minimum and maximum in each bucket.
    """

    length = len(values)
    if limit < 4:
        raise ValueError("Plot decimation limit must be at least 4.")
    if length <= limit:
        return list(range(length))

    interior_count = length - 2
    bucket_count = max(1, (limit - 2) // 2)
    selected = {0, length - 1}
    for bucket in range(bucket_count):
        start = 1 + (bucket * interior_count) // bucket_count
        stop = 1 + ((bucket + 1) * interior_count) // bucket_count
        if stop <= start:
            continue
        segment = range(start, stop)
        selected.add(min(segment, key=values.__getitem__))
        selected.add(max(segment, key=values.__getitem__))
    return sorted(selected)


@dataclass(frozen=True)
class MaterialExplorerSource:
    """One validated material file cached for an explorer session."""

    key: str
    path: Path
    label: str
    table: MaterialTable
    fingerprint: FileFingerprint | None
    color_index: int


@dataclass(frozen=True)
class MaterialExplorerSummary:
    """Compact direct-property and derived-loss ranges for one source."""

    label: str
    path: Path
    sample_count: int
    frequency_min_ghz: float
    frequency_max_ghz: float
    eps_real_range: tuple[float, float]
    eps_imag_range: tuple[float, float]
    mu_real_range: tuple[float, float]
    mu_imag_range: tuple[float, float]
    electric_loss_tangent_range: tuple[float, float] | None
    magnetic_loss_tangent_range: tuple[float, float] | None


@dataclass(frozen=True)
class MaterialExplorerSample:
    """One stored material row plus the two explicitly derived loss tangents."""

    label: str
    path: Path
    frequency_ghz: float
    eps_real: float
    eps_imag: float
    mu_real: float
    mu_imag: float
    electric_loss_tangent: float | None
    magnetic_loss_tangent: float | None


class MaterialExplorerModel:
    """Ordered, de-duplicated cache of validated FREDDY material files."""

    def __init__(
        self,
        reader: Callable[[Path], MaterialTable] = read_material_table,
    ) -> None:
        self._reader = reader
        self._sources: list[MaterialExplorerSource] = []
        self._summary_cache: dict[str, MaterialExplorerSummary] = {}
        self._next_color_index = 0

    @property
    def sources(self) -> tuple[MaterialExplorerSource, ...]:
        return tuple(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    def add_path(
        self,
        path: PathLike,
        label: str | None = None,
    ) -> MaterialExplorerSource | None:
        """Validate and add ``path``; return ``None`` when already loaded."""

        resolved = Path(path).expanduser().resolve(strict=False)
        key = _path_identity(resolved)
        if any(source.key == key for source in self._sources):
            return None

        # Bracket the read so a file being replaced during parsing cannot be
        # associated with metadata from a different version.
        before = _file_fingerprint(resolved)
        table = self._reader(resolved)
        after = _file_fingerprint(resolved)
        if before != after:
            raise RuntimeError(
                f"{resolved}: file changed while it was being read; try again."
            )
        shown_label = str(label).strip() if label is not None else ""
        source = MaterialExplorerSource(
            key=key,
            path=resolved,
            label=shown_label or resolved.name,
            table=table,
            fingerprint=after,
            color_index=self._next_color_index,
        )
        self._next_color_index += 1
        self._sources.append(source)
        return source

    def add_requests(
        self,
        requests: Iterable[SourceRequest],
    ) -> tuple[list[MaterialExplorerSource], list[Path], list[tuple[Path, Exception]]]:
        """Partially accept a batch and report additions, duplicates, and errors."""

        added: list[MaterialExplorerSource] = []
        duplicates: list[Path] = []
        errors: list[tuple[Path, Exception]] = []
        for request in requests:
            if isinstance(request, tuple):
                raw_path, label = request
            else:
                raw_path, label = request, None
            path = Path(raw_path).expanduser().resolve(strict=False)
            try:
                source = self.add_path(path, label)
            except Exception as exc:
                errors.append((path, exc))
            else:
                if source is None:
                    duplicates.append(path)
                else:
                    added.append(source)
        return added, duplicates, errors

    def source_for_key(self, key: str) -> MaterialExplorerSource | None:
        return next((source for source in self._sources if source.key == key), None)

    def remove_keys(self, keys: Iterable[str]) -> int:
        selected = {str(key) for key in keys}
        previous = len(self._sources)
        self._sources = [source for source in self._sources if source.key not in selected]
        for key in selected:
            self._summary_cache.pop(key, None)
        return previous - len(self._sources)

    def clear(self) -> None:
        self._sources.clear()
        self._summary_cache.clear()
        self._next_color_index = 0

    def reload_key(self, key: str) -> MaterialExplorerSource:
        """Reload one source atomically, retaining cached data on failure."""

        for index, source in enumerate(self._sources):
            if source.key != key:
                continue
            before = _file_fingerprint(source.path)
            table = self._reader(source.path)
            after = _file_fingerprint(source.path)
            if before != after:
                raise RuntimeError(
                    f"{source.path}: file changed while it was being read; try again."
                )
            replacement = MaterialExplorerSource(
                key=source.key,
                path=source.path,
                label=source.label,
                table=table,
                fingerprint=after,
                color_index=source.color_index,
            )
            self._sources[index] = replacement
            self._summary_cache.pop(source.key, None)
            return replacement
        raise KeyError(f"Unknown material explorer source: {key}")

    def source_state(self, source: MaterialExplorerSource) -> str:
        """Return ``current``, ``changed``, or ``missing`` for a cached source."""

        current = _file_fingerprint(source.path)
        if current is None:
            return "missing"
        if source.fingerprint is None or current != source.fingerprint:
            return "changed"
        return "current"

    def common_frequency_range(self) -> tuple[float, float] | None:
        """Return the inclusive overlap of every source without extrapolation."""

        if not self._sources:
            return None
        low = max(source.table.freq_ghz[0] for source in self._sources)
        high = min(source.table.freq_ghz[-1] for source in self._sources)
        return (low, high) if low <= high else None

    def summary(self, source: MaterialExplorerSource) -> MaterialExplorerSummary:
        cached = self._summary_cache.get(source.key)
        if cached is not None:
            return cached
        summary = MaterialExplorerSummary(
            label=source.label,
            path=source.path,
            sample_count=len(source.table.freq_ghz),
            frequency_min_ghz=source.table.freq_ghz[0],
            frequency_max_ghz=source.table.freq_ghz[-1],
            eps_real_range=_numeric_range(value.real for value in source.table.eps_r),
            eps_imag_range=_numeric_range(value.imag for value in source.table.eps_r),
            mu_real_range=_numeric_range(value.real for value in source.table.mu_r),
            mu_imag_range=_numeric_range(value.imag for value in source.table.mu_r),
            electric_loss_tangent_range=_optional_numeric_range(
                loss_tangent(value) for value in source.table.eps_r
            ),
            magnetic_loss_tangent_range=_optional_numeric_range(
                loss_tangent(value) for value in source.table.mu_r
            ),
        )
        self._summary_cache[source.key] = summary
        return summary

    def sample_at(self, source_index: int, sample_index: int) -> MaterialExplorerSample:
        source = self._sources[source_index]
        eps = source.table.eps_r[sample_index]
        mu = source.table.mu_r[sample_index]
        return MaterialExplorerSample(
            label=source.label,
            path=source.path,
            frequency_ghz=source.table.freq_ghz[sample_index],
            eps_real=eps.real,
            eps_imag=eps.imag,
            mu_real=mu.real,
            mu_imag=mu.imag,
            electric_loss_tangent=loss_tangent(eps),
            magnetic_loss_tangent=loss_tangent(mu),
        )

    def sample_at_frequency(
        self,
        source: MaterialExplorerSource,
        frequency_ghz: float,
    ) -> MaterialExplorerSample | None:
        """Linearly interpolate one source, returning ``None`` out of range."""

        frequencies = source.table.freq_ghz
        if frequency_ghz < frequencies[0] or frequency_ghz > frequencies[-1]:
            return None
        eps = interp_complex(frequency_ghz, frequencies, source.table.eps_r)
        mu = interp_complex(frequency_ghz, frequencies, source.table.mu_r)
        return MaterialExplorerSample(
            label=source.label,
            path=source.path,
            frequency_ghz=frequency_ghz,
            eps_real=eps.real,
            eps_imag=eps.imag,
            mu_real=mu.real,
            mu_imag=mu.imag,
            electric_loss_tangent=loss_tangent(eps),
            magnetic_loss_tangent=loss_tangent(mu),
        )

    def raw_row_count(self) -> int:
        return sum(len(source.table.freq_ghz) for source in self._sources)


try:
    from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
    from PySide6.QtGui import QColor, QIcon, QPixmap
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QSplitter,
        QTabWidget,
        QTableView,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except Exception:  # pragma: no cover - pure model remains usable without Qt
    QT_AVAILABLE = False
    QAbstractTableModel = QWidget = object  # type: ignore[assignment,misc]


if QT_AVAILABLE:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure

        from .plot import style_axis

        MPL_AVAILABLE = True
    except Exception:  # pragma: no cover - Qt table explorer still remains usable
        MPL_AVAILABLE = False
else:
    MPL_AVAILABLE = False


def _format_number(value: float) -> str:
    return f"{value:.7g}"


def _format_range(values: tuple[float, float] | None) -> str:
    if values is None:
        return "N/A"
    low, high = values
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-15):
        return _format_number(low)
    return f"{_format_number(low)} to {_format_number(high)}"


if QT_AVAILABLE:

    class _SummaryTableModel(QAbstractTableModel):
        HEADERS = (
            "Material",
            "Samples",
            "Coverage (GHz)",
            "Epsilon real range",
            "Epsilon imag range",
            "Mu real range",
            "Mu imag range",
            "Electric tan delta range (derived)",
            "Magnetic tan delta range (derived)",
            "Source",
        )

        def __init__(self, explorer: MaterialExplorerModel) -> None:
            super().__init__()
            self.explorer = explorer
            self.rows: list[MaterialExplorerSummary] = []
            self.refresh()

        def refresh(self) -> None:
            self.beginResetModel()
            self.rows = [self.explorer.summary(source) for source in self.explorer.sources]
            self.endResetModel()

        def rowCount(self, _parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
            return len(self.rows)

        def columnCount(self, _parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
            return len(self.HEADERS)

        def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: ANN201
            if not index.isValid() or not (0 <= index.row() < len(self.rows)):
                return None
            row = self.rows[index.row()]
            if role == Qt.ToolTipRole:
                return str(row.path)
            if role == Qt.TextAlignmentRole and index.column() not in (0, 9):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            if role != Qt.DisplayRole:
                return None
            coverage = (
                f"{_format_number(row.frequency_min_ghz)} to "
                f"{_format_number(row.frequency_max_ghz)}"
            )
            values = (
                row.label,
                str(row.sample_count),
                coverage,
                _format_range(row.eps_real_range),
                _format_range(row.eps_imag_range),
                _format_range(row.mu_real_range),
                _format_range(row.mu_imag_range),
                _format_range(row.electric_loss_tangent_range),
                _format_range(row.magnetic_loss_tangent_range),
                str(row.path),
            )
            return values[index.column()]

        def headerData(  # noqa: N802, ANN201
            self,
            section: int,
            orientation: Qt.Orientation,
            role: int = Qt.DisplayRole,
        ):
            if role == Qt.DisplayRole and orientation == Qt.Horizontal:
                return self.HEADERS[section]
            return super().headerData(section, orientation, role)


    class _RawTableModel(QAbstractTableModel):
        HEADERS = (
            "Material",
            "Frequency (GHz)",
            "Epsilon real",
            "Epsilon imag",
            "Mu real",
            "Mu imag",
            "Electric tan delta (derived)",
            "Magnetic tan delta (derived)",
        )

        def __init__(self, explorer: MaterialExplorerModel) -> None:
            super().__init__()
            self.explorer = explorer
            self.offsets: list[int] = []
            self.refresh()

        def refresh(self) -> None:
            self.beginResetModel()
            running = 0
            self.offsets = []
            for source in self.explorer.sources:
                running += len(source.table.freq_ghz)
                self.offsets.append(running)
            self.endResetModel()

        def rowCount(self, _parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
            return self.offsets[-1] if self.offsets else 0

        def columnCount(self, _parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
            return len(self.HEADERS)

        def _sample_for_row(self, row: int) -> MaterialExplorerSample:
            source_index = bisect.bisect_right(self.offsets, row)
            prior = self.offsets[source_index - 1] if source_index else 0
            return self.explorer.sample_at(source_index, row - prior)

        def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: ANN201
            if not index.isValid() or not (0 <= index.row() < self.rowCount()):
                return None
            sample = self._sample_for_row(index.row())
            if role == Qt.ToolTipRole:
                return str(sample.path)
            if role == Qt.TextAlignmentRole and index.column() != 0:
                return int(Qt.AlignRight | Qt.AlignVCenter)
            if role != Qt.DisplayRole:
                return None
            values = (
                sample.label,
                _format_number(sample.frequency_ghz),
                _format_number(sample.eps_real),
                _format_number(sample.eps_imag),
                _format_number(sample.mu_real),
                _format_number(sample.mu_imag),
                "N/A"
                if sample.electric_loss_tangent is None
                else _format_number(sample.electric_loss_tangent),
                "N/A"
                if sample.magnetic_loss_tangent is None
                else _format_number(sample.magnetic_loss_tangent),
            )
            return values[index.column()]

        def headerData(  # noqa: N802, ANN201
            self,
            section: int,
            orientation: Qt.Orientation,
            role: int = Qt.DisplayRole,
        ):
            if role == Qt.DisplayRole and orientation == Qt.Horizontal:
                return self.HEADERS[section]
            return super().headerData(section, orientation, role)


    class _FrequencyComparisonTableModel(QAbstractTableModel):
        HEADERS = (
            "Material",
            "Status",
            "Epsilon real",
            "Epsilon imag",
            "Mu real",
            "Mu imag",
            "Electric tan delta (derived)",
            "Magnetic tan delta (derived)",
            "Coverage (GHz)",
            "Source",
        )

        def __init__(self, explorer: MaterialExplorerModel) -> None:
            super().__init__()
            self.explorer = explorer
            self.frequency_ghz: float | None = None
            self.rows: list[
                tuple[MaterialExplorerSource, MaterialExplorerSample | None]
            ] = []
            self.refresh()

        def set_frequency(self, frequency_ghz: float | None) -> None:
            self.frequency_ghz = frequency_ghz
            self.refresh()

        def refresh(self) -> None:
            self.beginResetModel()
            self.rows = [
                (
                    source,
                    None
                    if self.frequency_ghz is None
                    else self.explorer.sample_at_frequency(
                        source, self.frequency_ghz
                    ),
                )
                for source in self.explorer.sources
            ]
            self.endResetModel()

        def rowCount(self, _parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
            return len(self.rows)

        def columnCount(self, _parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
            return len(self.HEADERS)

        def _status(
            self,
            source: MaterialExplorerSource,
            sample: MaterialExplorerSample | None,
        ) -> str:
            if self.frequency_ghz is None:
                return "Choose a frequency"
            if sample is None:
                return "Out of range"
            index = bisect.bisect_left(source.table.freq_ghz, self.frequency_ghz)
            if (
                index < len(source.table.freq_ghz)
                and source.table.freq_ghz[index] == self.frequency_ghz
            ):
                return "Stored sample"
            return "Linear interpolation"

        def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: ANN201
            if not index.isValid() or not (0 <= index.row() < len(self.rows)):
                return None
            source, sample = self.rows[index.row()]
            if role == Qt.ToolTipRole:
                return str(source.path)
            if role == Qt.TextAlignmentRole and index.column() not in (0, 1, 9):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            if role != Qt.DisplayRole:
                return None
            coverage = (
                f"{_format_number(source.table.freq_ghz[0])} to "
                f"{_format_number(source.table.freq_ghz[-1])}"
            )
            unavailable = "--"
            values = (
                source.label,
                self._status(source, sample),
                unavailable if sample is None else _format_number(sample.eps_real),
                unavailable if sample is None else _format_number(sample.eps_imag),
                unavailable if sample is None else _format_number(sample.mu_real),
                unavailable if sample is None else _format_number(sample.mu_imag),
                unavailable
                if sample is None or sample.electric_loss_tangent is None
                else _format_number(sample.electric_loss_tangent),
                unavailable
                if sample is None or sample.magnetic_loss_tangent is None
                else _format_number(sample.magnetic_loss_tangent),
                coverage,
                str(source.path),
            )
            return values[index.column()]

        def headerData(  # noqa: N802, ANN201
            self,
            section: int,
            orientation: Qt.Orientation,
            role: int = Qt.DisplayRole,
        ):
            if role == Qt.DisplayRole and orientation == Qt.Horizontal:
                return self.HEADERS[section]
            return super().headerData(section, orientation, role)


    class MaterialExplorerWidget(QWidget):
        """Read-only, session-scoped material comparison workspace."""

        def __init__(
            self,
            *,
            presets: Mapping[str, PathLike] | None = None,
            stack_source_provider: Callable[[], Iterable[SourceRequest]] | None = None,
            mix_source_provider: Callable[[], Iterable[SourceRequest]] | None = None,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.model = MaterialExplorerModel()
            self.presets = dict(presets or {})
            self.stack_source_provider = stack_source_provider
            self.mix_source_provider = mix_source_provider
            self._colors: Mapping[str, object] = {}
            self._last_folder = ""
            self._last_load_feedback = ""
            self.setAcceptDrops(True)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(8)

            title = QLabel("Material Explorer")
            title.setObjectName("ExplorerTitle")
            outer.addWidget(title)
            description = QLabel(
                "Compare FREDDY material CSVs without changing the layer "
                "stack or running a solver. Curves use each file's stored frequency "
                "grid; the tables preserve the signed values from the file."
            )
            description.setWordWrap(True)
            outer.addWidget(description)
            convention = QLabel(
                "Convention: e^(+j omega t). Passive loss uses negative imaginary "
                "epsilon and mu. Tan delta is the only derived value shown "
                "(-imaginary/real); all other values are direct file data."
            )
            convention.setObjectName("ExplorerNote")
            convention.setWordWrap(True)
            outer.addWidget(convention)

            add_row = QHBoxLayout()
            self.add_files_button = QPushButton("Add material files...")
            self.add_files_button.clicked.connect(self._choose_files)
            add_row.addWidget(self.add_files_button)
            self.add_stack_button = QPushButton("Add current stack materials")
            self.add_stack_button.setEnabled(stack_source_provider is not None)
            self.add_stack_button.clicked.connect(
                lambda: self._add_from_provider(
                    self.stack_source_provider,
                    "The current stack has no measured material files.",
                )
            )
            add_row.addWidget(self.add_stack_button)
            self.add_mix_button = QPushButton("Add Material Mix inputs")
            self.add_mix_button.setEnabled(mix_source_provider is not None)
            self.add_mix_button.clicked.connect(
                lambda: self._add_from_provider(
                    self.mix_source_provider,
                    "Material Mix has no input files.",
                )
            )
            add_row.addWidget(self.add_mix_button)
            self.add_air_button = QPushButton("Add Air reference")
            self.add_air_button.setEnabled(self._air_preset() is not None)
            self.add_air_button.clicked.connect(self._add_air_reference)
            add_row.addWidget(self.add_air_button)
            add_row.addStretch(1)
            outer.addLayout(add_row)

            manage_row = QHBoxLayout()
            self.reload_button = QPushButton("Reload selected")
            self.reload_button.clicked.connect(self._reload_selected)
            manage_row.addWidget(self.reload_button)
            self.remove_button = QPushButton("Remove selected")
            self.remove_button.clicked.connect(self._remove_selected)
            manage_row.addWidget(self.remove_button)
            self.clear_button = QPushButton("Clear")
            self.clear_button.clicked.connect(self._clear)
            manage_row.addWidget(self.clear_button)
            manage_row.addStretch(1)
            self.status_label = QLabel("No materials loaded.")
            self.status_label.setObjectName("ExplorerStatus")
            manage_row.addWidget(self.status_label)
            outer.addLayout(manage_row)

            body = QSplitter(Qt.Horizontal)
            outer.addWidget(body, 1)

            sources_group = QGroupBox("Loaded material files")
            sources_layout = QVBoxLayout(sources_group)
            source_hint = QLabel(
                "Drop five-column material CSVs here. File timestamp/size changes "
                "are checked when this view opens; reload is always explicit."
            )
            source_hint.setWordWrap(True)
            sources_layout.addWidget(source_hint)
            self.source_list = QListWidget()
            self.source_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.source_list.itemSelectionChanged.connect(self._sync_action_state)
            sources_layout.addWidget(self.source_list, 1)
            body.addWidget(sources_group)

            self.detail_tabs = QTabWidget()
            body.addWidget(self.detail_tabs)
            body.setStretchFactor(0, 0)
            body.setStretchFactor(1, 1)
            body.setSizes([285, 715])

            chart_page = QWidget()
            chart_layout = QVBoxLayout(chart_page)
            chart_layout.setContentsMargins(4, 4, 4, 4)
            self.chart_note = QLabel("")
            self.chart_note.setObjectName("ExplorerNote")
            self.chart_note.setVisible(False)
            chart_layout.addWidget(self.chart_note)
            if MPL_AVAILABLE:
                self.figure = Figure(figsize=(8.5, 5.4), dpi=100, constrained_layout=True)
                axes = self.figure.subplots(2, 2, sharex=True)
                self.axes = [axes[row][column] for row in range(2) for column in range(2)]
                self.canvas = FigureCanvas(self.figure)
                self.chart_toolbar = NavigationToolbar(self.canvas, chart_page)
                chart_layout.addWidget(self.chart_toolbar)
                chart_layout.addWidget(self.canvas)
            else:
                self.figure = None
                self.axes = []
                self.canvas = None
                self.chart_toolbar = None
                missing = QLabel(
                    "Matplotlib is unavailable. Summary and raw data tables are "
                    "still available."
                )
                missing.setWordWrap(True)
                chart_layout.addWidget(missing)
                chart_layout.addStretch(1)
            self.detail_tabs.addTab(chart_page, "Property curves")

            comparison_page = QWidget()
            comparison_layout = QVBoxLayout(comparison_page)
            comparison_layout.setContentsMargins(6, 6, 6, 6)
            comparison_controls = QHBoxLayout()
            comparison_controls.addWidget(QLabel("Compare at"))
            self.compare_frequency_edit = QLineEdit()
            self.compare_frequency_edit.setMaximumWidth(110)
            self.compare_frequency_edit.setPlaceholderText("GHz")
            self.compare_frequency_edit.returnPressed.connect(
                self._apply_compare_frequency
            )
            comparison_controls.addWidget(self.compare_frequency_edit)
            comparison_controls.addWidget(QLabel("GHz"))
            self.compare_button = QPushButton("Update comparison")
            self.compare_button.clicked.connect(self._apply_compare_frequency)
            comparison_controls.addWidget(self.compare_button)
            self.shared_midpoint_button = QPushButton("Use shared midpoint")
            self.shared_midpoint_button.clicked.connect(self._use_shared_midpoint)
            comparison_controls.addWidget(self.shared_midpoint_button)
            comparison_controls.addStretch(1)
            comparison_layout.addLayout(comparison_controls)
            compare_note = QLabel(
                "Values between stored samples use component-wise linear "
                "interpolation. Values outside a file's coverage are marked "
                "Out of range; extrapolation is never used."
            )
            compare_note.setWordWrap(True)
            compare_note.setObjectName("ExplorerNote")
            comparison_layout.addWidget(compare_note)
            self.compare_status_label = QLabel("Add materials to compare.")
            comparison_layout.addWidget(self.compare_status_label)
            self.frequency_table = QTableView()
            self.frequency_model = _FrequencyComparisonTableModel(self.model)
            self.frequency_table.setModel(self.frequency_model)
            self._configure_table(self.frequency_table)
            comparison_layout.addWidget(self.frequency_table, 1)
            self.detail_tabs.addTab(comparison_page, "Compare at frequency")

            self.summary_table = QTableView()
            self.summary_model = _SummaryTableModel(self.model)
            self.summary_table.setModel(self.summary_model)
            self._configure_table(self.summary_table)
            self.summary_table.horizontalHeader().setStretchLastSection(True)
            self.detail_tabs.addTab(self.summary_table, "File summary")

            self.raw_table = QTableView()
            self.raw_model = _RawTableModel(self.model)
            self.raw_table.setModel(self.raw_model)
            self._configure_table(self.raw_table)
            self.detail_tabs.addTab(self.raw_table, "Raw values")

            self._refresh_all()

        @staticmethod
        def _configure_table(table: QTableView) -> None:
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setWordWrap(False)
            table.verticalHeader().setVisible(False)
            header = table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
            header.setResizeContentsPrecision(200)

        def _air_preset(self) -> tuple[str, PathLike] | None:
            for label, path in self.presets.items():
                if "air" in label.casefold():
                    return label, path
            return None

        def _choose_files(self) -> None:
            paths, _selected_filter = QFileDialog.getOpenFileNames(
                self,
                "Add FREDDY material CSVs",
                self._last_folder,
                "Material CSVs (*.csv);;All files (*)",
            )
            if paths:
                self._last_folder = str(Path(paths[0]).parent)
                self.add_requests(paths)

        def _add_air_reference(self) -> None:
            preset = self._air_preset()
            if preset is not None:
                label, path = preset
                self.add_requests([(path, label)])

        def _add_from_provider(
            self,
            provider: Callable[[], Iterable[SourceRequest]] | None,
            empty_message: str,
        ) -> None:
            requests = list(provider()) if provider is not None else []
            if not requests:
                QMessageBox.information(self, "Material Explorer", empty_message)
                return
            self.add_requests(requests)

        def add_requests(
            self,
            requests: Iterable[SourceRequest],
        ) -> tuple[int, int, list[tuple[Path, Exception]]]:
            pending = list(requests)
            self.status_label.setText("Loading material files...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                added, duplicates, errors = self.model.add_requests(pending)
            finally:
                QApplication.restoreOverrideCursor()
            duplicate_keys = {_path_identity(path) for path in duplicates}
            selected = {source.key for source in added} | duplicate_keys
            feedback: list[str] = []
            if added:
                feedback.append(f"{len(added)} added")
            if duplicates:
                feedback.append(f"{len(duplicates)} already loaded")
            if errors:
                feedback.append(f"{len(errors)} rejected")
            self._last_load_feedback = " | ".join(feedback)
            self._refresh_all(selected_keys=selected)
            if errors:
                shown = [f"{path.name}: {error}" for path, error in errors[:12]]
                if len(errors) > len(shown):
                    shown.append(f"...and {len(errors) - len(shown)} more")
                accepted_lines = []
                if added:
                    accepted_lines.append(f"Loaded {len(added)} valid material file(s).")
                if duplicates:
                    accepted_lines.append(f"Already loaded: {len(duplicates)} file(s).")
                prefix = "\n".join(accepted_lines)
                if prefix:
                    prefix += "\n\n"
                QMessageBox.warning(
                    self,
                    "Material files not loaded",
                    prefix + "Rejected files:\n" + "\n".join(shown),
                )
            return len(added), len(duplicates), errors

        def _selected_keys(self) -> set[str]:
            keys: set[str] = set()
            for item in self.source_list.selectedItems():
                key = item.data(Qt.UserRole)
                if key:
                    keys.add(str(key))
            return keys

        def _reload_selected(self) -> None:
            selected = self._selected_keys()
            errors: list[tuple[Path, Exception]] = []
            self._last_load_feedback = ""
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                for key in selected:
                    source = self.model.source_for_key(key)
                    if source is None:
                        continue
                    try:
                        self.model.reload_key(key)
                    except Exception as exc:
                        errors.append((source.path, exc))
            finally:
                QApplication.restoreOverrideCursor()
            self._refresh_all(selected_keys=selected)
            if errors:
                QMessageBox.warning(
                    self,
                    "Material reload failed",
                    "The last valid in-memory data was retained:\n"
                    + "\n".join(f"{path.name}: {error}" for path, error in errors),
                )

        def _remove_selected(self) -> None:
            self._last_load_feedback = ""
            self.model.remove_keys(self._selected_keys())
            self._refresh_all()

        def _clear(self) -> None:
            self._last_load_feedback = ""
            self.model.clear()
            self._refresh_all()

        def _source_colors(self) -> list[str]:
            candidates: list[str] = []
            for key in ("plot_line_freq", "plot_line_angle", "plot_worst", "accent"):
                value = self._colors.get(key)
                if isinstance(value, str) and value not in candidates:
                    candidates.append(value)
            dark = str(self._colors.get("plot_bg", "")).casefold() in {
                "#0b1222",
                "#0f172a",
            }
            extras = (
                ("#2dd4bf", "#f472b6", "#fb923c", "#a3e635", "#e879f9", "#67e8f9")
                if dark
                else ("#0f766e", "#be123c", "#c2410c", "#4d7c0f", "#a21caf", "#0e7490")
            )
            for value in extras:
                if value not in candidates:
                    candidates.append(value)
            return candidates or ["#2563eb", "#7c3aed", "#b45309", "#0891b2"]

        def _source_color(self, index: int) -> str:
            colors = self._source_colors()
            return colors[index % len(colors)]

        def _display_labels(self) -> dict[str, str]:
            grouped: dict[str, list[MaterialExplorerSource]] = {}
            for source in self.model.sources:
                grouped.setdefault(source.label.casefold(), []).append(source)
            labels = {
                source.key: source.label for source in self.model.sources
            }
            for group in grouped.values():
                if len(group) < 2:
                    continue
                maximum_depth = max(len(source.path.parent.parts) for source in group)
                unique_suffixes: list[str] | None = None
                for depth in range(1, maximum_depth + 1):
                    suffixes = [
                        "/".join(source.path.parent.parts[-depth:]) for source in group
                    ]
                    if len({suffix.casefold() for suffix in suffixes}) == len(group):
                        unique_suffixes = suffixes
                        break
                if unique_suffixes is None:
                    unique_suffixes = [f"source {index}" for index in range(1, len(group) + 1)]
                for source, suffix in zip(group, unique_suffixes):
                    labels[source.key] = f"{source.label} - {suffix}"
            return labels

        def _refresh_source_list(self, selected_keys: set[str]) -> None:
            labels = self._display_labels()
            self.source_list.blockSignals(True)
            self.source_list.clear()
            for source in self.model.sources:
                state = self.model.source_state(source)
                state_text = "" if state == "current" else f" [{state}]"
                text = (
                    f"{labels[source.key]}{state_text}\n"
                    f"{len(source.table.freq_ghz)} samples | "
                    f"{_format_number(source.table.freq_ghz[0])} to "
                    f"{_format_number(source.table.freq_ghz[-1])} GHz"
                )
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, source.key)
                item.setToolTip(str(source.path))
                swatch = QPixmap(12, 12)
                swatch.fill(QColor(self._source_color(source.color_index)))
                item.setIcon(QIcon(swatch))
                self.source_list.addItem(item)
                item.setSelected(source.key in selected_keys)
            self.source_list.blockSignals(False)

        def _refresh_status(self) -> None:
            count = len(self.model)
            if count == 0:
                text = "No materials loaded."
            elif count == 1:
                source = self.model.sources[0]
                text = (
                    "1 material | coverage "
                    f"{_format_number(source.table.freq_ghz[0])} to "
                    f"{_format_number(source.table.freq_ghz[-1])} GHz"
                )
            else:
                overlap = self.model.common_frequency_range()
                if overlap is None:
                    text = f"{count} materials | no shared frequency range"
                else:
                    text = (
                        f"{count} materials | shared coverage "
                        f"{_format_number(overlap[0])} to "
                        f"{_format_number(overlap[1])} GHz"
                    )
            if self._last_load_feedback:
                text += " | " + self._last_load_feedback
            self.status_label.setText(text)

        def _draw_charts(self) -> None:
            if not MPL_AVAILABLE or self.figure is None or self.canvas is None:
                return
            colors = self._colors
            self.figure.patch.set_facecolor(str(colors.get("plot_bg", "#ffffff")))
            specs = (
                ("Relative permittivity - real", lambda value: value.real, "eps"),
                ("Relative permittivity - imaginary", lambda value: value.imag, "eps"),
                ("Relative permeability - real", lambda value: value.real, "mu"),
                ("Relative permeability - imaginary", lambda value: value.imag, "mu"),
            )
            labels = self._display_labels()
            for axis, (title, accessor, family) in zip(self.axes, specs):
                axis.clear()
                for source in self.model.sources:
                    values = source.table.eps_r if family == "eps" else source.table.mu_r
                    direct_values = [accessor(value) for value in values]
                    indices = extrema_preserving_indices(direct_values)
                    frequencies = [source.table.freq_ghz[index] for index in indices]
                    plotted = [direct_values[index] for index in indices]
                    axis.plot(
                        frequencies,
                        plotted,
                        color=self._source_color(source.color_index),
                        linewidth=1.6,
                        marker="o" if len(frequencies) == 1 else None,
                        markersize=4,
                        label=labels[source.key],
                    )
                compare_frequency = self.frequency_model.frequency_ghz
                if compare_frequency is not None and self.model.sources:
                    union_low = min(
                        source.table.freq_ghz[0] for source in self.model.sources
                    )
                    union_high = max(
                        source.table.freq_ghz[-1] for source in self.model.sources
                    )
                    if union_low <= compare_frequency <= union_high:
                        axis.axvline(
                            compare_frequency,
                            color=str(colors.get("plot_crosshair", "#475569")),
                            linewidth=1.0,
                            linestyle="--",
                            alpha=0.8,
                        )
                axis.set_title(title, fontsize=9)
                axis.set_xlabel("Frequency (GHz)", fontsize=8)
                axis.set_ylabel("Relative value", fontsize=8)
                if colors:
                    style_axis(axis, colors)  # type: ignore[arg-type]
                    axis.grid(
                        True,
                        color=str(colors.get("plot_grid", "#cbd5e1")),
                        alpha=0.3,
                    )
                else:
                    axis.grid(True, alpha=0.25)
                axis.tick_params(labelsize=8)
                if not self.model.sources:
                    axis.text(
                        0.5,
                        0.5,
                        "Add material CSVs to compare",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        color=str(colors.get("muted_text", "#64748b")),
                    )
            if self.model.sources:
                self.axes[0].legend(fontsize=7, loc="best")
            decimated = any(
                len(source.table.freq_ghz) > 5000 for source in self.model.sources
            )
            self.chart_note.setText(
                "Display decimated for responsiveness; every plotted bucket keeps "
                "its minimum and maximum so narrow extrema are retained. Raw values "
                "remain complete."
            )
            self.chart_note.setVisible(decimated)
            self.canvas.draw_idle()

        def _update_compare_status(self) -> None:
            count = len(self.model)
            frequency = self.frequency_model.frequency_ghz
            overlap = self.model.common_frequency_range()
            self.shared_midpoint_button.setEnabled(overlap is not None)
            if count == 0:
                text = "Add materials to compare."
            elif frequency is None:
                text = "Enter a positive comparison frequency in GHz."
            else:
                in_range = sum(
                    sample is not None for _source, sample in self.frequency_model.rows
                )
                text = (
                    f"{in_range} of {count} material(s) cover "
                    f"{_format_number(frequency)} GHz."
                )
                if overlap is None and count > 1:
                    text += " The loaded files have no shared frequency range."
                elif overlap is not None:
                    text += (
                        " Shared range: "
                        f"{_format_number(overlap[0])} to "
                        f"{_format_number(overlap[1])} GHz."
                    )
            self.compare_status_label.setText(text)

        def _refresh_frequency_comparison(self) -> None:
            if not self.model.sources:
                self.compare_frequency_edit.clear()
                self.frequency_model.set_frequency(None)
                self._update_compare_status()
                return
            if self.frequency_model.frequency_ghz is None:
                overlap = self.model.common_frequency_range()
                if overlap is not None:
                    midpoint = (overlap[0] + overlap[1]) / 2.0
                    self.compare_frequency_edit.setText(_format_number(midpoint))
                    self.frequency_model.set_frequency(midpoint)
                else:
                    self.frequency_model.refresh()
            else:
                self.frequency_model.refresh()
            self._update_compare_status()

        def _apply_compare_frequency(self) -> None:
            if not self.model.sources:
                QMessageBox.information(
                    self,
                    "Material Explorer",
                    "Add at least one material file before comparing a frequency.",
                )
                return
            try:
                frequency = float(self.compare_frequency_edit.text().strip())
                if not math.isfinite(frequency) or frequency <= 0.0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Comparison frequency",
                    "Comparison frequency must be a finite value greater than 0 GHz.",
                )
                return
            self.compare_frequency_edit.setText(_format_number(frequency))
            self.frequency_model.set_frequency(frequency)
            self._update_compare_status()
            self._draw_charts()

        def _use_shared_midpoint(self) -> None:
            overlap = self.model.common_frequency_range()
            if overlap is None:
                QMessageBox.information(
                    self,
                    "Shared frequency range",
                    "The loaded materials do not have a shared frequency range.",
                )
                return
            midpoint = (overlap[0] + overlap[1]) / 2.0
            self.compare_frequency_edit.setText(_format_number(midpoint))
            self.frequency_model.set_frequency(midpoint)
            self._update_compare_status()
            self._draw_charts()

        def _sync_action_state(self) -> None:
            selected = bool(self._selected_keys())
            self.reload_button.setEnabled(selected)
            self.remove_button.setEnabled(selected)
            self.clear_button.setEnabled(bool(len(self.model)))

        def _refresh_all(self, selected_keys: set[str] | None = None) -> None:
            selected = self._selected_keys() if selected_keys is None else selected_keys
            self._refresh_source_list(selected)
            self.summary_model.refresh()
            self.raw_model.refresh()
            self._refresh_frequency_comparison()
            self._refresh_status()
            self._sync_action_state()
            self._draw_charts()

        def refresh_external_state(self) -> None:
            """Refresh stale markers and provider button state without file reads."""

            self.add_stack_button.setEnabled(self.stack_source_provider is not None)
            self.add_mix_button.setEnabled(self.mix_source_provider is not None)
            selected = self._selected_keys()
            self._refresh_source_list(selected)
            self._refresh_status()
            self._sync_action_state()

        def apply_theme(self, colors: Mapping[str, object]) -> None:
            self._colors = colors
            selected = self._selected_keys()
            self._refresh_source_list(selected)
            self._draw_charts()

        def dragEnterEvent(self, event) -> None:  # noqa: N802, ANN001
            urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
            if any(url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() == ".csv" for url in urls):
                event.acceptProposedAction()
            else:
                event.ignore()

        def dropEvent(self, event) -> None:  # noqa: N802, ANN001
            paths = [
                url.toLocalFile()
                for url in event.mimeData().urls()
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() == ".csv"
            ]
            if paths:
                self.add_requests(paths)
                event.acceptProposedAction()
            else:
                event.ignore()


else:

    class MaterialExplorerWidget:  # pragma: no cover - friendly headless failure
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PySide6 is required for the Material Explorer GUI.")


__all__ = [
    "MPL_AVAILABLE",
    "QT_AVAILABLE",
    "MaterialExplorerModel",
    "MaterialExplorerSample",
    "MaterialExplorerSource",
    "MaterialExplorerSummary",
    "MaterialExplorerWidget",
    "SourceRequest",
    "extrema_preserving_indices",
    "loss_tangent",
]
