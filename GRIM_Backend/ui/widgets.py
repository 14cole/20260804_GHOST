"""Reusable presentation widgets; no dataset or solver operations."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)


def initial_window_size(available: QSize, preferred: QSize = QSize(1550, 900)) -> QSize:
    """Reserve room for native window borders and the title bar.

    QScreen.availableGeometry already excludes desktop taskbars/docks. Qt
    expresses both sizes in logical pixels, so no manual DPI scaling is used.
    """
    return QSize(
        min(preferred.width(), max(1, available.width() - 32)),
        min(preferred.height(), max(1, available.height() - 64)),
    )


class ClickableLabel(QLabel):
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
        else:
            super().mouseDoubleClickEvent(event)


class PlotSettingsPopup(QFrame):
    """Scrollable, searchable top-level popup for one plot context.

    Closing the window via its title bar untoggles the bound button so the
    button state mirrors visibility.  ``content_widget`` deliberately remains
    a plain widget: callers can keep using their existing grid layout while
    the popup owns the search bar and scroll-area plumbing.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str = "Plot Settings",
    ) -> None:
        super().__init__(parent)
        self._toggle_button: QToolButton | None = None
        self._filter_rows: list[tuple[str, tuple[QWidget, ...], QWidget | None]] = []
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(title)
        # Keep the popup useful on a compact display.  Larger contents scroll
        # in either direction instead of forcing the window beyond the screen.
        self.setMinimumSize(420, 280)
        self.resize(820, 540)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        filter_row = QHBoxLayout()
        filter_label = QLabel("Find setting")
        self.filter_edit = QLineEdit(self)
        self.filter_edit.setObjectName("plotSettingsFilter")
        self.filter_edit.setPlaceholderText("Search settings…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip(
            "Filter settings by label, control name, option, or help text."
        )
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self.filter_edit, 1)
        outer.addLayout(filter_row)

        self.no_matches_label = QLabel("No settings match this search.")
        self.no_matches_label.setObjectName("settingsNoMatches")
        self.no_matches_label.setVisible(False)
        outer.addWidget(self.no_matches_label)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("plotSettingsScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_widget = QWidget()
        self.content_widget.setObjectName("plotSettingsContent")
        self.scroll_area.setWidget(self.content_widget)
        outer.addWidget(self.scroll_area, 1)

        self.filter_edit.textChanged.connect(self._apply_filter)

    @staticmethod
    def _searchable_widget_text(widget: QWidget) -> str:
        parts: list[str] = []
        for accessor in ("text", "toolTip", "placeholderText", "windowTitle"):
            method = getattr(widget, accessor, None)
            if callable(method):
                value = method()
                if value:
                    parts.append(str(value))
        if isinstance(widget, QComboBox):
            parts.extend(widget.itemText(index) for index in range(widget.count()))
        return " ".join(parts)

    def register_filter_grid(
        self,
        grid: QGridLayout,
        *,
        search_prefix: str = "",
        section: QWidget | None = None,
        excluded_widgets: tuple[QWidget, ...] = (),
    ) -> None:
        """Register each grid row as one searchable unit.

        A matching row keeps all of its labels and controls together.  When a
        nested ``section`` is supplied, the section itself disappears if none
        of its rows match, avoiding an empty ISAR block during filtering.
        """

        excluded_ids = {id(widget) for widget in excluded_widgets}
        for row in range(grid.rowCount()):
            widgets: list[QWidget] = []
            seen: set[int] = set()
            for column in range(grid.columnCount()):
                item = grid.itemAtPosition(row, column)
                widget = item.widget() if item is not None else None
                if widget is None or id(widget) in excluded_ids or id(widget) in seen:
                    continue
                seen.add(id(widget))
                widgets.append(widget)
            if not widgets:
                continue
            haystack = " ".join(
                [search_prefix]
                + [self._searchable_widget_text(widget) for widget in widgets]
            ).casefold()
            self._filter_rows.append((haystack, tuple(widgets), section))

    def _apply_filter(self, text: str) -> None:
        tokens = tuple(part.casefold() for part in text.split() if part)
        section_matches: dict[QWidget, bool] = {}
        any_match = False
        for haystack, widgets, section in self._filter_rows:
            matches = all(token in haystack for token in tokens)
            any_match = any_match or matches
            for widget in widgets:
                widget.setVisible(matches)
            if section is not None:
                section_matches[section] = section_matches.get(section, False) or matches
        for section, matches in section_matches.items():
            section.setVisible(matches)
        self.no_matches_label.setVisible(bool(tokens) and not any_match)

    def set_toggle_button(self, button: QToolButton) -> None:
        self._toggle_button = button

    def closeEvent(self, event) -> None:
        if self._toggle_button is not None and self._toggle_button.isChecked():
            self._toggle_button.setChecked(False)
        super().closeEvent(event)


class CollapsibleSection(QWidget):
    """A titled panel whose body collapses when its header is clicked.

    Purely presentational — it organises the control dock into Datasets /
    Parameters / Operations / Plot Tools groups while holding the exact same
    widgets the app has always used.
    """

    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = True) -> None:
        super().__init__(parent)
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QToolButton()
        self.header.setObjectName("sectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._body = QWidget()
        self._body.setObjectName("sectionBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 8, 8, 8)
        self._body_layout.setSpacing(6)

        outer.addWidget(self.header)
        outer.addWidget(self._body)

        self.header.toggled.connect(self._sync)
        self._sync(expanded)

    def _sync(self, on: bool) -> None:
        self.header.setText(("▾  " if on else "▸  ") + self._title)
        self._body.setVisible(on)

    def addWidget(self, widget, stretch: int = 0) -> None:
        self._body_layout.addWidget(widget, stretch)

    def addLayout(self, layout, stretch: int = 0) -> None:
        self._body_layout.addLayout(layout, stretch)
