"""Copy picture geometry and crop settings between PowerPoint selections.

This module is importable on every platform so its mapping and formatting
logic can be tested with ordinary Python objects.  Live automation requires
Windows, desktop PowerPoint, and pywin32.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Sequence

try:  # pywin32 is intentionally an optional, Windows-only dependency.
    import pythoncom
    import win32com.client
except ImportError:  # pragma: no cover - normal on non-Windows systems
    pythoncom = None
    win32com = None

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from grim_palette import (
    APPLICATION_PALETTES,
    APPLICATION_PALETTE_SETTINGS_KEY,
    DEFAULT_APPLICATION_PALETTE,
    normalize_application_palette_name,
)

PP_SELECTION_SHAPES = 2
MSO_FALSE = 0
MSO_PICTURE = 13
MSO_LINKED_PICTURE = 11
MSO_PLACEHOLDER = 14
MSO_GROUP = 6


@dataclass(frozen=True)
class CropProfile:
    """PowerPoint PictureFormat crop margins, in points."""

    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class ShapeProfile:
    """The picture formatting copied from one selected source shape."""

    slide_index: int
    name: str
    left: float
    top: float
    width: float
    height: float
    crop: CropProfile | None


@dataclass(frozen=True)
class ApplyOptions:
    location: bool = True
    size: bool = True
    crop: bool = True

    def any_enabled(self) -> bool:
        return self.location or self.size or self.crop


def map_profiles_to_targets(
    profiles: Sequence[ShapeProfile], targets: Sequence[Any]
) -> list[tuple[ShapeProfile, Any]]:
    """Return deterministic source/target pairs.

    Equal counts pair in selection order.  When counts differ, ChartHelper's
    first captured profile broadcasts to every selected target picture.
    """

    if not profiles:
        raise ValueError("Capture at least one source picture first.")
    if not targets:
        raise ValueError("Select at least one destination picture in PowerPoint.")
    if len(profiles) != len(targets):
        return [(profiles[0], target) for target in targets]
    return list(zip(profiles, targets))


def _shape_slide_index(shape: Any) -> int:
    """Find a slide ancestor for either a top-level or grouped shape."""

    parent = getattr(shape, "Parent", None)
    for _ in range(8):
        if parent is None:
            break
        try:
            return int(parent.SlideIndex)
        except Exception:
            parent = getattr(parent, "Parent", None)
    return -1


def capture_profile_from_shape(shape: Any) -> ShapeProfile:
    """Read geometry and crop information from a picture-like COM shape."""

    crop: CropProfile | None = None
    try:
        picture_format = shape.PictureFormat
        crop = CropProfile(
            left=float(picture_format.CropLeft),
            top=float(picture_format.CropTop),
            right=float(picture_format.CropRight),
            bottom=float(picture_format.CropBottom),
        )
    except Exception:
        # Some picture placeholders expose geometry but not PictureFormat until
        # PowerPoint finishes materializing their image.  Location/size capture
        # remains useful, and the GUI clearly reports unavailable crop data.
        crop = None

    return ShapeProfile(
        slide_index=_shape_slide_index(shape),
        name=str(shape.Name),
        left=float(shape.Left),
        top=float(shape.Top),
        width=float(shape.Width),
        height=float(shape.Height),
        crop=crop,
    )


def apply_profile_to_shape(
    profile: ShapeProfile, shape: Any, options: ApplyOptions
) -> None:
    """Apply enabled parts of a profile to one destination shape."""

    if not options.any_enabled():
        raise ValueError("Enable Location, Size, and/or Crop before applying.")

    # Crop first: PowerPoint can change picture bounds while crop properties
    # are assigned.  Size and location are written afterward so the requested
    # final frame geometry wins.
    if options.crop:
        if profile.crop is None:
            raise ValueError(f"Crop information was unavailable for {profile.name!r}.")
        picture_format = shape.PictureFormat
        picture_format.CropLeft = profile.crop.left
        picture_format.CropTop = profile.crop.top
        picture_format.CropRight = profile.crop.right
        picture_format.CropBottom = profile.crop.bottom

    if options.size:
        original_lock = None
        try:
            original_lock = int(shape.LockAspectRatio)
            shape.LockAspectRatio = MSO_FALSE
        except Exception:
            original_lock = None
        try:
            shape.Width = profile.width
            shape.Height = profile.height
        finally:
            if original_lock is not None:
                try:
                    shape.LockAspectRatio = original_lock
                except Exception:
                    pass

    if options.location:
        shape.Left = profile.left
        shape.Top = profile.top


class PowerPointBridge:
    """Small adapter around the PowerPoint COM application."""

    def __init__(self) -> None:
        self._com_initialized = False

    @staticmethod
    def _require_com() -> None:
        if sys.platform != "win32":
            raise RuntimeError("Live PowerPoint automation requires Windows.")
        if pythoncom is None or win32com is None:
            raise RuntimeError(
                "pywin32 is not installed. Run: python -m pip install pywin32"
            )

    def get_ppt_app(self):
        """Connect to the running desktop PowerPoint instance."""

        self._require_com()
        if not self._com_initialized:
            pythoncom.CoInitialize()
            self._com_initialized = True
        try:
            return win32com.client.GetActiveObject("PowerPoint.Application")
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to PowerPoint. Open a presentation in desktop PowerPoint first."
            ) from exc

    def get_selected_shapes(self):
        app = self.get_ppt_app()
        if int(app.Presentations.Count) == 0:
            raise RuntimeError("No PowerPoint presentations are open.")
        window = app.ActiveWindow
        if window is None:
            raise RuntimeError("PowerPoint has no active presentation window.")
        selection = window.Selection
        if selection is None or int(selection.Type) != PP_SELECTION_SHAPES:
            raise RuntimeError("Select one or more pictures on the active slide first.")
        try:
            return selection.ShapeRange
        except Exception as exc:
            raise RuntimeError("PowerPoint did not return a selected shape range.") from exc

    @staticmethod
    def shape_is_picture_like(shape: Any) -> bool:
        try:
            shape_type = int(shape.Type)
        except Exception:
            return False
        if shape_type in (MSO_PICTURE, MSO_LINKED_PICTURE):
            return True
        if shape_type != MSO_PLACEHOLDER:
            return False
        try:
            contained_type = int(shape.PlaceholderFormat.ContainedType)
            return contained_type in (MSO_PICTURE, MSO_LINKED_PICTURE)
        except Exception:
            try:
                shape.PictureFormat
                return True
            except Exception:
                return False

    @classmethod
    def _collect_picture_shapes(cls, shape: Any) -> tuple[list[Any], int]:
        try:
            shape_type = int(shape.Type)
        except Exception:
            return [], 1

        if shape_type == MSO_GROUP:
            pictures: list[Any] = []
            skipped = 0
            try:
                group_items = shape.GroupItems
                for index in range(1, int(group_items.Count) + 1):
                    child_pictures, child_skipped = cls._collect_picture_shapes(
                        group_items.Item(index)
                    )
                    pictures.extend(child_pictures)
                    skipped += child_skipped
            except Exception:
                return [], 1
            return pictures, skipped

        if cls.shape_is_picture_like(shape):
            return [shape], 0
        return [], 1

    def selected_picture_shapes(self) -> tuple[list[Any], int]:
        shape_range = self.get_selected_shapes()
        pictures: list[Any] = []
        skipped = 0
        for index in range(1, int(shape_range.Count) + 1):
            found, ignored = self._collect_picture_shapes(shape_range.Item(index))
            pictures.extend(found)
            skipped += ignored
        if not pictures:
            raise RuntimeError("The current PowerPoint selection contains no pictures.")
        return pictures, skipped

    def capture_selected(self) -> tuple[list[ShapeProfile], int]:
        pictures, skipped = self.selected_picture_shapes()
        return [capture_profile_from_shape(shape) for shape in pictures], skipped

    def apply_to_selected(
        self, profiles: Sequence[ShapeProfile], options: ApplyOptions
    ) -> tuple[int, int]:
        targets, skipped = self.selected_picture_shapes()
        pairs = map_profiles_to_targets(profiles, targets)

        # Snapshot every destination.  If one COM write fails, restore all
        # destinations already touched rather than leave a partially formatted
        # slide with no clear indication of which pictures changed.
        snapshots = [capture_profile_from_shape(target) for _, target in pairs]
        try:
            for profile, target in pairs:
                apply_profile_to_shape(profile, target, options)
        except Exception as exc:
            rollback_options = ApplyOptions(location=True, size=True, crop=True)
            for snapshot, (_, target) in zip(snapshots, pairs):
                try:
                    if snapshot.crop is None:
                        rollback_options = ApplyOptions(location=True, size=True, crop=False)
                    else:
                        rollback_options = ApplyOptions(location=True, size=True, crop=True)
                    apply_profile_to_shape(snapshot, target, rollback_options)
                except Exception:
                    pass
            raise RuntimeError(f"PowerPoint rejected a formatting change: {exc}") from exc
        return len(pairs), skipped

    def close(self) -> None:
        if self._com_initialized and pythoncom is not None:
            try:
                pythoncom.CoUninitialize()
            finally:
                self._com_initialized = False


class MainWindow(QMainWindow):
    """ChartHelper's compact capture/imprint workflow with GRIM colors."""

    def __init__(self, *, settings: QSettings | None = None) -> None:
        super().__init__()
        self.bridge = PowerPointBridge()
        self.captured_profiles: list[ShapeProfile] = []
        settings = settings if settings is not None else QSettings("GRIM", "GRIM")
        self.application_palette_name = normalize_application_palette_name(
            settings.value(APPLICATION_PALETTE_SETTINGS_KEY, DEFAULT_APPLICATION_PALETTE)
        )
        self.application_palette = dict(APPLICATION_PALETTES[self.application_palette_name])

        self.setWindowTitle("PowerPoint Image Imprinter")
        self.resize(350, 300)
        self.setMinimumSize(350, 300)
        self._build_ui()
        self._apply_styles()
        self._set_status("Ready")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        options_box = QGroupBox("Imprint Options")
        options_layout = QVBoxLayout(options_box)
        options_layout.setSpacing(10)
        self.chk_location = QCheckBox("Imprint Location")
        self.chk_size = QCheckBox("Imprint Size")
        self.chk_crop = QCheckBox("Imprint Crop")
        for checkbox in (self.chk_location, self.chk_size, self.chk_crop):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._update_apply_enabled)
            options_layout.addWidget(checkbox)
        root.addWidget(options_box)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.btn_capture = QPushButton("Capture")
        self.btn_apply = QPushButton("Imprint")
        self.btn_clear = QPushButton("Clear")
        self.btn_apply.setEnabled(False)
        self.btn_capture.setToolTip("Capture the selected pictures in PowerPoint.")
        self.btn_apply.setToolTip(
            "Imprint the selected destination pictures. Equal counts pair in selection "
            "order; otherwise the first captured profile is used for every destination."
        )
        self.btn_capture.clicked.connect(self._capture)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_clear.clicked.connect(self._clear)
        for button in (self.btn_capture, self.btn_apply, self.btn_clear):
            button.setMinimumHeight(40)
            button_row.addWidget(button)
        root.addLayout(button_row)

        self.status_card = QFrame()
        self.status_card.setObjectName("StatusCard")
        status_layout = QHBoxLayout(self.status_card)
        status_layout.setContentsMargins(12, 10, 12, 10)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        status_layout.addWidget(self.status)
        root.addWidget(self.status_card)

    def _apply_styles(self) -> None:
        # Stylesheet section: use the same saved palette as the GRIM window.
        # Set only this helper's font/style so an embedded launch cannot change
        # the appearance of unrelated application windows.
        font = QFont(self.font())
        font.setPointSize(10)
        self.setFont(font)
        palette = self.application_palette
        self.setStyleSheet(
            f"""
            QMainWindow, QMessageBox {{ background: {palette['win_bg']}; }}
            QLabel, QCheckBox {{ color: {palette['text']}; background: transparent; }}
            QGroupBox {{
                background: {palette['panel_bg']}; color: {palette['text']};
                border: 1px solid {palette['border']}; border-radius: 8px;
                margin-top: 10px; padding: 12px; font-weight: 600;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
            QCheckBox {{ spacing: 8px; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px; border-radius: 3px;
                border: 1px solid {palette['border']}; background: {palette['win_bg']};
            }}
            QCheckBox::indicator:checked {{
                background: {palette['checked_bg']}; border-color: {palette['checked_border']};
            }}
            QCheckBox::indicator:hover, QCheckBox:focus {{
                border: 1px solid {palette['checked_border']};
            }}
            QPushButton {{
                background: {palette['panel_bg']}; color: {palette['text']};
                border: 1px solid {palette['border']}; border-radius: 6px;
                padding: 0 8px; font-weight: 600;
            }}
            QPushButton:hover, QPushButton:focus {{ border-color: {palette['checked_border']}; }}
            QPushButton:pressed {{ background: {palette['checked_bg']}; color: white; }}
            QPushButton:disabled {{
                background: {palette['head_bg']}; color: {palette['muted']};
                border-color: {palette['grid']};
            }}
            QFrame#StatusCard {{
                background: {palette['head_bg']}; border: 1px solid {palette['border']};
                border-radius: 8px;
            }}
            QFrame#StatusCard[error="true"] {{ border: 2px solid {palette['checked_border']}; }}
            QToolTip {{
                background: {palette['panel_bg']}; color: {palette['text']};
                border: 1px solid {palette['border']}; padding: 4px;
            }}
            """
        )

    def _selected_options(self) -> ApplyOptions:
        return ApplyOptions(
            location=self.chk_location.isChecked(),
            size=self.chk_size.isChecked(),
            crop=self.chk_crop.isChecked(),
        )

    def _update_apply_enabled(self, _checked: bool | None = None) -> None:
        self.btn_apply.setEnabled(
            bool(self.captured_profiles) and self._selected_options().any_enabled()
        )

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self.status.setText(message)
        self.status_card.setProperty("error", error)
        self.status_card.style().unpolish(self.status_card)
        self.status_card.style().polish(self.status_card)
        self.status_card.update()

    def _update_capture_button_text(self) -> None:
        count = len(self.captured_profiles)
        self.btn_capture.setText(f"Capture ({count})" if count else "Capture")

    def _capture(self) -> None:
        try:
            profiles, skipped = self.bridge.capture_selected()
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.captured_profiles = profiles
        self._update_capture_button_text()
        suffix = f" Skipped {skipped} non-picture shape(s)." if skipped else ""
        unavailable = sum(profile.crop is None for profile in profiles)
        if unavailable:
            suffix += f" Crop unavailable for {unavailable}; uncheck Imprint Crop to copy location/size."
        self._set_status(f"Captured {len(profiles)} picture profile(s).{suffix}")
        self._update_apply_enabled()

    def _apply(self) -> None:
        options = self._selected_options()
        try:
            applied, skipped = self.bridge.apply_to_selected(
                self.captured_profiles, options
            )
        except Exception as exc:
            self._show_error(str(exc))
            return
        mode = (
            "Paired in selection order."
            if applied == len(self.captured_profiles)
            else "Counts differ; used the first captured profile for every destination."
        )
        suffix = f" Skipped {skipped} non-picture shape(s)." if skipped else ""
        self._set_status(f"Imprinted {applied} picture(s). {mode}{suffix}")

    def _clear(self) -> None:
        self.captured_profiles = []
        self._update_capture_button_text()
        self._set_status("Ready")
        self._update_apply_enabled()

    def _show_error(self, message: str) -> None:
        self._set_status(message, error=True)
        QMessageBox.warning(self, "PowerPoint Image Imprinter", message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        self.bridge.close()
        super().closeEvent(event)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("PowerPoint Image Imprinter")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
