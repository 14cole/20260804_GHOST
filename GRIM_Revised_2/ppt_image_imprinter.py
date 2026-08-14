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

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
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

    One captured profile broadcasts to all selected target pictures.  Multiple
    profiles require the same number of targets; silently cycling or truncating
    pictures would make slide formatting difficult to audit.
    """

    if not profiles:
        raise ValueError("Capture at least one source picture first.")
    if not targets:
        raise ValueError("Select at least one destination picture in PowerPoint.")
    if len(profiles) == 1:
        return [(profiles[0], target) for target in targets]
    if len(profiles) != len(targets):
        raise ValueError(
            f"Captured {len(profiles)} pictures but selected {len(targets)} destinations. "
            "Select the same number, or capture one picture to apply it to all destinations."
        )
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


def _points_and_inches(value: float) -> str:
    return f"{value:.1f} pt ({value / 72.0:.2f} in)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.bridge = PowerPointBridge()
        self.captured_profiles: list[ShapeProfile] = []

        self.setWindowTitle("PowerPoint Image Imprinter")
        self.resize(560, 560)
        self.setMinimumSize(480, 460)
        self._build_ui()
        self._apply_styles()
        self._set_status("Ready — select source pictures in PowerPoint.")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel("PowerPoint Image Imprinter")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 17, QFont.Weight.DemiBold))
        root.addWidget(title)

        instructions = QLabel(
            "Capture picture formatting on one slide, select destination pictures "
            "on another slide, then apply it. One captured picture formats every "
            "destination; multiple pictures are paired in selection order."
        )
        instructions.setWordWrap(True)
        root.addWidget(instructions)

        options_box = QGroupBox("Apply options")
        options_layout = QHBoxLayout(options_box)
        self.chk_location = QCheckBox("Location")
        self.chk_size = QCheckBox("Size")
        self.chk_crop = QCheckBox("Crop")
        for checkbox in (self.chk_location, self.chk_size, self.chk_crop):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._update_apply_enabled)
            options_layout.addWidget(checkbox)
        options_layout.addStretch(1)
        root.addWidget(options_box)

        button_row = QHBoxLayout()
        self.btn_capture = QPushButton("1  Capture selected")
        self.btn_apply = QPushButton("2  Apply to selected")
        self.btn_clear = QPushButton("Clear")
        self.btn_apply.setEnabled(False)
        self.btn_capture.clicked.connect(self._capture)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_clear.clicked.connect(self._clear)
        button_row.addWidget(self.btn_capture)
        button_row.addWidget(self.btn_apply)
        button_row.addWidget(self.btn_clear)
        root.addLayout(button_row)

        captured_label = QLabel("Captured formatting")
        captured_label.setObjectName("sectionLabel")
        root.addWidget(captured_label)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("No source pictures captured.")
        root.addWidget(self.summary, 1)

        self.keep_on_top = QCheckBox("Keep helper above PowerPoint")
        self.keep_on_top.toggled.connect(self._toggle_on_top)
        root.addWidget(self.keep_on_top)

        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f3f5f8; color: #172033; }
            QLabel { color: #172033; }
            QLabel#title { color: #123b67; }
            QLabel#sectionLabel { font-weight: 600; margin-top: 3px; }
            QLabel#status { background: #e5edf7; border-radius: 5px; padding: 8px; }
            QGroupBox { border: 1px solid #bdc9d8; border-radius: 6px;
                        margin-top: 8px; padding-top: 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; }
            QPushButton { background: #1f5f99; color: white; border: none;
                          border-radius: 5px; padding: 9px 12px; font-weight: 600; }
            QPushButton:hover { background: #174d7d; }
            QPushButton:disabled { background: #a9b5c1; }
            QTextEdit { background: white; border: 1px solid #bdc9d8;
                        border-radius: 5px; padding: 6px; }
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
        self.status.setProperty("error", error)
        self.status.setStyleSheet(
            "background: #fde8e8; color: #8a1c1c; border-radius: 5px; padding: 8px;"
            if error
            else ""
        )

    def _capture(self) -> None:
        try:
            profiles, skipped = self.bridge.capture_selected()
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.captured_profiles = profiles
        lines = []
        for index, profile in enumerate(profiles, start=1):
            crop_text = (
                f"L {profile.crop.left:.1f}, T {profile.crop.top:.1f}, "
                f"R {profile.crop.right:.1f}, B {profile.crop.bottom:.1f} pt"
                if profile.crop is not None
                else "unavailable"
            )
            lines.append(
                f"{index}. Slide {profile.slide_index}: {profile.name}\n"
                f"   Position: {_points_and_inches(profile.left)}, "
                f"{_points_and_inches(profile.top)}\n"
                f"   Size: {_points_and_inches(profile.width)} × "
                f"{_points_and_inches(profile.height)}\n"
                f"   Crop: {crop_text}"
            )
        self.summary.setPlainText("\n\n".join(lines))
        suffix = f" Skipped {skipped} non-picture shape(s)." if skipped else ""
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
        enabled = ", ".join(
            name
            for name, value in (
                ("location", options.location),
                ("size", options.size),
                ("crop", options.crop),
            )
            if value
        )
        suffix = f" Skipped {skipped} non-picture shape(s)." if skipped else ""
        self._set_status(f"Applied {enabled} to {applied} picture(s).{suffix}")

    def _clear(self) -> None:
        self.captured_profiles = []
        self.summary.clear()
        self._set_status("Capture cleared.")
        self._update_apply_enabled()

    def _toggle_on_top(self, checked: bool) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, checked)
        self.show()

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
