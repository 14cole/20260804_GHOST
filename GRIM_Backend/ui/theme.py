"""Application stylesheet generation, independent of window construction."""

from __future__ import annotations

import base64
from collections.abc import Mapping


def _branch_arrow_uri(points: str, fill: str) -> str:
    """Return a base64 SVG data-URI for a small polygon arrow (used in QSS branch rules)."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">'
        f'<polygon points="{points}" fill="{fill}"/>'
        f'</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def build_qss(palette: Mapping[str, object]) -> str:
    arrow_right = _branch_arrow_uri(
        "2,1 6,4 2,7", str(palette["text"])
    )
    arrow_down = _branch_arrow_uri(
        "1,2 7,2 4,6", str(palette["text"])
    )
    is_dark = bool(palette.get("is_dark", True))
    success_bg = "#052e16" if is_dark else "#ecfdf5"
    success_border = palette.get("success", "#22c55e" if is_dark else "#15803d")
    warning_bg = "#422006" if is_dark else "#fffbeb"
    warning_border = palette.get("warning", "#f59e0b" if is_dark else "#b45309")
    danger_bg = "#450a0a" if is_dark else "#fef2f2"
    danger_border = palette.get("danger", "#ef4444" if is_dark else "#b91c1c")
    return f"""
    QMainWindow {{ background: {palette['win_bg']}; }}
    QMenuBar {{ background: {palette['panel_bg']}; color: {palette['text']}; }}
    QMenuBar::item {{ background: transparent; padding: 5px 10px; }}
    QMenuBar::item:selected {{ background: {palette['hover']}; }}
    QMenu {{
        background: {palette['panel_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']};
    }}
    QMenu::item {{ padding: 6px 28px 6px 24px; }}
    QMenu::item:selected {{ background: {palette['checked_bg']}; color: white; }}
    QMenu::indicator:checked {{ background: {palette['checked_border']}; }}
    QStatusBar {{ background: {palette['panel_bg']}; color: {palette['muted']}; }}
    QWidget {{ color: {palette['text']}; }}
    QDialog {{ background: {palette['win_bg']}; color: {palette['text']}; }}
    QFrame {{ background: {palette['panel_bg']}; border: 1px solid {palette['border']}; border-radius: 8px; }}
    QFrame#paramSeparator {{
        background: {palette['border']}; min-width: 2px; max-width: 2px; border: none; border-radius: 0px;
    }}
    QGroupBox {{ color: {palette['text']}; border: 1px solid {palette['border']}; border-radius: 8px; margin-top: 10px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
    QLabel {{ color: {palette['text']}; }}
    QTableWidget {{
        background: {palette['panel_bg']}; color: {palette['text']};
        alternate-background-color: {palette['head_bg']};
        border: 1px solid {palette['border']}; gridline-color: {palette['grid']};
    }}
    QPlainTextEdit, QTextEdit {{
        background: {palette['panel_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; border-radius: 6px;
        selection-background-color: {palette['checked_bg']};
        selection-color: white;
    }}
    QHeaderView::section {{ background: {palette['head_bg']}; color: {palette['text']}; border: none; padding: 6px; }}
    QTabWidget::pane {{ border: 1px solid {palette['border']}; background: {palette['panel_bg']}; }}
    QTabBar::tab {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; border-bottom: 0; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
    QTabBar::tab:selected {{ background: {palette['head_bg']}; color: {palette['text']}; border-color: {palette['checked_border']}; }}
    QTabBar::tab:hover {{ background: {palette['hover']}; }}
    QListWidget {{
        background: {palette['panel_bg']}; color: {palette['text']};
        alternate-background-color: {palette['head_bg']};
        border: 1px solid {palette['border']};
    }}
    QTreeWidget {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; }}
    QTreeWidget::item {{ border-bottom: 1px solid {palette['grid']}; padding: 3px 4px; }}
    QTreeWidget::item:selected {{ background: {palette['checked_bg']}; color: white; }}
    QTreeWidget::branch {{ background: {palette['panel_bg']}; }}
    QTreeWidget::branch:has-children:!open {{ image: url("{arrow_right}"); }}
    QTreeWidget::branch:has-children:open  {{ image: url("{arrow_down}"); }}
    QTreeWidget#assemblyTree::branch:has-children {{ image: none; }}
    QListWidget::item {{ border-bottom: 1px solid {palette['grid']}; padding: 4px 6px; }}
    QListWidget QLineEdit {{
        background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        padding: 2px 4px; min-height: 20px; font-size: 12px;
    }}
    QListWidget::item:selected {{
        background: {palette['checked_bg']}; color: white; border-bottom: 1px solid {palette['grid']};
    }}
    QToolButton, QPushButton, QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{
        background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        border-radius: 6px; padding: 6px;
    }}
    QCheckBox, QRadioButton {{
        background: transparent; color: {palette['text']}; border: none;
        padding: 4px 2px;
    }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {palette['border']};
        border-radius: 3px;
        background: {palette['panel_bg']};
    }}
    QCheckBox::indicator:checked {{
        background: {palette['checked_bg']};
        border-color: {palette['checked_border']};
    }}
    QToolButton:hover, QPushButton:hover {{ border-color: {palette['hover']}; }}
    QToolButton:focus, QPushButton:focus, QLineEdit:focus, QComboBox:focus,
    QDoubleSpinBox:focus, QSpinBox:focus {{
        border: 2px solid {palette['checked_border']};
    }}
    QToolButton:disabled, QPushButton:disabled, QLineEdit:disabled,
    QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{
        color: {palette['muted']}; border-color: {palette['grid']};
        background: {palette['head_bg']};
    }}
    QToolButton:checked {{ background: {palette['checked_bg']}; color: white; border-color: {palette['checked_border']}; }}
    QComboBox QAbstractItemView {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; }}
    QTableWidget::item:selected {{ background: {palette['checked_bg']}; color: white; }}
    QProgressBar {{
        background: {palette['head_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; border-radius: 5px;
        text-align: center;
    }}
    QProgressBar::chunk {{ background: {palette['checked_bg']}; }}
    QSlider::groove:horizontal {{
        height: 6px; background: {palette['grid']}; border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        width: 14px; margin: -5px 0; background: {palette['checked_border']};
        border: 1px solid {palette['border']}; border-radius: 7px;
    }}
    QLabel#hoverReadout {{
        background: {palette['head_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        border-radius: 4px; padding: 2px 6px; font-family: "Consolas","Courier New",monospace; font-size: 11px;
    }}
    QScrollArea#controlDock {{ background: {palette['win_bg']}; border: none; }}
    QScrollArea#featureBodyScroll, QScrollArea#featurePointScroll,
    QScrollArea#featureLineScroll,
    QScrollArea#featureReviewScroll {{ background: {palette['panel_bg']}; border: none; }}
    QScrollArea#plotSettingsScroll {{ background: {palette['panel_bg']}; border: none; }}
    QScrollArea#runsControlsScroll, QScrollArea#pptControlsScroll,
    QScrollArea#ghostSolverControlsScroll, QScrollArea#freddyWorkspaceScroll {{
        background: {palette['panel_bg']}; border: none;
    }}
    QScrollArea#runsControlsScroll > QWidget,
    QScrollArea#pptControlsScroll > QWidget,
    QWidget#runsControlsContent,
    QWidget#ghostSolverControlsContent,
    QWidget#pptControlsContent {{ background: {palette['panel_bg']}; }}
    QWidget#featureAssemblyContent {{ background: {palette['panel_bg']}; }}
    QWidget#featurePlacementUnitsBar {{
        background: {palette['head_bg']}; border: 1px solid {palette['border']};
        border-radius: 7px;
    }}
    QLabel#featurePanelIntro {{ font-size: 13px; font-weight: 600; padding: 2px 1px; }}
    QLabel#featureWorkflowSteps {{
        background: {palette['head_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; border-radius: 6px;
        padding: 7px 9px; font-weight: 600;
    }}
    QLabel#featureNextStep {{ color: {palette['text']}; padding: 1px 4px 3px 4px; }}
    QGroupBox#featureStepCard {{
        border-color: {palette['border']}; background: {palette['panel_bg']};
        font-weight: 600;
    }}
    QGroupBox#featureStepCard QLabel, QGroupBox#featureStepCard QLineEdit,
    QGroupBox#featureStepCard QComboBox, QGroupBox#featureStepCard QCheckBox,
    QGroupBox#featureStepCard QPushButton, QGroupBox#featureStepCard QTableWidget,
    QGroupBox#featureStepCard QTreeWidget {{
        font-weight: 400;
    }}
    QLabel#featureHint, QLabel#featureCsvSummary {{ color: {palette['muted']}; padding: 1px 2px; }}
    QLabel#featureSummary, QLabel#featureBuildSummary {{
        background: {palette['head_bg']}; border: 1px solid {palette['border']};
        border-radius: 5px; padding: 5px 7px;
    }}
    QLabel#featureEffectivePhysics {{
        background: {success_bg}; border: 1px solid {success_border};
        border-radius: 5px; padding: 6px 8px;
    }}
    QLabel#featureModelBoundary {{
        background: {warning_bg}; border-left: 3px solid {warning_border};
        border-radius: 5px; padding: 7px 9px;
    }}
    QLabel#featureValidationWarning {{
        background: {warning_bg}; border: 1px solid {warning_border};
        border-radius: 5px; padding: 6px 8px;
    }}
    QLabel#featureSurfaceBindingStatus {{
        background: {palette['head_bg']}; border-left: 3px solid {palette['border']};
        border-radius: 4px; padding: 6px 8px;
    }}
    QLabel#featureSurfaceBindingStatus[bindingState="valid"],
    QLabel#featureSurfaceBindingStatus[bindingState="not_required"] {{
        background: {success_bg}; border-left-color: {success_border};
    }}
    QLabel#featureSurfaceBindingStatus[bindingState="missing"],
    QLabel#featureSurfaceBindingStatus[bindingState="invalid"],
    QLabel#featureSurfaceBindingStatus[bindingState="unavailable"] {{
        background: {danger_bg}; border-left-color: {danger_border};
    }}
    QLabel#featureSurfaceBindingStatus[bindingState="stale"],
    QLabel#featureSurfaceBindingStatus[bindingState="unchecked"] {{
        background: {warning_bg}; border-left-color: {warning_border};
    }}
    QPushButton#featureWorkflowAction[primaryAction="true"] {{
        background: {palette['checked_bg']}; color: white;
        border: 1px solid {palette['checked_border']}; font-weight: 600;
    }}
    QPushButton#featureWorkflowAction[primaryAction="true"]:hover {{
        border: 2px solid {palette['checked_border']};
    }}
    QLabel#featureContract {{
        background: {palette['head_bg']}; border-left: 3px solid {palette['checked_border']};
        border-radius: 4px; padding: 6px 8px;
    }}
    QLabel#featureReadiness {{
        font-family: "Consolas","Courier New",monospace; padding: 4px 1px;
    }}
    QTreeWidget#featureReadinessChecklist {{
        background: {palette['panel_bg']}; border: 1px solid {palette['border']};
        border-radius: 5px;
    }}
    QLabel#featureAssemblyStatus {{
        background: {palette['head_bg']}; border: 1px solid {palette['border']};
        border-radius: 6px; padding: 6px 8px;
    }}
    QWidget#plotSettingsContent {{ background: {palette['panel_bg']}; }}
    QLabel#settingsNoMatches {{ color: {palette['muted']}; padding: 4px 2px; }}
    QWidget#dockBody {{ background: {palette['win_bg']}; }}
    QToolButton#sectionHeader {{
        background: {palette['head_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; border-radius: 6px;
        padding: 7px 10px; text-align: left; font-weight: 600;
    }}
    QToolButton#sectionHeader:hover {{ border-color: {palette['hover']}; }}
    QToolButton#sectionHeader:checked {{ background: {palette['head_bg']}; color: {palette['text']}; border-color: {palette['border']}; }}
    QWidget#sectionBody {{
        background: {palette['panel_bg']}; border: 1px solid {palette['border']};
        border-top: none; border-top-left-radius: 0px; border-top-right-radius: 0px;
        border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;
    }}
    QLabel#opsCategory {{ color: {palette['text']}; font-weight: 600; padding: 6px 2px 1px 2px; }}
    QLabel#paramHeader {{ color: {palette['text']}; font-weight: 600; padding: 2px; }}
    QLabel#plotTitle {{ color: {palette['text']}; font-weight: 700; font-size: 14px; padding: 2px 4px; }}
    QFrame#plotToolbar {{ background: {palette['head_bg']}; border: 1px solid {palette['border']}; border-radius: 8px; }}
    QFrame#datasetOpsPanel {{ background: {palette['panel_bg']}; border: 1px solid {palette['border']}; border-radius: 8px; }}
    """
