from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QLabel,
    QToolButton,
)



@dataclass
class PlotContext:
    btn_export_plot: QToolButton
    btn_dataset_ops: QToolButton
    btn_settings: QToolButton
    settings_frame: QFrame
    spin_plot_xmin: QDoubleSpinBox
    spin_plot_xmax: QDoubleSpinBox
    spin_plot_xstep: QDoubleSpinBox
    spin_plot_ymin: QDoubleSpinBox
    spin_plot_ymax: QDoubleSpinBox
    spin_plot_ystep: QDoubleSpinBox
    spin_plot_zmin: QDoubleSpinBox
    spin_plot_zmax: QDoubleSpinBox
    spin_plot_zstep: QDoubleSpinBox
    combo_plot_scale: QComboBox
    combo_polar_zero: QComboBox
    combo_colormap: QComboBox
    chk_colorbar: QCheckBox
    chk_colorbar_shared: QCheckBox
    chk_plot_grid_visible: QCheckBox
    chk_colormap_invert: QCheckBox
    combo_isar_window: QComboBox
    combo_isar_units: QComboBox
    chk_isar_az_interp: QCheckBox
    spin_isar_az_min: QDoubleSpinBox
    spin_isar_az_max: QDoubleSpinBox
    spin_isar_az_step: QDoubleSpinBox
    chk_isar_freq_band: QCheckBox
    spin_isar_freq_min: QDoubleSpinBox
    spin_isar_freq_max: QDoubleSpinBox
    combo_isar_recon: QComboBox
    spin_isar_l1_strength: QDoubleSpinBox
    spin_isar_l1_iters: QDoubleSpinBox
    chk_isar_flip_x: QCheckBox
    chk_isar_flip_y: QCheckBox
    chk_isar_aperture: QCheckBox
    spin_isar_ap_center: QDoubleSpinBox
    spin_isar_ap_width: QDoubleSpinBox
    btn_isar_ap_prev: QToolButton
    btn_isar_ap_next: QToolButton
    btn_isar_ap_play: QToolButton
    btn_isar_peak_scale: QToolButton
    spin_isar_peak_drop: QDoubleSpinBox
    chk_isar_square: QCheckBox
    btn_isar_apply: QToolButton
    btn_plot_bg: QToolButton
    btn_plot_grid: QToolButton
    btn_plot_text: QToolButton
    chk_plot_legend: QToolButton
    hover_readout: QLabel
    plot_figure: Figure
    plot_canvas: FigureCanvas
    plot_ax: Any
    plot_colorbars: list[Any]
    plot_axes: list[Any] | None
    plot_bg_color: str | None
    plot_grid_color: str | None
    plot_text_color: str | None
    last_plot_mode: str | None
