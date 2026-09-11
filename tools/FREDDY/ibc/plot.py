from __future__ import annotations

from typing import Mapping


def nearest_index(values: list[float], target: float) -> int:
    if not values:
        raise ValueError("Values cannot be empty.")
    return min(range(len(values)), key=lambda idx: abs(values[idx] - target))


def style_axis(axis: object, colors: Mapping[str, str]) -> None:
    axis.set_facecolor(colors["plot_axes_bg"])
    axis.xaxis.label.set_color(colors["plot_text"])
    axis.yaxis.label.set_color(colors["plot_text"])
    axis.title.set_color(colors["plot_text"])
    axis.tick_params(axis="x", colors=colors["plot_text"])
    axis.tick_params(axis="y", colors=colors["plot_text"])
    for spine in axis.spines.values():
        spine.set_color(colors["plot_spine"])


def style_colorbar(colorbar: object, colors: Mapping[str, str]) -> None:
    colorbar.ax.tick_params(colors=colors["plot_text"])
    colorbar.ax.yaxis.label.set_color(colors["plot_text"])
    colorbar.ax.set_facecolor(colors["plot_axes_bg"])
    colorbar.outline.set_edgecolor(colors["plot_spine"])
