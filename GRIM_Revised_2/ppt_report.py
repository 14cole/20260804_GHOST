"""Deterministic GRIM plot reports for Microsoft PowerPoint.

The planning and PNG-rendering portions of this module are deliberately free
of Qt and PowerPoint.  They can therefore drive both the GRIM slide preview
and ordinary unit tests.  Only :class:`PowerPointComBridge` knows about
Windows COM automation.

Live export starts its own hidden desktop PowerPoint instance; it does not
require PowerPoint to be open before the export begins.  ``pywin32`` remains
an optional, Windows-only runtime dependency.
"""

from __future__ import annotations

import math
import os
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence

try:  # pywin32 is intentionally optional outside live Windows export.
    import pythoncom
    import win32com.client
except ImportError:  # pragma: no cover - normal on non-Windows systems
    pythoncom = None
    win32com = None


POINTS_PER_INCH = 72.0
SLIDE_WIDTH_POINTS = 13.3333333333 * POINTS_PER_INCH
SLIDE_HEIGHT_POINTS = 7.5 * POINTS_PER_INCH
SLIDE_TITLE_FONT_SIZE_POINTS = 21.0
SLIDE_FOOTER_FONT_SIZE_POINTS = 8.5
SLIDE_PAGE_NUMBER_FONT_SIZE_POINTS = 8.5

PP_LAYOUT_BLANK = 12
PP_SAVE_AS_OPEN_XML_PRESENTATION = 24
MSO_FALSE = 0
MSO_TRUE = -1
MSO_TEXT_ORIENTATION_HORIZONTAL = 1
PP_ALIGN_LEFT = 1
PP_ALIGN_CENTER = 2
PP_ALIGN_RIGHT = 3

PlotKind = Literal["azimuth_rect", "azimuth_polar", "frequency"]
LayoutKind = Literal["azimuth_3x2", "frequency_single"]
RenderedImageKey = tuple[int, int]


@dataclass(frozen=True)
class Rect:
    """A rectangle in PowerPoint points on the canonical 16:9 slide."""

    left: float
    top: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.width, self.height)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Slide geometry must contain only finite values.")
        if self.left < 0 or self.top < 0:
            raise ValueError("Slide geometry cannot start outside the slide.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Slide geometry width and height must be positive.")

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def scaled(self, x_scale: float, y_scale: float) -> "Rect":
        return Rect(
            left=self.left * x_scale,
            top=self.top * y_scale,
            width=self.width * x_scale,
            height=self.height * y_scale,
        )


@dataclass(frozen=True)
class SlideGeometry:
    """Fixed title, plot, and footer geometry for one layout family."""

    width: float
    height: float
    title: Rect
    footer: Rect
    page_number: Rect
    plot_frames: tuple[Rect, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Slide dimensions must be positive.")
        for box in (self.title, self.footer, self.page_number, *self.plot_frames):
            if box.right > self.width + 1e-6 or box.bottom > self.height + 1e-6:
                raise ValueError("A slide element extends beyond the slide canvas.")

    def scaled_to(self, width: float, height: float) -> "SlideGeometry":
        """Scale canonical positions to the presentation's actual page size."""

        if width <= 0 or height <= 0:
            raise ValueError("PowerPoint returned invalid slide dimensions.")
        x_scale = width / self.width
        y_scale = height / self.height
        return SlideGeometry(
            width=float(width),
            height=float(height),
            title=self.title.scaled(x_scale, y_scale),
            footer=self.footer.scaled(x_scale, y_scale),
            page_number=self.page_number.scaled(x_scale, y_scale),
            plot_frames=tuple(
                frame.scaled(x_scale, y_scale) for frame in self.plot_frames
            ),
        )


def azimuth_3x2_geometry() -> SlideGeometry:
    """Return the canonical row-major 3-column by 2-row plot layout."""

    slide_w = SLIDE_WIDTH_POINTS
    slide_h = SLIDE_HEIGHT_POINTS
    margin_x = 34.0
    column_gap = 12.0
    row_gap = 10.0
    plot_top = 54.0
    plot_bottom = 503.0
    plot_width = (slide_w - 2.0 * margin_x - 2.0 * column_gap) / 3.0
    plot_height = (plot_bottom - plot_top - row_gap) / 2.0
    frames = tuple(
        Rect(
            margin_x + column * (plot_width + column_gap),
            plot_top + row * (plot_height + row_gap),
            plot_width,
            plot_height,
        )
        for row in range(2)
        for column in range(3)
    )
    return SlideGeometry(
        width=slide_w,
        height=slide_h,
        title=Rect(34.0, 13.0, slide_w - 68.0, 31.0),
        footer=Rect(34.0, 514.0, slide_w - 150.0, 15.0),
        page_number=Rect(slide_w - 105.0, 514.0, 71.0, 15.0),
        plot_frames=frames,
    )


def frequency_single_geometry() -> SlideGeometry:
    """Return the canonical single-frequency-sweep layout."""

    slide_w = SLIDE_WIDTH_POINTS
    slide_h = SLIDE_HEIGHT_POINTS
    return SlideGeometry(
        width=slide_w,
        height=slide_h,
        title=Rect(42.0, 15.0, slide_w - 84.0, 34.0),
        footer=Rect(42.0, 514.0, slide_w - 158.0, 15.0),
        page_number=Rect(slide_w - 105.0, 514.0, 63.0, 15.0),
        plot_frames=(Rect(54.0, 61.0, slide_w - 108.0, 438.0),),
    )


def geometry_for_layout(layout: LayoutKind) -> SlideGeometry:
    if layout == "azimuth_3x2":
        return azimuth_3x2_geometry()
    if layout == "frequency_single":
        return frequency_single_geometry()
    raise ValueError(f"Unsupported slide layout: {layout!r}")


@dataclass(frozen=True)
class PlotSeries:
    """One line to render into an individual plot PNG."""

    x: tuple[float, ...]
    y: tuple[float, ...]
    label: str = ""
    color: str | None = None
    line_style: str = "-"
    line_width: float = 1.5

    @classmethod
    def from_values(
        cls,
        x: Iterable[float],
        y: Iterable[float],
        *,
        label: str = "",
        color: str | None = None,
        line_style: str = "-",
        line_width: float = 1.5,
    ) -> "PlotSeries":
        return cls(
            x=tuple(float(value) for value in x),
            y=tuple(float(value) for value in y),
            label=str(label),
            color=color,
            line_style=str(line_style),
            line_width=float(line_width),
        )

    def __post_init__(self) -> None:
        if not self.x or len(self.x) != len(self.y):
            raise ValueError("Every plot series needs equally sized, non-empty x/y data.")
        if not all(math.isfinite(value) for value in self.x):
            raise ValueError("Plot x values must all be finite.")
        if any(not (math.isfinite(value) or math.isnan(value)) for value in self.y):
            raise ValueError("Plot y values may contain finite values or NaN gaps, not infinity.")
        if not any(math.isfinite(value) for value in self.y):
            raise ValueError("Every plot series needs at least one finite y value.")
        if self.line_width <= 0 or not math.isfinite(self.line_width):
            raise ValueError("Plot line width must be a positive finite number.")


@dataclass(frozen=True)
class PlotSpec:
    """A Qt-free description of one independently rendered plot."""

    plot_id: str
    kind: PlotKind
    title: str
    x_label: str
    y_label: str
    series: tuple[PlotSeries, ...]
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None
    x_tick_step: float | None = None
    y_tick_step: float | None = None
    show_grid: bool = True
    show_legend: bool = True

    def __post_init__(self) -> None:
        if not self.plot_id.strip():
            raise ValueError("Every plot needs a stable, non-empty plot_id.")
        if self.kind not in ("azimuth_rect", "azimuth_polar", "frequency"):
            raise ValueError(f"Unsupported report plot kind: {self.kind!r}")
        if not self.series:
            raise ValueError(f"Plot {self.plot_id!r} has no data series.")
        for name, limits in (("x", self.x_limits), ("y", self.y_limits)):
            if limits is None:
                continue
            low, high = (float(limits[0]), float(limits[1]))
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"{name}-axis limits must be finite and increasing.")
        for name, step in (("x", self.x_tick_step), ("y", self.y_tick_step)):
            if step is not None and (not math.isfinite(step) or step <= 0):
                raise ValueError(f"{name}-axis tick step must be positive and finite.")


@dataclass(frozen=True)
class PlotPlacement:
    """A plot assigned to one fixed frame on a slide."""

    plot: PlotSpec
    frame: Rect
    slot_index: int
    row_index: int
    column_index: int


@dataclass(frozen=True)
class SlidePlan:
    """One presentation slide, suitable for both preview and COM export."""

    title: str
    footer: str
    layout: LayoutKind
    plots: tuple[PlotPlacement, ...]

    def __post_init__(self) -> None:
        geometry = geometry_for_layout(self.layout)
        if not self.plots:
            raise ValueError("A report slide must contain at least one plot.")
        if len(self.plots) > len(geometry.plot_frames):
            raise ValueError(f"Layout {self.layout!r} cannot hold that many plots.")
        seen_slots: set[int] = set()
        for placement in self.plots:
            if placement.slot_index in seen_slots:
                raise ValueError("A slide cannot place two plots in the same slot.")
            if not 0 <= placement.slot_index < len(geometry.plot_frames):
                raise ValueError("Plot slot is outside the selected slide layout.")
            expected = geometry.plot_frames[placement.slot_index]
            if placement.frame != expected:
                raise ValueError("Plot placement does not match the fixed slide geometry.")
            expected_columns = 3 if self.layout == "azimuth_3x2" else 1
            expected_row, expected_column = divmod(
                placement.slot_index, expected_columns
            )
            if (placement.row_index, placement.column_index) != (
                expected_row,
                expected_column,
            ):
                raise ValueError("Plot row/column identity does not match its fixed slot.")
            seen_slots.add(placement.slot_index)


@dataclass(frozen=True)
class PresentationPlan:
    """A complete, deterministic report plan."""

    slides: tuple[SlidePlan, ...]

    def __post_init__(self) -> None:
        if not self.slides:
            raise ValueError("Select at least one plot before exporting PowerPoint.")
        ids = [placement.plot.plot_id for slide in self.slides for placement in slide.plots]
        duplicates = sorted({plot_id for plot_id in ids if ids.count(plot_id) > 1})
        if duplicates:
            raise ValueError(
                "plot_id values must be unique across a report: " + ", ".join(duplicates)
            )

    @property
    def plot_count(self) -> int:
        return sum(len(slide.plots) for slide in self.slides)


def _normalized_slide_titles(
    titles: str | Sequence[str], count: int
) -> tuple[str, ...]:
    if isinstance(titles, str):
        return (titles,) * count
    values = tuple(str(value) for value in titles)
    if len(values) != count:
        raise ValueError(f"Expected {count} slide titles but received {len(values)}.")
    return values


def plan_azimuth_slides(
    plots: Sequence[PlotSpec],
    *,
    slide_titles: str | Sequence[str] = "Azimuth Sweeps",
    footer: str = "",
) -> PresentationPlan:
    """Chunk azimuth plots into deterministic 3x2, row-major slides."""

    plot_values = tuple(plots)
    if not plot_values:
        raise ValueError("Select at least one azimuth plot.")
    if any(plot.kind not in ("azimuth_rect", "azimuth_polar") for plot in plot_values):
        raise ValueError("Azimuth slides can contain only rectangular or polar azimuth plots.")
    geometry = azimuth_3x2_geometry()
    slide_count = math.ceil(len(plot_values) / 6)
    titles = _normalized_slide_titles(slide_titles, slide_count)
    slides: list[SlidePlan] = []
    for slide_index in range(slide_count):
        chunk = plot_values[slide_index * 6 : (slide_index + 1) * 6]
        placements = tuple(
            PlotPlacement(
                plot,
                geometry.plot_frames[slot],
                slot,
                slot // 3,
                slot % 3,
            )
            for slot, plot in enumerate(chunk)
        )
        slides.append(
            SlidePlan(
                title=titles[slide_index],
                footer=str(footer),
                layout="azimuth_3x2",
                plots=placements,
            )
        )
    return PresentationPlan(tuple(slides))


def plan_frequency_slides(
    plots: Sequence[PlotSpec],
    *,
    slide_titles: str | Sequence[str] | None = None,
    footer: str = "",
) -> PresentationPlan:
    """Place each frequency-sweep plot on its own full-size slide."""

    plot_values = tuple(plots)
    if not plot_values:
        raise ValueError("Select at least one frequency-sweep plot.")
    if any(plot.kind != "frequency" for plot in plot_values):
        raise ValueError("Frequency-sweep slides can contain only frequency plots.")
    titles = (
        tuple(plot.title for plot in plot_values)
        if slide_titles is None
        else _normalized_slide_titles(slide_titles, len(plot_values))
    )
    geometry = frequency_single_geometry()
    slides = tuple(
        SlidePlan(
            title=titles[index],
            footer=str(footer),
            layout="frequency_single",
            plots=(PlotPlacement(plot, geometry.plot_frames[0], 0, 0, 0),),
        )
        for index, plot in enumerate(plot_values)
    )
    return PresentationPlan(slides)


def combine_plans(*plans: PresentationPlan) -> PresentationPlan:
    """Combine independently planned sections in the supplied order."""

    return PresentationPlan(tuple(slide for plan in plans for slide in plan.slides))


@dataclass(frozen=True)
class PlotRenderStyle:
    """Stable visual settings shared by slide preview and final export."""

    background: str = "#ffffff"
    axes_background: str = "#ffffff"
    text: str = "#172033"
    grid: str = "#b7c0ca"
    axes_edge: str = "#48566a"
    default_colors: tuple[str, ...] = (
        "#1f5f99",
        "#d97706",
        "#198754",
        "#8b5cf6",
        "#c2415d",
        "#0e7490",
    )
    font_family: str = "Arial"


def _safe_asset_name(plot_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", plot_id.strip()).strip("._")
    return stem[:80] or "plot"


def _inclusive_ticks(low: float, high: float, step: float) -> list[float]:
    """Return stable inclusive ticks while rejecting accidental huge grids."""

    count = int(math.floor((high - low) / step + 1e-9)) + 1
    if count > 1_000:
        raise ValueError("Tick step would create more than 1,000 tick marks.")
    return [low + index * step for index in range(max(count, 1))]


def _signed_degree_label(value: float) -> str:
    normalized = (value + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1e-9) and value > 0:
        normalized = 180.0
    if math.isclose(normalized, 0.0, abs_tol=1e-9):
        normalized = 0.0
    rounded = round(normalized)
    number = (
        str(int(rounded))
        if math.isclose(normalized, rounded, abs_tol=1e-9)
        else f"{normalized:g}"
    )
    return f"{number}°"


def polar_degree_ticks(
    limits: tuple[float, float] | None = None,
    step: float | None = None,
) -> tuple[tuple[float, str], ...]:
    """Return polar tick positions and signed-degree labels.

    Positions remain in the caller's input convention while labels are wrapped
    to ``[-180, 180]``.  A duplicated full-circle endpoint is omitted.
    """

    low, high = limits or (-180.0, 180.0)
    tick_step = step or 45.0
    if not all(math.isfinite(value) for value in (low, high, tick_step)):
        raise ValueError("Polar tick limits and step must be finite.")
    if low >= high or tick_step <= 0:
        raise ValueError("Polar tick limits must increase and step must be positive.")
    ticks = _inclusive_ticks(low, high, tick_step)
    if (
        len(ticks) > 1
        and math.isclose((ticks[-1] - ticks[0]) % 360.0, 0.0, abs_tol=1e-9)
    ):
        ticks.pop()
    return tuple((value, _signed_degree_label(value)) for value in ticks)


def render_plot_png(
    plot: PlotSpec,
    output_path: str | os.PathLike[str],
    *,
    width_points: float,
    height_points: float,
    dpi: int = 160,
    style: PlotRenderStyle = PlotRenderStyle(),
) -> Path:
    """Render one plot to an opaque PNG without importing Qt or pyplot."""

    if width_points <= 0 or height_points <= 0:
        raise ValueError("Rendered plot dimensions must be positive.")
    if dpi < 72:
        raise ValueError("Plot rendering DPI must be at least 72.")

    # Local imports keep planning and fake-COM tests lightweight.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(
        figsize=(width_points / POINTS_PER_INCH, height_points / POINTS_PER_INCH),
        dpi=dpi,
        facecolor=style.background,
    )
    FigureCanvasAgg(figure)
    if plot.kind == "azimuth_polar":
        axes = figure.add_subplot(111, projection="polar")
        figure.subplots_adjust(left=0.13, right=0.87, bottom=0.15, top=0.82)
    else:
        axes = figure.add_subplot(111)
        if width_points / height_points > 1.7:
            figure.subplots_adjust(left=0.10, right=0.975, bottom=0.145, top=0.89)
        else:
            figure.subplots_adjust(left=0.17, right=0.97, bottom=0.20, top=0.84)
    axes.set_facecolor(style.axes_background)
    axes.tick_params(colors=style.text, labelsize=7.5 if width_points < 400 else 10)
    for spine in axes.spines.values():
        spine.set_color(style.axes_edge)

    for index, series in enumerate(plot.series):
        x_values = series.x
        if plot.kind == "azimuth_polar":
            x_values = tuple(math.radians(value) for value in x_values)
        axes.plot(
            x_values,
            series.y,
            label=series.label or None,
            color=series.color or style.default_colors[index % len(style.default_colors)],
            linestyle=series.line_style,
            linewidth=series.line_width,
        )

    if plot.kind == "azimuth_polar":
        axes.set_theta_zero_location("N")
        axes.set_theta_direction(-1)
        polar_low, polar_high = plot.x_limits or (-180.0, 180.0)
        if plot.x_limits is not None:
            axes.set_thetamin(polar_low)
            axes.set_thetamax(polar_high)
        polar_step = plot.x_tick_step or 45.0
        polar_ticks = polar_degree_ticks((polar_low, polar_high), polar_step)
        axes.set_xticks([math.radians(value) for value, _label in polar_ticks])
        axes.set_xticklabels([label for _value, label in polar_ticks])
    elif plot.x_limits is not None:
        axes.set_xlim(*plot.x_limits)
    if plot.y_limits is not None:
        axes.set_ylim(*plot.y_limits)
    if plot.kind != "azimuth_polar" and plot.x_tick_step is not None:
        axes.xaxis.set_major_locator(MultipleLocator(plot.x_tick_step))
    if plot.y_tick_step is not None:
        axes.yaxis.set_major_locator(MultipleLocator(plot.y_tick_step))
    axes.set_title(
        plot.title,
        color=style.text,
        fontsize=9.5 if width_points < 400 else 14,
        fontweight="bold",
        fontfamily=style.font_family,
        pad=5,
    )
    if plot.kind != "azimuth_polar":
        axes.set_xlabel(
            plot.x_label,
            color=style.text,
            fontsize=8 if width_points < 400 else 11,
            fontfamily=style.font_family,
        )
    axes.set_ylabel(
        plot.y_label,
        color=style.text,
        fontsize=8 if width_points < 400 else 11,
        fontfamily=style.font_family,
    )
    axes.grid(plot.show_grid, color=style.grid, linewidth=0.55, alpha=0.7)
    labels = [series.label for series in plot.series if series.label]
    if plot.show_legend and labels:
        axes.legend(
            loc="upper right",
            fontsize=6.5 if width_points < 400 else 9,
            framealpha=0.88,
        )
    figure.savefig(
        destination,
        format="png",
        dpi=dpi,
        facecolor=style.background,
        transparent=False,
    )
    figure.clear()
    return destination


def render_plan_images(
    plan: PresentationPlan,
    output_directory: str | os.PathLike[str],
    *,
    dpi: int = 160,
    style: PlotRenderStyle = PlotRenderStyle(),
    renderer: Callable[..., Path] = render_plot_png,
) -> dict[RenderedImageKey, Path]:
    """Render every placement to its own predictably named PNG file."""

    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[RenderedImageKey, Path] = {}
    for slide_index, slide in enumerate(plan.slides):
        for placement_index, placement in enumerate(slide.plots):
            name = (
                f"slide_{slide_index + 1:03d}_slot_{placement.slot_index + 1}_"
                f"{_safe_asset_name(placement.plot.plot_id)}.png"
            )
            path = output_dir / name
            result = renderer(
                placement.plot,
                path,
                width_points=placement.frame.width,
                height_points=placement.frame.height,
                dpi=dpi,
                style=style,
            )
            result_path = Path(result).resolve()
            if not result_path.is_file():
                raise RuntimeError(f"Plot renderer did not create {result_path}.")
            rendered[(slide_index, placement_index)] = result_path
    return rendered


class PresentationWriter(Protocol):
    """Boundary used by :func:`export_powerpoint_report` for fake testing."""

    def write(
        self,
        plan: PresentationPlan,
        rendered_images: Mapping[RenderedImageKey, Path],
        output_path: Path,
        *,
        template_path: Path | None = None,
    ) -> None: ...


def _office_rgb(red: int, green: int, blue: int) -> int:
    return int(red) + int(green) * 256 + int(blue) * 65536


def _collection_item(collection: Any, index: int) -> Any:
    """Support both late-bound COM collections and simple test fakes."""

    try:
        return collection.Item(index)
    except (AttributeError, TypeError):
        return collection(index)


class PowerPointComBridge:
    """Write a planned report using a private desktop PowerPoint instance."""

    def __init__(self, application_factory: Callable[[], Any] | None = None) -> None:
        self._application_factory = application_factory

    @staticmethod
    def _require_com() -> None:
        if sys.platform != "win32":
            raise RuntimeError("PowerPoint export requires Windows.")
        if pythoncom is None or win32com is None:
            raise RuntimeError(
                "pywin32 is not installed. Run: python -m pip install pywin32"
            )

    def _new_application(self) -> tuple[Any, bool]:
        if self._application_factory is not None:
            return self._application_factory(), False
        self._require_com()
        assert pythoncom is not None and win32com is not None
        pythoncom.CoInitialize()
        try:
            application = win32com.client.DispatchEx("PowerPoint.Application")
        except Exception:
            pythoncom.CoUninitialize()
            raise
        return application, True

    def preflight(self) -> None:
        """Fail before plot rendering when desktop PowerPoint is unavailable."""

        application = None
        com_initialized = False
        try:
            application, com_initialized = self._new_application()
            try:
                application.Visible = MSO_FALSE
            except Exception:
                pass
        except Exception as exc:
            raise RuntimeError(
                f"PowerPoint export is unavailable: {exc}"
            ) from exc
        finally:
            if application is not None:
                try:
                    application.Quit()
                except Exception:
                    pass
            if com_initialized and pythoncom is not None:
                pythoncom.CoUninitialize()

    @staticmethod
    def _open_presentation(application: Any, template_path: Path | None) -> Any:
        if template_path is None:
            presentation = application.Presentations.Add(WithWindow=MSO_FALSE)
            presentation.PageSetup.SlideWidth = SLIDE_WIDTH_POINTS
            presentation.PageSetup.SlideHeight = SLIDE_HEIGHT_POINTS
            return presentation
        return application.Presentations.Open(
            str(template_path),
            ReadOnly=MSO_TRUE,
            Untitled=MSO_TRUE,
            WithWindow=MSO_FALSE,
        )

    @staticmethod
    def _clear_template_slides(presentation: Any) -> None:
        while int(presentation.Slides.Count) > 0:
            _collection_item(presentation.Slides, 1).Delete()

    @staticmethod
    def _add_text(
        slide: Any,
        box: Rect,
        text: str,
        *,
        size: float,
        bold: bool,
        alignment: int,
        color: int,
        font_name: str = "Arial",
    ) -> Any:
        shape = slide.Shapes.AddTextbox(
            MSO_TEXT_ORIENTATION_HORIZONTAL,
            box.left,
            box.top,
            box.width,
            box.height,
        )
        frame = shape.TextFrame
        frame.MarginLeft = 0
        frame.MarginRight = 0
        frame.MarginTop = 0
        frame.MarginBottom = 0
        text_range = frame.TextRange
        text_range.Text = str(text)
        text_range.Font.Name = font_name
        text_range.Font.Size = float(size)
        text_range.Font.Bold = MSO_TRUE if bold else MSO_FALSE
        text_range.Font.Color.RGB = color
        text_range.ParagraphFormat.Alignment = alignment
        return shape

    def _populate_presentation(
        self,
        presentation: Any,
        plan: PresentationPlan,
        rendered_images: Mapping[RenderedImageKey, Path],
    ) -> None:
        page_width = float(presentation.PageSetup.SlideWidth)
        page_height = float(presentation.PageSetup.SlideHeight)
        if page_width <= 0 or page_height <= 0:
            raise RuntimeError("PowerPoint returned an invalid slide size.")
        expected_ratio = SLIDE_WIDTH_POINTS / SLIDE_HEIGHT_POINTS
        actual_ratio = page_width / page_height
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.0, abs_tol=1.0e-3):
            raise RuntimeError(
                "The blank PowerPoint template must use a widescreen 16:9 "
                "slide size so the exported deck matches the GRIM preview."
            )
        total_slides = len(plan.slides)
        for slide_index, slide_plan in enumerate(plan.slides):
            base_geometry = geometry_for_layout(slide_plan.layout)
            geometry = base_geometry.scaled_to(page_width, page_height)
            x_scale = page_width / base_geometry.width
            y_scale = page_height / base_geometry.height
            slide = presentation.Slides.Add(
                int(presentation.Slides.Count) + 1,
                PP_LAYOUT_BLANK,
            )
            self._add_text(
                slide,
                geometry.title,
                slide_plan.title,
                size=SLIDE_TITLE_FONT_SIZE_POINTS * min(x_scale, y_scale),
                bold=True,
                alignment=PP_ALIGN_LEFT,
                color=_office_rgb(23, 32, 51),
            )
            for placement_index, placement in enumerate(slide_plan.plots):
                image_path = Path(rendered_images[(slide_index, placement_index)])
                if not image_path.is_file():
                    raise RuntimeError(f"Rendered plot image is missing: {image_path}")
                frame = placement.frame.scaled(x_scale, y_scale)
                picture = slide.Shapes.AddPicture(
                    str(image_path),
                    MSO_FALSE,
                    MSO_TRUE,
                    frame.left,
                    frame.top,
                    frame.width,
                    frame.height,
                )
                try:
                    picture.AlternativeText = placement.plot.title
                except Exception:
                    pass
            if slide_plan.footer:
                self._add_text(
                    slide,
                    geometry.footer,
                    slide_plan.footer,
                    size=SLIDE_FOOTER_FONT_SIZE_POINTS * min(x_scale, y_scale),
                    bold=False,
                    alignment=PP_ALIGN_LEFT,
                    color=_office_rgb(72, 86, 106),
                )
            self._add_text(
                slide,
                geometry.page_number,
                f"{slide_index + 1} / {total_slides}",
                size=SLIDE_PAGE_NUMBER_FONT_SIZE_POINTS * min(x_scale, y_scale),
                bold=False,
                alignment=PP_ALIGN_RIGHT,
                color=_office_rgb(72, 86, 106),
            )

    def write(
        self,
        plan: PresentationPlan,
        rendered_images: Mapping[RenderedImageKey, Path],
        output_path: Path,
        *,
        template_path: Path | None = None,
    ) -> None:
        """Create one PPTX and close all private COM objects before returning."""

        application = None
        presentation = None
        com_initialized = False
        try:
            application, com_initialized = self._new_application()
            try:
                application.Visible = MSO_FALSE
            except Exception:
                pass
            try:
                application.DisplayAlerts = MSO_FALSE
            except Exception:
                pass
            presentation = self._open_presentation(application, template_path)
            self._clear_template_slides(presentation)
            self._populate_presentation(presentation, plan, rendered_images)
            presentation.SaveAs(
                str(output_path),
                PP_SAVE_AS_OPEN_XML_PRESENTATION,
            )
        except Exception as exc:
            raise RuntimeError(f"PowerPoint report export failed: {exc}") from exc
        finally:
            if presentation is not None:
                try:
                    presentation.Close()
                except Exception:
                    pass
            if application is not None:
                try:
                    application.Quit()
                except Exception:
                    pass
            if com_initialized and pythoncom is not None:
                pythoncom.CoUninitialize()


def _validate_export_paths(
    destination: str | os.PathLike[str],
    template_path: str | os.PathLike[str] | None,
) -> tuple[Path, Path | None]:
    output = Path(destination).expanduser().resolve()
    if output.suffix.lower() != ".pptx":
        raise ValueError("PowerPoint report output must use the .pptx extension.")
    template: Path | None = None
    if template_path is not None and str(template_path).strip():
        template = Path(template_path).expanduser().resolve()
        if template.suffix.lower() not in (".pptx", ".potx"):
            raise ValueError("PowerPoint templates must use .pptx or .potx.")
        if not template.is_file():
            raise FileNotFoundError(f"PowerPoint template does not exist: {template}")
        if template == output:
            raise ValueError("Choose an output file different from the blank template.")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, template


def export_powerpoint_report(
    plan: PresentationPlan,
    destination: str | os.PathLike[str],
    *,
    template_path: str | os.PathLike[str] | None = None,
    writer: PresentationWriter | None = None,
    dpi: int = 160,
    style: PlotRenderStyle = PlotRenderStyle(),
    renderer: Callable[..., Path] = render_plot_png,
    temporary_parent: str | os.PathLike[str] | None = None,
) -> Path:
    """Render individual PNGs and safely publish a PPTX report.

    PowerPoint writes to a unique sibling staging file.  The requested output
    is replaced only after the writer returns and the staging file exists, so
    a failed render or COM operation leaves any previous report untouched.
    All plot images are kept in a temporary directory and removed on success
    or failure.
    """

    output, template = _validate_export_paths(destination, template_path)
    bridge: PresentationWriter = writer or PowerPointComBridge()
    preflight = getattr(bridge, "preflight", None)
    if callable(preflight):
        preflight()
    temp_parent_path = (
        Path(temporary_parent).expanduser().resolve()
        if temporary_parent is not None
        else None
    )
    if temp_parent_path is not None:
        temp_parent_path.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(
        f".{output.stem}.grim-{uuid.uuid4().hex}.tmp.pptx"
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="grim-ppt-",
            dir=str(temp_parent_path) if temp_parent_path is not None else None,
        ) as temporary_directory:
            rendered = render_plan_images(
                plan,
                Path(temporary_directory) / "plots",
                dpi=dpi,
                style=style,
                renderer=renderer,
            )
            bridge.write(
                plan,
                rendered,
                staging,
                template_path=template,
            )
            if not staging.is_file() or staging.stat().st_size <= 0:
                raise RuntimeError("PowerPoint did not create a valid staging presentation.")
            os.replace(staging, output)
    finally:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
    return output


__all__ = [
    "LayoutKind",
    "PlotKind",
    "PlotPlacement",
    "PlotRenderStyle",
    "PlotSeries",
    "PlotSpec",
    "PowerPointComBridge",
    "PresentationPlan",
    "PresentationWriter",
    "Rect",
    "RenderedImageKey",
    "SLIDE_FOOTER_FONT_SIZE_POINTS",
    "SLIDE_PAGE_NUMBER_FONT_SIZE_POINTS",
    "SLIDE_TITLE_FONT_SIZE_POINTS",
    "SlideGeometry",
    "SlidePlan",
    "azimuth_3x2_geometry",
    "combine_plans",
    "export_powerpoint_report",
    "frequency_single_geometry",
    "geometry_for_layout",
    "plan_azimuth_slides",
    "plan_frequency_slides",
    "polar_degree_ticks",
    "render_plan_images",
    "render_plot_png",
]
