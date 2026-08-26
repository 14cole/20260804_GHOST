from __future__ import annotations

import posixpath
import tempfile
import unittest
import math
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import xml.etree.ElementTree as ET

import ppt_report

from ppt_report import (
    DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
    DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
    MASTER_LEGEND_IMAGE_INDEX,
    MSO_BRING_TO_FRONT,
    MSO_FALSE,
    MSO_TRUE,
    POINTS_PER_INCH,
    PP_LAYOUT_BLANK,
    PlotSeries,
    PlotSpec,
    PowerPointComBridge,
    azimuth_3x2_geometry,
    combine_plans,
    export_powerpoint_report,
    frequency_single_geometry,
    plan_azimuth_slides,
    plan_frequency_slides,
    polar_degree_ticks,
    render_master_legend_png,
    render_plan_images,
    render_plot_png,
    _inclusive_ticks,
)


def make_plot(plot_id: str, kind: str = "azimuth_rect") -> PlotSpec:
    return PlotSpec(
        plot_id=plot_id,
        kind=kind,  # type: ignore[arg-type]
        title=f"Plot {plot_id}",
        x_label="Azimuth (deg)" if kind.startswith("azimuth") else "Frequency (GHz)",
        y_label="RCS (dBsm)",
        series=(
            PlotSeries.from_values(
                (0.0, 1.0, 2.0),
                (-10.0, -8.0, -9.0),
                label="HH",
            ),
        ),
    )


class BundledTemplateContractTests(unittest.TestCase):
    def test_bundled_template_keeps_named_layout_and_seed_contract(self):
        template = (
            Path(__file__).resolve().parent
            / "templates"
            / "GRIM_Report_Template.pptx"
        )
        self.assertTrue(template.is_file())
        presentation_ns = {
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main"
        }
        relationship_tag = (
            "{http://schemas.openxmlformats.org/package/2006/relationships}"
            "Relationship"
        )

        def resolve_target(source_part: str, target: str) -> str:
            if target.startswith("/"):
                return target.lstrip("/")
            return posixpath.normpath(
                posixpath.join(posixpath.dirname(source_part), target)
            )

        with zipfile.ZipFile(template) as archive:
            names = set(archive.namelist())
            presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
            slide_size = presentation.find("p:sldSz", presentation_ns)
            self.assertIsNotNone(slide_size)
            assert slide_size is not None
            self.assertEqual(
                (int(slide_size.attrib["cx"]), int(slide_size.attrib["cy"])),
                (12_192_000, 6_858_000),
            )

            master_parts = sorted(
                name
                for name in names
                if name.startswith("ppt/slideMasters/slideMaster")
                and name.endswith(".xml")
            )
            layout_parts = sorted(
                name
                for name in names
                if name.startswith("ppt/slideLayouts/slideLayout")
                and name.endswith(".xml")
            )
            slide_parts = sorted(
                name
                for name in names
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
            self.assertEqual(len(master_parts), 1)
            self.assertEqual(len(layout_parts), 2)
            self.assertEqual(len(slide_parts), 2)

            master_xml = ET.fromstring(archive.read(master_parts[0]))
            master_c_sld = master_xml.find("p:cSld", presentation_ns)
            self.assertIsNotNone(master_c_sld)
            assert master_c_sld is not None
            self.assertEqual(master_c_sld.attrib.get("name"), "GRIM Report Master")

            layout_names: dict[str, str] = {}
            for layout_part in layout_parts:
                layout_xml = ET.fromstring(archive.read(layout_part))
                layout_c_sld = layout_xml.find("p:cSld", presentation_ns)
                self.assertIsNotNone(layout_c_sld)
                assert layout_c_sld is not None
                layout_names[layout_part] = layout_c_sld.attrib.get("name", "")

                base_name = posixpath.basename(layout_part)
                rels_part = f"ppt/slideLayouts/_rels/{base_name}.rels"
                relationships = ET.fromstring(archive.read(rels_part))
                master_targets = [
                    resolve_target(layout_part, relationship.attrib["Target"])
                    for relationship in relationships.findall(relationship_tag)
                    if relationship.attrib.get("Type", "").endswith("/slideMaster")
                ]
                self.assertEqual(master_targets, master_parts)

            self.assertEqual(
                set(layout_names.values()),
                {
                    DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                    DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
                },
            )

            seeded_layouts: list[str] = []
            for slide_part in slide_parts:
                base_name = posixpath.basename(slide_part)
                rels_part = f"ppt/slides/_rels/{base_name}.rels"
                relationships = ET.fromstring(archive.read(rels_part))
                layout_targets = [
                    resolve_target(slide_part, relationship.attrib["Target"])
                    for relationship in relationships.findall(relationship_tag)
                    if relationship.attrib.get("Type", "").endswith("/slideLayout")
                ]
                self.assertEqual(len(layout_targets), 1)
                seeded_layouts.append(layout_names[layout_targets[0]])
            self.assertEqual(
                seeded_layouts,
                [
                    DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                    DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
                ],
            )

            core = ET.fromstring(archive.read("docProps/core.xml"))
            core_namespace = (
                "http://schemas.openxmlformats.org/package/2006/metadata/"
                "core-properties"
            )
            dc_namespace = "http://purl.org/dc/elements/1.1/"
            self.assertEqual(
                core.findtext(f"{{{dc_namespace}}}creator"), "GRIM"
            )
            self.assertEqual(
                core.findtext(f"{{{core_namespace}}}lastModifiedBy"), "GRIM"
            )
            application = ET.fromstring(archive.read("docProps/app.xml"))
            app_namespace = (
                "http://schemas.openxmlformats.org/officeDocument/2006/"
                "extended-properties"
            )
            self.assertEqual(
                application.findtext(f"{{{app_namespace}}}Application"), "GRIM"
            )


class LayoutPlanningTests(unittest.TestCase):
    def test_polar_ticks_use_signed_labels_without_duplicate_full_circle_ray(self):
        ticks = polar_degree_ticks((0.0, 360.0), 45.0)
        self.assertEqual([value for value, _label in ticks], list(range(0, 360, 45)))
        self.assertEqual(
            [label for _value, label in ticks],
            ["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"],
        )

    def test_fixed_ticks_start_at_the_user_minimum_and_are_bounded(self):
        self.assertEqual(_inclusive_ticks(1.0, 10.0, 2.0), [1.0, 3.0, 5.0, 7.0, 9.0])
        self.assertEqual(_inclusive_ticks(0.5, 1.0, 0.25), [0.5, 0.75, 1.0])
        with self.assertRaisesRegex(ValueError, "more than 1,000"):
            _inclusive_ticks(0.0, 1.0, 0.0001)

    def test_series_accepts_nan_gaps_but_rejects_invalid_axes(self):
        series = PlotSeries.from_values((0.0, 1.0, 2.0), (1.0, math.nan, 3.0))
        self.assertTrue(math.isnan(series.y[1]))
        with self.assertRaisesRegex(ValueError, "x values must all be finite"):
            PlotSeries.from_values((0.0, math.nan), (1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "at least one finite y"):
            PlotSeries.from_values((0.0, 1.0), (math.nan, math.nan))
        with self.assertRaisesRegex(ValueError, "not infinity"):
            PlotSeries.from_values((0.0, 1.0), (1.0, math.inf))

    def test_azimuth_layout_is_fixed_row_major_three_by_two(self):
        geometry = azimuth_3x2_geometry()
        self.assertEqual(len(geometry.plot_frames), 6)
        self.assertLess(geometry.plot_frames[0].left, geometry.plot_frames[1].left)
        self.assertEqual(geometry.plot_frames[0].top, geometry.plot_frames[2].top)
        self.assertGreater(geometry.plot_frames[3].top, geometry.plot_frames[0].top)
        for frame in geometry.plot_frames:
            self.assertLessEqual(frame.right, geometry.width)
            self.assertLessEqual(frame.bottom, geometry.height)

    def test_report_header_and_plot_positions_match_team_template(self):
        expected_title = (0.76, 0.42, 11.82, 0.36)
        expected_legend = (0.76, 1.05, 11.82, 0.22)
        for geometry in (azimuth_3x2_geometry(), frequency_single_geometry()):
            self.assertEqual(
                tuple(
                    round(value / POINTS_PER_INCH, 9)
                    for value in (
                        geometry.title.left,
                        geometry.title.top,
                        geometry.title.width,
                        geometry.title.height,
                    )
                ),
                expected_title,
            )
            self.assertEqual(
                tuple(
                    round(value / POINTS_PER_INCH, 9)
                    for value in (
                        geometry.master_legend.left,
                        geometry.master_legend.top,
                        geometry.master_legend.width,
                        geometry.master_legend.height,
                    )
                ),
                expected_legend,
            )
            self.assertLessEqual(geometry.title.bottom, geometry.master_legend.top)
            self.assertLess(geometry.master_legend.top, geometry.plot_frames[0].top)
            self.assertGreater(
                geometry.master_legend.bottom, geometry.plot_frames[0].top
            )
            self.assertAlmostEqual(
                geometry.plot_frames[0].top / POINTS_PER_INCH, 1.09, places=9
            )
        self.assertAlmostEqual(
            azimuth_3x2_geometry().plot_frames[0].left / POINTS_PER_INCH,
            0.47,
            places=9,
        )

    def test_seven_azimuth_plots_make_six_then_one(self):
        plan = plan_azimuth_slides(
            [make_plot(str(index)) for index in range(7)],
            slide_titles=("First", "Second"),
            footer="Program | Classification",
        )
        self.assertEqual(len(plan.slides), 2)
        self.assertEqual([len(slide.plots) for slide in plan.slides], [6, 1])
        self.assertEqual(
            [placement.slot_index for placement in plan.slides[0].plots],
            list(range(6)),
        )
        self.assertEqual(
            [
                (placement.row_index, placement.column_index)
                for placement in plan.slides[0].plots
            ],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        )
        self.assertEqual(plan.slides[1].plots[0].slot_index, 0)
        self.assertEqual(plan.slides[1].title, "Second")
        self.assertEqual(plan.plot_count, 7)

    def test_polarization_sections_paginate_independently_with_blank_slots(self):
        plots = tuple(
            [make_plot(f"vv-{index}") for index in range(7)]
            + [make_plot(f"hh-{index}") for index in range(7)]
        )
        plan = plan_azimuth_slides(
            plots,
            slide_titles="RCS Report",
            polarization_labels=("VV",) * 7 + ("HH",) * 7,
        )

        self.assertEqual([len(slide.plots) for slide in plan.slides], [6, 1, 6, 1])
        self.assertEqual(
            [slide.title for slide in plan.slides],
            ["RCS Report — VV", "RCS Report — VV", "RCS Report — HH", "RCS Report — HH"],
        )
        self.assertTrue(
            all(
                placement.plot.plot_id.startswith("vv-")
                for slide in plan.slides[:2]
                for placement in slide.plots
            )
        )
        self.assertTrue(
            all(
                placement.plot.plot_id.startswith("hh-")
                for slide in plan.slides[2:]
                for placement in slide.plots
            )
        )
        self.assertEqual([placement.slot_index for placement in plan.slides[1].plots], [0])
        self.assertEqual([placement.slot_index for placement in plan.slides[3].plots], [0])

    def test_polarization_section_label_count_must_match_plots(self):
        with self.assertRaisesRegex(ValueError, "2 polarization labels"):
            plan_azimuth_slides(
                (make_plot("one"), make_plot("two")),
                polarization_labels=("VV",),
            )

    def test_frequency_sweeps_are_one_per_slide(self):
        plots = [make_plot("a", "frequency"), make_plot("b", "frequency")]
        plan = plan_frequency_slides(plots)
        self.assertEqual(len(plan.slides), 2)
        self.assertTrue(all(len(slide.plots) == 1 for slide in plan.slides))
        self.assertEqual(plan.slides[0].plots[0].frame, frequency_single_geometry().plot_frames[0])
        self.assertEqual([slide.title for slide in plan.slides], ["Plot a", "Plot b"])

    def test_frequency_master_titles_use_only_deck_title_and_polarization(self):
        plots = (
            replace(
                make_plot("vv", "frequency"),
                title=(
                    "Frequency Sweep | P90 across Azimuth [90, 270] deg | "
                    "Elevation 0 deg | VV"
                ),
            ),
            replace(
                make_plot("hh", "frequency"),
                title=(
                    "Frequency Sweep | P90 across Azimuth [90, 270] deg | "
                    "Elevation 0 deg | HH"
                ),
            ),
        )
        plan = plan_frequency_slides(
            plots,
            slide_titles="RCS Report",
            polarization_labels=("VV", "HH"),
        )

        self.assertEqual(
            [slide.title for slide in plan.slides],
            ["RCS Report — VV", "RCS Report — HH"],
        )
        self.assertIn("P90 across Azimuth", plan.slides[0].plots[0].plot.title)
        self.assertNotIn("Frequency Sweep", plan.slides[0].title)

    def test_master_legend_is_one_slide_level_key_and_suppresses_plot_legends(self):
        plots = []
        for index in range(2):
            plots.append(
                PlotSpec(
                    plot_id=str(index),
                    kind="azimuth_rect",
                    title=f"Plot {index}",
                    x_label="Azimuth (deg)",
                    y_label="RCS (dBsm)",
                    series=(
                        PlotSeries.from_values((0, 1), (1, 2), label="Baseline"),
                        PlotSeries.from_values((0, 1), (2, 3), label="Modified"),
                    ),
                    show_legend=True,
                )
            )
        plan = plan_azimuth_slides(plots, master_legend=True)
        slide = plan.slides[0]
        self.assertEqual(
            [entry.label for entry in slide.master_legend],
            ["Baseline", "Modified"],
        )
        self.assertTrue(all(not placement.plot.show_legend for placement in slide.plots))

    def test_master_legend_requires_stable_series_order(self):
        first = make_plot("one")
        second = PlotSpec(
            plot_id="two",
            kind="azimuth_rect",
            title="Two",
            x_label="Azimuth (deg)",
            y_label="RCS (dBsm)",
            series=(PlotSeries.from_values((0, 1), (1, 2), label="Different"),),
        )
        with self.assertRaisesRegex(ValueError, "same labeled series"):
            plan_azimuth_slides((first, second), master_legend=True)

    def test_polar_axis_rejects_more_than_one_full_turn(self):
        with self.assertRaisesRegex(ValueError, "at most 360"):
            PlotSpec(
                plot_id="wide-polar",
                kind="azimuth_polar",
                title="Wide",
                x_label="Azimuth (deg)",
                y_label="RCS (dBsm)",
                series=(PlotSeries.from_values((0, 1), (1, 2), label="A"),),
                x_limits=(-181.0, 181.0),
            )

    def test_wrong_plot_family_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Azimuth slides"):
            plan_azimuth_slides([make_plot("frequency", "frequency")])
        with self.assertRaisesRegex(ValueError, "Frequency-sweep"):
            plan_frequency_slides([make_plot("azimuth")])

    def test_combined_report_rejects_duplicate_ids(self):
        azimuth = plan_azimuth_slides([make_plot("same")])
        frequency = plan_frequency_slides([make_plot("same", "frequency")])
        with self.assertRaisesRegex(ValueError, "plot_id values must be unique"):
            combine_plans(azimuth, frequency)


class RenderingTests(unittest.TestCase):
    def test_real_renderer_creates_an_opaque_png(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib is not installed in this headless test runtime.")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plot.png"
            render_plot_png(
                make_plot("real"),
                output,
                width_points=280.0,
                height_points=210.0,
                dpi=80,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1_000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_each_placement_gets_a_predictably_named_png(self):
        plan = plan_azimuth_slides([make_plot("6 GHz"), make_plot("12/GHz")])

        def fake_renderer(plot, output_path, **_kwargs):
            path = Path(output_path)
            path.write_bytes(b"fake png")
            return path

        with tempfile.TemporaryDirectory() as directory:
            rendered = render_plan_images(plan, directory, renderer=fake_renderer)
            self.assertEqual(set(rendered), {(0, 0), (0, 1)})
            self.assertEqual(
                [path.name for path in rendered.values()],
                [
                    "slide_001_slot_1_6_GHz.png",
                    "slide_001_slot_2_12_GHz.png",
                ],
            )

    def test_master_legend_gets_one_transparent_header_asset(self):
        plot = PlotSpec(
            plot_id="legend",
            kind="azimuth_rect",
            title="Legend",
            x_label="Azimuth (deg)",
            y_label="RCS (dBsm)",
            series=(
                PlotSeries.from_values((0, 1), (1, 2), label="Baseline"),
                PlotSeries.from_values((0, 1), (2, 3), label="Modified"),
            ),
        )
        plan = plan_azimuth_slides((plot,), master_legend=True)

        def fake_renderer(_value, output_path, **_kwargs):
            path = Path(output_path)
            path.write_bytes(b"png")
            return path

        with tempfile.TemporaryDirectory() as directory:
            rendered = render_plan_images(
                plan,
                directory,
                renderer=fake_renderer,
                legend_renderer=fake_renderer,
            )
            self.assertEqual(
                set(rendered),
                {(0, MASTER_LEGEND_IMAGE_INDEX), (0, 0)},
            )
            self.assertEqual(
                rendered[(0, MASTER_LEGEND_IMAGE_INDEX)].name,
                "slide_001_master_legend.png",
            )

    def test_real_master_legend_renderer_creates_png(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib is not installed in this headless test runtime.")
        plot = make_plot("legend")
        plan = plan_azimuth_slides((plot,), master_legend=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "legend.png"
            render_master_legend_png(
                plan.slides[0].master_legend,
                output,
                width_points=500.0,
                height_points=20.0,
                dpi=80,
            )
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_master_legend_fails_clearly_instead_of_clipping_long_names(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib is not installed in this headless test runtime.")
        plot = PlotSpec(
            plot_id="wide-legend",
            kind="azimuth_rect",
            title="Wide legend",
            x_label="Azimuth (deg)",
            y_label="RCS (dBsm)",
            series=tuple(
                PlotSeries.from_values(
                    (0.0, 1.0),
                    (0.0, 1.0),
                    label=f"Very long production dataset label number {index}",
                )
                for index in range(12)
            ),
        )
        plan = plan_azimuth_slides((plot,), master_legend=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "master legend does not fit"):
                render_master_legend_png(
                    plan.slides[0].master_legend,
                    Path(directory) / "legend.png",
                    width_points=500.0,
                    height_points=20.0,
                    dpi=80,
                )


class RecordingWriter:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []
        self.image_paths: list[Path] = []

    def write(
        self,
        plan,
        rendered_images,
        output_path,
        *,
        template_path=None,
        template_layouts=None,
    ):
        self.calls.append(
            (plan, output_path, template_path, dict(template_layouts or {}))
        )
        self.image_paths = list(rendered_images.values())
        self.images_existed_during_write = all(path.is_file() for path in self.image_paths)
        if self.fail:
            raise RuntimeError("simulated PowerPoint failure")
        Path(output_path).write_bytes(b"fake pptx")


class PreflightRecordingWriter(RecordingWriter):
    def __init__(self, *, fail_preflight: bool = False):
        super().__init__()
        self.preflight_called = False
        self.fail_preflight = fail_preflight

    def preflight(self):
        self.preflight_called = True
        if self.fail_preflight:
            raise RuntimeError("PowerPoint is unavailable")


def tiny_renderer(_plot, output_path, **_kwargs):
    path = Path(output_path)
    path.write_bytes(b"png")
    return path


class SafeExportTests(unittest.TestCase):
    def test_powerpoint_preflight_runs_before_any_plot_rendering(self):
        plan = plan_azimuth_slides([make_plot("a")])
        writer = PreflightRecordingWriter(fail_preflight=True)
        render_called = False

        def renderer(*_args, **_kwargs):
            nonlocal render_called
            render_called = True
            raise AssertionError("renderer must not run after failed preflight")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.pptx"
            with self.assertRaisesRegex(RuntimeError, "PowerPoint is unavailable"):
                export_powerpoint_report(
                    plan,
                    output,
                    writer=writer,
                    renderer=renderer,
                )
        self.assertTrue(writer.preflight_called)
        self.assertFalse(render_called)

    def test_success_uses_template_staging_and_cleans_plot_images(self):
        plan = plan_azimuth_slides([make_plot("a")])
        writer = RecordingWriter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "blank.potx"
            template.write_bytes(b"template")
            output = root / "report.pptx"
            temp_parent = root / "temporary"
            result = export_powerpoint_report(
                plan,
                output,
                template_path=template,
                writer=writer,
                renderer=tiny_renderer,
                temporary_parent=temp_parent,
            )
            self.assertEqual(result, output.resolve())
            self.assertEqual(output.read_bytes(), b"fake pptx")
            self.assertTrue(writer.images_existed_during_write)
            self.assertTrue(all(not path.exists() for path in writer.image_paths))
            self.assertEqual(writer.calls[0][2], template.resolve())
            self.assertNotEqual(writer.calls[0][1], output.resolve())
            self.assertEqual(list(temp_parent.iterdir()), [])
            self.assertEqual(list(root.glob(".*.tmp.pptx")), [])

    def test_named_layouts_are_normalized_and_forwarded(self):
        plan = plan_azimuth_slides([make_plot("a")])
        writer = RecordingWriter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "named-layouts.pptx"
            template.write_bytes(b"template")
            export_powerpoint_report(
                plan,
                root / "report.pptx",
                template_path=template,
                template_layouts={
                    "azimuth_3x2": f"  {DEFAULT_AZIMUTH_TEMPLATE_LAYOUT}  ",
                },
                writer=writer,
                renderer=tiny_renderer,
            )

        self.assertEqual(
            writer.calls[0][3],
            {"azimuth_3x2": DEFAULT_AZIMUTH_TEMPLATE_LAYOUT},
        )

    def test_named_layouts_without_template_fail_before_preflight_or_render(self):
        plan = plan_azimuth_slides([make_plot("a")])
        writer = PreflightRecordingWriter()
        renderer = mock.Mock(side_effect=AssertionError("must not render"))

        with self.assertRaisesRegex(ValueError, "require.*template"):
            export_powerpoint_report(
                plan,
                "report.pptx",
                template_layouts={
                    "azimuth_3x2": DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                },
                writer=writer,
                renderer=renderer,
            )

        self.assertFalse(writer.preflight_called)
        renderer.assert_not_called()

    def test_failed_write_preserves_existing_destination(self):
        plan = plan_azimuth_slides([make_plot("a")])
        writer = RecordingWriter(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.pptx"
            output.write_bytes(b"previous report")
            temp_parent = root / "temporary"
            with self.assertRaisesRegex(RuntimeError, "simulated PowerPoint failure"):
                export_powerpoint_report(
                    plan,
                    output,
                    writer=writer,
                    renderer=tiny_renderer,
                    temporary_parent=temp_parent,
                )
            self.assertEqual(output.read_bytes(), b"previous report")
            self.assertTrue(all(not path.exists() for path in writer.image_paths))
            self.assertEqual(list(root.glob(".*.tmp.pptx")), [])

    def test_template_is_never_allowed_to_equal_destination(self):
        plan = plan_azimuth_slides([make_plot("a")])
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "blank.pptx"
            template.write_bytes(b"template")
            with self.assertRaisesRegex(ValueError, "different from the template"):
                export_powerpoint_report(
                    plan,
                    template,
                    template_path=template,
                    writer=RecordingWriter(),
                    renderer=tiny_renderer,
                )


class FakeColor:
    RGB = 0


class FakeFont:
    def __init__(self):
        self.Name = ""
        self.Size = 0.0
        self.Bold = MSO_FALSE
        self.Color = FakeColor()


class FakeParagraphFormat:
    Alignment = 0


class FakeTextRange:
    def __init__(self):
        self.Text = ""
        self.Font = FakeFont()
        self.ParagraphFormat = FakeParagraphFormat()


class FakeTextFrame:
    def __init__(self):
        self.MarginLeft = 0
        self.MarginRight = 0
        self.MarginTop = 0
        self.MarginBottom = 0
        self.TextRange = FakeTextRange()


class FakeShape:
    def __init__(self):
        self.TextFrame = FakeTextFrame()
        self.AlternativeText = ""
        self.zorder_commands = []

    def ZOrder(self, command):
        self.zorder_commands.append(command)


class FakeShapes:
    def __init__(self, *, title_placeholder=False):
        self.textboxes = []
        self.pictures = []
        self._title = FakeShape() if title_placeholder else None

    @property
    def Title(self):
        if self._title is None:
            raise AttributeError("This slide has no title placeholder")
        return self._title

    def AddTextbox(self, *args):
        shape = FakeShape()
        self.textboxes.append((args, shape))
        return shape

    def AddPicture(self, *args):
        shape = FakeShape()
        self.pictures.append((args, shape))
        return shape


class FakeSlide:
    def __init__(self, owner=None, *, title_placeholder=False):
        self.Shapes = FakeShapes(title_placeholder=title_placeholder)
        self.owner = owner
        self.CustomLayout = None

    def Delete(self):
        self.owner.items.remove(self)


class FakeSlides:
    def __init__(self, seed_count=0):
        self.items = []
        self.add_calls = []
        self.add_slide_calls = []
        for _ in range(seed_count):
            self.items.append(FakeSlide(self))

    @property
    def Count(self):
        return len(self.items)

    def Item(self, index):
        return self.items[index - 1]

    def Add(self, index, layout):
        self.add_calls.append((index, layout))
        slide = FakeSlide(self)
        self.items.insert(index - 1, slide)
        return slide

    def AddSlide(self, index, custom_layout):
        self.add_slide_calls.append((index, custom_layout))
        slide = FakeSlide(
            self,
            title_placeholder=bool(
                getattr(custom_layout, "has_title_placeholder", False)
            ),
        )
        slide.CustomLayout = custom_layout
        self.items.insert(index - 1, slide)
        return slide


class FakeComCollection:
    def __init__(self, items=()):
        self.items = list(items)

    @property
    def Count(self):
        return len(self.items)

    def Item(self, index):
        return self.items[index - 1]


class FakeCustomLayout:
    def __init__(self, name, *, has_title_placeholder=False):
        self.Name = name
        self.has_title_placeholder = bool(has_title_placeholder)


class FakeSlideMaster:
    def __init__(self, name, layouts=()):
        self.Name = name
        self.CustomLayouts = FakeComCollection(layouts)


class FakeDesign:
    def __init__(self, name, master_name, layouts=()):
        self.Name = name
        self.SlideMaster = FakeSlideMaster(master_name, layouts)


class FakePageSetup:
    def __init__(self, width=800.0, height=450.0):
        self.SlideWidth = width
        self.SlideHeight = height


class FakePresentation:
    def __init__(
        self,
        seed_count=1,
        *,
        width=800.0,
        height=450.0,
        on_save=None,
        save_error=None,
        close_error=None,
        designs=(),
    ):
        self.Slides = FakeSlides(seed_count)
        self.PageSetup = FakePageSetup(width, height)
        self.closed = False
        self.saved = None
        self.Saved = MSO_FALSE
        self.on_save = on_save
        self.save_error = save_error
        self.close_error = close_error
        self.Designs = FakeComCollection(designs)

    def SaveAs(self, path, file_format):
        if self.save_error is not None:
            raise self.save_error
        self.saved = (Path(path), file_format)
        Path(path).write_bytes(b"pptx from fake COM")
        if self.on_save is not None:
            self.on_save()

    def Close(self):
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class FakePresentations:
    def __init__(self, presentation, *, initial_count=0):
        self.presentation = presentation
        self.initial_count = initial_count
        self.report_open = False
        self.open_args = None
        self.add_args = None

    @property
    def Count(self):
        report_count = int(self.report_open and not self.presentation.closed)
        return self.initial_count + report_count

    def Open(self, path, **kwargs):
        self.open_args = (Path(path), kwargs)
        self.report_open = True
        return self.presentation

    def Add(self, **kwargs):
        self.add_args = kwargs
        self.report_open = True
        return self.presentation


class FakeApplication:
    def __init__(
        self,
        presentation,
        *,
        initial_presentation_count=0,
        visible=True,
        display_alerts=7,
    ):
        self.Presentations = FakePresentations(
            presentation,
            initial_count=initial_presentation_count,
        )
        self.Visible = visible
        self.DisplayAlerts = display_alerts
        self.quit_called = False

    def Quit(self):
        self.quit_called = True


class ComBridgeFakeTests(unittest.TestCase):
    def test_bridge_clears_seed_and_writes_fixed_shapes(self):
        plan = plan_azimuth_slides(
            [make_plot("one"), make_plot("two")], footer="Program | Classification"
        )
        presentation = FakePresentation(seed_count=2)
        application = FakeApplication(presentation)
        bridge = PowerPointComBridge(application_factory=lambda: application)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "blank.pptx"
            template.write_bytes(b"template")
            images = {}
            for placement_index in range(2):
                image = root / f"plot-{placement_index}.png"
                image.write_bytes(b"png")
                images[(0, placement_index)] = image
            output = root / "report.pptx"
            bridge.write(plan, images, output, template_path=template)

            self.assertTrue(output.is_file())
            self.assertTrue(presentation.closed)
            self.assertEqual(presentation.Saved, MSO_TRUE)
            self.assertFalse(application.quit_called)
            self.assertTrue(application.Visible)
            self.assertEqual(application.DisplayAlerts, 7)
            self.assertEqual(application.Presentations.open_args[0], template)
            self.assertEqual(application.Presentations.open_args[1]["WithWindow"], MSO_FALSE)
            self.assertEqual(len(presentation.Slides.items), 1)
            slide = presentation.Slides.items[0]
            self.assertEqual(len(slide.Shapes.pictures), 2)
            self.assertEqual(len(slide.Shapes.textboxes), 2)  # title and footer only
            self.assertEqual(slide.Shapes.pictures[0][1].AlternativeText, "Plot one")

    def test_bridge_places_master_legend_last_and_brings_it_to_front(self):
        plan = plan_azimuth_slides((make_plot("one"),), master_legend=True)
        presentation = FakePresentation(seed_count=0, width=960.0, height=540.0)
        application = FakeApplication(presentation)
        bridge = PowerPointComBridge(application_factory=lambda: application)
        geometry = azimuth_3x2_geometry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legend = root / "legend.png"
            plot = root / "plot.png"
            legend.write_bytes(b"legend")
            plot.write_bytes(b"plot")
            bridge.write(
                plan,
                {
                    (0, MASTER_LEGEND_IMAGE_INDEX): legend,
                    (0, 0): plot,
                },
                root / "report.pptx",
            )
        slide = presentation.Slides.items[0]
        self.assertEqual(len(slide.Shapes.pictures), 2)
        self.assertEqual(slide.Shapes.pictures[0][1].AlternativeText, "Plot one")
        legend_args, legend_shape = slide.Shapes.pictures[-1]
        self.assertEqual(
            legend_args[3:],
            (
                geometry.master_legend.left,
                geometry.master_legend.top,
                geometry.master_legend.width,
                geometry.master_legend.height,
            ),
        )
        self.assertEqual(legend_shape.AlternativeText, "Dataset legend: HH")
        self.assertEqual(legend_shape.zorder_commands, [MSO_BRING_TO_FRONT])

    def test_new_deck_is_forced_to_widescreen(self):
        plan = plan_frequency_slides([make_plot("frequency", "frequency")])
        presentation = FakePresentation(seed_count=0)
        application = FakeApplication(presentation)
        bridge = PowerPointComBridge(application_factory=lambda: application)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "plot.png"
            image.write_bytes(b"png")
            bridge.write(plan, {(0, 0): image}, root / "report.pptx")
            self.assertAlmostEqual(
                presentation.PageSetup.SlideWidth / presentation.PageSetup.SlideHeight,
                16.0 / 9.0,
                places=6,
            )
            self.assertEqual(application.Presentations.add_args["WithWindow"], MSO_FALSE)
            self.assertEqual(presentation.Slides.add_calls, [(1, PP_LAYOUT_BLANK)])
            self.assertEqual(presentation.Slides.add_slide_calls, [])

    def test_template_uses_named_layout_for_each_plan_family(self):
        azimuth_plan = plan_azimuth_slides([make_plot("azimuth")])
        frequency_plan = plan_frequency_slides(
            [make_plot("frequency", "frequency")]
        )
        plan = combine_plans(azimuth_plan, frequency_plan)
        azimuth_layout = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT)
        frequency_layout = FakeCustomLayout(DEFAULT_FREQUENCY_TEMPLATE_LAYOUT)
        unrelated_layout = FakeCustomLayout("Unrelated")
        design = FakeDesign(
            "Temporary Design",
            "GRIM Report Master",
            (frequency_layout, unrelated_layout, azimuth_layout),
        )
        presentation = FakePresentation(seed_count=2, designs=(design,))
        application = FakeApplication(presentation)
        bridge = PowerPointComBridge(application_factory=lambda: application)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pptx"
            template.write_bytes(b"template")
            images = {}
            for slide_index in range(2):
                image = root / f"plot-{slide_index}.png"
                image.write_bytes(b"png")
                images[(slide_index, 0)] = image
            bridge.write(
                plan,
                images,
                root / "report.pptx",
                template_path=template,
                template_layouts={
                    "azimuth_3x2": DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                    "frequency_single": DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
                },
            )

        self.assertEqual(presentation.Slides.add_calls, [])
        self.assertEqual(
            presentation.Slides.add_slide_calls,
            [(3, azimuth_layout), (4, frequency_layout)],
        )
        self.assertIs(presentation.Slides.items[0].CustomLayout, azimuth_layout)
        self.assertIs(presentation.Slides.items[1].CustomLayout, frequency_layout)

    def test_named_layout_title_placeholder_keeps_master_formatting(self):
        plan = plan_frequency_slides(
            [make_plot("frequency", "frequency")],
            slide_titles="Vehicle RCS",
        )
        layout = FakeCustomLayout(
            DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
            has_title_placeholder=True,
        )
        presentation = FakePresentation(
            seed_count=1,
            designs=(FakeDesign("Design", "Master", (layout,)),),
        )
        application = FakeApplication(presentation)
        bridge = PowerPointComBridge(application_factory=lambda: application)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pptx"
            image = root / "plot.png"
            template.write_bytes(b"template")
            image.write_bytes(b"png")
            bridge.write(
                plan,
                {(0, 0): image},
                root / "report.pptx",
                template_path=template,
                template_layouts={
                    "frequency_single": DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
                },
            )

        slide = presentation.Slides.items[0]
        self.assertEqual(slide.Shapes.Title.TextFrame.TextRange.Text, "Vehicle RCS")
        self.assertEqual(slide.Shapes.textboxes, [])
        # The bridge assigns only the text; all master typography remains
        # inherited instead of being overwritten with GRIM's Arial fallback.
        self.assertEqual(slide.Shapes.Title.TextFrame.TextRange.Font.Name, "")
        self.assertEqual(slide.Shapes.Title.TextFrame.TextRange.Font.Size, 0.0)

    def test_bare_duplicate_layout_requires_master_qualifier_before_clearing(self):
        plan = plan_azimuth_slides([make_plot("azimuth")])
        first = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT)
        second = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT.lower())
        presentation = FakePresentation(
            seed_count=1,
            designs=(
                FakeDesign("Design A", "Master A", (first,)),
                FakeDesign("Design B", "Master B", (second,)),
            ),
        )
        bridge = PowerPointComBridge(
            application_factory=lambda: FakeApplication(presentation)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pptx"
            template.write_bytes(b"template")
            with self.assertRaisesRegex(RuntimeError, "Master :: Layout"):
                bridge.write(
                    plan,
                    {},
                    root / "report.pptx",
                    template_path=template,
                    template_layouts={
                        "azimuth_3x2": DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                    },
                )

        self.assertEqual(presentation.Slides.Count, 1)
        self.assertEqual(presentation.Slides.add_calls, [])
        self.assertEqual(presentation.Slides.add_slide_calls, [])
        self.assertIsNone(presentation.saved)
        self.assertTrue(presentation.closed)

    def test_master_qualified_layout_disambiguates_case_insensitively(self):
        plan = plan_azimuth_slides([make_plot("azimuth")])
        first = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT)
        second = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT)
        presentation = FakePresentation(
            seed_count=0,
            designs=(
                FakeDesign("Design A", "Master A", (first,)),
                FakeDesign("Design B", "Master B", (second,)),
            ),
        )
        bridge = PowerPointComBridge(
            application_factory=lambda: FakeApplication(presentation)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pptx"
            template.write_bytes(b"template")
            image = root / "plot.png"
            image.write_bytes(b"png")
            bridge.write(
                plan,
                {(0, 0): image},
                root / "report.pptx",
                template_path=template,
                template_layouts={
                    "azimuth_3x2": " master b :: grim azimuth 3X2 ",
                },
            )

        self.assertEqual(presentation.Slides.add_slide_calls, [(1, second)])

    def test_duplicate_layout_names_inside_one_master_require_renaming(self):
        plan = plan_azimuth_slides([make_plot("azimuth")])
        first = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT)
        second = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT)
        presentation = FakePresentation(
            seed_count=1,
            designs=(FakeDesign("Design A", "Master A", (first, second)),),
        )
        bridge = PowerPointComBridge(
            application_factory=lambda: FakeApplication(presentation)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pptx"
            template.write_bytes(b"template")
            with self.assertRaisesRegex(RuntimeError, "Rename the duplicate") as error:
                bridge.write(
                    plan,
                    {},
                    root / "report.pptx",
                    template_path=template,
                    template_layouts={
                        "azimuth_3x2": DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                    },
                )

        self.assertIn("rename required", str(error.exception))
        self.assertNotIn("Use 'Master :: Layout'", str(error.exception))
        self.assertEqual(presentation.Slides.Count, 1)

    def test_duplicate_master_names_offer_unique_design_qualifiers(self):
        plan = plan_azimuth_slides([make_plot("azimuth")])
        first = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT)
        second = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT)
        presentation = FakePresentation(
            seed_count=0,
            designs=(
                FakeDesign("Design One", "Shared Master", (first,)),
                FakeDesign("Design Two", "Shared Master", (second,)),
            ),
        )
        bridge = PowerPointComBridge(
            application_factory=lambda: FakeApplication(presentation)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pptx"
            template.write_bytes(b"template")
            with self.assertRaisesRegex(
                RuntimeError,
                "Design One :: GRIM Azimuth 3x2.*Design Two :: GRIM Azimuth 3x2",
            ):
                bridge.write(
                    plan,
                    {},
                    root / "ambiguous.pptx",
                    template_path=template,
                    template_layouts={
                        "azimuth_3x2": DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                    },
                )

            image = root / "plot.png"
            image.write_bytes(b"png")
            bridge.write(
                plan,
                {(0, 0): image},
                root / "report.pptx",
                template_path=template,
                template_layouts={
                    "azimuth_3x2": "design two :: grim azimuth 3x2",
                },
            )

        self.assertEqual(presentation.Slides.add_slide_calls, [(1, second)])

    def test_missing_required_layout_lists_available_before_clearing(self):
        plan = plan_frequency_slides([make_plot("frequency", "frequency")])
        other = FakeCustomLayout("Other Layout")
        presentation = FakePresentation(
            seed_count=1,
            designs=(FakeDesign("Design A", "Master A", (other,)),),
        )
        bridge = PowerPointComBridge(
            application_factory=lambda: FakeApplication(presentation)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pptx"
            template.write_bytes(b"template")
            with self.assertRaisesRegex(
                RuntimeError,
                "GRIM Frequency Sweep.*Master A :: Other Layout",
            ):
                bridge.write(
                    plan,
                    {},
                    root / "report.pptx",
                    template_path=template,
                    template_layouts={
                        "frequency_single": DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
                    },
                )

        self.assertEqual(presentation.Slides.Count, 1)
        self.assertEqual(presentation.Slides.add_calls, [])
        self.assertEqual(presentation.Slides.add_slide_calls, [])

    def test_unused_family_layout_is_not_required(self):
        plan = plan_azimuth_slides([make_plot("azimuth")])
        azimuth_layout = FakeCustomLayout(DEFAULT_AZIMUTH_TEMPLATE_LAYOUT)
        presentation = FakePresentation(
            seed_count=0,
            designs=(
                FakeDesign("Temporary Design", "GRIM Report Master", (azimuth_layout,)),
            ),
        )
        bridge = PowerPointComBridge(
            application_factory=lambda: FakeApplication(presentation)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pptx"
            template.write_bytes(b"template")
            image = root / "plot.png"
            image.write_bytes(b"png")
            bridge.write(
                plan,
                {(0, 0): image},
                root / "report.pptx",
                template_path=template,
                template_layouts={
                    "azimuth_3x2": DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                    "frequency_single": DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
                },
            )

        self.assertEqual(presentation.Slides.add_slide_calls, [(1, azimuth_layout)])

    def test_non_widescreen_template_is_rejected_before_adding_report_slides(self):
        plan = plan_azimuth_slides([make_plot("one")])
        presentation = FakePresentation(seed_count=1, width=720.0, height=540.0)
        application = FakeApplication(presentation)
        bridge = PowerPointComBridge(application_factory=lambda: application)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "legacy-4x3.pptx"
            template.write_bytes(b"template")
            image = root / "plot.png"
            image.write_bytes(b"png")
            with self.assertRaisesRegex(RuntimeError, "widescreen 16:9"):
                bridge.write(
                    plan,
                    {(0, 0): image},
                    root / "report.pptx",
                    template_path=template,
                )
        self.assertTrue(presentation.closed)
        self.assertFalse(application.quit_called)

    def test_shared_powerpoint_and_user_presentations_remain_open(self):
        plan = plan_frequency_slides([make_plot("frequency", "frequency")])
        presentation = FakePresentation(seed_count=0)
        application = FakeApplication(
            presentation,
            initial_presentation_count=2,
            visible=True,
            display_alerts=11,
        )
        bridge = PowerPointComBridge(application_factory=lambda: application)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "plot.png"
            image.write_bytes(b"png")
            bridge.write(plan, {(0, 0): image}, root / "report.pptx")

        self.assertTrue(presentation.closed)
        self.assertEqual(application.Presentations.Count, 2)
        self.assertFalse(application.quit_called)
        self.assertTrue(application.Visible)
        self.assertEqual(application.DisplayAlerts, 11)

    def test_preflight_does_not_quit_shared_powerpoint(self):
        presentation = FakePresentation(seed_count=0)
        application = FakeApplication(
            presentation,
            initial_presentation_count=1,
            visible=True,
            display_alerts=13,
        )

        PowerPointComBridge(application_factory=lambda: application).preflight()

        self.assertFalse(application.quit_called)
        self.assertEqual(application.Presentations.Count, 1)
        self.assertTrue(application.Visible)
        self.assertEqual(application.DisplayAlerts, 13)

    def test_user_deck_opened_during_export_remains_open(self):
        plan = plan_frequency_slides([make_plot("frequency", "frequency")])
        application = None

        def user_opens_presentation():
            application.Presentations.initial_count = 1

        presentation = FakePresentation(seed_count=0, on_save=user_opens_presentation)
        application = FakeApplication(presentation)
        bridge = PowerPointComBridge(application_factory=lambda: application)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "plot.png"
            image.write_bytes(b"png")
            bridge.write(plan, {(0, 0): image}, root / "report.pptx")

        self.assertTrue(presentation.closed)
        self.assertEqual(application.Presentations.Count, 1)
        self.assertFalse(application.quit_called)

    def test_active_powerpoint_is_reused_without_dispatch_or_quit(self):
        presentation = FakePresentation(seed_count=0)
        application = FakeApplication(
            presentation,
            initial_presentation_count=1,
        )
        pythoncom_fake = SimpleNamespace(
            CoInitialize=mock.Mock(),
            CoUninitialize=mock.Mock(),
        )
        client_fake = SimpleNamespace(
            GetActiveObject=mock.Mock(return_value=application),
            DispatchEx=mock.Mock(),
        )
        win32com_fake = SimpleNamespace(client=client_fake)

        with (
            mock.patch.object(ppt_report.sys, "platform", "win32"),
            mock.patch.object(ppt_report, "pythoncom", pythoncom_fake),
            mock.patch.object(ppt_report, "win32com", win32com_fake),
        ):
            PowerPointComBridge().preflight()

        client_fake.GetActiveObject.assert_called_once_with("PowerPoint.Application")
        client_fake.DispatchEx.assert_not_called()
        pythoncom_fake.CoInitialize.assert_called_once_with()
        pythoncom_fake.CoUninitialize.assert_called_once_with()
        self.assertFalse(application.quit_called)

    def test_preflight_never_quits_powerpoint_started_by_grim(self):
        presentation = FakePresentation(seed_count=0)
        application = FakeApplication(presentation)
        pythoncom_fake = SimpleNamespace(
            CoInitialize=mock.Mock(),
            CoUninitialize=mock.Mock(),
        )
        client_fake = SimpleNamespace(
            GetActiveObject=mock.Mock(side_effect=RuntimeError("not running")),
            DispatchEx=mock.Mock(return_value=application),
        )
        win32com_fake = SimpleNamespace(client=client_fake)

        with (
            mock.patch.object(ppt_report.sys, "platform", "win32"),
            mock.patch.object(ppt_report, "pythoncom", pythoncom_fake),
            mock.patch.object(ppt_report, "win32com", win32com_fake),
        ):
            PowerPointComBridge().preflight()

        client_fake.DispatchEx.assert_called_once_with("PowerPoint.Application")
        pythoncom_fake.CoUninitialize.assert_called_once_with()
        self.assertFalse(application.quit_called)

    def test_failed_export_in_shared_powerpoint_closes_only_grim_presentation(self):
        plan = plan_frequency_slides([make_plot("frequency", "frequency")])
        presentation = FakePresentation(
            seed_count=0,
            save_error=RuntimeError("save failed"),
        )
        application = FakeApplication(
            presentation,
            initial_presentation_count=2,
            visible=True,
            display_alerts=17,
        )
        bridge = PowerPointComBridge(application_factory=lambda: application)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "plot.png"
            image.write_bytes(b"png")
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                bridge.write(plan, {(0, 0): image}, root / "report.pptx")

        self.assertTrue(presentation.closed)
        self.assertEqual(presentation.Saved, MSO_TRUE)
        self.assertEqual(application.Presentations.Count, 2)
        self.assertFalse(application.quit_called)
        self.assertTrue(application.Visible)
        self.assertEqual(application.DisplayAlerts, 17)

    def test_close_failure_is_reported_without_quitting_powerpoint(self):
        plan = plan_frequency_slides([make_plot("frequency", "frequency")])
        presentation = FakePresentation(
            seed_count=0,
            close_error=RuntimeError("close failed"),
        )
        application = FakeApplication(presentation)
        bridge = PowerPointComBridge(application_factory=lambda: application)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "plot.png"
            image.write_bytes(b"png")
            with self.assertRaisesRegex(RuntimeError, "temporary presentation"):
                bridge.write(plan, {(0, 0): image}, root / "report.pptx")

        self.assertFalse(presentation.closed)
        self.assertFalse(application.quit_called)

    def test_operation_and_close_failures_are_both_reported(self):
        plan = plan_frequency_slides([make_plot("frequency", "frequency")])
        presentation = FakePresentation(
            seed_count=0,
            save_error=RuntimeError("save failed"),
            close_error=RuntimeError("close failed"),
        )
        application = FakeApplication(presentation)
        bridge = PowerPointComBridge(application_factory=lambda: application)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "plot.png"
            image.write_bytes(b"png")
            with self.assertRaisesRegex(
                RuntimeError,
                "save failed.*also could not close.*close failed",
            ):
                bridge.write(plan, {(0, 0): image}, root / "report.pptx")

        self.assertFalse(presentation.closed)
        self.assertFalse(application.quit_called)


if __name__ == "__main__":
    unittest.main()
