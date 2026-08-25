from __future__ import annotations

import tempfile
import unittest
import math
from pathlib import Path

from ppt_report import (
    MSO_FALSE,
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
    render_plan_images,
    render_plot_png,
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


class LayoutPlanningTests(unittest.TestCase):
    def test_polar_ticks_use_signed_labels_without_duplicate_full_circle_ray(self):
        ticks = polar_degree_ticks((0.0, 360.0), 45.0)
        self.assertEqual([value for value, _label in ticks], list(range(0, 360, 45)))
        self.assertEqual(
            [label for _value, label in ticks],
            ["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"],
        )

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

    def test_frequency_sweeps_are_one_per_slide(self):
        plots = [make_plot("a", "frequency"), make_plot("b", "frequency")]
        plan = plan_frequency_slides(plots)
        self.assertEqual(len(plan.slides), 2)
        self.assertTrue(all(len(slide.plots) == 1 for slide in plan.slides))
        self.assertEqual(plan.slides[0].plots[0].frame, frequency_single_geometry().plot_frames[0])
        self.assertEqual([slide.title for slide in plan.slides], ["Plot a", "Plot b"])

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


class RecordingWriter:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []
        self.image_paths: list[Path] = []

    def write(self, plan, rendered_images, output_path, *, template_path=None):
        self.calls.append((plan, output_path, template_path))
        self.image_paths = list(rendered_images.values())
        self.images_existed_during_write = all(path.is_file() for path in self.image_paths)
        if self.fail:
            raise RuntimeError("simulated PowerPoint failure")
        Path(output_path).write_bytes(b"fake pptx")


def tiny_renderer(_plot, output_path, **_kwargs):
    path = Path(output_path)
    path.write_bytes(b"png")
    return path


class SafeExportTests(unittest.TestCase):
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
            with self.assertRaisesRegex(ValueError, "different from the blank template"):
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


class FakeShapes:
    def __init__(self):
        self.textboxes = []
        self.pictures = []

    def AddTextbox(self, *args):
        shape = FakeShape()
        self.textboxes.append((args, shape))
        return shape

    def AddPicture(self, *args):
        shape = FakeShape()
        self.pictures.append((args, shape))
        return shape


class FakeSlide:
    def __init__(self, owner=None):
        self.Shapes = FakeShapes()
        self.owner = owner

    def Delete(self):
        self.owner.items.remove(self)


class FakeSlides:
    def __init__(self, seed_count=0):
        self.items = []
        for _ in range(seed_count):
            self.items.append(FakeSlide(self))

    @property
    def Count(self):
        return len(self.items)

    def Item(self, index):
        return self.items[index - 1]

    def Add(self, index, _layout):
        slide = FakeSlide(self)
        self.items.insert(index - 1, slide)
        return slide


class FakePageSetup:
    def __init__(self, width=800.0, height=450.0):
        self.SlideWidth = width
        self.SlideHeight = height


class FakePresentation:
    def __init__(self, seed_count=1, *, width=800.0, height=450.0):
        self.Slides = FakeSlides(seed_count)
        self.PageSetup = FakePageSetup(width, height)
        self.closed = False
        self.saved = None

    def SaveAs(self, path, file_format):
        self.saved = (Path(path), file_format)
        Path(path).write_bytes(b"pptx from fake COM")

    def Close(self):
        self.closed = True


class FakePresentations:
    def __init__(self, presentation):
        self.presentation = presentation
        self.open_args = None
        self.add_args = None

    def Open(self, path, **kwargs):
        self.open_args = (Path(path), kwargs)
        return self.presentation

    def Add(self, **kwargs):
        self.add_args = kwargs
        return self.presentation


class FakeApplication:
    def __init__(self, presentation):
        self.Presentations = FakePresentations(presentation)
        self.Visible = None
        self.DisplayAlerts = None
        self.quit_called = False

    def Quit(self):
        self.quit_called = True


class ComBridgeFakeTests(unittest.TestCase):
    def test_bridge_starts_private_app_clears_seed_and_writes_fixed_shapes(self):
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
            self.assertTrue(application.quit_called)
            self.assertEqual(application.Presentations.open_args[0], template)
            self.assertEqual(application.Presentations.open_args[1]["WithWindow"], MSO_FALSE)
            self.assertEqual(len(presentation.Slides.items), 1)
            slide = presentation.Slides.items[0]
            self.assertEqual(len(slide.Shapes.pictures), 2)
            self.assertEqual(len(slide.Shapes.textboxes), 3)  # title, footer, page
            self.assertEqual(slide.Shapes.pictures[0][1].AlternativeText, "Plot one")

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
        self.assertTrue(application.quit_called)


if __name__ == "__main__":
    unittest.main()
