from __future__ import annotations

import unittest
from unittest import mock

from PySide6.QtWidgets import QApplication

from ppt_image_imprinter import (
    ApplyOptions,
    CropProfile,
    MSO_GROUP,
    MSO_PICTURE,
    MainWindow,
    PowerPointBridge,
    ShapeProfile,
    apply_profile_to_shape,
    capture_profile_from_shape,
    map_profiles_to_targets,
)


class FakePictureFormat:
    def __init__(self, left=1.0, top=2.0, right=3.0, bottom=4.0):
        self.CropLeft = left
        self.CropTop = top
        self.CropRight = right
        self.CropBottom = bottom


class FakeSlide:
    def __init__(self, index):
        self.SlideIndex = index


class FakeShape:
    def __init__(
        self,
        name="Picture 1",
        *,
        left=10.0,
        top=20.0,
        width=100.0,
        height=50.0,
        slide=1,
    ):
        self.Type = MSO_PICTURE
        self.Name = name
        self.Left = left
        self.Top = top
        self._width = width
        self.Height = height
        self.Parent = FakeSlide(slide)
        self.PictureFormat = FakePictureFormat()
        self.LockAspectRatio = -1
        self.fail_next_width_write = False

    @property
    def Width(self):
        return self._width

    @Width.setter
    def Width(self, value):
        if self.fail_next_width_write:
            self.fail_next_width_write = False
            raise RuntimeError("simulated protected shape")
        self._width = value


class FakeCollection:
    def __init__(self, *items):
        self._items = list(items)
        self.Count = len(items)

    def Item(self, index):
        return self._items[index - 1]


class FakeGroup:
    Type = MSO_GROUP

    def __init__(self, *items):
        self.GroupItems = FakeCollection(*items)


class FakeBridge(PowerPointBridge):
    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def selected_picture_shapes(self):
        return self.targets, 0


def make_profile(name="source", value=1.0):
    return ShapeProfile(
        slide_index=1,
        name=name,
        left=10.0 * value,
        top=20.0 * value,
        width=30.0 * value,
        height=40.0 * value,
        crop=CropProfile(value, 2.0 * value, 3.0 * value, 4.0 * value),
    )


class ProfileMappingTests(unittest.TestCase):
    def test_one_profile_broadcasts_to_all_targets(self):
        profile = make_profile()
        targets = [object(), object(), object()]
        pairs = map_profiles_to_targets([profile], targets)
        self.assertEqual([target for _, target in pairs], targets)
        self.assertTrue(all(source is profile for source, _ in pairs))

    def test_mismatched_counts_use_first_profile_for_every_target(self):
        profiles = [make_profile("a"), make_profile("b")]
        for targets in ([object()], [object(), object(), object()]):
            with self.subTest(target_count=len(targets)):
                pairs = map_profiles_to_targets(profiles, targets)
                self.assertEqual(pairs, [(profiles[0], target) for target in targets])

    def test_multiple_profiles_map_in_order(self):
        profiles = [make_profile("a"), make_profile("b")]
        targets = [object(), object()]
        pairs = map_profiles_to_targets(profiles, targets)
        self.assertEqual(pairs, list(zip(profiles, targets)))


class ShapeFormattingTests(unittest.TestCase):
    def test_capture_reads_geometry_crop_and_slide(self):
        shape = FakeShape("Source", left=11, top=22, width=333, height=144, slide=7)
        shape.PictureFormat = FakePictureFormat(5, 6, 7, 8)
        profile = capture_profile_from_shape(shape)
        self.assertEqual(profile.slide_index, 7)
        self.assertEqual(profile.name, "Source")
        self.assertEqual((profile.left, profile.top), (11.0, 22.0))
        self.assertEqual((profile.width, profile.height), (333.0, 144.0))
        self.assertEqual(profile.crop, CropProfile(5.0, 6.0, 7.0, 8.0))

    def test_apply_writes_all_enabled_properties_and_restores_aspect_lock(self):
        target = FakeShape(left=1, top=2, width=3, height=4)
        target.LockAspectRatio = -1
        profile = make_profile(value=2.0)
        apply_profile_to_shape(profile, target, ApplyOptions())
        self.assertEqual((target.Left, target.Top), (20.0, 40.0))
        self.assertEqual((target.Width, target.Height), (60.0, 80.0))
        self.assertEqual(target.LockAspectRatio, -1)
        self.assertEqual(
            (
                target.PictureFormat.CropLeft,
                target.PictureFormat.CropTop,
                target.PictureFormat.CropRight,
                target.PictureFormat.CropBottom,
            ),
            (2.0, 4.0, 6.0, 8.0),
        )

    def test_apply_honors_individual_options(self):
        target = FakeShape(left=1, top=2, width=3, height=4)
        original_crop = capture_profile_from_shape(target).crop
        apply_profile_to_shape(
            make_profile(value=3.0),
            target,
            ApplyOptions(location=True, size=False, crop=False),
        )
        self.assertEqual((target.Left, target.Top), (30.0, 60.0))
        self.assertEqual((target.Width, target.Height), (3, 4))
        self.assertEqual(capture_profile_from_shape(target).crop, original_crop)

    def test_group_collection_recurses_and_counts_nonpictures(self):
        picture_a = FakeShape("A")
        picture_b = FakeShape("B")
        nonpicture = type("FakeText", (), {"Type": 17})()
        pictures, skipped = PowerPointBridge._collect_picture_shapes(
            FakeGroup(picture_a, FakeGroup(nonpicture, picture_b))
        )
        self.assertEqual(pictures, [picture_a, picture_b])
        self.assertEqual(skipped, 1)

    def test_failed_batch_rolls_back_previously_changed_targets(self):
        first = FakeShape("First", left=1, top=2, width=3, height=4)
        second = FakeShape("Second", left=5, top=6, width=7, height=8)
        first_before = capture_profile_from_shape(first)
        second_before = capture_profile_from_shape(second)
        second.fail_next_width_write = True
        bridge = FakeBridge([first, second])

        with self.assertRaisesRegex(RuntimeError, "PowerPoint rejected"):
            bridge.apply_to_selected(
                [make_profile("one", 2.0), make_profile("two", 3.0)],
                ApplyOptions(),
            )

        self.assertEqual(capture_profile_from_shape(first), first_before)
        self.assertEqual(capture_profile_from_shape(second), second_before)


class ImprinterWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_capture_imprint_options_mismatch_and_clear(self):
        window = MainWindow(settings=mock.Mock(value=mock.Mock(return_value="Dark")))
        self.addCleanup(window.close)
        first = FakeShape("Source A", left=111, top=222, width=80, height=90)
        second = FakeShape("Source B", left=333, top=444)
        window.bridge = FakeBridge([first, second])
        self.assertFalse(window.btn_apply.isEnabled())
        window.btn_capture.click()
        self.assertEqual(window.btn_capture.text(), "Capture (2)")
        self.assertTrue(window.btn_apply.isEnabled())

        targets = [FakeShape("Target", left=1, top=2, width=3, height=4) for _ in range(3)]
        window.bridge.targets = targets
        window.chk_size.setChecked(False)
        window.chk_crop.setChecked(False)
        window.btn_apply.click()
        for target in targets:
            self.assertEqual((target.Left, target.Top), (111, 222))
            self.assertEqual((target.Width, target.Height), (3, 4))
        self.assertIn("first captured profile", window.status.text())
        window.chk_location.setChecked(False)
        self.assertFalse(window.btn_apply.isEnabled())
        window.btn_clear.click()
        self.assertEqual(window.captured_profiles, [])
        self.assertEqual(window.btn_capture.text(), "Capture")
        self.assertEqual(window.status.text(), "Ready")

    def test_saved_grim_palette_and_legacy_name_work_without_main_gui(self):
        from grim_palette import APPLICATION_PALETTES

        for name in (*APPLICATION_PALETTES, "Raytheon-inspired", "unknown"):
            with self.subTest(name=name):
                settings = mock.Mock(value=mock.Mock(return_value=name))
                window = MainWindow(settings=settings)
                try:
                    expected = "Raytheon" if name == "Raytheon-inspired" else name
                    if expected not in APPLICATION_PALETTES:
                        expected = "Dark"
                    self.assertEqual(window.application_palette_name, expected)
                    self.assertIn(APPLICATION_PALETTES[expected]["win_bg"], window.styleSheet())
                    settings.setValue.assert_not_called()
                finally:
                    window.close()


if __name__ == "__main__":
    unittest.main()
