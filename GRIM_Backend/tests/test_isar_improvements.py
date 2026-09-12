"""Physical and workflow acceptance tests for the 2026 ISAR audit changes."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.modes import isar_mode as isar
from GRIM_Backend.isar.geometry import angular_bands, angular_sublooks, axis_edges
from GRIM_Backend.isar.quality import C0, plan_isar, image_contract, physical_coefficients
from GRIM_Backend.isar.operators import PolarPointOperator, native_image_residual
from GRIM_Backend.isar.comparison import compare_images, hydrate_band
from GRIM_Backend.isar.artifact import build_isar_manifest, save_isar_artifact, load_isar_artifact
from GRIM_Backend.isar.recipes import recipe_from_params, recipe_arguments, save_recipe, load_recipe


def grid(az=None, freq=None, field=None, unit='Hz', elevation=0.):
    az = np.linspace(-5., 5., 65) if az is None else np.asarray(az)
    freq = np.linspace(9e9, 11e9, 65) if freq is None else np.asarray(freq)
    field = np.ones((len(az), len(freq)), np.complex64) if field is None else field
    return RcsGrid(az, [elevation], freq, ['VV'], rcs=np.asarray(field, np.complex64)[:, None, :, None],
        units={'frequency': unit, 'azimuth': 'deg', 'elevation': 'deg', 'time_convention': 'exp(+jwt)'},
        extra={'phase_reference': 'fixed origin', 'range_phase_convention': 'S~exp(-j*2*k*R)',
               'measurement_geometry': 'far-field monostatic', 'motion_compensation': 'stable'})


def params(source, **options):
    return dict(dataset=source, bands=[list(range(len(source.azimuths)))],
        freq_indices_sorted=list(range(len(source.frequencies))), freq_hz=np.asarray(source.frequencies) * isar._unit_to_hz_scale(source.units['frequency']),
        elev_idx=0, elevation_deg=float(source.elevations[0]), pol_idx=0, recon='accurate', window_name='Rectangular',
        unit_name='m', az_target_deg=None, az_center_deg=None, l1_strength=.05, l1_iters=300,
        flip_x=False, flip_y=False, **options)


def formed(source=None, **options):
    source = grid() if source is None else source
    bands, elapsed = isar.form_isar(source, reconstruction='accurate', window='Rectangular', retain_complex=True, **options)
    return source, bands, build_isar_manifest(source, params(source), bands, elapsed)


class GeometryAndQuality(unittest.TestCase):
    def test_stride_and_disjoint_physical_sectors(self):
        source = grid(az=np.arange(0, 10, .1))
        bands, _ = isar.form_isar(source, azimuth_indices=list(range(0, 100, 2)))
        self.assertEqual(len(bands), 1)
        np.testing.assert_allclose(bands[0]['az_values'], np.arange(0, 10, .2))
        self.assertEqual(angular_bands(range(6), [0, .1, .2, 10, 10.1, 10.2]), [[0, 1, 2], [3, 4, 5]])

    def test_sublook_angular_bound_and_no_loss_with_density_change(self):
        az = np.r_[np.linspace(0, 5, 101), np.arange(6, 61)]
        for overlap in (0., .5):
            chunks = angular_sublooks(range(len(az)), az, overlap=overlap)
            self.assertEqual(set(sum(chunks, [])), set(range(len(az))))
            self.assertTrue(all(0 < np.ptp(az[c]) <= 10 + 1e-10 for c in chunks))
        with self.assertRaisesRegex(ValueError, 'second sample'):
            angular_sublooks([0, 1], [0, 11])

    def test_outer_edges_include_half_cells_and_descending(self):
        self.assertEqual(axis_edges([-1., 0, 1]), (-1.5, 1.5))
        self.assertEqual(axis_edges([1., 0, -1]), (1.5, -1.5))

    def test_native_sampling_bound_equals_corner_calculation(self):
        az, freq = np.linspace(-10, 10, 129), np.linspace(9e9, 11e9, 129)
        p = plan_isar(az, freq, scene_half_extent_m=(2.1, 1.5), reconstruction='auto')
        theta = np.deg2rad(az)
        worst = max(np.max(abs(np.diff(4*np.pi/C0*freq[-1]*(x*np.sin(theta)+y*np.cos(theta)))))
                    for x in (-2.1, 2.1) for y in (-1.5, 1.5))
        self.assertAlmostEqual(p['max_native_azimuth_phase_step_rad'], worst, places=12)
        self.assertEqual(p['selected_reconstruction'], 'accurate')
        small = plan_isar(np.linspace(-1, 1, 65), freq, scene_half_extent_m=(.05, .05), reconstruction='auto')
        self.assertEqual(small['selected_reconstruction'], 'fft')
        with self.assertRaises(ValueError):
            plan_isar(az, freq, reconstruction='unknown')

    def test_explicit_coherent_composite_and_crop(self):
        source = grid(az=np.linspace(-11, 11, 101))
        coherent, _ = isar.form_isar(source, reconstruction='auto', aperture_mode='coherent',
            scene_half_extent_m=(.5, .6), retain_complex=True)
        self.assertIn('complex_image', coherent[0])
        self.assertLessEqual(abs(coherent[0]['x_range']).max(), .5)
        composite, _ = isar.form_isar(source, aperture_mode='composite', composite_side=64)
        self.assertEqual(composite[0]['magnitude'].shape, (64, 64))
        self.assertTrue(all(x['span_degrees'] <= 10 for x in composite[0]['composite_sublooks']))
        self.assertNotIn('complex_image', composite[0])

    def test_masked_origin_gain_all_windows_and_support_patterns(self):
        rng = np.random.default_rng(901)
        for kind in ('random', 'sector'):
            field = np.ones((65, 65), np.complex64)
            if kind == 'random':
                field[rng.random(field.shape) < .17] = np.nan
            else:
                field[20:31, 12:30] = np.nan
            source = grid(field=field)
            for window in ('Rectangular', 'Hanning', 'Hamming', 'Blackman', 'Blackman-Harris', 'Kaiser β=15'):
                with self.subTest(kind=kind, window=window):
                    bands, _ = isar.form_isar(source, reconstruction='accurate', window=window)
                    self.assertAlmostEqual(float(bands[0]['magnitude'].max()), 1., places=5)
                    self.assertLess(bands[0]['phase_coverage'], 1.)

    def test_measured_window_psf_distinguishes_resolution_and_sidelobes(self):
        source = grid()
        rect, _ = isar.form_isar(source, reconstruction='accurate', window='Rectangular')
        hann, _ = isar.form_isar(source, reconstruction='accurate', window='Hanning')
        for axis in ('cross_range', 'range'):
            a, b = rect[0]['psf'][axis], hann[0]['psf'][axis]
            self.assertGreater(b['power_fwhm'], a['power_fwhm'])
            self.assertLess(b['pslr_db'], a['pslr_db'] - 10.)

    def test_cancel_and_progress(self):
        stages = []
        isar.form_isar(grid(), progress=stages.append)
        self.assertTrue(stages)
        with self.assertRaises((ValueError, InterruptedError)):
            isar.form_isar(grid(), cancel_check=lambda: True)


class NativeOperatorTests(unittest.TestCase):
    def test_direct_sum_and_adjoint_for_offgrid_multiple_distributed_points(self):
        rng = np.random.default_rng(84)
        az = rng.uniform(-.3, .3, 53)
        freq = rng.uniform(9e9, 11e9, 53)
        for count in (1, 3, 73):
            points = rng.uniform(-1., 1., (count, 2))
            coefficients = rng.normal(size=count) + 1j*rng.normal(size=count)
            samples = rng.normal(size=53) + 1j*rng.normal(size=53)
            op = PolarPointOperator(az, freq, points, elevation_degrees=37, maximum_block_bytes=4800)
            matrix = np.array([[np.exp(-4j*np.pi/C0*f*np.cos(np.deg2rad(37))*(x*np.sin(t)+y*np.cos(t)))
                for x,y in points] for t,f in zip(az,freq)])
            np.testing.assert_allclose(op.forward(coefficients), matrix @ coefficients, atol=4e-12, rtol=1e-12)
            np.testing.assert_allclose(op.adjoint(samples), matrix.conj().T @ samples, atol=4e-12, rtol=1e-12)
            np.testing.assert_allclose(np.vdot(op.forward(coefficients), samples), np.vdot(coefficients, op.adjoint(samples)), atol=4e-11)

    def test_native_limits_and_cancellation(self):
        with self.assertRaisesRegex(ValueError, 'interaction budget'):
            PolarPointOperator([0, .1], [1e9, 2e9], [[0, 0], [1, 1]], maximum_interactions=3)
        op = PolarPointOperator([0], [1e9], [[0, 0]], cancel_check=lambda: True)
        with self.assertRaises(InterruptedError):
            op.forward([1])

    def test_phase_contract_restores_coefficients_and_native_noise_residual(self):
        az, freq = np.linspace(-3, 3, 7), np.linspace(9e9, 11e9, 9)
        contract = image_contract(np.deg2rad(az), freq, 0., 0., 1.)
        x, y = np.array([-.2, .3]), np.array([.1, .4])
        coefficients = np.array([[1+2j, 0], [0, -.3j]])
        origin = contract['spatial_frequency_origin_hz']
        stored = coefficients * np.exp(-4j*np.pi/C0*(origin[0]*x[:,None]+origin[1]*y[None,:]))
        np.testing.assert_allclose(physical_coefficients(stored, x, y, contract), coefficients, atol=1e-12)
        t, f = np.meshgrid(np.deg2rad(az), freq, indexing='ij')
        op = PolarPointOperator(t.ravel(), f.ravel(), [[x[0], y[0]], [x[1], y[1]]])
        native = op.forward([1+2j, -.3j]).reshape(t.shape)
        r = native_image_residual(stored, x, y, contract, az, freq, lambda i,j: native[i,j])
        self.assertLess(r['relative_complex_l2_residual'], 1e-12)
        rng = np.random.default_rng(831)
        noise = .01*(rng.normal(size=t.shape)+1j*rng.normal(size=t.shape))
        r = native_image_residual(stored, x, y, contract, az, freq, lambda i,j: (native+noise)[i,j], maximum_samples=20)
        self.assertTrue(r['sampled'])
        self.assertEqual(r['sample_count'], 20)
        self.assertGreater(r['relative_complex_l2_residual'], .001)

    def test_sparse_gridded_convergence_does_not_hide_native_model_mismatch(self):
        az, freq = np.linspace(-10, 10, 129), np.linspace(9e9, 11e9, 129)
        field = np.exp(-4j*np.pi/C0*freq[None,:]*1.5*np.cos(np.deg2rad(az[:,None])))
        bands, _ = isar.form_isar(grid(az, freq, field), reconstruction='sparse', l1_strength=.05, l1_iterations=1000, retain_complex=True)
        b = bands[0]
        self.assertTrue(b['sparse_converged'])
        self.assertLess(b['sparse_output_relative_residual_norm'], .1)
        self.assertGreater(b['native_residual']['relative_complex_l2_residual'], 1.)
        self.assertTrue(b['native_residual']['high_model_mismatch'])


class ReproducibilityTests(unittest.TestCase):
    def test_quality_artifact_roundtrip_and_loader_budget(self):
        _, bands, manifest = formed()
        with tempfile.TemporaryDirectory() as directory:
            path = save_isar_artifact(Path(directory)/'test', bands, manifest)
            m, b = load_isar_artifact(path)
            np.testing.assert_array_equal(b[0]['complex_image'], bands[0]['complex_image'])
            for key in ('image_contract', 'psf', 'accuracy_plan', 'memory_budget', 'az_gap_count', 'freq_largest_gap'):
                self.assertEqual(m['bands'][0][key], manifest['bands'][0][key])
            with self.assertRaisesRegex(ValueError, 'budget'):
                load_isar_artifact(path, maximum_working_bytes=100)

    def test_recipe_replays_physical_samples_across_units(self):
        source = grid()
        p = params(source)
        p['bands'] = [list(range(0, 65, 2))]
        recipe = recipe_from_params(p)
        with tempfile.TemporaryDirectory() as directory:
            path = save_recipe(Path(directory)/'recipe.json', recipe)
            recipe = load_recipe(path)
        other = grid(freq=source.frequencies/1e9, unit='GHz')
        args = recipe_arguments(other, recipe)
        self.assertEqual(args['azimuth_indices'], p['bands'][0])
        a, _ = isar.form_isar(source, azimuth_indices=p['bands'][0], reconstruction='accurate', window='Rectangular')
        b, _ = isar.form_isar(other, **args)
        np.testing.assert_allclose(a[0]['magnitude'], b[0]['magnitude'], atol=2e-7)
        recipe['options']['untrusted_code'] = 'anything'
        with self.assertRaises(ValueError):
            recipe_arguments(other, recipe)

    def test_difference_sign_levels_and_live_decimation(self):
        _, bands, m = formed()
        a, b = copy.deepcopy(bands[0]), copy.deepcopy(bands[0])
        a['magnitude'] = np.ones_like(a['magnitude'])*2
        b['magnitude'] = np.ones_like(b['magnitude'])
        result = compare_images(m, a, m, b)
        np.testing.assert_allclose(result['delta_db'], 20*np.log10(2), atol=1e-5)
        a['complex_image'] = np.full(a['complex_image'].shape, 2j)
        a['magnitude'] = a['magnitude'][::3, ::3]
        result = compare_images(m, a, m, b)
        self.assertAlmostEqual(result['statistics']['mean_delta_db'], 20*np.log10(2), places=5)

    def test_comparison_rejects_unknown_frame_and_different_recipe(self):
        _, bands, m = formed()
        bad = copy.deepcopy(bands[0])
        bad['image_contract']['range_basis'] = [1, 2]
        with self.assertRaisesRegex(ValueError, 'coordinate frames'):
            compare_images(m, bands[0], m, bad)
        bad = copy.deepcopy(m)
        bad['formation']['window'] = 'Hanning'
        with self.assertRaisesRegex(ValueError, 'window'):
            compare_images(m, bands[0], bad, bands[0])
        bad = copy.deepcopy(m)
        bad['source']['selected_azimuth_values_native'][0] += .01
        with self.assertRaisesRegex(ValueError, 'acquired angular samples'):
            compare_images(m, bands[0], bad, bands[0])
        with self.assertRaisesRegex(ValueError, 'contract'):
            compare_images(m, bands[0], m, {'image_contract': {}})

    def test_resampling_uses_linear_intensity_not_db(self):
        _, bands, m = formed()
        a, b = copy.deepcopy(bands[0]), copy.deepcopy(bands[0])
        a.update(x_range=np.array([0., .5, 1.]), y_range=np.array([0., .5, 1.]), magnitude=np.ones((3, 3)))
        b.update(x_range=np.array([0., 1.]), y_range=np.array([0., 1.]), magnitude=np.array([[1., 1.], [3., 3.]]))
        with self.assertRaisesRegex(ValueError, 'grids differ'):
            compare_images(m, a, m, b)
        r = compare_images(m, a, m, b, allow_resample=True)
        self.assertAlmostEqual(float(r['b_db'][1, 1]), 10*np.log10(5), places=5)


class UiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from PySide6.QtWidgets import QWidget, QListWidget, QAbstractItemView
        from GRIM_Backend.tests.test_plot_settings_ui import _SettingsBuilderWindow
        from GRIM_Backend.ui.dataset_actions import DatasetOpsMixin
        self.window = _SettingsBuilderWindow()
        self.context = self.window._build_plot_left_context(QWidget(self.window), 'isar')
        for name in self.context.__dataclass_fields__:
            setattr(self.window, name, getattr(self.context, name))
        self.window._fill_list = DatasetOpsMixin._fill_list.__get__(self.window)
        self.window.active_dataset = grid()
        for name in ('list_pol', 'list_az', 'list_elev', 'list_freq'):
            widget = QListWidget(self.window)
            widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
            setattr(self.window, name, widget)
        for widget, values in ((self.window.list_pol, ['VV']), (self.window.list_az, self.window.active_dataset.azimuths),
                               (self.window.list_elev, [0.]), (self.window.list_freq, self.window.active_dataset.frequencies)):
            self.window._fill_list(widget, values)

    def tearDown(self):
        self.context.settings_frame.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_frequency_limits_units_and_irrelevant_controls(self):
        from GRIM_Backend.ui.isar_controls import sync_frequency_controls, sync_reconstruction_controls
        for unit, factor in (('Hz', 1.), ('GHz', 1e9), ('MHz', 1e6)):
            sync_frequency_controls(self.context, grid(freq=np.array([9e9, 11e9])/factor, unit=unit))
            self.assertEqual(self.context.spin_isar_freq_min.value(), 9e9/factor)
            self.assertEqual(self.context.spin_isar_freq_max.value(), 11e9/factor)
            self.assertIn(unit, self.context.spin_isar_freq_max.suffix())
        sync_reconstruction_controls(self.context)
        self.assertFalse(self.context.spin_isar_l1_iters.isEnabled())
        self.context.combo_isar_recon.setCurrentIndex(2)
        sync_reconstruction_controls(self.context)
        self.assertTrue(self.context.spin_isar_l1_iters.isEnabled())
        self.assertFalse(self.context.combo_isar_window.isEnabled())

    def test_recipe_exact_selection_center_and_atomic_validation(self):
        from PySide6.QtCore import Qt
        from GRIM_Backend.ui.isar_workflow import apply_recipe_controls
        p = params(self.window.active_dataset)
        p['bands'] = [list(range(0, 65, 2))]
        p['az_center_deg'] = 0.
        arguments = recipe_arguments(self.window.active_dataset, recipe_from_params(p))
        apply_recipe_controls(self.window, arguments)
        selected = [item.data(Qt.UserRole+1) for item in self.window.list_az.selectedItems()]
        self.assertEqual(selected, p['bands'][0])
        self.assertTrue(self.window.chk_isar_aperture.isChecked())
        self.assertEqual(self.window.spin_isar_ap_width.value(), 10.)
        self.assertFalse(self.window.list_az.signalsBlocked())
        bad = dict(arguments, scene_half_extent_m=[1e-8, 1.])
        with self.assertRaisesRegex(ValueError, 'GUI precision'):
            apply_recipe_controls(self.window, bad)
        self.assertEqual([item.data(Qt.UserRole+1) for item in self.window.list_az.selectedItems()], selected)

    def test_reopened_artifact_dialog_and_locked_comparison_axes(self):
        from GRIM_Backend.ui.isar_workflow import IsarResultDialog
        _, bands, m = formed()
        dialog = IsarResultDialog((m, bands), (m, bands))
        self.assertIsNotNone(dialog.comparison)
        self.assertTrue(dialog.axes[0].get_shared_x_axes().joined(*dialog.axes[:2]))
        dialog.profiles.setChecked(True)
        self.assertEqual(len(dialog.figure.axes), 9)  # six plots and three color bars
        dialog.roi()
        self.assertIn('ROI', dialog.details.toPlainText())
        dialog.close()
        dialog.deleteLater()


if __name__ == '__main__':
    unittest.main()
