"""Behavior checks for workflow shortcuts, including rejected/unchanged imports."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from matplotlib.figure import Figure
from GRIM_Backend.reports.workspace import PptWorkspace, DatasetCatalogEntry
from GRIM_Backend.reports.workflow import write_report_recipe, read_report_recipe, inspect_template, current_plot_report_setup
from GRIM_Backend.assembly.workflow import suggest_response_mapping
from GRIM_Backend.assembly.panel import FeatureAssemblyPanel, read_feature_assembly_recipe
from test_ppt_workspace import _grid
from GRIM_Backend.integrations.ghost import load_ghost_module


class WorkflowUpdatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.widgets=[]

    def tearDown(self):
        for widget in self.widgets:
            if hasattr(widget,'dispose'): widget.dispose()
            widget.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def ppt(self):
        ui=PptWorkspace()
        self.widgets.append(ui)
        ui.set_dataset_catalog([DatasetCatalogEntry('a','Body',_grid(),str(self.root/'body.grim')),
                                DatasetCatalogEntry('b','Total',_grid(scale=2),str(self.root/'total.grim'))])
        ui.select_frequencies([1.,3.])
        return ui

    def test_report_roundtrip_restores_order_axes_template_and_styles(self):
        ui=self.ppt()
        state=ui.capture_report_setup()
        state['order']=['b','a']
        state['selected']=['b','a']
        state['combos']['x_scale_mode_combo']='fixed'
        state['spins'].update(x_min_spin=0.,x_max_spin=270.,x_step_spin=45.)
        state['line_widths']={'b':2.5}
        state['line_styles']={'b':'--'}
        state['line_colors']={'b':'#ff0000'}
        state['texts']['deck_title_edit']='Variant report'
        path=self.root/'review.report.json'
        write_report_recipe(path,state)
        ui.apply_report_setup(read_report_recipe(path))
        self.assertEqual(ui.capture_report_setup(),state)
        plan=ui._build_plan()
        first=plan.slides[0].plots[0].plot.series[0]
        self.assertEqual((first.line_width,first.line_style,first.color),(2.5,'--','#ff0000'))

    def test_missing_frequency_or_channel_rejects_and_restores_controls(self):
        ui=self.ppt()
        before=ui.capture_report_setup()
        for section,key,value,message in [('frequencies',None,[99.],'frequencies'),('combos','polarization_combo','HV','polarization')]:
            recipe=copy.deepcopy(before)
            if key: recipe[section][key]=value
            else: recipe[section]=value
            recipe['texts']['deck_title_edit']='Should roll back'
            with self.assertRaisesRegex(ValueError,message): ui.apply_report_setup(recipe)
            self.assertEqual(ui.capture_report_setup(),before)

    def test_recipe_reuses_new_dataset_with_explicit_selection(self):
        ui=self.ppt()
        ui.select_dataset_ids(['b'])
        state=ui.capture_report_setup()
        ui.set_dataset_catalog([DatasetCatalogEntry('c','New variant',_grid(scale=3))])
        with self.assertRaisesRegex(ValueError,'missing or ambiguous'): ui.apply_report_setup(state)
        ui.apply_report_setup(state,current_datasets=True)
        self.assertEqual(ui.selected_dataset_ids(),('c',))
        self.assertEqual(ui.selected_frequencies(),(1.,3.))

    def test_angular_report_does_not_require_an_unused_fixed_azimuth_cut(self):
        ui=self.ppt()
        state=ui.capture_report_setup()
        ui.set_dataset_catalog([DatasetCatalogEntry('c','Different sweep',_grid(azimuths=(10.,20.,30.)))])
        ui.apply_report_setup(state,current_datasets=True)
        self.assertEqual(ui.selected_frequencies(),(1.,3.))
        self.assertEqual(ui._build_plan().slides[0].plots[0].plot.series[0].x,(10.,20.,30.))

    def test_native_angle_units_convert_on_recipe_reuse(self):
        import numpy as np
        ui=self.ppt()
        ui.azimuth_combo.setCurrentIndex(ui.azimuth_combo.findData(90.))
        state=ui.capture_report_setup()
        ui.set_dataset_catalog([DatasetCatalogEntry('c','Radians',_grid(azimuths=np.deg2rad([0,90,180,270]),angle_unit='rad'))])
        ui.apply_report_setup(state,current_datasets=True)
        self.assertAlmostEqual(ui.azimuth_combo.currentData(),np.pi/2)

    def test_main_plot_transfer_uses_frozen_selection_and_visible_axes(self):
        ui=self.ppt()
        fig=Figure()
        ax=fig.add_subplot()
        line,=ax.plot([0,90,180],[0,2,1],color='#ff0000',linewidth=2,linestyle='--')
        line._grim_dataset_key='b'
        ax.set_xlim(0,180); ax.set_ylim(-10,10)
        params=dict(azimuths=[0,90,180],elevations=[0.],frequencies=[3.],polarization='HH',reference_index=0,phase=False,scale='dbsm',show_legend=True)
        spec=('supported',(SimpleNamespace(dataset_id='b'),),('Total',),'azimuth_rect',params)
        window=SimpleNamespace(_active_plot_tab='isar',_plot_contexts={'plotting':SimpleNamespace(last_python_plot_spec=spec,plot_ax=ax)},ppt_workspace=ui)
        state=current_plot_report_setup(window)
        ui.apply_report_setup(state)
        self.assertEqual(ui.selected_dataset_ids(),('b',))
        self.assertEqual(ui.selected_frequencies(),(3.,))
        self.assertEqual(ui.x_max_spin.value(),180.)
        self.assertEqual(ui._series_line_colors,{'b':'#ff0000'})
        params['phase']=True
        with self.assertRaisesRegex(ValueError,'magnitude'): current_plot_report_setup(window)

    def test_template_checks_and_open_exported_path(self):
        ui=self.ppt()
        template=self.root/'template.potx'
        ns='http://schemas.openxmlformats.org/presentationml/2006/main'
        with zipfile.ZipFile(template,'w') as archive:
            archive.writestr('ppt/presentation.xml',f'<p:presentation xmlns:p="{ns}"><p:sldSz cx="1600" cy="900"/></p:presentation>')
            archive.writestr('ppt/slideLayouts/slideLayout1.xml',f'<p:sldLayout xmlns:p="{ns}"><p:cSld name="Six plots"/></p:sldLayout>')
        self.assertIn('widescreen',inspect_template(template,['Six plots']))
        with self.assertRaisesRegex(ValueError,'unavailable'): inspect_template(template,['Absent'])
        ui._export_succeeded(str(template))
        with mock.patch('PySide6.QtGui.QDesktopServices.openUrl',return_value=True) as opened:
            ui._open_exported_presentation()
        self.assertEqual(opened.call_args.args[0].toLocalFile(),str(template).replace('\\','/'))

    def test_library_suggestions_do_not_guess_ambiguous_or_overwrite_existing(self):
        (self.root/'sub').mkdir()
        for path in [self.root/'fastener.grim',self.root/'seam.grim',self.root/'sub'/'seam.grim']: path.touch()
        unique,ambiguous,missing=suggest_response_mapping(['fastener','seam','missing','kept'],self.root,{'kept':'old.grim'})
        self.assertEqual(set(unique),{'fastener'})
        self.assertEqual(len(ambiguous['seam']),2)
        self.assertEqual(missing,['missing'])

    def test_variant_separates_recipe_output_and_preserves_membership(self):
        ui=FeatureAssemblyPanel()
        self.widgets.append(ui)
        ui.model.values.excluded_point_placement_ids={'P2'}
        source=self.root/'baseline.assembly.json'
        ui.save_recipe_path(source)
        before=source.read_bytes()
        destination=self.root/'without_seam.assembly.json'
        ui.create_variant_path(destination,'Without seam')
        self.assertEqual(source.read_bytes(),before)
        saved=read_feature_assembly_recipe(destination)
        self.assertEqual(saved.variant,'Without seam')
        self.assertEqual(saved.values.excluded_point_placement_ids,{'P2'})
        self.assertEqual(Path(saved.values.output_grim).name,'without_seam_total.grim')
        self.assertFalse(ui._validated_plan_current)
        with self.assertRaisesRegex(ValueError,'new recipe'): ui.create_variant_path(source,'Again')

    def test_readiness_links_open_the_relevant_step_and_hidden_controls(self):
        from PySide6.QtWidgets import QTreeWidgetItem
        ui=FeatureAssemblyPanel(); self.widgets.append(ui)
        ui._go_to_requirement(QTreeWidgetItem(['Clean-body response']))
        self.assertEqual(ui.workflow_tabs.currentIndex(),0)
        ui._go_to_requirement(QTreeWidgetItem(['Surface mesh units']))
        self.assertTrue(ui.body_geometry_section.header.isChecked())
        ui._go_to_requirement(QTreeWidgetItem(['Advanced settings valid']))
        self.assertEqual(ui.workflow_tabs.currentIndex(),3)
        self.assertTrue(ui.advanced_section.header.isChecked())

    def test_shared_2d_setup_roundtrip_and_incompatible_hpc_rejected_atomically(self):
        module=load_ghost_module('ghost_backend.runs.setup')
        SolverTab=load_ghost_module('ghost_backend.ui.solver').SolverTab
        from GRIM_Backend.runs.workspace import RunsWorkspace
        local=SolverTab()
        cluster=RunsWorkspace(settings=QSettings(str(self.root/'settings.ini'),QSettings.IniFormat))
        self.widgets.extend([local,cluster])
        local.edit_freq_list.setText('1, 2.75')
        local.edit_elev_list.setText('-10, 0, 90')
        local.cmb_units.setCurrentText('meters')
        local.chk_mesh_certification.setChecked(True)
        local.cmb_accuracy_target.setCurrentIndex(local.cmb_accuracy_target.findData('tight'))
        local.cmb_lu_precision.setCurrentIndex(local.cmb_lu_precision.findData('double'))
        value=local._capture_run_setup()
        path=self.root/'target.run.json'
        module.save_setup(path,value)
        cluster._apply_saved_run_setup(module.read_setup(path))
        self.assertEqual(cluster._capture_run_setup(),value)
        other=copy.deepcopy(value)
        other['scattering']='bistatic'
        other['observation_angles_deg']=[0.,90.]
        other['solver_method']='direct'
        other['execution_options']['factorization']='dense'
        with self.assertRaisesRegex(ValueError,'monostatic'): cluster._apply_saved_run_setup(other)
        self.assertEqual(cluster._capture_run_setup(),value)
        local._apply_saved_run_setup(other)
        self.assertEqual(local._capture_run_setup(),other)
        bad=copy.deepcopy(value); bad['frequencies_ghz']=[float('nan')]
        with self.assertRaisesRegex(ValueError,'finite'): local._apply_saved_run_setup(bad)
        self.assertEqual(local._capture_run_setup(),other)

    def test_physical_dimensions_and_preflight_material_errors(self):
        module=load_ghost_module('ghost_backend.runs.setup')
        SolverTab=load_ghost_module('ghost_backend.ui.solver').SolverTab
        ui=SolverTab(); self.widgets.append(ui)
        geo=Path(__file__).resolve().parents[2]/'tools/GHOST/ghost_backend/validation/material_examples/01_pec.geo'
        if not geo.is_file():
            geo=next((Path(__file__).resolve().parents[2]/'tools/GHOST/ghost_backend/validation/material_examples').glob('*pec*.geo'))
        ui.edit_geo_path.setText(str(geo))
        snapshot,_,base=ui._load_geometry_for_solver()
        summary=ui._run_setup_summary(snapshot,base,ui._capture_run_setup())
        self.assertIn('checks passed',summary)
        self.assertIn('X span',module.geometry_dimensions(snapshot,'inches'))
        broken=copy.deepcopy(snapshot)
        broken['ibcs']=[['1','missing.csv']]
        with self.assertRaises(Exception): ui._run_setup_summary(broken,base,ui._capture_run_setup())
        ui.edit_freq_list.setText('1')
        ui.cmb_units.setCurrentText('meters')
        for example in geo.parent.glob('*.geo'):
            with self.subTest(material=example.name):
                ui.edit_geo_path.setText(str(example))
                snapshot,_,base=ui._load_geometry_for_solver()
                summary=ui._run_setup_summary(snapshot,base,ui._capture_run_setup())
                self.assertIn('VV + HH',summary)

    def test_background_preflight_finishes_without_solving_or_publishing(self):
        SolverTab=load_ghost_module('ghost_backend.ui.solver').SolverTab
        ui=SolverTab(); self.widgets.append(ui)
        geo=Path(__file__).resolve().parents[2]/'tools/GHOST/ghost_backend/validation/material_examples/type1_thin_dielectric.geo'
        ui.edit_geo_path.setText(str(geo))
        ui.edit_freq_list.setText('1')
        ui.cmb_units.setCurrentText('meters')
        published=[]
        ui.files_exported.connect(lambda *args:published.append(args))
        with mock.patch('ghost_backend.ui.solver._SolveWorker._run_2d') as solve:
            ui._check_run_setup()
            self.assertTrue(ui.job_is_running())
            deadline=time.monotonic()+5
            while ui.job_is_running() and time.monotonic()<deadline:
                self.app.processEvents(); time.sleep(.005)
            self.app.processEvents()
            solve.assert_not_called()
        self.assertFalse(ui.job_is_running())
        self.assertIn('VV + HH',ui.run_setup_notice.text())
        self.assertIsNone(ui.last_result)
        self.assertEqual(published,[])
        self.assertTrue(ui.btn_run.isEnabled())

    def test_ppt_template_probe_closes_owned_copy_on_layout_error(self):
        from GRIM_Backend.reports.report import PowerPointComBridge
        presentation=mock.Mock()
        app=mock.Mock()
        bridge=PowerPointComBridge(application_factory=lambda:app)
        with mock.patch.object(bridge,'_open_presentation',return_value=presentation), mock.patch.object(bridge,'_find_custom_layout',side_effect=ValueError('missing master')):
            with self.assertRaisesRegex(ValueError,'missing master'):
                bridge.preflight_template(self.root/'template.pptx',['Master :: Layout'])
        presentation.Close.assert_called_once()
        app.Quit.assert_not_called()


if __name__=='__main__': unittest.main()
