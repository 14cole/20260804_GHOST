import copy
import importlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import probe_workflows as p
from PySide6.QtWidgets import QFileDialog, QMessageBox
from GRIM_Backend.io.loaders import load_dataset

p.RESULT_FILE='extended-results.json'
f=p.f
out=p.AUDIT/'extended-artifacts'
out.mkdir(exist_ok=True)

def batch():
    p.window.main_tabs.setCurrentIndex(2)
    f._select_mode(f._mode_labels.index('IBC Batch'))
    f.ibc_batch_start_var.set('.1');f.ibc_batch_stop_var.set('.2');f.ibc_batch_step_var.set('.1')
    f.ibc_batch_output_dir_var.set(str(out));f.ibc_batch_prefix_var.set('batch')
    f._export_ibc_batch()
    p.wait(f.job_is_running)
    result=f.analysis_panels['IBC Batch'].result
    assert result is not None
    paths=list(out.glob('batch*.csv'))
    assert len(paths)==2,paths
    p.screenshot('batch-results')
    return {'exports':list(map(str,paths))}

def inverse():
    f._select_mode(f._mode_labels.index('Inverse Design'))
    f.layers[0].inv_t_min_in=.1;f.layers[0].inv_t_max_in=.2;f.layers[0].inv_t_accuracy_in=.1
    f._refresh_layers()
    for key,value in {'inv_target_start_var':'9','inv_target_stop_var':'11','inv_target_step_var':'1',
        'inv_angle_start_var':'0','inv_angle_stop_var':'0','inv_top_n_var':'2'}.items():getattr(f,key).set(value)
    f._run_inverse_design()
    p.wait(f.job_is_running)
    assert len(f.inverse_candidates)==2,f.status_var.get()
    assert f._inverse_checkpoint['next_index']==2
    assert f._inverse_checkpoint['total']==2
    assert not f._inverse_can_resume()
    p.screenshot('inverse-results')
    return {'candidates':len(f.inverse_candidates),'checkpoint_count':f._inverse_checkpoint['next_index']}

def tolerance():
    t=f.tolerance_workspace
    f._select_mode(f._mode_labels.index('Sensitivity & Yield'))
    f.layers[0].tolerances={'thickness':{'bound':5}}
    t.refresh_layers()
    t.restore_setup({'f_start':'9','f_stop':'11','f_step':'1','a_start':'0','a_stop':'30','a_step':'30',
        'points':'5','mode':'Sensitivity + statistical trials','samples':'16','seed':'42'})
    t.run()
    p.wait(f.job_is_running)
    assert t.result is not None
    assert t.result['trials']==16
    assert t.result['evaluated']==21
    records=[]
    for index in range(t.view.count()):
        t.view.setCurrentIndex(index);p.app.processEvents()
        assert t.figure.axes,t.view.currentText()
        records.append(t.view.currentText())
    p.screenshot('sensitivity-results')
    return {'trials':16,'evaluations':21,'views':records}

def ghost():
    source=out/'square.geo'
    source.write_text('Title: audit square\nSegment: body 2\nproperties: 2 0 0 0 0\n-0.02 -0.02 -0.02 0.02\n-0.02 0.02 0.02 0.02\n0.02 0.02 0.02 -0.02\n0.02 -0.02 -0.02 -0.02\nIBCS_Resistances:\nDielectrics:\n',encoding='utf-8')
    p.window.main_tabs.setCurrentIndex(3)
    g=p.window.ghost_integration.workspace
    with patch.object(QFileDialog,'getOpenFileName',return_value=(str(source),'')):
        assert g.geometry_tab.load_geo()
    solver=g.solver_tab
    g.setCurrentWidget(solver)
    solver.cmb_units.setCurrentText('meters')
    solver.edit_freq_list.setText('1')
    solver.edit_elev_list.setText('0,90')
    solver.edit_output.setText(str(out/'square.grim'))
    solver.chk_frequency_checkpoints.setChecked(False)
    solver._run_solver()
    p.wait(solver.job_is_running,120)
    assert solver.last_result is not None,solver.lbl_status.text()
    solver._export_last_result()
    p.wait(p.window._background_job_active)
    exports=list(out.glob('square*.grim'))
    assert exports,solver.lbl_status.text()
    for path in exports:
        grid=load_dataset(path)
        finite=grid.rcs_power[np.isfinite(grid.rcs_power)]
        assert finite.size>=2
        assert np.all(finite>=0)
    p.screenshot('ghost-solve')
    return {'exports':list(map(str,exports)),'status':solver.lbl_status.text(),'datasets_imported':p.window.table.rowCount()}

def assembly():
    from GRIM_Backend.assembly.values import FeatureAssemblyValues
    from GRIM_Backend.assembly.model import FeatureAssemblyFormModel
    panel=p.window.feature_assembly_panel
    from GRIM_Backend.datasets.grid import RcsGrid
    field=np.ones((3,1,2,3),complex)
    field[:,:,:,2]=0
    base=RcsGrid([0.,5.,10.],[0.],[9.,10.],['VV','HH','VH'],rcs=field,units=p.grid.units.copy())
    base.save(out/'full-pol-body.grim')
    model=FeatureAssemblyFormModel(FeatureAssemblyValues(base_grim=str(out/'full-pol-body.grim'),output_grim=str(out/'body-only.grim')))
    plan=model.prepare_preview(panel._service)
    dispatch=model.assemble_validated(panel._service)
    result=load_dataset(out/'body-only.grim')
    np.testing.assert_array_equal(result.rcs_power,base.rcs_power)
    np.testing.assert_array_equal(result.rcs_phase,base.rcs_phase)
    return {'output':str(out/'body-only.grim'),'shape':list(result.rcs_power.shape)}

def artifacts():
    p.load()
    p.isar()
    from GRIM_Backend.isar.artifact import save_isar_artifact,load_isar_artifact
    from GRIM_Backend.isar.comparison import hydrate_band,compare_images
    from GRIM_Backend.ui.isar_workflow import IsarResultDialog
    bands,manifest=p.window._last_isar_artifact
    path=save_isar_artifact(out/'point',bands,manifest)
    loaded=load_isar_artifact(path)
    np.testing.assert_allclose(loaded[1][0]['complex_image'],bands[0]['complex_image'])
    band=hydrate_band(loaded[1][0],loaded[0])
    doubled=dict(band,magnitude=band['magnitude']*2,complex_image=band['complex_image']*2)
    comparison=compare_images(loaded[0],doubled,loaded[0],band)
    bright=band['magnitude']>1e-5
    np.testing.assert_allclose(comparison['delta_db'][bright],20*np.log10(2),atol=2e-5)
    dialog=IsarResultDialog(loaded,loaded)
    assert np.max(abs(dialog.comparison['delta_db']))==0
    dialog.profiles.setChecked(True);dialog.roi();dialog.close();dialog.deleteLater()
    return {'artifact':str(path),'max_self_difference_db':0.,'scaled_difference_db':float(np.median(comparison['delta_db'][bright]))}

with patch.object(QMessageBox,'critical',side_effect=p.modal('critical')),patch.object(QMessageBox,'warning',side_effect=p.modal('warning')),patch.object(QMessageBox,'information',side_effect=p.modal('information')),patch.object(QMessageBox,'question',side_effect=p.modal('question')):
    for name,fn in [('FREDDY batch export',batch),('FREDDY exhaustive inverse grid',inverse),
                    ('FREDDY sensitivity and statistical trial views',tolerance),
                    ('ISAR numerical export reopen comparison',artifacts),
                    ('GHOST geometry to solve to export to GRIM',ghost),
                    ('Assembly body-only identity via real service',assembly)]:p.probe(name,fn)
p.window.hide()
p.sys.exit(1 if p.callback_errors or any(r['status'] != 'PASS' for r in p.records) else 0)
