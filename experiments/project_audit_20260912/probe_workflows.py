"""Independent integration probes: real widgets, workers, and numerical code."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import json
import copy
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QSettings, Qt, QItemSelectionModel, QPoint, QRect
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox, QAbstractButton
from GRIM_Backend.ui.app import GrimCutWindow
from GRIM_Backend.datasets.grid import RcsGrid

(AUDIT/'screens').mkdir(exist_ok=True)
app = QApplication([])
font = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0], 9))
callback_errors, records, dialogs = [], [], []
RESULT_FILE='workflow-results.json'
def exception_hook(typ, exc, tb):
    message = ''.join(traceback.format_exception(typ, exc, tb))
    callback_errors.append(message)
    print(message, file=sys.stderr, flush=True)
sys.excepthook = exception_hook
window = GrimCutWindow(settings=QSettings(str(AUDIT/'probe.ini'), QSettings.IniFormat))
window.resize(1280, 720)
window.show()

def wait(predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert not predicate(), 'Worker did not finish before timeout'

def probe(name, action):
    before = len(callback_errors)
    start = time.monotonic()
    try:
        details = action()
        app.processEvents()
        assert len(callback_errors) == before, callback_errors[before:]
        result = dict(name=name, status='PASS', details=details)
    except Exception:
        result = dict(name=name, status='FAIL', error=traceback.format_exc())
    result['seconds'] = round(time.monotonic()-start, 3)
    records.append(result)
    (AUDIT/RESULT_FILE).write_text(json.dumps(dict(results=records, dialogs=dialogs, callback_errors=callback_errors), indent=2, default=str), encoding='utf-8')
    print(json.dumps(result, default=str), flush=True)

def select(*rows):
    table = window.table
    table.clearSelection()
    for row in rows:
        table.setCurrentCell(row, 0, QItemSelectionModel.NoUpdate)
        table.selectionModel().select(table.model().index(row,0), QItemSelectionModel.Select|QItemSelectionModel.Rows)
        app.processEvents()

def parameters(az='all', freq='all', el=0, pol=0):
    for widget, value in ((window.list_az,az),(window.list_freq,freq),(window.list_elev,el),(window.list_pol,pol)):
        widget.clearSelection()
        if value == 'all': widget.selectAll()
        else:
            assert widget.item(value) is not None, (widget.objectName(), value, widget.count())
            widget.item(value).setSelected(True)
    app.processEvents()

def screenshot(name):
    app.processEvents()
    window.grab().save(str(AUDIT/'screens'/('workflow-'+name+'.png')))

az = np.linspace(-5,5,33)
freq = np.linspace(9,11,33)
phase = 4*np.pi*freq[None,None,:,None]*1e9/299792458*.15
field = np.broadcast_to(np.exp(1j*phase), (33,3,33,2)).copy()
grid = RcsGrid(az, [-1.,0.,1.],freq,['VV','HH'],rcs=field, units={
    'azimuth':'deg','elevation':'deg','frequency':'GHz','angular_coordinate_system':'conic',
    'rcs_log_unit':'dBsm','rcs_linear_quantity':'sigma_3d'})
grid.save(str(AUDIT/'known-a.grim'))
other = copy.deepcopy(grid)
other.rcs_power *= 4
other.save(str(AUDIT/'known-b.grim'))

def load():
    window._handle_files_dropped([str(AUDIT/'known-a.grim'), str(AUDIT/'known-b.grim')])
    wait(window._background_job_active)
    assert window.table.rowCount() == 2, window.status.currentMessage()
    select(0)
    return {'loaded_rows':2,'shape':list(window.active_dataset.rcs_power.shape)}

def plot(mode):
    window.main_tabs.setCurrentIndex(0)
    select(0,1)
    parameters(freq=0 if mode.startswith('azimuth') or mode in ('elevation_sweep','compare') else 'all',
               az=0 if mode in ('frequency','elevation_sweep') else 'all',
               el='all' if mode=='elevation_sweep' else 1)
    getattr(window, '_plot_'+mode)()
    app.processEvents()
    assert not any(word in window.status.currentMessage().lower() for word in ('requires', 'select exactly', 'failed')), window.status.currentMessage()
    lines = [line for ax in window.plot_figure.axes for line in ax.lines if len(line.get_ydata())]
    assert lines or any(ax.collections or ax.images for ax in window.plot_figure.axes), window.status.currentMessage()
    if mode in ('azimuth_rect','frequency','elevation_sweep'):
        values = [np.asarray(line.get_ydata(), dtype=float) for line in lines]
        assert any(np.allclose(v,0,atol=1e-10) for v in values), values
        assert any(np.allclose(v,10*np.log10(4),atol=1e-10) for v in values), values
    if mode=='delta_map':
        result=window.plot_ax._grim_delta_map
        np.testing.assert_allclose(result.delta_db,-10*np.log10(4),atol=1e-10)
    screenshot(mode)
    return {'axes':len(window.plot_figure.axes),'lines':len(lines),'status':window.status.currentMessage()}

def isar():
    window.main_tabs.setCurrentIndex(1)
    select(0)
    parameters(el=1)
    window.combo_isar_units.setCurrentText('m')
    window.combo_isar_recon.setCurrentIndex(0)
    window._on_explicit_isar_plot_clicked()
    wait(lambda: getattr(window,'_isar_busy',False), 90)
    artifact = getattr(window,'_last_isar_artifact',None)
    assert artifact is not None, window.status.currentMessage()
    from GRIM_Backend.isar.artifact import save_isar_artifact, load_isar_artifact
    bands, manifest = artifact
    result = bands[0]
    assert np.isfinite(result['magnitude']).all()
    ix, iy = np.unravel_index(np.argmax(result['magnitude']), result['magnitude'].shape)
    cross = float(result['x_range'][ix])
    down = float(result['y_range'][iy])
    dx = float(np.max(np.diff(result['x_range'])))
    dy = float(np.max(np.diff(result['y_range'])))
    assert abs(cross) <= dx, (cross, dx)
    assert abs(down + .15) <= dy, (down, dy)
    screenshot('isar')
    return {'shape':list(result['magnitude'].shape),'peak_cross_range_m':cross,
            'peak_down_range_m':down,'expected_down_range_m':-.15,
            'status':window.status.currentMessage()}

def header():
    window.main_tabs.setCurrentIndex(1)
    window.resize(1280,720)
    app.processEvents()
    clipped=[]
    for button in window.findChildren(QAbstractButton):
        if button.isVisible() and 'Settings' in button.text():
            bounds=QRect(button.mapTo(window,QPoint()),button.size())
            if not window.rect().contains(bounds): clipped.append({'text':button.text(),'bounds':bounds.getRect(),'window':window.rect().getRect()})
    assert not clipped, clipped
    return {'window_size':[window.width(),window.height()],'clipped':clipped}

def ppt():
    window.main_tabs.setCurrentIndex(5)
    p=window.ppt_workspace
    p._use_main_selection()
    assert p.build_preview(), p.lbl_status.text() if hasattr(p,'lbl_status') else 'Preview returned false'
    screenshot('ppt')
    return 'Rendered actual report preview from loaded dataset'

def replay():
    script=window.python_recorder.script
    (AUDIT/'recorded-workflow.py').write_text(script,encoding='utf-8')
    import subprocess
    proc=subprocess.run([sys.executable,str(AUDIT/'recorded-workflow.py')],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    return {'script_lines':len(script.splitlines()),'returncode':proc.returncode,'stderr':proc.stderr}

f=window.freddy_integration.workspace
fmod=sys.modules[type(f).__module__]
LayerConfig=fmod.LayerConfig
f.layers=[LayerConfig(.12,False,'','',0,material_source='constant',constant_eps_real=4.,constant_eps_imag=-.4)]
f._refresh_layers()
for key,val in {'f_start_var':'9','f_stop_var':'11','f_step_var':'1',
    'output_var':str(AUDIT/'impedance.csv'),'angle_f_start_var':'9','angle_f_stop_var':'11',
    'angle_f_step_var':'1','angle_start_var':'0','angle_stop_var':'60','angle_step_var':'30',
    'angle_output_var':str(AUDIT/'off-angle.csv'),'thk_f_start_var':'9','thk_f_stop_var':'11',
    'thk_f_step_var':'1','thk_start_var':'.1','thk_stop_var':'.2','thk_step_var':'.1',
    'thk_output_var':str(AUDIT/'thickness.csv')}.items(): getattr(f,key).set(val)

def freddy(mode,method):
    window.main_tabs.setCurrentIndex(2)
    f._select_mode(f._mode_labels.index(mode))
    getattr(f,method)()
    wait(f.job_is_running,90)
    result=f.analysis_panels[mode].result
    assert result is not None, f.status_var.get()
    screenshot('freddy-'+mode.lower().replace(' ','-'))
    return {'status':f.status_var.get(),'result_type':type(result).__name__}

def modal(kind):
    def called(*args,**kw):
        dialogs.append({'kind':kind,'args':[str(a) for a in args[1:]]})
        return QMessageBox.Yes if kind == 'question' else QMessageBox.Ok
    return called

def run_probes():
    with patch.object(QMessageBox,'critical',side_effect=modal('critical')), patch.object(QMessageBox,'warning',side_effect=modal('warning')), patch.object(QMessageBox,'information',side_effect=modal('information')), patch.object(QMessageBox,'question',side_effect=modal('question')):
        probe('Load native datasets through real background loader',load)
        for mode in ('azimuth_rect','azimuth_polar','frequency','elevation_sweep','waterfall','compare','delta_map'):
            probe('Plot '+mode,lambda mode=mode:plot(mode))
        probe('ISAR real background formation',isar)
        probe('ISAR settings accessible at 1280x720',header)
        probe('PPT real preview',ppt)
        probe('Python recorded workflow replay',replay)
        for mode,method in [('Impedance','_compute_impedance'),('Off Angle','_compute_off_angle'),('Thickness','_compute_thickness')]:
            probe('FREDDY '+mode,lambda mode=mode,method=method:freddy(mode,method))
    window.hide()

if __name__ == '__main__':
    run_probes()
    sys.exit(1 if callback_errors or any(r['status'] != 'PASS' for r in records) else 0)
