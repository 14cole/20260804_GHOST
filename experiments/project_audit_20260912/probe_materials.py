"""Independent real-widget material and cancellation edge probes."""
import importlib
from unittest.mock import patch
import numpy as np
import probe_workflows as p
from PySide6.QtWidgets import QMessageBox

p.RESULT_FILE = 'material-results.json'
f = p.f
out = p.AUDIT / 'material-artifacts'
out.mkdir(exist_ok=True)
paths = []
for name, eps in [('low', 2), ('high', 6)]:
    path = out / f'{name}.csv'
    path.write_text('frequency_hz,eps_real,eps_imag,mu_real,mu_imag\n'
                    f'9000000000,{eps},-0.1,1,0\n11000000000,{eps},-0.1,1,0\n', encoding='utf-8')
    paths.append(path)
material_io = importlib.import_module(type(f).__module__.rsplit('.', 1)[0] + '.io')

def mix():
    p.window.main_tabs.setCurrentIndex(2)
    f._select_mode(f._mode_labels.index('Material Mix'))
    f.mix_components = [f._coerce_mix_component({'file': str(path), 'parts': 1}) for path in paths]
    f._refresh_mix_components_list()
    f.mix_rule_var.set('linear')
    f.mix_freq_mode_var.set('Discrete list')
    f.mix_freq_list_var.set('9,10,11')
    f._preview_mix()
    assert f.mix_preview is not None, p.dialogs
    table, thickness, label = f._current_mix_material()
    np.testing.assert_allclose(table.eps_r, [4-.1j]*3, rtol=1e-12)
    np.testing.assert_allclose(table.mu_r, [1+0j]*3, rtol=1e-12)
    exported = out / 'equal-blend.csv'
    with patch.object(p.fmod.filedialog, 'asksaveasfilename', return_value=str(exported)):
        f._export_mix_material()
    loaded = material_io.read_material_table(exported)
    np.testing.assert_allclose(loaded.eps_r, table.eps_r)
    p.screenshot('material-mix')
    return {'expected_epsilon': '4-0.1j', 'frequency_ghz': list(table.freq_ghz), 'export': str(exported)}

def explorer():
    f._select_mode(f._mode_labels.index('Material Explorer'))
    explorer = f.material_explorer
    explorer.model.clear()
    malformed = out / 'malformed.csv'
    malformed.write_text('not,a,material\nx,y,z\n', encoding='utf-8')
    original = [path.read_bytes() for path in paths]
    added, duplicate, errors = explorer.add_requests([paths[0], paths[1], paths[0], malformed])
    assert (added, duplicate, len(errors)) == (2, 1, 1), (added, duplicate, errors)
    source = explorer.model.sources[0]
    sample = explorer.model.sample_at_frequency(source, 10)
    assert sample.eps_real == 2 and sample.eps_imag == -.1, sample
    assert explorer.model.sample_at_frequency(source, 8.999) is None
    assert explorer.model.sample_at_frequency(source, 11.001) is None
    assert explorer.model.raw_row_count() == 4
    assert original == [path.read_bytes() for path in paths]
    p.screenshot('material-explorer')
    return {'added': added, 'duplicates': duplicate, 'rejected': len(errors), 'interpolation_epsilon': '2-0.1j', 'outside_range': None}

def cancellation():
    solver = p.window.ghost_integration.workspace.solver_tab
    prior = {'samples': [{'prior': True}]}
    solver.last_result = prior
    with patch.object(QMessageBox, 'critical') as critical:
        for message in ('Solve canceled by user.', 'Solve cancelled by user.', ''):
            solver._pending_solve_context = {'uses_geometry_tab': True}
            solver._set_solving_state(True)
            solver.progress.setValue(35)
            solver._on_solver_canceled(message)
            assert not solver._is_solving
            assert solver._pending_solve_context is None
            assert solver.progress.value() == 0
            assert 'cancel' in solver.lbl_status.text().lower()
            assert solver.last_result is prior
            assert solver.btn_run.isEnabled()
            assert not solver.btn_cancel.isEnabled()
        critical.assert_not_called()
    return {'messages': 3, 'running_state_cleared': True, 'prior_result_preserved': True, 'critical_dialogs': 0}

with patch.object(QMessageBox, 'critical', side_effect=p.modal('critical')), patch.object(QMessageBox, 'warning', side_effect=p.modal('warning')), patch.object(QMessageBox, 'information', side_effect=p.modal('information')), patch.object(QMessageBox, 'question', side_effect=p.modal('question')):
    p.probe('Material Mix equal-volume numerical oracle and export', mix)
    p.probe('Material Explorer interpolation, bounds, duplicate and malformed batch', explorer)
    p.probe('GHOST cancellation state independent of message spelling', cancellation)
p.window.hide()
p.sys.exit(1 if p.callback_errors or any(r['status'] != 'PASS' for r in p.records) else 0)
