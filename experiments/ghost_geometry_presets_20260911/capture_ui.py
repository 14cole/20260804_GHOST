"""Capture collapsed and expanded solver forms at desktop and compact sizes."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'tools/GHOST/Backend'))
sys.path.insert(0, str(root / 'GRIM_Revised_2'))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from ghost_backend.ui.app import GhostWorkspace
from runs_workspace import RunsWorkspace
from test_runs_workspace import _MemorySettings

app = QApplication([])
font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
workspace = GhostWorkspace()
workspace.setCurrentWidget(workspace.solver_tab)
workspace.resize(1280, 720)
workspace.show()
app.processEvents()
destination = Path(__file__).parent
workspace.grab().save(str(destination / 'ghost-main.png'))
workspace.solver_tab.btn_advanced_settings.setChecked(True)
app.processEvents()
workspace.solver_tab.controls_scroll.ensureWidgetVisible(workspace.solver_tab.execution_options_widget)
app.processEvents()
workspace.grab().save(str(destination / 'ghost-advanced.png'))
workspace.solver_tab.btn_advanced_settings.setChecked(False)
workspace.solver_tab.cmb_elev_mode.setCurrentIndex(1)
workspace.solver_tab.edit_elev_stop.setText('360')
workspace.solver_tab.edit_elev_step.setText('1')
workspace.resize(1200, 680)
app.processEvents()
workspace.grab().save(str(destination / 'ghost-sweep-compact.png'))
runs = RunsWorkspace(settings=_MemorySettings())
runs.resize(1280, 720)
runs.show()
app.processEvents()
runs.controls_scroll.ensureWidgetVisible(runs.geometry_preset_combo)
app.processEvents()
runs.grab().save(str(destination / 'runs-main.png'))
workspace.close()
runs.close()
