import os
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'tools/GHOST/Backend'))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from ghost_backend.ui.execution import ExecutionOptionsWidget
from ghost_backend.ui.app import GhostWorkspace
app = QApplication([])
font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
widget = ExecutionOptionsWidget()
widget.set_value(dict(factorization='compressed', compressed_storage_mib=8192,
                      assembly_threads=4, blas_threads=2))
widget.resize(710, widget.sizeHint().height())
widget.show()
app.processEvents()
widget.grab().save(str(Path(__file__).parent/'execution-controls.png'))
workspace = GhostWorkspace()
workspace.resize(1440, 960)
workspace.solver_tab.execution_options_widget.set_value(widget.value())
workspace.setCurrentWidget(workspace.solver_tab)
workspace.show()
app.processEvents()
workspace.solver_tab.controls_scroll.ensureWidgetVisible(workspace.solver_tab.execution_options_widget)
app.processEvents()
workspace.grab().save(str(Path(__file__).parent/'ghost-workspace.png'))
workspace.close()
widget.close()
