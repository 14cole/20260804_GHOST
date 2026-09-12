import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import json
from pathlib import Path
import re
import sys
import traceback

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QTabWidget, QAbstractButton, QComboBox, QLabel
from GRIM_Backend.ui.app import GrimCutWindow

app = QApplication([])
font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9))
errors = []
sys.excepthook = lambda typ, exc, tb: errors.append(''.join(traceback.format_exception(typ, exc, tb)))
settings = QSettings(str(AUDIT / 'ui-test.ini'), QSettings.IniFormat)
window = GrimCutWindow(settings=settings)
window.resize(1280, 720)
window.show()
captures = AUDIT / 'screens'
captures.mkdir(exist_ok=True)
records = []

def capture(name):
    window.resize(1280, 720)
    app.processEvents()
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    window.grab().save(str(captures / (slug + '.png')))
    buttons = [dict(text=w.text(), enabled=w.isEnabled(), tooltip=w.toolTip())
               for w in window.findChildren(QAbstractButton) if w.isVisible() and w.text()]
    combos = [dict(name=w.objectName(), values=[w.itemText(i) for i in range(w.count())], current=w.currentText())
              for w in window.findChildren(QComboBox) if w.isVisible()]
    records.append(dict(page=name, size=[window.width(), window.height()], buttons=buttons, combos=combos))
    print(name, flush=True)

def subtabs(parent, prefix):
    for tabs in parent.findChildren(QTabWidget):
        if tabs is window.main_tabs or not tabs.isVisible():
            continue
        old = tabs.currentIndex()
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            capture(prefix + ' - ' + tabs.tabText(index))
        tabs.setCurrentIndex(old)

for index in range(window.main_tabs.count()):
    window.main_tabs.setCurrentIndex(index)
    name = window.main_tabs.tabText(index)
    capture(name)
    if name == 'FREDDY':
        freddy = window.freddy_integration.workspace
        for mode, label in enumerate(freddy._mode_labels):
            freddy._select_mode(mode)
            capture(name + ' - ' + label)
            subtabs(freddy, name + ' - ' + label)
    else:
        subtabs(window.main_tabs.widget(index), name)

(AUDIT / 'ui-inventory.json').write_text(json.dumps(dict(
    ghost_error=window.ghost_integration.load_error,
    freddy_error=window.freddy_integration.load_error,
    pages=records, errors=errors), indent=2), encoding='utf-8')
window.hide()
print('COMPLETE', len(records), 'pages; callback errors:', len(errors), flush=True)
