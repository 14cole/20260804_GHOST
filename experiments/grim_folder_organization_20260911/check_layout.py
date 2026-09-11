"""Check categorized files and source-checkout startup without editable imports."""
from pathlib import Path
import json
import re
import subprocess
import sys
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parents[2]
GRIM = ROOT / 'GRIM_Backend'
expected = {'run_gui.py', 'run_diagnostics.py', 'run_headless.py', 'run_image_imprinter.py'}
assert {p.name for p in GRIM.iterdir() if p.is_file()} == expected
assert not (ROOT / 'GRIM_Revised_2').exists()
assert not (GRIM / 'grim_backend').exists()
metadata = json.loads((Path(__file__).parent / 'moves.json').read_text())
assert all((GRIM / path).exists() for path in metadata['files'].values())
print('Root layout: four run scripts, categorized folders, no nested backend.')

dependencies = sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')})
with tempfile.TemporaryDirectory() as temporary:
    code = '''
import json, runpy, sys
from pathlib import Path
sys.path[:0] = json.loads(sys.argv[2])
namespace = runpy.run_path(sys.argv[1], run_name='bootstrap_check')
assert callable(namespace['main'])
import GRIM_Backend.datasets.grid
assert str(Path(GRIM_Backend.datasets.grid.__file__).resolve()) == sys.argv[3]
'''
    scripts = sorted(GRIM.glob('run_*.py')) + sorted(p for p in (GRIM/'examples').glob('*.py') if not p.name.startswith('_'))
    for path in scripts:
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(path), json.dumps(dependencies), str(GRIM/'datasets/grid.py')],
                                cwd=temporary, text=True, capture_output=True, timeout=45)
        assert result.returncode == 0, (path, result.stderr)
        print('Isolated bootstrap:', path.relative_to(ROOT))

for path in [ROOT/'README.md', ROOT/'ARCHITECTURE.md', *(GRIM/'docs').glob('*.md'), *(GRIM/'examples').glob('*.md')]:
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        if ':' in target or target.startswith('#'):
            continue
        target = target.split('#')[0]
        assert (path.parent/target).exists(), (path,target)
print('Documentation links: valid.')
