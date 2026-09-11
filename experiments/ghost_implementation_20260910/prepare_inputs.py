from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'ghost_ram_time_20260910'))
from common import snapshot, json, hashlib
from material_probe import KINDS
inputs = {kind:snapshot(kind, 384) for kind in KINDS}
inputs['airfoil'] = snapshot('airfoil')
inputs['ibc4096'] = snapshot('ibc', 4096)
(HERE/'inputs.json').write_text(json.dumps(inputs))
(HERE/'baseline-hashes.json').write_text(json.dumps({
    p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'baseline_backend').glob('*.py')}, indent=2))
