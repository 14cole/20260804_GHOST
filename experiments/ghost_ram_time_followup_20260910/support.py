"""Follow-up experiments only; use the previous audit's captured systems."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ghost_ram_time_20260910'))
from common import *
PRIOR=HERE
HERE=Path(__file__).resolve().parent
def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
if __name__=='__main__':write('source-manifest.json',source_hashes())
