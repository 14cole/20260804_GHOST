"""Run the preserved previous experiment against the same saved dense systems."""
from pathlib import Path
import sys,json
here=Path(__file__).resolve().parent
prior=here.parent/'ghost_ram_time_20260910'
sys.path[:0]=[str(here/'baseline_experiment'),str(prior)]
import streamed_probe as probe
probe.HERE=here/'baseline_experiment'
probe.PRIOR=prior
for pol in ('TE','TM'):
    probe.run(2.,pol,512,32.)
    result=json.loads((probe.HERE/('streamed-f2.0-{}-tile512-cut32.0.json'.format(pol))).read_text())
    if not result.get('accepted'):raise RuntimeError('Baseline qualification failed')

