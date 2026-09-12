"""Run every declared acceptance suite, retaining failures and complete logs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
sys.path.insert(0, str(ROOT))
from verify_project import development_suites

records = []
for number, (name, cwd, args) in enumerate(development_suites(ROOT, 'full'), 1):
    started = time.monotonic()
    log = AUDIT / f'{number:02d}-' / 'output.log'
    log.parent.mkdir(exist_ok=True)
    print(f'START {number}: {name}', flush=True)
    with log.open('w', encoding='utf-8') as handle:
        try:
            result = subprocess.run([sys.executable, *args], cwd=cwd,
                env=dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONIOENCODING='utf-8'),
                stdout=handle, stderr=subprocess.STDOUT, timeout=3600)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
    record = dict(name=name, cwd=str(cwd), command=[sys.executable, *args],
                  returncode=code, seconds=round(time.monotonic()-started, 2), log=str(log))
    records.append(record)
    (AUDIT / 'acceptance-results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'END {number}: {name}: {code}, {record["seconds"]}s', flush=True)
sys.exit(0 if all(r['returncode'] == 0 for r in records) else 1)
