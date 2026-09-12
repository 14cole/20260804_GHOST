"""Record exact audit counts and a hash manifest of the inspected source tree."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
results = json.loads((AUDIT/'acceptance-results.json').read_text())
total = skipped = failures = 0
for result in results:
    log = Path(result['log']).read_text(encoding='utf-8')
    match = re.search(r'^Ran (\d+) tests? in ', log, re.M)
    result['unittest_count'] = int(match[1]) if match else None
    result['skipped'] = int(re.search(r'\bskipped=(\d+)', log)[1]) if 'skipped=' in log else 0
    result['failures'] = int(re.search(r'\bfailures=(\d+)', log)[1]) if 'failures=' in log else 0
    total += result['unittest_count'] or 0
    skipped += result['skipped']
    failures += result['failures']
probes = []
for name in ('workflow-results.json','extended-results.json','material-results.json'):
    data = json.loads((AUDIT/name).read_text())
    probes.extend(data['results'])
powerpoint = json.loads((AUDIT/'powerpoint-results.json').read_text())
probes.append(dict(name='PowerPoint desktop export and render', status=powerpoint['status']))
summary = dict(suites=results, unittest_total=total, unittest_passed=total-skipped-failures,
               unittest_skipped=skipped, unittest_failed=failures, independent_probes=probes,
               independent_passed=sum(p['status']=='PASS' for p in probes))
(AUDIT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

extensions = {'.py','.pyw','.md','.toml','.json','.txt','.cpp','.c','.h','.hpp','.ps1','.bat','.cmd','.yaml','.yml'}
paths = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in extensions]
for tree in ('GRIM_Backend','tools/FREDDY','tools/GHOST','requirements'):
    paths.extend(p for p in (ROOT/tree).rglob('*') if p.is_file() and p.suffix in extensions
                 and not any(part in {'.venv','__pycache__','.git'} for part in p.parts))
manifest = []
for path in sorted(set(paths)):
    content = path.read_bytes()
    manifest.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
revision = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
status = subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
(AUDIT/'source-status.txt').write_text(status,encoding='utf-8')
(AUDIT/'source-snapshot.json').write_text(json.dumps(dict(revision=revision,scope='Root text files, GRIM_Backend, tools/FREDDY, tools/GHOST, requirements; excludes old experiment snapshots and environments',files=manifest),indent=2),encoding='utf-8')
print(json.dumps(dict(unittest_total=total,passed=total-skipped-failures,skipped=skipped,failed=failures,
                     independent_passed=summary['independent_passed'],source_files_hashed=len(manifest),revision=revision),indent=2))
