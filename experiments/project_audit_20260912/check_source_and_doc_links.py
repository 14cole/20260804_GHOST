from pathlib import Path
import ast
import json
import re
from urllib.parse import unquote

audit = Path(__file__).resolve().parent
root = audit.parents[1]
manifest = json.loads((audit/'source-snapshot.json').read_text())
syntax_errors, missing_links = [], []
python_count = document_count = 0
for entry in manifest['files']:
    path = root / entry['path']
    if path.suffix == '.py':
        python_count += 1
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except (SyntaxError, UnicodeError) as error:
            syntax_errors.append({'path': entry['path'], 'error': str(error)})
    if path.suffix == '.md':
        document_count += 1
        content = path.read_text(encoding='utf-8-sig')
        # Exclude fenced source examples; inspect local inline links only.
        content = re.sub(r'```.*?```', '', content, flags=re.S)
        for match in re.finditer(r'\[[^\]\n]*\]\(([^)\n]+)\)', content):
            target = match[1].strip('<>')
            if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#',1)[0])
            if not target or '$' in target:
                continue
            resolved = path.parent / target
            if not resolved.exists():
                missing_links.append({'document': entry['path'], 'target': target})
result = dict(python_files=python_count, syntax_errors=syntax_errors,
              markdown_files=document_count, missing_local_inline_links=missing_links)
(audit/'source-and-doc-checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
raise SystemExit(1 if syntax_errors or missing_links else 0)
