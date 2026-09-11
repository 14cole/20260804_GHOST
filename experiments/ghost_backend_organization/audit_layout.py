"""Inspect relocation imports, executable bodies, and documentation."""
import ast
import copy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).parent
BACKEND = ROOT / 'tools/GHOST/Backend'
MAPPING = json.loads((EVIDENCE / 'module-map.json').read_text())['modules']
FACADES = set(json.loads((EVIDENCE / 'module-map.json').read_text())['facades'])

guide = [
    '# GHOST import locations', '',
    'Use package imports when adding or updating code. The root entry modules',
    'marked **yes** remain callable under their established names.', '',
    '| Former module | Package import | Root entry retained |',
    '| --- | --- | --- |',
]
for old, new in sorted(MAPPING.items()):
    guide.append('| `{}` | `{}` | {} |'.format(old, new, 'yes' if old in FACADES else ''))
(BACKEND / 'IMPORTS.md').write_text('\n'.join(guide) + '\n', encoding='ascii')


class ExecutableBody(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def functions(tree):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


unchanged, changed = 0, []
documentation = []
for old, new in MAPPING.items():
    before = ast.parse((EVIDENCE / 'source' / (old + '.py')).read_text())
    path = BACKEND / (new.replace('.', '/') + '.py')
    after = ast.parse(path.read_text())
    a, b = functions(before), functions(after)
    assert [n.name for n in a] == [n.name for n in b], old
    for x, y in zip(a, b):
        xbody = ast.dump(ExecutableBody().visit(copy.deepcopy(x)))
        ybody = ast.dump(ExecutableBody().visit(copy.deepcopy(y)))
        if xbody == ybody:
            unchanged += 1
        else:
            changed.append([old, x.name])
    for node in [after] + [n for n in ast.walk(after) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]:
        doc = ast.get_docstring(node) or ''
        if re.search(r'historic|backstor|legacy|previously|originally|no longer|rejected|new production|phase[- ]\d|used to|why .*no', doc, re.I):
            documentation.append({'path': str(path.relative_to(ROOT)), 'name': getattr(node, 'name', 'module'), 'doc': doc})

result = {'unchanged_function_bodies': unchanged, 'changed_function_bodies': changed}
(EVIDENCE / 'function-comparison.json').write_text(json.dumps(result, indent=2))
(EVIDENCE / 'remaining-docstrings.json').write_text(json.dumps(documentation, indent=2))

strings, stale = [], []
for base in [ROOT / 'tools/GHOST', ROOT / 'GRIM_Revised_2', ROOT / 'tools/FREDDY']:
    for path in base.rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        source = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('ghost_backend.'):
                parent = parents.get(node)
                if isinstance(parent, (ast.Subscript, ast.Dict, ast.Compare, ast.keyword)):
                    strings.append([str(path.relative_to(ROOT)), node.lineno, ast.get_source_segment(source, parent)])
            if isinstance(node, ast.ImportFrom) and node.module in MAPPING:
                stale.append([str(path.relative_to(ROOT)), node.lineno, node.module])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in MAPPING:
                        stale.append([str(path.relative_to(ROOT)), node.lineno, alias.name])
(EVIDENCE / 'canonical-string-audit.json').write_text(json.dumps(strings, indent=2))
(EVIDENCE / 'flat-import-audit.json').write_text(json.dumps(stale, indent=2))
print(json.dumps(result))
print('Flagged docstrings:', len(documentation), 'data-context strings:', len(strings), 'flat imports:', len(stale))
