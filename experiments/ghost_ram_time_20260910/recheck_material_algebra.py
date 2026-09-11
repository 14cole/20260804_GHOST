"""Revalidate captured fixtures without repeating electromagnetic assembly."""
from common import *
from material_probe import algebra_checks
for path in HERE.glob('material-*.json'):
    if path.name=='material-errors.json':continue
    result=json.loads(path.read_text())
    if not isinstance(result,dict) or 'kind' not in result:continue
    kind=result['kind'];count=result['count']
    for pol in result['algebra']:
        result['algebra'][pol]=algebra_checks(HERE/'systems'/('{}-n{}-f0.6'.format(kind,count))/pol)
    write(path.name,result)
    print(kind,'checked',flush=True)
