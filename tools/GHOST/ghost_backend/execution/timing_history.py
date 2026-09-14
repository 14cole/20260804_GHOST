"""Host-local timing evidence for identical requests; never a numerical cache.

Only repeated successful measurements of at least two backends can change a
ranking. Changed source, materials, angles, meshes, tolerances, or CPU settings
invalidate evidence. A missing/unwritable cache leaves the work prior intact.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import uuid
import time
from ghost_backend.execution.runtime import unlink_if_exists

MAX_BYTES=2*1024**2
MAX_ENTRIES=128
MAX_AGE=14*24*3600


def cache_path():
    base=os.environ.get('GHOST_TIMING_CACHE_DIR')
    if not base:
        base=os.path.join(os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME') or
                          os.path.join(os.path.expanduser('~'),'.cache'),'GHOST','solver-timings')
    return Path(base)/'timings-v1.json'


def request_key(arguments,options,name):
    from ghost_backend.execution.provenance import backend_source_records, source_bundle_fingerprint
    from ghost_backend.execution.options import effective_assembly_threads
    from ghost_backend.twod.preparation import material_fingerprints
    from ghost_backend.twod.solver import _material_base_dir_for_snapshot
    from ghost_backend.linalg.refined_lu import requested_precision
    from ghost_backend.execution.thread_control import threadpool_info
    import numpy as np
    import scipy
    snapshot=arguments.get('geometry_snapshot')
    if not isinstance(snapshot,dict):return None
    ignored={'abort_event','progress_callback','geometry_snapshot','solver_method','material_base_dir'}
    payload={k:v for k,v in arguments.items() if k not in ignored and not k.startswith('_')}
    payload['geometry']={k:snapshot.get(k) for k in ('segments','ibcs','dielectrics')}
    payload['materials']=material_fingerprints(snapshot,_material_base_dir_for_snapshot(snapshot,arguments.get('material_base_dir')))
    payload['precision']=requested_precision()
    payload['blas']=[{k:v for k,v in item.items() if k in ('internal_api','version','threading_layer','architecture')}
                     for item in threadpool_info()]
    payload['options']={k:v for k,v in options.items() if k not in ('factorization','ram_budget_gib','temporary_directory')}
    payload['host']=(platform.node(),platform.machine(),platform.processor(),platform.platform(),
                     platform.python_version(),np.__version__,scipy.__version__,effective_assembly_threads())
    payload['source']=source_bundle_fingerprint(backend_source_records(str(Path(__file__).resolve().parents[1])))
    payload['entrypoint']=name
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def read():
    try:
        path=cache_path()
        if path.stat().st_size>MAX_BYTES:return {}
        result=json.loads(path.read_text(encoding='utf-8'))
        return result if isinstance(result,dict) and len(result)<=MAX_ENTRIES else {}
    except (OSError,ValueError):return {}


def measured_costs(key):
    entry=read().get(key,{})
    if not isinstance(entry,dict):return {}
    result={}
    for mode,rows in entry.items():
        if mode not in ('dense','compressed','fmm') or not isinstance(rows,list):continue
        values=[]
        for row in rows[-5:]:
            if not isinstance(row,list) or len(row)!=2:continue
            stamp,seconds=row
            if (isinstance(stamp,(int,float)) and isinstance(seconds,(int,float)) and
                math.isfinite(stamp) and math.isfinite(seconds) and
                0<=time.time()-stamp<=MAX_AGE and 0<seconds<=MAX_AGE):values.append(seconds)
        if len(values)>=2:result[mode]=statistics.median(values)
    return result


def adjust(selection,key,batch=False):
    candidates=selection.get('candidates',{})
    measured={m:t for m,t in measured_costs(key).items() if m in candidates}
    if len(measured)<2:return selection
    ratios=[t/candidates[m]['cost'] for m,t in measured.items()]
    scale=statistics.median(ratios)
    revised={m:dict(c,prior_cost=c['cost'],cost=measured.get(m,c['cost']*scale),
                    timing_evidence='measured_median' if m in measured else 'scaled_prior')
             for m,c in candidates.items()}
    from ghost_backend.execution.policy import rank_candidates
    budget=(candidates[selection['selected']]['peak_gb'] if batch else selection['admission_budget_gib'])
    ranked=rank_candidates(revised,budget,margin=0. if batch else .2)
    return dict(selection,selected=ranked[0],retry_order=ranked[1:],candidates=revised,
                timing_model='identical_request_host_median_v1',measured_backends=sorted(measured),
                prior_selected_backend=selection['selected'],batch_plan_revised=bool(batch and ranked[0]!=selection['selected']),
                reason='Ranked using repeated timings for this exact request and execution host; RAM admission retained.')


def record(key,mode,seconds,metadata):
    if (key is None or mode not in ('dense','compressed','fmm') or not math.isfinite(seconds) or seconds<=0 or
        seconds>MAX_AGE or not metadata.get('quality_gate',{}).get('passed') or
        metadata.get('backend_selection',{}).get('failed_attempts')):return
    if metadata.get('frequency_metadata') and any(
        row['metadata'].get('backend_selection',{}).get('selected',mode) != mode or
        row['metadata'].get('adaptive_mesh',{}).get('fallback')
        for row in metadata['frequency_metadata']):return
    adaptation=metadata.get('adaptive_mesh',{})
    if (adaptation.get('fallback') or any(step.get('failed_backends') or step.get('backend') != mode
                                        for step in adaptation.get('steps',[]))):return
    path=cache_path();temporary=None
    try:
        entries=read();entry=entries.setdefault(key,{})
        if not isinstance(entry,dict):entry={};entries[key]=entry
        rows=entry.get(mode,[])
        entry[mode]=(rows[-4:] if isinstance(rows,list) else [])+[[time.time(),seconds]]
        # Concurrent writers may lose a timing sample, never a solver result.
        while len(entries)>MAX_ENTRIES:entries.pop(next(iter(entries)))
        path.parent.mkdir(parents=True,exist_ok=True)
        # Windows tempfile can retry PermissionError for an effectively
        # unwritable directory billions of times. This optional cache gets one
        # exclusive creation attempt and never holds up a completed solve.
        candidate=path.with_name('.timings-'+uuid.uuid4().hex+'.tmp')
        with open(candidate,'x',encoding='utf-8') as stream:
            temporary=candidate
            json.dump(entries,stream,allow_nan=False)
        os.replace(temporary,path)
    except (OSError,ValueError):pass
    finally:
        if temporary is not None:
            try:unlink_if_exists(temporary)
            except OSError:pass
