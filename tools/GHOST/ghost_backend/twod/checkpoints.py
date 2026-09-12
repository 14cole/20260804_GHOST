"""Atomic, input-verified frequency checkpoints for desktop 2-D sweeps.

Archives contain column arrays and JSON only (never executable pickle). A
completed frequency is reusable only with matching inputs, material contents,
backend source, precision, quality settings, and an intact content digest.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import time
import numpy as np


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=lambda v: v.item())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(arguments, options, precision, certified):
    from ghost_backend.twod.geometry import _material_base_dir_for_snapshot
    from ghost_backend.twod.preparation import material_fingerprints
    snapshot = arguments['geometry_snapshot']
    base = _material_base_dir_for_snapshot(snapshot, arguments.get('material_base_dir'))
    files = material_fingerprints(snapshot, base)
    source = hashlib.sha256()
    backend = Path(__file__).resolve().parents[1]
    for folder in ('twod', 'compressed', 'linalg', 'execution', 'runs', 'geometry'):
        for path in sorted((backend / folder).rglob('*.py')):
            source.update(str(path.relative_to(backend)).replace('\\', '/').encode('utf-8'))
            source.update(path.read_bytes())
    inputs = {key: value for key,value in arguments.items()
              if key not in ('progress_callback', 'abort_event', 'frequencies_ghz')}
    # With a shared mesh reference, the sweep controls its conservative scale.
    if arguments.get('mesh_reference_ghz') is not None:
        inputs['mesh_frequencies_ghz'] = arguments['frequencies_ghz']
    return hashlib.sha256(_json(dict(inputs=inputs, options=options, precision=precision,
        certified=certified, materials=files, source=source.hexdigest(), schema=1)).encode('utf-8')).hexdigest()


class FrequencyCheckpoints:
    def __init__(self, directory, identity, certified):
        self.directory = Path(directory) / identity
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.certified = bool(certified)

    def _path(self, frequency):
        key = hashlib.sha256(float(frequency).hex().encode('ascii')).hexdigest()[:24]
        return self.directory / (key + '.npz')

    def save(self, frequency, result):
        if self.certified and result.get('metadata', {}).get('mesh_convergence_certified') is not True:
            raise ValueError('An uncertified frequency cannot enter a certified checkpoint.')
        rows = result['samples']
        columns = sorted(set(key for row in rows for key in row))
        arrays, encodings = {}, []
        for index,key in enumerate(columns):
            values = [row.get(key) for row in rows]
            # Store numeric solver fields directly; JSON is only a fallback for
            # heterogeneous extension fields. No object-dtype arrays are used.
            if all(type(value) in (int, float, bool, str) for value in values) and len({type(value) for value in values}) == 1:
                arrays['c' + str(index)] = np.asarray(values)
                encodings.append('scalar')
            else:
                arrays['c' + str(index)] = np.asarray([_json(value) for value in values])
                encodings.append('json')
            arrays['p' + str(index)] = np.asarray([key in row for row in rows], dtype=bool)
        header = {key: value for key,value in result.items() if key not in ('samples', 'co_solved_samples')}
        record = dict(identity=self.identity, frequency=float(frequency), certified=self.certified,
                      columns=columns, encodings=encodings, result=header, co_solved='co_solved_samples' in result)
        arrays['header'] = np.frombuffer(_json(record).encode('utf-8'), dtype=np.uint8)
        path = self._path(frequency)
        handle, temporary = tempfile.mkstemp(prefix='frequency-', suffix='.npz', dir=str(self.directory))
        try:
            with os.fdopen(handle, 'wb') as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            digest = _sha(temporary)
            os.replace(temporary, str(path))
            marker = dict(identity=self.identity, frequency=float(frequency), sha256=digest)
            handle, manifest = tempfile.mkstemp(prefix='digest-', suffix='.json', dir=str(self.directory))
            try:
                with os.fdopen(handle, 'w', encoding='utf-8') as stream:
                    stream.write(_json(marker))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(manifest, str(path.with_suffix('.json')))
            finally:
                if os.path.exists(manifest):
                    os.unlink(manifest)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, frequency):
        path = self._path(frequency)
        try:
            marker = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if marker != dict(identity=self.identity, frequency=float(frequency), sha256=_sha(path)):
                return None
            with np.load(str(path), allow_pickle=False) as data:
                record = json.loads(data['header'].tobytes().decode('utf-8'))
                if (record['identity'] != self.identity or record['frequency'] != float(frequency)
                        or record['certified'] != self.certified):
                    return None
                rows = None
                for index,key in enumerate(record['columns']):
                    values, present = data['c'+str(index)], data['p'+str(index)]
                    if rows is None:
                        rows = [{} for _ in values]
                    if len(values) != len(rows) or len(present) != len(rows):
                        return None
                    for row,value,exists in zip(rows, values, present):
                        if exists:
                            row[key] = value.item() if record['encodings'][index] == 'scalar' else json.loads(str(value))
            result = record['result']
            result['samples'] = rows or []
            if self.certified and result.get('metadata', {}).get('mesh_convergence_certified') is not True:
                return None
            if any(float(row['frequency_ghz']) != float(frequency) for row in result['samples']):
                return None
            if record['co_solved']:
                result['co_solved_samples'] = {pol: [row for row in result['samples'] if row['polarization'] == pol]
                                               for pol in ('VV', 'HH')}
            return result
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            return None


def run_checkpointed(solve, arguments, directory, options, precision, certified):
    from ghost_backend.twod.solver import _merge_frequency_results
    started = time.perf_counter()
    profiles = []
    frequencies = list(arguments['frequencies_ghz'])
    if len(set(frequencies)) != len(frequencies):
        raise ValueError('Duplicate frequencies are not supported in a co-polarized result grid.')
    identity = input_identity(arguments, options, precision, certified)
    store = FrequencyCheckpoints(directory, identity, certified)
    progress = arguments.get('progress_callback')
    abort = arguments.get('abort_event')
    reused = 0
    for index,frequency in enumerate(frequencies):
        if abort is not None and abort.is_set():
            raise InterruptedError('Solve canceled; completed frequency checkpoints were retained.')
        cached = store.load(frequency)
        if cached is not None:
            reused += 1
        else:
            def report(done, total, message):
                if progress:
                    progress(index*1000 + int(1000*done/max(total,1)), len(frequencies)*1000, message)
            result = solve(**dict(arguments, frequencies_ghz=[frequency], progress_callback=report))
            if result.get('metadata', {}).get('runtime_profile'):
                profiles.append(result['metadata']['runtime_profile'])
            store.save(frequency, result)
            del result
        del cached
        if progress:
            progress((index+1)*1000, len(frequencies)*1000,
                     'Frequency {:g} GHz saved; {} of {} complete ({} reused).'.format(frequency,index+1,len(frequencies),reused))
    if abort is not None and abort.is_set():
        raise InterruptedError('Solve canceled; completed frequency checkpoints were retained.')
    def completed():
        for frequency in frequencies:
            value = store.load(frequency)
            if value is None:
                raise IOError('A completed frequency checkpoint changed before result export. Rerun to recompute it.')
            yield value
    result = _merge_frequency_results(completed(), frequencies)
    result['metadata']['frequency_checkpoints'] = dict(directory=str(store.directory),
        completed=len(frequencies), reused=reused, input_sha256=identity)
    peaks = [p['sampled_peak_process_rss_bytes'] for p in profiles if p.get('sampled_peak_process_rss_bytes') is not None]
    result['metadata']['runtime_profile'] = dict(
        wall_seconds=time.perf_counter()-started,
        stage_seconds={key: sum(p.get('stage_seconds',{}).get(key,0.) for p in profiles)
                       for key in set(key for p in profiles for key in p.get('stage_seconds',{}))},
        stage_calls={key: sum(p.get('stage_calls',{}).get(key,0) for p in profiles)
                     for key in set(key for p in profiles for key in p.get('stage_calls',{}))},
        sampled_peak_process_rss_bytes=max(peaks) if peaks else None,
        stage_semantics='Current execution only; cached stages excluded; nested stages may overlap.',
        memory_semantics='Process samples from frequencies computed in this execution; cached samples excluded.')
    return result
