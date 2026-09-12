"""Immutable-input preparation shared for the lifetime of a 2-D run.

Material tables are captured once per run. No matrices, factors, or meshes are
retained here, and nothing is reused across independent runs.
"""
from contextlib import contextmanager
from functools import wraps
import json
import hashlib
from pathlib import Path

from ghost_backend.execution.runtime import ScopedValue

_ACTIVE = ScopedValue('ghost_2d_preparation', None)


@contextmanager
def preparation_scope():
    if _ACTIVE.get() is not None:
        yield _ACTIVE.get()
        return
    with _ACTIVE.override({'materials': {}, 'geometry': {}, 'fingerprints': {}, 'hits': 0}):
        yield _ACTIVE.get()


def prepared_execution(function):
    @wraps(function)
    def call(*args, **kwargs):
        with preparation_scope():
            return function(*args, **kwargs)
    return call


def material_fingerprints(snapshot, base_dir):
    """Verify checkpoint inputs against the material tables captured in this run."""
    from ghost_backend.geometry.io import material_filename_from_row
    from ghost_backend.twod.geometry import _resolve_material_file
    entries = [snapshot.get('ibcs', []) or [], snapshot.get('dielectrics', []) or []]
    fingerprints = {}
    for row in entries[0] + entries[1]:
        name = material_filename_from_row(row)
        if name:
            path = Path(_resolve_material_file(base_dir, name))
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            digest = digest.hexdigest()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Material file changed while being captured: {}'.format(path))
            fingerprints[str(path)] = digest
    state = _ACTIVE.get()
    key = (base_dir, json.dumps(entries, sort_keys=True))
    if state is not None and key in state['fingerprints'] and state['fingerprints'][key] != fingerprints:
        raise ValueError('Material files changed after run preparation. Start the run again with the updated files.')
    return fingerprints


def prepare_geometry(snapshot, material_base_dir=None, units='inches'):
    # Import lazily to preserve the public solver's validation hooks.
    from ghost_backend.twod import solver
    state = _ACTIVE.get()
    base_dir = solver._material_base_dir_for_snapshot(snapshot, material_base_dir)
    scale = solver._unit_scale_to_meters(units)
    entries = [snapshot.get('ibcs', []) or [], snapshot.get('dielectrics', []) or []]
    material_key = (base_dir, json.dumps(entries, sort_keys=True))
    geometry_key = (material_key, scale, hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode('utf-8')).hexdigest())
    if state is not None and geometry_key in state['geometry']:
        state['hits'] += 1
        return state['geometry'][geometry_key]
    library = state['materials'].get(material_key) if state is not None else None
    if library is None:
        fingerprints = material_fingerprints(snapshot, base_dir)
        library = solver.MaterialLibrary.from_entries(entries[0], entries[1], base_dir=base_dir)
        if material_fingerprints(snapshot, base_dir) != fingerprints:
            raise ValueError('Material files changed during run preparation.')
        if state is not None:
            state['materials'][material_key] = library
            state['fingerprints'][material_key] = fingerprints
    report = solver.validate_geometry_snapshot_for_solver(
        snapshot, base_dir=base_dir, meters_scale=scale, material_library=library)
    result = (base_dir, report, library, float(scale))
    if state is not None:
        state['geometry'][geometry_key] = result
    return result
