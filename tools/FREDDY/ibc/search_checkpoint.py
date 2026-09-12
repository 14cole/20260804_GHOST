"""Portable, atomic inverse-search checkpoints containing completed scores only."""
from __future__ import annotations

from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile

SCHEMA = 'freddy.inverse-checkpoint'
VERSION = 1
MAX_SCORE_BYTES = 512 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


def engine_identity():
    """Invalidate saved scores when the search or numerical implementation changes."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ('compute.py', 'design_search.py', 'inverse_grid.py',
                 'inverse_workflow.py', 'ui_options.py', 'io.py', 'search_checkpoint.py'):
        digest.update(name.encode('ascii'))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _validate_header(value):
    if not isinstance(value, dict):
        raise ValueError('Invalid search checkpoint metadata.')
    for key in ('identity', 'engine', 'scores_sha256'):
        if not isinstance(value.get(key), str) or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise ValueError(f'Invalid checkpoint {key}.')
    if value.get('schema') != SCHEMA or type(value.get('version')) is not int or value['version'] != VERSION:
        raise ValueError('Unsupported search checkpoint version.')
    done, total = value.get('next_index'), value.get('total')
    if type(done) is not int or type(total) is not int or not 0 <= done <= total or total < 1:
        raise ValueError('Invalid completed-combination count.')
    if type(value.get('plots_complete')) is not bool:
        raise ValueError('Invalid checkpoint plot status.')
    if done * 40 > MAX_SCORE_BYTES:
        raise ValueError('Checkpoint scores exceed the 512 MiB recovery-file limit.')


def save_checkpoint(path, checkpoint, *, overwrite=True):
    """Publish an archive only after its full score stream is durable.

    The caller must use a stable captured checkpoint on the search thread or
    an idle UI thread. Existing recovery files survive any failed write.
    """
    path = Path(path)
    rows = checkpoint['score_rows']
    if not isinstance(rows, array) or rows.typecode != 'd' or rows.itemsize != 8:
        raise ValueError('Checkpoint scores must be 64-bit floating-point values.')
    metadata = dict(schema=SCHEMA, version=VERSION, engine=engine_identity(),
                    identity=checkpoint['identity'], next_index=checkpoint['next_index'],
                    total=checkpoint['total'], plots_complete=checkpoint['plots_complete'],
                    scores_sha256='0' * 64)
    _validate_header(metadata)
    if len(rows) != metadata['next_index'] * 5 or not all(math.isfinite(v) for v in rows):
        raise ValueError('Checkpoint contains incomplete or nonfinite scores.')
    if sys.byteorder != 'little':
        rows = array('d', rows)
        rows.byteswap()
    data = memoryview(rows).cast('B')
    fd, temporary = tempfile.mkstemp(prefix='.freddy-checkpoint-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w+b') as stream:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
                digest = hashlib.sha256()
                with archive.open('scores.f64le', 'w', force_zip64=True) as output:
                    for start in range(0, len(data), CHUNK_BYTES):
                        block = data[start:start + CHUNK_BYTES]
                        output.write(block)
                        digest.update(block)
                metadata['scores_sha256'] = digest.hexdigest()
                archive.writestr('checkpoint.json', json.dumps(metadata, sort_keys=True, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Publish a fresh search without racing an existing file. Both
            # files are in the same directory/filesystem; link is atomic.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    """Validate format, size, code identity and content before returning scores.

    Resume also verifies the current physics inputs and material contents in
    run_inverse_search. Comparison plots are rebuilt after loading.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ['checkpoint.json', 'scores.f64le']:
                raise ValueError('Unexpected search checkpoint contents.')
            if archive.getinfo('checkpoint.json').file_size > 16384:
                raise ValueError('Search checkpoint metadata is too large.')
            metadata = json.loads(archive.read('checkpoint.json'))
            _validate_header(metadata)
            if metadata['engine'] != engine_identity():
                raise ValueError('Search implementation changed. Start a new analysis.')
            if archive.getinfo('scores.f64le').file_size != metadata['next_index'] * 40:
                raise ValueError('Checkpoint score size does not match completed combinations.')
            rows = array('d')
            digest = hashlib.sha256()
            with archive.open('scores.f64le') as source:
                for block in iter(lambda: source.read(CHUNK_BYTES), b''):
                    digest.update(block)
                    rows.frombytes(block)
            if digest.hexdigest() != metadata['scores_sha256']:
                raise ValueError('Checkpoint score checksum does not match.')
            if sys.byteorder != 'little':
                rows.byteswap()
            if not all(math.isfinite(v) for v in rows):
                raise ValueError('Checkpoint contains nonfinite scores.')
    except (zipfile.BadZipFile, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read search checkpoint: {exc}') from exc
    return dict(identity=metadata['identity'], score_rows=rows,
                next_index=metadata['next_index'], total=metadata['total'], plots_complete=False)
