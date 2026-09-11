"""Validated JSON configuration for the existing local and HPC driver entrypoints."""

import argparse
import ast
from ghost_backend.execution.runtime import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

SCHEMA = 'ghost.driver-config'


@dataclass(frozen=True)
class LoadedConfiguration:
    path: 'Path'
    sha256: 'str'

    def verify(self):
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError('Driver configuration changed after loading. Restart with a stable configuration.')


OPTIONAL_INTS = {'WORKERS', 'N_MODES', 'ARRAY_THROTTLE', 'CORES_PER_NODE', 'MAX_WORKERS_PER_NODE'}
POSITIVE_INTS = {'N_NODES', 'N_JOBS', 'MAX_PANELS', 'MAX_ELEMENTS',
                 'BLAS_THREADS_PER_WORKER', 'WORKERS_PER_UNIT', 'TASKS_PER_CHILD'}
OPTIONAL_TEXT = {'SLURM_ACCOUNT', 'SLURM_QOS', 'SLURM_TIME', 'MEM_PER_NODE',
                 'SLURM_MAIL_TYPE', 'SLURM_MAIL_USER'}
TEXT = {'FRD_DIR', 'OPN_DIR', 'OUTPUT_DIR', 'SLURM_PARTITION', 'PYTHON_EXE'}
NUMERIC_LISTS = {'FREQUENCIES_GHZ', 'AZIMUTHS_DEG', 'ELEVATIONS_DEG'}
TEXT_LISTS = {'GEOMETRY_DIRS', 'GEOMETRY_EXTS', 'SLURM_EXTRA_SBATCH', 'JOB_PROLOGUE'}
NUMBERS = {'BODY_AXIS_AZ_DEG', 'BODY_AXIS_EL_DEG', 'BODY_ROLL_DEG', 'CFIE_ALPHA',
           'MODE_TOL', 'STREAM_BUDGET_GB', 'MEMORY_HEADROOM', 'MEMORY_SAFETY',
           'CLAIM_STALE_SECONDS', 'MAX_SOLVE_GB'}
CHOICES = {'SOLVER_METHOD': ('auto', 'direct', 'experimental_cpu'), 'GEOMETRY_UNITS': ('inches', 'meters'), 'ACCURACY_TARGET': ('standard', 'tight'),
           'LU_PRECISION': ('double', 'mixed'), 'ASSEMBLY': ('auto', 'tables', 'streaming'),
           'TABLE_PRECISION': ('auto', 'single', 'double')}


def validate_settings(settings, allowed_keys):
    if not isinstance(settings, dict):
        raise ValueError('Driver settings must be a JSON object.')
    unknown = set(settings) - set(allowed_keys)
    if unknown:
        raise ValueError('Unsupported driver settings: ' + ', '.join(sorted(unknown)))
    result = {}
    for key, value in settings.items():
        valid = False
        if key == 'EXECUTION_OPTIONS':
            from ghost_backend.execution.options import validate_options
            value = validate_options(value)
            valid = True
        elif key in POSITIVE_INTS | OPTIONAL_INTS:
            valid = (value is None and key in OPTIONAL_INTS) or (type(value) is int and value >= 1)
        elif key in TEXT | OPTIONAL_TEXT:
            valid = (value is None and key in OPTIONAL_TEXT) or (isinstance(value, str) and bool(value.strip()) and '\n' not in value and '\r' not in value)
        elif key in NUMERIC_LISTS:
            valid = isinstance(value, (list, tuple)) and 0 < len(value) <= 100000 and all(
                type(v) in (float, int) and math.isfinite(v) and (key != 'FREQUENCIES_GHZ' or v > 0)
                for v in value)
            if valid:
                valid = len(value) == len(set(value))
                if key == 'FREQUENCIES_GHZ':
                    valid = valid and len(value) == len({f'{v:06.3f}' for v in value})
                value = list(value)
        elif key in TEXT_LISTS:
            valid = isinstance(value, (list, tuple)) and all(isinstance(v, str) and '\n' not in v and '\r' not in v for v in value)
            if valid:
                value = list(value)
                if key in {'GEOMETRY_DIRS', 'GEOMETRY_EXTS'}:
                    valid = bool(value) and all(v.strip() for v in value)
        elif key in NUMBERS:
            valid = (key == 'MAX_SOLVE_GB' and value is None) or (type(value) in (float, int) and math.isfinite(value))
            if valid and value is not None:
                if key == 'MEMORY_HEADROOM':
                    valid = 0 < value <= 1
                elif key == 'CFIE_ALPHA':
                    valid = 0 < value < 1
                elif key == 'MEMORY_SAFETY':
                    valid = value >= 1
                elif key == 'CLAIM_STALE_SECONDS':
                    valid = value >= 60
                elif key not in {'BODY_AXIS_AZ_DEG', 'BODY_AXIS_EL_DEG', 'BODY_ROLL_DEG'}:
                    valid = value > 0
        elif key in CHOICES:
            valid = isinstance(value, str) and value in CHOICES[key]
        elif key in {'MESH_CERTIFICATION', 'SUBMIT'}:
            valid = type(value) is bool
        elif key == 'ASSEMBLY_THREADS':
            valid = value == 'auto' or (type(value) is int and value >= 1)
        if not valid:
            raise ValueError(f'Invalid driver setting {key}: {value!r}')
        result[key] = value
    if result.get('SOLVER_METHOD') == 'experimental_cpu' and result.get('LU_PRECISION', 'double') != 'double':
        raise ValueError('Experimental CPU requires double LU precision.')
    return result


def settings_from_run_setup(value, kind):
    """Reuse the desktop 2-D recipe for the monostatic driver capabilities."""
    from ghost_backend.runs.setup import DEFAULT_QUALITY, validate_setup
    if kind != '2d':
        raise ValueError('A 2-D run setup cannot configure a BoR driver.')
    setup = validate_setup(value)
    if setup['scattering'] != 'monostatic' or setup['quality'] != DEFAULT_QUALITY:
        raise ValueError('Batch drivers require a monostatic setup with their standard quality thresholds.')
    return dict(FREQUENCIES_GHZ=setup['frequencies_ghz'], AZIMUTHS_DEG=setup['angles_deg'],
                GEOMETRY_UNITS=setup['units'], MESH_CERTIFICATION=setup['mesh_certification'],
                ACCURACY_TARGET=setup['accuracy'], LU_PRECISION=setup['lu_precision'],
                SOLVER_METHOD=setup.get('solver_method', 'direct'),
                EXECUTION_OPTIONS=setup['execution_options'])


def configuration_payload(kind, settings, allowed_keys, *, run_setup=None):
    if kind not in ('2d', 'bor'):
        raise ValueError('Driver kind must be 2d or bor.')
    merged = settings_from_run_setup(run_setup, kind) if run_setup is not None else {}
    checked = validate_settings(settings, allowed_keys)
    for key in merged.keys() & checked.keys():
        if merged[key] != checked[key]:
            raise ValueError(f'Driver setting {key} conflicts with the embedded run setup.')
    merged.update(checked)
    if (kind == '2d' and merged.get('LU_PRECISION') == 'mixed' and
            'SOLVER_METHOD' not in merged and 'SOLVER_METHOD' in allowed_keys):
        merged['SOLVER_METHOD'] = 'direct'
    from ghost_backend.runs.execution import reconcile_settings
    merged = reconcile_settings(merged)
    return dict(schema=SCHEMA, version=1, driver=kind,
                settings=validate_settings(merged, allowed_keys))


def write_configuration(path, payload):
    """Atomically publish a previously validated configuration."""
    path = Path(path)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix='.driver-config-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def driver_contract(driver):
    """Read literal declarations without executing or rewriting driver source."""
    tree = ast.parse(Path(driver).read_text(encoding='utf-8-sig'))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('_CONFIG_KEYS', '_CONFIG_KIND'):
                values[name] = ast.literal_eval(node.value)
    if set(values) != {'_CONFIG_KEYS', '_CONFIG_KIND'}:
        raise ValueError('Driver does not declare a supported configuration contract.')
    return values['_CONFIG_KIND'], values['_CONFIG_KEYS']


def load_driver_configuration(namespace, script, kind, allowed_keys):
    """Apply an explicit --config file or an adjacent .config.json, before use."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config')
    argv = sys.argv[1:] if namespace.get('__name__') in ('__main__', '__mp_main__') else []
    args, _ = parser.parse_known_args(argv)
    path = Path(args.config).expanduser().resolve() if args.config else Path(script).with_suffix('.config.json')
    if not path.is_file() and not args.config:
        return None
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Driver configuration exceeds 4 MiB.')
    data = path.read_bytes()
    raw = json.loads(data.decode('utf-8'))
    if (not isinstance(raw, dict) or set(raw) - {'schema', 'version', 'driver', 'settings', 'run_setup'} or
            raw.get('schema') != SCHEMA or type(raw.get('version')) is not int or raw['version'] != 1 or
            raw.get('driver') != kind or 'settings' not in raw):
        raise ValueError('Unsupported driver configuration schema, version, or solver kind.')
    checked = configuration_payload(kind, raw['settings'], allowed_keys, run_setup=raw.get('run_setup'))
    namespace.update(checked['settings'])
    return LoadedConfiguration(path.resolve(), hashlib.sha256(data).hexdigest())


def configuration_source_records(script, configuration):
    records = {'driver_configured.py': str(Path(script).resolve())}
    if configuration is not None:
        configuration.verify()
        records['driver_configured.config.json'] = str(configuration.path)
    return records


def copy_configuration(configuration, driver_copy):
    """Keep submission and worker provenance identical under stable logical names."""
    if configuration is not None:
        configuration.verify()
        destination = Path(driver_copy).with_suffix('.config.json')
        shutil.copy2(configuration.path, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != configuration.sha256:
            raise ValueError('Driver configuration changed while staging the worker copy.')
