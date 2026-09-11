"""Write a reusable 10 GHz, 361-azimuth compressed 2D run setup."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'tools/GHOST/Backend'))
from ghost_backend.runs.setup import DEFAULT_QUALITY, save_setup, read_setup
from ghost_backend.execution.options import validate_options
path = Path(__file__).with_name('airfoil-10GHz.run.json')
options = validate_options(dict(factorization='compressed', compressed_storage_mib=8192,
                               assembly_threads=4, blas_threads=2))
save_setup(path, dict(schema='grim.2d-run-setup', version=2, frequencies_ghz=[10.0],
    angles_deg=list(range(361)), units='inches', mesh_certification=True, accuracy='standard',
    lu_precision='double', solver_method='experimental_cpu', scattering='monostatic',
    observation_angles_deg=[], quality=dict(DEFAULT_QUALITY), execution_options=options))
assert read_setup(path)['execution_options'] == options
print(path)
