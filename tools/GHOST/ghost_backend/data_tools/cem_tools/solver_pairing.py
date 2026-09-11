"""Load the solver's canonical filename and OPN/FRD pairing implementation."""

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType

from .errors import CemToolError


def solver_backend_path() -> 'Path':
    configured = os.environ.get("CEM_SOLVER_BACKEND_PATH")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if (candidate / 'ghost_backend' / 'run_gui.py').is_file():
            return candidate / 'ghost_backend'
        return candidate
    return Path(__file__).resolve().parents[2]


def pairing_module() -> 'ModuleType':
    """Load filename operations from one complete, consistent backend tree."""
    backend = solver_backend_path().resolve()
    module_path = backend / "io" / "naming.py"
    if not module_path.is_file():
        raise CemToolError(
            f"solver pairing library not found at {module_path}; set "
            "CEM_SOLVER_BACKEND_PATH to the project's Backend folder"
        )
    for name, loaded in tuple(sys.modules.items()):
        if name == 'ghost_backend' or name.startswith('ghost_backend.'):
            if name == 'ghost_backend':
                roots = {Path(value).resolve() for value in getattr(loaded, '__path__', ())}
                if roots != {backend}:
                    raise CemToolError('Conflicting GHOST package; restart CEM Tools with one backend.')
                continue
            origin = getattr(loaded, '__file__', None)
            if not origin or backend not in Path(origin).resolve().parents:
                raise CemToolError(
                    f"Conflicting GHOST module {name}; restart CEM Tools "
                    "with one solver backend directory."
                )
    backend_text = str(backend)
    sys.path[:] = [entry for entry in sys.path if entry != backend_text]
    sys.path.insert(0, backend_text)
    sys.path.insert(0, str(backend.parent))
    module = importlib.import_module('ghost_backend.io.naming')
    if Path(module.__file__).resolve() != module_path:
        raise CemToolError(f"cannot import solver pairing library {module_path}")
    return module
