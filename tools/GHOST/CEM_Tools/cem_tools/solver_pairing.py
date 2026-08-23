"""Load the solver's canonical filename and OPN/FRD pairing implementation."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

from .errors import CemToolError


def solver_backend_path() -> 'Path':
    configured = os.environ.get("CEM_SOLVER_BACKEND_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "Backend"


def pairing_module() -> 'ModuleType':
    module_path = solver_backend_path() / "grim_naming.py"
    if not module_path.is_file():
        raise CemToolError(
            f"solver pairing library not found at {module_path}; set "
            "CEM_SOLVER_BACKEND_PATH to the project's Backend folder"
        )
    module_name = "_cem_tools_solver_grim_naming"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise CemToolError(f"cannot import solver pairing library {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module
