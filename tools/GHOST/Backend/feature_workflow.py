"""Public entry point for ghost_backend.assembly.workflow."""
import sys
from ghost_backend.assembly import workflow as _implementation

sys.modules[__name__] = _implementation
