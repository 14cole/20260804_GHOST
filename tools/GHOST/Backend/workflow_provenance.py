"""Public entry point for ghost_backend.execution.provenance."""
import sys
from ghost_backend.execution import provenance as _implementation

sys.modules[__name__] = _implementation
