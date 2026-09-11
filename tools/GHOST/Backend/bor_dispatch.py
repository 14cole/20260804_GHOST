"""Public entry point for ghost_backend.bor.dispatch."""
import sys
from ghost_backend.bor import dispatch as _implementation

sys.modules[__name__] = _implementation
