"""Public entry point for ghost_backend.twod.solver."""
import sys
from ghost_backend.twod import solver as _implementation

sys.modules[__name__] = _implementation
