"""Public entry point for ghost_backend.io.viewer_bridge."""
import sys
from ghost_backend.io import viewer_bridge as _implementation

sys.modules[__name__] = _implementation
