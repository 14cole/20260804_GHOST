"""Public entry point for ghost_backend.ui.app."""
import sys
from ghost_backend.ui import app as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
