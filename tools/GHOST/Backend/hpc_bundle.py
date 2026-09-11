"""Public entry point for ghost_backend.hpc.bundle."""
import sys
from ghost_backend.hpc import bundle as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
