"""Public entry point for ghost_backend.ui.app."""
if not __package__:
    import sys
    from pathlib import Path
    if Path(__file__).resolve().parent.name == "ghost_backend":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sys
from ghost_backend.ui import app as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
