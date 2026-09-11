"""Public entry point for ghost_backend.assembly.fields."""
import sys
from ghost_backend.assembly import fields as _implementation

sys.modules[__name__] = _implementation
