"""Small runtime adapters for the Python 3.6+ headless GHOST backend.

Desktop Python uses its standard dataclasses and context variables. Python
3.6 uses the bundled, unmodified dataclasses 0.8 backport and thread-local
solver settings. The latter support synchronous nested/threaded solves;
they do not implement asyncio task contexts on Python 3.6.
"""
from contextlib import contextmanager
import sys
import threading

if sys.version_info < (3, 7):
    from _ghost_dataclasses import asdict, dataclass, field, replace
else:
    from dataclasses import asdict, dataclass, field, replace

try:
    from contextvars import ContextVar
except ImportError:
    ContextVar = None


class ScopedValue:
    """A solver setting restored after nested calls, exceptions, and threads."""

    def __init__(self, name, default):
        self._default = default
        self._context = ContextVar(name, default=default) if ContextVar else None
        self._local = threading.local() if self._context is None else None

    def get(self):
        if self._context is not None:
            return self._context.get()
        return getattr(self._local, 'value', self._default)

    @contextmanager
    def override(self, value):
        if self._context is not None:
            token = self._context.set(value)
            try:
                yield
            finally:
                self._context.reset(token)
        else:
            previous = self.get()
            self._local.value = value
            try:
                yield
            finally:
                self._local.value = previous


def unlink_if_exists(path):
    """Remove a file, ignoring only absence (Path.unlink on Python 3.6)."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_text_lf(path, text):
    """Write UTF-8 text with LF newlines on every supported interpreter."""
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        return stream.write(text)
