"""Expose the bundled BLAS thread controls supported by this Python runtime."""
import sys

if sys.version_info >= (3, 9):
    from . import _threadpoolctl as implementation
else:
    from . import _threadpoolctl_py36 as implementation

__version__ = implementation.__version__
threadpool_limits = implementation.threadpool_limits
threadpool_info = implementation.threadpool_info

__all__ = ['threadpool_limits', 'threadpool_info']
