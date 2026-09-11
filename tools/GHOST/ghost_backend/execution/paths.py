"""Locate the source and native-artifact root for this backend."""
from pathlib import Path


def backend_root():
    """Return the GHOST package directory containing the run scripts."""
    return Path(__file__).resolve().parent.parent


def native_kernel_root():
    """Return the directory containing BoR native source and libraries."""
    return backend_root() / 'bor' / 'native'
