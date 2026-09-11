"""Locate the source and native-artifact root for this backend."""
from pathlib import Path


def backend_root():
    """Return the Backend directory containing drivers and native libraries."""
    return Path(__file__).resolve().parent.parent
