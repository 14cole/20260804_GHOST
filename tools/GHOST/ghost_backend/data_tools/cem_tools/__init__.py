"""Headless computational-electromagnetics dataset utilities."""

from .operations import (
    BatchResult,
    concatenate_frequencies,
    concatenate_polarizations,
    convert_files,
    rename_files,
    subtract_datasets,
)

__all__ = [
    "BatchResult",
    "concatenate_frequencies",
    "concatenate_polarizations",
    "convert_files",
    "rename_files",
    "subtract_datasets",
]
