"""Dataset format dispatch and deterministic folder loading."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import glob
import os

from GRIM_Backend.datasets.combine import combine_datasets
from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.coordinates import canonical_angular_coordinate_system
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.csv import has_flat_csv_signature, load_flat_csv as _load_flat_csv_schema


SUPPORTED_EXTENSIONS = (
    ".grim",
    ".csv",
    ".cst_data",
    ".txt",
    ".dat",
    ".asc",
    ".ascii",
    ".tsv",
    ".out",
    ".pio",
    ".cmplx_di",
    ".ptm",
    ".ss",
)


class UnrecognizedTableError(ValueError):
    """No standard reader recognized a text table; the GUI may offer mapping."""


def is_supported_path(path: str) -> bool:
    return str(path).lower().endswith(SUPPORTED_EXTENSIONS)


def load_flat_csv(path: str) -> RcsGrid:
    """Load a supported flat RCS table into an RcsGrid."""

    return _load_flat_csv_schema(
        path,
        grid_class=RcsGrid,
        canonical_angular_coordinate_system=canonical_angular_coordinate_system,
        c0=C0,
    )


def read_CST(path: str) -> RcsGrid:
    """Read a supported CST far-field table.

    Accepts wide theta/phi CSV and row-oriented ``.cst_data`` tables.
    """
    return RcsGrid.read_CST(path)


def read_SENTRi(path: str) -> RcsGrid:
    """Read a CREATE-RF SENTRi RCS table with its vendor conventions."""

    return RcsGrid.read_SENTRi(path)


def load_dataset(path: str, *, allow_legacy_pickle=False) -> RcsGrid:
    """Load any GRIM-supported dataset without importing Qt."""
    path = str(path)
    lower = path.lower()
    if lower.endswith(".grim"):
        return RcsGrid.load(path, allow_legacy_pickle=allow_legacy_pickle)
    if lower.endswith(".out"):
        return RcsGrid.load_out(path)
    if lower.endswith(".ss"):
        return RcsGrid.load_ss(path)
    if lower.endswith(".ptm"):
        return RcsGrid.load_ptm(path)
    if lower.endswith((".pio", ".cmplx_di")):
        return RcsGrid.load_pio(path)
    if lower.endswith(".cst_data"):
        return read_CST(path)
    if lower.endswith((".csv", ".txt")) and RcsGrid.has_SENTRi_signature(path):


        return read_SENTRi(path)
    if lower.endswith((".csv", ".txt")) and has_flat_csv_signature(path):


        return load_flat_csv(path)
    loaders = (
        (load_flat_csv, read_CST)
        if lower.endswith(".csv")
        else (
            RcsGrid.load_theta_phi_txt,
            load_flat_csv,
            read_CST,
        )
    )
    errors = []
    for loader in loaders:
        try:
            return loader(path)
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    error_type = (UnrecognizedTableError if lower.endswith(
        (".csv", ".txt", ".dat", ".asc", ".ascii", ".tsv")) else ValueError)
    raise error_type("; ".join(errors))


def _matching_dataset_paths(folder: str, *, pattern="*", recursive=False):
    search = (
        os.path.join(str(folder), "**", pattern)
        if recursive
        else os.path.join(str(folder), pattern)
    )
    paths = [
        path
        for path in sorted(glob.glob(search, recursive=recursive))
        if os.path.isfile(path) and is_supported_path(path)
    ]
    if not paths:
        raise ValueError(f"no supported datasets matched {search!r}")
    return paths


def load_folder(
    folder: str,
    *,
    pattern="*",
    recursive=False,
    operation="join",
    workers=1,
    overlap="error",
    max_output_bytes=None,
    coherent_metadata_attested=False,
    stitch_policy="priority-first",
    tol=1.0e-6,
):
    """Load matching files and combine them in deterministic pathname order."""
    paths = _matching_dataset_paths(
        folder, pattern=pattern, recursive=recursive
    )
    workers = max(1, int(workers))
    if workers == 1:
        grids = [load_dataset(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            grids = list(pool.map(load_dataset, paths))
    return combine_datasets(
        grids,
        operation,
        overlap=overlap,
        max_output_bytes=max_output_bytes,
        coherent_metadata_attested=coherent_metadata_attested,
        stitch_policy=stitch_policy,
        tol=tol,
    )
