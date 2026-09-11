"""Public GRIM headless API."""
from GRIM_Backend.scripting.cli import _audit_json_default, _parser, audit_dataset, main
from GRIM_Backend.datasets.combine import combine_datasets
from GRIM_Backend.io.loaders import (
    SUPPORTED_EXTENSIONS,
    _matching_dataset_paths,
    is_supported_path,
    load_dataset,
    load_flat_csv,
    load_folder,
    read_CST,
    read_SENTRi,
)
from GRIM_Backend.plotting.modes.isar_mode import form_isar

if __name__ == "__main__":
    raise SystemExit(main())
