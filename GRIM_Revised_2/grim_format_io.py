"""Native archive loading and composition of the RcsGrid format adapters."""
from __future__ import annotations

from dataclasses import replace
import json
import csv
import os
import re
import tempfile
import warnings
import numpy as np


from grim_cst_io import CstFormatMixin
from grim_sentri_io import SentriFormatMixin
from grim_legacy_io import LegacyFormatMixin
from grim_pio_io import PioFormatMixin


class RcsGridFormatMixin(
    CstFormatMixin, SentriFormatMixin, LegacyFormatMixin, PioFormatMixin
):
    """Format adapters inherited by the public RcsGrid data model.

    Model validation helpers are imported at call time to avoid an import
    cycle and retain the existing model-level allocation and format policies.
    """

    @classmethod
    def load(
        cls,
        path,
        mmap_mode: str | None = None,
        *,
        allow_legacy_pickle: bool = False,
        max_output_bytes=None,
    ):
        """Load a grid from a .grim (npz) file.

        Args:
            path: Input path, with or without .grim.
            mmap_mode: Retained for API compatibility. ``.npz`` members cannot
                be memory-mapped; a warning is emitted when this is supplied.
            allow_legacy_pickle: Explicitly opt in to legacy object-array files.
                Never enable this for an untrusted file.
            max_output_bytes: Optional reviewed cap for the exact native NPZ
                payload plus the power/phase sanitation copies. By default one
                load may use at most half of currently available memory (or
                the conservative 2 GiB fallback when memory is unknown).

        Returns:
            RcsGrid instance loaded from disk.
        """
        from grim_dataset import (
            _preflight_native_archive_allocation,
        )
        path = os.fspath(path)
        if not path.casefold().endswith(".grim"):
            path = f"{path}.grim"
        _preflight_native_archive_allocation(
            path,
            allow_legacy_pickle=bool(allow_legacy_pickle),
            max_output_bytes=max_output_bytes,
        )
        if mmap_mode is not None:
            warnings.warn(
                "mmap_mode has no effect for .grim/.npz archives; arrays are loaded eagerly",
                RuntimeWarning,
                stacklevel=2,
            )
        # ``NpzFile`` owns a ZipFile reader in addition to the caller-owned
        # stream.  Close both deterministically: relying on garbage collection
        # can retain archive buffers (and, on Windows, file locks) beyond the
        # lifetime of the returned eager ``RcsGrid``.
        with open(path, "rb") as f, np.load(
            f, allow_pickle=bool(allow_legacy_pickle)
        ) as data:

            units = {}
            if "units" in data:
                raw_units = data["units"]
                if isinstance(raw_units, np.ndarray):
                    raw_units = raw_units.item()
                if isinstance(raw_units, bytes):
                    raw_units = raw_units.decode("utf-8")
                if isinstance(raw_units, str) and raw_units:
                    try:
                        units = json.loads(raw_units)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path} contains corrupt units metadata; refusing to "
                            "guess frequency, RCS, or angular conventions"
                        ) from exc
                elif isinstance(raw_units, dict):
                    units = raw_units
                if not isinstance(units, dict):
                    raise ValueError(
                        f"{path} contains invalid units metadata (expected a JSON object)"
                    )

            source_path_raw = data["source_path"].item() if "source_path" in data else None
            source_path = source_path_raw if source_path_raw else None
            history_raw = data["history"].item() if "history" in data else None
            history = history_raw if history_raw else None
            required = ("azimuths", "elevations", "frequencies", "polarizations", "rcs_power", "rcs_phase")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(
                    f"{path} is not a supported .grim file (missing keys: {', '.join(missing)})"
                )

            # Load only the raw-field members needed for consistency checking
            # before validating the core payload.  Large independent ancillary
            # meshes/profiles remain unopened until the required axes and RCS
            # grids have passed validation.
            raw_extra = {
                key: data[key]
                for key in (
                    "rcs_amp_real",
                    "rcs_amp_imag",
                    "raw_complex_amplitude_preserved",
                )
                if key in data
            }

            (
                azimuths,
                elevations,
                frequencies,
                polarizations,
                rcs_power,
                rcs_phase,
                units,
            ) = cls._validate_native_payload(
                path=path,
                azimuths=data["azimuths"],
                elevations=data["elevations"],
                frequencies=data["frequencies"],
                polarizations=data["polarizations"],
                rcs_power=data["rcs_power"],
                rcs_phase=data["rcs_phase"],
                units=units,
                extra=raw_extra,
            )

            # Keys this class does not model (including the already-validated
            # raw complex pair and solver provenance) ride along in ``extra``
            # so save() can put them back -- see _extra_to_write.
            extra = {
                key: (
                    raw_extra[key]
                    if key in raw_extra
                    else data[key]
                )
                for key in getattr(data, "files", [])
                if key not in cls._RESERVED_KEYS
            }

            return cls(
                azimuths,
                elevations,
                frequencies,
                polarizations,
                rcs_power=rcs_power,
                rcs_phase=rcs_phase,
                rcs_domain="power_phase",
                source_path=source_path,
                history=history,
                units=units,
                extra=extra,
            )
