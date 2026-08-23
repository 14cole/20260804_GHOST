"""Declarative tool definitions consumed by the GUI."""

from dataclasses import dataclass
from typing import Any, Callable

from .grim_bridge import OUTPUT_EXTENSIONS
from .operations import (
    concatenate_frequencies,
    concatenate_polarizations,
    convert_files,
    rename_files,
    subtract_datasets,
)


@dataclass(frozen=True)
class FieldSpec:
    name: 'str'
    label: 'str'
    kind: 'str'
    required: 'bool' = True
    default: 'Any' = None
    options: 'tuple[str, ...]' = ()


@dataclass(frozen=True)
class ToolSpec:
    identifier: 'str'
    title: 'str'
    description: 'str'
    function: 'Callable[..., Any]'
    fields: 'tuple[FieldSpec, ...]'


class ToolRegistry:
    def __init__(self) -> 'None':
        self._tools: 'dict[str, ToolSpec]' = {}

    def register(self, spec: 'ToolSpec') -> 'None':
        if spec.identifier in self._tools:
            raise ValueError(f"duplicate tool identifier: {spec.identifier}")
        self._tools[spec.identifier] = spec

    def tools(self) -> 'tuple[ToolSpec, ...]':
        return tuple(self._tools.values())


DIR = "directory"
TEXT = "text"
CHECK = "checkbox"
CHOICE = "choice"


def default_registry() -> 'ToolRegistry':
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "subtract", "Subtract Datasets",
        "Coherently subtract the FRD (clean) complex far field from the OPN "
        "(featured) field. One FRD may serve multiple OPN cases when its "
        "parameters are a compatible subset. Inputs require final _OPN/_FRD "
        "markers and preserved solver raw amplitude arrays.",
        subtract_datasets,
        (
            FieldSpec("opn_dir", "OPN library", DIR),
            FieldSpec("frd_dir", "FRD library", DIR),
            FieldSpec("output_dir", "Output folder", DIR),
            FieldSpec("overwrite", "Replace existing output files", CHECK, False, False),
        ),
    ))
    registry.register(ToolSpec(
        "concat_pols", "Concatenate Polarizations",
        "Combine matching GRIM files across the solver's VV/HH/VH/HV, "
        "TM/TE, V/H, or VERTICAL/HORIZONTAL aliases.",
        concatenate_polarizations,
        (
            FieldSpec("input_dir", "Original folder", DIR),
            FieldSpec("output_dir", "Joined_Pols folder", DIR),
            FieldSpec("overwrite", "Replace existing output files", CHECK, False, False),
        ),
    ))
    registry.register(ToolSpec(
        "concat_freqs", "Concatenate Frequencies",
        "Combine matching GRIM files across frequency tokens such as 3GHz.",
        concatenate_frequencies,
        (
            FieldSpec("input_dir", "Original folder", DIR),
            FieldSpec("output_dir", "Joined_Freqs folder", DIR),
            FieldSpec("overwrite", "Replace existing output files", CHECK, False, False),
        ),
    ))
    registry.register(ToolSpec(
        "rename", "Rename Files",
        "Replace a keyword in every matching filename, either in place or into "
        "a separate folder.",
        rename_files,
        (
            FieldSpec("input_dir", "Input folder", DIR),
            FieldSpec("output_dir", "Output folder", DIR, False),
            FieldSpec("keyword", "Keyword", TEXT),
            FieldSpec("replacement", "Replacement", TEXT, False, ""),
            FieldSpec("in_place", "Rename in place", CHECK, False, False),
            FieldSpec("overwrite", "Replace existing output files", CHECK, False, False),
        ),
    ))
    registry.register(ToolSpec(
        "convert", "Convert Files",
        "Convert every recognized GRIM dataset in a folder. .ss is available "
        "as an input but remains unavailable as an output.",
        convert_files,
        (
            FieldSpec("input_dir", "Input folder", DIR),
            FieldSpec("output_dir", "Output folder", DIR),
            FieldSpec("extension", "Output extension", CHOICE, True, ".grim",
                      OUTPUT_EXTENSIONS),
            FieldSpec("overwrite", "Replace existing output files", CHECK, False, False),
        ),
    ))
    return registry
