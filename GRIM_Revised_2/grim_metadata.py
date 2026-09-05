"""Scalar convention metadata interpretation, independent of datasets and views.

Inspection preserves evidence and its state. The scalar compatibility adapter
retains the existing advisory/strict policy; callers choose how to present it.
No function here changes field values or guesses a physical transformation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re

import numpy as np


ADVISORY_METADATA_KEYS = frozenset({
    "amplitude_version", "phase_reference", "time_convention",
    "polarization_basis", "amplitude_convention", "complex_field_domain",
})


def canonical_time_convention(value) -> str:
    compact = (
        str(value or "").strip().casefold()
        .replace("ω", "omega").replace("*", "").replace(" ", "")
    )
    if re.search(r"exp\(\+?j(?:omega|w)t\)", compact):
        return "+jwt"
    if re.search(r"exp\(-j(?:omega|w)t\)", compact):
        return "-jwt"
    return compact


@dataclass(frozen=True)
class ScalarMetadata:
    key: str
    declarations: tuple[str, ...]
    sources: tuple[str, ...]
    malformed_sources: tuple[str, ...]
    conflicting: bool

    @property
    def status(self) -> str:
        if self.malformed_sources:
            return "malformed"
        if self.conflicting:
            return "conflicting"
        return "consistent" if self.declarations else "missing"

    def scalar(self, *, advisory: bool) -> str:
        """Keep legacy eligibility behavior while inspection stays lossless."""
        if self.malformed_sources and not advisory:
            raise ValueError(f"metadata {self.key!r} must be scalar")
        if self.conflicting:
            if advisory:
                return ""
            raise ValueError(f"dataset contains contradictory {self.key} metadata")
        return self.declarations[0] if self.declarations else ""


def inspect_scalar_metadata(
    key: str,
    units: Mapping | None,
    extra: Mapping | None,
    *,
    canonicalizer: Callable[[str], str] | None = None,
) -> ScalarMetadata:
    declarations, sources, malformed = [], [], []
    for name, container in (("units", units or {}), ("extra", extra or {})):
        if key not in container:
            continue
        raw = np.asarray(container[key])
        if raw.size != 1:
            malformed.append(name)
            continue
        value = raw.reshape(-1)[0]
        if isinstance(value, np.generic):
            value = value.item()
        # False and zero are evidence, particularly for motion compensation.
        text = "" if value is None else str(value).strip()
        if text:
            declarations.append(text)
            sources.append(name)
    normalize = canonicalizer or (
        canonical_time_convention if key == "time_convention"
        else lambda value: " ".join(value.split()).casefold()
    )
    return ScalarMetadata(
        key, tuple(declarations), tuple(sources), tuple(malformed),
        len({normalize(value) for value in declarations}) > 1,
    )
