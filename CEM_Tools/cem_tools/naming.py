"""Filename grouping rules for solver and GRIM dataset libraries."""

import re
from pathlib import Path


POLARIZATIONS = {
    "VV", "HH", "VH", "HV", "TM", "TE",
    "V", "H", "VERTICAL", "HORIZONTAL", "CROSS",
}
_FREQUENCY_TOKEN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:GHz)$",
    re.IGNORECASE,
)
_ROLE_TOKEN = re.compile(r"^(?:OPN|FRD)$", re.IGNORECASE)


def _tokens(path: 'str | Path') -> 'list[str]':
    return Path(path).stem.split("_")


def is_polarization_token(token: 'str') -> 'bool':
    return token.upper() in POLARIZATIONS


def is_frequency_token(token: 'str') -> 'bool':
    return bool(_FREQUENCY_TOKEN.fullmatch(token))


def is_role_token(token: 'str') -> 'bool':
    return bool(_ROLE_TOKEN.fullmatch(token))


def group_stem(
    path: 'str | Path',
    remove: 'str',
) -> 'str':
    """Return a stable output/group stem after removing selected tokens."""

    kept: 'list[str]' = []
    for token in _tokens(path):
        drop = False
        if remove in {"polarization", "axes", "design"}:
            drop = drop or is_polarization_token(token)
        if remove in {"frequency", "axes", "design"}:
            drop = drop or is_frequency_token(token)
        if remove == "design":
            drop = drop or is_role_token(token)
        if not drop and token:
            kept.append(token)
    return "_".join(kept) or "joined"


def role_neutral_stem(path: 'str | Path') -> 'str':
    kept = [token for token in _tokens(path) if not is_role_token(token)]
    return "_".join(kept) or "delta"
