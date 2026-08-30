from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from PySide6.QtCore import QByteArray, QMimeData, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QFont, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QRadioButton,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from grim_headless import is_supported_path

# MIME sent FROM the Datasets table TO the tree (single loaded dataset)
MIME_DATASET = "application/x-grim-dataset"

# MIME sent FROM the tree TO the Datasets table (branch drag marker)
MIME_BRANCH = "application/x-grim-branch"

# QTreeWidgetItem data roles
_ROLE_TYPE    = Qt.UserRole        # "root" | "branch" | "leaf"
_ROLE_NAME    = Qt.UserRole + 1    # dataset name string (leaves only)
_ROLE_GRID    = Qt.UserRole + 2    # RcsGrid object (leaves only, may be None)
_ROLE_MODE    = Qt.UserRole + 3    # "auto" | "coh" | "incoh" (non-roots)
_ROLE_PURPOSE = Qt.UserRole + 4    # "response" | "preview"
_ROLE_PREVIEW_KEY = Qt.UserRole + 5  # stable runtime scene key (preview only)

# Tree columns. ``Use`` controls response membership; ``Show`` is deliberately
# preview-only so hiding visual clutter can never silently change a result.
_COLUMN_NAME       = 0
_COLUMN_MODE       = 1
_COLUMN_INCLUDED   = 2
_COLUMN_VISIBILITY = 3

_TYPE_ROOT   = "root"
_TYPE_BRANCH = "branch"
_TYPE_LEAF   = "leaf"
_TYPE_PREVIEW_ROOT = "preview_root"
_TYPE_PREVIEW_GROUP = "preview_group"
_TYPE_PREVIEW_ITEM = "preview_item"

_PURPOSE_RESPONSE = "response"
_PURPOSE_PREVIEW = "preview"

_MODE_COH    = "coh"
_MODE_INCOH  = "incoh"
_MODE_AUTO   = "auto"
_DEFAULT_LEAF_MODE = _MODE_INCOH
_DEFAULT_BRANCH_MODE = _MODE_AUTO

_MODE_LABEL = {
    _MODE_AUTO: "auto",
    _MODE_COH: "field +",
    _MODE_INCOH: "power +",
}
_MODE_COLOUR = {
    _MODE_AUTO: QColor("#9ca3af"),
    _MODE_COH: QColor("#3b82f6"),
    _MODE_INCOH: QColor("#f59e0b"),
}

# Dataset-side response semantics.  ``combine_role`` describes whether the
# stored phase may participate in a field sum.  ``assembly_response_role``
# distinguishes a complete body response from a feature-only delta so a
# downstream tree cannot accidentally count the same clean body twice.
_COMBINE_ROLE_KEY = "combine_role"
_RESPONSE_ROLE_KEY = "assembly_response_role"
_ASSEMBLY_PROVENANCE_KEY = "assembly_provenance_json"
_FEATURE_PROVENANCE_KEY = "feature_provenance_json"
_SOURCE_MONOSTATIC_SHA256_KEY = "source_monostatic_sha256"
_ASSEMBLY_BASE_SHA256_KEY = "assembly_base_sha256"
_ASSEMBLY_BASE_RESPONSE_SHA256_KEY = "assembly_base_response_sha256"

_COMBINE_ROLE_COHERENT = "coherent"
_COMBINE_ROLE_POWER = "power"
_RESPONSE_ROLE_BODY_PLUS_FEATURES = "body_plus_features"
_RESPONSE_ROLE_FEATURES_ONLY_DELTA = "features_only_delta"
_RESPONSE_ROLE_COHERENT_SUM = "coherent_field_sum"
_RESPONSE_ROLE_INCOHERENT_SUM = "incoherent_power_sum"
_RESPONSE_ROLE_MIXED_SUM = "coherent_field_plus_incoherent_power"
_COHERENT_VEHICLE_FRAME_ATTESTATION = (
    "Every Field + response was solved or translated into one vehicle frame "
    "with a common phase center, common attitude, and common earth V/H basis."
)
_KNOWN_RESPONSE_ROLES = {
    _RESPONSE_ROLE_BODY_PLUS_FEATURES,
    _RESPONSE_ROLE_FEATURES_ONLY_DELTA,
    _RESPONSE_ROLE_COHERENT_SUM,
    _RESPONSE_ROLE_INCOHERENT_SUM,
    _RESPONSE_ROLE_MIXED_SUM,
}

# Scalar metadata that remains meaningful after axis alignment and response
# addition.  Source-only solver certificates and shape-dependent arrays are
# deliberately absent.  A key is copied to a multi-input result only when all
# inputs declare the same value; explicit disagreements are errors for the
# coordinate/convention keys below rather than silently discarded evidence.
_SEMANTIC_SCALAR_KEYS = (
    "phase_reference",
    "time_convention",
    "polarization_basis",
    "amplitude_convention",
    "complex_field_domain",
    "source_format",
    "assembly_angular_coordinate_contract",
    "elevation_coordinate_convention",
    "sentri_elevation_convention",
    "sentri_coordinate_mapping",
    "sentri_polarization_mapping",
    "sentri_phase_mapping",
    "sentri_zero_360_seam_policy",
    "sentri_zero_360_precedence_used",
    "sentri_signed_180_seam_policy",
    "sentri_signed_180_precedence_used",
    "sentri_units_row_present",
)
_STRICT_SEMANTIC_KEYS = set(_SEMANTIC_SCALAR_KEYS) - {"source_format"}
_DYNAMIC_RESULT_METADATA_KEYS = {
    _COMBINE_ROLE_KEY,
    "combine_role_note",
    _RESPONSE_ROLE_KEY,
    _ASSEMBLY_PROVENANCE_KEY,
    _FEATURE_PROVENANCE_KEY,
    _SOURCE_MONOSTATIC_SHA256_KEY,
    _ASSEMBLY_BASE_SHA256_KEY,
    _ASSEMBLY_BASE_RESPONSE_SHA256_KEY,
    "coherent_metadata_attestation_json",
    "response_role_validation",
    "rcs_amp_real",
    "rcs_amp_imag",
    "raw_complex_amplitude_preserved",
}


def _item_purpose(item: QTreeWidgetItem) -> str:
    """Return response for legacy nodes that predate the purpose role."""
    purpose = item.data(0, _ROLE_PURPOSE)
    return _PURPOSE_PREVIEW if purpose == _PURPOSE_PREVIEW else _PURPOSE_RESPONSE


def _is_preview_item(item: QTreeWidgetItem | None) -> bool:
    return item is not None and _item_purpose(item) == _PURPOSE_PREVIEW


def _preview_key(item: QTreeWidgetItem) -> str | None:
    value = item.data(0, _ROLE_PREVIEW_KEY)
    return value if isinstance(value, str) and value else None


def _preview_icon_type(node_type: str) -> str:
    return {
        _TYPE_PREVIEW_ROOT: _TYPE_ROOT,
        _TYPE_PREVIEW_GROUP: _TYPE_BRANCH,
        _TYPE_PREVIEW_ITEM: _TYPE_LEAF,
    }.get(node_type, node_type)


def _visibility_state(item: QTreeWidgetItem):
    """Return the item's explicit preview check state."""
    return item.checkState(_COLUMN_VISIBILITY)


def _set_visibility_state(item: QTreeWidgetItem, visible: bool) -> None:
    """Set one item's preview state without changing its descendants."""
    item.setCheckState(
        _COLUMN_VISIBILITY,
        Qt.Checked if bool(visible) else Qt.Unchecked,
    )


def _set_subtree_visibility(item: QTreeWidgetItem, visible: bool) -> None:
    """Apply one preview state to a node and every descendant."""
    _set_visibility_state(item, visible)
    for index in range(item.childCount()):
        _set_subtree_visibility(item.child(index), visible)


def _subtree_items(item: QTreeWidgetItem) -> list[QTreeWidgetItem]:
    """Return a node and its descendants in stable pre-order."""
    items = [item]
    for index in range(item.childCount()):
        items.extend(_subtree_items(item.child(index)))
    return items


def _sync_visibility_from_children(item: QTreeWidgetItem | None) -> None:
    """Make each ancestor's checkbox summarize the states of its children."""
    current = item
    while current is not None:
        states = [
            _visibility_state(current.child(index))
            for index in range(current.childCount())
        ]
        if states:
            if all(state == Qt.Checked for state in states):
                state = Qt.Checked
            elif all(state == Qt.Unchecked for state in states):
                state = Qt.Unchecked
            else:
                state = Qt.PartiallyChecked
            current.setCheckState(_COLUMN_VISIBILITY, state)
        current = current.parent()


def _item_preview_visible(item: QTreeWidgetItem) -> bool:
    """Whether ``item`` is visible in a preview, including ancestor masks.

    Leaves normally have only checked/unchecked states. A partially checked
    branch remains an active container because at least one descendant is
    visible. This helper is intentionally unrelated to physical inclusion in
    :func:`build_assembly_grid`.
    """
    current = item
    while current is not None:
        if _visibility_state(current) == Qt.Unchecked:
            return False
        current = current.parent()
    return True


def _inclusion_state(item: QTreeWidgetItem):
    """Return one response node's explicit Build Platform check state."""
    return item.checkState(_COLUMN_INCLUDED)


def _set_inclusion_state(item: QTreeWidgetItem, included: bool) -> None:
    """Set one response node's build state without changing descendants."""
    if _is_preview_item(item):
        return
    item.setCheckState(
        _COLUMN_INCLUDED,
        Qt.Checked if bool(included) else Qt.Unchecked,
    )


def _response_subtree_items(item: QTreeWidgetItem) -> list[QTreeWidgetItem]:
    """Return response nodes in a subtree, excluding runtime preview nodes."""
    if _is_preview_item(item):
        return []
    items = [item]
    for index in range(item.childCount()):
        items.extend(_response_subtree_items(item.child(index)))
    return items


def _set_subtree_included(item: QTreeWidgetItem, included: bool) -> None:
    """Apply one Build Platform inclusion state to a response subtree."""
    for descendant in _response_subtree_items(item):
        _set_inclusion_state(descendant, included)


def _sync_inclusion_from_children(item: QTreeWidgetItem | None) -> None:
    """Make response ancestors summarize their response children's Use state."""
    current = item
    while current is not None:
        children = [
            current.child(index)
            for index in range(current.childCount())
            if not _is_preview_item(current.child(index))
        ]
        states = [_inclusion_state(child) for child in children]
        if states:
            if all(state == Qt.Checked for state in states):
                state = Qt.Checked
            elif all(state == Qt.Unchecked for state in states):
                state = Qt.Unchecked
            else:
                state = Qt.PartiallyChecked
            current.setCheckState(_COLUMN_INCLUDED, state)
        current = current.parent()


def _item_response_included(item: QTreeWidgetItem) -> bool:
    """Whether a response node participates in builds, including ancestors."""
    if _is_preview_item(item):
        return False
    current = item
    while current is not None:
        if _inclusion_state(current) == Qt.Unchecked:
            return False
        current = current.parent()
    return True


def _inherit_container_states(
    item: QTreeWidgetItem, parent: QTreeWidgetItem | None
) -> None:
    """Keep a newly attached subtree masked when its container is off."""
    if parent is None:
        return
    if _visibility_state(parent) == Qt.Unchecked:
        _set_subtree_visibility(item, False)
    if not _is_preview_item(item) and _inclusion_state(parent) == Qt.Unchecked:
        _set_subtree_included(item, False)


def _apply_mode_badge(item: QTreeWidgetItem, mode: str | None) -> None:
    """Paint the mode column for a node. Root nodes have no parent, so they
    have no add-mode — their second column stays blank."""
    if mode is None:
        item.setText(1, "")
        item.setForeground(1, QBrush())
        return
    item.setText(1, _MODE_LABEL.get(mode, ""))
    item.setForeground(1, QBrush(_MODE_COLOUR.get(mode, QColor("#9ca3af"))))
    if item.data(0, _ROLE_TYPE) == _TYPE_BRANCH:
        item.setToolTip(
            _COLUMN_MODE,
            (
                "Auto: this branch enters its parent using the combine_role "
                "of the response built inside it."
                if mode == _MODE_AUTO
                else "Explicit branch override; use Auto to adopt the built "
                "response's declared combine_role."
            ),
        )
    font = item.font(1)
    font.setBold(True)
    item.setFont(1, font)


def _set_node_mode(item: QTreeWidgetItem, mode: str) -> None:
    """Persist a node's add-mode and refresh its badge. No-op for roots."""
    if _is_preview_item(item) or item.data(0, _ROLE_TYPE) == _TYPE_ROOT:
        item.setData(0, _ROLE_MODE, None)
        _apply_mode_badge(item, None)
        return
    node_type = item.data(0, _ROLE_TYPE)
    allowed = (
        (_MODE_AUTO, _MODE_COH, _MODE_INCOH)
        if node_type == _TYPE_BRANCH
        else (_MODE_COH, _MODE_INCOH)
    )
    if mode not in allowed:
        mode = (
            _DEFAULT_BRANCH_MODE
            if node_type == _TYPE_BRANCH
            else _DEFAULT_LEAF_MODE
        )
    item.setData(0, _ROLE_MODE, mode)
    _apply_mode_badge(item, mode)


def _node_mode(item: QTreeWidgetItem) -> str:
    """Read one persisted add mode with type-appropriate safe defaults."""
    if _is_preview_item(item):
        raise ValueError("preview-only nodes do not have an assembly add mode")
    raw = item.data(0, _ROLE_MODE)
    node_type = item.data(0, _ROLE_TYPE)
    if raw in (_MODE_COH, _MODE_INCOH):
        return raw
    if node_type == _TYPE_BRANCH and raw == _MODE_AUTO:
        return raw
    return (
        _DEFAULT_BRANCH_MODE
        if node_type == _TYPE_BRANCH
        else _DEFAULT_LEAF_MODE
    )


def _scalar_metadata_value(raw, *, key: str):
    """Return one JSON-safe scalar metadata value or reject malformed data."""

    array = np.asarray(raw)
    if array.size != 1:
        raise ValueError(f"metadata {key!r} must contain exactly one value")
    value = array.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"metadata {key!r} is not valid UTF-8") from exc
    if not isinstance(value, (str, bool, int, float)) and value is not None:
        raise ValueError(f"metadata {key!r} must be a scalar string/number/bool")
    return value


def _canonical_metadata_value(key: str, value):
    """Canonical comparison form without inventing a declaration."""

    if isinstance(value, str):
        text = " ".join(value.split())
        if key == "time_convention":
            # RcsGrid owns the public convention normalizer.  This local form
            # handles the two spellings Assembly itself emits without making
            # the tree dependent on a private instance merely to compare text.
            compact = (
                text.casefold()
                .replace("ω", "omega")
                .replace("*", "")
                .replace(" ", "")
            )
            for positive in ("exp(+jomegat)", "exp(jomegat)", "exp(+jwt)", "exp(jwt)"):
                if positive in compact:
                    return "+jwt"
            for negative in ("exp(-jomegat)", "exp(-jwt)"):
                if negative in compact:
                    return "-jwt"
        return text.casefold()
    return value


def _declared_extra_scalar(grid, key: str):
    """Read one scalar from units/extra and refuse internal contradictions."""

    declarations = []
    for container in (getattr(grid, "units", None) or {}, getattr(grid, "extra", None) or {}):
        if key in container:
            declarations.append(_scalar_metadata_value(container[key], key=key))
    nonblank = [
        value for value in declarations
        if not isinstance(value, str) or bool(value.strip())
    ]
    canonical = {_canonical_metadata_value(key, value) for value in nonblank}
    if len(canonical) > 1:
        raise ValueError(f"dataset contains contradictory {key} metadata")
    return nonblank[0] if nonblank else None


def _declared_combine_role(grid) -> str | None:
    raw = _declared_extra_scalar(grid, _COMBINE_ROLE_KEY)
    if raw is None:
        return None
    role = str(raw).strip().lower()
    if role not in {_COMBINE_ROLE_COHERENT, _COMBINE_ROLE_POWER}:
        raise ValueError(
            f"unknown combine_role {role!r}; expected 'coherent' or 'power'"
        )
    return role


def _declared_response_role(grid) -> str | None:
    raw = _declared_extra_scalar(grid, _RESPONSE_ROLE_KEY)
    if raw is None:
        return None
    role = str(raw).strip().lower()
    if role not in _KNOWN_RESPONSE_ROLES:
        raise ValueError(
            f"unknown assembly_response_role {role!r}; file must be regenerated "
            "or its response semantics reviewed"
        )
    return role


def _mode_for_grid(grid) -> str:
    """Adopt an explicit file role; untyped responses default to power-add."""

    role = _declared_combine_role(grid) if grid is not None else None
    return _MODE_COH if role == _COMBINE_ROLE_COHERENT else _MODE_INCOH


def _resolved_node_mode(item: QTreeWidgetItem, built_grid) -> str:
    """Resolve branch Auto from its built response; leaves never use Auto."""

    mode = _node_mode(item)
    if mode != _MODE_AUTO:
        return mode
    if item.data(0, _ROLE_TYPE) != _TYPE_BRANCH:
        return _DEFAULT_LEAF_MODE
    return _mode_for_grid(built_grid)


def _validate_hash(value, *, context: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{context} must be a 64-character SHA-256 hex digest")
    return digest


def _declared_base_response_hash(grid) -> str | None:
    raw = _declared_extra_scalar(grid, _ASSEMBLY_BASE_RESPONSE_SHA256_KEY)
    if raw is None:
        return None
    return _validate_hash(raw, context=_ASSEMBLY_BASE_RESPONSE_SHA256_KEY)


def _assembly_response_physics_sha256(grid) -> str:
    """Match GHOST's package-independent clean-response physics digest."""

    channel_indices = {}
    for index, raw in enumerate(np.asarray(grid.polarizations).ravel()):
        value = str(raw).strip().upper()
        canonical = (
            "VV" if value in {"VV", "V", "VERTICAL"}
            else "HH" if value in {"HH", "H", "HORIZONTAL"}
            else "VH" if value in {"VH", "HV"}
            else value
        )
        if canonical not in {"VV", "HH", "VH"}:
            raise ValueError(
                "clean Assembly base response physics digest requires exactly "
                "VV, HH, and reciprocal VH/HV channels"
            )
        if canonical in channel_indices:
            raise ValueError(
                f"duplicate polarization alias for {canonical} in Assembly base"
            )
        channel_indices[canonical] = index
    if set(channel_indices) != {"VV", "HH", "VH"}:
        raise ValueError(
            "clean Assembly base response physics digest requires exactly VV, "
            "HH, and reciprocal VH/HV channels"
        )
    channels = ["VV", "HH", "VH"]
    order = [channel_indices[channel] for channel in channels]

    real = (grid.extra or {}).get("rcs_amp_real")
    imag = (grid.extra or {}).get("rcs_amp_imag")
    if real is None or imag is None:
        raise ValueError(
            "matching a clean body to a features-only delta requires its "
            "authoritative rcs_amp_real/rcs_amp_imag arrays"
        )
    real = np.asarray(real, dtype=np.float64)
    imag = np.asarray(imag, dtype=np.float64)
    if real.shape != grid.rcs_power.shape or imag.shape != grid.rcs_power.shape:
        raise ValueError(
            "clean body raw amplitude arrays do not match its response grid"
        )
    amplitude = np.asarray(real + 1j * imag, dtype=np.complex128)[..., order]

    digest = hashlib.sha256()
    digest.update(b"ghost.assembly-base-response-physics-v1\0")

    def update_array(name, value, dtype):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(str(name).encode("utf-8") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(memoryview(array).cast("B"))

    update_array("azimuths", grid.azimuths, "<f8")
    update_array("elevations", grid.elevations, "<f8")
    update_array("frequencies", grid.frequencies, "<f8")
    update_array("polarizations", np.asarray(channels, dtype="U2"), "<U2")
    update_array("complex_amplitude", amplitude, "<c16")
    for key in (
        "phase_reference",
        "amplitude_convention",
        "complex_field_domain",
    ):
        raw = _declared_extra_scalar(grid, key)
        if raw is None:
            raise ValueError(
                f"clean Assembly base is missing required {key} metadata"
            )
        encoded = str(raw).encode("utf-8")
        digest.update(key.encode("ascii") + b"\0")
        digest.update(len(encoded).to_bytes(8, "little") + encoded)
    return digest.hexdigest()


def _json_metadata(grid, key: str):
    raw = _declared_extra_scalar(grid, key)
    if raw is None:
        return None
    try:
        return json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"metadata {key!r} is not valid JSON") from exc


def _hashes_from_record(record, key: str, *, context: str) -> set[str]:
    """Read a scalar-or-list hash field from one durable provenance record."""

    if record is None:
        return set()
    if not isinstance(record, dict):
        raise ValueError("assembly_provenance_json must decode to an object")
    raw_values = record.get(key, [])
    if raw_values is None:
        return set()
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    if not isinstance(raw_values, list):
        raise ValueError(f"assembly provenance {key} must be a list")
    return {
        _validate_hash(value, context=context) for value in raw_values
    }


def _feature_provenance_source_hashes(grid) -> tuple[set[str], bool]:
    """Return feature source hashes and whether legacy total semantics exist."""

    feature_records = _json_metadata(grid, _FEATURE_PROVENANCE_KEY)
    if feature_records is None:
        return set(), False
    if not isinstance(feature_records, list) or not all(
        isinstance(record, dict) for record in feature_records
    ):
        raise ValueError(
            "feature_provenance_json must decode to a list of objects"
        )
    hashes = set()
    legacy_body_total = False
    for record in feature_records:
        raw_hash = record.get(_SOURCE_MONOSTATIC_SHA256_KEY)
        if raw_hash is not None:
            hashes.add(
                _validate_hash(
                    raw_hash, context="feature provenance source hash"
                )
            )
        if (
            str(record.get("schema", "")).strip()
            == "ghost.workflow.coherent-feature-addition.v1"
        ):
            legacy_body_total = True
    return hashes, legacy_body_total


def _body_plus_features_identities(grid) -> tuple[set[str], set[str]]:
    """Return source-file and physical-response IDs already containing a body."""

    role = _declared_response_role(grid)
    direct_sources = set()
    for key in (_SOURCE_MONOSTATIC_SHA256_KEY, _ASSEMBLY_BASE_SHA256_KEY):
        raw = _declared_extra_scalar(grid, key)
        if raw is not None:
            direct_sources.add(_validate_hash(raw, context=key))
    if len(direct_sources) > 1:
        raise ValueError(
            "dataset has contradictory source_monostatic_sha256 and "
            "assembly_base_sha256 metadata"
        )
    direct_response = _declared_base_response_hash(grid)
    direct_responses = {direct_response} if direct_response else set()

    assembly_record = _json_metadata(grid, _ASSEMBLY_PROVENANCE_KEY)
    assembly_sources = _hashes_from_record(
        assembly_record,
        "body_plus_features_source_sha256",
        context="assembly provenance source hash",
    )
    assembly_responses = _hashes_from_record(
        assembly_record,
        "body_plus_features_base_response_sha256",
        context="assembly provenance body-response hash",
    )
    # Records written before dual-identity tracking used this field for every
    # base-response context. It is body occupancy only when the same record
    # also says a body is physically present.
    if assembly_sources and not assembly_responses:
        assembly_responses.update(
            _hashes_from_record(
                assembly_record,
                "assembly_base_response_sha256",
                context="legacy assembly base-response hash",
            )
        )

    feature_sources, legacy_body_total = (
        _feature_provenance_source_hashes(grid)
    )
    if role == _RESPONSE_ROLE_FEATURES_ONLY_DELTA:
        return set(), set()

    if role == _RESPONSE_ROLE_BODY_PLUS_FEATURES:
        sources = direct_sources | assembly_sources | feature_sources
        responses = direct_responses | assembly_responses
        if len(sources) != 1 or len(responses) != 1:
            raise ValueError(
                "body_plus_features response must identify exactly one source "
                "file SHA-256 and one assembly base-response SHA-256"
            )
        return sources, responses
    if role in {
        _RESPONSE_ROLE_COHERENT_SUM,
        _RESPONSE_ROLE_INCOHERENT_SUM,
        _RESPONSE_ROLE_MIXED_SUM,
    }:
        return assembly_sources, assembly_responses
    if legacy_body_total:
        if len(feature_sources) != 1:
            raise ValueError(
                "legacy body-plus-features provenance must identify exactly "
                "one source body hash"
            )
        return feature_sources, direct_responses
    # A direct source hash without an explicit delta role fails safe: it may be
    # a legacy total, and treating it as a delta could reintroduce the body.
    return direct_sources | assembly_sources, direct_responses | assembly_responses


def _assembly_base_reference_hashes(grid) -> set[str]:
    """Return every source-file base context, including feature-only deltas."""

    hashes: set[str] = set()
    for key in (_ASSEMBLY_BASE_SHA256_KEY, _SOURCE_MONOSTATIC_SHA256_KEY):
        raw = _declared_extra_scalar(grid, key)
        if raw is not None:
            hashes.add(_validate_hash(raw, context=key))
    feature_hashes, _legacy_total = _feature_provenance_source_hashes(grid)
    hashes.update(feature_hashes)
    assembly_record = _json_metadata(grid, _ASSEMBLY_PROVENANCE_KEY)
    for key in (
        "assembly_base_sha256",
        "body_plus_features_source_sha256",
        "feature_delta_base_sha256",
    ):
        hashes.update(
            _hashes_from_record(
                assembly_record,
                key,
                context=f"assembly provenance {key}",
            )
        )
    role = _declared_response_role(grid)
    if role in {
        _RESPONSE_ROLE_BODY_PLUS_FEATURES,
        _RESPONSE_ROLE_FEATURES_ONLY_DELTA,
    } and len(hashes) != 1:
        raise ValueError(
            f"{role} response must identify exactly one consistent "
            "assembly_base_sha256"
        )
    return hashes


def _assembly_base_response_reference_hashes(grid) -> set[str]:
    """Return every physical clean-response base context carried by a grid."""

    hashes = set()
    direct = _declared_base_response_hash(grid)
    if direct is not None:
        hashes.add(direct)
    assembly_record = _json_metadata(grid, _ASSEMBLY_PROVENANCE_KEY)
    for key in (
        "assembly_base_response_sha256",
        "body_plus_features_base_response_sha256",
        "feature_delta_base_response_sha256",
    ):
        hashes.update(
            _hashes_from_record(
                assembly_record,
                key,
                context=f"assembly provenance {key}",
            )
        )
    role = _declared_response_role(grid)
    if role in {
        _RESPONSE_ROLE_BODY_PLUS_FEATURES,
        _RESPONSE_ROLE_FEATURES_ONLY_DELTA,
    } and len(hashes) != 1:
        raise ValueError(
            f"{role} response must identify exactly one consistent "
            "assembly_base_response_sha256"
        )
    return hashes


def _feature_delta_context(grid) -> tuple[set[str], set[str], bool]:
    """Return durable feature-delta base context and delta-only status."""

    role = _declared_response_role(grid)
    if role == _RESPONSE_ROLE_FEATURES_ONLY_DELTA:
        return (
            _assembly_base_reference_hashes(grid),
            _assembly_base_response_reference_hashes(grid),
            True,
        )
    assembly_record = _json_metadata(grid, _ASSEMBLY_PROVENANCE_KEY)
    sources = _hashes_from_record(
        assembly_record,
        "feature_delta_base_sha256",
        context="assembly feature-delta source hash",
    )
    responses = _hashes_from_record(
        assembly_record,
        "feature_delta_base_response_sha256",
        context="assembly feature-delta base-response hash",
    )
    delta_only = False
    if assembly_record is not None:
        raw_delta_only = assembly_record.get("feature_delta_only", False)
        if type(raw_delta_only) is not bool:
            raise ValueError(
                "assembly provenance feature_delta_only must be boolean"
            )
        delta_only = raw_delta_only
    if (
        not sources
        and not responses
        and role in {
            _RESPONSE_ROLE_COHERENT_SUM,
            _RESPONSE_ROLE_INCOHERENT_SUM,
            _RESPONSE_ROLE_MIXED_SUM,
        }
    ):
        occupied_sources = _hashes_from_record(
            assembly_record,
            "body_plus_features_source_sha256",
            context="legacy assembly occupied-body source hash",
        )
        feature_sources, _legacy_total = (
            _feature_provenance_source_hashes(grid)
        )
        direct_source = _declared_extra_scalar(
            grid, _ASSEMBLY_BASE_SHA256_KEY
        )
        direct_response = _declared_base_response_hash(grid)
        if (
            not occupied_sources
            and direct_response is not None
            and feature_sources
        ):
            if len(feature_sources) != 1:
                raise ValueError(
                    "legacy feature-delta provenance must identify exactly "
                    "one base source hash"
                )
            if direct_source is not None:
                validated_source = _validate_hash(
                    direct_source,
                    context="legacy feature-delta base source hash",
                )
                if feature_sources != {validated_source}:
                    raise ValueError(
                        "legacy feature-delta provenance has contradictory "
                        "base source hashes"
                    )
            sources = set(feature_sources)
            responses = {direct_response}
            delta_only = True
    if delta_only and (len(sources) != 1 or len(responses) != 1):
        raise ValueError(
            "a feature-delta-only derived response must retain exactly one "
            "source-file and one physical base-response context"
        )
    return sources, responses, delta_only


def _feature_component_signatures(grid) -> set[str]:
    """Collect durable placed-component identities from source/assembly JSON."""

    signatures: set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            signature = value.get("component_signature")
            if signature is not None:
                signatures.add(
                    _validate_hash(signature, context="feature component signature")
                )
            listed = value.get("feature_component_signatures")
            if listed is not None:
                if not isinstance(listed, list):
                    raise ValueError(
                        "assembly feature_component_signatures must be a list"
                    )
                signatures.update(
                    _validate_hash(item, context="assembly component signature")
                    for item in listed
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for key in (_FEATURE_PROVENANCE_KEY, _ASSEMBLY_PROVENANCE_KEY):
        decoded = _json_metadata(grid, key)
        if decoded is not None:
            visit(decoded)
    return signatures


def _assert_no_duplicate_feature_components(grids) -> set[str]:
    seen: dict[str, int] = {}
    for index, grid in enumerate(grids, start=1):
        for signature in _feature_component_signatures(grid):
            previous = seen.get(signature)
            if previous is not None:
                raise ValueError(
                    "refusing to combine responses containing the same placed "
                    f"feature component ({signature[:12]}…; inputs {previous} "
                    f"and {index}). Use the body total or its feature delta once, "
                    "not both."
                )
            seen[signature] = index
    return set(seen)


def _assert_no_shared_body_totals(grids) -> tuple[set[str], set[str]]:
    """Validate body/delta provenance and return dual body occupancy IDs."""

    grids = list(grids)
    occupied_sources: dict[str, int] = {}
    occupied_responses: dict[str, int] = {}
    delta_sources: set[str] = set()
    delta_responses: set[str] = set()
    delta_only_indices: set[int] = set()
    hashable_unoccupied = []

    def occupy(mapping, digest, index, *, identity_label):
        previous = mapping.get(digest)
        if previous is not None:
            raise ValueError(
                "refusing to combine two responses containing the same clean "
                f"body by {identity_label} ({digest[:12]}…; inputs {previous} "
                f"and {index}). This would count the body twice. Combine the "
                "clean body once with features-only deltas instead."
            )
        mapping[digest] = index

    for index, grid in enumerate(grids, start=1):
        sources, responses = _body_plus_features_identities(grid)
        for digest in sources:
            occupy(
                occupied_sources,
                digest,
                index,
                identity_label="source-file SHA-256",
            )
        for digest in responses:
            occupy(
                occupied_responses,
                digest,
                index,
                identity_label="physical base-response SHA-256",
            )

        feature_sources, feature_responses, delta_only = (
            _feature_delta_context(grid)
        )
        if delta_only:
            delta_sources.update(feature_sources)
            delta_responses.update(feature_responses)
            delta_only_indices.add(index)
        elif not sources and not responses:
            hashable_unoccupied.append((index, grid))

    if len(delta_sources) > 1 or len(delta_responses) > 1:
        raise ValueError(
            "refusing to combine feature deltas from different Assembly base "
            "source or physical-response hashes"
        )
    if bool(delta_sources) != bool(delta_responses):
        raise ValueError(
            "feature-delta provenance must retain both source-file and "
            "physical base-response hashes"
        )

    # If tagged body occupancy exists, compare every otherwise-untagged
    # authoritative 3-D response against it. This catches clean-body +
    # body-plus-features totals even after either file was repackaged/resaved.
    # The same proof supplies the body identity required by a delta-only input.
    need_physics_proof = bool(occupied_responses or delta_responses)
    proof_errors = []
    for index, grid in hashable_unoccupied if need_physics_proof else ():
        try:
            candidate = _assembly_response_physics_sha256(grid)
        except ValueError as exc:
            proof_errors.append(f"input {index}: {exc}")
            continue
        occupy(
            occupied_responses,
            candidate,
            index,
            identity_label="physical response SHA-256",
        )

    # Delta-only responses do not contain their body. If any other input is
    # present, prove that its exact base occurs once in the assembled response.
    if delta_sources and len(delta_only_indices) != len(grids):
        expected_source = next(iter(delta_sources))
        expected_response = next(iter(delta_responses))
        if (
            occupied_sources
            and expected_source not in occupied_sources
            and expected_response not in occupied_responses
        ):
            raise ValueError(
                "refusing to combine responses from different assembly base "
                "hashes (the feature-delta source does not match the body)"
            )
        if expected_response not in occupied_responses:
            detail = (
                "; ".join(proof_errors[:2])
                if proof_errors
                else "no response physics hash matched"
            )
            raise ValueError(
                "features-only deltas require exactly one proven matching "
                "clean-body response; " + detail
            )
        occupied_sources.setdefault(
            expected_source,
            occupied_responses[expected_response],
        )

    return set(occupied_sources), set(occupied_responses)

# Icon pixel size
_ICON = 14


def _node_icon(node_type: str, expanded: bool = False, has_data: bool = True) -> QIcon:
    """
    Draw a small 14×14 icon that encodes both node type and state:
      Root (closed) — blue folder with tab
      Root (open)   — open folder (lighter, tab raised)
      Branch (closed) — teal rounded rect
      Branch (open)   — teal rounded rect, lighter fill
      Leaf (loaded)   — green circle
      Leaf (empty)    — grey circle  (also italicised/dimmed by _apply_leaf_style)
    """
    node_type = _preview_icon_type(node_type)
    pix = QPixmap(_ICON, _ICON)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)

    if node_type == _TYPE_ROOT:
        body  = QColor("#3b82f6") if not expanded else QColor("#93c5fd")
        tab   = QColor("#1d4ed8") if not expanded else QColor("#60a5fa")
        p.setBrush(tab)
        p.drawRoundedRect(1, 1, 6, 4, 1, 1)      # folder tab
        p.setBrush(body)
        p.drawRoundedRect(1, 4, 12, 9, 2, 2)     # folder body

    elif node_type == _TYPE_BRANCH:
        color = QColor("#0891b2") if not expanded else QColor("#22d3ee")
        p.setBrush(color)
        p.drawRoundedRect(2, 2, 10, 10, 2, 2)

    else:  # leaf
        color = QColor("#34d399") if has_data else QColor("#6b7280")
        p.setBrush(color)
        p.drawEllipse(2, 2, 10, 10)

    p.end()
    return QIcon(pix)


# ─────────────────────────────────────────────────────────────────────────────
# RcsGrid ↔ base-64 helpers (mirrors RcsGrid.save / RcsGrid.load exactly,
# but works with BytesIO instead of a file path)
# ─────────────────────────────────────────────────────────────────────────────

def _grid_to_b64(grid) -> str:
    """Serialise an RcsGrid to a base-64 string."""
    buf = io.BytesIO()
    units_payload = json.dumps(grid.units) if grid.units else ""
    payload = dict(grid._extra_to_write())
    payload.update(
        azimuths=grid.azimuths,
        elevations=grid.elevations,
        frequencies=grid.frequencies,
        polarizations=grid.polarizations,
        rcs_power=grid.rcs_power,
        rcs_phase=grid.rcs_phase,
        rcs_domain="power_phase",
        power_domain="linear_rcs",
        source_path=grid.source_path if grid.source_path is not None else "",
        history=grid.history if grid.history is not None else "",
        units=units_payload,
    )
    for tag in ("rcs_domain", "power_domain"):
        if tag in grid.extra:
            payload[tag] = grid.extra[tag]
    object_keys = [
        str(key)
        for key, value in payload.items()
        if np.asarray(value).dtype.hasobject
    ]
    if object_keys:
        raise ValueError(
            "cannot embed this dataset in a pickle-free .asy file because metadata "
            "contains object-typed value(s): "
            + ", ".join(sorted(object_keys))
            + ". Convert those values to numeric/string arrays or JSON text."
        )
    np.savez(buf, **payload)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_grid(b64: str):
    """Reconstruct an RcsGrid from a base-64 string."""
    from grim_dataset import RcsGrid
    buf = io.BytesIO(base64.b64decode(b64))
    data = np.load(buf, allow_pickle=False)

    units: dict = {}
    if "units" in data:
        raw = data["units"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str) and raw:
            try:
                units = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "embedded dataset contains corrupt units metadata; refusing "
                    "to guess its physical conventions"
                ) from exc
        if not isinstance(units, dict):
            raise ValueError("embedded dataset units metadata must be a JSON object")

    source_path_raw = data["source_path"].item() if "source_path" in data else None
    source_path     = source_path_raw if source_path_raw else None
    history_raw     = data["history"].item() if "history" in data else None
    history         = history_raw if history_raw else None
    extra = {
        key: data[key]
        for key in getattr(data, "files", [])
        if key not in RcsGrid._RESERVED_KEYS
    }

    return RcsGrid(
        data["azimuths"],
        data["elevations"],
        data["frequencies"],
        data["polarizations"],
        rcs_power=data["rcs_power"],
        rcs_phase=data["rcs_phase"],
        rcs_domain="power_phase",
        source_path=source_path,
        history=history,
        units=units,
        extra=extra,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tree item serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _item_to_dict(item: QTreeWidgetItem) -> dict | None:
    """Serialize one response node; preview-only geometry is runtime state."""
    if _is_preview_item(item):
        return None
    node_type = item.data(0, _ROLE_TYPE)
    children = []
    for index in range(item.childCount()):
        child = _item_to_dict(item.child(index))
        if child is not None:
            children.append(child)
    d: dict = {
        "name":     item.text(0),
        "type":     node_type,
        "included": _inclusion_state(item) != Qt.Unchecked,
        # Preview visibility is persisted independently from response
        # inclusion. A partial branch is considered visible; its individual
        # child states reproduce the partial state when the tree is loaded.
        "visible":  _visibility_state(item) != Qt.Unchecked,
        "children": children,
    }
    if node_type != _TYPE_ROOT:
        d["mode"] = _node_mode(item)
    if node_type == _TYPE_LEAF:
        d["dataset"] = item.data(0, _ROLE_NAME)
        grid = item.data(0, _ROLE_GRID)
        if grid is not None:
            try:
                d["data"] = _grid_to_b64(grid)
            except Exception as exc:
                raise ValueError(
                    "Could not serialize the embedded dataset for response "
                    f"leaf {item.text(0)!r}: {exc}"
                ) from exc
    return d


def _dict_to_item(d: dict) -> QTreeWidgetItem:
    node_type = d.get("type", _TYPE_BRANCH)
    if node_type not in (_TYPE_ROOT, _TYPE_BRANCH, _TYPE_LEAF):
        raise ValueError(
            f"assembly file contains unsupported response node type {node_type!r}"
        )
    item = QTreeWidgetItem([d["name"]])
    item.setData(0, _ROLE_TYPE, node_type)
    item.setData(0, _ROLE_PURPOSE, _PURPOSE_RESPONSE)
    _apply_flags(item, node_type)
    included = d.get("included", True)
    if type(included) is not bool:
        raise ValueError(
            f"assembly response node {d.get('name')!r} has a non-boolean "
            "'included' value"
        )
    _set_inclusion_state(item, included)
    # Version 1/2 .asy files did not carry preview visibility, and versions
    # 1-3 did not carry response inclusion. Both default checked so historical
    # assemblies retain their appearance and build behavior.
    _set_visibility_state(item, bool(d.get("visible", True)))

    if node_type == _TYPE_LEAF:
        item.setData(0, _ROLE_NAME, d.get("dataset"))
        grid = None
        if "data" in d:
            try:
                grid = _b64_to_grid(d["data"])
            except Exception as exc:
                raise ValueError(
                    "Could not decode the embedded dataset for response "
                    f"leaf {item.text(0)!r}: {exc}"
                ) from exc
        item.setData(0, _ROLE_GRID, grid)
        _apply_leaf_style(item, grid is not None)
        item.setIcon(0, _node_icon(_TYPE_LEAF, has_data=(grid is not None)))
    else:
        item.setIcon(0, _node_icon(node_type, expanded=False))
        if node_type == _TYPE_ROOT:
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)

    if node_type == _TYPE_ROOT:
        _apply_mode_badge(item, None)
    else:
        if "mode" in d:
            mode = d["mode"]
            allowed_modes = (
                (_MODE_AUTO, _MODE_COH, _MODE_INCOH)
                if node_type == _TYPE_BRANCH
                else (_MODE_COH, _MODE_INCOH)
            )
            if mode not in allowed_modes:
                raise ValueError(
                    f"assembly response node {d.get('name')!r} has invalid "
                    f"add mode {mode!r}"
                )
        elif node_type == _TYPE_LEAF:
            mode = _mode_for_grid(item.data(0, _ROLE_GRID))
        else:
            mode = _DEFAULT_BRANCH_MODE
        if (
            node_type == _TYPE_LEAF
            and mode == _MODE_COH
            and _declared_combine_role(item.data(0, _ROLE_GRID))
            == _COMBINE_ROLE_POWER
        ):
            raise ValueError(
                f"assembly response leaf {d.get('name')!r} requests coherent "
                "Field + but its dataset declares combine_role='power'"
            )
        _set_node_mode(item, mode)

    for child in d.get("children", []):
        item.addChild(_dict_to_item(child))
    if item.childCount():
        _sync_visibility_from_children(item)
        _sync_inclusion_from_children(item)
    return item


def _clone_response_item(item: QTreeWidgetItem) -> QTreeWidgetItem:
    """Deep-copy one response subtree without aliasing any RcsGrid state."""
    if _is_preview_item(item):
        raise ValueError("runtime preview nodes cannot be duplicated")
    node_type = item.data(0, _ROLE_TYPE)
    if node_type not in (_TYPE_ROOT, _TYPE_BRANCH, _TYPE_LEAF):
        raise ValueError(f"unsupported response node type {node_type!r}")

    clone = QTreeWidgetItem([item.text(0)])
    clone.setData(0, _ROLE_TYPE, node_type)
    clone.setData(0, _ROLE_PURPOSE, _PURPOSE_RESPONSE)
    _apply_flags(clone, node_type)
    _set_visibility_state(clone, _visibility_state(item) != Qt.Unchecked)
    _set_inclusion_state(clone, _inclusion_state(item) != Qt.Unchecked)

    if node_type == _TYPE_LEAF:
        clone.setData(0, _ROLE_NAME, item.data(0, _ROLE_NAME))
        grid = item.data(0, _ROLE_GRID)
        grid_copy = None if grid is None else copy.deepcopy(grid)
        clone.setData(0, _ROLE_GRID, grid_copy)
        _apply_leaf_style(clone, grid_copy is not None)
        clone.setIcon(0, _node_icon(_TYPE_LEAF, has_data=(grid_copy is not None)))
    else:
        clone.setIcon(0, _node_icon(node_type, expanded=item.isExpanded()))
        if node_type == _TYPE_ROOT:
            font = clone.font(0)
            font.setBold(True)
            clone.setFont(0, font)

    if node_type == _TYPE_ROOT:
        _apply_mode_badge(clone, None)
    else:
        _set_node_mode(clone, _node_mode(item))

    for index in range(item.childCount()):
        child = item.child(index)
        if not _is_preview_item(child):
            clone.addChild(_clone_response_item(child))
    if clone.childCount():
        _sync_visibility_from_children(clone)
        _sync_inclusion_from_children(clone)
    clone.setExpanded(item.isExpanded())
    return clone


def _apply_flags(item: QTreeWidgetItem, node_type: str) -> None:
    base = (
        Qt.ItemIsEnabled
        | Qt.ItemIsSelectable
        | Qt.ItemIsEditable
        | Qt.ItemIsUserCheckable
    )
    if node_type == _TYPE_LEAF:
        item.setFlags(base | Qt.ItemIsDragEnabled)
    else:
        item.setFlags(base | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled)


def _apply_preview_flags(item: QTreeWidgetItem) -> None:
    """Preview nodes are selectable/checkable but never response drag targets."""
    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)


def _apply_leaf_style(item: QTreeWidgetItem, has_data: bool) -> None:
    """Italicise and dim leaves that have no RcsGrid loaded yet."""
    font = item.font(0)
    font.setItalic(not has_data)
    item.setFont(0, font)
    if has_data:
        item.setForeground(0, QBrush())          # default colour
    else:
        item.setForeground(0, QBrush(QColor("#888888")))
        item.setToolTip(0, "No dataset data loaded – drag from the Datasets table")


# ─────────────────────────────────────────────────────────────────────────────
# Tree widget
# ─────────────────────────────────────────────────────────────────────────────

class AssemblyTree(QTreeWidget):
    """
    Node hierarchy:   Root  →  Branch(es)  →  Leaf (stores RcsGrid data)

    Drop IN (onto a branch or root):
      • MIME_DATASET from the Datasets table → leaf; RcsGrid copied from
        DatasetTable._pending_drag_data set during startDrag
      • dataset file URLs from the file explorer
        (.grim/.csv/.cst_data/.txt/.out/.pio/.cmplx_di/.ptm/.ss)
        → .grim creates leaves with loaded data;
        also emits files_to_load so the main window adds them to the table

    Drag OUT (branch → Datasets table):
      • Stores list[(name, RcsGrid|None)] in _pending_branch_data for the
        table to retrieve.  Tree is never modified.

    Internal drag:
      • Branches/roots: reparented manually (MIME_BRANCH used as marker).
      • Leaves: Qt InternalMove.
    """

    files_to_load = Signal(list)  # list of supported dataset file paths
    # Emitted synchronously before every full tree clear/replacement.  Runtime
    # preview owners use this to remove their scene artists before an .asy load
    # discards the corresponding typed preview nodes.
    tree_clearing = Signal()
    # Emitted synchronously while a typed preview subtree is still attached.
    # Scene owners can inspect/clear their runtime bindings before deletion.
    preview_removing = Signal(object)
    # Emitted after a leaf or branch preview checkbox changes. Consumers should
    # refresh their lightweight scene from the tree; assembly physics is not
    # affected by this signal or by the checkbox states.
    visibility_changed = Signal(object, bool)
    # Response-tree mutations only. Runtime feature-preview nodes are excluded
    # because they are rebuilt by the feature workflow and never serialized.
    content_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("assemblyTree")
        self.setColumnCount(4)
        self.setHeaderLabels(["Assembly", "Mode", "Use", "Show"])
        self.setColumnWidth(0, 220)
        self.setColumnWidth(1, 56)
        self.setColumnWidth(2, 48)
        self.setColumnWidth(3, 54)
        self.headerItem().setToolTip(
            _COLUMN_MODE,
            "How this response enters its parent: Field + is a complex sum and "
            "requires a common phase center, time convention, polarization "
            "basis, and coordinate frame. Power + is the safe default for an "
            "untyped leaf. A branch defaults to Auto, which adopts the "
            "combine_role of the response built inside that branch; use the "
            "context menu only for an explicit branch override.",
        )
        self.headerItem().setToolTip(
            _COLUMN_INCLUDED,
            "Include this response node in Build Platform. Uncheck a body or "
            "feature branch to compare assembly variants without deleting it.",
        )
        self.headerItem().setToolTip(
            _COLUMN_VISIBILITY,
            "Show or hide this node in the Assembly 3-D preview. "
            "This does not include or exclude response data from Build Platform.",
        )
        # InternalMove prevents Qt's C++ QAbstractItemView::dropEvent from
        # calling model()->dropMimeData() for external drops, which would
        # interfere with items we add manually in our Python override.
        # Our dragEnterEvent/dragMoveEvent/dropEvent overrides still accept
        # external MIME_DATASET and URL drops; internal leaf moves fall
        # through to super().dropEvent() which InternalMove handles correctly.
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.invisibleRootItem().setFlags(
            self.invisibleRootItem().flags() | Qt.ItemIsDropEnabled
        )
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self.itemExpanded.connect(self._on_item_expanded)
        self.itemCollapsed.connect(self._on_item_collapsed)
        self._updating_visibility = False
        self._updating_inclusion = False
        self.itemChanged.connect(self._on_item_changed)
        self._branch_drag_item: QTreeWidgetItem | None = None
        self._pending_branch_data: list | None = None

    def edit(self, index, trigger, event):
        """Allow direct text editing only for the visible Name column.

        Mode is a rendered view of ``_ROLE_MODE`` and must only change through
        the explicit context-menu controls; editing its text would otherwise
        make the badge disagree with the role used by physics.
        """

        if index.isValid() and index.column() != _COLUMN_NAME:
            return False
        return super().edit(index, trigger, event)

    def clear(self) -> None:
        """Clear all nodes after notifying owners of runtime-only previews."""

        self.tree_clearing.emit()
        super().clear()

    # ── typed, preview-only nodes ------------------------------------------

    @staticmethod
    def item_purpose(item: QTreeWidgetItem) -> str:
        return _item_purpose(item)

    @staticmethod
    def item_preview_key(item: QTreeWidgetItem) -> str | None:
        return _preview_key(item)

    def _walk_items(self, parent: QTreeWidgetItem | None = None):
        root = self.invisibleRootItem() if parent is None else parent
        for index in range(root.childCount()):
            child = root.child(index)
            yield child
            yield from self._walk_items(child)

    def preview_item_for_key(self, stable_key: str) -> QTreeWidgetItem | None:
        key = str(stable_key)
        for item in self._walk_items():
            if _is_preview_item(item) and _preview_key(item) == key:
                return item
        return None

    def _new_preview_key(self) -> str:
        return f"preview:{uuid.uuid4().hex}"

    def _make_preview_node(
        self,
        name: str,
        node_type: str,
        *,
        parent: QTreeWidgetItem | None,
        stable_key: str | None,
    ) -> QTreeWidgetItem:
        if node_type not in (
            _TYPE_PREVIEW_ROOT,
            _TYPE_PREVIEW_GROUP,
            _TYPE_PREVIEW_ITEM,
        ):
            raise ValueError(f"unsupported preview node type {node_type!r}")
        if node_type == _TYPE_PREVIEW_ROOT:
            if parent is not None:
                raise ValueError("a preview root must be top-level")
        elif (
            parent is None
            or not _is_preview_item(parent)
            or parent.data(0, _ROLE_TYPE) == _TYPE_PREVIEW_ITEM
        ):
            raise ValueError(
                "preview groups/items require a preview root or group parent"
            )

        key = self._new_preview_key() if stable_key is None else str(stable_key)
        if not key or key != key.strip():
            raise ValueError(
                "preview stable_key must be nonempty without surrounding whitespace"
            )
        if self.preview_item_for_key(key) is not None:
            raise ValueError(f"duplicate preview stable_key {key!r}")

        item = QTreeWidgetItem([str(name)])
        item.setData(0, _ROLE_TYPE, node_type)
        item.setData(0, _ROLE_PURPOSE, _PURPOSE_PREVIEW)
        item.setData(0, _ROLE_PREVIEW_KEY, key)
        item.setData(0, _ROLE_MODE, None)
        _apply_preview_flags(item)
        _set_visibility_state(item, True)
        item.setIcon(0, _node_icon(node_type, expanded=False, has_data=True))
        _apply_mode_badge(item, None)
        item.setToolTip(
            0,
            "3-D preview only. This node is not a response dataset and is "
            "never included in Build Platform.",
        )
        if node_type == _TYPE_PREVIEW_ROOT:
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            self.invisibleRootItem().addChild(item)
        else:
            parent.addChild(item)
            parent.setExpanded(True)
            self._refresh_visibility_after_structure_change(parent)
        self.visibility_changed.emit(item, True)
        return item

    def add_preview_root(
        self,
        name: str,
        stable_key: str | None = None,
    ) -> QTreeWidgetItem:
        return self._make_preview_node(
            name,
            _TYPE_PREVIEW_ROOT,
            parent=None,
            stable_key=stable_key,
        )

    def add_preview_group(
        self,
        parent: QTreeWidgetItem,
        name: str,
        stable_key: str | None = None,
    ) -> QTreeWidgetItem:
        return self._make_preview_node(
            name,
            _TYPE_PREVIEW_GROUP,
            parent=parent,
            stable_key=stable_key,
        )

    def add_preview_item(
        self,
        parent: QTreeWidgetItem,
        name: str,
        stable_key: str | None = None,
    ) -> QTreeWidgetItem:
        return self._make_preview_node(
            name,
            _TYPE_PREVIEW_ITEM,
            parent=parent,
            stable_key=stable_key,
        )

    def remove_preview_key(self, stable_key: str) -> bool:
        item = self.preview_item_for_key(stable_key)
        if item is None:
            return False
        self._remove_item(item)
        return True

    def remove_preview_root(self, root_or_key: QTreeWidgetItem | str) -> bool:
        item = (
            self.preview_item_for_key(root_or_key)
            if isinstance(root_or_key, str)
            else root_or_key
        )
        if item is None:
            return False
        if (
            not _is_preview_item(item)
            or item.data(0, _ROLE_TYPE) != _TYPE_PREVIEW_ROOT
            or item.parent() is not None
        ):
            raise ValueError("remove_preview_root requires a preview root or its key")
        self._remove_item(item)
        return True

    # ── preview visibility --------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if (
            column == _COLUMN_INCLUDED
            and not _is_preview_item(item)
            and not self._updating_inclusion
        ):
            state = _inclusion_state(item)
            self._updating_inclusion = True
            try:
                if state in (Qt.Checked, Qt.Unchecked) and item.childCount():
                    included = state == Qt.Checked
                    for index in range(item.childCount()):
                        _set_subtree_included(item.child(index), included)
                _sync_inclusion_from_children(item.parent())
            finally:
                self._updating_inclusion = False
            self.content_changed.emit()
            return
        if (
            not self._updating_visibility
            and not self._updating_inclusion
            and not _is_preview_item(item)
        ):
            self.content_changed.emit()
        if self._updating_visibility or column != _COLUMN_VISIBILITY:
            return
        state = _visibility_state(item)
        changed_items = [item]
        self._updating_visibility = True
        try:
            # A user checking/unchecking a container applies that choice to
            # its complete subtree. Partial states are summaries created from
            # differing child states and never cascade.
            if state in (Qt.Checked, Qt.Unchecked) and item.childCount():
                visible = state == Qt.Checked
                changed_items = _subtree_items(item)
                for index in range(item.childCount()):
                    _set_subtree_visibility(item.child(index), visible)
            _sync_visibility_from_children(item.parent())
        finally:
            self._updating_visibility = False
        for changed_item in changed_items:
            self.visibility_changed.emit(
                changed_item, _item_preview_visible(changed_item)
            )

    def set_item_preview_visible(
        self, item: QTreeWidgetItem, visible: bool
    ) -> None:
        """Show/hide ``item`` and its subtree in previews only.

        This programmatic counterpart to clicking the checkbox emits one
        :attr:`visibility_changed` signal and leaves coherent/incoherent build
        membership untouched.
        """
        if item is None:
            return
        changed_items = _subtree_items(item)
        self._updating_visibility = True
        try:
            _set_subtree_visibility(item, visible)
            _sync_visibility_from_children(item.parent())
        finally:
            self._updating_visibility = False
        for changed_item in changed_items:
            self.visibility_changed.emit(
                changed_item, _item_preview_visible(changed_item)
            )
        if any(not _is_preview_item(changed_item) for changed_item in changed_items):
            self.content_changed.emit()

    @staticmethod
    def item_visible(item: QTreeWidgetItem) -> bool:
        """Return effective preview visibility for one tree item."""
        return _item_preview_visible(item)

    # Descriptive compatibility alias for callers developed alongside the
    # initial Assembly workspace prototype.
    item_preview_visible = item_visible

    def set_item_included(self, item: QTreeWidgetItem, included: bool) -> None:
        """Include/exclude a response subtree from Build Platform."""
        if item is None:
            return
        if _is_preview_item(item):
            raise ValueError("preview-only nodes cannot be included in response builds")
        self._updating_inclusion = True
        try:
            _set_subtree_included(item, included)
            _sync_inclusion_from_children(item.parent())
        finally:
            self._updating_inclusion = False
        self.content_changed.emit()

    @staticmethod
    def item_included(item: QTreeWidgetItem) -> bool:
        """Return effective Build Platform membership for one response node."""
        return _item_response_included(item)

    def _refresh_visibility_after_structure_change(
        self, parent: QTreeWidgetItem | None
    ) -> None:
        """Refresh tri-state ancestors after an item is attached or removed."""
        if parent is None:
            return
        self._updating_visibility = True
        self._updating_inclusion = True
        try:
            _sync_visibility_from_children(parent)
            _sync_inclusion_from_children(parent)
        finally:
            self._updating_visibility = False
            self._updating_inclusion = False
        self.visibility_changed.emit(parent, _item_preview_visible(parent))

    # ── branch indicators (plus/minus box) ───────────────────────────────────

    def drawBranches(self, painter: QPainter, rect, index) -> None:
        super().drawBranches(painter, rect, index)
        item = self.itemFromIndex(index)
        if item is None or item.childCount() == 0:
            return
        sz     = 9
        indent = self.indentation()
        x      = rect.right() - indent + (indent - sz) // 2
        y      = rect.top() + (rect.height() - sz) // 2
        mid_x  = x + sz // 2
        mid_y  = y + sz // 2
        color  = self.palette().color(QPalette.ColorRole.Text)
        bg     = self.palette().color(QPalette.ColorRole.Base)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(QPen(color, 1))
        painter.setBrush(bg)
        painter.drawRect(x, y, sz - 1, sz - 1)
        painter.drawLine(x + 2, mid_y, x + sz - 3, mid_y)
        if not item.isExpanded():
            painter.drawLine(mid_x, y + 2, mid_x, y + sz - 3)
        painter.restore()

    # ── outbound drag ────────────────────────────────────────────────────────

    def startDrag(self, supported_actions) -> None:
        item = self.currentItem()
        if item is None or _is_preview_item(item):
            return
        node_type = item.data(0, _ROLE_TYPE)
        if node_type in (_TYPE_ROOT, _TYPE_BRANCH):
            leaf_data = self._collect_leaf_data(item)
            self._pending_branch_data = leaf_data
            self._branch_drag_item    = item
            mime = QMimeData()
            mime.setData(MIME_BRANCH, QByteArray(item.text(0).encode("utf-8")))
            drag = QDrag(self)
            drag.setMimeData(mime)
            drag.exec(Qt.CopyAction | Qt.MoveAction)
            self._pending_branch_data = None
            self._branch_drag_item    = None
        else:
            super().startDrag(supported_actions)

    def _collect_leaf_data(self, item: QTreeWidgetItem) -> list[tuple[str, object]]:
        result: list[tuple[str, object]] = []
        if not _item_response_included(item):
            return result
        for i in range(item.childCount()):
            child = item.child(i)
            if _is_preview_item(child):
                continue
            if child.data(0, _ROLE_TYPE) == _TYPE_LEAF:
                name = child.data(0, _ROLE_NAME) or child.text(0)
                grid = child.data(0, _ROLE_GRID)
                result.append((name, grid))
            else:
                result.extend(self._collect_leaf_data(child))
        return result

    # ── inbound drag ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        mime = event.mimeData()
        if mime.hasFormat(MIME_DATASET) or mime.hasFormat(MIME_BRANCH) or mime.hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        mime = event.mimeData()
        if mime.hasFormat(MIME_DATASET) or mime.hasFormat(MIME_BRANCH) or mime.hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        mime    = event.mimeData()
        vp_pos  = self.viewport().mapFrom(self, event.position().toPoint())
        target  = self.itemAt(vp_pos)

        # Preview geometry is deliberately not a response drop target. It is
        # populated only through the typed preview API/service plan.
        if _is_preview_item(target):
            event.ignore()
            return

        # ── internal branch reparent ─────────────────────────────────────────
        if mime.hasFormat(MIME_BRANCH) and event.source() is self:
            item = self._branch_drag_item
            if _branch_drop_would_create_cycle(item, target):
                event.ignore()
                return
            old_parent = item.parent()
            (old_parent or self.invisibleRootItem()).removeChild(item)
            new_parent = None
            if target is not None and target.data(0, _ROLE_TYPE) in (_TYPE_ROOT, _TYPE_BRANCH):
                _inherit_container_states(item, target)
                target.addChild(item)
                target.setExpanded(True)
                new_parent = target
            elif target is not None and target.data(0, _ROLE_TYPE) == _TYPE_LEAF:
                parent = target.parent() or self.invisibleRootItem()
                if parent is not self.invisibleRootItem():
                    _inherit_container_states(item, parent)
                parent.addChild(item)
                parent.setExpanded(True)
                new_parent = None if parent is self.invisibleRootItem() else parent
            else:
                self.invisibleRootItem().addChild(item)
            self._refresh_visibility_after_structure_change(old_parent)
            if new_parent is not old_parent:
                self._refresh_visibility_after_structure_change(new_parent)
            self.content_changed.emit()
            event.acceptProposedAction()
            return

        # ── dataset dragged from the Datasets table ──────────────────────────
        if mime.hasFormat(MIME_DATASET) and event.source() is not self:
            src = event.source()
            entries: list[tuple[str, object]] = []
            if hasattr(src, "_pending_drag_data") and src._pending_drag_data:
                entries = src._pending_drag_data  # list of (name, RcsGrid|None)
            else:
                entries = [(bytes(mime.data(MIME_DATASET)).decode("utf-8"), None)]
            for name, grid in entries:
                _attach(self, self._make_leaf(name, grid), target)
            event.acceptProposedAction()
            return

        # ── dataset files dropped from the file explorer ─────────────────────
        if mime.hasUrls() and event.source() is not self:
            from grim_dataset import RcsGrid
            supported_paths = [
                u.toLocalFile() for u in mime.urls()
                if u.isLocalFile() and is_supported_path(u.toLocalFile())
            ]
            grim_paths = [path for path in supported_paths if path.lower().endswith(".grim")]
            if grim_paths:
                for path in grim_paths:
                    name = os.path.splitext(os.path.basename(path))[0]
                    try:
                        grid = RcsGrid.load(path)
                    except Exception:
                        grid = None
                    leaf = self._make_leaf(name, grid)
                    _attach(self, leaf, target)
            if supported_paths:
                self.files_to_load.emit(supported_paths)
                event.acceptProposedAction()
                return

        super().dropEvent(event)
        if event.isAccepted():
            self.content_changed.emit()

    # ── node factories ───────────────────────────────────────────────────────

    def _make_leaf(self, dataset_name: str, grid=None) -> QTreeWidgetItem:
        item = QTreeWidgetItem([dataset_name])
        item.setData(0, _ROLE_TYPE, _TYPE_LEAF)
        item.setData(0, _ROLE_PURPOSE, _PURPOSE_RESPONSE)
        item.setData(0, _ROLE_NAME, dataset_name)
        item.setData(0, _ROLE_GRID, grid)
        _apply_flags(item, _TYPE_LEAF)
        _set_inclusion_state(item, True)
        _set_visibility_state(item, True)
        _apply_leaf_style(item, grid is not None)
        item.setIcon(0, _node_icon(_TYPE_LEAF, has_data=(grid is not None)))
        try:
            mode = _mode_for_grid(grid)
        except ValueError as exc:
            mode = _MODE_INCOH
            item.setToolTip(
                _COLUMN_MODE,
                f"Unsafe response metadata: {exc}. Build Platform will refuse "
                "this dataset until its role metadata is repaired.",
            )
        _set_node_mode(item, mode)
        return item

    def _make_node(
        self,
        name: str,
        node_type: str,
        parent: QTreeWidgetItem | None = None,
        edit: bool = True,
    ) -> QTreeWidgetItem:
        if node_type not in (_TYPE_ROOT, _TYPE_BRANCH):
            raise ValueError(f"unsupported response container type {node_type!r}")
        if node_type == _TYPE_ROOT and parent is not None:
            raise ValueError("response roots must be top-level")
        if parent is not None and (
            _is_preview_item(parent)
            or parent.data(0, _ROLE_TYPE) == _TYPE_LEAF
        ):
            raise ValueError("response branches require a root or branch parent")
        item = QTreeWidgetItem([name])
        item.setData(0, _ROLE_TYPE, node_type)
        item.setData(0, _ROLE_PURPOSE, _PURPOSE_RESPONSE)
        _apply_flags(item, node_type)
        _set_inclusion_state(item, True)
        _set_visibility_state(item, True)
        item.setIcon(0, _node_icon(node_type, expanded=False))
        if node_type == _TYPE_ROOT:
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
        if node_type == _TYPE_ROOT:
            _apply_mode_badge(item, None)
        else:
            _set_node_mode(item, _DEFAULT_BRANCH_MODE)
        if parent is not None:
            _inherit_container_states(item, parent)
            parent.addChild(item)
            parent.setExpanded(True)
            self._refresh_visibility_after_structure_change(parent)
        else:
            self.invisibleRootItem().addChild(item)
            self.visibility_changed.emit(item, _item_preview_visible(item))
        if edit:
            self.scrollToItem(item)
            self.editItem(item, 0)
        self.content_changed.emit()
        return item

    def duplicate_response_subtree(
        self, item: QTreeWidgetItem
    ) -> QTreeWidgetItem:
        """Insert an independent, clearly named copy beside ``item``."""
        if item is None or _is_preview_item(item):
            raise ValueError("select a response root, branch, or dataset")

        parent = item.parent()
        container = parent or self.invisibleRootItem()
        sibling_names = {
            container.child(index).text(0)
            for index in range(container.childCount())
        }
        base_name = f"{item.text(0)} Copy"
        copy_name = base_name
        suffix = 2
        while copy_name in sibling_names:
            copy_name = f"{base_name} {suffix}"
            suffix += 1

        clone = _clone_response_item(item)
        clone.setText(0, copy_name)
        source_index = container.indexOfChild(item)
        container.insertChild(source_index + 1, clone)
        clone.setExpanded(item.isExpanded())
        if parent is not None:
            self._refresh_visibility_after_structure_change(parent)
        else:
            self.visibility_changed.emit(clone, _item_preview_visible(clone))
        self.setCurrentItem(clone)
        self.scrollToItem(clone)
        self.content_changed.emit()
        return clone

    def confirm_remove_item(self, item: QTreeWidgetItem) -> bool:
        """Require confirmation before deleting loaded data or a subtree."""

        def _counts(node: QTreeWidgetItem) -> tuple[int, int]:
            nodes = 1
            loaded = int(
                node.data(0, _ROLE_TYPE) == _TYPE_LEAF
                and node.data(0, _ROLE_GRID) is not None
            )
            for child_index in range(node.childCount()):
                child_nodes, child_loaded = _counts(node.child(child_index))
                nodes += child_nodes
                loaded += child_loaded
            return nodes, loaded

        node_count, loaded_count = _counts(item)
        if node_count == 1 and loaded_count == 0:
            return True
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        answer = QMessageBox.question(
            self,
            "Delete Response Data?",
            f"Delete '{item.text(0)}' and its {node_count - 1} descendant(s)?\n\n"
            f"This removes {loaded_count} embedded dataset(s) and cannot be undone.",
            buttons.Yes | buttons.No,
            buttons.No,
        )
        return answer == buttons.Yes

    def _remove_item(self, item: QTreeWidgetItem) -> None:
        parent = item.parent()
        response_item = not _is_preview_item(item)
        if _is_preview_item(item):
            self.preview_removing.emit(item)
        (parent or self.invisibleRootItem()).removeChild(item)
        # The detached item still carries its runtime scene binding, so emit
        # it before the last Python reference can disappear. This also makes
        # programmatic remove_preview_key(child_key) hide that child's artist.
        self.visibility_changed.emit(item, False)
        if parent is not None:
            self._refresh_visibility_after_structure_change(parent)
        if response_item:
            self.content_changed.emit()

    # ── expand / collapse icon updates ───────────────────────────────────────

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        node_type = item.data(0, _ROLE_TYPE)
        if node_type in (
            _TYPE_ROOT, _TYPE_BRANCH, _TYPE_PREVIEW_ROOT, _TYPE_PREVIEW_GROUP
        ):
            item.setIcon(0, _node_icon(node_type, expanded=True))

    def _on_item_collapsed(self, item: QTreeWidgetItem) -> None:
        node_type = item.data(0, _ROLE_TYPE)
        if node_type in (
            _TYPE_ROOT, _TYPE_BRANCH, _TYPE_PREVIEW_ROOT, _TYPE_PREVIEW_GROUP
        ):
            item.setIcon(0, _node_icon(node_type, expanded=False))

    # ── context menu ─────────────────────────────────────────────────────────

    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        preview_only = _is_preview_item(item)
        menu = QMenu(self)
        act_root   = menu.addAction("Add Root")
        act_branch = menu.addAction("Add Branch")
        act_duplicate = menu.addAction("Duplicate Subtree")
        act_del    = menu.addAction("Delete")
        menu.addSeparator()
        act_expand   = menu.addAction("Expand")
        act_collapse = menu.addAction("Collapse")
        menu.addSeparator()
        act_rename = menu.addAction("Rename")
        if preview_only:
            act_root.setVisible(False)
            act_branch.setVisible(False)
            act_duplicate.setVisible(False)
            act_del.setVisible(False)
            act_rename.setVisible(False)
        elif item is None:
            act_duplicate.setEnabled(False)
            act_del.setEnabled(False)
            act_rename.setEnabled(False)

        # Per-node add-mode setters (only meaningful for non-root nodes).
        act_set_coh = None
        act_set_inc = None
        act_set_auto = None
        if (
            item is not None
            and not preview_only
            and item.data(0, _ROLE_TYPE) != _TYPE_ROOT
        ):
            menu.addSeparator()
            current = _node_mode(item)
            if item.data(0, _ROLE_TYPE) == _TYPE_BRANCH:
                act_set_auto = menu.addAction(
                    "Set Add Mode: Auto (adopt built response role)"
                )
                act_set_auto.setCheckable(True)
                act_set_auto.setChecked(current == _MODE_AUTO)
            act_set_coh = menu.addAction(
                "Set Add Mode: Pre-aligned coherent field (+)"
            )
            act_set_inc = menu.addAction("Set Add Mode: Incoherent power (+)")
            act_set_coh.setCheckable(True)
            act_set_inc.setCheckable(True)
            act_set_coh.setChecked(current == _MODE_COH)
            act_set_inc.setChecked(current == _MODE_INCOH)
            if item.data(0, _ROLE_TYPE) == _TYPE_LEAF:
                try:
                    declared_role = _declared_combine_role(
                        item.data(0, _ROLE_GRID)
                    )
                except ValueError as exc:
                    act_set_coh.setEnabled(False)
                    act_set_coh.setToolTip(str(exc))
                else:
                    if declared_role == _COMBINE_ROLE_POWER:
                        act_set_coh.setEnabled(False)
                        act_set_coh.setToolTip(
                            "This dataset explicitly declares phase-unknown "
                            "power and cannot enter a coherent field sum."
                        )

        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        if chosen == act_root:
            self._make_node("New Root", _TYPE_ROOT, parent=None)
        elif chosen == act_branch:
            branch_parent = item
            if (
                branch_parent is not None
                and branch_parent.data(0, _ROLE_TYPE) == _TYPE_LEAF
            ):
                branch_parent = branch_parent.parent()
            self._make_node("New Branch", _TYPE_BRANCH, parent=branch_parent)
        elif chosen == act_duplicate and item is not None:
            self.duplicate_response_subtree(item)
        elif chosen == act_del and item is not None:
            if self.confirm_remove_item(item):
                self._remove_item(item)
        elif chosen == act_expand and item is not None:
            self.expandItem(item)
            for i in range(item.childCount()):
                self.expandItem(item.child(i))
        elif chosen == act_collapse and item is not None:
            self.collapseItem(item)
        elif chosen == act_rename and item is not None:
            self.editItem(item, 0)
        elif chosen == act_set_auto and item is not None:
            _set_node_mode(item, _MODE_AUTO)
        elif chosen == act_set_coh and item is not None:
            _set_node_mode(item, _MODE_COH)
        elif chosen == act_set_inc and item is not None:
            _set_node_mode(item, _MODE_INCOH)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _attach(
    tree: QTreeWidget, item: QTreeWidgetItem, target: QTreeWidgetItem | None
) -> None:
    if _is_preview_item(target) or _is_preview_item(item):
        raise ValueError(
            "preview-only geometry cannot be attached through the response tree drag API"
        )
    attached_parent = None
    if target is not None and target.data(0, _ROLE_TYPE) in (_TYPE_ROOT, _TYPE_BRANCH):
        _inherit_container_states(item, target)
        target.addChild(item)
        target.setExpanded(True)
        attached_parent = target
    elif target is not None and target.data(0, _ROLE_TYPE) == _TYPE_LEAF:
        # Drop on a leaf → insert into the leaf's parent container
        parent = target.parent() or tree.invisibleRootItem()
        if parent is not tree.invisibleRootItem():
            _inherit_container_states(item, parent)
        parent.addChild(item)
        parent.setExpanded(True)
        if parent is not tree.invisibleRootItem():
            attached_parent = parent
    else:
        tree.invisibleRootItem().addChild(item)
    refresh = getattr(tree, "_refresh_visibility_after_structure_change", None)
    if callable(refresh):
        if attached_parent is None:
            tree.visibility_changed.emit(item, _item_preview_visible(item))
        else:
            refresh(attached_parent)
    changed = getattr(tree, "content_changed", None)
    if changed is not None:
        changed.emit()


def _is_ancestor(candidate: QTreeWidgetItem | None, item: QTreeWidgetItem) -> bool:
    if candidate is None:
        return False
    p = item.parent()
    while p is not None:
        if p is candidate:
            return True
        p = p.parent()
    return False


def _branch_drop_would_create_cycle(
    item: QTreeWidgetItem | None,
    target: QTreeWidgetItem | None,
) -> bool:
    """Whether reparenting ``item`` beneath ``target`` would form a cycle."""

    return bool(
        item is None
        or target is item
        or (target is not None and _is_ancestor(item, target))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Panel widget
# ─────────────────────────────────────────────────────────────────────────────

class AssemblyTreePanel(QWidget):
    files_to_load = Signal(list)
    # (platform_name, combined_RcsGrid, history_string)
    platform_built = Signal(str, object, str)
    dirty_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Assembly Tree"))
        title_row.addStretch(1)
        self.chk_show_all = QCheckBox("Show All")
        self.chk_show_all.setTristate(True)
        self.chk_show_all.setChecked(True)
        self.chk_show_all.setToolTip(
            "Show or hide every assembly branch in the 3-D preview. "
            "This does not change which responses Build Platform combines."
        )
        title_row.addWidget(self.chk_show_all)
        layout.addLayout(title_row)

        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self.btn_add_root   = QToolButton(text="+ Root")
        self.btn_add_branch = QToolButton(text="+ Branch")
        self.btn_duplicate  = QToolButton(text="Duplicate")
        self.btn_duplicate.setToolTip(
            "Copy the selected response subtree and all of its dataset data. "
            "Use the Use checkboxes to compare variants."
        )
        self.btn_delete     = QToolButton(text="Delete")
        row1.addWidget(self.btn_add_root)
        row1.addWidget(self.btn_add_branch)
        row1.addWidget(self.btn_duplicate)
        row1.addWidget(self.btn_delete)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self.btn_expand   = QToolButton(text="Expand")
        self.btn_collapse = QToolButton(text="Collapse")
        self.btn_save = QToolButton(text="Save .asy")
        self.btn_load = QToolButton(text="Load .asy")
        row2.addWidget(self.btn_expand)
        row2.addWidget(self.btn_collapse)
        row2.addStretch(1)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(4)
        row3.addWidget(self.btn_save)
        row3.addWidget(self.btn_load)
        row3.addStretch(1)
        layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(4)
        self.btn_build = QToolButton(text="Build Platform")
        self.btn_build.setToolTip(
            "Recursively combine the selected root/branch into a single dataset, "
            "honouring Use checkboxes and each node's pre-aligned field / "
            "incoherent power add-mode, then aligning axes using the strategy "
            "chosen in the dialog."
        )
        row4.addWidget(self.btn_build)
        row4.addStretch(1)
        layout.addLayout(row4)

        self.tree = AssemblyTree()
        layout.addWidget(self.tree, 1)
        self._syncing_show_all = False
        self._suppress_dirty = False
        self._dirty = False
        self._assembly_path: Path | None = None

        self.btn_add_root.clicked.connect(
            lambda: self.tree._make_node("New Root", _TYPE_ROOT)
        )
        self.btn_add_branch.clicked.connect(self._add_branch)
        self.btn_duplicate.clicked.connect(self._duplicate_selected)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_expand.clicked.connect(self._expand_selected)
        self.btn_collapse.clicked.connect(self._collapse_selected)
        self.btn_save.clicked.connect(lambda: self._save())
        self.btn_load.clicked.connect(self._load)
        self.btn_build.clicked.connect(self._build)
        self.tree.files_to_load.connect(self.files_to_load)
        self.tree.visibility_changed.connect(self._sync_show_all_checkbox)
        self.tree.content_changed.connect(self._mark_dirty)
        self.chk_show_all.stateChanged.connect(self._set_show_all)

    @property
    def assembly_path(self) -> Path | None:
        return self._assembly_path

    def is_dirty(self) -> bool:
        return bool(self._dirty)

    def _set_dirty(self, dirty: bool) -> None:
        value = bool(dirty)
        if value == self._dirty:
            return
        self._dirty = value
        self.dirty_changed.emit(value)

    def _mark_dirty(self, *_args) -> None:
        if not self._suppress_dirty:
            self._set_dirty(True)

    def _document(self) -> dict:
        root = self.tree.invisibleRootItem()
        nodes = []
        for index in range(root.childCount()):
            serialized = _item_to_dict(root.child(index))
            if serialized is not None:
                nodes.append(serialized)
        # Version 5 adds the explicit branch-only Auto add mode. Older files
        # remain readable; a branch with no mode adopts Auto on load.
        return {"version": 5, "tree": nodes}

    def _confirm_unsaved_changes(
        self,
        *,
        action: str,
        parent: QWidget | None = None,
    ) -> bool:
        if not self.is_dirty():
            return True
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        shown_name = self._assembly_path.name if self._assembly_path else "Untitled assembly"
        answer = QMessageBox.warning(
            parent or self,
            "Unsaved Assembly",
            f"'{shown_name}' has unsaved response-tree changes. "
            f"Save them before {action}?",
            buttons.Save | buttons.Discard | buttons.Cancel,
            buttons.Save,
        )
        if answer == buttons.Cancel:
            return False
        if answer == buttons.Save:
            return self._save(path=self._assembly_path)
        return True

    def request_close(self, parent: QWidget | None = None) -> bool:
        return self._confirm_unsaved_changes(action="closing GRIM", parent=parent)

    def _set_show_all(self, raw_state: int) -> None:
        """Apply the global preview toggle to every top-level subtree."""
        if self._syncing_show_all:
            return
        state = Qt.CheckState(raw_state)
        if state == Qt.PartiallyChecked:
            return
        visible = state == Qt.Checked
        root = self.tree.invisibleRootItem()
        for index in range(root.childCount()):
            self.tree.set_item_preview_visible(root.child(index), visible)
        self._sync_show_all_checkbox()

    def _sync_show_all_checkbox(self, *_args) -> None:
        """Reflect checked, hidden, or mixed top-level preview state."""
        root = self.tree.invisibleRootItem()
        states = [
            root.child(index).checkState(_COLUMN_VISIBILITY)
            for index in range(root.childCount())
        ]
        if not states or all(state == Qt.Checked for state in states):
            state = Qt.Checked
        elif all(state == Qt.Unchecked for state in states):
            state = Qt.Unchecked
        else:
            state = Qt.PartiallyChecked
        self._syncing_show_all = True
        try:
            self.chk_show_all.setCheckState(state)
        finally:
            self._syncing_show_all = False

    def _add_branch(self) -> None:
        parent = self.tree.currentItem()
        if _is_preview_item(parent):
            self._notify(
                "Feature preview branches are managed by the feature workflow, "
                "not by response assembly controls."
            )
            return
        if parent is not None and parent.data(0, _ROLE_TYPE) == _TYPE_LEAF:
            parent = parent.parent()
        self.tree._make_node("New Branch", _TYPE_BRANCH, parent=parent)

    def _delete_selected(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        if _is_preview_item(item):
            self._notify(
                "Feature preview nodes are managed by the feature workflow and "
                "are replaced when its inputs are validated."
            )
            return
        if self.tree.confirm_remove_item(item):
            self.tree._remove_item(item)

    def _duplicate_selected(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            self._notify("Select a response root, branch, or dataset to duplicate.")
            return
        if _is_preview_item(item):
            self._notify(
                "Feature preview nodes are regenerated from Place Features and "
                "cannot be duplicated here. Duplicate a response branch instead."
            )
            return
        try:
            duplicate = self.tree.duplicate_response_subtree(item)
        except Exception as exc:
            self._notify(f"Could not duplicate assembly subtree: {exc}")
            return
        self._notify(
            f"Duplicated '{item.text(0)}' as '{duplicate.text(0)}'. "
            "Uncheck Use on either variant for a trade study."
        )

    def _expand_selected(self) -> None:
        item = self.tree.currentItem()
        if item is not None:
            self.tree.expandItem(item)
            for i in range(item.childCount()):
                self.tree.expandItem(item.child(i))

    def _collapse_selected(self) -> None:
        item = self.tree.currentItem()
        if item is not None:
            self.tree.collapseItem(item)

    def _save(self, path: str | os.PathLike[str] | None = None) -> bool:
        target = Path(path).expanduser() if path is not None else None
        if target is None:
            default_path = str(self._assembly_path or Path("assembly.asy"))
            selected, _ = QFileDialog.getSaveFileName(
                self,
                "Save Assembly Tree",
                default_path,
                "Assembly Files (*.asy)",
            )
            if not selected:
                return False
            target = Path(selected).expanduser()
        if target.suffix.lower() != ".asy":
            target = target.with_suffix(".asy")
        target = target.resolve(strict=False)

        temporary_path: Path | None = None
        try:
            if not target.parent.is_dir():
                raise FileNotFoundError(
                    f"Save directory does not exist: {target.parent}"
                )
            if target.exists() and not target.is_file():
                raise OSError(f"Save target is not a regular file: {target}")
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self._document(), stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        except Exception as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            QMessageBox.critical(
                self,
                "Assembly Save Failed",
                f"Could not save the assembly tree:\n{exc}",
            )
            return False

        self._assembly_path = target
        self._set_dirty(False)
        self._notify(f"Assembly tree saved: {target}")
        return True

    def _build(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            roots = [
                self.tree.invisibleRootItem().child(i)
                for i in range(self.tree.invisibleRootItem().childCount())
                if not _is_preview_item(self.tree.invisibleRootItem().child(i))
            ]
            if len(roots) == 1:
                item = roots[0]
            else:
                self._notify("Select a root or branch to build first.")
                return
        if _is_preview_item(item):
            self._notify(
                "Feature preview geometry cannot be built as a response assembly. "
                "Use the feature workflow Build action."
            )
            return
        if item.data(0, _ROLE_TYPE) not in (_TYPE_ROOT, _TYPE_BRANCH):
            self._notify("Select a root or branch (not a leaf) to build.")
            return

        dlg = BuildDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        axis_mode = dlg.axis_mode()
        coherent_metadata_attested = dlg.coherent_metadata_attested()

        try:
            grid, history = build_assembly_grid(
                item,
                axis_mode=axis_mode,
                coherent_metadata_attested=coherent_metadata_attested,
            )
        except Exception as exc:
            self._notify(f"Build failed: {exc}")
            return
        if grid is None:
            self._notify(
                "Build produced no data (subtree has no enabled, loaded leaves)."
            )
            return
        self.platform_built.emit(item.text(0), grid, history)

    def _notify(self, text: str) -> None:
        # Surface a transient error through the main window's status bar if
        # we can find it; otherwise just print so the user has *something*.
        try:
            self.window().status.showMessage(text)
        except Exception:
            print(text)

    def _load(self) -> bool:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Assembly Tree", "", "Assembly Files (*.asy)"
        )
        if not path:
            return False
        if not self._confirm_unsaved_changes(
            action="loading another assembly", parent=self
        ):
            return False

        try:
            with open(path, "r", encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                raise ValueError("Assembly file root must be a JSON object.")
            version = data.get("version", 1)
            if type(version) is not int or not 1 <= version <= 5:
                raise ValueError(
                    "Assembly file 'version' must be an integer from 1 through 5."
                )
            raw_nodes = data.get("tree")
            if not isinstance(raw_nodes, list):
                raise ValueError("Assembly file 'tree' must be a JSON list.")
            # Decode and validate every response node while the current tree and
            # runtime preview are still intact. A malformed .asy must not destroy
            # the user's live workspace before reporting its error.
            loaded_items = [_dict_to_item(node_dict) for node_dict in raw_nodes]
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Assembly Load Failed",
                f"Could not load the assembly tree. The current tree was kept.\n\n{exc}",
            )
            return False

        self._suppress_dirty = True
        try:
            self.tree.clear()
            for item in loaded_items:
                self.tree.invisibleRootItem().addChild(item)
            self.tree.expandAll()
            self._sync_show_all_checkbox()
        finally:
            self._suppress_dirty = False
        self._assembly_path = Path(path).expanduser().resolve(strict=False)
        self._set_dirty(False)
        self._notify(f"Assembly tree loaded: {self._assembly_path}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Build dialog + recursive accumulator
# ─────────────────────────────────────────────────────────────────────────────


class BuildDialog(QDialog):
    """Choose how to handle axis mismatch when summing parts in a subtree."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build Platform")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Axis alignment across parts (each leaf's grid may differ "
            "in azimuth / elevation / frequency / polarization):"
        ))
        self._btn_group = QButtonGroup(self)
        self._radio_intersect = QRadioButton(
            "Intersect — keep only axis values present in every part (no interpolation, lossless)."
        )
        self._radio_interp = QRadioButton(
            "Interpolate — power-only builds; resample onto a common grid "
            "(no extrapolation)."
        )
        self._radio_strict = QRadioButton(
            "Strict — require every part to share exactly the same axes (error if any differs)."
        )
        self._radio_intersect.setChecked(True)
        self._btn_group.addButton(self._radio_intersect, 0)
        self._btn_group.addButton(self._radio_interp, 1)
        self._btn_group.addButton(self._radio_strict, 2)
        layout.addWidget(self._radio_intersect)
        layout.addWidget(self._radio_interp)
        layout.addWidget(self._radio_strict)

        self.chk_coherent_attestation = QCheckBox(
            "I attest every Field + response was solved or translated into "
            "the same vehicle frame, with one phase center, attitude, and "
            "V/H basis."
        )
        self.chk_coherent_attestation.setToolTip(
            "Required whenever a build contains two or more Field + inputs, "
            "even when their metadata labels match. This confirms physical "
            "registration in one vehicle frame, not merely matching text. "
            "Explicitly conflicting declarations can never be overridden; "
            "the confirmation is recorded in provenance and history."
        )
        layout.addWidget(self.chk_coherent_attestation)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def axis_mode(self) -> str:
        if self._radio_interp.isChecked():
            return "interp"
        if self._radio_strict.isChecked():
            return "strict"
        return "intersect"

    def coherent_metadata_attested(self) -> bool:
        return bool(self.chk_coherent_attestation.isChecked())


def _intersect_numeric_axes(arrays, tol: float = 1e-6) -> np.ndarray:
    """Intersect a list of numeric 1-D arrays element-wise, with absolute
    tolerance for floating-point matches. Preserves the first array's
    ordering.
    """
    from grim_dataset import RcsGrid

    return RcsGrid._axis_intersection(arrays, tol=tol)


def _intersect_categorical_axes(arrays) -> np.ndarray:
    """Intersection of categorical (string) axes preserving the first axis order."""
    if not arrays:
        return np.asarray([], dtype=object)
    first = list(arrays[0])
    others = [set(arr.tolist() if hasattr(arr, "tolist") else list(arr)) for arr in arrays[1:]]
    keep = [v for v in first if all(v in s for s in others)]
    return np.asarray(keep)


def _interp_target_axes(grids) -> tuple:
    """For interp mode: target axes = the first grid's values restricted to the
    common axis range across all grids, plus the categorical intersection of
    polarizations. This keeps every output sample within every part's support
    so RcsGrid.align_to('interp') never has to extrapolate.
    """
    ref = grids[0]

    def _clipped(axis_name: str) -> np.ndarray:
        ref_axis = np.asarray(getattr(ref, axis_name), dtype=float)
        if ref_axis.size == 0:
            return ref_axis
        lo = max(np.asarray(getattr(g, axis_name), dtype=float).min() for g in grids)
        hi = min(np.asarray(getattr(g, axis_name), dtype=float).max() for g in grids)
        mask = (ref_axis >= lo - 1e-9) & (ref_axis <= hi + 1e-9)
        # Values admitted by the round-off tolerance must still be placed
        # strictly inside every source grid's support before align_to().
        return np.unique(np.clip(ref_axis[mask], lo, hi))

    az = _clipped("azimuths")
    el = _clipped("elevations")
    f = _clipped("frequencies")
    pol = _intersect_categorical_axes([g.polarizations for g in grids])
    return az, el, f, pol


def _axes_only_grid(az, el, f, pol, reference=None):
    """Construct a stub RcsGrid carrying only axis arrays — used as the target
    passed to RcsGrid.align_to() so we can align every part to the same set
    of axes without inventing a synthetic data array each time."""
    from grim_dataset import RcsGrid

    shape = (len(az), len(el), len(f), len(pol))
    zero = np.zeros(shape, dtype=np.float32)
    return RcsGrid(
        np.asarray(az), np.asarray(el), np.asarray(f), np.asarray(pol),
        rcs=None, rcs_power=zero, rcs_phase=zero,
        rcs_domain="power_phase",
        units=dict(reference.units or {}) if reference is not None else None,
        extra=(
            {"phase_reference": reference.extra["phase_reference"]}
            if reference is not None and "phase_reference" in reference.extra
            else None
        ),
    )


def _coherent_target_indices(grid, target) -> tuple[list[int], ...]:
    """Map one target grid to source indices using GRIM's physical tolerances."""

    from grim_dataset import _ANGLE_UNITS, _FREQUENCY_UNITS

    az_unit = grid._supported_unit(
        "azimuth", _ANGLE_UNITS, "deg"
    )
    el_unit = grid._supported_unit(
        "elevation", _ANGLE_UNITS, "deg"
    )
    canonical_frequency = grid._supported_unit(
        "frequency", _FREQUENCY_UNITS, "GHz"
    )
    tolerances = (
        float(np.deg2rad(1.0e-6)) if az_unit == "rad" else 1.0e-6,
        float(np.deg2rad(1.0e-6)) if el_unit == "rad" else 1.0e-6,
        {"Hz": 1.0e3, "kHz": 1.0, "MHz": 1.0e-3, "GHz": 1.0e-6}[
            canonical_frequency
        ],
        0.0,
    )
    indices = []
    for source_axis, target_axis, tolerance, label in zip(
        (
            grid.azimuths,
            grid.elevations,
            grid.frequencies,
            grid.polarizations,
        ),
        (
            target.azimuths,
            target.elevations,
            target.frequencies,
            target.polarizations,
        ),
        tolerances,
        ("azimuth", "elevation", "frequency", "polarization"),
    ):
        matched = grid._indices_for_axis_values(
            source_axis, target_axis, tol=tolerance
        )
        if matched is None or len(matched) != len(target_axis):
            raise ValueError(
                f"could not map coherent {label} samples to the assembly target"
            )
        indices.append(matched)
    return tuple(indices)


def _align_coherent_grid(grid, target, *, mode: str):
    """Align authoritative complex field samples without float32 round-trip."""

    from grim_dataset import RcsGrid

    if mode == "intersect":
        indices = _coherent_target_indices(grid, target)
        field = grid.rcs[np.ix_(*indices)]
    elif mode == "interp":
        polarization_indices = grid._indices_for_axis_values(
            grid.polarizations, target.polarizations, tol=0.0
        )
        if (
            polarization_indices is None
            or len(polarization_indices) != len(target.polarizations)
        ):
            raise ValueError("polarization axis mismatch for coherent interp")
        field = grid.rcs[..., polarization_indices]
        for axis, old, new, label in (
            (0, grid.azimuths, target.azimuths, "azimuth"),
            (1, grid.elevations, target.elevations, "elevation"),
            (2, grid.frequencies, target.frequencies, "frequency"),
        ):
            grid._check_axis_sorted(old, label)
            field = grid._interp_complex_axis(field, old, new, axis)
    else:
        raise ValueError(f"unsupported coherent alignment mode {mode!r}")
    aligned = RcsGrid(
        target.azimuths,
        target.elevations,
        target.frequencies,
        target.polarizations,
        rcs=np.asarray(field, dtype=np.complex128),
        units=copy.deepcopy(grid.units or {}),
    )
    # RcsGrid stores its public interchange representation as power/phase.
    # Keep the exact aligned field out-of-band for this one in-memory build so
    # a near-null does not round-trip through angle/square-root reconstruction.
    aligned._assembly_authoritative_complex = np.asarray(
        field, dtype=np.complex128
    )
    return aligned


def _align_grids_for_assembly(
    grids,
    axis_mode: str,
    *,
    coherent_count: int | None = None,
) -> list:
    """Align every grid in `grids` to a shared set of axes per `axis_mode`.

    strict: validate every grid has identical axes; no resampling.
    intersect: take the pairwise intersection of every axis; no interpolation.
    interp: build a common grid clipped to the overlapping range and resample
            power-only contributors. Coherent Field + interpolation is refused
            because it can change field phase/amplitude relationships.
    """
    if not grids:
        return []
    if coherent_count is None:
        coherent_count = 0
    if (
        type(coherent_count) is not int
        or coherent_count < 0
        or coherent_count > len(grids)
    ):
        raise ValueError("coherent_count must index the leading Field + grids")
    if axis_mode == "interp" and coherent_count:
        raise ValueError(
            "Interpolate is disabled when a build contains Field + responses. "
            "Use Strict for identical sampling or Intersect for shared exact "
            "axis values; coherent fields are never resampled."
        )
    ref = grids[0]
    if axis_mode == "strict":
        for g in grids[1:]:
            ref._assert_compatible(g)
        return list(grids)

    if axis_mode == "intersect":
        az = _intersect_numeric_axes([g.azimuths for g in grids])
        el = _intersect_numeric_axes([g.elevations for g in grids])
        f = _intersect_numeric_axes([g.frequencies for g in grids])
        pol = _intersect_categorical_axes([g.polarizations for g in grids])
        if az.size == 0 or el.size == 0 or f.size == 0 or pol.size == 0:
            raise ValueError(
                "intersect: parts have no common azimuth/elevation/frequency/polarization values"
            )
        target = _axes_only_grid(az, el, f, pol, ref)
        return [
            _align_coherent_grid(grid, target, mode="intersect")
            if index < coherent_count
            else grid.align_to(target, mode="intersect")
            for index, grid in enumerate(grids)
        ]

    if axis_mode == "interp":
        az, el, f, pol = _interp_target_axes(grids)
        if az.size == 0 or el.size == 0 or f.size == 0 or pol.size == 0:
            raise ValueError(
                "interp: parts have no overlapping axis range across all parts"
            )
        target = _axes_only_grid(az, el, f, pol, ref)
        # Numeric interpolation requires identical categorical axes. Subset
        # and reorder polarization first without discarding the source numeric
        # samples needed to support interpolation.
        aligned = []
        for index, grid in enumerate(grids):
            if index < coherent_count:
                aligned.append(_align_coherent_grid(grid, target, mode="interp"))
                continue
            prepared = (
                grid
                if np.array_equal(grid.polarizations, pol)
                else grid.axis_crop(polarizations=pol)
            )
            aligned.append(prepared.align_to(target, mode="interp"))
        return aligned

    raise ValueError(f"unknown axis_mode {axis_mode!r}")


def _validate_coherent_sources(grids, *, metadata_attested: bool) -> None:
    if not isinstance(metadata_attested, (bool, np.bool_)):
        raise TypeError("coherent_metadata_attested must be True or False")
    for grid in grids:
        if _declared_combine_role(grid) == _COMBINE_ROLE_POWER:
            raise ValueError(
                "a response explicitly tagged combine_role='power' cannot be "
                "used as a coherent pre-aligned Field + child"
            )
    if len(grids) < 2:
        return
    if not metadata_attested:
        raise ValueError(
            "two or more Field + responses require explicit attestation that "
            "every coherent child was solved or translated into one common "
            "vehicle frame, phase center, attitude, and earth V/H basis; "
            "matching metadata labels alone are not proof of registration"
        )
    reference = grids[0]
    for grid in grids[1:]:
        reference._assert_coherent_metadata_compatible(
            grid, metadata_attested=metadata_attested
        )


def _result_response_role(
    source_grids,
    occupied_sources,
    occupied_responses,
    *,
    has_coh,
    has_incoh,
):
    if has_coh and has_incoh:
        return _RESPONSE_ROLE_MIXED_SUM
    if has_incoh:
        return _RESPONSE_ROLE_INCOHERENT_SUM
    explicit_roles = [_declared_response_role(grid) for grid in source_grids]
    body_count = max(len(occupied_sources), len(occupied_responses))
    if body_count:
        return (
            _RESPONSE_ROLE_BODY_PLUS_FEATURES
            if body_count == 1
            else _RESPONSE_ROLE_COHERENT_SUM
        )
    if explicit_roles and all(
        role == _RESPONSE_ROLE_FEATURES_ONLY_DELTA for role in explicit_roles
    ):
        return _RESPONSE_ROLE_FEATURES_ONLY_DELTA
    return _RESPONSE_ROLE_COHERENT_SUM


def _combined_semantic_extra(
    source_grids,
    source_coh_grids,
    *,
    has_coh: bool,
    has_incoh: bool,
    coherent_metadata_attested: bool,
    body_occupancy=None,
    component_signatures=None,
):
    """Build honest scalar metadata for a derived assembly response."""

    source_grids = list(source_grids)
    extra = {}
    coherent_keys = {
        "phase_reference",
        "time_convention",
        "amplitude_convention",
        "complex_field_domain",
    }
    for key in _SEMANTIC_SCALAR_KEYS:
        relevant = source_coh_grids if key in coherent_keys else source_grids
        if not relevant:
            continue
        values = [_declared_extra_scalar(grid, key) for grid in relevant]
        declared = [value for value in values if value is not None]
        canonical = {
            _canonical_metadata_value(key, value) for value in declared
        }
        if len(canonical) > 1 and key in _STRICT_SEMANTIC_KEYS:
            raise ValueError(
                f"refusing to combine parts with contradictory {key} metadata"
            )
        # A shared result declaration is valid only when every contributing
        # source made it. Unknown inputs do not inherit another input's claim.
        if len(declared) == len(relevant) and len(canonical) == 1:
            extra[key] = copy.deepcopy(declared[0])

    # No unique phase/time/amplitude exists after any incoherent contribution.
    if has_incoh:
        for key in coherent_keys:
            extra.pop(key, None)

    if body_occupancy is None:
        occupied_sources, occupied_responses = (
            _assert_no_shared_body_totals(source_grids)
        )
    else:
        occupied_sources = set(body_occupancy[0])
        occupied_responses = set(body_occupancy[1])
    if component_signatures is None:
        component_signatures = _assert_no_duplicate_feature_components(
            source_grids
        )
    else:
        component_signatures = set(component_signatures)
    base_source_hashes = set()
    base_response_hashes = set()
    delta_source_hashes = set()
    delta_response_hashes = set()
    delta_only_flags = []
    for grid in source_grids:
        base_source_hashes.update(_assembly_base_reference_hashes(grid))
        base_response_hashes.update(
            _assembly_base_response_reference_hashes(grid)
        )
        delta_sources, delta_responses, delta_only = (
            _feature_delta_context(grid)
        )
        delta_source_hashes.update(delta_sources)
        delta_response_hashes.update(delta_responses)
        delta_only_flags.append(delta_only)
    feature_delta_only = bool(delta_only_flags) and all(delta_only_flags)
    response_role = _result_response_role(
        source_grids,
        occupied_sources,
        occupied_responses,
        has_coh=has_coh,
        has_incoh=has_incoh,
    )
    combine_role = (
        _COMBINE_ROLE_COHERENT
        if has_coh and not has_incoh
        else _COMBINE_ROLE_POWER
    )
    extra[_COMBINE_ROLE_KEY] = combine_role
    extra["combine_role_note"] = (
        "pre-aligned complex field sum"
        if combine_role == _COMBINE_ROLE_COHERENT
        else "incoherent power result; no common output phase"
    )
    extra[_RESPONSE_ROLE_KEY] = response_role
    if (
        response_role == _RESPONSE_ROLE_BODY_PLUS_FEATURES
        and len(occupied_sources) == 1
    ):
        digest = next(iter(occupied_sources))
        extra[_SOURCE_MONOSTATIC_SHA256_KEY] = digest
        extra[_ASSEMBLY_BASE_SHA256_KEY] = digest
    elif response_role == _RESPONSE_ROLE_FEATURES_ONLY_DELTA:
        if len(base_source_hashes) != 1:
            raise ValueError(
                "features_only_delta sum must retain exactly one assembly base hash"
            )
        extra[_ASSEMBLY_BASE_SHA256_KEY] = next(iter(base_source_hashes))
    elif delta_source_hashes and len(base_source_hashes) == 1:
        # Power+ or mixed transitions must not erase the base context of a
        # feature delta; otherwise a later unrelated body could accept it.
        extra[_ASSEMBLY_BASE_SHA256_KEY] = next(iter(base_source_hashes))
    if (
        len(base_response_hashes) == 1
        and max(len(occupied_sources), len(occupied_responses)) <= 1
    ):
        extra[_ASSEMBLY_BASE_RESPONSE_SHA256_KEY] = next(
            iter(base_response_hashes)
        )

    # Preserve source provenance for a one-input conversion. For a real
    # multi-input build the aggregate record below replaces, rather than
    # impersonates, one child's feature/assembly provenance.
    if len(source_grids) == 1:
        source = source_grids[0]
        for key in (_FEATURE_PROVENANCE_KEY, _ASSEMBLY_PROVENANCE_KEY):
            value = _declared_extra_scalar(source, key)
            if value is not None:
                extra[key] = copy.deepcopy(value)

    record = {
        "schema": "grim.assembly-response.v1",
        "response_role": response_role,
        "combine_role": combine_role,
        "input_count": len(source_grids),
        "coherent_input_count": len(source_coh_grids),
        "incoherent_input_count": len(source_grids) - len(source_coh_grids),
        "body_plus_features_source_sha256": sorted(occupied_sources),
        "body_plus_features_base_response_sha256": sorted(
            occupied_responses
        ),
        "assembly_base_sha256": sorted(base_source_hashes),
        "assembly_base_response_sha256": sorted(base_response_hashes),
        "feature_delta_base_sha256": sorted(delta_source_hashes),
        "feature_delta_base_response_sha256": sorted(
            delta_response_hashes
        ),
        "feature_delta_only": feature_delta_only,
        "feature_component_signatures": sorted(component_signatures),
        "coherent_metadata_attested": bool(
            coherent_metadata_attested and len(source_coh_grids) > 1
        ),
    }
    extra[_ASSEMBLY_PROVENANCE_KEY] = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

    if coherent_metadata_attested and len(source_coh_grids) > 1:
        _history, attested_extra = source_coh_grids[0]._coherent_attestation_provenance(
            source_coh_grids[1:],
            operation="assembly-field-add",
            metadata_attested=True,
        )
        if attested_extra:
            raw_record = attested_extra.get(
                "coherent_metadata_attestation_json"
            )
            if raw_record:
                attestation_record = json.loads(raw_record)
                attestation_record["attested_scope"] = [
                    "common_vehicle_frame",
                    "common_phase_center",
                    "common_attitude",
                    "common_earth_vh_basis",
                    "phasor_time_convention",
                ]
                attestation_record["attestation_statement"] = (
                    _COHERENT_VEHICLE_FRAME_ATTESTATION
                )
                attested_extra["coherent_metadata_attestation_json"] = (
                    json.dumps(
                        attestation_record,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
            extra.update(attested_extra)
    return extra


def _combined_result_units(ref, source_grids, semantic_extra, *, has_incoh: bool):
    """Retain physical units without laundering one input's convention tags."""

    units = copy.deepcopy(ref.units or {})
    for key in list(units):
        key_text = str(key)
        if (
            key in _DYNAMIC_RESULT_METADATA_KEYS
            or key_text.endswith("_provenance_json")
            or key_text.startswith("combine_")
            or key_text.startswith("feature_delta_")
            or key_text.startswith("body_plus_features_")
            or (
                key_text.startswith("assembly_")
                and key_text != "assembly_angular_coordinate_contract"
            )
        ):
            units.pop(key, None)
    field_only_keys = {
        "phase_reference",
        "time_convention",
        "amplitude_convention",
        "complex_field_domain",
    }
    for key in _SEMANTIC_SCALAR_KEYS:
        units.pop(key, None)
        if has_incoh and key in field_only_keys:
            continue
        if key not in semantic_extra:
            continue
        # Keep a shared convention in units when at least one producer used
        # that modeled container. It also remains in ``extra`` so older readers
        # that looked there continue to see the same, noncontradictory value.
        if any(key in (grid.units or {}) for grid in source_grids):
            units[key] = copy.deepcopy(semantic_extra[key])
    return units


def _combine_children(
    coh_grids,
    incoh_grids,
    ref,
    *,
    metadata_sources=None,
    coherent_metadata_attested=False,
    validated_body_occupancy=None,
    validated_component_signatures=None,
):
    """Coherent-field + incoherent-power combination of pre-aligned children.

    coh_grids contribute by complex sum   →  C_coh = Σ rcs
    incoh_grids contribute by power sum  →  P_incoh = Σ rcs_power
    output power   = |C_coh|² + P_incoh
    output phase   = arg(C_coh)   (NaN if there are no coherent contributors)
    """
    from grim_dataset import C0, RcsGrid

    if not isinstance(coherent_metadata_attested, (bool, np.bool_)):
        raise TypeError("coherent_metadata_attested must be True or False")
    if metadata_sources is None:
        source_grids = list(coh_grids) + list(incoh_grids)
    else:
        source_grids = list(metadata_sources)
    if len(source_grids) != len(coh_grids) + len(incoh_grids):
        raise ValueError("metadata_sources must correspond one-for-one with children")
    source_coh_grids = source_grids[:len(coh_grids)]
    _validate_coherent_sources(
        source_coh_grids,
        metadata_attested=bool(coherent_metadata_attested),
    )
    body_occupancy = (
        _assert_no_shared_body_totals(source_grids)
        if validated_body_occupancy is None
        else (
            set(validated_body_occupancy[0]),
            set(validated_body_occupancy[1]),
        )
    )
    component_signatures = (
        _assert_no_duplicate_feature_components(source_grids)
        if validated_component_signatures is None
        else set(validated_component_signatures)
    )

    C_coh = None
    if coh_grids:
        for g in coh_grids:
            missing = np.isfinite(g.rcs_power) & ~np.isfinite(g.rcs_phase)
            if np.any(missing):
                raise ValueError(
                    "a branch containing incoherent or phase-unknown data cannot "
                    "be used as a coherent assembly child"
                )
        def _field(grid):
            aligned_field = getattr(
                grid, "_assembly_authoritative_complex", None
            )
            return grid.rcs if aligned_field is None else aligned_field

        C_coh = np.array(_field(coh_grids[0]), copy=True)
        for g in coh_grids[1:]:
            C_coh = C_coh + _field(g)

    P_incoh = None
    if incoh_grids:
        P_incoh = np.array(incoh_grids[0].rcs_power, copy=True)
        for g in incoh_grids[1:]:
            P_incoh = P_incoh + g.rcs_power

    if C_coh is not None and P_incoh is not None:
        total_power = np.abs(C_coh) ** 2 + P_incoh
        # No single field phase exists for a coherent-field plus statistically
        # independent power sum. Mark it unknown so an ancestor cannot silently
        # reinterpret the incoherent contribution as a coherent field.
        total_phase = np.full(total_power.shape, np.nan, dtype=total_power.dtype)
    elif C_coh is not None:
        total_power = np.abs(C_coh) ** 2
        total_phase = np.angle(C_coh)
    else:
        total_power = np.array(P_incoh, copy=True)
        total_phase = np.full(total_power.shape, np.nan, dtype=total_power.dtype)

    extra = _combined_semantic_extra(
        source_grids,
        source_coh_grids,
        has_coh=C_coh is not None,
        has_incoh=P_incoh is not None,
        coherent_metadata_attested=bool(coherent_metadata_attested),
        body_occupancy=body_occupancy,
        component_signatures=component_signatures,
    )
    result_units = _combined_result_units(
        ref,
        source_grids,
        extra,
        has_incoh=P_incoh is not None,
    )

    if C_coh is not None and P_incoh is None:
        quantity = ref.linear_quantity()
        if quantity == "sigma_3d":
            raw_amplitude = C_coh / np.sqrt(4.0 * np.pi)
        elif quantity == "sigma_2d":
            frequency_hz = ref._frequency_value_to_hz(ref.frequencies)
            k0 = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float) / C0
            raw_amplitude = C_coh * (
                2.0 * np.sqrt(k0)[None, None, :, None]
            )
        else:
            raw_amplitude = None
        if raw_amplitude is not None:
            extra["rcs_amp_real"] = np.asarray(
                raw_amplitude.real, dtype=np.float64
            )
            extra["rcs_amp_imag"] = np.asarray(
                raw_amplitude.imag, dtype=np.float64
            )
            extra["raw_complex_amplitude_preserved"] = True

    result = RcsGrid(
        ref.azimuths, ref.elevations, ref.frequencies, ref.polarizations,
        rcs=None,
        rcs_power=total_power,
        rcs_phase=total_phase,
        rcs_domain="power_phase",
        units=result_units,
        extra=extra,
    )
    if C_coh is not None and P_incoh is None:
        result._assembly_authoritative_complex = np.asarray(
            C_coh, dtype=np.complex128
        )
    return result


def build_assembly_grid(
    node: QTreeWidgetItem,
    *,
    axis_mode: str = "intersect",
    coherent_metadata_attested: bool = False,
):
    """Recursively materialise an assembly subtree into a single RcsGrid.

    Leaves return their stored grid (or None if empty). Branches/roots gather
    every enabled, non-None child grid, separate them by add-mode, align all
    children to a common axis grid per `axis_mode`, and combine coherent+
    incoherent contributions as in `_combine_children`.

    Returns (grid, history_string). Both are None / "" if the subtree has
    no loaded data.
    """
    if not isinstance(coherent_metadata_attested, (bool, np.bool_)):
        raise TypeError("coherent_metadata_attested must be True or False")
    if _is_preview_item(node):
        raise ValueError(
            "preview-only geometry is not a response assembly; use the feature "
            "workflow to apply point and line responses"
        )
    if not _item_response_included(node):
        return None, ""
    node_type = node.data(0, _ROLE_TYPE)
    if node_type == _TYPE_LEAF:
        grid = node.data(0, _ROLE_GRID)
        if grid is None:
            return None, ""
        return grid, node.text(0)

    coh: list = []
    incoh: list = []
    for i in range(node.childCount()):
        child = node.child(i)
        if _is_preview_item(child):
            continue
        child_grid, child_history = build_assembly_grid(
            child,
            axis_mode=axis_mode,
            coherent_metadata_attested=coherent_metadata_attested,
        )
        if child_grid is None:
            continue
        stored_mode = _node_mode(child)
        child_mode = _resolved_node_mode(child, child_grid)
        if stored_mode == _MODE_AUTO:
            child_history += f" [Auto -> {_MODE_LABEL[child_mode]}]"
        bucket = coh if child_mode == _MODE_COH else incoh
        bucket.append((child_history, child_grid))

    if not coh and not incoh:
        return None, ""

    all_pairs = coh + incoh
    grids_in_order = [g for _, g in all_pairs]

    # Axis values alone do not establish physical compatibility. Refuse unit,
    # quantity, angular-frame, and convention mismatches before alignment can
    # discard source provenance.
    ref_units = grids_in_order[0]
    for grid in grids_in_order[1:]:
        ref_units._assert_physical_metadata_compatible(grid)

    source_coh = grids_in_order[:len(coh)]
    _validate_coherent_sources(
        source_coh,
        metadata_attested=bool(coherent_metadata_attested),
    )
    body_occupancy = _assert_no_shared_body_totals(grids_in_order)
    component_signatures = _assert_no_duplicate_feature_components(
        grids_in_order
    )

    # A one-field branch is only a structural passthrough. Avoid reconstructing
    # it (and thereby laundering feature/SENTRi provenance) when its declared
    # operation is already coherent.
    if len(grids_in_order) == 1 and coh:
        return grids_in_order[0], (
            f"Σ {node.text(0)} ({axis_mode}): coh[{coh[0][0]}]"
        )

    aligned = _align_grids_for_assembly(
        grids_in_order,
        axis_mode=axis_mode,
        coherent_count=len(coh),
    )
    n_coh = len(coh)
    aligned_coh = aligned[:n_coh]
    aligned_incoh = aligned[n_coh:]
    ref = aligned[0]
    result = _combine_children(
        aligned_coh,
        aligned_incoh,
        ref,
        metadata_sources=grids_in_order,
        coherent_metadata_attested=coherent_metadata_attested,
        validated_body_occupancy=body_occupancy,
        validated_component_signatures=component_signatures,
    )

    parts = []
    if coh:
        parts.append("coh[" + " + ".join(h for h, _ in coh) + "]")
    if incoh:
        parts.append("incoh[" + " + ".join(h for h, _ in incoh) + "]")
    history = f"Σ {node.text(0)} ({axis_mode}): " + " ⊕ ".join(parts)
    if len(coh) > 1:
        history += " | User attestation: " + _COHERENT_VEHICLE_FRAME_ATTESTATION
    return result, history
