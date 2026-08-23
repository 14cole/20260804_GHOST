#!/usr/bin/env python3
"""Validate topology and clearance of the checked-in gap geometry sweep."""

import sys
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "Backend"))
sys.path.insert(0, str(STUDY_DIR))

from generate_gap_geometry_study import (  # noqa: E402
    C0_M_PER_S,
    DEFAULT_MAX_BACKING_CHORD_LAMBDA,
    INCHES_PER_METER,
    _minimum_feature_clearance,
)
from geometry_io import parse_geometry  # noqa: E402


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate():
    paths = sorted((STUDY_DIR / "generated").rglob("*.geo"))
    _require(len(paths) == 90, f"expected 90 geometries, found {len(paths)}")

    pairs = {}
    segment_counts = []
    for path in paths:
        frequency_ghz = float(path.parts[-4][:-3])
        role = path.parts[-3]
        clearance_tag = path.parts[-2]
        target_lambda = float(
            clearance_tag[1:-len("lambda")].replace("p", ".")
        )

        title, segments, ibcs, dielectrics = parse_geometry(
            path.read_text(encoding="ascii")
        )
        _require(len(segments) == 1, f"{path}: expected one segment")
        _require(not ibcs, f"{path}: unexpected IBC definitions")
        _require(not dielectrics, f"{path}: unexpected dielectric definitions")

        segment = segments[0]
        edges = list(zip(
            zip(segment.x[0::2], segment.y[0::2]),
            zip(segment.x[1::2], segment.y[1::2]),
        ))
        metadata = dict(
            token.split("=", 1)
            for token in title.split()
            if "=" in token
        )
        backing_segment_count = int(metadata["backing_segments"])
        segment_counts.append(backing_segment_count)
        expected_edges = backing_segment_count + (3 if role == "FRD" else 5)
        _require(
            len(edges) == expected_edges,
            f"{path}: expected {expected_edges} edges, found {len(edges)}",
        )
        _require(
            all(start != end for start, end in edges),
            f"{path}: zero-length edge",
        )
        for index, (_start, end) in enumerate(edges):
            next_start = edges[(index + 1) % len(edges)][0]
            _require(
                abs(end[0] - next_start[0]) <= 1.0e-9
                and abs(end[1] - next_start[1]) <= 1.0e-9,
                f"{path}: disconnected edge {index}",
            )

        twice_area = sum(
            start[0] * end[1] - end[0] * start[1]
            for start, end in edges
        )
        _require(twice_area < 0.0, f"{path}: expected clockwise winding")

        backing_edges = edges[-backing_segment_count:]
        backing = [backing_edges[0][0]] + [
            end for _start, end in backing_edges
        ]
        wavelength_in = (
            C0_M_PER_S / (frequency_ghz * 1.0e9) * INCHES_PER_METER
        )
        achieved_lambda = _minimum_feature_clearance(backing) / wavelength_in
        maximum_chord_lambda = max(
            ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
            for start, end in backing_edges
        ) / wavelength_in
        _require(
            achieved_lambda >= target_lambda * (1.0 - 2.0e-8),
            f"{path}: achieved {achieved_lambda:g} lambda, "
            f"expected at least {target_lambda:g}",
        )
        _require(
            maximum_chord_lambda
            <= DEFAULT_MAX_BACKING_CHORD_LAMBDA * (1.0 + 2.0e-8),
            f"{path}: maximum backing chord is "
            f"{maximum_chord_lambda:g} lambda",
        )

        key = (frequency_ghz, clearance_tag)
        pairs.setdefault(key, {})[role] = tuple(backing_edges)

    _require(len(pairs) == 45, f"expected 45 pairs, found {len(pairs)}")
    for key, roles in pairs.items():
        _require(set(roles) == {"FRD", "OPN"}, f"{key}: incomplete pair")
        _require(
            roles["FRD"] == roles["OPN"],
            f"{key}: FRD/OPN backing mismatch",
        )

    print(
        "Validated 90 geometries: 45 matched FRD/OPN pairs, closed "
        "clockwise, requested clearance achieved, maximum backing chord "
        f"<= {DEFAULT_MAX_BACKING_CHORD_LAMBDA:g} lambda "
        f"({min(segment_counts)}--{max(segment_counts)} chords)."
    )


if __name__ == "__main__":
    validate()
