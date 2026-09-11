#!/usr/bin/env python3
"""Coordinate frames and polarization conventions for feature placement."""

import numpy as np


CAD2AXIS = np.array([[0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]])


AXIS_AZ_DEG = 0.0
AXIS_EL_DEG = 0.0
ROLL_DEG = 0.0

UNIT_SCALE = {"meters": 1.0, "m": 1.0, "mm": 1e-3, "millimeters": 1e-3,
              "inches": 0.0254, "in": 0.0254, "inch": 0.0254,
              "ft": 0.3048, "feet": 0.3048}


def scale_for(units):
    key = str(units).strip().lower()
    if key not in UNIT_SCALE:
        raise SystemExit(f"unknown UNITS {units!r} -- use one of "
                         f"{sorted(set(UNIT_SCALE))}.")
    return UNIT_SCALE[key]


def to_axis_frame(v):
    """CAD coordinates -> solver coordinates (any array whose last axis is xyz)."""
    return np.asarray(v, float) @ CAD2AXIS.T
