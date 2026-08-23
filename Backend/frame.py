#!/usr/bin/env python3
"""
THE FRAME CONVENTION -- stated once, used by every step.

You draw everything (the STL and the door coordinates) in the CAD frame:

        +y = NOSE        +x = RIGHT        +z = UP

The BoR solve cannot be reoriented: it is axisymmetric about its own +z by
construction, so the steps rotate your CAD coordinates into that solver frame
(solver x = CAD up, y = CAD right, z = CAD nose).  You never see the solver
frame.  A body profile is frame-free -- it is (rho, z), rho = distance from the
axis, z = along the nose.

ATTITUDE IS FIXED, NOT A KNOB: the vehicle is LEVEL, UPRIGHT, NOSE AT AZIMUTH 0.
So the output azimuth/elevation are VEHICLE-RELATIVE angles, and every door shows
up at the angle where it actually sits in your CAD:

        right side (+x) -> azimuth 270        left side (-x) -> azimuth  90
        top       (+z) -> elevation +90       bottom   (-z) -> elevation -90
        nose      (+y) -> azimuth   0

A heading or a bank angle is a rigid rotation of the finished result; it belongs
in a scene-level step, not in a component trade study.
"""

import numpy as np

# v_solver = v_cad @ CAD2AXIS.T   (a pure rotation, det = +1)
CAD2AXIS = np.array([[0.0, 0.0, 1.0],       # solver x = CAD up
                     [1.0, 0.0, 0.0],       # solver y = CAD right
                     [0.0, 1.0, 0.0]])      # solver z = CAD nose

# the fixed attitude: level, upright, nose at azimuth 0
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
