"""Hydraulic formula helpers for the Saint-Venant model."""

import numpy as np


def compute_slot_width(flow_depths, diameters):
    """
    Compute Preissmann slot width for circular conduits.

    The width is based on normalized flow depth. If normalized depth is
    greater than 1.78, the slot width is set to 1% of the conduit diameter.
    Otherwise, it uses the Sjoberg equation from SWMM.
    """
    y_norm = flow_depths / diameters
    width_max = diameters

    return np.where(
        y_norm > 1.78,
        0.01 * width_max,
        width_max * 0.5423 * np.exp(-np.power(y_norm, 2.4)),
    )
