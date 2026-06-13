"""Hydraulic formula helpers for the Saint-Venant model."""

import numpy as np
import scipy.optimize as optimize


def compute_churchill_friction_factor(reynolds, roughness, diameter):
    """Compute Darcy friction factor with Churchill's equation."""
    c = (7 / reynolds) ** 0.9 + 0.27 * roughness / diameter
    a = (-2.457 * np.log(c)) ** 16
    b = (37530 / reynolds) ** 16
    return 8 * ((8 / reynolds) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)


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


def compute_upstream_weight_alpha(froude_number, is_full):
    """
    Compute upstream-weighting alpha from Froude number and pressurization state.

    Alpha is 1.0 for Froude numbers up to 0.5, decreases linearly between
    0.5 and 1.0, and is zero for supercritical or pressurized flow.
    """
    alpha = np.zeros_like(froude_number, dtype=float)

    low_froude = froude_number <= 0.5
    transitional_froude = np.logical_and(froude_number > 0.5, froude_number < 1.0)

    alpha[low_froude] = 1.0
    alpha[transitional_froude] = 2 * (1 - froude_number[transitional_froude])
    alpha[is_full] = 0.0

    return alpha


def circular_segment_area(depth, diameter):
    """Calculate flow area in a circular conduit at a given depth."""
    radius = diameter / 2
    theta = 2 * np.arccos((radius - depth) / radius)
    return (radius**2 / 2) * (theta - np.sin(theta))


def circular_wetted_perimeter(depth, diameter):
    """Calculate wetted perimeter in a circular conduit at a given depth."""
    radius = diameter / 2
    theta = 2 * np.arccos((radius - depth) / radius)
    return radius * theta


def circular_hydraulic_radius(depth, diameter):
    """Calculate hydraulic radius in a circular conduit at a given depth."""
    area = circular_segment_area(depth, diameter)
    perimeter = circular_wetted_perimeter(depth, diameter)
    return area / perimeter


def critical_depth_residual(depth, flowrate, gravity, diameter):
    """Calculate the circular-conduit critical-depth residual."""
    perimeter = circular_wetted_perimeter(depth, diameter)
    area = circular_segment_area(depth, diameter)
    return (flowrate**2 * perimeter) / (gravity * area**3) - 1


def find_critical_depth(flowrate, diameter, gravity=9.81):
    """Find critical depth in a circular conduit."""
    initial_guess = 1.01 * (flowrate**2 / gravity)**0.25 / (diameter**0.26)
    return optimize.fsolve(
        critical_depth_residual,
        initial_guess,
        args=(flowrate, gravity, diameter),
    )[0]
