import numpy as np

from openkarst.models.hydraulics import (
    circular_hydraulic_radius,
    circular_segment_area,
    circular_wetted_perimeter,
    compute_churchill_friction_factor,
    compute_conduit_slope_cosines,
    compute_slot_width,
    compute_upstream_weight_alpha,
    critical_depth_residual,
    find_critical_depth,
)


def test_compute_slot_width_uses_sjoberg_equation_below_threshold():
    flow_depths = np.array([1.0])
    diameters = np.array([1.0])

    result = compute_slot_width(flow_depths, diameters)

    expected = diameters * 0.5423 * np.exp(-np.power(flow_depths / diameters, 2.4))
    np.testing.assert_allclose(result, expected)


def test_compute_churchill_friction_factor_matches_formula():
    reynolds = np.logspace(0, 8, num=9)
    roughness = np.full_like(reynolds, 0.03)
    diameter = np.full_like(reynolds, 1.0)

    result = compute_churchill_friction_factor(reynolds, roughness, diameter)

    c = (7 / reynolds) ** 0.9 + 0.27 * roughness / diameter
    a = (-2.457 * np.log(c)) ** 16
    b = (37530 / reynolds) ** 16
    expected = 8 * ((8 / reynolds) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)
    np.testing.assert_allclose(result, expected)


def test_compute_slot_width_uses_one_percent_diameter_above_threshold():
    flow_depths = np.array([1.79, 6.0])
    diameters = np.array([1.0, 3.0])

    result = compute_slot_width(flow_depths, diameters)

    np.testing.assert_allclose(result, np.array([0.01, 0.03]))


def test_compute_upstream_weight_alpha_uses_froude_ranges_and_pressurization():
    froude_number = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 0.25])
    is_full = np.array([False, False, False, False, False, True])

    result = compute_upstream_weight_alpha(froude_number, is_full)

    np.testing.assert_allclose(result, np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]))


def test_compute_conduit_slope_cosines_from_endpoint_elevations():
    z1 = np.array([0.0, 1.0, 0.0])
    z2 = np.array([0.0, 2.0, 2.0])
    lengths = np.array([3.0, np.sqrt(2.0), 2.0])

    result = compute_conduit_slope_cosines(z1, z2, lengths)

    np.testing.assert_allclose(result, np.array([1.0, 1.0 / np.sqrt(2.0), 0.0]))


def test_circular_geometry_helpers_at_half_full_depth():
    depth = 1.0
    diameter = 2.0

    np.testing.assert_allclose(circular_segment_area(depth, diameter), np.pi / 2)
    np.testing.assert_allclose(circular_wetted_perimeter(depth, diameter), np.pi)
    np.testing.assert_allclose(circular_hydraulic_radius(depth, diameter), 0.5)


def test_find_critical_depth_solves_residual():
    flowrate = 1.0
    diameter = 2.0
    gravity = 9.81

    depth = find_critical_depth(flowrate, diameter, gravity)

    assert 0.0 < depth < diameter
    np.testing.assert_allclose(
        critical_depth_residual(depth, flowrate, gravity, diameter),
        0.0,
        atol=1e-9,
    )
