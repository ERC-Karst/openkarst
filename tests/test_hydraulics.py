import numpy as np

from openkarst.models.hydraulics import (
    circular_hydraulic_radius,
    circular_segment_area,
    circular_wetted_perimeter,
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
