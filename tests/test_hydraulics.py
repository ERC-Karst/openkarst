import numpy as np

from openkarst.models.hydraulics import compute_slot_width


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
