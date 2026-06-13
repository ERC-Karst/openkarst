import numpy as np

from openkarst.models.cross_section_geometry import (
    CircularAnalyticalGeometry,
    CircularTabulatedGeometry,
    create_cross_section_geometry,
)


def test_circular_tabulated_geometry_matches_analytical_geometry():
    depth_factors = np.array(
        [0.0, 1e-8, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0, 1.1],
        dtype=float,
    )

    for diameter in (0.25, 1.0, 3.0):
        depths = depth_factors * diameter
        analytical = CircularAnalyticalGeometry(diameter)
        tabulated = CircularTabulatedGeometry(diameter, n_points=5001)

        np.testing.assert_allclose(
            tabulated.full_area(),
            analytical.full_area(),
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            tabulated.full_perimeter(),
            analytical.full_perimeter(),
            rtol=1e-14,
            atol=1e-14,
        )

        strict_mask = depth_factors != 1e-8

        np.testing.assert_allclose(
            tabulated.area(depths[strict_mask]),
            analytical.area(depths[strict_mask]),
            rtol=3e-6,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            tabulated.wetted_perimeter(depths[strict_mask]),
            analytical.wetted_perimeter(depths[strict_mask]),
            rtol=2e-4,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            tabulated.hydraulic_radius(depths[strict_mask]),
            analytical.hydraulic_radius(depths[strict_mask]),
            rtol=2e-4,
            atol=1e-12,
        )

        tiny_depth = 1e-8 * diameter
        assert np.isfinite(tabulated.area(tiny_depth))
        assert np.isfinite(tabulated.wetted_perimeter(tiny_depth))
        assert np.isfinite(tabulated.hydraulic_radius(tiny_depth))
        assert tabulated.area(tiny_depth) >= 0.0
        assert tabulated.wetted_perimeter(tiny_depth) >= 0.0
        assert tabulated.hydraulic_radius(tiny_depth) >= 0.0


def test_circular_tabulated_area_is_monotonic_and_clamped():
    diameter = 2.0
    geometry = CircularTabulatedGeometry(diameter)
    depths = np.linspace(-0.1 * diameter, 1.1 * diameter, 2000)

    areas = geometry.area(depths)
    valid_depth_areas = geometry.area(np.linspace(0.0, diameter, 2000))

    assert np.all(areas >= 0.0)
    assert np.all(np.diff(valid_depth_areas) >= -1e-14)
    assert geometry.area(-diameter) == 0.0
    np.testing.assert_allclose(geometry.area(2.0 * diameter), geometry.full_area())
    np.testing.assert_allclose(
        geometry.wetted_perimeter(2.0 * diameter),
        geometry.full_perimeter(),
    )


def test_circular_geometry_backends_support_per_conduit_diameters():
    diameters = np.array([0.5, 1.0, 2.0])
    depths = 0.5 * diameters

    for geometry in (
        CircularAnalyticalGeometry(diameters),
        CircularTabulatedGeometry(diameters),
    ):
        np.testing.assert_allclose(
            geometry.area(depths),
            np.pi * diameters**2 / 8.0,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            geometry.wetted_perimeter(depths),
            np.pi * diameters / 2.0,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            geometry.hydraulic_radius(depths),
            diameters / 4.0,
            rtol=1e-12,
            atol=1e-12,
        )


def test_geometry_factory_passes_table_points_to_tabulated_backend():
    geometry = create_cross_section_geometry(
        "circular_tabulated",
        diameters=np.array([1.0, 2.0]),
        table_points=1234,
    )

    assert geometry.n_points == 1234
