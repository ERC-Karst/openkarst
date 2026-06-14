import numpy as np

from openkarst.models.cross_section_geometry import (
    CircularAnalyticalGeometry,
    CircularTabulatedGeometry,
    create_cross_section_geometry,
)


def _write_normalized_circular_table(path, n_points=2001):
    eta = np.linspace(0.0, 1.0, n_points)
    theta = 2.0 * np.arccos(np.clip(1.0 - 2.0 * eta, -1.0, 1.0))
    area_norm = 0.125 * (theta - np.sin(theta))
    perimeter_norm = 0.5 * theta
    top_width_norm = 2.0 * np.sqrt(np.maximum(eta - eta**2, 0.0))

    area_norm[0] = 0.0
    area_norm[-1] = np.pi / 4.0
    perimeter_norm[0] = 0.0
    perimeter_norm[-1] = np.pi
    top_width_norm[0] = 0.0
    top_width_norm[-1] = 0.0

    table = np.column_stack((eta, area_norm, perimeter_norm, top_width_norm))
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="eta,area_norm,perimeter_norm,top_width_norm",
        comments="",
    )


def _write_absolute_circular_table(path, diameter=1.0, n_points=2001):
    eta = np.linspace(0.0, 1.0, n_points)
    depth = eta * diameter
    theta = 2.0 * np.arccos(np.clip(1.0 - 2.0 * eta, -1.0, 1.0))
    area = 0.125 * diameter**2 * (theta - np.sin(theta))
    perimeter = 0.5 * diameter * theta
    top_width = 2.0 * diameter * np.sqrt(np.maximum(eta - eta**2, 0.0))

    area[0] = 0.0
    area[-1] = np.pi * diameter**2 / 4.0
    perimeter[0] = 0.0
    perimeter[-1] = np.pi * diameter
    top_width[0] = 0.0
    top_width[-1] = 0.0

    table = np.column_stack((depth, area, perimeter, top_width))
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="depth,area,wetted_perimeter,top_width",
        comments="",
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
        np.testing.assert_allclose(
            tabulated.top_width(depths[strict_mask]),
            analytical.top_width(depths[strict_mask]),
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
        np.testing.assert_allclose(
            geometry.top_width(depths),
            diameters,
            rtol=1e-6,
            atol=1e-12,
        )


def test_geometry_factory_passes_table_points_to_tabulated_backend():
    geometry = create_cross_section_geometry(
        "circular_tabulated",
        diameters=np.array([1.0, 2.0]),
        table_points=1234,
    )

    assert geometry.n_points == 1234


def test_user_tabulated_geometry_from_normalized_csv_matches_analytical_circle(tmp_path):
    table_file = tmp_path / "normalized_circle.csv"
    _write_normalized_circular_table(table_file)

    diameters = np.array([0.5, 2.0])
    depth_factors = np.array([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0, 1.1])
    depths = depth_factors[:, None] * diameters[None, :]
    analytical = CircularAnalyticalGeometry(diameters)
    tabulated = create_cross_section_geometry(
        "tabulated",
        diameters,
        table_file=str(table_file),
        scale_by_diameter=True,
    )

    np.testing.assert_allclose(
        tabulated.area(depths),
        analytical.area(depths),
        rtol=3e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tabulated.wetted_perimeter(depths),
        analytical.wetted_perimeter(depths),
        rtol=2e-4,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tabulated.hydraulic_radius(depths),
        analytical.hydraulic_radius(depths),
        rtol=2e-4,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tabulated.top_width(depths),
        analytical.top_width(depths),
        rtol=2e-4,
        atol=1e-12,
    )


def test_user_tabulated_geometry_can_use_one_absolute_table_for_all_conduits(tmp_path):
    table_file = tmp_path / "absolute_circle.csv"
    _write_absolute_circular_table(table_file, diameter=1.0)

    geometry = create_cross_section_geometry(
        "tabulated",
        diameters=np.array([1.0, 2.0]),
        table_file=str(table_file),
        scale_by_diameter=False,
    )

    depths = np.array([0.5, 0.5])
    np.testing.assert_allclose(geometry.full_depths, np.array([1.0, 1.0]))
    np.testing.assert_allclose(geometry.area(depths), np.array([np.pi / 8.0] * 2))
    np.testing.assert_allclose(geometry.top_width(depths), np.array([1.0, 1.0]))
