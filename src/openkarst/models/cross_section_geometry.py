"""Cross-section geometry backends for closed conduit hydraulics.

This module provides conduit geometry calculations to provide A(y), P(y),
and R(y). This either come from circular analytical formulas, circular
interpolation tables, or potentially user-defined cross sections (future).
"""

import numpy as np
from scipy.interpolate import PchipInterpolator


GEOMETRY_BACKENDS = ("circular_analytical", "circular_tabulated")


class CrossSectionGeometry:
    """Small base class for per-conduit closed cross-section geometry.

    Each method accepts arrays, so one geometry object can evaluate all
    conduits in a network even when every conduit has its own diameter.
    """

    name = "base"

    def __init__(self, diameters):
        self.diameters = np.asarray(diameters, dtype=float)
        if np.any(self.diameters <= 0.0):
            raise ValueError("Conduit diameters must be positive.")

        # For circular conduits the crown depth is the diameter. Future
        # geometries can replace this with their own full-depth values.
        self.full_depths = self.diameters

    # every geometry must define these
    def area(self, depths):
        raise NotImplementedError

    def wetted_perimeter(self, depths):
        raise NotImplementedError

    def hydraulic_radius(self, depths, areas=None):
        """Compute R(y) = A(y) / P(y).

        The solver commonly already has A(y), so areas can be passed in to avoid
        recalculating it inside computationally expensive Picard loops.
        """
        if areas is None:
            areas = self.area(depths)

        perimeters = self.wetted_perimeter(depths)
        radii = np.zeros_like(areas, dtype=float)
        wet = perimeters > 0.0
        radii[wet] = areas[wet] / perimeters[wet]
        return np.where(
            np.asarray(depths, dtype=float) >= self.full_depths,
            self.full_hydraulic_radius(),
            radii,
        )

    def is_full(self, depths):
        """Return True where the water depth has reached the conduit ceiling."""
        return np.asarray(depths, dtype=float) >= self.full_depths

    def full_area(self):
        raise NotImplementedError

    def full_perimeter(self):
        raise NotImplementedError

    def full_hydraulic_radius(self):
        return self.full_area() / self.full_perimeter()

    def full_hydraulic_diameter(self):
        return 4.0 * self.full_hydraulic_radius()


class CircularAnalyticalGeometry(CrossSectionGeometry):
    """Analytical circular closed-conduit geometry.

    This provides the same calculations as in the FLowSimulation class before.
    It is the default and will be the reference behavior.
    """

    name = "circular_analytical"

    def __init__(self, diameters):
        super().__init__(diameters)
        self.radii = 0.5 * self.diameters
        self._full_area = np.pi * self.radii**2
        self._full_perimeter = np.pi * self.diameters
        self._full_hydraulic_radius = self.diameters / 4.0
        self._full_hydraulic_diameter = self.diameters

    def area(self, depths):
        """Return circular-segment area A(y), clipped to full area for y >= D."""
        depths = np.asarray(depths, dtype=float)
        clipped_depths = np.clip(depths, 0.0, self.diameters)
        theta = self._theta(clipped_depths)
        areas = 0.5 * self.radii**2 * (theta - np.sin(theta))
        return np.where(depths >= self.diameters, self._full_area, areas)

    def wetted_perimeter(self, depths):
        """Return wetted arc length P(y), clipped to full perimeter for y >= D."""
        depths = np.asarray(depths, dtype=float)
        clipped_depths = np.clip(depths, 0.0, self.diameters)
        perimeters = self.radii * self._theta(clipped_depths)
        return np.where(
            depths >= self.diameters,
            self._full_perimeter,
            perimeters,
        )

    def hydraulic_radius(self, depths, areas=None):
        if areas is None:
            areas = self.area(depths)

        depths = np.asarray(depths, dtype=float)
        perimeters = self.wetted_perimeter(depths)
        radii = np.zeros_like(areas, dtype=float)
        wet = perimeters > 0.0
        radii[wet] = areas[wet] / perimeters[wet]
        return np.where(depths >= self.diameters, self._full_hydraulic_radius, radii)

    def full_area(self):
        return self._full_area

    def full_perimeter(self):
        return self._full_perimeter

    def full_hydraulic_radius(self):
        return self._full_hydraulic_radius

    def full_hydraulic_diameter(self):
        return self._full_hydraulic_diameter

    def _theta(self, depths):
        """Central angle of the wetted circular segment."""
        return 2.0 * np.arccos(np.clip((self.radii - depths) / self.radii, -1.0, 1.0))


class CircularTabulatedGeometry(CrossSectionGeometry):
    """Pre-computed depth-normalized tables for a circular closed-conduit geometry.

    This is a circular test case potential future arbitrary-table geometries:
    It precomputes A, P, and R over eta = y / D, then interpolates during the solve.
    """

    name = "circular_tabulated"

    def __init__(self, diameters, n_points=1000):
        super().__init__(diameters)
        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer greater than or equal to 2.")

        self.n_points = n_points
        self._full_area = (np.pi / 4.0) * self.diameters**2
        self._full_perimeter = np.pi * self.diameters
        self._full_hydraulic_radius = self.diameters / 4.0
        self._full_hydraulic_diameter = self.diameters

        eta, area_norm, perimeter_norm, radius_norm = _circular_normalized_table(n_points)
        self.eta = eta
        self.area_norm = area_norm
        self.perimeter_norm = perimeter_norm
        self.radius_norm = radius_norm
        self._area_interp = PchipInterpolator(eta, area_norm, extrapolate=False)
        self._perimeter_interp = PchipInterpolator(
            eta,
            perimeter_norm,
            extrapolate=False,
        )
        self._radius_interp = PchipInterpolator(eta, radius_norm, extrapolate=False)

    def area(self, depths):
        return self._interpolate(depths, self._area_interp, np.pi / 4.0, scale_power=2)

    def wetted_perimeter(self, depths):
        return self._interpolate(depths, self._perimeter_interp, np.pi, scale_power=1)

    def hydraulic_radius(self, depths, areas=None):
        return self._interpolate(depths, self._radius_interp, 0.25, scale_power=1)

    def full_area(self):
        return self._full_area

    def full_perimeter(self):
        return self._full_perimeter

    def full_hydraulic_radius(self):
        return self._full_hydraulic_radius

    def full_hydraulic_diameter(self):
        return self._full_hydraulic_diameter

    def _interpolate(self, depths, interpolator, full_norm, scale_power):
        depths = np.asarray(depths, dtype=float)
        eta = depths / self.diameters
        eta_clamped = np.clip(eta, 0.0, 1.0)

        norm_value = np.asarray(interpolator(eta_clamped), dtype=float)

        # Keep the geometry responsible only for the closed conduit. The
        # Preissmann slot correction is applied later by FlowSimulation.
        norm_value = np.where(eta <= 0.0, 0.0, norm_value)
        norm_value = np.where(eta >= 1.0, full_norm, norm_value)
        norm_value = np.maximum(norm_value, 0.0)
        return norm_value * self.diameters**scale_power


def create_cross_section_geometry(backend, diameters, table_points=1000):
    """Create the geometry object requested by geometry_settings.backend.

    This is in fact already done in the property validation. As I may remove that at
    some point this is here to future proof.
    """
    backend = str(backend).lower()
    if backend == CircularAnalyticalGeometry.name:
        return CircularAnalyticalGeometry(diameters)
    if backend == CircularTabulatedGeometry.name:
        return CircularTabulatedGeometry(diameters, n_points=table_points)
    raise ValueError(
        f"Unknown geometry_settings backend '{backend}'. "
        f"Allowed values are: {', '.join(GEOMETRY_BACKENDS)}."
    )


def _circular_normalized_table(n_points):
    """Build circular geometry tables for eta = y / D from 0 to 1."""
    eta = np.linspace(0.0, 1.0, n_points)
    theta = 2.0 * np.arccos(np.clip(1.0 - 2.0 * eta, -1.0, 1.0))

    # Normalized circular formulas:
    # A / D^2 = (theta - sin(theta)) / 8
    # P / D = theta / 2
    area_norm = 0.125 * (theta - np.sin(theta))
    perimeter_norm = 0.5 * theta

    # Force exact endpoints and monotonic area despite floating-point noise.
    area_norm[0] = 0.0
    area_norm[-1] = np.pi / 4.0
    area_norm = np.maximum.accumulate(area_norm)
    perimeter_norm[0] = 0.0
    perimeter_norm[-1] = np.pi

    radius_norm = np.zeros_like(area_norm, dtype=float)
    wet = perimeter_norm > 0.0
    radius_norm[wet] = area_norm[wet] / perimeter_norm[wet]
    radius_norm[-1] = 0.25
    return eta, area_norm, perimeter_norm, radius_norm
