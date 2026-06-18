"""Cross-section geometry backends for closed conduit hydraulics.

This module provides conduit geometry calculations to provide A(y), P(y),
R(y), and top width. These can come from circular analytical formulas,
circular interpolation tables, or user-defined cross-section tables.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator


GEOMETRY_BACKENDS = ("circular_analytical", "circular_tabulated", "tabulated")
INTERPOLATION_METHODS = ("pchip", "linear")


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

    def top_width(self, depths):
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

    def top_width(self, depths):
        """Return free-surface top width, not including the Preissmann slot."""
        depths = np.asarray(depths, dtype=float)
        clipped_depths = np.clip(depths, 0.0, self.diameters)
        width_argument = self.diameters * clipped_depths - clipped_depths**2
        widths = 2.0 * np.sqrt(np.maximum(width_argument, 0.0))
        return np.where(
            (depths <= 0.0) | (depths >= self.diameters),
            0.0,
            widths,
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

    def __init__(self, diameters, n_points=100, interpolation_method="pchip"):
        super().__init__(diameters)
        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer greater than or equal to 2.")

        self.n_points = n_points
        self.interpolation_method = _validate_interpolation_method(
            interpolation_method
        )
        self._full_area = (np.pi / 4.0) * self.diameters**2
        self._full_perimeter = np.pi * self.diameters
        self._full_hydraulic_radius = self.diameters / 4.0
        self._full_hydraulic_diameter = self.diameters

        (
            eta,
            area_norm,
            perimeter_norm,
            radius_norm,
            top_width_norm,
        ) = _circular_normalized_table(n_points)
        self.eta = eta
        self.area_norm = area_norm
        self.perimeter_norm = perimeter_norm
        self.radius_norm = radius_norm
        self.top_width_norm = top_width_norm
        self._area_interp = _make_interpolator(
            eta,
            area_norm,
            self.interpolation_method,
        )
        self._perimeter_interp = _make_interpolator(
            eta,
            perimeter_norm,
            self.interpolation_method,
        )
        self._radius_interp = _make_interpolator(
            eta,
            radius_norm,
            self.interpolation_method,
        )
        self._top_width_interp = _make_interpolator(
            eta,
            top_width_norm,
            self.interpolation_method,
        )

    def area(self, depths):
        return self._interpolate(depths, self._area_interp, np.pi / 4.0, scale_power=2)

    def wetted_perimeter(self, depths):
        return self._interpolate(depths, self._perimeter_interp, np.pi, scale_power=1)

    def hydraulic_radius(self, depths, areas=None):
        return self._interpolate(depths, self._radius_interp, 0.25, scale_power=1)

    def top_width(self, depths):
        return self._interpolate(depths, self._top_width_interp, 0.0, scale_power=1)

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


class TabulatedGeometry(CrossSectionGeometry):
    """User-defined symmetric width-depth table for one closed-conduit geometry.

    If scale_by_diameter is True, the CSV is interpreted as normalized data:
    eta, width_norm. Each conduit scales the same shape by its own diameter.

    If scale_by_diameter is False, the CSV is interpreted as physical data:
    depth, width. The same absolute geometry is used for every conduit.

    Area and wetted perimeter are precomputed from the width-depth table once
    during initialization. During the simulation only interpolation is used.
    """

    name = "tabulated"

    def __init__(
        self,
        diameters,
        table_file,
        scale_by_diameter=True,
        interpolation_method="pchip",
    ):
        super().__init__(diameters)
        if not isinstance(scale_by_diameter, bool):
            raise ValueError("scale_by_diameter must be True or False.")

        self.table_file = table_file
        self.scale_by_diameter = scale_by_diameter
        self.interpolation_method = _validate_interpolation_method(
            interpolation_method
        )

        table = _load_tabulated_geometry_csv(table_file, self.scale_by_diameter)
        self.depth_table = table["depth"]
        self.top_width_table = table["top_width"]
        self.area_table, self.perimeter_table = _precompute_from_width_table(
            self.depth_table,
            self.top_width_table,
        )
        self.radius_table = _hydraulic_radius_from_area_perimeter(
            self.area_table,
            self.perimeter_table,
        )

        self._area_interp = _make_interpolator(
            self.depth_table,
            self.area_table,
            self.interpolation_method,
        )
        self._perimeter_interp = _make_interpolator(
            self.depth_table,
            self.perimeter_table,
            self.interpolation_method,
        )
        self._radius_interp = _make_interpolator(
            self.depth_table,
            self.radius_table,
            self.interpolation_method,
        )
        self._top_width_interp = _make_interpolator(
            self.depth_table,
            self.top_width_table,
            self.interpolation_method,
        )

        if self.scale_by_diameter:
            self.full_depths = self.diameters
            self._full_area = self.area_table[-1] * self.diameters**2
            self._full_perimeter = self.perimeter_table[-1] * self.diameters
        else:
            self.full_depths = np.full_like(self.diameters, self.depth_table[-1])
            self._full_area = np.full_like(self.diameters, self.area_table[-1])
            self._full_perimeter = np.full_like(
                self.diameters,
                self.perimeter_table[-1],
            )

        self._full_hydraulic_radius = self._full_area / self._full_perimeter
        self._full_hydraulic_diameter = 4.0 * self._full_hydraulic_radius

    def area(self, depths):
        return self._interpolate(
            depths,
            self._area_interp,
            full_value=self._full_area,
            scale_power=2,
        )

    def wetted_perimeter(self, depths):
        return self._interpolate(
            depths,
            self._perimeter_interp,
            full_value=self._full_perimeter,
            scale_power=1,
        )

    def hydraulic_radius(self, depths, areas=None):
        return self._interpolate(
            depths,
            self._radius_interp,
            full_value=self._full_hydraulic_radius,
            scale_power=1,
        )

    def top_width(self, depths):
        return self._interpolate(
            depths,
            self._top_width_interp,
            full_value=0.0,
            scale_power=1,
        )

    def full_area(self):
        return self._full_area

    def full_perimeter(self):
        return self._full_perimeter

    def full_hydraulic_radius(self):
        return self._full_hydraulic_radius

    def full_hydraulic_diameter(self):
        return self._full_hydraulic_diameter

    def _interpolate(self, depths, interpolator, full_value, scale_power):
        depths = np.asarray(depths, dtype=float)
        if self.scale_by_diameter:
            table_depths = depths / self.diameters
        else:
            table_depths = depths

        table_depths_clipped = np.clip(
            table_depths,
            self.depth_table[0],
            self.depth_table[-1],
        )
        values = np.asarray(interpolator(table_depths_clipped), dtype=float)

        if self.scale_by_diameter:
            values = values * self.diameters**scale_power

        values = np.where(table_depths <= self.depth_table[0], 0.0, values)
        values = np.where(table_depths >= self.depth_table[-1], full_value, values)
        return np.maximum(values, 0.0)


def create_cross_section_geometry(
    backend,
    diameters,
    table_points=100,
    table_file=None,
    scale_by_diameter=True,
    interpolation_method="pchip",
):
    """Create the geometry object requested by geometry_settings.backend.

    This is in fact already done in the property validation. As I may remove that at
    some point this is here to future proof.
    """
    backend = str(backend).lower()
    if backend == CircularAnalyticalGeometry.name:
        return CircularAnalyticalGeometry(diameters)
    if backend == CircularTabulatedGeometry.name:
        return CircularTabulatedGeometry(
            diameters,
            n_points=table_points,
            interpolation_method=interpolation_method,
        )
    if backend == TabulatedGeometry.name:
        return TabulatedGeometry(
            diameters,
            table_file=table_file,
            scale_by_diameter=scale_by_diameter,
            interpolation_method=interpolation_method,
        )
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
    top_width_norm = 2.0 * np.sqrt(np.maximum(eta - eta**2, 0.0))

    # Force exact endpoints and monotonic area despite floating-point noise.
    area_norm[0] = 0.0
    area_norm[-1] = np.pi / 4.0
    area_norm = np.maximum.accumulate(area_norm)
    perimeter_norm[0] = 0.0
    perimeter_norm[-1] = np.pi
    top_width_norm[0] = 0.0
    top_width_norm[-1] = 0.0

    radius_norm = np.zeros_like(area_norm, dtype=float)
    wet = perimeter_norm > 0.0
    radius_norm[wet] = area_norm[wet] / perimeter_norm[wet]
    radius_norm[-1] = 0.25
    return eta, area_norm, perimeter_norm, radius_norm, top_width_norm


def _load_tabulated_geometry_csv(table_file, scale_by_diameter):
    if not table_file:
        raise ValueError("A table_file is required for the 'tabulated' geometry backend.")

    table = np.genfromtxt(table_file, delimiter=",", names=True, dtype=float, encoding=None)
    if table.dtype.names is None:
        raise ValueError("Geometry table CSV must include a header row.")

    columns = {name.lower(): table[name] for name in table.dtype.names}
    if scale_by_diameter:
        depth = _table_column(columns, "eta", "depth_norm", "normalized_depth")
        top_width = _table_column(
            columns,
            "width_norm",
            "top_width_norm",
            "surface_width_norm",
            "width",
            "top_width",
            "surface_width",
        )
    else:
        depth = _table_column(columns, "depth", "y", "depth_m")
        top_width = _table_column(
            columns,
            "width",
            "top_width",
            "surface_width",
            "width_m",
        )

    depth = np.atleast_1d(np.asarray(depth, dtype=float))
    top_width = np.atleast_1d(np.asarray(top_width, dtype=float))

    _validate_tabulated_geometry(depth, top_width, scale_by_diameter)
    return {
        "depth": depth,
        "top_width": top_width,
    }


def _validate_interpolation_method(interpolation_method):
    method = str(interpolation_method).lower()
    if method not in INTERPOLATION_METHODS:
        raise ValueError(
            "interpolation_method must be one of: "
            + ", ".join(INTERPOLATION_METHODS)
        )
    return method


def _make_interpolator(x, values, interpolation_method):
    method = _validate_interpolation_method(interpolation_method)
    if method == "pchip":
        return PchipInterpolator(x, values, extrapolate=False)
    if method == "linear":
        return LinearInterpolator(x, values)
    raise ValueError(f"Unknown interpolation method '{interpolation_method}'.")


class LinearInterpolator:
    """Tiny callable wrapper around numpy's one-dimensional linear interpolation."""

    def __init__(self, x, values):
        self.x = np.asarray(x, dtype=float)
        self.values = np.asarray(values, dtype=float)

    def __call__(self, x_new):
        return np.interp(x_new, self.x, self.values)


def _table_column(columns, *names):
    for name in names:
        if name in columns:
            return columns[name]
    raise ValueError(
        "Geometry table CSV is missing one of these columns: "
        + ", ".join(names)
    )


def _validate_tabulated_geometry(depth, top_width, scale_by_diameter):
    lengths = {len(depth), len(top_width)}
    if len(lengths) != 1 or len(depth) < 2:
        raise ValueError("Geometry table columns must have the same length >= 2.")

    for name, values in (
        ("depth", depth),
        ("width", top_width),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Geometry table column '{name}' must be finite.")

    if not np.isclose(depth[0], 0.0):
        raise ValueError("Geometry table depth must start at 0.")
    if np.any(np.diff(depth) <= 0.0):
        raise ValueError("Geometry table depth values must be strictly increasing.")
    if scale_by_diameter and not np.isclose(depth[-1], 1.0):
        raise ValueError("Normalized geometry table depth must end at eta = 1.")

    if np.any(top_width < 0.0):
        raise ValueError("Geometry table width must be nonnegative.")

    if not np.isclose(top_width[0], 0.0):
        raise ValueError("Closed-conduit width must be zero at depth 0.")
    if not np.isclose(top_width[-1], 0.0):
        raise ValueError("Closed-conduit width must be zero at full depth.")


def _precompute_from_width_table(depth, top_width):
    """Precompute A(y) and P(y) from a symmetric width-depth table.

    The table describes a symmetric cross section with x = +/- width(y) / 2.
    Area is the integral of width over depth. Wetted perimeter is the length of
    the two side-wall polylines implied by the table.
    """
    area = np.zeros_like(depth, dtype=float)
    depth_steps = np.diff(depth)
    width_avg = 0.5 * (top_width[:-1] + top_width[1:])
    area[1:] = np.cumsum(width_avg * depth_steps)
    area = np.maximum.accumulate(area)

    half_width_steps = 0.5 * np.diff(top_width)
    side_lengths = np.sqrt(depth_steps**2 + half_width_steps**2)
    perimeter = np.zeros_like(depth, dtype=float)
    perimeter[1:] = 2.0 * np.cumsum(side_lengths)

    return area, perimeter


def _hydraulic_radius_from_area_perimeter(area, perimeter):
    radius = np.zeros_like(area, dtype=float)
    wet = perimeter > 0.0
    radius[wet] = area[wet] / perimeter[wet]
    return radius
