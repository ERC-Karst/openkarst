"""Numba kernels for closed-conduit geometry calculations."""

import math

from openkarst.models.numba_support import (
    NUMBA_AVAILABLE,
    ensure_numba_available,
    njit,
    prange,
)


if NUMBA_AVAILABLE:

    # Circular analytical geometry

    @njit(cache=True)
    def _clip_scalar(value, lower, upper):
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value


    @njit(cache=True)
    def _circular_geometry_scalar(depth, diameter):
        radius = 0.5 * diameter
        clipped_depth = _clip_scalar(depth, 0.0, diameter)
        theta_arg = _clip_scalar((radius - clipped_depth) / radius, -1.0, 1.0)
        theta = 2.0 * math.acos(theta_arg)

        full_area = math.pi * radius * radius
        if depth >= diameter:
            area = full_area
            top_width = 0.0
            hydraulic_radius = 0.25 * diameter
            is_full = True
        else:
            area = 0.5 * radius * radius * (theta - math.sin(theta))
            width_argument = diameter * clipped_depth - clipped_depth * clipped_depth
            if depth <= 0.0 or width_argument <= 0.0:
                top_width = 0.0
            else:
                top_width = 2.0 * math.sqrt(width_argument)

            perimeter = radius * theta
            if perimeter > 0.0:
                hydraulic_radius = area / perimeter
            else:
                hydraulic_radius = 0.0
            is_full = False

        return area, top_width, hydraulic_radius, is_full


    @njit(cache=True)
    def _slot_width_scalar(depth, diameter):
        y_norm = depth / diameter
        if y_norm > 1.78:
            return 0.01 * diameter
        return diameter * 0.5423 * math.exp(-(y_norm ** 2.4))


    @njit(parallel=True, cache=True)
    def compute_circular_geometry_numba(
        y1,
        y2,
        y_mid,
        diameters,
        conduit_lengths,
        min_waterdepth,
        a1,
        a2,
        a_mid,
        r1,
        r2,
        r_mid,
        w_mid,
        surface_a1,
        surface_a2,
        is_full_y1,
        is_full_y2,
        is_full_y_mid,
    ):
        for k in prange(y_mid.size):
            diameter = diameters[k]

            area1, conduit_w1, radius1, full1 = _circular_geometry_scalar(
                y1[k],
                diameter,
            )
            area2, conduit_w2, radius2, full2 = _circular_geometry_scalar(
                y2[k],
                diameter,
            )
            area_mid, conduit_w_mid, radius_mid, full_mid = (
                _circular_geometry_scalar(
                    y_mid[k],
                    diameter,
                )
            )

            a1[k] = area1
            a2[k] = area2
            a_mid[k] = area_mid
            r1[k] = radius1
            r2[k] = radius2
            r_mid[k] = radius_mid
            is_full_y1[k] = full1
            is_full_y2[k] = full2
            is_full_y_mid[k] = full_mid

            if full1 and y1[k] > min_waterdepth:
                width1 = _slot_width_scalar(y1[k], diameter)
            else:
                width1 = conduit_w1

            if full2 and y2[k] > min_waterdepth:
                width2 = _slot_width_scalar(y2[k], diameter)
            else:
                width2 = conduit_w2

            if full_mid and (y1[k] > min_waterdepth or y2[k] > min_waterdepth):
                width_mid = _slot_width_scalar(y_mid[k], diameter)
            else:
                width_mid = conduit_w_mid

            w_mid[k] = width_mid
            surface_a1[k] = 0.25 * (width1 + width_mid) * conduit_lengths[k]
            surface_a2[k] = 0.25 * (width_mid + width2) * conduit_lengths[k]


    # Linear tabulated geometry

    @njit(cache=True)
    def _linear_interp_scalar(x, xp, fp):
        if x <= xp[0]:
            return fp[0]
        last = xp.size - 1
        if x >= xp[last]:
            return fp[last]

        lo = 0
        hi = last
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xp[mid] <= x:
                lo = mid
            else:
                hi = mid

        span = xp[hi] - xp[lo]
        if span == 0.0:
            return fp[lo]
        weight = (x - xp[lo]) / span
        return fp[lo] + weight * (fp[hi] - fp[lo])


    @njit(cache=True)
    def _tabulated_geometry_scalar(
        depth,
        diameter,
        full_depth,
        full_area,
        full_hydraulic_radius,
        depth_table,
        area_table,
        radius_table,
        top_width_table,
        scale_by_diameter,
    ):
        if scale_by_diameter:
            table_depth = depth / diameter
        else:
            table_depth = depth

        if table_depth <= depth_table[0]:
            return 0.0, 0.0, 0.0, False

        if depth >= full_depth or table_depth >= depth_table[depth_table.size - 1]:
            return full_area, 0.0, full_hydraulic_radius, True

        area_value = _linear_interp_scalar(table_depth, depth_table, area_table)
        radius_value = _linear_interp_scalar(table_depth, depth_table, radius_table)
        top_width_value = _linear_interp_scalar(
            table_depth,
            depth_table,
            top_width_table,
        )

        if scale_by_diameter:
            area_value = area_value * diameter * diameter
            radius_value = radius_value * diameter
            top_width_value = top_width_value * diameter

        if area_value < 0.0:
            area_value = 0.0
        if radius_value < 0.0:
            radius_value = 0.0
        if top_width_value < 0.0:
            top_width_value = 0.0

        return area_value, top_width_value, radius_value, False


    @njit(parallel=True, cache=True)
    def compute_tabulated_geometry_numba(
        y1,
        y2,
        y_mid,
        diameters,
        full_depths,
        full_areas,
        full_hydraulic_radii,
        conduit_lengths,
        min_waterdepth,
        depth_table,
        area_table,
        radius_table,
        top_width_table,
        scale_by_diameter,
        a1,
        a2,
        a_mid,
        r1,
        r2,
        r_mid,
        w_mid,
        surface_a1,
        surface_a2,
        is_full_y1,
        is_full_y2,
        is_full_y_mid,
    ):
        for k in prange(y_mid.size):
            diameter = diameters[k]
            full_depth = full_depths[k]
            full_area = full_areas[k]
            full_radius = full_hydraulic_radii[k]

            area1, conduit_w1, radius1, full1 = _tabulated_geometry_scalar(
                y1[k],
                diameter,
                full_depth,
                full_area,
                full_radius,
                depth_table,
                area_table,
                radius_table,
                top_width_table,
                scale_by_diameter,
            )
            area2, conduit_w2, radius2, full2 = _tabulated_geometry_scalar(
                y2[k],
                diameter,
                full_depth,
                full_area,
                full_radius,
                depth_table,
                area_table,
                radius_table,
                top_width_table,
                scale_by_diameter,
            )
            area_mid, conduit_w_mid, radius_mid, full_mid = (
                _tabulated_geometry_scalar(
                    y_mid[k],
                    diameter,
                    full_depth,
                    full_area,
                    full_radius,
                    depth_table,
                    area_table,
                    radius_table,
                    top_width_table,
                    scale_by_diameter,
                )
            )

            a1[k] = area1
            a2[k] = area2
            a_mid[k] = area_mid
            r1[k] = radius1
            r2[k] = radius2
            r_mid[k] = radius_mid
            is_full_y1[k] = full1
            is_full_y2[k] = full2
            is_full_y_mid[k] = full_mid

            if full1 and y1[k] > min_waterdepth:
                width1 = _slot_width_scalar(y1[k], full_depth)
            else:
                width1 = conduit_w1

            if full2 and y2[k] > min_waterdepth:
                width2 = _slot_width_scalar(y2[k], full_depth)
            else:
                width2 = conduit_w2

            if full_mid and (y1[k] > min_waterdepth or y2[k] > min_waterdepth):
                width_mid = _slot_width_scalar(y_mid[k], full_depth)
            else:
                width_mid = conduit_w_mid

            w_mid[k] = width_mid
            surface_a1[k] = 0.25 * (width1 + width_mid) * conduit_lengths[k]
            surface_a2[k] = 0.25 * (width_mid + width2) * conduit_lengths[k]


else:

    def compute_circular_geometry_numba(*args):
        """Raise a clear error when the optional Numba backend is unavailable."""
        ensure_numba_available()


    def compute_tabulated_geometry_numba(*args):
        """Raise a clear error when the optional Numba backend is unavailable."""
        ensure_numba_available()
