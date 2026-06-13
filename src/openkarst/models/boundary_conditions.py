#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:56:06 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import numpy as np

VALID_BC_TYPES = {'volumetric', 'flux'}


def normalize_target_ids(target_ids):
    """
    Convert one or more boundary-condition target ids to a list of integers.

    Boundary-condition setters accept either a single node id, such as `3`,
    or multiple ids, such as `[3, 4, 5]` or `np.array([3, 4, 5])`.
    This helper normalizes those input forms to one internal representation.
    """
    if isinstance(target_ids, (int, np.integer)):
        return [int(target_ids)]
    return [int(target_id) for target_id in list(target_ids)]


def broadcast_boundary_values(target_ids, values):
    """
    Match boundary-condition values to the normalized target ids.

    If `values` is a scalar, a 0-D NumPy array, or a boundary-condition tuple
    such as `("box", value, t_start, t_end)`, the same value definition is
    applied to every target id.

    If `values` is a list or 1-D NumPy array, it is treated as one value per
    target id and must have the same length as `target_ids`.
    """
    if isinstance(values, tuple):
        return [values] * len(target_ids)

    if isinstance(values, np.ndarray) and values.ndim == 0:
        return [values.item()] * len(target_ids)

    if isinstance(values, (list, np.ndarray)):
        if len(values) != len(target_ids):
            raise ValueError(
                f"Length mismatch: {len(target_ids)} target ids but {len(values)} values."
            )
        return list(values)

    return [values] * len(target_ids)


class BoundaryCondition:
    """Abstract base class for time-dependent boundary conditions.

    Subclasses must implement the `get_value(t)` method to define
    time-dependent behavior for boundary conditions applied to nodes.

    Attributes:
        target_ids (list[int]): IDs of the nodes where the boundary condition applies.
        bc_type (str): Type of boundary condition, either 'volumetric' or 'flux'.
    """

    def __init__(self, target_ids, bc_type='volumetric'):
        """Initializes a boundary condition.

        Args:
            target_ids (list[int]): List of node indices the BC applies to.
            bc_type (str): Type of boundary condition. Must be one of {'volumetric', 'flux'}.

        Raises:
            ValueError: If `bc_type` is not in the allowed set.
        """
        if bc_type not in VALID_BC_TYPES:
            raise ValueError(f"Invalid bc_type: {bc_type}. Must be one of {VALID_BC_TYPES}")
        self.target_ids = target_ids
        self.bc_type = bc_type

    def get_value(self, t):
        """Returns the boundary condition value at time t.

        Args:
            t (float): The time at which to evaluate the boundary condition.

        Returns:
            float: The boundary condition value.

        Raises:
            NotImplementedError: This method must be overridden in subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_value(t)")


class ConstantBC(BoundaryCondition):
    """Constant boundary condition over time.

    Attributes:
        value (float): Constant value applied at all times.
    """

    def __init__(self, target_ids, value, bc_type='volumetric'):
        """Initializes a constant boundary condition.

        Args:
            target_ids (list[int]): List of node indices the BC applies to.
            value (float): The constant value for the boundary condition.
            bc_type (str, optional): Type of BC ('volumetric' or 'flux'). Defaults to 'volumetric'.
        """
        super().__init__(target_ids, bc_type)
        self.value = value

    def get_value(self, t):
        """Returns the constant value.

        Args:
            t (float): Time (ignored).

        Returns:
            float: The constant value.
        """
        return self.value


class BoxBC(BoundaryCondition):
    """Boundary condition active in a specific time window, with optional values outside.

    Attributes:
        v_during (float): Value applied during [t0, t1].
        v_before (float): Value before t0.
        v_after (float): Value after t1.
        t0 (float): Start time of application.
        t1 (float): End time of application.
    """

    def __init__(self, target_ids, v_during, t_start, t_end,
                 v_before=0.0, v_after=0.0, bc_type='volumetric'):
        """Initializes a box-style time-dependent boundary condition.

        Args:
            target_ids (list[int]): List of node indices the BC applies to.
            v_during (float): Value to apply during the time window.
            t_start (float): Start time of the boundary condition.
            t_end (float): End time of the boundary condition.
            v_before (float, optional): Value before `t_start`. Defaults to 0.0.
            v_after (float, optional): Value after `t_end`. Defaults to 0.0.
            bc_type (str, optional): Type of BC ('volumetric' or 'flux'). Defaults to 'volumetric'.
        """
        super().__init__(target_ids, bc_type)
        self.v_during = v_during
        self.v_before = v_before
        self.v_after = v_after
        self.t0 = t_start
        self.t1 = t_end

    def get_value(self, t):
        """Returns the value at time t based on box condition logic.

        Args:
            t (float): Time at which to evaluate the BC.

        Returns:
            float: The BC value at the given time.
        """
        if t < self.t0:
            return self.v_before
        elif t <= self.t1:
            return self.v_during
        else:
            return self.v_after


class TimeSeriesBC(BoundaryCondition):
    """Time-dependent boundary condition using linear interpolation.

    Attributes:
        times (np.ndarray): Time points defining the time series.
        values (np.ndarray): Corresponding boundary condition values.
        extrapolate (str): Behavior outside defined time range ('hold' or 'zero').
    """

    def __init__(self, target_ids, times, values, bc_type='volumetric', extrapolate='hold'):
        """Initializes a time-series-based boundary condition.

        Args:
            target_ids (list[int]): List of node indices the BC applies to.
            times (array-like): Time points (must be sorted).
            values (array-like): Corresponding values at each time point.
            bc_type (str, optional): Type of BC ('volumetric' or 'flux'). Defaults to 'volumetric'.
            extrapolate (str, optional): Extrapolation behavior outside the time range.
                - 'hold': use nearest value (i.e. first and last point).
                - 'zero': set to zero.

        Raises:
            ValueError: If `extrapolate` is not 'hold' or 'zero'.
        """
        super().__init__(target_ids, bc_type)
        self.times = np.asarray(times)
        self.values = np.asarray(values)
        self.extrapolate = extrapolate.lower()

        if self.extrapolate not in {'hold', 'zero'}:
            raise ValueError(f"Invalid extrapolate='{self.extrapolate}'. Use 'hold' or 'zero'.")

    def get_value(self, t):
        """Returns interpolated value at time `t`.

        Args:
            t (float): Time at which to evaluate the BC.

        Returns:
            float: The interpolated or extrapolated boundary condition value.
        """
        if t < self.times[0]:
            return 0.0 if self.extrapolate == 'zero' else float(self.values[0])
        elif t > self.times[-1]:
            return 0.0 if self.extrapolate == 'zero' else float(self.values[-1])
        else:
            return float(np.interp(t, self.times, self.values))


# from numba import njit

# @njit(cache=True)
# def _ts_eval_with_cursor(t, times, values, slopes, j, extrapolate_zero):
#     """
#     Evaluate piecewise-linear time series at time t using a segment cursor.
#     Returns (value, new_j).
#     - times: strictly increasing, len >= 2
#     - slopes: len = len(times) - 1, (values[i+1]-values[i])/(times[i+1]-times[i])
#     - j: cursor such that times[j] <= t < times[j+1] (if within range)
#     - extrapolate_zero: bool (True -> zero outside; False -> hold endpoint)
#     """
#     n = times.size

#     # extrapolation ends
#     if t <= times[0]:
#         return (0.0 if extrapolate_zero else values[0], 0)
#     if t >= times[n-1]:
#         return (0.0 if extrapolate_zero else values[n-1], n-2)

#     # ensure cursor is at the correct segment
#     if t < times[j]:
#         # rare backward jump: reposition once
#         jj = np.searchsorted(times, t) - 1
#         if jj < 0:
#             jj = 0
#         elif jj > n - 2:
#             jj = n - 2
#         j = jj
#     else:
#         # typical forward-only advance; usually 0 or 1 increments
#         while j < n - 2 and t >= times[j + 1]:
#             j += 1

#     # linear interpolation using precomputed slope
#     val = values[j] + slopes[j] * (t - times[j])
#     return (val, j)


# class TimeSeriesBC(BoundaryCondition):
#     """Time-dependent BC with segment cursor + Numba-accelerated evaluator."""

#     def __init__(self, target_ids, times, values, bc_type='volumetric', extrapolate='hold'):
#         super().__init__(target_ids, bc_type)

#         t = np.asarray(times, dtype=np.float64)
#         v = np.asarray(values, dtype=np.float64)
#         if t.ndim != 1 or v.ndim != 1 or t.size != v.size:
#             raise ValueError("times and values must be 1D and same length.")
#         if t.size < 2:
#             raise ValueError("TimeSeriesBC needs at least two points.")
#         if not np.all(np.diff(t) > 0):
#             raise ValueError("times must be strictly increasing.")

#         self.times = np.ascontiguousarray(t)
#         self.values = np.ascontiguousarray(v)

#         dt = np.diff(self.times)
#         dv = np.diff(self.values)
#         self._slopes = np.ascontiguousarray(dv / dt, dtype=np.float64)

#         ex = extrapolate.lower()
#         if ex not in {'hold', 'zero'}:
#             raise ValueError("extrapolate must be 'hold' or 'zero'.")
#         self._extrapolate_zero = (ex == 'zero')

#         self._j = 0
#         self._last_t = -np.inf  # optional, not required by the kernel

#     def reset_cursor(self):
#         self._j = 0
#         self._last_t = -np.inf

#     def get_value(self, t):
#         t = float(t)
#         val, new_j = _ts_eval_with_cursor(
#             t, self.times, self.values, self._slopes, self._j, self._extrapolate_zero
#         )
#         self._j = new_j
#         self._last_t = t
#         return float(val)

