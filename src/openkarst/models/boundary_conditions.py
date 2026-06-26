#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:56:06 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

from numbers import Real

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


class SpringBC:
    """Head-dependent spring outflow boundary.

    The boundary computes an outflow from the simulated node head to an
    external spring node with defined outlet elevation. It never returns
    negative flow, so it cannot inject water back into the model domain.
    """

    def __init__(
        self,
        target_ids,
        outlet_elevation,
        coefficient=None,
        exponent=1.0,
        rating_curve=None,
    ):
        """Initialize a spring boundary condition.

        Args:
            target_ids (list[int]): List of node indices the BC applies to.
            outlet_elevation (float): Absolute spring outlet elevation [m].
            coefficient (float, optional): Power-law coefficient for
                ``Q = coefficient * excess_head**exponent``.
            exponent (float, optional): Power-law exponent. Defaults to 1.0.
            rating_curve (tuple, optional): ``(stages, discharges)`` arrays,
                where stages are excess heads above the outlet [m] and
                discharges are non-negative outflows [m^3/s].
        """
        if not isinstance(outlet_elevation, Real) or not np.isfinite(outlet_elevation):
            raise ValueError("Spring outlet_elevation must be a finite scalar.")

        if rating_curve is not None and coefficient is not None:
            raise ValueError(
                "SpringBC accepts either coefficient/exponent or rating_curve, not both."
            )

        self.target_ids = target_ids
        self.outlet_elevation = float(outlet_elevation)
        self.coefficient = None
        self.exponent = None
        self.rating_stages = None
        self.rating_discharges = None

        if rating_curve is None:
            if coefficient is None:
                raise ValueError(
                    "SpringBC requires coefficient when rating_curve is not provided."
                )
            if not isinstance(coefficient, Real) or not np.isfinite(coefficient):
                raise ValueError("Spring coefficient must be a finite scalar.")
            if coefficient < 0.0:
                raise ValueError("Spring coefficient must be non-negative.")
            if not isinstance(exponent, Real) or not np.isfinite(exponent):
                raise ValueError("Spring exponent must be a finite scalar.")
            if exponent <= 0.0:
                raise ValueError("Spring exponent must be positive.")

            self.coefficient = float(coefficient)
            self.exponent = float(exponent)
        else:
            try:
                stages, discharges = rating_curve
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Spring rating_curve must be a (stages, discharges) pair."
                ) from exc

            stages = np.asarray(stages, dtype=float)
            discharges = np.asarray(discharges, dtype=float)

            if stages.ndim != 1 or discharges.ndim != 1:
                raise ValueError("Spring rating_curve arrays must be one-dimensional.")
            if stages.size < 2:
                raise ValueError("Spring rating_curve needs at least two points.")
            if stages.size != discharges.size:
                raise ValueError(
                    "Spring rating_curve stages and discharges must have the same length."
                )
            if not np.all(np.isfinite(stages)) or not np.all(np.isfinite(discharges)):
                raise ValueError("Spring rating_curve values must be finite.")
            if np.any(stages < 0.0):
                raise ValueError("Spring rating_curve stages must be non-negative.")
            if np.any(np.diff(stages) <= 0.0):
                raise ValueError(
                    "Spring rating_curve stages must be strictly increasing."
                )
            if np.any(discharges < 0.0):
                raise ValueError(
                    "Spring rating_curve discharges must be non-negative."
                )

            self.rating_stages = stages
            self.rating_discharges = discharges

    def compute_outflow(self, head):
        """Return spring outflow for a node with defined hydraulic head [m]."""
        if not isinstance(head, Real) or not np.isfinite(head):
            raise ValueError("Spring head must be a finite scalar.")

        excess_head = float(head) - self.outlet_elevation
        if excess_head <= 0.0:
            return 0.0

        if self.rating_stages is not None:
            outflow = np.interp(
                excess_head,
                self.rating_stages,
                self.rating_discharges,
                left=0.0,
                right=float(self.rating_discharges[-1]),
            )
        else:
            outflow = self.coefficient * excess_head**self.exponent

        return max(float(outflow), 0.0)
