#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:56:06 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import numpy as np

VALID_BC_TYPES = {'volumetric', 'flux'}

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