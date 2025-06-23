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
    """
    Abstract base class for time-dependent boundary conditions applied to nodes.

    Subclasses must implement `get_value(t)` to define time-dependent behavior.

    Attributes:
        target_ids (list[int]): IDs of the nodes where the boundary condition is applied.
        bc_type (str): Type of boundary condition, 'volumetric' or 'flux'.
    """
    def __init__(self, target_ids, bc_type='volumetric'):
        if bc_type not in VALID_BC_TYPES:
            raise ValueError(f"Invalid bc_type: {bc_type}. Must be one of {VALID_BC_TYPES}")
        self.target_ids = target_ids
        self.bc_type = bc_type

    def get_value(self, t):
        raise NotImplementedError("Subclasses must implement get_value(t)")


class ConstantBC(BoundaryCondition):
    """
    Constant boundary condition.

    Attributes:
        value (float): Constant value applied at all times.
    """
    def __init__(self, target_ids, value, bc_type='volumetric'):
        super().__init__(target_ids, bc_type)
        self.value = value

    def get_value(self, t):
        return self.value


class BoxBC(BoundaryCondition):
    """
    Boundary condition active between [t0, t1] with optional values outside.

    Attributes:
        v_during (float): Value during the interval [t0, t1].
        v_before (float): Value before t0.
        v_after (float): Value after t1.
        t0 (float): Start time of activation.
        t1 (float): End time of activation.
    """
    def __init__(self, target_ids, v_during, t_start, t_end,
                 v_before=0.0, v_after=0.0, bc_type='volumetric'):
        super().__init__(target_ids, bc_type)
        self.v_during = v_during
        self.v_before = v_before
        self.v_after = v_after
        self.t0 = t_start
        self.t1 = t_end

    def get_value(self, t):
        if t < self.t0:
            return self.v_before
        elif t <= self.t1:
            return self.v_during
        else:
            return self.v_after


class TimeSeriesBC(BoundaryCondition):
    """
    Time-dependent boundary condition using linear interpolation.

    Attributes:
        times (np.ndarray): Array of time points.
        values (np.ndarray): Corresponding values.
        extrapolate (str): Behavior outside the time window: 'hold' or 'zero'.
    """
    def __init__(self, target_ids, times, values, bc_type='volumetric', extrapolate='hold'):
        super().__init__(target_ids, bc_type)
        self.times = np.asarray(times)
        self.values = np.asarray(values)
        self.extrapolate = extrapolate.lower()

        if self.extrapolate not in {'hold', 'zero'}:
            raise ValueError(f"Invalid extrapolate='{self.extrapolate}'. Use 'hold' or 'zero'.")

    def get_value(self, t):
        if t < self.times[0]:
            return 0.0 if self.extrapolate == 'zero' else float(self.values[0])
        elif t > self.times[-1]:
            return 0.0 if self.extrapolate == 'zero' else float(self.values[-1])
        else:
            return float(np.interp(t, self.times, self.values))
