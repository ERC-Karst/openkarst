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
    """
    def __init__(self, target_ids, times, values, bc_type='volumetric'):
        super().__init__(target_ids, bc_type)
        self.times = np.asarray(times)
        self.values = np.asarray(values)

    def get_value(self, t):
        return float(np.interp(t, self.times, self.values))

