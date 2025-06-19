import numpy as np

class BoundaryCondition:
    """
    Abstract base class for time-dependent boundary conditions.

    This class defines the interface for boundary condition objects applied to 
    specific nodes or conduits. Subclasses must implement the `get_value(t)` method.

    Attributes:
        target_ids (list of int): IDs of the target nodes or conduits where the 
            boundary condition is applied.
        target_type (str): Type of target, either 'node' or 'conduit'.
        bc_type (str): Type of boundary condition ('volumetric' or 'flux').
    """

    def __init__(self, target_ids, target_type='node', bc_type='volumetric'):
        """
        Initialize a boundary condition.

        Args:
            target_ids (list of int): IDs of the nodes or conduits to which this 
                boundary condition applies.
            target_type (str, optional): 'node' or 'conduit'. Defaults to 'node'.
            bc_type (str, optional): Type of boundary condition ('volumetric' or 'flux').
                Defaults to 'volumetric'.
        """
        self.target_ids = target_ids
        self.target_type = target_type
        self.bc_type = bc_type

    def get_value(self, t):
        """
        Compute the boundary condition value at a given time.

        Args:
            t (float): Time at which to evaluate the boundary condition.

        Returns:
            float: The value of the boundary condition at time t.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement get_value")


class ConstantBC(BoundaryCondition):
    """
    Constant boundary condition applied to one or more nodes/conduits.

    Attributes:
        value (float): Constant value of the boundary condition.
    """

    def __init__(self, target_ids, value, target_type='node', bc_type='volumetric'):
        """
        Initialize a constant boundary condition.

        Args:
            target_ids (list of int): IDs of the target nodes or conduits.
            value (float): Constant value to apply.
            target_type (str, optional): 'node' or 'conduit'. Defaults to 'node'.
            bc_type (str, optional): 'volumetric' or 'flux'. Defaults to 'volumetric'.
        """
        super().__init__(target_ids, target_type, bc_type)
        self.value = value

    def get_value(self, t):
        """
        Return the constant boundary condition value.

        Args:
            t (float): Time (ignored for constant BC).

        Returns:
            float: The constant value.
        """
        return self.value


class RampBC(BoundaryCondition):
    """
    Linearly ramped boundary condition between two values over a time interval.

    Attributes:
        v0 (float): Initial value at t0.
        v1 (float): Final value at t1.
        t0 (float): Start time of ramp.
        t1 (float): End time of ramp.
    """

    def __init__(self, target_ids, value_start, value_end, t_start, t_end,
                 target_type='node', bc_type='volumetric'):
        """
        Initialize a ramped boundary condition.

        Args:
            target_ids (list of int): IDs of the target nodes or conduits.
            value_start (float): Value at the start of the ramp.
            value_end (float): Value at the end of the ramp.
            t_start (float): Start time of the ramp.
            t_end (float): End time of the ramp.
            target_type (str, optional): 'node' or 'conduit'. Defaults to 'node'.
            bc_type (str, optional): 'volumetric' or 'flux'. Defaults to 'volumetric'.
        """
        super().__init__(target_ids, target_type, bc_type)
        self.v0 = value_start
        self.v1 = value_end
        self.t0 = t_start
        self.t1 = t_end

    def get_value(self, t):
        """
        Return the boundary condition value at time t.

        Args:
            t (float): Time at which to evaluate the value.

        Returns:
            float: The value at time t, based on linear interpolation.
        """
        if t <= self.t0:
            return self.v0
        elif t >= self.t1:
            return self.v1
        else:
            return self.v0 + (t - self.t0) / (self.t1 - self.t0) * (self.v1 - self.v0)


class TimeSeriesBC(BoundaryCondition):
    """
    Time series-based boundary condition using linear interpolation.

    Attributes:
        times (array-like): List or array of time points.
        values (array-like): List or array of corresponding values.
    """

    def __init__(self, target_ids, times, values,
                 target_type='node', bc_type='volumetric'):
        """
        Initialize a time series boundary condition.

        Args:
            target_ids (list of int): IDs of the target nodes or conduits.
            times (list or np.ndarray): Times at which values are specified.
            values (list or np.ndarray): Values corresponding to the time points.
            target_type (str, optional): 'node' or 'conduit'. Defaults to 'node'.
            bc_type (str, optional): 'volumetric' or 'flux'. Defaults to 'volumetric'.
        """
        super().__init__(target_ids, target_type, bc_type)
        self.times = times
        self.values = values

    def get_value(self, t):
        """
        Interpolate the value at a given time using the supplied time series.

        Args:
            t (float): Time at which to evaluate the boundary condition.

        Returns:
            float: Interpolated value at time t.
        """
        return np.interp(t, self.times, self.values)

