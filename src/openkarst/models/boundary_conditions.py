class BoundaryCondition:
    def __init__(self, target_ids, target_type='node'):
        self.target_ids = target_ids
        self.target_type = target_type

    def get_value(self, t):
        raise NotImplementedError("Subclasses must implement get_value")


class ConstantBC(BoundaryCondition):
    def __init__(self, target_ids, value, target_type='node'):
        super().__init__(target_ids, target_type)
        self.value = value

    def get_value(self, t):
        return self.value


class RampBC(BoundaryCondition):
    def __init__(self, target_ids, value_start, value_end, t_start, t_end):
        super().__init__(target_ids)
        self.v0 = value_start
        self.v1 = value_end
        self.t0 = t_start
        self.t1 = t_end

    def get_value(self, t):
        if t <= self.t0:
            return self.v0
        elif t >= self.t1:
            return self.v1
        else:
            return self.v0 + (t - self.t0) / (self.t1 - self.t0) * (self.v1 - self.v0)


class TimeSeriesBC(BoundaryCondition):
    def __init__(self, target_ids, times, values):
        super().__init__(target_ids)
        self.times = times
        self.values = values

    def get_value(self, t):
        import numpy as np
        return np.interp(t, self.times, self.values)
