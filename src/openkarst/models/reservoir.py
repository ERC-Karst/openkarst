#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reservoir model coupled to single openKARST nodes."""

from numbers import Real

import numpy as np


class UnconfinedReservoir:
    """
    Single-node unconfined reservoir with a base equal to the connected node height.

    The Saint-Venant controls the coupling timing:

    1. compute_exchange is called once at the start of a timestep using the
       accepted node water depth.
    2. The returned exchange is kept fixed during the hydraulic Picard solve.
    3. advance_reservoirs is called after the hydraulic timestep is accepted.

    Positive exchange means flow from the reservoir into the connected node.
    """

    def __init__(
        self,
        node,
        base_elevation,
        area,
        specific_yield,
        initial_water_depth,
        conductance,
        recharge=0.0,
        time=None,
        recharge_extrapolate='hold',
    ):
        self.node = int(node)

        try:
            self.base_elevation = float(base_elevation)
            self.area = float(area)
            self.specific_yield = float(specific_yield)
            self.water_depth = float(initial_water_depth)
            self.conductance = float(conductance)
            self.recharge_extrapolate = recharge_extrapolate.lower()
            self.current_t = float(0.0)
        except (TypeError, ValueError) as error:
            raise ValueError("Reservoir parameters must be numeric values.") from error

        parameters = (
            self.base_elevation,
            self.area,
            self.specific_yield,
            self.water_depth,
            self.conductance,
        )
        if not all(np.isfinite(value).all() for value in parameters):
            raise ValueError("Reservoir parameters must be finite values.")
        if self.recharge_extrapolate not in {'hold', 'zero'}:
            raise ValueError("recharge_extrapolate must be 'hold' or 'zero'.")
        if self.area <= 0.0:
            raise ValueError("area must be greater than zero.")
        if self.specific_yield <= 0.0:
            raise ValueError("specific_yield must be greater than zero.")
        if self.water_depth < 0.0:
            raise ValueError("initial_water_depth must be non-negative.")
        if self.conductance < 0.0:
            raise ValueError("conductance must be non-negative.")

        if time is None:
            if not isinstance(recharge, Real):
                raise ValueError("Constant recharge must be a scalar value.")
            self.recharge = float(recharge)
            self.time = None
        else:
            self.time = np.asarray(time, dtype=float)
            self.recharge = np.asarray(recharge, dtype=float)
            if self.time.ndim != 1 or self.recharge.ndim != 1:
                raise ValueError("Recharge time series must use 1D arrays.")
            if len(self.time) == 0:
                raise ValueError("Recharge time series must contain at least one value.")
            if len(self.time) != len(self.recharge):
                raise ValueError("Recharge times and values must have the same length.")
            if not np.all(np.isfinite(self.time)) or not np.all(np.isfinite(self.recharge)):
                raise ValueError("Recharge time series values must be finite.")
            if len(self.time) > 1 and not np.all(np.diff(self.time) > 0.0):
                raise ValueError("Recharge times must be strictly increasing.")

        # Exchange from the most recently accepted timestep
        self.last_exchange_rate = 0.0
        self.last_recharge_rate = self._get_recharge_value(self.current_t)

    def get_hydraulic_head(self):
        """Return the current reservoir hydraulic head [m]."""
        return self.base_elevation + self.water_depth

    def get_storage(self):
        """Return the current drainable reservoir storage [m^3]."""
        return self.area * self.specific_yield * self.water_depth
    
    def _get_recharge_value(self, t):
        """Returns interpolated value at time `t`.
        Args:
            t (float): Time at which to evaluate the BC.
        Returns:
            float: The interpolated or extrapolated boundary condition value.
        """
        #constant recharge: return value
        if self.time is None:
            return float(self.recharge)
        
        #timeseries: interpolate or extrapolate
        if t < self.time[0]:
            return 0.0 if self.recharge_extrapolate == 'zero' else float(self.recharge[0])
        elif t > self.time[-1]:
            return 0.0 if self.recharge_extrapolate == 'zero' else float(self.recharge[-1])
        else:
            return float(np.interp(t, self.time, self.recharge))

    def compute_exchange(self, node_water_depth, dt):
        """
        Compute reservoir-node exchange for the next hydraulic timestep.

        Args:
            node_water_depth (float): Accepted node water depth at timestep start [m].
            dt (float): Hydraulic timestep [s].

        Returns:
            float: Reservoir-node exchange rate [m^3/s]. Can be positive or negative.
        """
        # TODO (Jenny): Implement Q = conductance * (water_depth - node_water_depth),
        # or similar fashion. We discussed C_ex but in principle other forms may work (better)
        # Need to discuss how to possibly validate or compare...
        # Positive Q supplies the node. Limit positive Q so the accepted timestep
        # cannot withdraw more water than the reservoir contains.
        #calculate Q

        # I think instead of self.get_hydraulic_head() we should have only the water depth in the reservoir
        # We assume for now that the reservoir base is always at the same height as the connected node height.
        Q = self.conductance * (self.get_hydraulic_head() - node_water_depth)

        # Take into account self.dt as Q is a volumetric rate?
        Q = min(Q, self.get_storage())  # limit positive Q to available storage
        return Q 

    def advance(self, exchange_rate, dt):
        """
        Advance reservoir storage after an accepted hydraulic timestep.

        Args:
            exchange_rate (float): Cached reservoir-to-node exchange rate [m^3/s].
            dt (float): Accepted hydraulic timestep [s].
        """
        # TODO (Jenny): Here we need to ipdate the reservoir state e.g.
        # storage_old = self.get_storage()
        # storage_new = storage_old + (self.recharge - exchange_rate) * dt
        # Enforce non-negative storage, then update self.water_depth:
        # self.water_depth = storage_new / (self.area * self.specific_yield)

        # Lets find a way to use self.current_time from FlowSimulation
        self.current_t += dt
        
        storage_old = self.get_storage()
        current_recharge = self._get_recharge_value(self.current_t)
        self.last_recharge_rate = current_recharge
        storage_new = storage_old + (current_recharge - exchange_rate) * dt
        self.water_depth = max(0.0, storage_new / (self.area * self.specific_yield)) #enfore non-negative storage 
