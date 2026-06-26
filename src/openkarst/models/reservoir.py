#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reservoir model coupled to single openKARST nodes."""

import math
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
        time=None
    ):
        self.node = int(node)

        try:
            self.base_elevation = float(base_elevation)
            self.area = float(area)
            self.specific_yield = float(specific_yield)
            self.water_depth = float(initial_water_depth)
            self.conductance = float(conductance)
            self.recharge = recharge #can be float or array
            self.time = time
            self.current_t = float(0.0)
            self.exchange_history = []
            self.time_history = []
            self.water_depth_history = []
            self.recharge_history = []
        except (TypeError, ValueError) as error:
            raise ValueError("Reservoir parameters must be numeric values.") from error

        parameters = (
            self.base_elevation,
            self.area,
            self.specific_yield,
            self.water_depth,
            self.conductance,
            self.recharge,
        )
        if not all(np.isfinite(value).all() for value in parameters):
            raise ValueError("Reservoir parameters must be finite values.")
        if self.area <= 0.0:
            raise ValueError("area must be greater than zero.")
        if self.specific_yield <= 0.0:
            raise ValueError("specific_yield must be greater than zero.")
        if self.water_depth < 0.0:
            raise ValueError("initial_water_depth must be non-negative.")
        if self.conductance < 0.0:
            raise ValueError("conductance must be non-negative.")

        # Exchange from the most recently accepted timestep
        self.last_exchange_rate = 0.0

    def get_hydraulic_head(self):
        """Return the current reservoir hydraulic head [m]."""
        return self.base_elevation + self.water_depth

    def get_storage(self):
        """Return the current drainable reservoir storage [m^3]."""
        return self.area * self.specific_yield * self.water_depth
    
    def _record_history(self, exchange_rate): 
        """Record exchange rates + water depth"""
        self.time_history.append(self.current_t)
        self.exchange_history.append(exchange_rate)
        self.water_depth_history.append(self.water_depth)
        self.recharge_history.append(self._get_recharge_value(self.current_t))


    def _get_recharge_value(self, t, extrapolate_mode = 'zero'):
        """Returns interpolated value at time `t`.
        Args:
            t (float): Time at which to evaluate the BC.
        Returns:
            float: The interpolated or extrapolated boundary condition value.
        """
        #constant recharge: return value
        if type(self.recharge) == float or type(self.recharge) == int:
            return float(self.recharge)
        
        #timeseries: interpolate or extrapolate
        if t < self.time[0]:
            return 0.0 if extrapolate_mode == 'zero' else float(self.values[0])
        elif t > self.time[-1]:
            return 0.0 if extrapolate_mode == 'zero' else float(self.values[-1])
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
        storage_new = storage_old + (current_recharge - exchange_rate) * dt
        self.water_depth = max(0.0, storage_new / (self.area * self.specific_yield)) #enfore non-negative storage 
        self._record_history(exchange_rate) #record history of exchange and water depth