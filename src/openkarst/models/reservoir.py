#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reservoir model coupled to single openKARST nodes."""

import math


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
    ):
        self.node = int(node)

        try:
            self.base_elevation = float(base_elevation)
            self.area = float(area)
            self.specific_yield = float(specific_yield)
            self.water_depth = float(initial_water_depth)
            self.conductance = float(conductance)
            self.recharge = float(recharge)
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
        if not all(math.isfinite(value) for value in parameters):
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
        raise NotImplementedError(
            "TODO (Jenny): implement UnconfinedReservoir.compute_exchange()."
        )

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
        raise NotImplementedError(
            "TODO (Jenny): implement UnconfinedReservoir.advance()."
        )
