#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level settings container for Saint-Venant simulations."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from openkarst.config.geometry_settings import GeometrySettings
from openkarst.config.physical_properties import PhysicalProperties
from openkarst.config.simulation_settings import SimulationSettings
from openkarst.config.solver_settings import SolverSettings
from openkarst.config.transport_settings import TransportSettings
from openkarst.config.validate_settings import validate_settings


@dataclass
class SaintVenantSettings:
    """Canonical settings object used by the Saint-Venant simulation."""

    physical: PhysicalProperties = field(default_factory=PhysicalProperties)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    solver: SolverSettings = field(default_factory=SolverSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    transport: TransportSettings = field(default_factory=TransportSettings)

    @staticmethod
    def from_user_input(
        physical_properties: Optional[Mapping[str, Any] | PhysicalProperties] = None,
        geometry_settings: Optional[Mapping[str, Any] | GeometrySettings] = None,
        solver_settings: Optional[Mapping[str, Any] | SolverSettings] = None,
        simulation_settings: Optional[Mapping[str, Any] | SimulationSettings] = None,
        transport_settings: Optional[Mapping[str, Any] | TransportSettings] = None,
    ) -> "SaintVenantSettings":
        """Build settings from user dictionaries or settings objects."""

        physical = (
            physical_properties
            if isinstance(physical_properties, PhysicalProperties)
            else PhysicalProperties(**(physical_properties or {}))
        )
        geometry = (
            geometry_settings
            if isinstance(geometry_settings, GeometrySettings)
            else GeometrySettings(**(geometry_settings or {}))
        )
        solver = (
            solver_settings
            if isinstance(solver_settings, SolverSettings)
            else SolverSettings(**(solver_settings or {}))
        )
        simulation = (
            simulation_settings
            if isinstance(simulation_settings, SimulationSettings)
            else SimulationSettings(**(simulation_settings or {}))
        )
        transport = (
            transport_settings
            if isinstance(transport_settings, TransportSettings)
            else TransportSettings(**(transport_settings or {}))
        )

        return SaintVenantSettings(
            physical=physical,
            geometry=geometry,
            solver=solver,
            simulation=simulation,
            transport=transport,
        )

    def validate(self, logger):
        """Validate the combined settings tree."""

        validate_settings(
            self.physical,
            self.geometry,
            self.solver,
            self.simulation,
            self.transport,
            logger,
        )
