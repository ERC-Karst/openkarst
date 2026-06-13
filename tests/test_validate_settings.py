import logging

import pytest

from openkarst.config.physical_properties import PhysicalProperties
from openkarst.config.simulation_settings import SimulationSettings
from openkarst.config.solver_settings import SolverSettings
from openkarst.config.transport_settings import TransportSettings
from openkarst.config.validate_settings import validate_settings


def _valid_settings():
    return (
        PhysicalProperties(),
        SolverSettings(),
        SimulationSettings(dt_init=1.0, dt_max=1.0, t_max=10.0),
        TransportSettings(),
        logging.getLogger("test"),
    )


def test_validate_settings_accepts_valid_defaults_with_required_times():
    validate_settings(*_valid_settings())


def test_validate_settings_rejects_invalid_physical_properties():
    physical_properties, solver_settings, simulation_settings, transport_settings, logger = _valid_settings()
    physical_properties.water_density = 0.0

    with pytest.raises(ValueError, match="water_density"):
        validate_settings(
            physical_properties,
            solver_settings,
            simulation_settings,
            transport_settings,
            logger,
        )


def test_validate_settings_rejects_missing_constant_timestep():
    physical_properties, solver_settings, simulation_settings, transport_settings, logger = _valid_settings()
    simulation_settings.dt_init = None

    with pytest.raises(ValueError, match="dt_init"):
        validate_settings(
            physical_properties,
            solver_settings,
            simulation_settings,
            transport_settings,
            logger,
        )
