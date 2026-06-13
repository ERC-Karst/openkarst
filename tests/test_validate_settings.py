import logging

import pytest

from openkarst.config.physical_properties import PhysicalProperties
from openkarst.config.geometry_settings import GeometrySettings
from openkarst.config.simulation_settings import SimulationSettings
from openkarst.config.solver_settings import SolverSettings
from openkarst.config.transport_settings import TransportSettings
from openkarst.config.validate_settings import validate_settings


def _valid_settings():
    return (
        PhysicalProperties(),
        GeometrySettings(),
        SolverSettings(),
        SimulationSettings(dt_init=1.0, dt_max=1.0, t_max=10.0),
        TransportSettings(),
        logging.getLogger("test"),
    )


def test_validate_settings_accepts_valid_defaults_with_required_times():
    validate_settings(*_valid_settings())


def test_validate_settings_rejects_invalid_physical_properties():
    physical_properties, geometry_settings, solver_settings, simulation_settings, transport_settings, logger = _valid_settings()
    physical_properties.water_density = 0.0

    with pytest.raises(ValueError, match="water_density"):
        validate_settings(
            physical_properties,
            geometry_settings,
            solver_settings,
            simulation_settings,
            transport_settings,
            logger,
        )


def test_validate_settings_rejects_unknown_geometry_backend():
    physical_properties, geometry_settings, solver_settings, simulation_settings, transport_settings, logger = _valid_settings()
    geometry_settings.backend = "unknown"

    with pytest.raises(ValueError, match="backend"):
        validate_settings(
            physical_properties,
            geometry_settings,
            solver_settings,
            simulation_settings,
            transport_settings,
            logger,
        )


def test_validate_settings_rejects_invalid_geometry_table_points():
    physical_properties, geometry_settings, solver_settings, simulation_settings, transport_settings, logger = _valid_settings()
    geometry_settings.table_points = 1

    with pytest.raises(ValueError, match="table_points"):
        validate_settings(
            physical_properties,
            geometry_settings,
            solver_settings,
            simulation_settings,
            transport_settings,
            logger,
        )


def test_validate_settings_rejects_missing_constant_timestep():
    physical_properties, geometry_settings, solver_settings, simulation_settings, transport_settings, logger = _valid_settings()
    simulation_settings.dt_init = None

    with pytest.raises(ValueError, match="dt_init"):
        validate_settings(
            physical_properties,
            geometry_settings,
            solver_settings,
            simulation_settings,
            transport_settings,
            logger,
        )
