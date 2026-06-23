import logging

from openkarst.config import SaintVenantSettings, SimulationSettings


def test_saint_venant_settings_builds_from_user_dicts():
    settings = SaintVenantSettings.from_user_input(
        simulation_settings={
            "adaptive_timesteps": False,
            "dt_init": 1.0,
            "dt_max": 1.0,
            "t_max": 200.0,
        }
    )

    assert settings.simulation.t_max == 200.0
    assert settings.simulation.dt_init == 1.0
    settings.validate(logging.getLogger("test"))


def test_saint_venant_settings_accepts_settings_objects():
    simulation = SimulationSettings(dt_init=1.0, dt_max=1.0, t_max=200.0)

    settings = SaintVenantSettings.from_user_input(simulation_settings=simulation)

    assert settings.simulation is simulation
