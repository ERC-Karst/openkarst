import logging
from types import SimpleNamespace

import numpy as np
import pytest

from openkarst.io.results_handling import initialize_results_container, store_results


def test_initialize_results_container_uses_enabled_valid_outputs_only():
    results = initialize_results_container(
        {
            "output_interval": 10.0,
            "time": True,
            "flowrates": True,
            "water_depths": False,
        },
        logging.getLogger("test"),
    )

    assert results == {"time": [], "flowrates": []}


def test_initialize_results_container_rejects_unknown_outputs():
    with pytest.raises(ValueError, match="Invalid keys"):
        initialize_results_container({"not_a_result": True}, logging.getLogger("test"))


def test_store_results_appends_copies_of_mutable_arrays():
    simulation = SimpleNamespace(
        convergence_fails=1,
        Q=np.array([1.0, 2.0]),
        _v_mid_last=np.array([0.5, 0.6]),
        y=np.array([0.1, 0.2, 0.3]),
        current_time=4.0,
        dt=0.25,
        relative_y_l2_norm=1e-4,
        Re_conduit=np.array([100.0, 200.0]),
        picard_iterations_last=3,
        C=np.array([0.0, 0.1, 0.2]),
        M=np.array([0.0, 1.0, 2.0]),
    )
    results = {
        "convergence_fails": [],
        "flowrates": [],
        "velocities": [],
        "water_depths": [],
        "time": [],
        "time_step_size": [],
        "l2_norms": [],
        "reynolds_numbers": [],
        "picard_iterations": [],
        "concentrations": [],
        "mass": [],
    }

    stored = store_results(simulation, results)
    simulation.Q[0] = 99.0
    simulation.y[0] = 99.0

    assert stored["convergence_fails"] == [1]
    assert stored["time"] == [4.0]
    assert stored["time_step_size"] == [0.25]
    np.testing.assert_array_equal(stored["flowrates"][0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(stored["water_depths"][0], np.array([0.1, 0.2, 0.3]))
