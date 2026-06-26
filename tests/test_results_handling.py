import logging
from types import SimpleNamespace

import numpy as np
import pytest

from openkarst.io.results_handling import initialize_results_container, store_results


def _reservoir(
    node,
    water_depth,
    hydraulic_head,
    storage,
    exchange_rate,
    recharge_rate,
):
    reservoir = SimpleNamespace(
        node=node,
        reservoir_water_depth=water_depth,
        last_exchange_rate=exchange_rate,
        last_recharge_rate=recharge_rate,
    )
    reservoir.get_hydraulic_head = lambda: hydraulic_head
    reservoir.get_storage = lambda: storage
    return reservoir


def test_initialize_results_container_uses_enabled_valid_outputs_only():
    results = initialize_results_container(
        {
            "output_interval": 10.0,
            "time": True,
            "flowrates": True,
            "y_l2_norms": True,
            "Q_l2_norms": True,
            "picard_iterations_total": True,
            "reservoir_storage": True,
            "water_depths": False,
        },
        logging.getLogger("test"),
    )

    assert results == {
        "time": [],
        "flowrates": [],
        "y_l2_norms": [],
        "Q_l2_norms": [],
        "picard_iterations_total": [],
        "reservoir_storage": [],
    }


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
        relative_Q_l2_norm=3e-4,
        Re_conduit=np.array([100.0, 200.0]),
        picard_iterations_last=3,
        picard_iterations_total=8,
        C=np.array([0.0, 0.1, 0.2]),
        M=np.array([0.0, 1.0, 2.0]),
        reservoirs=[
            _reservoir(1, 2.0, 12.0, 200.0, 0.003, 0.001),
            _reservoir(3, 1.5, 11.5, 150.0, -0.002, 0.0),
        ],
    )
    results = {
        "convergence_fails": [],
        "flowrates": [],
        "velocities": [],
        "water_depths": [],
        "time": [],
        "time_step_size": [],
        "l2_norms": [],
        "y_l2_norms": [],
        "Q_l2_norms": [],
        "reynolds_numbers": [],
        "picard_iterations": [],
        "picard_iterations_total": [],
        "concentrations": [],
        "mass": [],
        "reservoir_nodes": [],
        "reservoir_water_depths": [],
        "reservoir_heads": [],
        "reservoir_storage": [],
        "reservoir_exchange": [],
        "reservoir_recharge": [],
    }

    stored = store_results(simulation, results)
    simulation.Q[0] = 99.0
    simulation.y[0] = 99.0

    assert stored["convergence_fails"] == [1]
    assert stored["time"] == [4.0]
    assert stored["time_step_size"] == [0.25]
    assert stored["l2_norms"] == [1e-4]
    assert stored["y_l2_norms"] == [1e-4]
    assert stored["Q_l2_norms"] == [3e-4]
    assert stored["picard_iterations"] == [3]
    assert stored["picard_iterations_total"] == [8]
    np.testing.assert_array_equal(stored["flowrates"][0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(stored["water_depths"][0], np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(stored["reservoir_nodes"][0], np.array([1, 3]))
    np.testing.assert_array_equal(stored["reservoir_water_depths"][0], np.array([2.0, 1.5]))
    np.testing.assert_array_equal(stored["reservoir_heads"][0], np.array([12.0, 11.5]))
    np.testing.assert_array_equal(stored["reservoir_storage"][0], np.array([200.0, 150.0]))
    np.testing.assert_array_equal(stored["reservoir_exchange"][0], np.array([0.003, -0.002]))
    np.testing.assert_array_equal(stored["reservoir_recharge"][0], np.array([0.001, 0.0]))
