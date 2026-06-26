from types import SimpleNamespace

import numpy as np
import pytest

from openkarst.io.observation_recorder import ObservationRecorder


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


def test_connected_abs_flowrate_sums_absolute_connected_conduit_flows():
    flow_sim = SimpleNamespace(
        y_new=np.array([0.1, 0.2, 0.3]),
        Q_new=np.array([1.5, -2.5]),
        n_indices1=np.array([0, 1]),
        n_indices2=np.array([1, 2]),
    )
    recorder = ObservationRecorder(
        nodes=[0, 1, 2],
        variables=["water_depth", "connected_abs_flowrate"],
        interval=1.0,
    )

    recorder.record(5.0, flow_sim)
    df = recorder.to_dataframe()

    np.testing.assert_array_equal(df["water_depth"], np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(
        df["connected_abs_flowrate"],
        np.array([1.5, 4.0, 2.5]),
    )


def test_connected_net_flowrate_sums_signed_connected_conduit_flows_into_node():
    flow_sim = SimpleNamespace(
        y_new=np.array([0.1, 0.2, 0.3]),
        Q_new=np.array([1.5, -2.5]),
        n_indices1=np.array([0, 1]),
        n_indices2=np.array([1, 2]),
    )
    recorder = ObservationRecorder(
        nodes=[0, 1, 2],
        variables=["connected_net_flowrate"],
        interval=1.0,
    )

    recorder.record(5.0, flow_sim)
    df = recorder.to_dataframe()

    np.testing.assert_array_equal(
        df["connected_net_flowrate"],
        np.array([-1.5, 4.0, -2.5]),
    )


def test_inflow_is_not_supported_as_observation_variable():
    with pytest.raises(ValueError, match="Unsupported observation variable 'inflow'"):
        ObservationRecorder(nodes=[0], variables=["inflow"], interval=1.0)


def test_reservoir_observation_variables_are_recorded_by_node():
    flow_sim = SimpleNamespace(
        y_new=np.array([0.1, 0.2, 0.3]),
        Q_new=np.array([1.5, -2.5]),
        n_indices1=np.array([0, 1]),
        n_indices2=np.array([1, 2]),
        reservoirs=[
            _reservoir(1, 2.0, 12.0, 200.0, 0.003, 0.001),
        ],
    )
    recorder = ObservationRecorder(
        nodes=[1],
        variables=[
            "reservoir_water_depth",
            "reservoir_head",
            "reservoir_storage",
            "reservoir_exchange",
            "reservoir_recharge",
        ],
        interval=1.0,
    )

    recorder.record(5.0, flow_sim)
    df = recorder.to_dataframe()

    assert df.loc[0, "node"] == 1
    assert df.loc[0, "reservoir_water_depth"] == 2.0
    assert df.loc[0, "reservoir_head"] == 12.0
    assert df.loc[0, "reservoir_storage"] == 200.0
    assert df.loc[0, "reservoir_exchange"] == 0.003
    assert df.loc[0, "reservoir_recharge"] == 0.001
