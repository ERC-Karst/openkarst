from types import SimpleNamespace

import numpy as np
import pytest

from openkarst.io.observation_recorder import ObservationRecorder


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
