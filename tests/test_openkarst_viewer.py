import importlib.util

import numpy as np
import pytest


VIEWER_DEPS_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("plotly") is not None
)
pytestmark = pytest.mark.skipif(
    not VIEWER_DEPS_AVAILABLE,
    reason="Dash/Plotly viewer dependencies are not installed",
)


def _figure_builder():
    from openkarst.visualization.openkarst_viewer import _build_convergence_figure

    return _build_convergence_figure


def _observation_helpers():
    from openkarst.visualization.openkarst_viewer import (
        _build_observation_figure,
        _precompute_observation_context,
    )

    return _precompute_observation_context, _build_observation_figure


def test_build_convergence_figure_plots_norm_histories():
    results = {
        "time": np.array([0.0, 10.0, 20.0]),
        "y_l2_norms": np.array([1e-1, 1e-3, 0.0]),
        "Q_l2_norms": np.array([2e-1, 4e-3, 5e-4]),
    }

    fig = _figure_builder()(results, time_idx=1)

    assert [trace.name for trace in fig.data] == ["Water depth L2", "Discharge L2"]
    assert fig.layout.yaxis.type == "log"
    assert fig.layout.yaxis.tickformat == ".0e"
    assert fig.layout.yaxis.exponentformat == "e"
    assert fig.layout.yaxis.showexponent == "all"
    assert fig.layout.legend.y == 0.98
    assert fig.layout.legend.xanchor == "right"
    assert fig.layout.shapes[0].x0 == 10.0
    np.testing.assert_array_equal(fig.data[0].x, np.array([0.0, 10.0]))
    np.testing.assert_array_equal(fig.data[0].y, np.array([1e-1, 1e-3]))


def test_build_convergence_figure_handles_missing_norm_histories():
    fig = _figure_builder()({"time": np.array([0.0, 10.0])}, time_idx=0)

    assert len(fig.data) == 0
    assert fig.layout.annotations[0].text == "No positive convergence norms"


def test_observation_context_uses_numeric_non_l2_columns_and_prefers_flow():
    import pandas as pd

    precompute_context, _ = _observation_helpers()
    obs_df = pd.DataFrame({
        "time": [0.0, 10.0],
        "node": [1, 1],
        "water_depth": [0.2, 0.3],
        "connected_abs_flowrate": [1.0, 1.2],
        "connected_net_flowrate": [0.8, 0.9],
        "y_l2_norms": [1e-1, 1e-2],
        "label": ["a", "b"],
    })

    context = precompute_context(obs_df)

    assert list(context["property_specs"]) == [
        "water_depth",
        "connected_abs_flowrate",
        "connected_net_flowrate",
    ]
    assert context["default_property"] == "connected_net_flowrate"


def test_build_observation_figure_plots_selected_property():
    import pandas as pd

    precompute_context, build_figure = _observation_helpers()
    results = {"time": np.array([0.0, 10.0, 20.0])}
    obs_df = pd.DataFrame({
        "time": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
        "node": [1, 1, 1, 2, 2, 2],
        "water_depth": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "connected_abs_flowrate": [1.0, 1.2, 1.3, 2.0, 2.2, 2.3],
    })
    context = precompute_context(obs_df)

    fig = build_figure(
        results,
        obs_df,
        context,
        time_idx=1,
        node_ids=[1],
        obs_node_colors={1: "#1f77b4"},
        observation_property="water_depth",
    )

    assert [trace.name for trace in fig.data] == ["Water depth [m] - node 1"]
    assert fig.layout.yaxis.title.text == "Water depth [m]"
    np.testing.assert_array_equal(fig.data[0].x, np.array([0.0, 10.0]))
    np.testing.assert_array_equal(fig.data[0].y, np.array([0.1, 0.2]))
