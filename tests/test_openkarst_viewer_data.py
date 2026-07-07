import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None,
    reason="Plotly viewer dependencies are not installed",
)


def _sample_geometry():
    return {
        "pore.coords": np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
        "throat.conns": np.array([
            [0, 1],
            [1, 2],
        ]),
    }


def test_profile_context_normalizes_transposed_water_depths():
    from openkarst.visualization.viewer.data import _precompute_profile_context

    results = {
        "time": np.array([0.0, 1.0]),
        "water_depths": np.array([
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ]),
    }

    context = _precompute_profile_context(_sample_geometry(), results)

    expected = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    np.testing.assert_array_equal(context["water_depths"], expected)
    np.testing.assert_array_equal(context["fields"]["depth"]["array"], expected)


def test_profile_figure_uses_normalized_water_depths_for_selected_nodes():
    from openkarst.visualization.viewer.data import _precompute_profile_context
    from openkarst.visualization.viewer.figures import _build_profile_figure

    results = {
        "time": np.array([0.0, 1.0]),
        "water_depths": np.array([
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ]),
    }
    context = _precompute_profile_context(_sample_geometry(), results)

    fig = _build_profile_figure(
        results,
        context,
        time_idx=1,
        node_ids=[2],
        node_step=1,
        depth_scale=1.0,
        color_field="depth",
        use_log_color_scale=False,
        show_vertical_bars=False,
        diameter_aware=False,
        color_edges_by_diameter=False,
        diameter_scale=1.0,
        camera=None,
        obs_node_colors={2: "#1f77b4"},
    )

    selected_node_trace = fig.data[-1]
    assert selected_node_trace.x[0] == pytest.approx(2.0)
    assert selected_node_trace.z[0] == pytest.approx(6.0)
