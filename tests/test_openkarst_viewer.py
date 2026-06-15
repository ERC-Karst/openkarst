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
    assert fig.layout.shapes[0].x0 == 10.0
    np.testing.assert_array_equal(fig.data[0].x, np.array([0.0, 10.0]))
    np.testing.assert_array_equal(fig.data[0].y, np.array([1e-1, 1e-3]))


def test_build_convergence_figure_handles_missing_norm_histories():
    fig = _figure_builder()({"time": np.array([0.0, 10.0])}, time_idx=0)

    assert len(fig.data) == 0
    assert fig.layout.annotations[0].text == "No positive convergence norms"
