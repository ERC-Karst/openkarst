#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility imports for the browser-based openKARST viewer.

The implementation lives in :mod:`openkarst.visualization.viewer`.  This
module keeps the original import path stable for existing examples, notebooks,
and tests.
"""

from .viewer.camera import (
    _camera_for_view,
    _camera_from_relayout,
    _default_3d_camera,
    _normalize_camera,
    _set_nested,
)
from .viewer.constants import (
    COLOR_CYCLE,
    DEFAULT_DEPTH_SCALE,
    DEFAULT_OBS_COMPACT_LEGEND_MAX_TRACES,
    DEFAULT_OBS_INLINE_LEGEND_MAX_TRACES,
    DEFAULT_OBS_RENDER_POINTS,
    DEFAULT_PLAY_STRIDE,
    DIAMETER_CANDIDATES,
    DIAMETER_COLOR_SCALE,
    FIELD_COLOR_SCALES,
    FIELD_LABELS,
    FIELD_RESULT_CANDIDATES,
    FIELD_SYMMETRIC_RANGE,
    FLOW_COLUMN_CANDIDATES,
    HEADER_LOGO,
    OBSERVATION_AXIS_LABELS,
    OBSERVATION_BASE_COLUMNS,
    OBSERVATION_LABELS,
    VIEW_CAMERAS,
)
from .viewer.data import (
    _as_time_node_array,
    _build_diameter_segments,
    _default_observation_property,
    _diameter_values,
    _field_specs,
    _finite_range,
    _flow_column,
    _format_number,
    _int_value,
    _is_l2_norm_column,
    _is_observation_base_column,
    _log10_color_values,
    _log_color_range,
    _log_colorbar_ticks,
    _number_value,
    _observation_axis_label,
    _observation_label,
    _observation_nodes,
    _observation_property_specs,
    _positive_range,
    _precompute_observation_context,
    _precompute_profile_context,
    _require_time_node_array,
    _thin_frame,
    _throat_to_node_values,
    _time_axis_range,
)
from .viewer.figures import (
    _build_convergence_figure,
    _build_observation_figure,
    _build_profile_figure,
    _compact_colorbar,
    _convergence_series,
    _empty_observation_figure,
    _scene_aspectratio,
)

__all__ = [
    "create_openkarst_viewer_app",
    "launch_openkarst_viewer",
]


def __getattr__(name):
    if name in {"create_openkarst_viewer_app", "_viewer_assets_dir"}:
        from .viewer import app

        return getattr(app, name)
    if name in {
        "launch_openkarst_viewer",
        "_in_google_colab",
        "_print_colab_proxy_url",
        "_run_dash_server",
        "_server_url",
        "_show_colab_iframe",
    }:
        from .viewer import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
