#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:40:08 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import threading
import time
import webbrowser
from copy import deepcopy
from pathlib import Path

import dash
from dash import Input, Output, State, dcc, html
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go


COLOR_CYCLE = pc.qualitative.Plotly
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_PLAY_STRIDE = 1
DEFAULT_OBS_RENDER_POINTS = 1200
FLOW_COLUMN_CANDIDATES = (
    "connected_abs_flowrate",
    "Q",
    "flowrate",
    "flowrates",
)
OBSERVATION_BASE_COLUMNS = {"time", "node"}
OBSERVATION_LABELS = {
    "water_depth": "Water depth [m]",
    "connected_abs_flowrate": "Connected |Q| [m3/s]",
    "Q": "Flow rate [m3/s]",
    "q": "Flow rate [m3/s]",
    "flowrate": "Flow rate [m3/s]",
    "flowrates": "Flow rate [m3/s]",
    "concentrations": "Concentration [kg/m3]",
    "concentration": "Concentration [kg/m3]",
    "C": "Concentration [kg/m3]",
    "c": "Concentration [kg/m3]",
    "mass": "Mass [kg]",
}
OBSERVATION_AXIS_LABELS = {
    "water_depth": "Water depth [m]",
    "connected_abs_flowrate": "Connected |Q| [m<sup>3</sup>/s]",
    "Q": "Flow rate [m<sup>3</sup>/s]",
    "q": "Flow rate [m<sup>3</sup>/s]",
    "flowrate": "Flow rate [m<sup>3</sup>/s]",
    "flowrates": "Flow rate [m<sup>3</sup>/s]",
    "concentrations": "Concentration [kg/m<sup>3</sup>]",
    "concentration": "Concentration [kg/m<sup>3</sup>]",
    "C": "Concentration [kg/m<sup>3</sup>]",
    "c": "Concentration [kg/m<sup>3</sup>]",
    "mass": "Mass [kg]",
}
FIELD_RESULT_CANDIDATES = {
    "flowrate": ("flowrates", "flowrate", "Q", "q", "flow", "flows", "discharge", "discharges"),
    "velocity": ("velocities", "velocity", "v"),
    "concentration": ("concentrations", "concentration", "C", "c"),
}
FIELD_LABELS = {
    "depth": "Depth [m]",
    "flowrate": "Flow rate [m3/s]",
    "velocity": "Velocity [m/s]",
    "concentration": "Concentration [kg/m3]",
}
FIELD_COLOR_SCALES = {
    "depth": "Viridis",
    "flowrate": "RdBu",
    "velocity": "Plasma",
    "concentration": "Turbo",
}
FIELD_SYMMETRIC_RANGE = {"flowrate"}
DIAMETER_CANDIDATES = (
    "throat.diameter",
    "throat.diameters",
    "diameters",
    "diameter",
)
DIAMETER_COLOR_SCALE = "Cividis"
HEADER_LOGO = "openkarst_header_color.png"
VIEW_CAMERAS = {
    "3d": dict(
        eye=dict(x=1.4, y=1.4, z=1.4),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="perspective"),
    ),
    "xy": dict(
        eye=dict(x=0, y=0, z=2.5),
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="orthographic"),
    ),
    "xz": dict(
        eye=dict(x=0, y=-2.5, z=0),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="orthographic"),
    ),
    "yz": dict(
        eye=dict(x=2.5, y=0, z=0),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="orthographic"),
    ),
}


def _in_google_colab():
    """Return True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _viewer_assets_dir():
    for parent in Path(__file__).resolve().parents:
        assets_dir = parent / "assets"
        if (assets_dir / HEADER_LOGO).is_file():
            return assets_dir
    return None


def _observation_nodes(obs_df):
    if obs_df is None or obs_df.empty or "node" not in obs_df.columns:
        return []
    return sorted(int(node) for node in obs_df["node"].dropna().unique())


def _flow_column(obs_df):
    if obs_df is None:
        return None
    for column in FLOW_COLUMN_CANDIDATES:
        if column in obs_df.columns:
            return column
    return None


def _is_l2_norm_column(column):
    return "l2" in str(column).lower()


def _is_observation_base_column(column):
    return str(column).lower() in OBSERVATION_BASE_COLUMNS


def _observation_label(column):
    return OBSERVATION_LABELS.get(column, str(column).replace("_", " ").title())


def _observation_axis_label(column):
    return OBSERVATION_AXIS_LABELS.get(column, _observation_label(column))


def _observation_property_specs(obs_df):
    if obs_df is None or obs_df.empty:
        return {}

    specs = {}
    for column in obs_df.columns:
        if _is_observation_base_column(column) or _is_l2_norm_column(column):
            continue
        try:
            values = np.asarray(obs_df[column], dtype=float)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).any():
            continue
        specs[column] = {
            "label": _observation_label(column),
            "axis_label": _observation_axis_label(column),
            "range": _finite_range(values),
        }
    return specs


def _default_observation_property(property_specs, obs_df):
    if not property_specs:
        return None

    flow_column = _flow_column(obs_df)
    if flow_column in property_specs:
        return flow_column

    return next(iter(property_specs))


def _finite_range(values, symmetric=False):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [0.0, 1.0]

    if symmetric:
        limit = max(float(np.max(np.abs(finite))), 1e-12)
        return [-limit, limit]

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-12)
        return [vmin - pad, vmax + pad]
    return [vmin, vmax]


def _positive_range(values):
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return None

    vmin = float(np.min(positive))
    vmax = float(np.max(positive))
    if np.isclose(vmin, vmax):
        return None
    return [vmin, vmax]


def _log_color_range(field_id, values):
    if field_id in FIELD_SYMMETRIC_RANGE:
        return None
    return _positive_range(values)


def _log10_color_values(values, positive_min):
    arr = np.asarray(values, dtype=float)
    clipped = np.where(np.isfinite(arr), np.maximum(arr, positive_min), np.nan)
    return np.log10(clipped)


def _log_colorbar_ticks(positive_range):
    vmin, vmax = positive_range
    log_min = float(np.log10(vmin))
    log_max = float(np.log10(vmax))
    first_power = int(np.ceil(log_min))
    last_power = int(np.floor(log_max))
    powers = list(range(first_power, last_power + 1))

    if 2 <= len(powers) <= 6:
        tickvals = powers
        ticktext = [f"{10 ** power:g}" for power in powers]
    else:
        tickvals = np.linspace(log_min, log_max, 5)
        ticktext = [f"{10 ** value:g}" for value in tickvals]

    return {
        "tickvals": tickvals,
        "ticktext": ticktext,
    }


def _number_value(value, default=1.0, minimum=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if not np.isfinite(result):
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    return result


def _int_value(value, default=1, minimum=None):
    return int(round(_number_value(value, default=default, minimum=minimum)))


def _format_number(value, integer=False):
    if integer:
        return str(int(round(value)))
    return f"{float(value):g}"


def _throat_to_node_values(throat_values, conns, n_nodes):
    throat_values = np.asarray(throat_values, dtype=float)
    if throat_values.ndim == 1:
        throat_values = throat_values[np.newaxis, :]

    node_values = np.zeros((throat_values.shape[0], n_nodes), dtype=float)
    counts = np.zeros(n_nodes, dtype=float)
    for throat_idx, (i, j) in enumerate(conns):
        vals = throat_values[:, throat_idx]
        node_values[:, i] += vals
        node_values[:, j] += vals
        counts[i] += 1.0
        counts[j] += 1.0

    counts[counts == 0.0] = 1.0
    return node_values / counts


def _as_time_node_array(values, n_times, n_nodes, conns):
    arr = np.asarray(values, dtype=float)
    n_throats = len(conns)

    if arr.ndim == 1:
        if arr.shape[0] == n_nodes:
            return np.tile(arr, (n_times, 1))
        if arr.shape[0] == n_throats:
            node_values = _throat_to_node_values(arr, conns, n_nodes)
            return np.tile(node_values, (n_times, 1))
        return None

    if arr.ndim != 2:
        return None

    if arr.shape == (n_times, n_nodes):
        return arr
    if arr.shape == (n_nodes, n_times):
        return arr.T
    if arr.shape == (n_times, n_throats):
        return _throat_to_node_values(arr, conns, n_nodes)
    if arr.shape == (n_throats, n_times):
        return _throat_to_node_values(arr.T, conns, n_nodes)
    return None


def _field_specs(results, geometry):
    t = results["time"]
    coords = geometry["pore.coords"]
    conns = geometry["throat.conns"]
    n_times = len(t)
    n_nodes = len(coords)
    water_depths = _as_time_node_array(results["water_depths"], n_times, n_nodes, conns)

    specs = {
        "depth": {
            "label": FIELD_LABELS["depth"],
            "array": water_depths,
            "range": _finite_range(water_depths),
            "log_range": _log_color_range("depth", water_depths),
            "colorscale": FIELD_COLOR_SCALES["depth"],
        }
    }

    for field_id, candidates in FIELD_RESULT_CANDIDATES.items():
        for key in candidates:
            if key not in results:
                continue
            field_values = _as_time_node_array(results[key], n_times, n_nodes, conns)
            if field_values is None:
                continue
            specs[field_id] = {
                "label": FIELD_LABELS[field_id],
                "array": field_values,
                "range": _finite_range(
                    field_values,
                    symmetric=field_id in FIELD_SYMMETRIC_RANGE,
                ),
                "log_range": _log_color_range(field_id, field_values),
                "colorscale": FIELD_COLOR_SCALES[field_id],
                "source": key,
            }
            break

    return specs


def _diameter_values(geometry, n_throats):
    for key in DIAMETER_CANDIDATES:
        if key not in geometry:
            continue
        values = np.asarray(geometry[key], dtype=float).reshape(-1)
        if values.size == n_throats:
            return values
    return None


def _build_diameter_segments(x, y, z, conns, diameters):
    if diameters is None:
        return []

    finite = diameters[np.isfinite(diameters)]
    if finite.size == 0:
        return []

    bin_count = min(5, max(1, finite.size))
    quantiles = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bin_count + 1)))
    if quantiles.size <= 2:
        bin_ids = np.zeros(len(diameters), dtype=int)
        bin_count = 1
    else:
        bin_ids = np.digitize(diameters, quantiles[1:-1], right=True)
        bin_count = len(quantiles) - 1

    dmin = float(np.min(finite))
    dmax = float(np.max(finite))
    dspan = max(dmax - dmin, 1e-12)
    colors = pc.sample_colorscale(
        DIAMETER_COLOR_SCALE,
        np.linspace(0.12, 0.88, bin_count),
    )
    segments = []

    for bin_id in range(bin_count):
        throat_ids = np.where(bin_ids == bin_id)[0]
        if throat_ids.size == 0:
            continue
        median_diameter = float(np.nanmedian(diameters[throat_ids]))
        lower = float(np.nanmin(diameters[throat_ids]))
        upper = float(np.nanmax(diameters[throat_ids]))
        width = 1.5 + 5.5 * (median_diameter - dmin) / dspan
        line_x = []
        line_y = []
        line_z = []
        for throat_idx in throat_ids:
            i, j = conns[throat_idx]
            line_x.extend([float(x[i]), float(x[j]), None])
            line_y.extend([float(y[i]), float(y[j]), None])
            line_z.extend([float(z[i]), float(z[j]), None])
        segments.append({
            "x": line_x,
            "y": line_y,
            "z": line_z,
            "width": float(width),
            "diameter": median_diameter,
            "label": f"{lower:.2f}-{upper:.2f} m",
            "color": colors[bin_id],
        })

    return segments


def _compact_colorbar(title, x_position, length):
    return dict(
        title=title,
        orientation="h",
        x=x_position,
        y=-0.12,
        xanchor="center",
        yanchor="top",
        len=length,
        thickness=10,
        title_font=dict(size=9),
        tickfont=dict(size=8),
    )


def _empty_observation_figure(message="No observation points"):
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#6b7280"),
            )
        ],
    )
    return fig


def _convergence_series(results, key, n_times):
    if key not in results:
        return None

    values = np.asarray(results[key], dtype=float).reshape(-1)
    if values.size == 0:
        return None

    return values[: min(values.size, n_times)]


def _time_axis_range(t):
    start = float(t[0])
    end = float(t[-1])
    if not np.isclose(start, end):
        return [start, end]

    pad = max(abs(start) * 0.05, 1.0)
    return [start - pad, end + pad]


def _build_convergence_figure(results, time_idx):
    t = np.asarray(results["time"], dtype=float).reshape(-1)
    if t.size == 0:
        return _empty_observation_figure("No convergence data")

    current_idx = int(np.clip(time_idx, 0, t.size - 1))
    current_time = float(t[current_idx])
    fig = go.Figure()
    traces_added = 0

    for key, label, color in (
        ("y_l2_norms", "Water depth L2", "#1f77b4"),
        ("Q_l2_norms", "Discharge L2", "#d62728"),
    ):
        values = _convergence_series(results, key, t.size)
        if values is None:
            continue

        series_time = t[: values.size]
        end_idx = int(np.searchsorted(series_time, current_time, side="right"))
        visible_values = np.where(values[:end_idx] > 0.0, values[:end_idx], np.nan)
        if not np.isfinite(np.where(values > 0.0, values, np.nan)).any():
            continue

        fig.add_trace(
            go.Scattergl(
                x=series_time[:end_idx],
                y=visible_values,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=(
                    "Time: %{x:g} s<br>"
                    f"{label}: %{{y:.3e}}"
                    "<extra></extra>"
                ),
            )
        )
        traces_added += 1

    if traces_added == 0:
        return _empty_observation_figure("No positive convergence norms")

    fig.add_vline(x=current_time, line_width=1, line_dash="dot", line_color="#374151")
    fig.update_layout(
        margin=dict(l=50, r=30, t=42, b=42),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="#dbe1ea",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(title="Time [s]", range=_time_axis_range(t), fixedrange=True),
        yaxis=dict(
            title="Relative L2 norm",
            type="log",
            tickformat=".0e",
            exponentformat="e",
            showexponent="all",
            fixedrange=True,
        ),
    )
    return fig


def _thin_frame(df, max_points=DEFAULT_OBS_RENDER_POINTS):
    if len(df) <= max_points:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, max_points, dtype=int))
    return df.iloc[idx]


def _precompute_profile_context(geometry, results):
    coords = geometry["pore.coords"]
    conns = geometry["throat.conns"]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    line_x = []
    line_y = []
    line_z = []
    diameter_line_color = []
    diameters = _diameter_values(geometry, len(conns))
    for i, j in conns:
        line_x.extend([float(x[i]), float(x[j]), None])
        line_y.extend([float(y[i]), float(y[j]), None])
        line_z.extend([float(z[i]), float(z[j]), None])
    if diameters is not None:
        for diameter in diameters:
            diameter_line_color.extend([float(diameter), float(diameter), np.nan])

    water_depths = results["water_depths"]
    diameter_range = _finite_range(diameters) if diameters is not None else None
    return {
        "x": x,
        "y": y,
        "z": z,
        "conns": conns,
        "line_x": line_x,
        "line_y": line_y,
        "line_z": line_z,
        "x_range": [float(np.min(x)), float(np.max(x))],
        "y_range": [float(np.min(y)), float(np.max(y))],
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "depth_min": float(np.min(water_depths)),
        "depth_max": float(np.max(water_depths)),
        "fields": _field_specs(results, geometry),
        "diameter_line_color": diameter_line_color,
        "diameter_range": diameter_range,
        "diameter_segments": _build_diameter_segments(x, y, z, conns, diameters),
    }


def _precompute_observation_context(obs_df):
    if (
        obs_df is None
        or obs_df.empty
        or "node" not in obs_df.columns
        or "time" not in obs_df.columns
    ):
        return None

    property_specs = _observation_property_specs(obs_df)
    if not property_specs:
        return None

    sorted_df = obs_df.sort_values(["node", "time"])
    by_node = {
        int(node): node_df.reset_index(drop=True)
        for node, node_df in sorted_df.groupby("node", sort=True)
    }
    context = {
        "by_node": by_node,
        "property_specs": property_specs,
        "default_property": _default_observation_property(property_specs, obs_df),
        "x_range": _time_axis_range(np.asarray(sorted_df["time"], dtype=float)),
    }
    return context


def _scene_aspectratio(profile_context, z_max_total):
    x_min, x_max = profile_context["x_range"]
    y_min, y_max = profile_context["y_range"]
    z_min = profile_context["z_min"]
    x_span = max(1e-9, float(x_max - x_min))
    y_span = max(1e-9, float(y_max - y_min))
    z_span = max(1e-9, float(z_max_total - z_min))
    max_span = max(x_span, y_span, z_span)
    return {
        "x": max(0.05, x_span / max_span),
        "y": max(0.08, y_span / max_span),
        "z": max(0.18, z_span / max_span),
    }


def _default_3d_camera(default_camera):
    return deepcopy(default_camera or dict(eye=dict(x=1.4, y=1.4, z=1.4)))


def _normalize_camera(camera, default_camera):
    if isinstance(camera, dict):
        if any(key in camera for key in ("eye", "up", "center", "projection")):
            return deepcopy(camera)
        if isinstance(camera.get("3d"), dict):
            return deepcopy(camera["3d"])
    return _default_3d_camera(default_camera)


def _camera_for_view(view_mode, default_camera):
    if view_mode in VIEW_CAMERAS:
        return deepcopy(VIEW_CAMERAS[view_mode])
    return _default_3d_camera(default_camera)


def _set_nested(target, path, value):
    cursor = target
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


def _camera_from_relayout(existing_camera, relayout_data):
    if not relayout_data:
        return None
    camera = deepcopy(existing_camera) if isinstance(existing_camera, dict) else {}
    if isinstance(relayout_data.get("scene.camera"), dict):
        for key, value in relayout_data["scene.camera"].items():
            if isinstance(value, dict) and isinstance(camera.get(key), dict):
                camera[key].update(deepcopy(value))
            else:
                camera[key] = deepcopy(value)
        return camera

    changed = False
    for key, value in relayout_data.items():
        if key.startswith("scene.camera."):
            _set_nested(camera, key.split(".")[2:], value)
            changed = True
    return camera if changed else None


def _build_profile_figure(
    results,
    profile_context,
    time_idx,
    node_ids,
    node_step,
    depth_scale,
    color_field,
    use_log_color_scale,
    show_vertical_bars,
    diameter_aware,
    color_edges_by_diameter,
    diameter_scale,
    camera,
    obs_node_colors,
):
    x = profile_context["x"]
    y = profile_context["y"]
    z = profile_context["z"]
    t = results["time"]
    water_depths = results["water_depths"]

    time_idx = int(np.clip(time_idx, 0, len(t) - 1))
    depth_scale = _number_value(depth_scale, default=1.0, minimum=0.0)
    diameter_scale = _number_value(diameter_scale, default=1.0, minimum=0.0)
    field_specs = profile_context["fields"]
    field_spec = field_specs.get(color_field, field_specs["depth"])
    field_values = field_spec["array"][time_idx, :]
    log_range = field_spec.get("log_range")
    use_log_color_scale = bool(use_log_color_scale and log_range)
    wd = water_depths[time_idx, :]
    visual_water_surface_z = z + wd * depth_scale

    n_nodes = len(x)
    selected = {int(node) for node in (node_ids or []) if 0 <= int(node) < n_nodes}
    step = _int_value(node_step, default=1, minimum=1)
    base_idx = set(range(0, n_nodes, step))
    idx = np.asarray(sorted(base_idx | selected), dtype=int)

    fig = go.Figure()
    has_diameter_data = bool(profile_context["diameter_segments"])
    show_diameter_segments = has_diameter_data and diameter_aware
    show_diameter_colorbar = bool(color_edges_by_diameter and profile_context["diameter_range"])
    show_two_colorbars = bool(show_vertical_bars and show_diameter_colorbar)
    bar_label = field_spec["label"]
    bar_cmin, bar_cmax = field_spec["range"]
    if use_log_color_scale:
        bar_label = f"{field_spec['label']} (log10)"
        bar_cmin = float(np.log10(log_range[0]))
        bar_cmax = float(np.log10(log_range[1]))

    bar_colorbar = _compact_colorbar(
        bar_label,
        0.32 if show_two_colorbars else 0.5,
        0.36 if show_two_colorbars else 0.58,
    )
    if use_log_color_scale:
        bar_colorbar.update(_log_colorbar_ticks(log_range))

    diameter_colorbar = _compact_colorbar(
        "Diameter [m]",
        0.72 if show_two_colorbars else 0.5,
        0.36 if show_two_colorbars else 0.58,
    )
    if show_diameter_segments:
        for segment in profile_context["diameter_segments"]:
            line_width = segment["width"] if diameter_aware else 2.0
            line_color = segment["color"] if color_edges_by_diameter else "#9ca3af"
            fig.add_trace(
                go.Scatter3d(
                    x=segment["x"],
                    y=segment["y"],
                    z=segment["z"],
                    mode="lines",
                    line=dict(
                        color=line_color,
                        width=max(0.5, line_width * diameter_scale),
                    ),
                    hoverinfo="none",
                    showlegend=False,
                )
            )
        if color_edges_by_diameter:
            dmin, dmax = profile_context["diameter_range"]
            fig.add_trace(
                go.Scatter3d(
                    x=[float(x[0]), float(x[0])],
                    y=[float(y[0]), float(y[0])],
                    z=[float(z[0]), float(z[0])],
                    mode="markers",
                    marker=dict(
                        size=0.01,
                        opacity=0.0,
                        color=[dmin, dmax],
                        colorscale=DIAMETER_COLOR_SCALE,
                        cmin=dmin,
                        cmax=dmax,
                        colorbar=diameter_colorbar,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    else:
        conduit_line = dict(color="#9ca3af", width=max(0.5, 2.0 * diameter_scale))
        if color_edges_by_diameter and profile_context["diameter_line_color"]:
            dmin, dmax = profile_context["diameter_range"]
            conduit_line = dict(
                color=profile_context["diameter_line_color"],
                colorscale=DIAMETER_COLOR_SCALE,
                cmin=dmin,
                cmax=dmax,
                width=max(0.5, 2.0 * diameter_scale),
                colorbar=diameter_colorbar,
            )
        fig.add_trace(
            go.Scatter3d(
                x=profile_context["line_x"],
                y=profile_context["line_y"],
                z=profile_context["line_z"],
                mode="lines",
                line=conduit_line,
                hoverinfo="none",
                showlegend=False,
            )
        )

    xi = x[idx]
    yi = y[idx]
    zi = z[idx]
    wdi = wd[idx]
    wsi = visual_water_surface_z[idx]
    field_i = field_values[idx]

    line_x = np.empty(3 * len(idx), dtype=float)
    line_y = np.empty(3 * len(idx), dtype=float)
    line_z = np.empty(3 * len(idx), dtype=float)
    color_vals = np.empty(3 * len(idx), dtype=float)

    line_x[::3] = xi
    line_x[1::3] = xi
    line_x[2::3] = np.nan
    line_y[::3] = yi
    line_y[1::3] = yi
    line_y[2::3] = np.nan
    line_z[::3] = zi
    line_z[1::3] = wsi
    line_z[2::3] = np.nan
    color_vals[::3] = field_i
    color_vals[1::3] = field_i
    color_vals[2::3] = np.nan
    bar_color_vals = (
        _log10_color_values(color_vals, log_range[0])
        if use_log_color_scale
        else color_vals
    )

    if show_vertical_bars:
        fig.add_trace(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line=dict(
                    color=bar_color_vals,
                    colorscale=field_spec["colorscale"],
                    cmin=bar_cmin,
                    cmax=bar_cmax,
                    width=4,
                    colorbar=bar_colorbar,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for node_id in sorted(selected):
        physical_surface_z = z[node_id] + wd[node_id]
        fig.add_trace(
            go.Scatter3d(
                x=[float(x[node_id])],
                y=[float(y[node_id])],
                z=[float(visual_water_surface_z[node_id])],
                mode="markers",
                marker=dict(
                    size=6,
                    color=obs_node_colors.get(node_id, "red"),
                    symbol="circle",
                    opacity=0.9,
                ),
                customdata=[[
                    float(wd[node_id]),
                    float(physical_surface_z),
                    float(field_values[node_id]),
                ]],
                hovertemplate=(
                    f"Node {node_id}<br>"
                    "Depth: %{customdata[0]:.3f} m<br>"
                    "Physical surface z: %{customdata[1]:.3f} m<br>"
                    f"{field_spec['label']}: "
                    "%{customdata[2]:.3g}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    max_z_total = float(profile_context["z_max"] + profile_context["depth_max"] * depth_scale)
    z_range = [float(profile_context["z_min"]), max_z_total]
    scene_camera = _normalize_camera(
        camera,
        dict(eye=dict(x=1.4, y=1.4, z=1.4)),
    )
    fig.update_layout(
        autosize=True,
        margin=dict(
            l=10,
            r=10,
            b=58 if (show_vertical_bars or show_diameter_colorbar) else 10,
            t=62,
        ),
        uirevision="openkarst-session-3d",
        scene=dict(
            xaxis=dict(title="X [m]", range=profile_context["x_range"], autorange=False, nticks=6),
            yaxis=dict(title="Y [m]", range=profile_context["y_range"], autorange=False, nticks=6),
            zaxis=dict(title="Elevation + scaled depth", range=z_range, autorange=False, nticks=5),
            aspectmode="manual",
            aspectratio=_scene_aspectratio(profile_context, max_z_total),
            camera=scene_camera,
            uirevision="openkarst-session-3d",
        ),
        title=dict(
            text=(
                "Network state"
                "<br>"
                f"t = {t[time_idx]:.1f} s | depth scale {depth_scale:g}x"
                "<br>"
                f"{'bars: ' + bar_label if show_vertical_bars else 'vertical bars hidden'}"
            ),
            x=0.01,
            xanchor="left",
            font=dict(size=12),
        ),
    )

    return fig


def _build_observation_figure(
    results,
    obs_df,
    obs_context,
    time_idx,
    node_ids,
    obs_node_colors,
    observation_property,
):
    if obs_df is None or obs_df.empty or not node_ids:
        return _empty_observation_figure()
    if obs_context is None:
        return _empty_observation_figure("No plottable observations")

    property_specs = obs_context["property_specs"]
    if observation_property not in property_specs:
        observation_property = obs_context["default_property"]
    if observation_property is None:
        return _empty_observation_figure("No plottable observations")

    current_time = float(results["time"][int(np.clip(time_idx, 0, len(results["time"]) - 1))])
    property_spec = property_specs[observation_property]
    fig = go.Figure()
    traces_added = 0

    for node_id in node_ids:
        node_id = int(node_id)
        df_full = obs_context["by_node"].get(node_id)
        if df_full is None or df_full.empty or observation_property not in df_full.columns:
            continue

        end_idx = int(np.searchsorted(df_full["time"].to_numpy(), current_time, side="right"))
        df_visible = _thin_frame(df_full.iloc[:end_idx])
        if df_visible.empty:
            continue

        trace_mode = "lines" if len(df_visible) > 500 else "lines+markers"

        fig.add_trace(go.Scattergl(
            x=df_visible["time"],
            y=np.asarray(df_visible[observation_property], dtype=float),
            mode=trace_mode,
            name=f"{property_spec['label']} - node {node_id}",
            line=dict(color=obs_node_colors.get(node_id, "blue"), width=2),
            marker=dict(size=5),
            legendgroup=f"node-{node_id}",
            hovertemplate=(
                "Time: %{x:g} s<br>"
                f"Node {node_id}<br>"
                f"{property_spec['label']}: %{{y:.3g}}"
                "<extra></extra>"
            ),
        ))
        traces_added += 1

    if traces_added == 0:
        return _empty_observation_figure("No selected observations")

    x_min, x_max = obs_context["x_range"]
    y_min, y_max = property_spec["range"]
    y_pad = max(1e-12, 0.08 * (y_max - y_min))

    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="#dbe1ea",
            borderwidth=1,
            font=dict(size=9),
        ),
        xaxis=dict(title="Time [s]", range=[x_min, x_max], fixedrange=True),
        yaxis=dict(
            title=property_spec["axis_label"],
            range=[y_min - y_pad, y_max + y_pad],
            fixedrange=True,
        ),
    )
    fig.add_vline(x=current_time, line_width=1, line_dash="dot", line_color="#374151")

    return fig


def create_openkarst_viewer_app(
    results,
    geometry,
    obs_df=None,
    *,
    depth_scale=DEFAULT_DEPTH_SCALE,
):
    """Build and return the openKARST Dash viewer app.

    This function intentionally does not start a web server or open a browser.
    Keeping app creation separate from display logic makes the viewer reusable
    in scripts, notebooks, Google Colab, and tests.
    """
    t = results["time"]
    profile_context = _precompute_profile_context(geometry, results)
    obs_context = _precompute_observation_context(obs_df)
    obs_nodes = _observation_nodes(obs_df)
    selected_nodes = obs_nodes[: min(10, len(obs_nodes))]
    observation_property_specs = obs_context["property_specs"] if obs_context else {}
    observation_property_options = [
        {"label": spec["label"], "value": column}
        for column, spec in observation_property_specs.items()
    ]
    default_observation_property = (
        obs_context["default_property"] if obs_context else None
    )
    field_options = [
        {"label": spec["label"], "value": field_id}
        for field_id, spec in profile_context["fields"].items()
    ]
    has_diameters = bool(profile_context["diameter_segments"])

    obs_node_colors = {
        node_id: COLOR_CYCLE[i % len(COLOR_CYCLE)]
        for i, node_id in enumerate(obs_nodes)
    }
    default_camera = _camera_for_view("3d", dict(eye=dict(x=1.4, y=1.4, z=1.4)))

    assets_dir = _viewer_assets_dir()
    dash_kwargs = {"assets_folder": str(assets_dir)} if assets_dir is not None else {}
    app = dash.Dash(__name__, **dash_kwargs)
    app.title = "openKARST 3D Viewer"
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                html, body, #react-entry-point {
                    height: 100%;
                    margin: 0;
                }
                * {
                    box-sizing: border-box;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """

    page_style = {
        "height": "100vh",
        "margin": "0",
        "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "background": "#f7f8fb",
        "color": "#111827",
        "overflow": "hidden",
    }
    shell_style = {
        "display": "grid",
        "gridTemplateColumns": "clamp(165px, 15.5vw, 200px) minmax(0, 1fr)",
        "height": "100vh",
    }
    sidebar_style = {
        "background": "#ffffff",
        "borderRight": "1px solid #dbe1ea",
        "padding": "8px",
        "overflowY": "auto",
    }
    title_style = {
        "fontSize": "19px",
        "fontWeight": "650",
        "letterSpacing": "0",
        "margin": "0 0 5px",
    }
    header_logo_style = {
        "display": "block",
        "width": "100%",
        "maxWidth": "130px",
        "height": "auto",
        "margin": "0 0 5px",
    }
    section_style = {
        "border": "1px solid #dbe1ea",
        "borderRadius": "6px",
        "padding": "5px 6px",
        "margin": "0 0 5px",
        "background": "#fbfcfe",
    }
    legend_style = {
        "fontSize": "10px",
        "fontWeight": "700",
        "padding": "0 4px",
        "color": "#374151",
    }
    label_style = {
        "display": "block",
        "fontSize": "10px",
        "fontWeight": "650",
        "color": "#4b5563",
        "marginBottom": "3px",
    }
    control_style = {
        "marginBottom": "5px",
    }
    input_style = {
        "width": "100%",
        "height": "28px",
        "boxSizing": "border-box",
        "border": "1px solid #cfd6e1",
        "borderRadius": "6px",
        "padding": "4px 8px",
        "fontSize": "12px",
        "background": "#ffffff",
    }
    stepper_style = {
        "display": "grid",
        "gridTemplateColumns": "26px minmax(0, 1fr) 26px",
        "gap": "4px",
        "alignItems": "center",
    }
    stepper_button_style = {
        "height": "28px",
        "border": "1px solid #cfd6e1",
        "borderRadius": "6px",
        "background": "#eef2f7",
        "color": "#111827",
        "fontSize": "16px",
        "fontWeight": "700",
        "cursor": "pointer",
    }
    stepper_input_style = {
        **input_style,
        "textAlign": "center",
        "padding": "4px",
    }
    button_style = {
        "width": "100%",
        "height": "30px",
        "border": "1px solid #1f6feb",
        "borderRadius": "6px",
        "background": "#1f6feb",
        "color": "#ffffff",
        "fontWeight": "650",
        "cursor": "pointer",
    }
    checklist_style = {
        "fontSize": "12px",
        "lineHeight": "1.2",
    }
    main_style = {
        "display": "grid",
        "gridTemplateRows": "82px minmax(0, 1fr)",
        "minWidth": "0",
        "height": "100vh",
        "overflow": "hidden",
    }
    timebar_style = {
        "background": "#ffffff",
        "borderBottom": "1px solid #dbe1ea",
        "padding": "14px 22px 8px",
    }
    plot_area_style = {
        "display": "grid",
        "gridTemplateColumns": "minmax(0, 1.7fr) minmax(320px, 0.9fr)",
        "gridTemplateRows": "minmax(0, 1fr) minmax(220px, 0.72fr)",
        "gap": "10px",
        "minHeight": "0",
        "padding": "10px",
        "alignItems": "stretch",
    }
    graph_style = {
        "width": "100%",
        "height": "100%",
        "minHeight": "0",
        "background": "#ffffff",
        "border": "1px solid #dbe1ea",
        "borderRadius": "8px",
        "overflow": "hidden",
    }

    def control(label, component):
        return html.Div([
            html.Label(label, style=label_style),
            component,
        ], style=control_style)

    def stepper(component_id, value, integer=False):
        return html.Div([
            html.Button(
                "-",
                id=f"{component_id}-minus",
                n_clicks=0,
                style=stepper_button_style,
            ),
            dcc.Input(
                id=component_id,
                type="text",
                value=_format_number(value, integer=integer),
                style=stepper_input_style,
            ),
            html.Button(
                "+",
                id=f"{component_id}-plus",
                n_clicks=0,
                style=stepper_button_style,
            ),
        ], style=stepper_style)

    def section(title, children):
        return html.Fieldset([
            html.Legend(title, style=legend_style),
            *children,
        ], style=section_style)

    def log_scale_container_style(field_id):
        field_spec = profile_context["fields"].get(field_id)
        if field_spec and field_spec.get("log_range"):
            return {"display": "block", "margin": "-3px 0 3px"}
        return {"display": "none"}

    header_component = (
        html.Img(
            src=app.get_asset_url(HEADER_LOGO),
            alt="openKARST",
            style=header_logo_style,
        )
        if assets_dir is not None
        else html.H2("openKARST Viewer", style=title_style)
    )

    app.layout = html.Div([
        html.Div([
            html.Aside([
                header_component,

                section("Playback", [
                    control("Stride", stepper("stride-input", DEFAULT_PLAY_STRIDE, integer=True)),
                    html.Button("Play", id="play-button", n_clicks=0, style=button_style),
                ]),

                section("View", [
                    control("Camera view", dcc.Dropdown(
                        id="view-mode-selector",
                        options=[
                            {"label": "3D network", "value": "3d"},
                            {"label": "Map view (x-y)", "value": "xy"},
                            {"label": "Longitudinal profile (x-z)", "value": "xz"},
                            {"label": "Cross-section (y-z)", "value": "yz"},
                        ],
                        value="3d",
                        clearable=False,
                        style={"fontSize": "12px"},
                    )),
                ]),

                section("Water Bars", [
                    control("Color by", dcc.Dropdown(
                        id="color-field-selector",
                        options=field_options,
                        value=field_options[0]["value"],
                        clearable=False,
                        style={"fontSize": "12px"},
                    )),
                    html.Div(
                        dcc.Checklist(
                            id="bar-log-scale-toggle",
                            options=[{"label": "Log color scale", "value": "log"}],
                            value=[],
                            style=checklist_style,
                        ),
                        id="bar-log-scale-container",
                        style=log_scale_container_style(field_options[0]["value"]),
                    ),
                    dcc.Checklist(
                        id="vertical-bars-toggle",
                        options=[{
                            "label": "Vertical bars",
                            "value": "bars",
                        }],
                        value=["bars"],
                        style=checklist_style,
                    ),
                    control("Node step", stepper("node-step-input", 1, integer=True)),
                    control("Depth scale", stepper("depth-scale-input", depth_scale)),
                ]),

                section("Conduits", [
                    dcc.Checklist(
                        id="diameter-aware-toggle",
                        options=[{
                            "label": "Width by diameter",
                            "value": "diameter",
                            "disabled": not has_diameters,
                        }],
                        value=[],
                        style=checklist_style,
                    ),
                    dcc.Checklist(
                        id="edge-color-toggle",
                        options=[{
                            "label": "Color by diameter",
                            "value": "diameter-color",
                            "disabled": not has_diameters,
                        }],
                        value=[],
                        style={**checklist_style, "marginTop": "4px"},
                    ),
                    html.Div(style={"height": "2px"}),
                    control("Diameter scale", stepper("diameter-scale-input", 1)),
                ]),

                section("Observations", [
                    control("Property", dcc.Dropdown(
                        id="observation-property-selector",
                        options=observation_property_options,
                        value=default_observation_property,
                        clearable=False,
                        disabled=not bool(observation_property_options),
                        placeholder="None",
                        style={"fontSize": "12px"},
                    )),
                    control("Observation nodes", dcc.Dropdown(
                        id="node-selector",
                        options=[{"label": str(n), "value": n} for n in obs_nodes],
                        value=selected_nodes,
                        multi=True,
                        disabled=not bool(obs_nodes),
                        placeholder="None",
                        style={"fontSize": "12px"},
                    )),
                ]),
            ], style=sidebar_style),

            html.Main([
                html.Div([
                    html.Label("Time", style={
                        "display": "block",
                        "fontSize": "12px",
                        "fontWeight": "700",
                        "color": "#4b5563",
                        "marginBottom": "6px",
                    }),
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=len(t) - 1,
                        step=1,
                        value=0,
                        marks={
                            0: "0s",
                            len(t) // 2: f"{int(t[len(t) // 2])}s",
                            len(t) - 1: f"{int(t[-1])}s",
                        },
                        updatemode="drag",
                    ),
                ], style=timebar_style),

                html.Div([
                    dcc.Graph(
                        id="3d-profile",
                        style={
                            **graph_style,
                            "gridColumn": "1",
                            "gridRow": "1 / span 2",
                        },
                    ),
                    dcc.Graph(
                        id="obs-plot",
                        style={
                            **graph_style,
                            "gridColumn": "2",
                            "gridRow": "1",
                        },
                    ),
                    dcc.Graph(
                        id="convergence-plot",
                        style={
                            **graph_style,
                            "gridColumn": "2",
                            "gridRow": "2",
                        },
                    ),
                ], style=plot_area_style),
            ], style=main_style),
        ], style=shell_style),

        dcc.Interval(id="interval", interval=120, n_intervals=0, disabled=True),
        dcc.Store(id="stored-camera", data=default_camera),
    ], style=page_style)

    def register_stepper(component_id, default, minimum, step, integer=False):
        @app.callback(
            Output(component_id, "value"),
            Input(f"{component_id}-minus", "n_clicks"),
            Input(f"{component_id}-plus", "n_clicks"),
            State(component_id, "value"),
            prevent_initial_call=True,
        )
        def update_stepper(_minus_clicks, _plus_clicks, current_value):
            triggered = dash.callback_context.triggered
            if not triggered:
                return _format_number(default, integer=integer)

            trigger_id = triggered[0]["prop_id"].split(".")[0]
            value = _number_value(current_value, default=default, minimum=minimum)
            if trigger_id.endswith("-minus"):
                value -= step
            else:
                value += step
            value = max(minimum, value)
            return _format_number(value, integer=integer)

    register_stepper("stride-input", DEFAULT_PLAY_STRIDE, 1, 1, integer=True)
    register_stepper("node-step-input", 1, 1, 1, integer=True)
    register_stepper("depth-scale-input", depth_scale, 0, 1)
    register_stepper("diameter-scale-input", 1, 0, 0.25)

    @app.callback(
        Output("interval", "disabled"),
        Input("play-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks):
        return n_clicks % 2 == 0

    @app.callback(
        Output("time-slider", "value"),
        Input("interval", "n_intervals"),
        State("time-slider", "value"),
        State("stride-input", "value"),
    )
    def advance_slider(_n_intervals, current_idx, stride):
        stride = _int_value(stride, default=1, minimum=1)
        return 0 if (current_idx + stride) >= len(t) else current_idx + stride

    @app.callback(
        Output("bar-log-scale-container", "style"),
        Output("bar-log-scale-toggle", "value"),
        Input("color-field-selector", "value"),
        State("bar-log-scale-toggle", "value"),
    )
    def update_log_scale_visibility(color_field, current_value):
        field_spec = profile_context["fields"].get(color_field)
        if field_spec and field_spec.get("log_range"):
            return log_scale_container_style(color_field), current_value or []
        return log_scale_container_style(color_field), []

    @app.callback(
        Output("3d-profile", "figure"),
        Output("stored-camera", "data"),
        Output("3d-profile", "relayoutData"),
        Input("time-slider", "value"),
        Input("view-mode-selector", "value"),
        Input("node-selector", "value"),
        Input("node-step-input", "value"),
        Input("depth-scale-input", "value"),
        Input("color-field-selector", "value"),
        Input("bar-log-scale-toggle", "value"),
        Input("vertical-bars-toggle", "value"),
        Input("diameter-aware-toggle", "value"),
        Input("edge-color-toggle", "value"),
        Input("diameter-scale-input", "value"),
        State("3d-profile", "relayoutData"),
        State("stored-camera", "data"),
    )
    def update_3d_plot(
        time_idx,
        view_mode,
        node_ids,
        node_step,
        current_depth_scale,
        color_field,
        bar_log_scale_toggle,
        vertical_bars_toggle,
        diameter_toggle,
        edge_color_toggle,
        current_diameter_scale,
        relayout_data,
        camera,
    ):
        use_log_color_scale = "log" in (bar_log_scale_toggle or [])
        show_vertical_bars = "bars" in (vertical_bars_toggle or [])
        diameter_aware = "diameter" in (diameter_toggle or [])
        color_edges_by_diameter = "diameter-color" in (edge_color_toggle or [])
        triggered = dash.callback_context.triggered
        trigger_id = triggered[0]["prop_id"].split(".")[0] if triggered else None

        if trigger_id == "view-mode-selector":
            current_camera = _camera_for_view(view_mode, default_camera)
            next_relayout_data = None
        else:
            current_camera = _normalize_camera(camera, default_camera)
            relayout_camera = _camera_from_relayout(current_camera, relayout_data)
            if relayout_camera is not None:
                current_camera = relayout_camera
            next_relayout_data = dash.no_update

        figure = _build_profile_figure(
            results,
            profile_context,
            time_idx,
            node_ids,
            node_step,
            current_depth_scale,
            color_field,
            use_log_color_scale,
            show_vertical_bars,
            diameter_aware,
            color_edges_by_diameter,
            current_diameter_scale,
            current_camera,
            obs_node_colors,
        )
        return figure, current_camera, next_relayout_data

    @app.callback(
        Output("obs-plot", "figure"),
        Input("time-slider", "value"),
        Input("node-selector", "value"),
        Input("observation-property-selector", "value"),
    )
    def update_obs_plot_callback(time_idx, node_ids, observation_property):
        return _build_observation_figure(
            results,
            obs_df,
            obs_context,
            time_idx,
            node_ids,
            obs_node_colors,
            observation_property,
        )

    @app.callback(
        Output("convergence-plot", "figure"),
        Input("time-slider", "value"),
    )
    def update_convergence_plot_callback(time_idx):
        return _build_convergence_figure(results, time_idx)

    return app



def _server_url(host, port):
    """Return a browser-friendly URL for a Dash server."""
    browser_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{browser_host}:{port}/"


def _run_dash_server(app, host, port):
    """Run a Dash server with options that work in background threads."""
    run = app.run if hasattr(app, "run") else app.run_server
    run(host=host, port=port, debug=False, use_reloader=False)


def _show_colab_iframe(port, iframe_height):
    """Display a running Dash app through Google Colab's port proxy."""
    from google.colab import output
    output.serve_kernel_port_as_iframe(
        port,
        path="/",
        height=iframe_height,
    )


def _print_colab_proxy_url(port):
    """Print a fallback proxied URL for Google Colab, when available."""
    from google.colab import output
    proxy_url = output.eval_js(f"google.colab.kernel.proxyPort({port})")
    print(f"Open the openKARST viewer through Colab here: {proxy_url}")


def launch_openkarst_viewer(
    results,
    geometry,
    obs_df=None,
    *,
    depth_scale=DEFAULT_DEPTH_SCALE,
    host="127.0.0.1",
    port=8050,
    open_browser=True,
    mode="auto",
    iframe_height=700,
):
    """Create, launch, and display the openKARST Dash viewer.

    Parameters
    ----------
    results, geometry, obs_df
        Simulation output, network geometry, and optional observation dataframe.
    depth_scale : float, default DEFAULT_DEPTH_SCALE
        Visual scaling factor for water-depth bars.
    host : str, default "127.0.0.1"
        Host used by the Dash server. In Colab, ``mode="auto"`` switches this
        to ``"0.0.0.0"`` so Colab's port proxy can reach the server.
    port : int, default 8050
        Port used by the Dash server.
    open_browser : bool, default True
        Backward-compatible local behavior. Ignored in Colab auto mode, where
        the viewer is shown in an iframe instead.
    mode : {"auto", "colab", "browser", "none"}, default "auto"
        Display mode. ``"auto"`` uses a Colab iframe inside Google Colab and a
        normal browser locally. ``"none"`` starts the server and only prints
        the URL, which is useful for advanced/custom embedding.
    iframe_height : int, default 700
        Height of the Colab iframe.

    Returns
    -------
    dash.Dash
        The Dash app instance. This keeps the old convenience API while still
        allowing advanced users to inspect or reuse the app.
    """
    valid_modes = {"auto", "colab", "browser", "none"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")

    in_colab = _in_google_colab()
    if mode == "auto":
        if in_colab:
            mode = "colab"
        elif open_browser:
            mode = "browser"
        else:
            mode = "none"

    if mode == "colab" and host == "127.0.0.1":
        # Colab's port proxy can reach the server reliably when it listens on
        # all interfaces inside the runtime container.
        host = "0.0.0.0"

    app = create_openkarst_viewer_app(
        results,
        geometry,
        obs_df,
        depth_scale=depth_scale,
    )

    thread = threading.Thread(
        target=_run_dash_server,
        args=(app, host, port),
        daemon=True,
    )
    thread.start()
    time.sleep(1.0)

    if mode == "colab":
        try:
            _show_colab_iframe(port, iframe_height)
        except Exception as exc:
            print("Could not show the openKARST viewer as a Colab iframe.")
            print(f"Reason: {exc}")
            try:
                _print_colab_proxy_url(port)
            except Exception:
                print(f"Viewer is running at {_server_url(host, port)}")
    elif mode == "browser":
        webbrowser.open(_server_url(host, port))
    else:
        print(f"Viewer is running at {_server_url(host, port)}")

    return app
