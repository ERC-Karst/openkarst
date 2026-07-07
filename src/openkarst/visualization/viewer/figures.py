"""Plotly figure builders for the openKARST viewer."""

import numpy as np
import plotly.graph_objects as go

from .camera import _normalize_camera
from .constants import (
    DEFAULT_OBS_COMPACT_LEGEND_MAX_TRACES,
    DEFAULT_OBS_INLINE_LEGEND_MAX_TRACES,
    DIAMETER_COLOR_SCALE,
)
from .data import (
    _int_value,
    _log10_color_values,
    _log_colorbar_ticks,
    _number_value,
    _thin_frame,
    _time_axis_range,
)


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
    water_depths = profile_context["water_depths"]

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
        df_property = df_full.iloc[:end_idx]
        if df_property.empty:
            continue
        y_values = np.asarray(df_property[observation_property], dtype=float)
        finite_values = np.isfinite(y_values)
        if not np.any(finite_values):
            continue
        df_visible = _thin_frame(df_property.loc[finite_values])
        y_values = np.asarray(df_visible[observation_property], dtype=float)

        trace_mode = "lines" if len(df_visible) > 500 else "lines+markers"

        fig.add_trace(go.Scattergl(
            x=df_visible["time"],
            y=y_values,
            mode=trace_mode,
            name=f"n {node_id}",
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
    show_legend = traces_added <= DEFAULT_OBS_COMPACT_LEGEND_MAX_TRACES

    if traces_added <= DEFAULT_OBS_INLINE_LEGEND_MAX_TRACES:
        legend = dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.62)",
            bordercolor="rgba(219,225,234,0.55)",
            borderwidth=1,
            font=dict(size=8),
            itemsizing="constant",
        )
        margin = dict(l=40, r=35, t=34, b=40)
    else:
        legend = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=8),
            itemsizing="constant",
        )
        margin = dict(l=40, r=35, t=54 if show_legend else 34, b=40)

    fig.update_layout(
        margin=margin,
        showlegend=show_legend,
        legend=legend,
        xaxis=dict(title="Time [s]", range=[x_min, x_max], fixedrange=True),
        yaxis=dict(
            title=property_spec["axis_label"],
            range=[y_min - y_pad, y_max + y_pad],
            fixedrange=True,
        ),
    )
    fig.add_vline(x=current_time, line_width=1, line_dash="dot", line_color="#374151")

    return fig
