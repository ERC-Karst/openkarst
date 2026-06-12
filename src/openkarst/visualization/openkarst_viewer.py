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

import dash
from dash import Input, Output, State, dcc, html
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLOR_CYCLE = pc.qualitative.Plotly
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_PLAY_STRIDE = 1
DEFAULT_OBS_RENDER_POINTS = 1200
FLOW_COLUMN_CANDIDATES = ("inflow", "Q", "flowrate", "flowrates")


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
    for i, j in conns:
        line_x.extend([float(x[i]), float(x[j]), None])
        line_y.extend([float(y[i]), float(y[j]), None])
        line_z.extend([float(z[i]), float(z[j]), None])

    water_depths = results["water_depths"]
    return {
        "x": x,
        "y": y,
        "z": z,
        "line_x": line_x,
        "line_y": line_y,
        "line_z": line_z,
        "x_range": [float(np.min(x)), float(np.max(x))],
        "y_range": [float(np.min(y)), float(np.max(y))],
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "depth_min": float(np.min(water_depths)),
        "depth_max": float(np.max(water_depths)),
    }


def _precompute_observation_context(obs_df):
    flow_column = _flow_column(obs_df)
    if obs_df is None or obs_df.empty or flow_column is None:
        return None

    sorted_df = obs_df.sort_values(["node", "time"])
    by_node = {
        int(node): node_df.reset_index(drop=True)
        for node, node_df in sorted_df.groupby("node", sort=True)
    }
    has_concentration = "concentrations" in sorted_df.columns
    context = {
        "by_node": by_node,
        "flow_column": flow_column,
        "has_concentration": has_concentration,
        "x_range": [float(sorted_df["time"].min()), float(sorted_df["time"].max())],
        "flow_range": [float(sorted_df[flow_column].min()), float(sorted_df[flow_column].max())],
    }
    if has_concentration:
        context["concentration_range"] = [
            float(sorted_df["concentrations"].min()),
            float(sorted_df["concentrations"].max()),
        ]
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


def _build_profile_figure(
    results,
    profile_context,
    time_idx,
    node_ids,
    node_step,
    depth_scale,
    camera,
    obs_node_colors,
):
    x = profile_context["x"]
    y = profile_context["y"]
    z = profile_context["z"]
    t = results["time"]
    water_depths = results["water_depths"]

    time_idx = int(np.clip(time_idx, 0, len(t) - 1))
    depth_scale = max(0.0, float(depth_scale or 1.0))
    wd = water_depths[time_idx, :]
    visual_water_surface_z = z + wd * depth_scale

    n_nodes = len(x)
    selected = {int(node) for node in (node_ids or []) if 0 <= int(node) < n_nodes}
    step = max(1, int(node_step)) if node_step else 1
    base_idx = set(range(0, n_nodes, step))
    idx = np.asarray(sorted(base_idx | selected), dtype=int)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=profile_context["line_x"],
            y=profile_context["line_y"],
            z=profile_context["line_z"],
            mode="lines",
            line=dict(color="#9ca3af", width=2),
            hoverinfo="none",
            showlegend=False,
        )
    )

    xi = x[idx]
    yi = y[idx]
    zi = z[idx]
    wdi = wd[idx]
    wsi = visual_water_surface_z[idx]

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
    color_vals[::3] = wdi
    color_vals[1::3] = wdi
    color_vals[2::3] = np.nan

    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line=dict(
                color=color_vals,
                colorscale="Viridis",
                cmin=profile_context["depth_min"],
                cmax=profile_context["depth_max"],
                width=4,
                colorbar=dict(
                    title="Depth [m]",
                    xanchor="left",
                    title_font=dict(size=10),
                    tickfont=dict(size=8),
                ),
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
                customdata=[[float(wd[node_id]), float(physical_surface_z)]],
                hovertemplate=(
                    f"Node {node_id}<br>"
                    "Depth: %{customdata[0]:.3f} m<br>"
                    "Physical surface z: %{customdata[1]:.3f} m<extra></extra>"
                ),
                showlegend=False,
            )
        )

    max_z_total = float(profile_context["z_max"] + profile_context["depth_max"] * depth_scale)
    z_range = [float(profile_context["z_min"]), max_z_total]
    fig.update_layout(
        autosize=True,
        margin=dict(l=10, r=10, b=10, t=40),
        uirevision="openkarst-session",
        scene=dict(
            xaxis=dict(title="X [m]", range=profile_context["x_range"], autorange=False, nticks=6),
            yaxis=dict(title="Y [m]", range=profile_context["y_range"], autorange=False, nticks=6),
            zaxis=dict(title="Elevation + scaled depth", range=z_range, autorange=False, nticks=5),
            aspectmode="manual",
            aspectratio=_scene_aspectratio(profile_context, max_z_total),
            camera=camera,
        ),
        title=f"Water depth at t = {t[time_idx]:.1f} s (visual scale {depth_scale:g}x)",
    )

    return fig


def _build_observation_figure(results, obs_df, obs_context, time_idx, node_ids, obs_node_colors):
    if obs_df is None or obs_df.empty or not node_ids:
        return _empty_observation_figure()
    if obs_context is None:
        return _empty_observation_figure("No flow observations")

    current_time = float(results["time"][int(np.clip(time_idx, 0, len(results["time"]) - 1))])
    flow_column = obs_context["flow_column"]
    has_concentration = bool(obs_context["has_concentration"])
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if has_concentration else go.Figure()

    for node_id in node_ids:
        node_id = int(node_id)
        df_full = obs_context["by_node"].get(node_id)
        if df_full is None or df_full.empty:
            continue

        end_idx = int(np.searchsorted(df_full["time"].to_numpy(), current_time, side="right"))
        df_visible = _thin_frame(df_full.iloc[:end_idx])
        trace_mode = "lines" if len(df_visible) > 500 else "lines+markers"

        flow_trace = go.Scattergl(
            x=df_visible["time"],
            y=df_visible[flow_column],
            mode=trace_mode,
            name=f"Q - node {node_id}",
            line=dict(color=obs_node_colors.get(node_id, "blue"), width=2),
            marker=dict(size=5),
            legendgroup=f"node-{node_id}",
        )
        if has_concentration:
            fig.add_trace(flow_trace, secondary_y=False)
        else:
            fig.add_trace(flow_trace)

        if has_concentration:
            fig.add_trace(
                go.Scattergl(
                    x=df_visible["time"],
                    y=df_visible["concentrations"],
                    mode="lines",
                    name=f"C - node {node_id}",
                    line=dict(color=obs_node_colors.get(node_id, "blue"), dash="dash", width=2),
                    legendgroup=f"node-{node_id}",
                ),
                secondary_y=True,
            )

    x_min, x_max = obs_context["x_range"]
    y1_min, y1_max = obs_context["flow_range"]
    y1_pad = max(1e-12, 0.08 * (y1_max - y1_min))

    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Time [s]", range=[x_min, x_max], fixedrange=True),
    )
    fig.add_vline(x=current_time, line_width=1, line_dash="dot", line_color="#374151")

    if has_concentration:
        y2_min, y2_max = obs_context["concentration_range"]
        y2_pad = max(1e-12, 0.08 * (y2_max - y2_min))
        fig.update_yaxes(
            title_text="Flow [m<sup>3</sup>/s]",
            range=[y1_min - y1_pad, y1_max + y1_pad],
            fixedrange=True,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Concentration [kg/m<sup>3</sup>]",
            range=[y2_min - y2_pad, y2_max + y2_pad],
            fixedrange=True,
            secondary_y=True,
        )
    else:
        fig.update_yaxes(
            title_text="Flow [m<sup>3</sup>/s]",
            range=[y1_min - y1_pad, y1_max + y1_pad],
            fixedrange=True,
        )

    return fig


def launch_openkarst_viewer(
    results,
    geometry,
    obs_df=None,
    *,
    depth_scale=DEFAULT_DEPTH_SCALE,
    host="127.0.0.1",
    port=8050,
    open_browser=True,
):
    t = results["time"]
    profile_context = _precompute_profile_context(geometry, results)
    obs_context = _precompute_observation_context(obs_df)
    obs_nodes = _observation_nodes(obs_df)
    selected_nodes = obs_nodes[: min(10, len(obs_nodes))]

    obs_node_colors = {
        node_id: COLOR_CYCLE[i % len(COLOR_CYCLE)]
        for i, node_id in enumerate(obs_nodes)
    }
    default_camera = dict(eye=dict(x=1.4, y=1.4, z=1.4))

    app = dash.Dash(__name__)
    app.title = "openKARST 3D Viewer"

    app.layout = html.Div([
        html.H2("openKARST 3D Viewer", style={"textAlign": "center"}),

        html.Div([
            html.Label("Stride:"),
            dcc.Input(
                id="stride-input",
                type="number",
                value=DEFAULT_PLAY_STRIDE,
                min=1,
                step=1,
                style={"width": "50px"},
            ),

            html.Button("Play", id="play-button", n_clicks=0, style={"marginLeft": "5px"}),

            html.Label("Node step:", style={"marginLeft": "20px"}),
            dcc.Input(
                id="node-step-input",
                type="number",
                value=1,
                min=1,
                step=1,
                style={"width": "60px"},
            ),

            html.Label("Depth scale:", style={"marginLeft": "20px"}),
            dcc.Input(
                id="depth-scale-input",
                type="number",
                value=depth_scale,
                min=0,
                step=1,
                style={"width": "70px"},
            ),

            html.Label("Observation node(s):", style={"marginLeft": "20px"}),
            dcc.Dropdown(
                id="node-selector",
                options=[{"label": str(n), "value": n} for n in obs_nodes],
                value=selected_nodes,
                multi=True,
                disabled=not bool(obs_nodes),
                placeholder="None",
                style={"minWidth": "220px", "maxWidth": "420px"},
            ),
        ], style={
            "display": "flex",
            "alignItems": "center",
            "gap": "10px",
            "margin": "10px",
            "flexWrap": "wrap",
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

        dcc.Interval(id="interval", interval=120, n_intervals=0, disabled=True),

        html.Div([
            dcc.Graph(id="3d-profile", style={
                "flex": "1",
                "minWidth": "300px",
                "width": "100%",
                "height": "100%",
            }),
            dcc.Graph(id="obs-plot", style={
                "flex": "1",
                "minWidth": "300px",
                "width": "100%",
                "height": "100%",
            }),
        ], style={
            "display": "flex",
            "flexDirection": "row",
            "flexWrap": "wrap",
            "height": "70vh",
            "gap": "10px",
        }),

        dcc.Store(id="stored-camera", data=default_camera),
    ])

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
        stride = max(1, int(stride)) if stride else 1
        return 0 if (current_idx + stride) >= len(t) else current_idx + stride

    @app.callback(
        Output("stored-camera", "data"),
        Input("3d-profile", "relayoutData"),
        State("stored-camera", "data"),
    )
    def update_camera_from_user(relayout_data, current_camera):
        if relayout_data and "scene.camera" in relayout_data:
            return relayout_data["scene.camera"]
        return current_camera

    @app.callback(
        Output("3d-profile", "figure"),
        Input("time-slider", "value"),
        Input("node-selector", "value"),
        Input("node-step-input", "value"),
        Input("depth-scale-input", "value"),
        State("stored-camera", "data"),
    )
    def update_3d_plot(time_idx, node_ids, node_step, current_depth_scale, camera):
        return _build_profile_figure(
            results,
            profile_context,
            time_idx,
            node_ids,
            node_step,
            current_depth_scale,
            camera,
            obs_node_colors,
        )

    @app.callback(
        Output("obs-plot", "figure"),
        Input("time-slider", "value"),
        Input("node-selector", "value"),
    )
    def update_obs_plot_callback(time_idx, node_ids):
        return _build_observation_figure(
            results,
            obs_df,
            obs_context,
            time_idx,
            node_ids,
            obs_node_colors,
        )

    def run_app():
        run = app.run if hasattr(app, "run") else app.run_server
        run(host=host, port=port, debug=False, use_reloader=False)

    threading.Thread(target=run_app).start()
    time.sleep(1)
    if open_browser:
        webbrowser.open(f"http://{host}:{port}/")
