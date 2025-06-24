#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:40:08 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:40:08 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""


import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import threading
import time
import webbrowser


def launch_openkarst_viewer(results, geometry, obs_df=None):
    coords = geometry['pore.coords']
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    t = results['time']
    water_depths = results['water_depths']

    max_z_total = float(np.max(z + water_depths))
    x_range = [float(np.min(x)), float(np.max(x))]
    y_range = [float(np.min(y)), float(np.max(y))]
    z_range = [float(np.min(z)), max_z_total]

    default_camera = dict(eye=dict(x=1.25, y=1.25, z=1.25))

    global_min = float(np.min(water_depths))
    global_max = float(np.max(water_depths))

    app = dash.Dash(__name__)
    app.title = "openKARST 3D Viewer"

    app.layout = html.Div([
        html.H2("openKARST 3D Viewer", style={'textAlign': 'center'}),

        html.Div([
            html.Label("Stride:"),
            dcc.Input(id="stride-input", type="number", value=1, min=1, step=1, style={'width': '80px'}),
            html.Button("Play", id="play-button", n_clicks=0),
        ], style={"margin": "10px"}),

        dcc.Slider(
            id='time-slider',
            min=0,
            max=len(t) - 1,
            step=1,
            value=0,
            marks={0: '0s', len(t)//2: f'{int(t[len(t)//2])}s', len(t)-1: f'{int(t[-1])}s'},
            updatemode='drag'  # or 'mouseup' for performance
        ),

        dcc.Interval(id="interval", interval=10, n_intervals=0, disabled=True),

        html.Div([
            html.Label("Observation Node:"),
            dcc.Dropdown(
                id='node-selector',
                options=[{'label': f'Node {n}', 'value': n} for n in sorted(obs_df['node'].unique())],
                value=sorted(obs_df['node'].unique())[0],  # default to first node
                clearable=False,
                style={'width': '200px'}
            )
        ], style={'margin': '10px'}),

        html.Div([
            dcc.Graph(id='3d-profile', style={'flex': '1', 'height': '700px'}),
            dcc.Graph(id='obs-plot', style={'flex': '1', 'height': '700px'})
        ], style={'display': 'flex', 'flexDirection': 'row'}),

        dcc.Store(id="stored-camera", data=default_camera)
    ])

    @app.callback(
        Output("interval", "disabled"),
        Input("play-button", "n_clicks"),
        prevent_initial_call=True
    )
    def toggle_play(n_clicks):
        return n_clicks % 2 == 0

    @app.callback(
        Output("time-slider", "value"),
        Input("interval", "n_intervals"),
        State("time-slider", "value"),
        State("stride-input", "value")
    )
    def advance_slider(n_intervals, current_idx, stride):
        stride = max(1, int(stride)) if stride else 1
        return 0 if (current_idx + stride) >= len(t) else current_idx + stride

    @app.callback(
        Output("stored-camera", "data"),
        Input("3d-profile", "relayoutData"),
        State("stored-camera", "data")
    )
    def update_camera_from_user(relayout_data, current_camera):
        if relayout_data and "scene.camera" in relayout_data:
            return relayout_data["scene.camera"]
        return current_camera

    @app.callback(
        Output("3d-profile", "figure"),
        Input("time-slider", "value"),
        State("stored-camera", "data")
    )
    def update_3d_plot(time_idx, camera):
        wd = water_depths[time_idx, :]
        water_surface_z = z + wd

        # Interleave x, y, z with None for separating line segments
        line_x = np.empty(3 * len(x))
        line_y = np.empty(3 * len(y))
        line_z = np.empty(3 * len(z))
        color_vals = np.empty(3 * len(z))

        line_x[::3] = x
        line_x[1::3] = x
        line_x[2::3] = None

        line_y[::3] = y
        line_y[1::3] = y
        line_y[2::3] = None

        line_z[::3] = z
        line_z[1::3] = water_surface_z
        line_z[2::3] = None

        color_vals[::3] = wd
        color_vals[1::3] = wd
        color_vals[2::3] = np.nan  # ignored

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode='lines',
            line=dict(
                color=color_vals,
                colorscale='Viridis',
                cmin=global_min,
                cmax=global_max,
                width=4,
                colorbar=dict(title="Water Depth [m]")
            ),
            showlegend=False
        ))

        fig.update_layout(
            uirevision='openkarst-session',
            scene=dict(
                xaxis=dict(title='X [m]', range=x_range),
                yaxis=dict(title='Y [m]', range=y_range),
                zaxis=dict(title='Elevation [m]', range=z_range),
            ),
            title=f"Water Depth at t = {t[time_idx]:.1f} s",
            height=700,
            margin=dict(l=10, r=10, b=10, t=40)
        )

        return fig


    @app.callback(
        Output("obs-plot", "figure"),
        Input("time-slider", "value"),
        Input("node-selector", "value")
    )
    def update_obs_plot_callback(time_idx, node_id):
        if obs_df is None:
            return go.Figure()

        df_full = obs_df[obs_df['node'] == node_id]

        current_time = results['time'][time_idx]
        df_visible = df_full[df_full['time'] <= current_time]

        x_min, x_max = df_full['time'].min(), df_full['time'].max()
        y_min, y_max = df_full['inflow'].min(), df_full['inflow'].max()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_visible['time'],
            y=df_visible['inflow'],
            mode='lines+markers',
            name='Outflow',
            line=dict(color='blue')
        ))

        fig.update_layout(
            title=f'Cumulative Outflow at Node {node_id}',
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(
                title='Time [s]',
                range=[x_min, x_max],
                fixedrange=True
            ),
            yaxis=dict(
                title='Flow [m³/s]',
                range=[y_min, y_max],
                fixedrange=True
            )
        )

        return fig

    def run_app():
        app.run_server(debug=False, use_reloader=False)

    threading.Thread(target=run_app).start()
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8050/")
