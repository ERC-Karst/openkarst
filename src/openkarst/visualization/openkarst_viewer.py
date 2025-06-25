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
import plotly.colors as pc

COLOR_CYCLE = pc.qualitative.Plotly


def launch_openkarst_viewer(results, geometry, obs_df=None):
    coords = geometry['pore.coords']
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    t = results['time']
    water_depths = results['water_depths']

    max_z_total = float(np.max(z + water_depths))
    x_range = [float(np.min(x)), float(np.max(x))]
    y_range = [float(np.min(y)), float(np.max(y))]
    z_range = [float(np.min(z)), max_z_total]

    default_camera = dict(eye=dict(x=1.4, y=1.4, z=1.4))

    global_min = float(np.min(water_depths))
    global_max = float(np.max(water_depths))

    app = dash.Dash(__name__)
    app.title = "openKARST 3D Viewer"

    OBS_NODE_COLORS = {}
    if obs_df is not None:
        unique_nodes = sorted(obs_df['node'].unique())
        OBS_NODE_COLORS = {
            node_id: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, node_id in enumerate(unique_nodes)
        }

    app.layout = html.Div([
        
        html.H2("openKARST 3D Viewer", style={'textAlign': 'center'}),

        # html.Div([
        #     html.Label("Stride:"),
        #     dcc.Input(id="stride-input", type="number", value=1, min=1, step=1, style={'width': '80px'}),
        #     html.Button("Play", id="play-button", n_clicks=0),
        # ], style={"margin": "10px"}),
        html.Div([
            html.Label("Stride:"),
            dcc.Input(id="stride-input", type="number", value=1, min=1, step=1, style={'width': '40px'}),

            html.Label(" "),
            html.Button("Play", id="play-button", n_clicks=0, style={'marginLeft': '5px'}),

            html.Label(" "),
            html.Label("Observation node(s):", style={'marginLeft': '20px'}),
            dcc.Checklist(
                id='node-selector',
                options=[{'label': str(n), 'value': n} for n in sorted(obs_df['node'].unique())],
                value=[sorted(obs_df['node'].unique())[0]],
                inline=True,
                style={'maxWidth': '300px', 'overflowX': 'auto', 'whiteSpace': 'nowrap'}
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'margin': '10px'}),

        dcc.Slider(
            id='time-slider',
            min=0,
            max=len(t) - 1,
            step=1,
            value=0,
            marks={0: '0s', len(t)//2: f'{int(t[len(t)//2])}s', len(t)-1: f'{int(t[-1])}s'},
            updatemode='drag'
        ),

        dcc.Interval(id="interval", interval=10, n_intervals=0, disabled=True),

        html.Div([
            dcc.Graph(id='3d-profile', style={'flex': '1', 'minWidth': '300px', 'width': '100%', 'height': '100%'}),
            dcc.Graph(id='obs-plot', style={'flex': '1', 'minWidth': '300px', 'width': '100%', 'height': '100%'})
        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'flexWrap': 'wrap',
            'height': '70vh',
            'gap': '10px'
        }),

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
        Input("node-selector", "value"),
        State("stored-camera", "data")
    )
    def update_3d_plot(time_idx, node_ids, camera):

        wd = water_depths[time_idx, :]
        water_surface_z = z + wd

        fig = go.Figure()

        # Add static conduit lines between nodes
        conns = geometry['throat.conns']
        x_lines, y_lines, z_lines = [], [], []
        for i, j in conns:
            x_lines += [x[i], x[j], None]
            y_lines += [y[i], y[j], None]
            z_lines += [z[i], z[j], None]

        fig.add_trace(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color='lightgray', width=2),
            hoverinfo='none',
            showlegend=False
        ))

        # Add vertical water depth bars
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
        color_vals[2::3] = np.nan

        

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
                colorbar=dict(
                    title="Depth [m]",
                    #thickness=10,
                    #len=0.5,
                    #x=1.05,
                    xanchor='left',
                    title_font=dict(size=10),
                    tickfont=dict(size=8)
                )
            ),
            showlegend=False
        ))

        for node_id in node_ids:
            fig.add_trace(go.Scatter3d(
                x=[x[node_id]],
                y=[y[node_id]],
                z=[z[node_id] + wd[node_id]],
                mode='markers',
                marker=dict(
                    size=6,
                    color=OBS_NODE_COLORS.get(node_id, 'red'),
                    symbol='circle',
                    opacity=0.9
                ),
                showlegend=False
            ))

        fig.update_layout(
            autosize=True,
            margin=dict(l=10, r=10, b=10, t=40),
            uirevision='openkarst-session',
            scene=dict(
                xaxis=dict(title='X [m]', range=x_range),
                yaxis=dict(title='Y [m]', range=y_range),
                zaxis=dict(title='Elevation [m]', range=z_range),
            ),
            title=f"Water depth at t = {t[time_idx]:.1f} s"
        )

        return fig

    @app.callback(
        Output("obs-plot", "figure"),
        Input("time-slider", "value"),
        Input("node-selector", "value")
    )
    def update_obs_plot_callback(time_idx, node_ids):
        if obs_df is None or not node_ids:
            return go.Figure()

        current_time = results['time'][time_idx]
        fig = go.Figure()

        all_x = []
        all_y = []

        for node_id in node_ids:
            df_full = obs_df[obs_df['node'] == node_id]
            df_visible = df_full[df_full['time'] <= current_time]

            fig.add_trace(go.Scatter(
                x=df_visible['time'],
                y=df_visible['inflow'],
                mode='lines+markers',
                name=f'Node {node_id}',
                line=dict(color=OBS_NODE_COLORS.get(node_id, 'blue'))
            ))

            all_x.extend(df_full['time'])
            all_y.extend(df_full['inflow'])

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        fig.update_layout(
            title='Cumulative outflow at selected nodes',
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(
                title='Time [s]',
                range=[x_min, x_max],
                fixedrange=True
            ),
            yaxis=dict(
                title='Flow [m<sup>3</sup>/s]',
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

