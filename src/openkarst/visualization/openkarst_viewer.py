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

def launch_openkarst_viewer(results, geometry):
    coords = geometry['pore.coords']
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    t = results['time']
    water_depths = results['water_depths']

    max_z_total = float(np.max(z + water_depths))
    x_range = [float(np.min(x)), float(np.max(x))]
    y_range = [float(np.min(y)), float(np.max(y))]
    z_range = [float(np.min(z)), max_z_total]

    default_camera = dict(eye=dict(x=1.25, y=1.25, z=1.25))

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
            drag_value = True
        ),

        dcc.Interval(id="interval", interval=10, n_intervals=0, disabled=True),

        dcc.Graph(id='3d-profile'),
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

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color='gray'),
            name="Conduit"
        ))

        for i in range(len(x)):
            fig.add_trace(go.Scatter3d(
                x=[x[i], x[i]],
                y=[y[i], y[i]],
                z=[z[i], water_surface_z[i]],
                mode='lines',
                line=dict(color='blue', width=4),
                showlegend=False
            ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X [m]', range=x_range),
                yaxis=dict(title='Y [m]', range=y_range),
                zaxis=dict(title='Elevation [m]', range=z_range),
                camera=camera
            ),
            title=f"Water Depth at t = {t[time_idx]:.1f} s",
            height=700,
            margin=dict(l=10, r=10, b=10, t=40)
        )
        return fig

    def run_app():
        app.run_server(debug=False, use_reloader=False)

    threading.Thread(target=run_app).start()
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8050/")

