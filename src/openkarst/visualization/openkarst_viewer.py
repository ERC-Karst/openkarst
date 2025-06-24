#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:40:08 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

def launch_openkarst_viewer(results, geometry):
    t = results['time']
    y = results['water_depths']  # shape: (time, node)
    coords = geometry['pore.coords']  # shape: (Np, 3)
    x = coords[:, 0]
    y_pos = coords[:, 1]  # optional, in case you want real 3D layout
    z = coords[:, 2]       # elevation

    node_ids = np.arange(y.shape[1])

    app = dash.Dash(__name__)
    app.title = "openKARST Viewer"

    app.layout = html.Div([
        html.H1("openKARST 3D Depth Viewer", style={'textAlign': 'center'}),

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
            marks={
                0: '0s',
                len(t)//2: f'{int(t[len(t)//2])}s',
                len(t)-1: f'{int(t[-1])}s'
            }
        ),

        dcc.Interval(id="interval", interval=300, n_intervals=0, disabled=True),

        dcc.Graph(id='profile-plot')
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
        Input("time-slider", "value"),
        Input("stride-input", "value")
    )
    def advance_slider(n_intervals, current_idx, stride):
        stride = max(1, int(stride)) if stride else 1
        next_idx = current_idx + stride
        return 0 if next_idx >= len(t) else next_idx

    @app.callback(
        Output("profile-plot", "figure"),
        Input("time-slider", "value")
    )
    def update_profile_plot(time_idx):
        water_depth = y[time_idx, :]
        water_surface = z + water_depth

        fig = go.Figure()

        # Plot conduit network
        fig.add_trace(go.Scatter3d(
            x=x, y=y_pos, z=z,
            mode='lines+markers',
            line=dict(color='gray'),
            marker=dict(size=3),
            name='Conduit Elevation'
        ))

        # Water depth lines
        for i in node_ids:
            fig.add_trace(go.Scatter3d(
                x=[x[i], x[i]],
                y=[y_pos[i], y_pos[i]],
                z=[z[i], water_surface[i]],
                mode='lines',
                line=dict(color='blue'),
                showlegend=False
            ))

        fig.update_layout(
            title=f"3D Water Depth at t = {t[time_idx]:.1f} s",
            scene=dict(
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Elevation [m]'
            ),
            height=600
        )
        return fig

    app.run_server(debug=False)