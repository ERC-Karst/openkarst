"""Dash callback registration for the openKARST viewer."""

import dash
from dash import Input, Output, State

from .camera import _camera_for_view, _camera_from_relayout, _normalize_camera
from .constants import DEFAULT_PLAY_STRIDE
from .data import _format_number, _int_value, _number_value
from .figures import (
    _build_convergence_figure,
    _build_observation_figure,
    _build_profile_figure,
)
from .layout import log_scale_container_style


def register_viewer_callbacks(app, viewer_state, depth_scale):
    results = viewer_state.results
    obs_df = viewer_state.obs_df
    profile_context = viewer_state.profile_context
    obs_context = viewer_state.obs_context
    obs_nodes = viewer_state.obs_nodes
    obs_node_colors = viewer_state.obs_node_colors
    default_camera = viewer_state.default_camera
    t = viewer_state.t

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
            return log_scale_container_style(profile_context, color_field), current_value or []
        return log_scale_container_style(profile_context, color_field), []

    @app.callback(
        Output("node-selector", "value"),
        Input("select-all-observation-nodes", "n_clicks"),
        Input("clear-observation-nodes", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_selected_observation_nodes(_select_all_clicks, _clear_clicks):
        triggered = dash.callback_context.triggered
        trigger_id = triggered[0]["prop_id"].split(".")[0] if triggered else None

        if trigger_id == "select-all-observation-nodes":
            return obs_nodes
        if trigger_id == "clear-observation-nodes":
            return []
        return dash.no_update

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
