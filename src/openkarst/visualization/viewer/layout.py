"""Dash layout for the openKARST viewer."""

from dash import dcc, html

from .constants import DEFAULT_PLAY_STRIDE, HEADER_LOGO
from .data import _format_number


PAGE_STYLE = {
    "height": "100vh",
    "margin": "0",
    "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "background": "#f7f8fb",
    "color": "#111827",
    "overflow": "hidden",
}
SHELL_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "clamp(165px, 15.5vw, 200px) minmax(0, 1fr)",
    "height": "100vh",
}
SIDEBAR_STYLE = {
    "background": "#ffffff",
    "borderRight": "1px solid #dbe1ea",
    "padding": "8px",
    "overflowY": "auto",
}
TITLE_STYLE = {
    "fontSize": "19px",
    "fontWeight": "650",
    "letterSpacing": "0",
    "margin": "0 0 5px",
}
HEADER_LOGO_STYLE = {
    "display": "block",
    "width": "100%",
    "maxWidth": "130px",
    "height": "auto",
    "margin": "0 0 5px",
}
SECTION_STYLE = {
    "border": "1px solid #dbe1ea",
    "borderRadius": "6px",
    "padding": "5px 6px",
    "margin": "0 0 5px",
    "background": "#fbfcfe",
}
LEGEND_STYLE = {
    "fontSize": "10px",
    "fontWeight": "700",
    "padding": "0 4px",
    "color": "#374151",
}
LABEL_STYLE = {
    "display": "block",
    "fontSize": "10px",
    "fontWeight": "650",
    "color": "#4b5563",
    "marginBottom": "3px",
}
CONTROL_STYLE = {
    "marginBottom": "5px",
}
INPUT_STYLE = {
    "width": "100%",
    "height": "28px",
    "boxSizing": "border-box",
    "border": "1px solid #cfd6e1",
    "borderRadius": "6px",
    "padding": "4px 8px",
    "fontSize": "12px",
    "background": "#ffffff",
}
STEPPER_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "26px minmax(0, 1fr) 26px",
    "gap": "4px",
    "alignItems": "center",
}
STEPPER_BUTTON_STYLE = {
    "height": "28px",
    "border": "1px solid #cfd6e1",
    "borderRadius": "6px",
    "background": "#eef2f7",
    "color": "#111827",
    "fontSize": "16px",
    "fontWeight": "700",
    "cursor": "pointer",
}
STEPPER_INPUT_STYLE = {
    **INPUT_STYLE,
    "textAlign": "center",
    "padding": "4px",
}
BUTTON_STYLE = {
    "width": "100%",
    "height": "30px",
    "border": "1px solid #1f6feb",
    "borderRadius": "6px",
    "background": "#1f6feb",
    "color": "#ffffff",
    "fontWeight": "650",
    "cursor": "pointer",
}
SECONDARY_BUTTON_STYLE = {
    "height": "26px",
    "border": "1px solid #cbd5e1",
    "borderRadius": "6px",
    "background": "#ffffff",
    "color": "#334155",
    "fontSize": "12px",
    "fontWeight": "650",
    "cursor": "pointer",
}
BUTTON_ROW_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr",
    "gap": "6px",
    "marginTop": "-4px",
}
CHECKLIST_STYLE = {
    "fontSize": "12px",
    "lineHeight": "1.2",
}
MAIN_STYLE = {
    "display": "grid",
    "gridTemplateRows": "82px minmax(0, 1fr)",
    "minWidth": "0",
    "height": "100vh",
    "overflow": "hidden",
}
TIMEBAR_STYLE = {
    "background": "#ffffff",
    "borderBottom": "1px solid #dbe1ea",
    "padding": "14px 22px 8px",
}
TIME_LABEL_STYLE = {
    "display": "block",
    "fontSize": "12px",
    "fontWeight": "700",
    "color": "#4b5563",
    "marginBottom": "6px",
}
PLOT_AREA_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "minmax(0, 1.7fr) minmax(320px, 0.9fr)",
    "gridTemplateRows": "minmax(0, 1fr) minmax(220px, 0.72fr)",
    "gap": "10px",
    "minHeight": "0",
    "padding": "10px",
    "alignItems": "stretch",
}
GRAPH_STYLE = {
    "width": "100%",
    "height": "100%",
    "minHeight": "0",
    "background": "#ffffff",
    "border": "1px solid #dbe1ea",
    "borderRadius": "8px",
    "overflow": "hidden",
}
DROPDOWN_STYLE = {"fontSize": "12px"}


def control(label, component):
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        component,
    ], style=CONTROL_STYLE)


def stepper(component_id, value, integer=False):
    return html.Div([
        html.Button(
            "-",
            id=f"{component_id}-minus",
            n_clicks=0,
            style=STEPPER_BUTTON_STYLE,
        ),
        dcc.Input(
            id=component_id,
            type="text",
            value=_format_number(value, integer=integer),
            style=STEPPER_INPUT_STYLE,
        ),
        html.Button(
            "+",
            id=f"{component_id}-plus",
            n_clicks=0,
            style=STEPPER_BUTTON_STYLE,
        ),
    ], style=STEPPER_STYLE)


def section(title, children):
    return html.Fieldset([
        html.Legend(title, style=LEGEND_STYLE),
        *children,
    ], style=SECTION_STYLE)


def log_scale_container_style(profile_context, field_id):
    field_spec = profile_context["fields"].get(field_id)
    if field_spec and field_spec.get("log_range"):
        return {"display": "block", "margin": "-3px 0 3px"}
    return {"display": "none"}


def _time_slider_marks(t):
    return {
        0: "0s",
        len(t) // 2: f"{int(t[len(t) // 2])}s",
        len(t) - 1: f"{int(t[-1])}s",
    }


def build_viewer_layout(app, viewer_state, depth_scale, assets_dir):
    field_options = viewer_state.field_options
    obs_nodes = viewer_state.obs_nodes
    t = viewer_state.t

    header_component = (
        html.Img(
            src=app.get_asset_url(HEADER_LOGO),
            alt="openKARST",
            style=HEADER_LOGO_STYLE,
        )
        if assets_dir is not None
        else html.H2("openKARST Viewer", style=TITLE_STYLE)
    )

    return html.Div([
        html.Div([
            html.Aside([
                header_component,

                section("Playback", [
                    control("Stride", stepper("stride-input", DEFAULT_PLAY_STRIDE, integer=True)),
                    html.Button("Play", id="play-button", n_clicks=0, style=BUTTON_STYLE),
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
                        style=DROPDOWN_STYLE,
                    )),
                ]),

                section("Water Bars", [
                    control("Color by", dcc.Dropdown(
                        id="color-field-selector",
                        options=field_options,
                        value=field_options[0]["value"],
                        clearable=False,
                        style=DROPDOWN_STYLE,
                    )),
                    html.Div(
                        dcc.Checklist(
                            id="bar-log-scale-toggle",
                            options=[{"label": "Log color scale", "value": "log"}],
                            value=[],
                            style=CHECKLIST_STYLE,
                        ),
                        id="bar-log-scale-container",
                        style=log_scale_container_style(
                            viewer_state.profile_context,
                            field_options[0]["value"],
                        ),
                    ),
                    dcc.Checklist(
                        id="vertical-bars-toggle",
                        options=[{
                            "label": "Vertical bars",
                            "value": "bars",
                        }],
                        value=["bars"],
                        style=CHECKLIST_STYLE,
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
                            "disabled": not viewer_state.has_diameters,
                        }],
                        value=[],
                        style=CHECKLIST_STYLE,
                    ),
                    dcc.Checklist(
                        id="edge-color-toggle",
                        options=[{
                            "label": "Color by diameter",
                            "value": "diameter-color",
                            "disabled": not viewer_state.has_diameters,
                        }],
                        value=[],
                        style={**CHECKLIST_STYLE, "marginTop": "4px"},
                    ),
                    html.Div(style={"height": "2px"}),
                    control("Diameter scale", stepper("diameter-scale-input", 1)),
                ]),

                section("Observations", [
                    control("Property", dcc.Dropdown(
                        id="observation-property-selector",
                        options=viewer_state.observation_property_options,
                        value=viewer_state.default_observation_property,
                        clearable=False,
                        disabled=not bool(viewer_state.observation_property_options),
                        placeholder="None",
                        style=DROPDOWN_STYLE,
                    )),
                    control("Observation nodes", dcc.Dropdown(
                        id="node-selector",
                        options=[{"label": str(n), "value": n} for n in obs_nodes],
                        value=viewer_state.selected_nodes,
                        multi=True,
                        disabled=not bool(obs_nodes),
                        placeholder="None",
                        style=DROPDOWN_STYLE,
                    )),
                    html.Div([
                        html.Button(
                            "All",
                            id="select-all-observation-nodes",
                            n_clicks=0,
                            disabled=not bool(obs_nodes),
                            style=SECONDARY_BUTTON_STYLE,
                        ),
                        html.Button(
                            "None",
                            id="clear-observation-nodes",
                            n_clicks=0,
                            disabled=not bool(obs_nodes),
                            style=SECONDARY_BUTTON_STYLE,
                        ),
                    ], style=BUTTON_ROW_STYLE),
                ]),
            ], style=SIDEBAR_STYLE),

            html.Main([
                html.Div([
                    html.Label("Time", style=TIME_LABEL_STYLE),
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=len(t) - 1,
                        step=1,
                        value=0,
                        marks=_time_slider_marks(t),
                        updatemode="drag",
                    ),
                ], style=TIMEBAR_STYLE),

                html.Div([
                    dcc.Graph(
                        id="3d-profile",
                        style={
                            **GRAPH_STYLE,
                            "gridColumn": "1",
                            "gridRow": "1 / span 2",
                        },
                    ),
                    dcc.Graph(
                        id="obs-plot",
                        style={
                            **GRAPH_STYLE,
                            "gridColumn": "2",
                            "gridRow": "1",
                        },
                    ),
                    dcc.Graph(
                        id="convergence-plot",
                        style={
                            **GRAPH_STYLE,
                            "gridColumn": "2",
                            "gridRow": "2",
                        },
                    ),
                ], style=PLOT_AREA_STYLE),
            ], style=MAIN_STYLE),
        ], style=SHELL_STYLE),

        dcc.Interval(id="interval", interval=120, n_intervals=0, disabled=True),
        dcc.Store(id="stored-camera", data=viewer_state.default_camera),
    ], style=PAGE_STYLE)
