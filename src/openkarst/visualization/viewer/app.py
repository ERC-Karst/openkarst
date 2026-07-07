"""Dash app assembly for the openKARST viewer."""

from pathlib import Path

import dash

from .callbacks import register_viewer_callbacks
from .camera import _camera_for_view
from .constants import COLOR_CYCLE, DEFAULT_DEPTH_SCALE, HEADER_LOGO
from .data import (
    _observation_nodes,
    _precompute_observation_context,
    _precompute_profile_context,
)
from .layout import build_viewer_layout
from .state import ViewerState


INDEX_STRING = """
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


def _viewer_assets_dir():
    for parent in Path(__file__).resolve().parents:
        assets_dir = parent / "assets"
        if (assets_dir / HEADER_LOGO).is_file():
            return assets_dir
    return None


def _build_viewer_state(results, geometry, obs_df):
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
    obs_node_colors = {
        node_id: COLOR_CYCLE[i % len(COLOR_CYCLE)]
        for i, node_id in enumerate(obs_nodes)
    }
    default_camera = _camera_for_view("3d", dict(eye=dict(x=1.4, y=1.4, z=1.4)))

    return ViewerState(
        results=results,
        geometry=geometry,
        obs_df=obs_df,
        t=results["time"],
        profile_context=profile_context,
        obs_context=obs_context,
        obs_nodes=obs_nodes,
        selected_nodes=selected_nodes,
        observation_property_options=observation_property_options,
        default_observation_property=default_observation_property,
        field_options=field_options,
        has_diameters=bool(profile_context["diameter_segments"]),
        obs_node_colors=obs_node_colors,
        default_camera=default_camera,
    )


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
    viewer_state = _build_viewer_state(results, geometry, obs_df)
    assets_dir = _viewer_assets_dir()
    dash_kwargs = {"assets_folder": str(assets_dir)} if assets_dir is not None else {}

    app = dash.Dash(__name__, **dash_kwargs)
    app.title = "openKARST 3D Viewer"
    app.index_string = INDEX_STRING
    app.layout = build_viewer_layout(app, viewer_state, depth_scale, assets_dir)
    app.openkarst_viewer_state = viewer_state

    register_viewer_callbacks(app, viewer_state, depth_scale)
    return app
