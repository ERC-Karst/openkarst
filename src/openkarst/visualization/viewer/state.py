"""Small data container used while building the Dash viewer."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ViewerState:
    results: Any
    geometry: Any
    obs_df: Any
    t: Any
    profile_context: dict
    obs_context: dict | None
    obs_nodes: list[int]
    selected_nodes: list[int]
    observation_property_options: list[dict]
    default_observation_property: str | None
    field_options: list[dict]
    has_diameters: bool
    obs_node_colors: dict[int, str]
    default_camera: dict
