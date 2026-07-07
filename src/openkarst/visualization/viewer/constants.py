"""Shared constants for the browser-based openKARST viewer."""

import plotly.colors as pc


COLOR_CYCLE = pc.qualitative.Plotly
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_PLAY_STRIDE = 1
DEFAULT_OBS_RENDER_POINTS = 1200
DEFAULT_OBS_INLINE_LEGEND_MAX_TRACES = 5
DEFAULT_OBS_COMPACT_LEGEND_MAX_TRACES = 10

FLOW_COLUMN_CANDIDATES = (
    "connected_net_flowrate",
    "connected_abs_flowrate",
    "Q",
    "flowrate",
    "flowrates",
)

OBSERVATION_BASE_COLUMNS = {"time", "node"}
OBSERVATION_LABELS = {
    "water_depth": "Water depth [m]",
    "connected_abs_flowrate": "Connected |Q| [m3/s]",
    "connected_net_flowrate": "Connected net Q [m3/s]",
    "Q": "Flow rate [m3/s]",
    "q": "Flow rate [m3/s]",
    "flowrate": "Flow rate [m3/s]",
    "flowrates": "Flow rate [m3/s]",
    "concentrations": "Concentration [kg/m3]",
    "concentration": "Concentration [kg/m3]",
    "C": "Concentration [kg/m3]",
    "c": "Concentration [kg/m3]",
    "mass": "Mass [kg]",
    "reservoir_water_depth": "Reservoir water depth [m]",
    "reservoir_head": "Reservoir head [m]",
    "reservoir_storage": "Reservoir storage [m3]",
    "reservoir_exchange": "Reservoir exchange [m3/s]",
    "reservoir_recharge": "Reservoir recharge [m3/s]",
}
OBSERVATION_AXIS_LABELS = {
    "water_depth": "Water depth [m]",
    "connected_abs_flowrate": "Connected |Q| [m<sup>3</sup>/s]",
    "connected_net_flowrate": "Connected net Q [m<sup>3</sup>/s]",
    "Q": "Flow rate [m<sup>3</sup>/s]",
    "q": "Flow rate [m<sup>3</sup>/s]",
    "flowrate": "Flow rate [m<sup>3</sup>/s]",
    "flowrates": "Flow rate [m<sup>3</sup>/s]",
    "concentrations": "Concentration [kg/m<sup>3</sup>]",
    "concentration": "Concentration [kg/m<sup>3</sup>]",
    "C": "Concentration [kg/m<sup>3</sup>]",
    "c": "Concentration [kg/m<sup>3</sup>]",
    "mass": "Mass [kg]",
    "reservoir_water_depth": "Reservoir water depth [m]",
    "reservoir_head": "Reservoir head [m]",
    "reservoir_storage": "Reservoir storage [m<sup>3</sup>]",
    "reservoir_exchange": "Reservoir exchange [m<sup>3</sup>/s]",
    "reservoir_recharge": "Reservoir recharge [m<sup>3</sup>/s]",
}

FIELD_RESULT_CANDIDATES = {
    "flowrate": ("flowrates", "flowrate", "Q", "q", "flow", "flows", "discharge", "discharges"),
    "velocity": ("velocities", "velocity", "v"),
    "concentration": ("concentrations", "concentration", "C", "c"),
}
FIELD_LABELS = {
    "depth": "Depth [m]",
    "flowrate": "Flow rate [m3/s]",
    "velocity": "Velocity [m/s]",
    "concentration": "Concentration [kg/m3]",
}
FIELD_COLOR_SCALES = {
    "depth": "Viridis",
    "flowrate": "RdBu",
    "velocity": "Plasma",
    "concentration": "Turbo",
}
FIELD_SYMMETRIC_RANGE = {"flowrate"}

DIAMETER_CANDIDATES = (
    "throat.diameter",
    "throat.diameters",
    "diameters",
    "diameter",
)
DIAMETER_COLOR_SCALE = "Cividis"
HEADER_LOGO = "openkarst_header_color.png"

VIEW_CAMERAS = {
    "3d": dict(
        eye=dict(x=1.4, y=1.4, z=1.4),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="perspective"),
    ),
    "xy": dict(
        eye=dict(x=0, y=0, z=2.5),
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="orthographic"),
    ),
    "xz": dict(
        eye=dict(x=0, y=-2.5, z=0),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="orthographic"),
    ),
    "yz": dict(
        eye=dict(x=2.5, y=0, z=0),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="orthographic"),
    ),
}
