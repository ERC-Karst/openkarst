"""Data normalization helpers for the openKARST viewer."""

import numpy as np
import plotly.colors as pc

from .constants import (
    DIAMETER_CANDIDATES,
    DIAMETER_COLOR_SCALE,
    DEFAULT_OBS_RENDER_POINTS,
    FIELD_COLOR_SCALES,
    FIELD_LABELS,
    FIELD_RESULT_CANDIDATES,
    FIELD_SYMMETRIC_RANGE,
    FLOW_COLUMN_CANDIDATES,
    OBSERVATION_AXIS_LABELS,
    OBSERVATION_BASE_COLUMNS,
    OBSERVATION_LABELS,
)


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


def _is_l2_norm_column(column):
    return "l2" in str(column).lower()


def _is_observation_base_column(column):
    return str(column).lower() in OBSERVATION_BASE_COLUMNS


def _observation_label(column):
    return OBSERVATION_LABELS.get(column, str(column).replace("_", " ").title())


def _observation_axis_label(column):
    return OBSERVATION_AXIS_LABELS.get(column, _observation_label(column))


def _observation_property_specs(obs_df):
    if obs_df is None or obs_df.empty:
        return {}

    specs = {}
    for column in obs_df.columns:
        if _is_observation_base_column(column) or _is_l2_norm_column(column):
            continue
        try:
            values = np.asarray(obs_df[column], dtype=float)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).any():
            continue
        specs[column] = {
            "label": _observation_label(column),
            "axis_label": _observation_axis_label(column),
            "range": _finite_range(values),
        }
    return specs


def _default_observation_property(property_specs, obs_df):
    if not property_specs:
        return None

    flow_column = _flow_column(obs_df)
    if flow_column in property_specs:
        return flow_column

    return next(iter(property_specs))


def _finite_range(values, symmetric=False):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [0.0, 1.0]

    if symmetric:
        limit = max(float(np.max(np.abs(finite))), 1e-12)
        return [-limit, limit]

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-12)
        return [vmin - pad, vmax + pad]
    return [vmin, vmax]


def _positive_range(values):
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return None

    vmin = float(np.min(positive))
    vmax = float(np.max(positive))
    if np.isclose(vmin, vmax):
        return None
    return [vmin, vmax]


def _log_color_range(field_id, values):
    if field_id in FIELD_SYMMETRIC_RANGE:
        return None
    return _positive_range(values)


def _log10_color_values(values, positive_min):
    arr = np.asarray(values, dtype=float)
    clipped = np.where(np.isfinite(arr), np.maximum(arr, positive_min), np.nan)
    return np.log10(clipped)


def _log_colorbar_ticks(positive_range):
    vmin, vmax = positive_range
    log_min = float(np.log10(vmin))
    log_max = float(np.log10(vmax))
    first_power = int(np.ceil(log_min))
    last_power = int(np.floor(log_max))
    powers = list(range(first_power, last_power + 1))

    if 2 <= len(powers) <= 6:
        tickvals = powers
        ticktext = [f"{10 ** power:g}" for power in powers]
    else:
        tickvals = np.linspace(log_min, log_max, 5)
        ticktext = [f"{10 ** value:g}" for value in tickvals]

    return {
        "tickvals": tickvals,
        "ticktext": ticktext,
    }


def _number_value(value, default=1.0, minimum=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if not np.isfinite(result):
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    return result


def _int_value(value, default=1, minimum=None):
    return int(round(_number_value(value, default=default, minimum=minimum)))


def _format_number(value, integer=False):
    if integer:
        return str(int(round(value)))
    return f"{float(value):g}"


def _throat_to_node_values(throat_values, conns, n_nodes):
    throat_values = np.asarray(throat_values, dtype=float)
    if throat_values.ndim == 1:
        throat_values = throat_values[np.newaxis, :]

    node_values = np.zeros((throat_values.shape[0], n_nodes), dtype=float)
    counts = np.zeros(n_nodes, dtype=float)
    for throat_idx, (i, j) in enumerate(conns):
        vals = throat_values[:, throat_idx]
        node_values[:, i] += vals
        node_values[:, j] += vals
        counts[i] += 1.0
        counts[j] += 1.0

    counts[counts == 0.0] = 1.0
    return node_values / counts


def _as_time_node_array(values, n_times, n_nodes, conns):
    arr = np.asarray(values, dtype=float)
    n_throats = len(conns)

    if arr.ndim == 1:
        if arr.shape[0] == n_nodes:
            return np.tile(arr, (n_times, 1))
        if arr.shape[0] == n_throats:
            node_values = _throat_to_node_values(arr, conns, n_nodes)
            return np.tile(node_values, (n_times, 1))
        return None

    if arr.ndim != 2:
        return None

    if arr.shape == (n_times, n_nodes):
        return arr
    if arr.shape == (n_nodes, n_times):
        return arr.T
    if arr.shape == (n_times, n_throats):
        return _throat_to_node_values(arr, conns, n_nodes)
    if arr.shape == (n_throats, n_times):
        return _throat_to_node_values(arr.T, conns, n_nodes)
    return None


def _require_time_node_array(values, n_times, n_nodes, conns, label):
    arr = _as_time_node_array(values, n_times, n_nodes, conns)
    if arr is None:
        raise ValueError(
            f"{label} must contain node or throat values with a supported time shape"
        )
    return arr


def _field_specs(results, geometry, water_depths=None):
    t = results["time"]
    coords = geometry["pore.coords"]
    conns = geometry["throat.conns"]
    n_times = len(t)
    n_nodes = len(coords)
    if water_depths is None:
        water_depths = _require_time_node_array(
            results["water_depths"],
            n_times,
            n_nodes,
            conns,
            "results['water_depths']",
        )

    specs = {
        "depth": {
            "label": FIELD_LABELS["depth"],
            "array": water_depths,
            "range": _finite_range(water_depths),
            "log_range": _log_color_range("depth", water_depths),
            "colorscale": FIELD_COLOR_SCALES["depth"],
        }
    }

    for field_id, candidates in FIELD_RESULT_CANDIDATES.items():
        for key in candidates:
            if key not in results:
                continue
            field_values = _as_time_node_array(results[key], n_times, n_nodes, conns)
            if field_values is None:
                continue
            specs[field_id] = {
                "label": FIELD_LABELS[field_id],
                "array": field_values,
                "range": _finite_range(
                    field_values,
                    symmetric=field_id in FIELD_SYMMETRIC_RANGE,
                ),
                "log_range": _log_color_range(field_id, field_values),
                "colorscale": FIELD_COLOR_SCALES[field_id],
                "source": key,
            }
            break

    return specs


def _diameter_values(geometry, n_throats):
    for key in DIAMETER_CANDIDATES:
        if key not in geometry:
            continue
        values = np.asarray(geometry[key], dtype=float).reshape(-1)
        if values.size == n_throats:
            return values
    return None


def _build_diameter_segments(x, y, z, conns, diameters):
    if diameters is None:
        return []

    finite = diameters[np.isfinite(diameters)]
    if finite.size == 0:
        return []

    bin_count = min(5, max(1, finite.size))
    quantiles = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bin_count + 1)))
    if quantiles.size <= 2:
        bin_ids = np.zeros(len(diameters), dtype=int)
        bin_count = 1
    else:
        bin_ids = np.digitize(diameters, quantiles[1:-1], right=True)
        bin_count = len(quantiles) - 1

    dmin = float(np.min(finite))
    dmax = float(np.max(finite))
    dspan = max(dmax - dmin, 1e-12)
    colors = pc.sample_colorscale(
        DIAMETER_COLOR_SCALE,
        np.linspace(0.12, 0.88, bin_count),
    )
    segments = []

    for bin_id in range(bin_count):
        throat_ids = np.where(bin_ids == bin_id)[0]
        if throat_ids.size == 0:
            continue
        median_diameter = float(np.nanmedian(diameters[throat_ids]))
        lower = float(np.nanmin(diameters[throat_ids]))
        upper = float(np.nanmax(diameters[throat_ids]))
        width = 1.5 + 5.5 * (median_diameter - dmin) / dspan
        line_x = []
        line_y = []
        line_z = []
        for throat_idx in throat_ids:
            i, j = conns[throat_idx]
            line_x.extend([float(x[i]), float(x[j]), None])
            line_y.extend([float(y[i]), float(y[j]), None])
            line_z.extend([float(z[i]), float(z[j]), None])
        segments.append({
            "x": line_x,
            "y": line_y,
            "z": line_z,
            "width": float(width),
            "diameter": median_diameter,
            "label": f"{lower:.2f}-{upper:.2f} m",
            "color": colors[bin_id],
        })

    return segments


def _time_axis_range(t):
    start = float(t[0])
    end = float(t[-1])
    if not np.isclose(start, end):
        return [start, end]

    pad = max(abs(start) * 0.05, 1.0)
    return [start - pad, end + pad]


def _thin_frame(df, max_points=DEFAULT_OBS_RENDER_POINTS):
    if len(df) <= max_points:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, max_points, dtype=int))
    return df.iloc[idx]


def _precompute_profile_context(geometry, results):
    coords = geometry["pore.coords"]
    conns = geometry["throat.conns"]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    n_times = len(results["time"])
    n_nodes = len(coords)

    water_depths = _require_time_node_array(
        results["water_depths"],
        n_times,
        n_nodes,
        conns,
        "results['water_depths']",
    )

    line_x = []
    line_y = []
    line_z = []
    diameter_line_color = []
    diameters = _diameter_values(geometry, len(conns))
    for i, j in conns:
        line_x.extend([float(x[i]), float(x[j]), None])
        line_y.extend([float(y[i]), float(y[j]), None])
        line_z.extend([float(z[i]), float(z[j]), None])
    if diameters is not None:
        for diameter in diameters:
            diameter_line_color.extend([float(diameter), float(diameter), np.nan])

    diameter_range = _finite_range(diameters) if diameters is not None else None
    return {
        "x": x,
        "y": y,
        "z": z,
        "conns": conns,
        "water_depths": water_depths,
        "line_x": line_x,
        "line_y": line_y,
        "line_z": line_z,
        "x_range": [float(np.min(x)), float(np.max(x))],
        "y_range": [float(np.min(y)), float(np.max(y))],
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "depth_min": float(np.min(water_depths)),
        "depth_max": float(np.max(water_depths)),
        "fields": _field_specs(results, geometry, water_depths=water_depths),
        "diameter_line_color": diameter_line_color,
        "diameter_range": diameter_range,
        "diameter_segments": _build_diameter_segments(x, y, z, conns, diameters),
    }


def _precompute_observation_context(obs_df):
    if (
        obs_df is None
        or obs_df.empty
        or "node" not in obs_df.columns
        or "time" not in obs_df.columns
    ):
        return None

    property_specs = _observation_property_specs(obs_df)
    if not property_specs:
        return None

    sorted_df = obs_df.sort_values(["node", "time"])
    by_node = {
        int(node): node_df.reset_index(drop=True)
        for node, node_df in sorted_df.groupby("node", sort=True)
    }
    return {
        "by_node": by_node,
        "property_specs": property_specs,
        "default_property": _default_observation_property(property_specs, obs_df),
        "x_range": _time_axis_range(np.asarray(sorted_df["time"], dtype=float)),
    }
