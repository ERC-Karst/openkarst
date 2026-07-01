# 3. Observations and results

This tutorial adds observation points to the minimal straight-conduit example.
Observation points are useful when you want compact time series for selected
nodes.

## Add observations before running

```python
flow.set_observation_points(
    nodes=[0, 19],
    variables=[
        "water_depth",
        "connected_abs_flowrate",
        "connected_net_flowrate",
    ],
    interval=1.0,
    name="boundary_nodes",
)
```

Each call defines one observation recorder. All variables in that call must be
valid for all nodes in that call.

The `name` argument is optional. If you omit it, openKARST assigns names such as
`observation_0`, `observation_1`, and so on. Names are only needed when you want
to identify separate recorder outputs from `get_observation_dataframes()`.

## Run and retrieve data

```python
results = flow.run_simulation(desired_outputs=outputs)
obs_df = flow.get_observation_dataframe()

print(obs_df.head())
```

## Observation variables

| Variable | Meaning |
| --- | --- |
| `water_depth` | Water depth at each observed node. |
| `connected_abs_flowrate` | Sum of absolute conduit flow rates connected to the node. |
| `connected_net_flowrate` | Signed net conduit flow rate into the node. |
| `concentrations` | Concentration at each observed node when transport is enabled. |
| `mass` | Mass at each observed node when transport is enabled. |
| `reservoir_water_depth` | Reservoir water depth at each observed reservoir node. |
| `reservoir_head` | Reservoir hydraulic head at each observed reservoir node. |
| `reservoir_storage` | Reservoir storage at each observed reservoir node. |
| `reservoir_exchange` | Reservoir-node exchange rate at each observed reservoir node. |
| `reservoir_recharge` | Reservoir recharge rate at each observed reservoir node. |

Reservoir variables can only be requested for nodes where a reservoir has been
registered with `add_reservoir(...)`. If a recorder requests a reservoir
variable for a non-reservoir node, openKARST raises a `ValueError`.

Reservoir nodes are still normal network nodes, so they can record standard
node variables and reservoir variables in the same recorder:

```python
flow.set_observation_points(
    nodes=reservoir_nodes,
    variables=[
        "water_depth",
        "connected_net_flowrate",
        "reservoir_water_depth",
        "reservoir_storage",
        "reservoir_exchange",
    ],
    interval=1.0,
    name="reservoirs",
)
```

## Mixed node types

Use separate recorder calls when different node groups need different
variables:

```python
flow.set_observation_points(
    nodes=all_observation_nodes,
    variables=["water_depth", "connected_net_flowrate"],
    interval=1.0,
    name="nodes",
)

flow.set_observation_points(
    nodes=reservoir_nodes,
    variables=["reservoir_water_depth", "reservoir_storage", "reservoir_exchange"],
    interval=1.0,
    name="reservoirs",
)
```

`get_observation_dataframe()` returns one combined dataframe, merged by `time`
and `node`. Columns that were not recorded for a row contain `NaN`.

The `name` arguments above are optional, but they make the individual recorder
tables easier to identify. If you need those individual outputs, use:

```python
observation_tables = flow.get_observation_dataframes()
```

## Simple plot

```python
import matplotlib.pyplot as plt

for node, group in obs_df.groupby("node"):
    plt.plot(group["time"], group["water_depth"], label=f"node {node}")

plt.xlabel("Time (s)")
plt.ylabel("Water depth (m)")
plt.legend()
plt.show()
```

## Results versus observations

Use results when you need whole-network arrays at output times. Use observation
points when you need clean node time series for analysis, plotting, or the
openKARST viewer.
