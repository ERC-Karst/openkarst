# Results and observations

openKARST has two complementary output mechanisms:

- the **results container**, which stores arrays for the whole network;
- **observation recorders**, which store compact time series at selected nodes.

## Results container

Pass `desired_outputs` to `run_simulation()`:

```python
outputs = {
    "output_interval": 1.0,
    "time": True,
    "flowrates": True,
    "water_depths": True,
    "reynolds_numbers": True,
}

results = flow.run_simulation(desired_outputs=outputs)
```

Common result shapes:

| Key | Shape |
| --- | --- |
| `time` | one value per stored output time |
| `flowrates` | output time x conduit |
| `water_depths` | output time x node |
| `reynolds_numbers` | output time x conduit |

## Observation points

Observation points are useful when you only need a few node time series:

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

Each call to `set_observation_points()` creates one recorder. The requested
variables must be valid for every node in that recorder. For example,
reservoir variables such as `reservoir_storage` can only be requested for nodes
that have a registered reservoir.

The `name` argument is optional. If it is not provided, openKARST assigns names
such as `observation_0` and `observation_1`. Names are useful only when you want
to retrieve separate recorder tables with `get_observation_dataframes()`.

Reservoir nodes can record standard node variables and reservoir variables in
the same recorder, provided every node in that call has a reservoir:

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

For mixed node groups, split the observation setup:

```python
flow.set_observation_points(
    nodes=all_observation_nodes,
    variables=["water_depth", "connected_net_flowrate"],
    interval=1.0,
    name="nodes",
)

flow.set_observation_points(
    nodes=reservoir_nodes,
    variables=["reservoir_water_depth", "reservoir_storage"],
    interval=1.0,
    name="reservoirs",
)
```

After the run:

```python
obs_df = flow.get_observation_dataframe()
```

`get_observation_dataframe()` returns one combined dataframe, merged by `time`
and `node`. Variables that were not recorded for a row appear as `NaN`. This
single table is convenient for plotting, exporting to CSV, or synchronizing
with the 3D viewer.

Use `get_observation_dataframes()` when you need the separate recorder tables
keyed by recorder name.
