# Results and observations

openKARST has two complementary output mechanisms:

- the **results container**, which stores arrays for the whole network;
- the **observation recorder**, which stores compact time series at selected nodes.

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
)
```

After the run:

```python
obs_df = flow.get_observation_dataframe()
```

The dataframe is convenient for plotting, exporting to CSV, or synchronizing
with the 3D viewer.
