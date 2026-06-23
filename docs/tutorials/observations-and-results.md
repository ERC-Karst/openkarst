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
)
```

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
