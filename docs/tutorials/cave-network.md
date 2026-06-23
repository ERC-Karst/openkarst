# 4. Cave network import

This tutorial imports a cave survey represented by nodes and conduits, then
uses the resulting network in the same workflow as the straight-conduit
examples.

## Cave data format

A cave dataset consists of three semicolon-delimited CSV files. Each file has a
header row.

| File | Required columns | Content |
| --- | --- | --- |
| Nodes | `id;x;y;z` | Consecutive node IDs and three-dimensional coordinates in m |
| Edges | `from_id;to_id` | The two node IDs connected by each conduit |
| Diameters | `id;cswidth;csheight` | Cross-section width and height in m for every node |

Example rows:

=== "nodes.csv"

    ```text
    id;x;y;z
    0;630228.94;178484.48;1774.66
    1;630230.51;178486.20;1774.46
    ```

=== "edges.csv"

    ```text
    from_id;to_id
    0;1
    0;2
    ```

=== "diameters.csv"

    ```text
    id;cswidth;csheight
    0;0.90;0.90
    1;0.98;0.98
    ```

Node IDs must start at zero and remain consecutive without gaps. Every node
must have a diameter entry. The loader first averages `cswidth` and `csheight`
for each node, then assigns each conduit the mean diameter of its two endpoint
nodes. Conduit lengths are calculated from the node coordinates.

## Load external files

```python
from openkarst.io import load_cave_data

network = load_cave_data(
    nodes_file="nodes.csv",
    edges_file="edges.csv",
    diameters_file="diameters.csv",
)
network["throat.epsilon"] = 0.03
```

The returned OpenPNM network is a graph object that stores the imported
geometry and properties. openKARST performs the hydraulic computations.

## Load the built-in dataset

The package includes the Seefeldhoehle dataset:

```python
from openkarst.io import load_cave_data

network = load_cave_data(cave="seefeldhoehle")
network["throat.epsilon"] = 0.03
```

## Choose boundary nodes

Inlet and outlet nodes are modeling choices that should reflect the surveyed
system and the purpose of the simulation. Useful information includes:

- surveyed inlet and outlet labels;
- node elevations;
- nodes connected to only one conduit;
- a plot or 3D inspection of the network geometry.

```python
inflow_nodes = [
    319, 401, 431, 315, 224,
    460, 423, 216, 416, 358,
    367, 405, 388, 391, 412,
]

outflow_nodes = [
    249, 174, 420, 27,
    466, 374, 99, 94,
    199, 74, 212, 467, 347,
]

flow.set_inflow_BC(nodes=inflow_nodes, values=0.01)
flow.set_waterdepth_BC(nodes=outflow_nodes, values=0.01)
```

## Complete example

The complete example uses the selected Seefeldhoehle inlet and outlet nodes for
this teaching run.

```python
import numpy as np

from openkarst.io import load_cave_data
from openkarst.models import FlowSimulation


network = load_cave_data(cave="seefeldhoehle")
network["throat.epsilon"] = 0.03

simulation_settings = {
    "adaptive_timesteps": True,
    "dt_init": 0.01,
    "dt_max": 0.1,
    "t_max": 500.0,
    "print_info_interval": 500,
}

flow = FlowSimulation(network, simulation_settings=simulation_settings)

flow.set_initial_conditions(
    initial_Q=np.zeros(network.Nt),
    initial_y=np.full(network.Np, 0.01),
)

inflow_nodes = [
    319, 401, 431, 315, 224,
    460, 423, 216, 416, 358,
    367, 405, 388, 391, 412,
]

# Keep the 13 selected outlets for this teaching run.
outflow_nodes = [
    249, 174, 420, 27,
    466, 374, 99, 94,
    199, 74, 212, 467, 347,
]

flow.set_inflow_BC(nodes=inflow_nodes, values=0.01)
flow.set_waterdepth_BC(nodes=outflow_nodes, values=0.01)

flow.set_observation_points(
    nodes=inflow_nodes + outflow_nodes,
    variables=["water_depth", "connected_abs_flowrate", "connected_net_flowrate"],
    interval=1.0,
)

outputs = {
    "output_interval": 1.0,
    "time": True,
    "flowrates": True,
    "water_depths": True,
}

results = flow.run_simulation(desired_outputs=outputs)
observations = flow.get_observation_dataframe()
```
