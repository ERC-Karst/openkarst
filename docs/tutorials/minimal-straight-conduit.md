# 1. Minimal straight conduit

This tutorial builds the smallest useful openKARST example: a straight conduit
with constant inflow on the left and constant water depth on the right.

## Goal

You will learn how to:

- create a simple network;
- assign conduit properties;
- create a `FlowSimulation` object;
- run a transient simulation;
- inspect final water depths.

## Code

```python
import numpy as np
import openpnm as op

from openkarst.models import FlowSimulation
from openkarst.network_generation import compute_conduit_lengths


def build_network():
    network = op.network.Cubic(shape=[20, 1, 1], connectivity=6, spacing=1.0)
    network = compute_conduit_lengths(network)
    network["throat.diameters"] = 1.0
    network["throat.epsilon"] = 0.03
    return network


def main():
    network = build_network()

    simulation_settings = {
        "adaptive_timesteps": True,
        "dt_init": 0.01,
        "dt_max": 0.5,
        "t_max": 30.0,
        "print_info_interval": 500,
    }

    flow = FlowSimulation(network, simulation_settings=simulation_settings)

    flow.set_initial_conditions(
        initial_Q=np.zeros(network.Nt),
        initial_y=np.full(network.Np, 0.01),
    )

    flow.set_inflow_BC(nodes=0, values=0.01, inflow_type="volumetric")
    flow.set_waterdepth_BC(nodes=19, values=0.01)

    outputs = {
        "output_interval": 1.0,
        "time": True,
        "flowrates": True,
        "water_depths": True,
        "time_step_size": True,
        "y_l2_norms": True,
        "Q_l2_norms": True,
    }

    results = flow.run_simulation(desired_outputs=outputs)

    print("Stored times:", results["time"])
    print("Final water depths:", results["water_depths"][-1])

    return results, network


if __name__ == "__main__":
    main()
```

## Explanation

`shape=[20, 1, 1]` creates 20 nodes in a line. Because the network has 20
nodes, valid node indices are `0` through `19`.

The initial flow rate has one value per conduit:

```python
np.zeros(network.Nt)
```

The initial water depth has one value per node:

```python
np.full(network.Np, 0.01)
```

The inflow boundary condition injects `0.01 m^3/s` at node `0`. The water-depth
boundary fixes the outlet depth at node `19`.

## Next step

Continue with [Boundary conditions](boundary-conditions.md) to replace the
constant inflow with time-dependent forcing.
