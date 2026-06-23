# Quick start

This quick start runs a small transient simulation on a straight conduit. It is
shorter than the full example scripts and is intended as the first thing to try
after installation.

## Complete script

Create a file named `quick_start.py` and run it from an environment where
openKARST is installed.

```python
import numpy as np
import openpnm as op

from openkarst.models import FlowSimulation
from openkarst.network_generation import compute_conduit_lengths


network = op.network.Cubic(shape=[20, 1, 1], connectivity=6, spacing=1.0)
network = compute_conduit_lengths(network)
network["throat.diameters"] = 1.0
network["throat.epsilon"] = 0.03

simulation_settings = {
    "adaptive_timesteps": True,
    "dt_init": 0.01,
    "dt_max": 0.5,
    "t_max": 30.0,
    "print_info_interval": 500,
}

flow = FlowSimulation(network, simulation_settings=simulation_settings)

initial_Q = np.zeros(network.Nt)
initial_y = np.full(network.Np, 0.01)
flow.set_initial_conditions(initial_Q, initial_y)

flow.set_inflow_BC(nodes=0, values=0.01, inflow_type="volumetric")
flow.set_waterdepth_BC(nodes=19, values=0.01)

outputs = {
    "output_interval": 1.0,
    "time": True,
    "flowrates": True,
    "water_depths": True,
    "time_step_size": True,
}

results = flow.run_simulation(desired_outputs=outputs)

print(results.keys())
print(results["time"][-1])
print(results["water_depths"][-1])
```

## What the script does

The script follows the standard openKARST lifecycle:

1. Build a network.
2. Assign conduit lengths, diameters, and roughness.
3. Create a `FlowSimulation` object.
4. Set initial flow rates and water depths.
5. Apply an inlet and an outlet boundary condition.
6. Run the simulation and inspect selected outputs.

## Expected result

The printed dictionary keys should match the outputs requested in `outputs`.
The final water-depth array has one value per node.
