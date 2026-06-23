# Simulation workflow

Most openKARST scripts follow the same sequence.

## 1. Define settings

Settings are passed as dictionaries. Missing entries use defaults, but
`dt_init`, `dt_max`, and `t_max` should usually be set explicitly for transient
runs.

```python
simulation_settings = {
    "adaptive_timesteps": True,
    "dt_init": 0.01,
    "dt_max": 0.5,
    "t_max": 100.0,
}
```

## 2. Create the `FlowSimulation` object

```python
from openkarst.models import FlowSimulation

flow = FlowSimulation(
    network,
    physical_properties=physical_properties,
    solver_settings=solver_settings,
    simulation_settings=simulation_settings,
)
```

## 3. Set initial conditions

The initial flow-rate array has one entry per conduit. The initial water-depth
array has one entry per node.

```python
initial_Q = np.zeros(network.Nt)
initial_y = np.full(network.Np, 0.01)
flow.set_initial_conditions(initial_Q, initial_y)
```

## 4. Apply boundary conditions

Use separate methods for inflow and water depth:

```python
flow.set_inflow_BC(nodes=0, values=0.01, inflow_type="volumetric")
flow.set_waterdepth_BC(nodes=19, values=0.01)
```

openKARST checks for incompatible boundary-condition assignments before the
simulation begins and reports them with a `ValueError`.

## 5. Choose outputs

Only requested outputs are stored:

```python
outputs = {
    "output_interval": 1.0,
    "time": True,
    "flowrates": True,
    "water_depths": True,
}
```

## 6. Run

```python
results = flow.run_simulation(desired_outputs=outputs)
```

The result values are NumPy arrays after the simulation finishes.
