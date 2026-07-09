# Parameters

This page summarizes commonly used settings. Values are passed as dictionaries
to a `FlowSimulation` object or `run_simulation()`.

## Physical properties

```python
physical_properties = {
    "water_density": 1000.0,
    "gravity": 9.81,
    "dynamic_viscosity": 0.001,
    "geometry_channel": False,
    "friction_model": "churchill",
    "steep_slope_correction": False,
}
```

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `water_density` | Water density in kg/m^3. | `1000.0` |
| `gravity` | Gravitational acceleration in m/s^2. | `9.81` |
| `dynamic_viscosity` | Dynamic viscosity in Pa s. | `0.001` |
| `geometry_channel` | Use channel-style geometry. | `False` |
| `channel_type` | `finite` or `infinite` channel. | `finite` |
| `channel_width` | Width for finite channel geometry. | `1.0` |
| `channel_manning` | Manning coefficient for channel geometry. | `0.03` |
| `friction_model` | Closed-conduit friction model. | `churchill` or `hybrid` |
| `steep_slope_correction` | Apply free-surface steep-slope pressure projection and Manning friction scaling using conduit slope angles. | `False` |

## Solver settings

```python
solver_settings = {
    "relaxation_factor": 0.6,
    "max_iterations": 100,
    "picard_depth_tol": 1e-7,
    "ss_rel_l2tol": 1e-3,
    "parallelization": False,
    "num_threads": None,
}
```

| Parameter | Meaning |
| --- | --- |
| `relaxation_factor` | Damping factor for Picard iterations. |
| `max_iterations` | Maximum Picard iterations per time step. |
| `picard_depth_tol` | Water-depth tolerance for Picard convergence. |
| `ss_rel_l2tol` | Relative L2 tolerance for steady-state detection. |
| `parallelization` | Use optional Numba kernels for geometry and flow updates. |
| `num_threads` | Number of Numba worker threads; `None` uses the Numba default. |

## Simulation settings

```python
simulation_settings = {
    "min_waterdepth": 1e-10,
    "min_flowrate": 1e-10,
    "courant": 0.8,
    "adaptive_timesteps": True,
    "dt_init": 0.01,
    "dt_max": 0.1,
    "steady_state": False,
    "t_max": 100.0,
    "print_info_interval": 500,
}
```

| Parameter | Meaning |
| --- | --- |
| `min_waterdepth` | Minimum water depth used for numerical stability. |
| `min_flowrate` | Minimum flow rate used for numerical stability. |
| `courant` | Courant number for time-step control. |
| `adaptive_timesteps` | Whether time step changes during the run. |
| `dt_init` | Initial or constant time step. |
| `dt_max` | Maximum adaptive time step. |
| `steady_state` | Stop based on steady-state convergence. |
| `t_max` | End time for transient simulations. |
| `print_info_interval` | Number of steps between progress messages. |

## Output settings

```python
outputs = {
    "output_interval": 1.0,
    "time": True,
    "time_step_size": True,
    "flowrates": True,
    "water_depths": True,
}
```

| Output key | Stored value |
| --- | --- |
| `time` | Simulation time. |
| `time_step_size` | Time-step size. |
| `flowrates` | Flow rate per conduit. |
| `velocities` | Velocity per conduit. |
| `water_depths` | Water depth per node. |
| `y_l2_norms` | Relative water-depth L2 norm. |
| `Q_l2_norms` | Relative flow-rate L2 norm. |
| `convergence_fails` | Cumulative convergence failures. |
| `reynolds_numbers` | Reynolds number per conduit. |
| `picard_iterations` | Picard iterations at stored time. |
| `picard_iterations_total` | Cumulative Picard iterations. |

Only keys set to `True` are stored.

## Logging settings

```python
logging_settings = {
    "base_dir": "runs/example_001",
}
```

Pass `logging_settings` when creating a `FlowSimulation` object:

```python
flow = FlowSimulation(
    network,
    simulation_settings=simulation_settings,
    logging_settings=logging_settings,
)
```

| Parameter | Meaning |
| --- | --- |
| `base_dir` | Base directory where the `logs/` folder is created. Defaults to the current working directory. |
| `log_file` | Optional log filename. If omitted, openKARST creates a timestamped file such as `simulation_20260709_091530_482317.log`. If provided, openKARST appends to that named file. |

For most runs, set only `base_dir`. This creates one log file per simulation
inside `<base_dir>/logs/`. Use `log_file` when several runs should append to a
specific shared log:

```python
logging_settings = {
    "base_dir": "runs/example_001",
    "log_file": "simulation.log",
}
```

The normal log records setup and run-level information: validated settings,
network summary, boundary conditions, observation recorders, requested outputs,
stop reason, final convergence totals, and stored result count. Timestep
progress controlled by `print_info_interval` is printed to the console and is
not written to the normal log. If the time history of convergence failures is
needed, request `convergence_fails` in `outputs`.
