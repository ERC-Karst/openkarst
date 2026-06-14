# Simulation Parameters

This page summarizes the hydraulic simulation parameters used by `FlowSimulation`
and `run_simulation`. It reflects the current code defaults and validation rules.

Omitted dictionary entries use their dataclass defaults, but some defaults are
placeholders that must be replaced for a valid run. In particular, transient
simulations require `t_max > 0`, and all simulations require a numeric
`dt_init` and `dt_max`.

## Physical Properties

Passed to `FlowSimulation` as `physical_properties={...}`.

| Parameter | Type | Description | Valid values | Default |
| --- | --- | --- | --- | --- |
| `water_density` | `int` or `float` | Density of water. | `> 0` | `1000.0` |
| `gravity` | `int` or `float` | Gravitational acceleration. | `> 0` | `9.81` |
| `dynamic_viscosity` | `int` or `float` | Dynamic viscosity of water. | `> 0` | `0.001` |
| `geometry_channel` | `bool` | Whether to use open-channel geometry, mainly for analytical channel examples. | `True` or `False` | `False` |
| `channel_type` | `str` | Width treatment for open-channel geometry. Only used when `geometry_channel=True`. | `"finite"` or `"infinite"` | `"finite"` |
| `channel_width` | `int` or `float` | Channel width. Required when `geometry_channel=True` and `channel_type="finite"`. | `> 0` | `1.0` |
| `channel_manning` | `int` or `float` | Manning roughness coefficient for open-channel geometry. Only used when `geometry_channel=True`. | `>= 0` | `0.03` |
| `friction_model` | `str` | Closed-conduit friction model. Used when `geometry_channel=False`; ignored when `geometry_channel=True`. | `"hybrid"` or `"churchill"` | `"churchill"` |

## Solver Settings

Passed to `FlowSimulation` as `solver_settings={...}`.

| Parameter | Type | Description | Valid values | Default |
| --- | --- | --- | --- | --- |
| `relaxation_factor` | `int` or `float` | Relaxation factor for Picard iterations. | `0 < value <= 1` | `0.6` |
| `max_iterations` | `int` | Maximum number of Picard iterations. | `10 < value <= 1000` | `500` |
| `picard_depth_tol` | `int` or `float` | Picard depth convergence tolerance. | `1e-11 < value <= 1e-2` | `1e-9` |
| `ss_rel_l2tol` | `int` or `float` | Relative L2 tolerance for steady-state convergence. | `1e-10 < value <= 1e-2` | `1e-3` |
| `ss_rel_madtol` | `int` or `float` | Relative mean absolute difference tolerance for steady-state convergence. | `1e-10 < value <= 1e-2` | `1e-7` |

## Simulation Settings

Passed to `FlowSimulation` as `simulation_settings={...}`.

| Parameter | Type | Description | Valid values | Default |
| --- | --- | --- | --- | --- |
| `min_waterdepth` | `int` or `float` | Minimum water depth used for numerical stability. | `1e-14 <= value < 1e-5` | `1e-10` |
| `min_flowrate` | `int` or `float` | Minimum flow rate used for numerical stability. | `0 <= value < 1e-5` | `1e-10` |
| `courant` | `int` or `float` | Courant number. Lower values can help convergence. | `0 < value <= 2` | `0.5` |
| `adaptive_timesteps` | `bool` | Whether to adapt the time step during the simulation. | `True` or `False` | `False` |
| `dt_init` | `int` or `float` | Constant time-step size when `adaptive_timesteps=False`; initial time-step size when `adaptive_timesteps=True`. | Required; `> 0` | `None` |
| `dt_max` | `int` or `float` | Maximum time-step size used by adaptive time stepping. A numeric value is currently required by validation. | Numeric value | `None` |
| `steady_state` | `bool` | Whether to run until steady-state convergence instead of a fixed end time. | `True` or `False` | `False` |
| `t_max` | `int` or `float` | Maximum simulation time for transient simulations. | Required and `> 0` when `steady_state=False` | `0.0` |
| `print_info_interval` | `int` | Number of time steps between printed progress messages. | `>= 1` | `1` |

## Output Settings

Passed to `run_simulation` as `desired_outputs={...}`. Boolean output keys are
stored only when explicitly set to `True`; omitted keys and keys set to `False`
are not stored. Unknown keys raise a `ValueError`.

| Parameter | Type | Description | Valid values | Default |
| --- | --- | --- | --- | --- |
| `output_interval` | `int` or `float` | Simulation-time interval between stored result snapshots. | Positive value expected | `1.0` |
| `time` | `bool` | Store simulation time. | `True` or `False` | Not stored |
| `time_step_size` | `bool` | Store time-step size. | `True` or `False` | Not stored |
| `flowrates` | `bool` | Store conduit flow rates. | `True` or `False` | Not stored |
| `velocities` | `bool` | Store conduit velocities. | `True` or `False` | Not stored |
| `water_depths` | `bool` | Store nodal water depths. | `True` or `False` | Not stored |
| `l2_norms` | `bool` | Store relative L2 error norms. | `True` or `False` | Not stored |
| `mad_norms` | `bool` | Store relative mean absolute difference error norms. | `True` or `False` | Not stored |
| `convergence_fails` | `bool` | Store the convergence failure count. | `True` or `False` | Not stored |
| `reynolds_numbers` | `bool` | Store conduit Reynolds numbers. | `True` or `False` | Not stored |
| `picard_iterations` | `bool` | Store the Picard iteration count from each stored time step. | `True` or `False` | Not stored |

## Logging Settings

Passed to `FlowSimulation` as `logging_settings={...}`.

| Parameter | Type | Description | Valid values | Default |
| --- | --- | --- | --- | --- |
| `base_dir` | `str` | Base directory where the `logs` folder is created. | Non-empty string | Current working directory |
| `log_file` | `str` | Log filename inside the `logs` folder. | Non-empty string | `"simulation.log"` |
