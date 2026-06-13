# Parameter Reference

This section provides a detailed overview of the input parameters required for setting up simulations, including their allowed ranges and types. The parameters are validated using the following criteria:

## Physical Properties

### `water_density`
- **Type:** `int` or `float`
- **Description:** Density of water.
- **Allowed Range:** `> 0`
- **Default:** `1000.0`

### `gravity`
- **Type:** `int` or `float`
- **Description:** Gravitational acceleration.
- **Allowed Range:** `> 0`
- **Default:** `9.81`

### `dynamic_viscosity`
- **Type:** `int` or `float`
- **Description:** Dynamic viscosity of water.
- **Allowed Range:** `> 0`
- **Default:** `0.001`

### `geometry_channel`
- **Type:** `bool`
- **Description:** Whether to use a channel geometry (e.g. for analytical solutions).
- **Allowed Values:** `True` or `False`
- **Default:** `False`

### `channel_type`
- **Type:** `str`
- **Description:** Type of channel geometry (in terms of width).
- **Allowed Values:** `"finite"` or `"infinite"`
- **Condition:** Required if `geometry_channel` is `True`
- **Default:** `"finite"`

### `channel_width`
- **Type:** `int` or `float`
- **Description:** Width of the channel.
- **Allowed Range:** `> 0`
- **Condition:** Required if `channel_type` is `"finite"`
- **Default:** `1.0`

### `channel_manning`
- **Type:** `int` or `float`
- **Description:** Manning's roughness coefficient for the channel. Roughness of conduits is controlled via the epsilon roughness factor as part of the geometry object.
- **Allowed Range:** `>= 0`
- **Condition:** Required if `geometry_channel` is `True`
- **Default:** `0.03`

## Geometry Settings

### `backend`
- **Type:** `str`
- **Description:** Closed-conduit cross-section geometry backend used when `geometry_channel` is `False`.
- **Allowed Values:** `"circular_analytical"` or `"circular_tabulated"`
- **Default:** `"circular_analytical"`

### `table_points`
- **Type:** `int`
- **Description:** Number of normalized depth points used by tabulated geometry backends.
- **Allowed Range:** `>= 2`
- **Default:** `1000`

## Solver Settings

### `relaxation_factor`
- **Type:** `int` or `float`
- **Description:** Relaxation factor for the solver.
- **Allowed Range:** `0 < value <= 1`
- **Default:** `0.6`

### `max_iterations`
- **Type:** `int`
- **Description:** Maximum number of Picard iterations.
- **Allowed Range:** `10 < value <= 1000`
- **Default:** `20`

### `picard_depth_tol`
- **Type:** `int` or `float`
- **Description:** Picard depth tolerance.
- **Allowed Range:** `1e-11 < value <= 1e-2`
- **Default:** `1e-9`

### `ss_rel_l2tol`
- **Type:** `int` or `float`
- **Description:** L2 tolerance for steady-state.
- **Allowed Range:** `1e-8 < value <= 1e-2`
- **Default:** `1e-7`

## Simulation Settings

### `min_waterdepth`
- **Type:** `int` or `float`
- **Description:** Minimum water depth.
- **Allowed Range:** `1e-12 <= value < 1e-5`
- **Default:** `1e-10`

### `min_flowrate`
- **Type:** `int` or `float`
- **Description:** Minimum flow rate.
- **Allowed Range:** `1e-12 <= value < 1e-5`
- **Default:** `1e-10`

### `courant`
- **Type:** `int` or `float`
- **Description:** Courant number. Lower values may help to achieve convergence.
- **Allowed Range:** `0 < value <= 2`
- **Default:** `0.5`

### `adaptive_timesteps`
- **Type:** `bool`
- **Description:** Whether to use adaptive time stepping.
- **Allowed Values:** `True` or `False`
- **Default:** `False`

### `dt_init`
- **Type:** `int` or `float`
- **Description:** This value is the constant time step size when no adaptive time stepping is not used. Otherwise it is the initial time step size, subject to modification by the adaptive algorithm.
- **Condition:** Required and must be greater than 0 when `adaptive_timesteps` is `False` or `True`
- **Default:** `-`

### `dt_max`
- **Type:** `int` or `float`
- **Description:** Maximum allowable time step.
- **Condition:** Required and must be greater than 0 when `adaptive_timesteps` is `True`
- **Default:** `1.0`

### `steady_state`
- **Type:** `bool`
- **Description:** Steady-state (True) or transient (False). Simulations are run until errors are below the L2 tolerance.
- **Allowed Values:** `True` or `False`
- **Default:** `False`

### `t_max`
- **Type:** `int` or `float`
- **Description:** Maximum time for transient simulations.
- **Condition:** Required and must be greater than 0 for transient simulations (`steady_state` is `False`)
- **Default:** `0.0`

### `print_info_interval`
- **Type:** `int`
- **Description:** Print info every # time steps.
- **Allowed Range:** `>= 1`
- **Default:** `1`

## Output Settings

### `output_interval`
- **Type:** `int` or `float`
- **Description:** Interval for outputting simulation data ( in seconds).
- **Allowed Range:** `> 0`
- **Default:** `1.0`

### `time`
- **Type:** `bool`
- **Description:** Whether to output time data.
- **Allowed Values:** `True` or `False`
- **Default:** `True`

### `time_step_size`
- **Type:** `bool`
- **Description:** Whether to output time step size data.
- **Allowed Values:** `True` or `False`
- **Default:** `True`

### `flowrates`
- **Type:** `bool`
- **Description:** Whether to output flow rate data.
- **Allowed Values:** `True` or `False`
- **Default:** `True`

### `water_depths`
- **Type:** `bool`
- **Description:** Whether to output water depth data.
- **Allowed Values:** `True` or `False`
- **Default:** `True`

### `l2_norms`
- **Type:** `bool`
- **Description:** Whether to output L2 norm data.
- **Allowed Values:** `True` or `False`
- **Default:** `True`

### `convergence_fails`
- **Type:** `bool`
- **Description:** Whether to output convergence failure data.
- **Allowed Values:** `True` or `False`
- **Default:** `True`

### `reynolds_numbers`
- **Type:** `bool`
- **Description:** Whether to output Reynolds number data.
- **Allowed Values:** `True` or `False`
- **Default:** `True`
