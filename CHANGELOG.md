# Changelog

## [0.4.0] - 2026-07-01
### Added
- Preliminary dynamic unconfined reservoir support coupled to conduit nodes, including
  constant or time-series recharge, reservoir storage, hydraulic head, and
  reservoir-node exchange tracking.
- Reservoir result outputs for `run_simulation()`, including reservoir nodes,
  water depths, heads, storage, exchange rates, and recharge rates.
- Reservoir observation variables for `set_observation_points()`, including
  reservoir water depth, head, storage, exchange, and recharge.
- Multiple observation recorders for mixed reservoir and non-reservoir node
  groups, with optional recorder names and separate recorder-table retrieval.
- Combined observation dataframe output compatible with the openKARST viewer.
- Head-dependent spring boundary condition support.
- Cross-section geometry backends for circular analytical geometry and
  tabulated depth-width geometry with optional diameter scaling.
- Signed connected-flow observation output with `connected_net_flowrate`.
- Whole-simulation diagnostic outputs for water-depth and discharge L2 norms,
  Picard iterations, and total Picard iterations.
- Fundamental pytest suite
- Restructured manual with new getting started section, general concepts, howto guides,
  tutorials and viewer documentation.

### Changed
- Viewer layout and controls were expanded with standard 3D views, Google Colab
  support, observation and property selection, convergence plots, log-scale display
  options, and improved legend behavior.
- Saint-Venant settings fowarding simplified
- Initial refactoring of into hydraulic helper routines
- Cave data loading was simplified via single public loading function.
- Example scripts and documentation were updated to use the current viewer,
  observation, geometry, and settings workflows.

### Fixed
- Preissmann slot handling no longer affects discharge area and hydraulic
  radius calculations.
- User supplied `dt_init` is also used as the lower limit for adaptive time
  stepping.
- Steady-state convergence checks now include discharge as well as water depth.
- Scalar inflow boundary conditions preserve their flux or volumetric type.
- Observation flowrate outputs now distinguish absolute connected flowrate from
  signed net connected flowrate.

### Removed
- Legacy PyVista example documentation and outdated parameter reference pages
- Legacy MAD norm setting and related steady-state documentation.

## [0.3.0] - 2025-11-20
### Added
- New boundary condition class (constant, box, timeseries) with stationary and transient modes
- New friction factor modes (pure Churchill or Churchill/Manning hybrid)
- New observation recorder class to set observation nodes
- New BC checks to avoid wrong combinations
- New plotly/Dash browser-based visualization tool
- Preliminary AD transport model
- Code efficiency (vectorized and cached BC interpolation for timeseries)

## [0.1.0] - 2024-07-25
### Added
- Initial release of openkarst
- Basic flow simulation setup
- Example scripts for usage
