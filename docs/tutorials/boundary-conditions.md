# 2. Boundary conditions

Boundary conditions define how water enters the network and how the hydraulic
state is constrained at selected nodes. openKARST provides two principal
hydraulic boundary types:

| Boundary type | Method | Prescribed quantity |
| --- | --- | --- |
| Inflow | `set_inflow_BC()` | Volumetric flow rate in m^3/s, or flux in m/s for channel geometries |
| Water depth | `set_waterdepth_BC()` | Water depth in m |

Both methods accept one node or several nodes, and both support constant,
box-shaped, and time-series values.

## Assigning nodes and values

Pass a single node as an integer:

```python
flow.set_inflow_BC(nodes=0, values=0.01)
flow.set_waterdepth_BC(nodes=19, values=0.01)
```

For several nodes, a scalar or boundary definition is applied to every node:

```python
flow.set_inflow_BC(nodes=[0, 3, 7], values=0.01)
```

Alternatively, pass one value or definition per node:

```python
flow.set_waterdepth_BC(
    nodes=[18, 19],
    values=[0.01, 0.02],
)
```

The number of entries in `values` must match the number of entries in `nodes`
when node-specific values are used.

## Supported value definitions

The `values` argument has the same structure for inflow and water-depth
boundaries:

| Form | Definition | Behavior |
| --- | --- | --- |
| Constant | `0.01` | Holds one value throughout the simulation. |
| Box | `("box", value, t_start, t_end)` | Uses `value` between the start and end times and zero outside the interval. |
| Box with outer values | `("box", value, t_start, t_end, before, after)` | Uses explicit values before and after the interval. |
| Time series | `("timeseries", times, values)` | Linearly interpolates between supplied points. |

### Constant boundaries

```python
flow.set_inflow_BC(
    nodes=0,
    values=0.01,
    inflow_type="volumetric",
)

flow.set_waterdepth_BC(
    nodes=19,
    values=0.01,
)
```

For channel geometries, inflow may instead be defined as an area-normalized
flux:

```python
flow.set_inflow_BC(
    nodes=0,
    values=1e-5,
    inflow_type="flux",
)
```

### Box profiles

The following inflow is `0.04 m^3/s` from `t = 100 s` through `t = 300 s` and
zero before and after:

```python
flow.set_inflow_BC(
    nodes=0,
    values=("box", 0.04, 100.0, 300.0),
)
```

The same form can prescribe a time-dependent water depth. Here the depth is
`0.10 m` before the interval, `0.20 m` during it, and `0.15 m` afterward:

```python
flow.set_waterdepth_BC(
    nodes=19,
    values=("box", 0.20, 100.0, 300.0, 0.10, 0.15),
)
```

### Time series

Time-series boundaries are linearly interpolated:

```python
times = [0.0, 50.0, 100.0, 200.0]
inflow = [0.0, 0.03, 0.06, 0.02]

flow.set_inflow_BC(
    nodes=0,
    values=("timeseries", times, inflow),
    extrapolate="zero",
)
```

They can also be used for water depth:

```python
times = [0.0, 100.0, 200.0]
depth = [0.10, 0.15, 0.12]

flow.set_waterdepth_BC(
    nodes=19,
    values=("timeseries", times, depth),
    extrapolate="hold",
)
```

`extrapolate="hold"` uses the first or last value outside the supplied time
range. `extrapolate="zero"` uses zero instead.

## Replacing and removing boundaries

The default `mode="add"` prevents a second boundary of the same type from being
assigned to an existing node. Use `mode="overwrite"` to replace it:

```python
flow.set_inflow_BC(nodes=0, values=0.02, mode="overwrite")
```

Use `mode="remove"` to remove it. The value is ignored in this mode:

```python
flow.set_inflow_BC(nodes=0, values=None, mode="remove")
flow.set_waterdepth_BC(nodes=19, values=None, mode="remove")
```

## Consistency checks

Each boundary node should have one hydraulic role. A typical model prescribes
inflow at source nodes and water depth at separate outlet or control nodes.
Before a simulation starts, openKARST checks for incompatible combinations,
including inflow and prescribed water depth on the same node, and raises a
`ValueError` that identifies the affected nodes.
