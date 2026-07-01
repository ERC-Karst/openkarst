# Overview

openKARST simulates flow through karst conduit networks using a dynamic-wave
formulation. A model combines a graph-like conduit geometry, hydraulic
properties, numerical settings, initial conditions, and boundary conditions.

## Mental model

An openKARST simulation can be read as:

```text
Where can water move?
  Network geometry

What are the conduit properties?
  Lengths, diameters, roughness, and elevations

How should the solver behave?
  Time step, Courant number, Picard iteration settings

How is the system forced?
  Inflow and water-depth boundary conditions

What should be saved?
  Result arrays and observation time series
```

## Main objects

| Object | Role |
| --- | --- |
| OpenPNM network | Provides a graph object and stores node and conduit properties. It is not used for hydraulic computations. |
| `FlowSimulation` object | Holds the hydraulic state and advances the model in time. |
| Boundary conditions | Prescribe inflow or water depth at selected nodes. |
| Results container | Stores selected arrays at output intervals. |
| Observation recorders | Store node-based time series for selected variables and selected node groups. |

## Units

Use SI units consistently:

| Quantity | Unit |
| --- | --- |
| Length, diameter, water depth | m |
| Time | s |
| Volumetric flow rate | m^3/s |
| Water density | kg/m^3 |
| Dynamic viscosity | Pa s |
| Gravity | m/s^2 |

## Modeling scope

openKARST is designed for conduit-network flow simulations. It supports
transient and steady-state style runs, circular conduit geometries, channel
geometry for analytical comparisons, and browser-based inspection of results.
Development is actively extending the modeling framework toward coupled
conduit-aquifer simulations and physically based infiltration models.
