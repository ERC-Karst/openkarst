# Networks and geometry

openKARST uses OpenPNM to create the network graph and store geometry and
property arrays. OpenPNM is not used for the hydraulic or numerical
computations performed by openKARST.

In OpenPNM terminology:

- **pores** are network nodes.
- **throats** are connections between nodes.
- In openKARST, throats correspond to **conduits**.

## Required geometry fields

For conduit-flow simulations, the network should contain:

| Field | Location | Meaning |
| --- | --- | --- |
| `pore.coords` | node | 3D coordinates of each node. |
| `throat.conns` | conduit | Node pairs connected by each conduit. |
| `throat.lengths` | conduit | Conduit length in m. |
| `throat.diameters` | conduit | Conduit diameter in m. |
| `throat.epsilon` | conduit | Roughness height in m. |

## Linear network example

```python
import openpnm as op

from openkarst.network_generation import compute_conduit_lengths

network = op.network.Cubic(shape=[20, 1, 1], connectivity=6, spacing=1.0)
network = compute_conduit_lengths(network)
network["throat.diameters"] = 1.0
network["throat.epsilon"] = 0.03
```

## Elevation

Node elevation is taken from the third coordinate:

```python
z = network["pore.coords"][:, 2]
```

To add a sloping conduit, modify the z-coordinate before creating the
simulation:

```python
network["pore.coords"][:, 2] = -0.01 * network["pore.coords"][:, 0]
```

## Cave CSV data

`load_cave_data()` reads three CSV files:

| File | Content |
| --- | --- |
| nodes | node id and x/y/z coordinates |
| edges | connected node pairs |
| diameters | cross-section width and height per node |

The loader creates an OpenPNM network, computes conduit lengths, and assigns
average conduit diameters.
