# Export to VTK

Use VTK export when you want to inspect results in ParaView or another
scientific visualization tool.

## Basic export

```python
import os

from openkarst.io import VtkDataExporter

results = flow.run_simulation(desired_outputs=outputs)

Q_history = results["flowrates"]
y_history = results["water_depths"]
t_history = results["time"]

output_dir = os.path.join(os.getcwd(), "vtk_output")
vtk_exporter = VtkDataExporter(output_dir)
vtk_exporter.export(network, Q_history, y_history, t_history)
```

## Mark inlet and outlet nodes

```python
vtk_exporter.export(
    network,
    Q_history,
    y_history,
    t_history,
    inflow_nodes=[0],
    outflow_nodes=[19],
)
```

This also writes a static marker file for inlet and outlet nodes.

## Exported fields

| Field | VTK location |
| --- | --- |
| Flow rate | conduit/cell data |
| Water depth | node/point data |
| Roughness | conduit/cell data, if available |
| Diameter | conduit/cell data, if available |
| Time | field data |

## ParaView workflow

1. Open the generated `output_timestep_*.vtk` files.
2. Apply time-aware loading if ParaView detects the sequence.
3. Color conduits by `Flowrate`.
4. Color nodes or points by `WaterDepth`.
5. Load `static_node_markers.vtk` if inlet/outlet labels are helpful.
