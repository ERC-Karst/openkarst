# Glossary

| Term | Meaning |
| --- | --- |
| Boundary condition | A prescribed value that forces the model, such as inflow or water depth at a node. |
| Conduit | A hydraulic connection between two nodes. In OpenPNM this is a throat. |
| Courant number | Stability-related control for time-step selection. |
| Flow rate | Volumetric discharge through a conduit, usually in m^3/s. |
| Hydraulic head | Elevation plus water depth. |
| Inflow node | Node where water enters the model through a prescribed boundary. |
| Manning coefficient | Roughness coefficient often used for open-channel flow. |
| Node | A point in the network. In OpenPNM this is a pore. |
| Observation point | A node where selected time series are recorded during simulation. |
| Outlet node | Node where a water-depth or other outlet condition is prescribed. |
| Picard iteration | Iterative nonlinear solve used within a time step. |
| Pore | OpenPNM term for a network node. |
| Stationary simulation | Simulation that advances until a steady hydraulic state is reached rather than stopping at a prescribed final time. |
| Throat | OpenPNM term for a connection between two pores; openKARST treats this as a conduit. |
| Transient simulation | Simulation that evolves through time until `t_max`. |
| Water depth | Depth of water at a node, measured from the local conduit or channel base. |
