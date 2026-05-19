<p align="center">
  <img src="assets/openkarst_header_color.png" alt="openKARST logo" width="420">
</p>

<p align="center">
  <strong>Open-source flow simulation for karst conduit networks</strong>
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.cageo.2025.106066">
    <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.cageo.2025.106066-blue" alt="DOI">
  </a>
  <img src="https://img.shields.io/badge/version-0.3.0-blue.svg" alt="version">
  <a href="https://erc-karst.github.io/openkarst/">
    <img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="docs">
  </a>
</p>

<p align="center">
  <a href="https://openkarst.org">Project Website → openkarst.org</a>
</p>

openKARST is a Python package for modeling and simulating flow in karst networks. It provides tools for setting up, running, and visualizing flow simulations.

## Features

- Steady-state and transient flow simulation in karst networks using the Saint-Venant equation (dynamic wave equation) 
- Dynamic switching between free surface and pressurized flows
- Laminar and turbulent flow dynamics
- Two friction factor modes (Churchill and Churchill/Manning hybrid)
- Support for circular conduits and free surface channels
- Data loader for ERC KARST cave network files
- Data exporter for VTK files (e.g. Visualization via Paraview)
- Browser-based Plotly/Dash data visualization
- Compatibility with OpenPNM and NetworkX for network creation and manipulation

## Documentation

For full documentation, visit the [openKARST Documentation](https://erc-karst.github.io/openkarst/).


## Contributing

Contributions are welcome! Please read the [contributing.md](docs/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Acknowledgments

- The [OpenPNM](https://github.com/PMEAL/OpenPNM) library for pore network modeling.
- The [Plotly/Dash](https://github.com/plotly/dash) library for 3D visualization.

## Citation

If you use openKARST in your research, please cite:

Kordilla, J., et al. (2026). *openKARST: A novel open-source flow simulator for karst systems*. Computers & Geosciences, 106066.  
https://doi.org/10.1016/j.cageo.2025.106066
