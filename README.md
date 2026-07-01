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
  <img src="https://img.shields.io/badge/version-0.4.0-blue.svg" alt="version">
  <a href="https://erc-karst.github.io/openkarst/">
    <img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="docs">
  </a>
</p>

<p align="center">
  <a href="https://openkarst.org">Project Website → openkarst.org</a>
</p>

openKARST is a Python package for modeling and simulating flow in karst networks. It provides tools for setting up, running, and visualizing flow simulations.

## Features

- Steady-state and transient karst conduit flow simulation using the Saint-Venant equations
- Dynamic switching between free-surface and pressurized flow
- Circular analytical, tabulated depth-width, and channel-style cross-section geometry support
- Churchill and hybrid Churchill/Manning friction formulations for laminar, turbulent and mixed flow regimes
- Constant, box, and time-series boundary conditions for inflow, water depth, springs
- Flexible observation recorders for selected node time series
- Result outputs for flow rates, water depths, Reynolds numbers, transport variables, reservoirs, convergence norms, and solver iteration counts
- Browser-based Plotly/Dash viewer for 3D inspection, time navigation, observation plots, convergence diagnostics, and Colab/browser use
- Built-in cave data loading utilities, including packaged example cave datasets
- VTK export for external visualization workflows such as ParaView
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
