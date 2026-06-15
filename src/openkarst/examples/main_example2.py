#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:56:06 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import os

import numpy as np

from openkarst.io.cave_data_loader import CaveDataLoader
from openkarst.models import FlowSimulation
from openkarst.visualization.openkarst_viewer import launch_openkarst_viewer


def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup flow simulation parameters
    physical_properties = {
        'water_density': 1000,        # kg/m^3
        'gravity': 9.81,              # m/s^2
        'dynamic_viscosity': 0.001,   # Pa.s (kg/m.s)
        'geometry_channel': False,    # Channel geometry for analytical solutions (Default False)
        'channel_type': 'infinite',   # 'infinite' for infinitely wide channel, 'finite' for defined width
        'channel_width': 1.0,         # Width of the channel (only used if channel_type is 'finite')
    }
    
    solver_settings = {
        'relaxation_factor': 0.6,    # Dimensionless
        'max_iterations': 20,        # Maximum Picard iterations
        'picard_depth_tol': 1e-4,    # Picard depth tolerance (meters)
        'ss_rel_l2tol': 1e-3         # L2 tolerance for steady-state
    }
    
    simulation_settings = {
        'min_waterdepth': 1e-12,      # Minimum water depth (meters)
        'min_flowrate': 1e-12,        # Minimum flow rate (m^3/s)
        'courant': 0.8,               # Courant number
        'adaptive_timesteps': True,   # Use adaptive timestepping
        'dt_init': 0.01,             # Initial (or constant) timestep (seconds)
        'dt_max': 0.1,                # Maximum allowable time step
        'steady_state': False,         # Steady-state (True) or transient (False)
        't_max': 2000.0,              # Maximum time for transient simulations (seconds)
        'print_info_interval': 1000,     # Print info every # time steps
    }
    
    output_settings = {
        'output_interval': 1.0,
        'time': True,
        'time_step_size': True,
        'flowrates': True,
        'water_depths': True,
        'y_l2_norms': True,
        'Q_l2_norms': True,
        'convergence_fails': True,
        'reynolds_numbers': True,
    }
    
    logging_settings = {
        'base_dir': base_dir,
        'log_file': 'simulation.log'
    }
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the nodes and conduit files
    nodes_file_path = os.path.join(current_dir, '../cave_data/reve_eveille/nodes.csv')
    edges_file_path = os.path.join(current_dir, '../cave_data/reve_eveille/edges.csv')
    diameters_file_path = os.path.join(current_dir, '../cave_data/reve_eveille/diameters.csv')

    # Initialize CaveDataLoader and load data
    loader = CaveDataLoader(nodes_file_path, edges_file_path, diameters_file_path)
    cn_geometry = loader.load_cave_data()
  
    # Assign conduit properties
    cn_geometry['throat.epsilon'] = 0.03
   

    # Create flow network object
    flow_network = FlowSimulation(cn_geometry,
                                  physical_properties = physical_properties,
                                  solver_settings = solver_settings,
                                  simulation_settings = simulation_settings,
                                  logging_settings = logging_settings)
    
    # Set initial conditions
    initial_Q = np.full(cn_geometry.Nt, 0.0, dtype=float)   # Initial flows at each conduit (Nt throats)
    initial_y = np.full(cn_geometry.Np, 0.01, dtype=float)  # Initial water depths at each node (Np pores)    
    flow_network.set_initial_conditions(initial_Q, initial_y)
    
    # Set boundary conditions
    outflow_nodes = [21]
    inflow_nodes = [71]

    # Define constant BC values
    flowrate = 0.01       # Volumetric inflow in m³/s
    water_depth = 0.01    # Constant water depth in m

    # Apply constant volumetric inflow at inflow nodes
    flow_network.set_inflow_BC(
        nodes=inflow_nodes,
        values=flowrate,          # Single float, default constant BC
        mode='add',               # Default
        inflow_type='volumetric'  # Default
    )

    # Apply constant water depth at outflow nodes
    flow_network.set_waterdepth_BC(
        nodes=outflow_nodes,
        values=water_depth,       # Single float, default constant BC
        mode='add'                # Default
    )

    # Record inlet and outlet time series for the viewer observation panel
    flow_network.set_observation_points(
        nodes=inflow_nodes + outflow_nodes,
        variables=['water_depth', 'inflow'],
        interval=output_settings['output_interval']
    )

    # Run simulation and store results
    results = flow_network.run_simulation(desired_outputs = output_settings)

    obs_df = flow_network.get_observation_dataframe()

    launch_openkarst_viewer(results, cn_geometry, obs_df)

    return results, cn_geometry, obs_df

if __name__ == '__main__':
    main()
    input("Viewer is running at http://127.0.0.1:8050. Press Enter to stop.\n")
