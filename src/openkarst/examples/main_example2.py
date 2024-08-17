#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:56:06 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Needs pip install imageio-ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/jkordil_idaea/Downloads/ffmpeg" 

from openkarst.io.cave_data_loader import CaveDataLoader
from openkarst.models import FlowSimulation
from openkarst.visualization.animation_pyvista import animate_network


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
        'ss_rel_l2tol': 1e-3,         # L2 tolerance for steady-state
        'ss_rel_madtol': 1e-8         # Median tolerance for steady-state
    }
    
    simulation_settings = {
        'min_waterdepth': 1e-12,      # Minimum water depth (meters)
        'min_flowrate': 1e-12,        # Minimum flow rate (m^3/s)
        'courant': 0.8,               # Courant number
        'adaptive_timesteps': True,   # Use adaptive timestepping
        'dt_init': 0.0001,             # Initial (or constant) timestep (seconds)
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
        'l2_norms': True,
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
    
    flowrate  = 0.01     # Flowrate in m^3/s
    water_depth = 0.01  # Water depths in m
    
    inflow_boundary_left = {node: flowrate for node in inflow_nodes}
    waterdepth_boundary_right = {node: water_depth for node in outflow_nodes}
    
    flow_network.set_boundary_conditions(
    inflow_boundary=inflow_boundary_left,
    inflow_type='constant',
    )
    
    flow_network.set_boundary_conditions(waterdepth_boundary=waterdepth_boundary_right)
    
    # Run simulation and store results
    results = flow_network.run_simulation(desired_outputs = output_settings)
    
    # Get arrays from results container
    Q_history = results['flowrates']
    y_history = results['water_depths']
    t_history = results['time']
    
    
    animation_settings = {
        'update_interval': 1,
        'conduit_plotradius': 0.5,
        'bar_plotradius': 0.5,
        'node_plotsize': 5,
        'depthscaling': 20,
        'fig_width': 1600,
        'fig_height': 800,
        'zoom_factor': 1.0,
        'background_color': 'black',
        'isometric_view': False,
        'create_animation': False,
        'filename': "network_animation2.mp4"
    }
    
    animate_network(cn_geometry=cn_geometry, 
                    Q_history=Q_history, 
                    y_history=y_history, 
                    t_history=t_history, 
                    **animation_settings)
    
    return results, cn_geometry

if __name__ == '__main__':
    results, cn_geometry = main()
    
# Get arrays from results container
Q_history = results['flowrates']
y_history = results['water_depths']
t_history = results['time']

# Extract flow rates at node 21
node_id = 21
connected_throats = cn_geometry.find_neighbor_throats(pores=[node_id])
flowrates_at_node = Q_history[:, connected_throats]
total_flowrates_at_node = np.sum(flowrates_at_node, axis=1)

# Plot the outflow at the conduit connected to the boundary node
plt.figure()
plt.plot(t_history, total_flowrates_at_node, label=f'Node {node_id}')
plt.xlabel('Time (s)')
plt.ylabel('Flowrate (m^3/s)')
plt.title(f'Flowrate at Node {node_id} Over Time')
plt.legend()
plt.show()