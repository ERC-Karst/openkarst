#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:56:06 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import sys
import os

import openpnm as op
import numpy as np

# Needs pip install imageio-ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/jkordil_idaea/Downloads/ffmpeg" 

from openkarst.visualization.animation_pyvista import animate_network
from openkarst.models import FlowSimulation


def main():
    
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
        'picard_depth_tol': 1e-7,    # Picard depth tolerance (meters)
        'ss_rel_l2tol': 1e-7,        # L2 tolerance for steady-state
    }
    
    simulation_settings = {
        'min_waterdepth': 1e-10,      # Minimum water depth (meters)
        'min_flowrate': 1e-10,        # Minimum flow rate (m^3/s)
        'courant': 0.8,               # Courant number
        'adaptive_timesteps': True,   # Use adaptive timestepping
        'dt_init': 0.001,             # Initial (or constant) timestep (seconds)
        'dt_max': 1.0,                # Maximum allowable time step
        'steady_state': False,         # Steady-state (True) or transient (False)
        't_max': 1000.0,              # Maximum time for transient simulations (seconds)
        'print_info_interval': 1000,     # Print info every # time steps
    }
    
    output_settings = {
        'output_interval': 10.0,
        'time': True,
        'time_step_size': True,
        'flowrates': True,
        'water_depths': True,
        'l2_norms': True,
        'convergence_fails': True,
        'reynolds_numbers': True,
    }
    
    # Create network object using OpenPNM
    dl = 1 # Constant spacing between nodes (meters)
    cn_geometry = op.network.Cubic(shape=[200, 1, 1], connectivity=6, spacing=dl)
    
    # Compute and assign conduit lengths 
    coords_diff = np.diff(cn_geometry.coords[cn_geometry.conns], axis=1).squeeze()
    squared_diffs = coords_diff**2
    sum_squared_diffs = np.sum(squared_diffs, axis=1)
    conduit_lengths = np.sqrt(sum_squared_diffs)
    cn_geometry['throat.lengths'] = conduit_lengths
    
    # Assign conduit properties
    cn_geometry['throat.epsilon'] = 0.03
    cn_geometry['throat.diameters'] = 1.0
       
    # Create flow network object
    flow_network = FlowSimulation(cn_geometry,
                                  physical_properties = physical_properties,
                                  solver_settings = solver_settings,
                                  simulation_settings = simulation_settings)
    
    # Set initial conditions
    initial_Q = np.full(cn_geometry.Nt, 0.0, dtype=float)   # Initial flows at each conduit (Nt throats)
    initial_y = np.full(cn_geometry.Np, 0.01, dtype=float)  # Initial water depths at each node (Np pores)    
    flow_network.set_initial_conditions(initial_Q, initial_y)
    
    # Set boundary conditions
    right_nodes = [199]
    left_nodes = [0]
    
    flowrate  = 0.1     # Flowrate in m^3/s
    water_depth = 0.01  # Water depths in m
    
    inflow_boundary_left = {node: flowrate for node in left_nodes}
    waterdepth_boundary_right = {node: water_depth for node in right_nodes}
    
    flow_network.set_boundary_conditions(inflow_boundary = inflow_boundary_left)
    flow_network.set_boundary_conditions(waterdepth_boundary = waterdepth_boundary_right)
    
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

if __name__ == '__main__':
    main()