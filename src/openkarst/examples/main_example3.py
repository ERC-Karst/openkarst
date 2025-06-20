#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:56:06 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import os

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# Needs pip install imageio-ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/jkordil_idaea/Downloads/ffmpeg" 

from openkarst.network_generation import compute_conduit_lengths
from openkarst.visualization.animation_pyvista import animate_network
from openkarst.models import FlowSimulation


def main():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup flow simulation parameters
    physical_properties = {
        'water_density': 1000,        # kg/m^3
        'gravity': 9.81,              # m/s^2
        'dynamic_viscosity': 0.001,   # Pa.s (kg/m.s)
        'geometry_channel': True,     # Channel geometry for analytical solutions (Default False)
        'channel_type': 'infinite',   # 'infinite' for infinitely wide channel, 'finite' for defined width
        'channel_width': 0.12,        # Width of the channel (only used if channel_type is 'finite')
        'channel_manning': 0.03,      # Constant Manning roughness applied to all channel segments
    }
    
    solver_settings = {
        'relaxation_factor': 0.6,    # Dimensionless
        'max_iterations': 20,        # Maximum Picard iterations
        'picard_depth_tol': 1e-5,    # Picard depth tolerance (meters)
        'ss_rel_l2tol': 1e-3,         # L2 tolerance for steady-state
        'ss_rel_madtol': 1e-8         # Median tolerance for steady-state
    }
    
    simulation_settings = {
        'min_waterdepth': 1e-10,      # Minimum water depth (meters)
        'min_flowrate': 1e-10,        # Minimum flow rate (m^3/s)
        'courant': 0.8,               # Courant number
        'adaptive_timesteps': True,   # Use adaptive timestepping
        'dt_init': 0.01,             # Initial (or constant) timestep (seconds)
        'dt_max': 1.0,               # Maximum allowable time step
        'steady_state': False,        # Steady-state (True) or transient (False)
        't_max': 4000.0,              # Maximum time for transient simulations (seconds)
        'print_info_interval': 100,   # Print info every # time steps
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
    
    logging_settings = {
        'base_dir': base_dir,
        'log_file': 'simulation.log'
    }
     
    # Create network object using OpenPNM
    dl = 1 # Constant spacing between nodes (meters)
    cn_geometry = op.network.Cubic(shape=[5000, 1, 1], connectivity=6, spacing=dl)
    
    # Compute conduit lengths using the utility function
    cn_geometry = compute_conduit_lengths(cn_geometry)
    
    # Create a height field according to the analytical steady-state solution (3.3.1.9) from:
    # Delestre, O. (2010): Simulation du ruissellement d’eau de pluie sur des surfaces agricoles.
    # Mathématiques. Université d’Orléans
    # https://theses.hal.science/tel-00531377v1

    # Parameters
    x = np.linspace(0, 5000, 5000)
    g = 9.81  # gravity (m/s^2)
    q = 2     # flow rate (m^2/s)
    n = 0.03  # Manning's roughness coefficient
    water_height = 9/8 +  1/4 * np.sin(np.pi * x / 500)
    
    # ODE function
    def dz_dx(z, x):
        h = h_interp(x)
        # Derivative from Delestre (2010)
        dh_dx = (np.pi / 2000) * np.cos(np.pi * x / 500)
        return (q**2 / (g * h**3) - 1) * dh_dx - Sf(h)
    
    # Friction slope function
    def Sf(h):
        return n**2 * q * np.abs(q) / (h**(10/3))
    # Interpolate h(x) for the ODE function
    h_interp = interp1d(x, water_height, kind='cubic', fill_value="extrapolate")
    
    # Initial condition for z
    z0 = [0]  # Bed elevation starts at 0
    
    # Solve the ODE with the given dh/dx
    z = odeint(dz_dx, z0, x).flatten()
    
    # Assign the heights of the channel to the z-dimension of the geometry object
    cn_geometry['pore.coords'][:, 2] = z
    
    # Create flow network object
    flow_network = FlowSimulation(cn_geometry,
                                  physical_properties = physical_properties,
                                  solver_settings = solver_settings,
                                  simulation_settings = simulation_settings,
                                  logging_settings = logging_settings)
    
    # Set initial conditions
    initial_Q = np.full(cn_geometry.Nt, 0.0, dtype=float)   # Initial flows at each conduit (Nt throats)
    initial_y = np.full(cn_geometry.Np, 0.0, dtype=float)  # Initial water depths at each node (Np pores)    
    flow_network.set_initial_conditions(initial_Q, initial_y)
      
    # Define inflow and water depth boundary conditions
    flow_rate = 2.0  # Volumetric inflow in m^3/s
    inflow_node = 0
    outflow_node = 4999

    # Apply constant inflow at node 0
    flow_network.set_inflow_BC(
        nodes=inflow_node,
        values=flow_rate,          # Constant BC (single float)
        mode='add',                # Default
        inflow_type='volumetric'   # Default
    )

    # Apply constant water depth at node 4999
    flow_network.set_waterdepth_BC(
        nodes=outflow_node,
        values=water_height[4999],  # Constant BC (single float)
        mode='add'                  # Default
    )
    
    
    # Run simulation and store results
    results = flow_network.run_simulation(desired_outputs = output_settings)
    
    return cn_geometry, results, water_height

if __name__ == '__main__':
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    cn_geometry, results, water_height = main()
    
######################################################################
############## Animation of solution #################################
######################################################################

# Get arrays from results container
Q_history = results['flowrates']
y_history = results['water_depths']
t_history = results['time']

# Compute hydraulic head    
H_history = cn_geometry['pore.coords'][:, 2] + y_history

plt.rcParams['font.size'] = 10

width_cm = 18
height_cm = 10
width_in = width_cm / 2.54
height_in = height_cm / 2.54

fig, ax = plt.subplots(figsize=(width_in, height_in))
distances = np.arange(len(H_history[0]))
lc = LineCollection([], colors='blue', lw=1)  # Empty line collection
ax.add_collection(lc)  # Add the collection to the axes

def init():
    ax.set_xlim(0, 5000)
    ymin, ymax = -15.0, 3.0
    ax.set_ylim(ymin, ymax)
    
    base = cn_geometry['pore.coords'][:, 2]
    
    # Draw the static and analytical lines first with a lower zorder
    ax.plot(distances, base, color='black', lw=0.8, zorder=1)
    ax.plot(distances, base + water_height, color='red', lw=0.8, linestyle='dotted', zorder=2)
    ax.fill_between(distances, ymin, base, color=(0.8, 0.8, 0.8), zorder=0)
    
    # Proxy element for the numerical solution
    numerical_line = Line2D([0], [0], color='blue', lw=1, label='Numerical solution')
    
    # Add all legend entries
    legend_entries = [
        Line2D([0], [0], color='black', lw=0.8, label='Static boundary'),
        Line2D([0], [0], color='red', lw=0.8, linestyle='dotted', label='Analytical solution'),
        Line2D([0], [0], color='blue', lw=0.8, label='Numerical solution')
    ]

    # Set axis labels
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m)')

    # Add the legend without the box
    ax.legend(handles=legend_entries, loc='upper right', frameon=False)
    return [lc]

def update(frame):
    head = H_history[frame]
    points = np.array([distances, head]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc.set_segments(segments)
    lc.set_zorder(3)  # Ensure the blue line is on top
    ax.set_title(f'Time: {t_history[frame]:.1f} (s)')
    return [lc]

update_interval = 1 # plot every x steps
ani = animation.FuncAnimation(fig,
                              update, 
                              frames=range(0, len(H_history), 
                              update_interval),
                              init_func=init,
                              blit=False,
                              repeat=False)

# Hardcoded ffmpeg path
ffmpeg_path = '/Users/jkordil_idaea/Downloads/ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

# Save the animation
video_path= 'test.mp4'
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, codec='h264', metadata=dict(artist='Me'))
ani.save(video_path, writer=writer)


######################################################################
############## Plot solution and error ###############################
######################################################################
distances = np.arange(len(H_history[0]))
base = cn_geometry['pore.coords'][:, 2]
head_numerical = H_history[-1]  # Last time step's hydraulic head data
#head_numerical2 = H_history[10000]  # Last time step's hydraulic head data
head_analytical = base + water_height  # Analytical solution

# Calculate error 
depth_numerical = head_numerical - base
percent_error = ((depth_numerical - water_height) / water_height) * 100

width_cm = 18
height_cm = 9

# Convert centimeters to inches
width_in = width_cm / 2.54
height_in = height_cm / 2.54
# Creating figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_in, height_in))

# Plotting the last time step on the first subplot
ax1.set_xlim(0, 5000)
ymin, ymax = -15.0, 3.0
ax1.set_ylim(ymin, ymax)

ax1.plot(distances, base, 'black', lw=0.75, label='Channel base')
ax1.plot(distances, head_analytical, 'r--', lw=0.75, label='Analytical')
ax1.plot(distances, head_numerical, 'b', lw=0.75, label='Numerical (4000s)')
ax1.fill_between(distances, ymin, base, color=(0.8, 0.8, 0.8))
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Elevation (m)')
ax1.legend(loc='upper right', frameon=False)

spine_linewidth = {spine: ax1.spines[spine].get_linewidth() for spine in ax1.spines}

# Print the linewidth of each spine
print("Linewidth of axes spines:", spine_linewidth)

# Plotting the error
ax2.set_xlim(0, 5000)
ax2.set_ylim(-5, 5)
ax2.plot(distances, percent_error, color='black', lw=0.75, label='Error')
ax2.axhline(0, color='gray', lw=0.8, ls='--')  # Adding a line at y=0 for reference
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Error (%)')
ax2.legend(frameon=False)

plt.tight_layout()
plt.show()

fig.savefig('analytical_solution.eps', format='eps', bbox_inches='tight')