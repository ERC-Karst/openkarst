#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:54:37 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

def apply_settings(simulation, physical_properties, solver_settings, simulation_settings, logger):
    """
    Apply various settings to the simulation object.

    This function sets multiple physical properties, solver settings, and 
    simulation settings to the provided simulation object. It also logs the 
    application of these settings.

    Args:
        simulation: The simulation object to which settings are applied.
        physical_properties: An object containing physical properties such as 
            water density, gravity, dynamic viscosity, geometry channel, channel 
            type, channel width, and channel Manning.
        solver_settings: An object containing solver settings such as relaxation 
            factor, maximum iterations, Picard depth tolerance, relative L2 tolerance, 
            and relative MAD tolerance.
        simulation_settings: An object containing simulation settings such as 
            minimum water depth, minimum flow rate, Courant number, adaptive 
            timesteps, initial timestep, maximum timestep, maximum simulation 
            time, steady state flag, and print info interval.
        logger: A logging object used to log the application of settings.

    Returns:
        None
    """
    
    simulation.rho = physical_properties.water_density
    simulation.gravity = physical_properties.gravity
    simulation.dyn_viscosity = physical_properties.dynamic_viscosity
    simulation.geometry_channel =  physical_properties.geometry_channel 
    simulation.channel_type = physical_properties.channel_type
    simulation.channel_width =  physical_properties.channel_width
    simulation.channel_manning =  physical_properties.channel_manning
    simulation.friction_model =  physical_properties.friction_model
    simulation.w = solver_settings.relaxation_factor
    simulation.max_iterations = solver_settings.max_iterations
    simulation.picard_depth_tol = solver_settings.picard_depth_tol
    simulation.ss_rel_l2tol = solver_settings.ss_rel_l2tol
    simulation.ss_rel_madtol = solver_settings.ss_rel_madtol
    simulation.min_waterdepth = simulation_settings.min_waterdepth
    simulation.min_flowrate = simulation_settings.min_flowrate
    simulation.courant = simulation_settings.courant
    simulation.adaptive_timesteps = simulation_settings.adaptive_timesteps
    simulation.dt_init = simulation_settings.dt_init
    simulation.dt_max = simulation_settings.dt_max
    simulation.t_max = simulation_settings.t_max
    simulation.steady_state = simulation_settings.steady_state
    simulation.print_info_interval = simulation_settings.print_info_interval
    
    logger.info('Settings applied')