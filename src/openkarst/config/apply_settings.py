#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:54:37 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

def apply_settings(simulation, physical_properties, geometry_settings,
                   solver_settings, simulation_settings, transport_settings, logger):
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
        geometry_settings: An object containing the closed-conduit geometry
            backend and tabulation settings.
        solver_settings: An object containing solver settings such as relaxation 
            factor, maximum iterations, Picard depth tolerance, and relative L2
            tolerance.
        simulation_settings: An object containing simulation settings such as 
            minimum water depth, minimum flow rate, Courant number, adaptive 
            timesteps, initial timestep, maximum timestep, maximum simulation 
            time, steady state flag, print info interval, enable transport flag,
            molecular diffusivity, longitudinal dispersivity, decay rate and transport CFL.
        logger: A logging object used to log the application of settings.

    Returns:
        None
    """
    
    simulation.rho = physical_properties.water_density
    simulation.gravity = physical_properties.gravity
    simulation.dyn_viscosity = physical_properties.dynamic_viscosity
    simulation.geometry_channel =  physical_properties.geometry_channel
    simulation.geometry_backend = geometry_settings.backend
    simulation.geometry_table_points = geometry_settings.table_points
    simulation.geometry_table_file = geometry_settings.table_file
    simulation.geometry_scale_by_diameter = geometry_settings.scale_by_diameter
    simulation.geometry_interpolation_method = geometry_settings.interpolation_method
    simulation.channel_type = physical_properties.channel_type
    simulation.channel_width =  physical_properties.channel_width
    simulation.channel_manning =  physical_properties.channel_manning
    simulation.friction_model =  physical_properties.friction_model
    simulation.w = solver_settings.relaxation_factor
    simulation.max_iterations = solver_settings.max_iterations
    simulation.picard_depth_tol = solver_settings.picard_depth_tol
    simulation.ss_rel_l2tol = solver_settings.ss_rel_l2tol
    simulation.min_waterdepth = simulation_settings.min_waterdepth
    simulation.min_flowrate = simulation_settings.min_flowrate
    simulation.courant = simulation_settings.courant
    simulation.adaptive_timesteps = simulation_settings.adaptive_timesteps
    simulation.dt_init = simulation_settings.dt_init
    simulation.dt_max = simulation_settings.dt_max
    simulation.t_max = simulation_settings.t_max
    simulation.steady_state = simulation_settings.steady_state
    simulation.print_info_interval = simulation_settings.print_info_interval
    simulation.enable_transport = simulation_settings.enable_transport
    simulation.molecular_diffusivity = transport_settings.molecular_diffusivity
    simulation.alpha_l = transport_settings.alpha_l
    simulation.decay_rate = transport_settings.decay_rate
    simulation.transport_cfl = transport_settings.transport_cfl
    
    logger.info('Settings applied')
