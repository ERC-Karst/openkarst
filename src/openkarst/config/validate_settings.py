#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:38:20 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

def validate_settings(physical_properties, solver_settings, simulation_settings, logger):
       
    if not (isinstance(physical_properties.water_density, (int, float)) and 
            physical_properties.water_density > 0):
        raise ValueError("Error: 'water_density' must be type int/float and greater than 0 (kg/m^3).")
    
    if not (isinstance(physical_properties.gravity, (int, float)) and 
            physical_properties.gravity > 0):
        raise ValueError("Error: 'gravity' must be type int/float and greater than 0 (m/s^2).")
    
    if not (isinstance(physical_properties.dynamic_viscosity, (int, float)) and 
            physical_properties.dynamic_viscosity > 0):
        raise ValueError("Error: 'dynamic_viscosity' must be type int/float and greater than 0 "
                         "(kg m^-1 s^-1).")
        
    if physical_properties.geometry_channel:
        if not (physical_properties.channel_type in ['finite', 'infinite']):
            raise ValueError("Error: 'channel_type' must be either 'finite' or 'infinite'.")
        
        if physical_properties.channel_type == 'finite':
            if not (isinstance(physical_properties.channel_width, (int, float)) and 
                    physical_properties.channel_width > 0):
                raise ValueError("Error: 'channel_width' must be type int/float and greater than 0 (meters).")
        
        if not (isinstance(physical_properties.channel_manning, (int, float)) and 
                physical_properties.channel_manning >= 0):
            raise ValueError("Error: 'channel_manning' must be type int/float and greater than or equal to 0.")
        
    if not (isinstance(solver_settings.relaxation_factor, (int, float)) and 
            0 < solver_settings.relaxation_factor <= 1):
        raise ValueError("Error: 'relaxation_factor' must be type int/float and greater than 0 and "
                         "less than or equal to 1.")
    if not (isinstance(solver_settings.max_iterations, int) and 
            10 < solver_settings.max_iterations <= 1000):
        raise ValueError("Error: 'max_iterations' must be type int and greater than 10 and less "
                         "than 1000.")
    
    if not (isinstance(solver_settings.picard_depth_tol, (int, float)) and 
            1e-11 < solver_settings.picard_depth_tol <= 1e-2):
        raise ValueError("Error: 'picard_depth_tol' must be type int/float and greater than 1e-11 "
                         "(meter) and less than 1e-3 (meter).")
    
    if not (isinstance(solver_settings.ss_rel_l2tol, (int, float)) and 
            1e-8 < solver_settings.ss_rel_l2tol <= 1e-2):
        raise ValueError("Error: 'ss_rel_l2tol' must be type int/float and greater than 1e-8 and "
                         "less than 1e-3.")

    if not (isinstance(simulation_settings.min_waterdepth, (int, float)) and 
            1e-12 <= simulation_settings.min_waterdepth < 1e-5):
        raise ValueError("Error: 'min_waterdepth' must be type int/float and greater equal 1e-10 and "
                         "less than 1e-5 (meter).")
    
    if not (isinstance(simulation_settings.min_flowrate, (int, float)) and 
            1e-12 <= simulation_settings.min_flowrate < 1e-5):
        raise ValueError("Error: 'min_flowrate' must be type int/float and greater equal 1e-10 and "
                         "less than 1e-5 (m^3/s).")

    if not (isinstance(simulation_settings.courant, (int, float)) and 
            0 < simulation_settings.courant <= 2):
        raise ValueError("Error: 'courant' must be type int/float and greater than 0 and less "
                         "than 2.")
           
    if not isinstance(simulation_settings.adaptive_timesteps, bool):
        raise ValueError("Error: 'adaptive_timesteps' must be a boolean True or False.")
        
    if not simulation_settings.adaptive_timesteps and (
            simulation_settings.dt_init is None or simulation_settings.dt_init <= 0
            ):
        raise ValueError("Error: 'dt_init' (constant step size) is required and must be "
                         "greater than 0 when 'adaptive_timesteps' is False.")
        
    if simulation_settings.adaptive_timesteps and (
            simulation_settings.dt_init is None or simulation_settings.dt_init <= 0
            ):
        raise ValueError("Error: 'dt_init' (initial step size) is required and must be "
                          "greater than 0 when 'adaptive_timesteps' is True.")

    if not isinstance(simulation_settings.dt_init, (int, float)):
        raise ValueError("Error: 'dt_init' must be type int/float.")
        
    if not simulation_settings.adaptive_timesteps and (
            simulation_settings.dt_init is None or simulation_settings.dt_init <= 0
            ):
        raise ValueError("Error: 'dt_max' is required and must be "
                          "greater than 0 when 'adaptive_timesteps' is True.")
        
    if not isinstance(simulation_settings.dt_max, (int, float)):
        raise ValueError("Error: 'dt_max' must be type int/float.") 
    
    if not isinstance(simulation_settings.steady_state, bool):
        raise ValueError("Error: 'steady_state' must be a boolean True or False.")
        
    if not simulation_settings.steady_state:
        if not (isinstance(simulation_settings.t_max, (int, float)) and
                simulation_settings.t_max > 0
                ):
            raise ValueError("Error: 't_max' must be provided and greater than 0 for "
                             "transient simulations ('steady_state' is False).")
           
    if not (isinstance(simulation_settings.print_info_interval, int) and 
            simulation_settings.print_info_interval >= 1):
        raise ValueError("Error: 'print_info_interval' must be an integer greater than or equal to 1.")
    
    logger.info('Settings validated')
    

