#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:39:15 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import numpy as np
from typing import Dict

def initialize_results_container(desired_outputs: Dict[str, bool], logger):
    """
    Initializes a results container based on the desired outputs.

    This function creates a dictionary to store simulation results for the keys 
    specified in `desired_outputs` that are set to True. It logs the creation of 
    the results container using the provided logger.

    Args:
        desired_outputs (Dict[str, bool]): A dictionary specifying which results to store. 
            Keys are the names of the results, and values are booleans indicating whether 
            to store that result.
        logger (logging.Logger): Logger instance for logging the creation of the results container.

    Returns:
        Dict[str, list]: A dictionary initialized to store lists of the specified results.
    """
    
    results_container = {key: [] for key in desired_outputs if desired_outputs[key]}
    logger.info('Results container created for: %s', desired_outputs)
    return results_container

def store_results(simulation_instance, results_container):
    """
    Stores the current state of the simulation in the results container.

    This function appends the current state of various simulation variables to the 
    corresponding lists in the results container. The variables to store are specified 
    by the keys of the results container.

    Args:
        simulation_instance: An instance of the simulation class containing the current 
            state of the simulation variables.
        results_container (Dict[str, list]): A dictionary containing lists to store the 
            simulation results.

    Returns:
        Dict[str, list]: The updated results container with the current simulation state appended.
    """
    
    if 'convergence_fails' in results_container:
        results_container['convergence_fails'].append(simulation_instance.convergence_fails)
    if 'flowrates' in results_container:
        results_container['flowrates'].append(np.copy(simulation_instance.Q))
    if 'water_depths' in results_container:
        results_container['water_depths'].append(np.copy(simulation_instance.y))
    if 'time' in results_container:
        results_container['time'].append(simulation_instance.current_time)
    if 'time_step_size' in results_container:
        results_container['time_step_size'].append(simulation_instance.dt)
    if 'l2_norms' in results_container:
        l2_norm = np.linalg.norm(simulation_instance.y_new - simulation_instance.y_old_t)
        relative_l2_norm = l2_norm / np.linalg.norm(simulation_instance.y_new)

        results_container['l2_norms'].append(relative_l2_norm)
    if 'mad_norms' in results_container:   
        mad = np.median(np.abs(simulation_instance.y_new - simulation_instance.y_old_t))
        relative_mad_norm = mad / np.median(np.abs(simulation_instance.y_new))
        
        results_container['mad_norms'].append(relative_mad_norm)
    if 'reynolds_numbers' in results_container:
        results_container['reynolds_numbers'].append(np.copy(simulation_instance.Re_conduit))

    return results_container