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
    Initializes a results container for storing selected simulation outputs.

    This function validates the user-specified `desired_outputs` dictionary against a
    predefined set of supported output keys. It then creates a container dictionary 
    where each valid key maps to an empty list to store time-dependent results.

    Args:
        desired_outputs (Dict[str, bool]): A dictionary specifying which results to store.
            Keys must be among the supported output types, and values should be True
            to enable storing that result.
        logger (logging.Logger): Logger instance for logging messages about the setup.

    Returns:
        Dict[str, list]: A dictionary with result keys initialized to empty lists.

    Raises:
        ValueError: If `desired_outputs` contains unknown or unsupported keys.
    """

    allowed_keys = {
        'convergence_fails',
        'flowrates',
        'water_depths',
        'time',
        'time_step_size',
        'l2_norms',
        'mad_norms',
        'reynolds_numbers',
        'picard_iterations'
    }

    invalid_keys = [key for key in desired_outputs if key not in allowed_keys and key != 'output_interval']
    if invalid_keys:
        raise ValueError(f"Invalid keys in desired_outputs: {invalid_keys}. "
                         f"Allowed keys are: {sorted(allowed_keys)}")

    results_container = {key: [] for key in desired_outputs if desired_outputs.get(key, False) and key in allowed_keys}
    logger.info('Results container created for: %s', list(results_container.keys()))
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
        results_container['l2_norms'].append(simulation_instance.relative_l2_norm)
    if 'mad_norms' in results_container:           
        results_container['mad_norms'].append(simulation_instance.relative_mad_norm)
    if 'reynolds_numbers' in results_container:
        results_container['reynolds_numbers'].append(np.copy(simulation_instance.Re_conduit))
    if 'picard_iterations' in results_container:
        results_container['picard_iterations'].append(simulation_instance.picard_iterations_last)

    return results_container