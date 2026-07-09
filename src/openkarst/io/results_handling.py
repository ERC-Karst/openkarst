#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:39:15 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import numpy as np
from typing import Dict


def _reservoir_nodes(simulation_instance):
    return np.array([reservoir.node for reservoir in simulation_instance.reservoirs], dtype=int)


def _reservoir_water_depths(simulation_instance):
    return np.array(
        [reservoir.reservoir_water_depth for reservoir in simulation_instance.reservoirs],
        dtype=float,
    )


def _reservoir_heads(simulation_instance):
    return np.array([reservoir.get_hydraulic_head() for reservoir in simulation_instance.reservoirs], dtype=float)


def _reservoir_storage(simulation_instance):
    return np.array([reservoir.get_storage() for reservoir in simulation_instance.reservoirs], dtype=float)


def _reservoir_exchange(simulation_instance):
    return np.array([reservoir.last_exchange_rate for reservoir in simulation_instance.reservoirs], dtype=float)


def _reservoir_recharge(simulation_instance):
    return np.array([reservoir.last_recharge_rate for reservoir in simulation_instance.reservoirs], dtype=float)


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
        'velocities',
        'water_depths',
        'time',
        'time_step_size',
        'l2_norms',
        'y_l2_norms',
        'Q_l2_norms',
        'reynolds_numbers',
        'picard_iterations',
        'picard_iterations_total',
        'concentrations',
        'mass',
        'reservoir_nodes',
        'reservoir_water_depths',
        'reservoir_heads',
        'reservoir_storage',
        'reservoir_exchange',
        'reservoir_recharge',
    }

    invalid_keys = [key for key in desired_outputs if key not in allowed_keys and key != 'output_interval']
    if invalid_keys:
        raise ValueError(f"Invalid keys in desired_outputs: {invalid_keys}. "
                         f"Allowed keys are: {sorted(allowed_keys)}")

    return {key: [] for key in desired_outputs if desired_outputs.get(key, False) and key in allowed_keys}

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
    if 'velocities' in results_container:
        results_container['velocities'].append(np.copy(simulation_instance._v_mid_last))
    if 'water_depths' in results_container:
        results_container['water_depths'].append(np.copy(simulation_instance.y))
    if 'time' in results_container:
        results_container['time'].append(simulation_instance.current_time)
    if 'time_step_size' in results_container:
        results_container['time_step_size'].append(simulation_instance.dt)
    if 'l2_norms' in results_container:
        results_container['l2_norms'].append(simulation_instance.relative_y_l2_norm)
    if 'y_l2_norms' in results_container:
        results_container['y_l2_norms'].append(simulation_instance.relative_y_l2_norm)
    if 'Q_l2_norms' in results_container:
        results_container['Q_l2_norms'].append(simulation_instance.relative_Q_l2_norm)
    if 'reynolds_numbers' in results_container:
        results_container['reynolds_numbers'].append(np.copy(simulation_instance.Re_conduit))
    if 'picard_iterations' in results_container:
        results_container['picard_iterations'].append(simulation_instance.picard_iterations_last)
    if 'picard_iterations_total' in results_container:
        results_container['picard_iterations_total'].append(
            simulation_instance.picard_iterations_total
        )
    if 'concentrations' in results_container:
        results_container['concentrations'].append(np.copy(simulation_instance.C))
    if 'mass' in results_container:
        results_container['mass'].append(np.copy(simulation_instance.M))
    if 'reservoir_nodes' in results_container:
        results_container['reservoir_nodes'].append(_reservoir_nodes(simulation_instance))
    if 'reservoir_water_depths' in results_container:
        results_container['reservoir_water_depths'].append(
            _reservoir_water_depths(simulation_instance)
        )
    if 'reservoir_heads' in results_container:
        results_container['reservoir_heads'].append(_reservoir_heads(simulation_instance))
    if 'reservoir_storage' in results_container:
        results_container['reservoir_storage'].append(_reservoir_storage(simulation_instance))
    if 'reservoir_exchange' in results_container:
        results_container['reservoir_exchange'].append(_reservoir_exchange(simulation_instance))
    if 'reservoir_recharge' in results_container:
        results_container['reservoir_recharge'].append(_reservoir_recharge(simulation_instance))

    return results_container
