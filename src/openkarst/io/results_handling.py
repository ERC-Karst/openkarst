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
    results_container = {key: [] for key in desired_outputs if desired_outputs[key]}
    logger.info('Results container created for: %s', desired_outputs)
    return results_container

def store_results(simulation_instance, results_container):
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
        relative_l2_norm = l2_norm / np.linalg.norm(simulation_instance.y_old_t)
        results_container['l2_norms'].append(relative_l2_norm)
    if 'reynolds_numbers' in results_container:
        results_container['reynolds_numbers'].append(np.copy(simulation_instance.Re_conduit))

    return results_container