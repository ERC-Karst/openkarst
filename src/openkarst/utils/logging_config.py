#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:03:16 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import os
import logging

def setup_logging():
    """
    Sets up logging for the simulation. Logs are saved in a 'logs' folder located in the same
    directory as the main script.

    This function configures the logging module to log messages to a file named 'simulation.log'
    within a 'logs' directory. If the 'logs' directory does not exist, it is created. The log file
    is overwritten each time the function is called.

    The log format includes the timestamp, logger name, log level, and the log message.

    Returns:
        logging.Logger: Configured logger for the simulation.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    log_dir = os.path.join(current_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, 'simulation.log')
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        force=True
    )
    logger = logging.getLogger('FlowSimulation')
    return logger