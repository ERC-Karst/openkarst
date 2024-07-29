#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:03:16 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import os
import logging
from typing import Optional

def setup_logging(logging_settings: Optional[dict] = None):
    """
    Sets up logging for the simulation. Logs are saved in a 'logs' folder located in the specified
    base directory and validates the settings.

    This function configures the logging module to log messages to a file named 'simulation.log'
    within a 'logs' directory. If the 'logs' directory does not exist, it is created. The log file
    is appended to each time the function is called, ensuring that previous log messages are
    retained. If not specific folder is given by the logging settings the current working directory
    is used.

    The log format includes the timestamp, logger name, log level, and the log message.

    Args:
        logging_settings (Optional[dict]): A dictionary containing logging configuration keys 'base_dir' 
                                           and 'log_file'. If None, default values are used.

    Returns:
        logging.Logger: Configured logger for the simulation.
    """
    
    # Default values
    default_base_dir = os.getcwd()
    default_log_file = 'simulation.log'
    
    # Use provided settings or defaults
    base_dir = logging_settings.get('base_dir', default_base_dir) if logging_settings else default_base_dir
    log_file = logging_settings.get('log_file', default_log_file) if logging_settings else default_log_file

    # Validate settings
    validate_logging_settings({
        'base_dir': base_dir,
        'log_file': log_file
    })

    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        filename=log_path,
        filemode='a',  # Append to the log file if it exists
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.DEBUG,
        force=True
    )
    logger = logging.getLogger('FlowSimulation')
    return logger

def validate_logging_settings(logging_settings: dict):
    """
    Validates the logging settings.

    Args:
        logging_settings (dict): A dictionary containing logging configuration keys 'base_dir', 
                                 'log_level', and 'log_file'.

    Raises:
        ValueError: If any of the logging settings are invalid.
    """
    if not isinstance(logging_settings.get('base_dir'), str) or not logging_settings['base_dir']:
        raise ValueError("Error: 'base_dir' must be a non-empty string.")
    
    if not isinstance(logging_settings.get('log_file'), str) or not logging_settings['log_file']:
        raise ValueError("Error: 'log_file' must be a non-empty string.")