#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:03:16 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import os
import logging
from datetime import datetime
from typing import Optional


def _timestamped_log_file(log_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = f'simulation_{timestamp}.log'
    counter = 1
    while os.path.exists(os.path.join(log_dir, log_file)):
        log_file = f'simulation_{timestamp}_{counter}.log'
        counter += 1
    return log_file


def setup_logging(logging_settings: Optional[dict] = None):
    """
    Sets up logging for the simulation. Logs are saved in a 'logs' folder located in the specified
    base directory and validates the settings.

    This function configures the FlowSimulation logger to log messages to a file
    within a 'logs' directory. If 'log_file' is omitted, a timestamped log file is
    created for the run. If 'log_file' is provided, messages are appended to that
    explicit file. If no base directory is given, the current working directory is used.

    The log format includes the timestamp, log level, logger name, and message.

    Args:
        logging_settings (Optional[dict]): A dictionary containing logging configuration keys 'base_dir' 
                                           and 'log_file'. If None, default values are used.

    Returns:
        logging.Logger: Configured logger for the simulation.
    """
    
    # Use provided settings or defaults
    default_base_dir = os.getcwd()
    base_dir = logging_settings.get('base_dir', default_base_dir) if logging_settings else default_base_dir

    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = logging_settings.get('log_file') if logging_settings else None
    if log_file is None:
        log_file = _timestamped_log_file(log_dir)

    # Validate settings
    validate_logging_settings({
        'base_dir': base_dir,
        'log_file': log_file
    })
    
    logger = logging.getLogger('FlowSimulation')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    log_path = os.path.join(log_dir, log_file)
    handler = logging.FileHandler(log_path, mode='a')
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logger.addHandler(handler)
    logger.info('----- openKARST log started -----')
    logger.info('Log file: path=%s', log_path)
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
