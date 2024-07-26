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
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, 'simulation.log')
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        force=True
    )
    logger = logging.getLogger('FlowSimulation')
    return logger