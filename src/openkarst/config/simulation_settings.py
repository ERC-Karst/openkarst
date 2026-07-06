#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:13:47 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationSettings:
    min_waterdepth: float = 1e-10
    min_flowrate: float = 1e-10
    courant: float = 0.8
    adaptive_timesteps: bool = False
    dt_init: Optional[float] = None
    dt_max: Optional[float] = None
    t_max: float = 0.0
    steady_state: bool = False
    print_info_interval: int = 1
    enable_transport: bool = False
