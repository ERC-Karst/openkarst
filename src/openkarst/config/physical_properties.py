#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:12:58 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

from dataclasses import dataclass

@dataclass
class PhysicalProperties:
    water_density: float = 1000.0
    gravity: float = 9.81
    dynamic_viscosity: float = 0.001
    geometry_channel: bool = False
    channel_type: str = 'finite'  
    channel_width: float = 1.0
    channel_manning: float = 0.03
    friction_model: str = 'hybrid'