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
class TransportSettings:
    molecular_diffusivity: float = 1e-9 # [m^2/s]
    alpha_l: float = 0.1 # longitudial dispersivity [m]
    decay_rate: float = 0.0 # first-order decay [1/s]
    transport_cfl: float = 0.8 
   