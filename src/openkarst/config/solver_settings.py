#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:13:21 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

from dataclasses import dataclass

@dataclass
class SolverSettings:
    relaxation_factor: float = 0.6
    max_iterations: int = 500
    picard_depth_tol: float = 1e-9
    ss_rel_l2tol: float = 1e-7