#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 08:41:41 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import numpy as np
import time
from contextlib import contextmanager

@contextmanager
def time_this(label: str):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print(f'[{label}] Elapsed time = {t1 - t0:.2f} seconds')