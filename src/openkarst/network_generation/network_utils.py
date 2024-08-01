#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:14:34 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import numpy as np

def compute_conduit_lengths(cn_geometry):
    """
    Computes the conduit lengths for a given network geometry and assigns 
    them to the 'throat.lengths' attribute of the geometry.
    
    This function calculates the Euclidean distance between connected nodes 
    (throats) in the network geometry, which is defined by `cn_geometry.coords` 
    and `cn_geometry.conns`. The computed lengths are stored in the 
    'throat.lengths' attribute of `cn_geometry`.
    
    Args:
        cn_geometry (OpenPNM Network): The network geometry object which 
            contains the coordinates and connections of the network nodes.
    
    Returns:
        OpenPNM Network: The updated network geometry object with the 
        computed 'throat.lengths' attribute.
    """
    
    coords_diff = np.diff(cn_geometry.coords[cn_geometry.conns], axis=1).squeeze()
    squared_diffs = coords_diff**2
    sum_squared_diffs = np.sum(squared_diffs, axis=1)
    conduit_lengths = np.sqrt(sum_squared_diffs)
    cn_geometry['throat.lengths'] = conduit_lengths
    
    return cn_geometry