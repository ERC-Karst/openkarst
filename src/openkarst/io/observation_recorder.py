#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 07:05:12 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""
import numpy as np
import pandas as pd


CONNECTED_ABS_FLOWRATE = "connected_abs_flowrate"
SUPPORTED_OBSERVATION_VARIABLES = {
    "water_depth",
    CONNECTED_ABS_FLOWRATE,
    "concentrations",
    "mass",
}


def _validated_observation_variables(variables):
    if isinstance(variables, str):
        variables = [variables]

    normalized = []
    for variable in variables:
        if variable not in SUPPORTED_OBSERVATION_VARIABLES:
            supported = ", ".join(sorted(SUPPORTED_OBSERVATION_VARIABLES))
            raise ValueError(
                f"Unsupported observation variable '{variable}'. "
                f"Supported variables are: {supported}."
            )
        if variable not in normalized:
            normalized.append(variable)
    return normalized


def _connected_abs_flowrate(flow_sim, node):
    connected = (flow_sim.n_indices1 == node) | (flow_sim.n_indices2 == node)
    if not np.any(connected):
        return 0.0
    return float(np.sum(np.abs(flow_sim.Q_new[connected])))


class ObservationRecorder:
    """Records simulation outputs at specified nodes and time intervals.

    This class records user-defined variables (e.g., water depth or connected
    conduit flowrate) at a list of observation nodes during a transient
    simulation. Currently does not support conduit values such as Reynolds
    numbers.

    Attributes:
        nodes (list of int): Indices of nodes to observe.
        variables (list of str): Variables to record. Options include:
            - 'water_depth': records water depth at the node.
            - 'connected_abs_flowrate': records the sum of absolute flowrates
              through conduits connected to the node.
            - 'concentrations': records concentrations at the node (AD-Transport)
            - 'mass': records mass at the node (AD-Transport)
        interval (float): Time interval between recordings, in seconds.
        next_record_time (float): Simulation time at which the next recording is due.
        records (list of dict): Internal buffer storing observation rows.
    """

    def __init__(self, nodes, variables, interval=1.0):
        """Initializes the observation recorder.

        Args:
            nodes (list of int): Node indices to track.
            variables (list of str): List of variables to observe. 
                Supported values are 'water_depth', 'connected_abs_flowrate',
                'concentrations', and 'mass'.
            interval (float, optional): Recording interval in seconds. Defaults to 1.0.
        """
        self.nodes = nodes
        self.variables = _validated_observation_variables(variables)
        self.interval = interval
        self.next_record_time = 0.0
        self.records = []  # Each dict becomes a row in the dataframe

    def record(self, current_time, flow_sim):
        """Records values from the simulation at the current time.

        Args:
            current_time (float): The current simulation time (in seconds).
            flow_sim (FlowSimulation): A simulation object providing `y_new`,
                `Q_new`, `n_indices1`, and `n_indices2` arrays.

        Note:
            - `y_new` must be available in `flow_sim` for water depth.
            - `Q_new`, `n_indices1`, and `n_indices2` must be available for
              connected conduit flowrate tracking.
        """
        for node in self.nodes:
            row = {
                'time': current_time,
                'node': node
            }
            if 'water_depth' in self.variables:
                row['water_depth'] = flow_sim.y_new[node]
            if CONNECTED_ABS_FLOWRATE in self.variables:
                row[CONNECTED_ABS_FLOWRATE] = _connected_abs_flowrate(flow_sim, node)
            if 'concentrations' in self.variables:
                row['concentrations'] = flow_sim.C[node]
            if 'mass' in self.variables:
                row['mass'] = flow_sim.M[node]
            self.records.append(row)

    def to_dataframe(self):
        """Converts the recorded observations into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with one row per observation and columns including
                'time', 'node', and the selected variables (e.g.,
                'water_depth', 'connected_abs_flowrate').
        """
        return pd.DataFrame(self.records)

    def reset(self):
        """Clears all recorded observations and resets internal time counter."""
        self.records.clear()
        self.next_record_time = 0.0
