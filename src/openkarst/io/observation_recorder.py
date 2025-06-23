#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 07:05:12 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""
import pandas as pd

class ObservationRecorder:
    """Records simulation outputs at specified nodes and time intervals.

    This class records user-defined variables (e.g., water depth or inflow)
    at a list of observation nodes during a transient simulation. Currently
    does not support conduit values such as Reynolds numbers.

    Attributes:
        nodes (list of int): Indices of nodes to observe.
        variables (list of str): Variables to record. Options include:
            - 'water_depth': records water depth at the node.
            - 'inflow': records inflow (from dQ_new) into the node.
        interval (float): Time interval between recordings, in seconds.
        next_record_time (float): Simulation time at which the next recording is due.
        records (list of dict): Internal buffer storing observation rows.
    """

    def __init__(self, nodes, variables, interval=1.0):
        """Initializes the observation recorder.

        Args:
            nodes (list of int): Node indices to track.
            variables (list of str): List of variables to observe. 
                Supported values are 'water_depth' and 'inflow'.
            interval (float, optional): Recording interval in seconds. Defaults to 1.0.
        """
        self.nodes = nodes
        self.variables = variables
        self.interval = interval
        self.next_record_time = 0.0
        self.records = []  # Each dict becomes a row in the dataframe

    def record(self, current_time, flow_sim):
        """Records values from the simulation at the current time.

        Args:
            current_time (float): The current simulation time (in seconds).
            flow_sim (FlowSimulation): A simulation object providing `y_new` and `dQ_new` arrays.

        Note:
            - `y_new` must be available in `flow_sim` for water depth.
            - `dQ_new` must be available for inflow tracking.
        """
        for node in self.nodes:
            row = {
                'time': current_time,
                'node': node
            }
            if 'water_depth' in self.variables:
                row['water_depth'] = flow_sim.y_new[node]
            if 'inflow' in self.variables:
                row['inflow'] = flow_sim.dQ_new[node]
            self.records.append(row)

    def to_dataframe(self):
        """Converts the recorded observations into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with one row per observation and columns including
                'time', 'node', and the selected variables (e.g., 'water_depth', 'inflow').
        """
        return pd.DataFrame(self.records)

    def reset(self):
        """Clears all recorded observations and resets internal time counter."""
        self.records.clear()
        self.next_record_time = 0.0
