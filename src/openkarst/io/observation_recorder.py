#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 07:05:12 2025

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""
import pandas as pd

class ObservationRecorder:
    def __init__(self, nodes, variables, interval=1.0):
        self.nodes = nodes
        self.variables = variables
        self.interval = interval
        self.next_record_time = 0.0
        self.records = []  # List of dicts; each becomes a row in the dataframe

    def record(self, current_time, flow_sim):
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
        return pd.DataFrame(self.records)

    def reset(self):
        self.records.clear()
        self.next_record_time = 0.0
