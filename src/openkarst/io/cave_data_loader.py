#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:22:54 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import networkx as nx
import openpnm as op
import numpy as np

class CaveDataLoader:
    """
    A class to load cave network data from CSV files and create an OpenPNM geometry object.

    This class reads node coordinates, edge connections, and diameters from
    respective CSV files and constructs a NetworkX graph with the data. It then
    converts the graph into an OpenPNM geometry object with assigned conduit lengths
    and diameters.

    Attributes:
        nodes_file (str): Path to the CSV file containing node coordinates.
        edges_file (str): Path to the CSV file containing edge connections.
        diameters_file (str): Path to the CSV file containing node diameters.

    Methods:
        load_cave_data(): Loads the cave data from the CSV files and constructs
            a NetworkX graph with node coordinates, edge connections, and
            edge diameters. Returns an OpenPNM geometry object.
    """
    
    def __init__(self, nodes_file: str, edges_file: str, diameters_file: str):
        """
        Initializes the CaveDataLoader with file paths.

        Args:
            nodes_file (str): Path to the CSV file containing node coordinates.
            edges_file (str): Path to the CSV file containing edge connections.
            diameters_file (str): Path to the CSV file containing node diameters.
        """
        
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.diameters_file = diameters_file
    
    def load_cave_data(self):
        """
        Loads the cave data from the CSV files and constructs a NetworkX graph.

        This method reads the node coordinates, edge connections, and node diameters
        from their respective CSV files. It constructs a NetworkX graph with the
        loaded data, where nodes have coordinates and edges have average diameters
        based on the connected nodes. The two diameters available at each node are
        currently averaged. The graph is then converted into an OpenPNM geometry
        object with assigned conduit lengths and diameters.

        Returns:
            openpnm.network.GenericNetwork: An OpenPNM geometry object representing
                the network with assigned conduit lengths and diameters.
        """
        
        G = nx.Graph()
        node_diameters = {}

        # Load nodes and their coordinates from the file, skipping the header
        with open(self.nodes_file, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                node_id, x, y, z = line.strip().split(';')
                G.add_node(int(node_id), coords=[float(x), float(y), float(z)])
        
        # Load edges from the file, skipping the header
        with open(self.edges_file, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                node_a, node_b = map(int, line.strip().split(';'))
                G.add_edge(node_a, node_b)
        
        # Load diameters from the file, skipping the header
        with open(self.diameters_file, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                node_id, cswidth, csheight = line.strip().split(';')
                average_diameter = (float(cswidth) + float(csheight)) / 2
                node_diameters[int(node_id)] = average_diameter
        
        # Assign average diameters to each edge by averaging diameters of connected nodes
        edge_diameters = {}
        for node_a, node_b in G.edges():
            avg_diameter = (node_diameters[node_a] + node_diameters[node_b]) / 2
            edge_diameters[tuple(sorted((node_a, node_b)))] = avg_diameter
            
        # Create an openPNM geometry object
        cn_geometry = op.io.network_from_networkx(G)
        
        # Compute and assign conduit lengths 
        coords_diff = np.diff(cn_geometry.coords[cn_geometry.conns], axis=1).squeeze()
        squared_diffs = coords_diff**2
        sum_squared_diffs = np.sum(squared_diffs, axis=1)
        conduit_lengths = np.sqrt(sum_squared_diffs)
        cn_geometry['throat.lengths'] = conduit_lengths
        
        # Assign conduit diameters to openPNM geometry object
        cn_geometry['throat.diameters'] = [edge_diameters[tuple(sorted(edge))] for edge in cn_geometry['throat.conns']]
 
        return cn_geometry