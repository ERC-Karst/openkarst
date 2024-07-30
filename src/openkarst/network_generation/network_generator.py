#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:47:33 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class NetworkGenerator:
    """
    A class to generate different types of network geometries.
    """
    
    def __init__(self):
        """
        Initializes the NetworkGenerator with an empty graph and a node ID counter.
        """
        
        self.graph = nx.Graph()
        self.node_id = 0

    def create_linear_network_with_deadends(self, total_length, spacing, n_dead_ends, length_deadends, 
                                            distribution='equal', mean=None, sigma=None):
        """
        Creates a linear network with specified dead-ends.
        
        Args:
            total_length (float): The total length of the main linear path.
            spacing (float): The spacing between nodes along the main linear path.
            n_dead_ends (int): The number of dead-ends to add to the network.
            length_deadends (float): The length of each dead-end branch (used if distribution='equal').
            distribution (str): The type of distribution for dead-end lengths ('equal' or 'lognormal').
            mean (float): The mean of the lognormal distribution (used if distribution='lognormal').
            sigma (float): The standard deviation of the lognormal distribution (used if distribution='lognormal').
        
        Returns:
            nx.Graph: The generated network with linear and dead-end branches.
        """

        self.graph.clear()
        self.node_id = 0
        
        # Initialize the graph with the root node
        self._add_node([0.0, 0.0, 0.0])
        self.node_id += 1
        
        # Create the main linear path
        for i in range(1, int(total_length / spacing) + 1):
            self._add_node([float(i * spacing), 0.0, 0.0])
            self.graph.add_edge(self.node_id - 1, self.node_id)
            self.node_id += 1
        
        # Add dead-ends at equally spaced intervals along the main path
        main_path_nodes = list(self.graph.nodes)
        dead_end_spacing = len(main_path_nodes) // (n_dead_ends + 1)
        
        for i in range(1, n_dead_ends + 1):
            base_node_id = main_path_nodes[i * dead_end_spacing]
            x, y, _ = self.graph.nodes[base_node_id]['coords']
            
            if distribution == 'equal':
                length = length_deadends
            elif distribution == 'lognormal':
                if mean is None or sigma is None:
                    raise ValueError("Mean and sigma must be provided for lognormal distribution")
                length = np.random.lognormal(mean, sigma)
            else:
                raise ValueError("Unsupported distribution type. Use 'equal' or 'lognormal'")
            
            # Add dead-end extending in positive y direction
            self._add_deadend(base_node_id, x, y, spacing, length, 1)
            
            # Reset base_node_id to extend in negative y direction
            base_node_id = main_path_nodes[i * dead_end_spacing]
            
            # Add dead-end extending in negative y direction
            self._add_deadend(base_node_id, x, y, spacing, length, -1)
        
        return self.graph
    
    def _add_node(self, coords):
        """
        Adds a node to the graph with specified coordinates.

        Args:
            coords (list): The coordinates of the node.
        """
        
        self.graph.add_node(self.node_id, coords=coords)
    
    def _add_deadend(self, base_node_id, x, y, spacing, length_deadends, direction):
        """
        Adds a dead-end branch to the graph.

        Args:
            base_node_id (int): The ID of the node to start the dead-end from.
            x (float): The x-coordinate of the base node.
            y (float): The y-coordinate of the base node.
            spacing (float): The spacing between nodes along the dead-end.
            length_deadends (float): The length of the dead-end branch.
            direction (int): The direction of the dead-end branch (1 for positive y, -1 for negative y).
        """
        
        for j in range(1, int(length_deadends / spacing) + 1):
            self._add_node([float(x), float(y + j * spacing * direction), 0.0])
            self.graph.add_edge(base_node_id, self.node_id)
            base_node_id = self.node_id
            self.node_id += 1

    def create_rectilinear_network(self, levels, scale_factor=1, node_spacing=10):
        """
        Creates a rectilinear network with branching.

        Args:
            levels (int): The number of levels of branching.
            scale_factor (float, optional): The scaling factor for the length of branches. Defaults to 1.
            node_spacing (float, optional): The spacing between nodes. Defaults to 10.

        Returns:
        nx.Graph: The generated rectilinear network.
        """
       
        self.graph.clear()
        self.node_id = 0
        
        # Initialize the graph with the root node
        self._add_node([0.0, 0.0, 0.0])
        self.node_id += 1
        
        def add_branch(x, y, depth, length, direction="x", parent_id=None):
            if depth >= levels:
                return
            
            if direction == "x":
                # Extend in the x direction by the length of the longest y branch
                end_x = x + length
                last_node_id = parent_id
                for new_x in np.arange(x + node_spacing, end_x + node_spacing, node_spacing):
                    next_coords = [float(new_x), float(y), 0.0]
                    self._add_node(next_coords)
                    if last_node_id is not None:
                        self.graph.add_edge(last_node_id, self.node_id)
                    last_node_id = self.node_id
                    self.node_id += 1
                add_branch(end_x, y, depth + 1, length / 2, "y", last_node_id)
            elif direction == "y":
                # Extend in both positive and negative y directions, with reduced length
                end_y_pos = y + length
                end_y_neg = y - length
                last_node_id_pos = parent_id
                for new_y in np.arange(y + node_spacing, end_y_pos + node_spacing, node_spacing):
                    next_coords = [float(x), float(new_y), 0.0]
                    self._add_node(next_coords)
                    if last_node_id_pos is not None:
                        self.graph.add_edge(last_node_id_pos, self.node_id)
                    last_node_id_pos = self.node_id
                    self.node_id += 1
                add_branch(x, end_y_pos, depth, length, "x", last_node_id_pos)
                
                last_node_id_neg = parent_id
                for new_y in np.arange(y - node_spacing, end_y_neg - node_spacing, -node_spacing):
                    next_coords = [float(x), float(new_y), 0.0]
                    self._add_node(next_coords)
                    if last_node_id_neg is not None:
                        self.graph.add_edge(last_node_id_neg, self.node_id)
                    last_node_id_neg = self.node_id
                    self.node_id += 1
                add_branch(x, end_y_neg, depth, length, "x", last_node_id_neg)
        
        # Scale the initial length
        initial_length = scale_factor * (2**(levels - 1))
        add_branch(0.0, 0.0, 0, initial_length, "x", 0)
        
        return self.graph

    def plot_graph(self, title="Network"):
        """
        Plots the generated network.
        
        Args:
            title (str, optional): The title of the plot. Defaults to "Network".
        """

        plt.figure(figsize=(10, 8))
        
        # Use node coordinates as positions
        pos = {node: (data['coords'][0], data['coords'][1]) for node, data in self.graph.nodes(data=True)}
        
        # Draw the graph with fixed positions and equal aspect ratio
        nx.draw(self.graph, pos, with_labels=True, node_size=50, node_color="lightgreen", font_size=10, font_weight="bold")
        
        plt.title(title)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

# Example usage
# generator = NetworkGenerator()
# G1 = generator.create_linear_network_with_deadends(total_length=100, spacing=1, n_dead_ends=20, length_deadends=5)
# generator.plot_graph(title="Linear Network with Dead-ends")

# G2 = generator.create_rectilinear_network(levels=5, scale_factor=100, node_spacing=20)
# generator.plot_graph(title="Symmetric Rectilinear Branchwork with Scaled Lengths and User-Provided Node Spacing")