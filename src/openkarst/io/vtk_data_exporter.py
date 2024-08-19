#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:46:32 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import os
import numpy as np
import pyvista as pv

class VtkDataExporter:
    """
    A class to export simulation results to VTK files.

    This class takes in a network geometry, flow rate history, water depth history,
    and time history, optionally inflow and outflow node indices, and exports the results to
    VTK files for visualization in ParaView.

    Attributes:
        output_dir (str): Directory where VTK files will be saved.

    Methods:
        export(cn_geometry, Q_history, y_history, t_history, inflow_nodes=None, outflow_nodes=None):
            Exports the simulation results to VTK files. If inflow or outflow nodes are specified,
            it also exports a static VTK file with inflow and outflow node markers.
    """
    
    def __init__(self, output_dir: str):
        """
        Initializes the VTKExporter with the output directory.

        Args:
            output_dir (str): Directory where VTK files will be saved.
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def vtk_safe(self, name):
        """Convert a string to a valid VTK field name."""
        import re
        if not re.match(r'^[a-zA-Z]', name):
            name = 'field_' + name
        name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
        return name

    def export(self, cn_geometry, Q_history, y_history, t_history, inflow_nodes=None, outflow_nodes=None):
        """
        Exports the simulation results to VTK files.

        Args:
            cn_geometry (openpnm.network): The network geometry.
            Q_history (list): Flow rate history for each edge.
            y_history (list): Water depth history for each node.
            t_history (list): Time history for the simulation.
            inflow_nodes (list, optional): List of inflow node indices. Defaults to None.
            outflow_nodes (list, optional): List of outflow node indices. Defaults to None.
        """
        points = cn_geometry['pore.coords']
        connections = cn_geometry['throat.conns']
        
        # Save static node markers if inflow or outflow nodes are provided
        if inflow_nodes is not None or outflow_nodes is not None:
            self.export_static_node_markers(cn_geometry, inflow_nodes, outflow_nodes)
        
        for i, (Q, y, t) in enumerate(zip(Q_history, y_history, t_history)):
            polydata = pv.PolyData()
            polydata.points = points
            lines = np.hstack([np.full((connections.shape[0], 1), 2), connections]).flatten()
            polydata.lines = lines
            polydata.cell_data[self.vtk_safe('Flowrate')] = Q
            polydata.point_data[self.vtk_safe('WaterDepth')] = y
        
            if 'throat.epsilon' in cn_geometry:
                polydata.cell_data[self.vtk_safe('Epsilon roughness')] = cn_geometry['throat.epsilon']
            if 'throat.diameters' in cn_geometry:
                polydata.cell_data[self.vtk_safe('Diameter')] = cn_geometry['throat.diameters']
                
            # Add time as field data
            polydata.field_data['Time'] = np.array([t], dtype=float)
            
            filename = os.path.join(self.output_dir, f'output_timestep_{i:04d}.vtk')
            polydata.save(filename)

    def export_static_node_markers(self, cn_geometry, inflow_nodes=None, outflow_nodes=None):
        """
        Exports a static VTK file with inflow and outflow node markers.

        Args:
            cn_geometry (openpnm.network): The network geometry.
            inflow_nodes (list, optional): List of inflow node indices. Defaults to None.
            outflow_nodes (list, optional): List of outflow node indices. Defaults to None.
        """
        points = cn_geometry['pore.coords']
        polydata = pv.PolyData()
        polydata.points = points

        # Optionally add inflow nodes
        if inflow_nodes is not None:
            inflow_array = np.zeros(points.shape[0], dtype=int)
            inflow_array[inflow_nodes] = 1
            polydata.point_data[self.vtk_safe('InflowNodes')] = inflow_array

        # Optionally add outflow nodes
        if outflow_nodes is not None:
            outflow_array = np.zeros(points.shape[0], dtype=int)
            outflow_array[outflow_nodes] = 1
            polydata.point_data[self.vtk_safe('OutflowNodes')] = outflow_array
        
        # Save the static node markers to a VTK file
        static_filename = os.path.join(self.output_dir, 'static_node_markers.vtk')
        polydata.save(static_filename)

            
          