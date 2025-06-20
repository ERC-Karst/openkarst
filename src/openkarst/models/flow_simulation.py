#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:56:06 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import numpy as np
import math
import scipy.optimize as optimize

from termcolor import colored
from typing import Optional, Dict, Any

from openkarst.config.physical_properties import PhysicalProperties
from openkarst.config.solver_settings import SolverSettings
from openkarst.config.simulation_settings import SimulationSettings
from openkarst.config.validate_settings import validate_settings
from openkarst.config.apply_settings import apply_settings

from openkarst.io.results_handling import initialize_results_container, store_results

from openkarst.utils.helpers import time_this
from openkarst.utils.logging_config import setup_logging

from openkarst.models.boundary_conditions import ConstantBC, BoxBC, TimeSeriesBC


class FlowSimulation:
    """
    Simulates free surface and pressurized flow through a karst conduit network using the
    dynamic wave equation.

    This class models the flow of water through a network of conduits, considering both
    free surface and pressurized flow conditions. It utilizes the dynamic wave equation
    to compute the flow rates, water depths, and other relevant properties over time.

    Attributes:
        GEOMETRY_CHANNEL (int): Indicator for channel geometry. Set to 1 for channel validation.
        logger (Logger): Logger instance for logging simulation information and debugging.
        physical_properties (PhysicalProperties): Object containing the physical properties of the simulation.
        solver_settings (SolverSettings): Object containing the solver settings.
        simulation_settings (SimulationSettings): Object containing the simulation settings.
        logging_settings (LoggingSettings): Object containing the logger settings.
        network (openpnm.network.GenericNetwork): The OpenPNM network used in the simulation.
        waterdepth_boundary (dict): Dictionary storing water depth boundary conditions.
        inflow_boundary (dict): Dictionary storing inflow boundary conditions.
        critical_depth_boundary (dict): Dictionary storing critical depth boundary conditions.

    Methods:
        __init__(self, openpnm_network, physical_properties=None, solver_settings=None, simulation_settings=None):
            Initializes the FlowSimulation class with the provided settings and network.
        _initialize_arrays(self):
            Initializes the arrays used in the simulation.
        _initialize_conduit_properties(self):
            Initializes the properties of the conduits in the network.
        set_initial_conditions(self, initial_Q, initial_y):
            Sets the initial conditions for the simulation.
        set_boundary_conditions(self, waterdepth_boundary=None, inflow_boundary=None, critical_depth_boundary=None):
            Sets the boundary conditions for the simulation.
        run_simulation(self, desired_outputs):
            Runs the simulation and returns the results.
        _dynamic_wave(self):
            Computes the dynamic wave for the current timestep.
        _update_network_state(self):
            Updates the state of the network with the new values.
        _initialize_state_variables(self):
            Initializes the state variables at the beginning of each timestep.
        _compute_conduit_state(self, y1, y2, y_mid):
            Computes the pressurization state of the conduits.
        _compute_surface_area(self, y1, y2, y_mid):
            Computes the surface areas and widths of the conduits.
        _compute_discharge_areas(self, y1, y2, y_mid, slot_widths):
            Computes the discharge areas of the conduits.
        _compute_alpha(self, froude_number):
            Computes the alpha value for upstream weighting.
        _compute_new_dt(self, v_mid, froude):
            Computes the new timestep based on the Courant number and Froude number.
        _compute_hydraulic_radius(self, flow_depths, flow_areas, slot_width, is_full):
            Computes the hydraulic radius of the conduits.
        _compute_upstream_weighted_radii(self, r1, r2, r_mid, h1, h2, alpha):
            Computes the upstream weighted hydraulic radii.
        _compute_upstream_weighted_areas(self, a1, a2, a_mid, h1, h2, alpha):
            Computes the upstream weighted areas.
        _compute_flows(self, a1, a2, a_mid_upwtd, r_mid_upwtd, r_mid, h1, h2, alpha, v_mid, w_mid):
            Computes the flow rates for the conduits.
        _compute_water_depths(self, n_surface_a):
            Computes the water depths at each node.
        _adjust_flowrates_dry_nodes(self):
            Adjusts the flow rates for dry nodes.
        _print_timestep_info(self, iteration, froude):
            Prints information about the current timestep.
        _check_picard_convergence(self):
            Checks the convergence of the Picard iterations.
        _compute_error_norms(self):
            Computes the L2 and MAD error norm 
        _check_steady_state_convergence(self):
            Checks the convergence to steady state using the L2 and MAD norm.
        _flow_area_cdepth(self, depth, diameter):
            Computes the flow area at critical depth.
        _wetted_perimeter_cdepth(self, depth, diameter):
            Computes the wetted perimeter at critical depth.
        _hydraulic_radius_cdepth(self, depth, diameter):
            Computes the hydraulic radius at critical depth.
        _critical_depth(self, depth, Q, g, diameter):
            Computes the critical depth for a given flow rate.
        _find_critical_depth(self, Q, diameter, g=9.81):
            Finds the critical depth for a given flow rate and diameter.
        __del__(self):
            Destructor for the FlowSimulation class, closes the logger.
    """
    
    #GEOMETRY_CHANNEL = 1     # Geometry = 1 for channel validation
    
    def __init__(self, openpnm_network,
                 physical_properties: Optional[Dict[str, Any]] = None,
                 solver_settings: Optional[Dict[str, Any]] = None,
                 simulation_settings: Optional[Dict[str, Any]] = None,
                 logging_settings: Optional[Dict[str, Any]] = None):
        """
        Initializes the FlowSimulation class with provided settings and network.

        Args:
            openpnm_network: The OpenPNM network to be used in the simulation.
            physical_properties (Optional[Dict[str, Any]]): Physical properties settings.
            solver_settings (Optional[Dict[str, Any]]): Solver settings.
            simulation_settings (Optional[Dict[str, Any]]): Simulation settings.
            logging_settings (Optional[Dict[str, Any]]): Logging configuration settings.
        
        Attributes:
            logger: Logger for logging information and debugging.
            physical_properties: Instance of PhysicalProperties with provided or default settings.
            solver_settings: Instance of SolverSettings with provided or default settings.
            simulation_settings: Instance of SimulationSettings with provided or default settings.
            network: The OpenPNM network to be used in the simulation.
            waterdepth_boundary (dict): Dictionary for water depth boundary conditions.
            inflow_boundary (dict): Dictionary for inflow boundary conditions.
            critical_depth_boundary (dict): Dictionary for critical depth boundary conditions.
        """
        
        # Set up logger
        self.logger = setup_logging(logging_settings)
        
        self.physical_properties = (PhysicalProperties(**physical_properties) 
                                    if physical_properties else PhysicalProperties()
                                    )
        self.solver_settings = (SolverSettings(**solver_settings)
                                if solver_settings else SolverSettings()
                                )
        self.simulation_settings= (SimulationSettings(**simulation_settings)
                                   if simulation_settings else SimulationSettings()
                                   )

        validate_settings(self.physical_properties,
                          self.solver_settings,
                          self.simulation_settings,
                          self.logger
                          )
        
        apply_settings(self,
                       self.physical_properties,
                       self.solver_settings,
                       self.simulation_settings,
                       self.logger
                       )
        
        #Get OpenPNM network (this will later come from another class)
        self.network = openpnm_network
        
        # Constant boundary conditions dictionary {node_index: value}
        #self.waterdepth_boundary = {}
        #self.inflow_boundary = {}
        self.critical_depth_boundary = {}
        
        # Stop conditions dictionary
        self.flowrate_condition = {}
        # Initialize in case user does not specify this
        self.stop_condition_set = False
        
        self._initialize_arrays()
        
        self._initialize_conduit_properties()
        
        self.logger.info('FlowSimulation initialized with physical properties: %s',
                         self.physical_properties)
        self.logger.info('FlowSimulation initialized with solver settings: %s',
                         self.solver_settings)
        self.logger.info('FlowSimulation initialized with simulation settings: %s',
                         self.simulation_settings)
        
        
    def _initialize_arrays(self):
        """
        Initialize arrays for flow simulation.

        This method initializes various arrays used in the flow simulation 
        process, including arrays for flow rates, water depths, discharge 
        areas, change in flow, and pressurization states. It also sets up 
        arrays for node indices and node heights.
    
        Initializes:
            Q (ndarray): Array for flow rates.
            Q_new (ndarray): Array for updated flow rates.
            Q_prev_i (ndarray): Array for previous iteration flow rates.
            Q_old_t (ndarray): Array for old time step flow rates.
            y (ndarray): Array for water depths.
            y_new (ndarray): Array for updated water depths.
            y_prev_i (ndarray): Array for previous iteration water depths.
            y_old_t (ndarray): Array for old time step water depths.
            dydt (ndarray): Array for rate of change of water depths.
            a_mid (ndarray): Array for discharge areas.
            a_mid_new (ndarray): Array for updated discharge areas.
            a_mid_old_t (ndarray): Array for old time step discharge areas.
            dQ (ndarray): Array for change in flow.
            dQ_new (ndarray): Array for updated change in flow.
            dQ_old_t (ndarray): Array for old time step change in flow.
            is_full_y1 (ndarray): Array indicating pressurization state of y1.
            is_full_y2 (ndarray): Array indicating pressurization state of y2.
            is_full_y_mid (ndarray): Array indicating pressurization state of y_mid.
            n_indices1 (ndarray): Array of node indices 1.
            n_indices2 (ndarray): Array of node indices 2.
            Z (ndarray): Array of node heights.
            z1 (ndarray): Array of node heights for n_indices1.
            z2 (ndarray): Array of node heights for n_indices2.
        """

        self.Q = np.zeros(self.network.Nt, dtype=float)  # Flow rates
        self.Q_new = np.zeros(self.network.Nt, dtype=float)
        self.Q_prev_i = np.zeros(self.network.Nt, dtype=float)
        self.Q_old_t = np.zeros(self.network.Nt, dtype=float)
        
        self.y = np.zeros(self.network.Np, dtype=float)  # Water depths
        self.y_new = np.zeros(self.network.Np, dtype=float)
        self.y_prev_i = np.zeros(self.network.Np, dtype=float)
        self.y_old_t = np.zeros(self.network.Np, dtype=float)
        self.dydt = np.zeros(self.network.Np, dtype=float)
        
        self.a_mid = np.zeros(self.network.Nt, dtype=float) # Discharge areas           
        self.a_mid_new = np.zeros(self.network.Nt, dtype=float)
        self.a_mid_old_t = np.zeros(self.network.Nt, dtype=float)
        
        self.dQ = np.zeros(self.network.Np, dtype=float) # Change in flow 
        self.dQ_new = np.zeros(self.network.Np, dtype=float)
        self.dQ_old_t = np.zeros(self.network.Np, dtype=float)
        
        self.is_full_y1 = np.full(self.network.Nt, False, dtype=bool) # Pressurization state
        self.is_full_y2 = np.full(self.network.Nt, False, dtype=bool)
        self.is_full_y_mid = np.full(self.network.Nt, False, dtype=bool)

        self.n_indices1, self.n_indices2 = self.network.conns.T
        
        # Compute coordination numbers for all nodes (used for recharge computation)
        self.coordination_numbers = np.zeros(self.network.Np, dtype=int)
        for node in range(self.network.Np):
            self.coordination_numbers[node] = np.sum(
                (self.n_indices1 == node) | (self.n_indices2 == node)
            )
        
        # Node heights
        self.Z = np.full(self.network.Np, self.network['pore.coords'][:, 2], dtype=float)
        self.z1 = self.Z[self.n_indices1]
        self.z2 = self.Z[self.n_indices2]
        
        self.logger.info('Arrays initialized')
        
         
    def _initialize_conduit_properties(self):
        """
        Initialize the conduit or channel properties for the flow simulation.

        This method sets up various properties related to the conduits/channels in the 
        network, including diameters, lengths, roughness coefficients, and 
        Manning coefficients. It also initializes and updates the maximum 
        depth for each node based on the connected conduits. The method computes the equivalent
        Manning coefficient used for pressurized conduits. In the case of open channel flow the 
        Manning coefficient is directly applied instead of calculating an equivalent one based
        on the friction coefficient and given epsilon values of the conduits. The method also
        logs the initialization status.

        
        """
        if self.geometry_channel == True:
            
            # Get Manning coefficient from physical property settings
            self.conduit_manning = np.full(self.network.Nt, self.channel_manning, dtype=float)
            
            self.conduit_lengths = np.full(
                self.network.Nt, self.network['throat.lengths'], dtype=float
            )
            
            # Set max_depths to a default value for open channel flow
            # This can affect the calculation of the second adaptive dt criterion
            self.max_depths = np.full(self.network.Np, 1.0, dtype=float)
            
            self.Re_conduit = np.zeros(self.network.Nt, dtype=float)
        
        else:
                
            self.conduit_diameters = np.full(
                self.network.Nt, self.network['throat.diameters'], dtype=float
            )
            self.conduit_lengths = np.full(
                self.network.Nt, self.network['throat.lengths'], dtype=float
            )
            self.conduit_epsilon = np.full(
                self.network.Nt, self.network['throat.epsilon'], dtype=float
            )
             
            # Initialize array to store the maximum depth for each node
            # At each node max_depth is the diameter of the largest connected conduit
            self.max_depths = np.zeros(self.network.Np)
            
            # Update max_depth based on the connected conduits
            # For nodes connected at n_indices1
            for i, node in enumerate(self.n_indices1):
                self.max_depths[node] = max(self.max_depths[node], self.conduit_diameters[i])
            # For nodes connected at n_indices2
            for i, node in enumerate(self.n_indices2):
                self.max_depths[node] = max(self.max_depths[node], self.conduit_diameters[i])
            
            self.Re_conduit = np.zeros(self.network.Nt, dtype=float)
            
            # Compute equivalent Manning coefficient at f(epsilon, Re->infty)
            # This is the equivalent Manning coefficient used for pressurized conduits
            RE_INFTY = 1e7
            f = self._compute_friction_churchill(RE_INFTY)
            self.conduit_manning = (
               1 / (np.sqrt(8 * self.gravity)) 
               * np.sqrt(f) 
               * (0.5 * self.conduit_diameters)**(1 / 3)
            )
        
        
            
        self.logger.info('Conduit properties initialized')
            
    def _compute_friction_churchill(self, Reynolds):
        """
        Compute the friction factor using Churchill's equation.

        This method calculates the friction factor for turbulent flow 
        using Churchill's equation, which provides a smooth transition 
        between laminar and turbulent flow regimes.

        Args:
            Reynolds (float): The Reynolds number for the flow in the conduit.

        Returns:
            float: The calculated friction factor.
        """
        
        C = (7/Reynolds)**0.9 + 0.27*self.conduit_epsilon/(self.conduit_diameters)
        A = (-2.457*np.log(C))**16
        B = (37530/Reynolds)**16
        f = 8*((8/Reynolds)**12 + 1/(A + B)**1.5)**(1/12)
        
        return f
    
            
    def run_simulation(self, desired_outputs: Dict[str, bool]):
        """
        Run the flow simulation.

        This method performs the dynamic wave computation for each time step
        until the simulation reaches a steady state or the maximum time is exceeded.

        Args:
            desired_outputs (Dict[str, bool]): Dictionary specifying which results
            to store and the output interval.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the simulation results
            stored as numpy arrays.
        """
        
        results_container = initialize_results_container(desired_outputs, self.logger)
        output_interval = desired_outputs.get('output_interval', 1.0)
        next_output_time = output_interval
        
        with time_this('run_simulation'):

            self.convergence_fails = 0
            self.dt = self.dt_init
            self.current_time = 0.0
            self.current_timestep = 0
            self.relative_l2_norm = 0.0
            self.relative_mad_norm = 0.0
            
            while True:
                self._initialize_state_variables()
               
                # Perform the dynamic wave computation for the current time step
                converged = self._dynamic_wave()
                
                if not converged:
                    print(colored(
                        f'[run_simulation] Not converged at time = '
                        f'{self.current_time:.1f}', 'red'
                    ))
                    self.convergence_fails += 1
                    
                # Update the network state with the new values
                self._update_network_state()

                # Compute L2 and MAD error norms for each timestep
                self._compute_error_norms()
                
                # Compute new step size based on Froude and Courant number 
                if self.adaptive_timesteps:
                    self._compute_new_dt(self._v_mid_last, self._froude_last)
        
                # Store the results if the current time exceeds the next output interval
                if self.current_time >= next_output_time:
                    results_container = store_results(self, results_container)
                    next_output_time += output_interval
                
                # Check flowrate stop condition if set
                if self.stop_condition_set:
                    for node_index, flowrate_value in self.flowrate_condition.items():
                        connected_conduits = np.where(
                            (self.n_indices1 == node_index) | (self.n_indices2 == node_index)
                        )[0]
                        
                        total_flowrate = np.sum(np.abs(self.Q_new[connected_conduits]))
                        
                        if math.fmod(self.current_timestep, self.print_info_interval) == 0:
                            
                            print(f'Outflow rate = {100 * (total_flowrate / flowrate_value):.2f}%')
                                        
                        if total_flowrate > self.flowrate_threshold * flowrate_value:
                            self.logger.info(
                                f'Flowrate threshold reached at node {node_index}: Simulation finished at time = {self.current_time:.2f}s'
                            )
                           
                            percentage_fails = 100 * self.convergence_fails / self.current_timestep
                            self.logger.info('Percentage convergence fails: {:.2f}'.format(percentage_fails))
                            print(f'[run_simulation] Percentage convergence fails = {percentage_fails:.2f}%')
                            print(colored('[run_simulation] Flowrate threshold reached (99% of flow rate)', 'green'))
                           
                            break
                    else:
                        # Continue the loop if the break was not triggered
                        self.current_time += self.dt
                        self.current_timestep += 1
                        continue
                    break
                
                # Check if steady-state achieved and exit (only if stop condition not set)
                if not self.stop_condition_set and self.steady_state:
                    if self._check_steady_state_convergence() and self.current_timestep > 1:
                        self.logger.info(
                            f'Steady state reached: Simulation finished at time = {self.current_time:.2f}s'
                        )
                       
                        percentage_fails = 100 * self.convergence_fails / self.current_timestep
                        self.logger.info('Percentage convergence fails: {:.2f}'.format(percentage_fails))
                        print(f'[run_simulation] Percentage convergence fails = {percentage_fails:.2f}%')
                        print(colored('[run_simulation] Steady state reached', 'green'))
                       
                        break
                
                # Increment the time by dt
                self.current_time += self.dt
                self.current_timestep += 1
                
                # If no steady-state simulation, exit when t_max is exceeded (only if stop condition not set)
                if not self.stop_condition_set and not self.steady_state and self.current_time > self.t_max:
                    
                    self.logger.info(
                        f'Maximum time reached: Simulation finished at time = {self.current_time:.2f}s'
                    )
                    
                    percentage_fails = 100 * self.convergence_fails / self.current_timestep
                    self.logger.info('Percentage convergence fails: {:.2f}'.format(percentage_fails))
                    print(colored('[run_simulation] t_max reached', 'green'))
                    print(f'[run_simulation] Percentage convergence fails = {percentage_fails:.2f}%')
                    
                    break
                
        # Convert lists to numpy arrays
        for key in results_container:
            results_container[key] = np.array(results_container[key])
        self.logger.info('Results stored.')
     
        return results_container
    
    def __del__(self):
        """
        Destructor to close all handlers associated with the logger.

        This method ensures that all handlers associated with the logger
        are properly closed when the object is deleted.
        """
    
        self.logger.info('Logger closed. Object deleted.\n')
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
    
    def set_initial_conditions(self, initial_Q, initial_y):
        """
        Set the initial conditions for flow rates and water depths.

        This method sets the initial flow rates and water depths for the
        simulation.

        Args:
            initial_Q (array-like): Initial flow rates for the conduits.
            initial_y (array-like): Initial water depths for the nodes.
        """
        
        np.copyto(self.Q, initial_Q)
        np.copyto(self.y, initial_y)


    def set_waterdepth_BC(self, nodes, values, mode='add'):
        """
        Set water depth boundary conditions at specified nodes.

        Parameters
        ----------
        nodes : int or list of int
            Index or indices of nodes to which water depth boundary conditions are applied.

        values : float, tuple, or list
            Values to assign to each node. If a single value is provided, it will be broadcast 
            to all nodes. Supported formats include:
            - float: constant water depth.
            - tuple:
                - ('ramp', v0, v1, t0, t1): ramps linearly from v0 to v1 between t0 and t1.
                - ('timeseries', times, values): interpolated time series using numpy.interp.
                - ('box', value, t0, t1 [, value_before=0.0, value_after=0.0]):
                  constant value applied between t0 and t1; optional values before and after.

        mode : str, optional
            Defines how the new BCs should interact with existing ones:
            - 'add' (default): adds new BCs; raises error if a BC already exists at the node.
            - 'overwrite': replaces any existing BC at the specified nodes.
            - 'remove': removes BCs from the specified nodes.
        """

        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}

        if 'waterdepth' not in self.boundary_conditions:
            self.boundary_conditions['waterdepth'] = []

        if not isinstance(nodes, list):
            nodes = [nodes]

        if mode == 'remove':
            self.boundary_conditions['waterdepth'] = [
                bc for bc in self.boundary_conditions['waterdepth']
                if all(n not in nodes for n in bc.target_ids)
            ]
            return

        if not isinstance(values, list):
            values = [values] * len(nodes)

        for node, val in zip(nodes, values):
            # Remove existing BC if overwrite
            if mode == 'overwrite':
                self.boundary_conditions['waterdepth'] = [
                    bc for bc in self.boundary_conditions['waterdepth']
                    if node not in bc.target_ids
                ]
            elif mode == 'add':
                for bc in self.boundary_conditions['waterdepth']:
                    if node in bc.target_ids:
                        raise ValueError(f"Water depth BC already exists at node {node}. Use mode='overwrite' to replace it.")

            # Create new BC object
            if isinstance(val, (int, float)):
                bc = ConstantBC([node], value=val)
            elif isinstance(val, tuple) and val[0] == 'box':
                _, v_during, t0, t1, *rest = val
                v_before = rest[0] if len(rest) > 0 else 0.0
                v_after = rest[1] if len(rest) > 1 else 0.0
                bc = BoxBC([node], v_during, t0, t1, v_before, v_after)
            elif isinstance(val, tuple) and val[0] == 'timeseries':
                _, times, vals = val
                bc = TimeSeriesBC([node], times=times, values=vals)
            else:
                raise ValueError(f"Unrecognized value for BC at node {node}: {val}")

            self.boundary_conditions['waterdepth'].append(bc)


    def set_inflow_BC(self, nodes, values, mode='add', inflow_type='volumetric'):
        """
        Set inflow boundary conditions at specified nodes.

        Parameters
        ----------
        nodes : int or list of int
            Index or indices of nodes where inflow boundary conditions are applied.

        values : float, tuple, or list
            Values to assign to each node. If a single value is provided, it will be broadcast 
            to all specified nodes. Supported formats include:
            - float: constant inflow (in m³/s if 'volumetric', or m/s if 'flux').
            - tuple:
                - ('ramp', q0, q1, t0, t1): ramps linearly from q0 to q1 between t0 and t1.
                - ('timeseries', times, values): interpolated time series using numpy.interp.
                - ('box', value, t0, t1 [, value_before=0.0, value_after=0.0]):
                  constant value between t0 and t1, with optional values before and after.

        mode : str, optional
            Defines how the new BCs should interact with existing ones:
            - 'add' (default): adds new BCs; raises an error if a BC already exists at the node.
            - 'overwrite': replaces any existing BCs at the specified nodes.
            - 'remove': removes BCs from the specified nodes.

        inflow_type : str, optional
            Specifies the type of inflow:
            - 'volumetric' (default): inflow is treated as a total flow rate (m³/s).
            - 'flux': inflow is treated as a flux (m/s) and will be converted to a volumetric
              flow during the simulation using the local geometry (e.g., conduit area).
        """

        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}

        if 'inflow' not in self.boundary_conditions:
            self.boundary_conditions['inflow'] = []

        if not isinstance(nodes, list):
            nodes = [nodes]

        if mode == 'remove':
            self.boundary_conditions['inflow'] = [
                bc for bc in self.boundary_conditions['inflow']
                if all(n not in nodes for n in bc.target_ids)
            ]
            return

        if not isinstance(values, list):
            values = [values] * len(nodes)

        for node, val in zip(nodes, values):
            if mode == 'overwrite':
                self.boundary_conditions['inflow'] = [
                    bc for bc in self.boundary_conditions['inflow']
                    if node not in bc.target_ids
                ]
            elif mode == 'add':
                for bc in self.boundary_conditions['inflow']:
                    if node in bc.target_ids:
                        raise ValueError(f"Inflow BC already exists at node {node}. Use mode='overwrite' to replace it.")

            # Create BC object and attach type
            if isinstance(val, (int, float)):
                bc = ConstantBC([node], value=val, bc_type=inflow_type)
            elif isinstance(val, tuple) and val[0] == 'box':
                _, v_during, t0, t1, *rest = val
                v_before = rest[0] if len(rest) > 0 else 0.0
                v_after = rest[1] if len(rest) > 1 else 0.0
                bc = BoxBC([node], v_during, t0, t1, v_before, v_after, bc_type=inflow_type)
            elif isinstance(val, tuple) and val[0] == 'timeseries':
                _, times, flow_values = val
                bc = TimeSeriesBC([node], times=times, values=flow_values,
                                bc_type=inflow_type)
            else:
                raise ValueError(f"Unrecognized inflow BC format at node {node}: {val}")

            self.boundary_conditions['inflow'].append(bc)


    # def set_boundary_conditions(
    #         self,
    #         waterdepth_boundary=None,
    #         inflow_boundary=None,
    #         critical_depth_boundary=None,
    #         inflow_type='constant',  # constant, constant_timespan or ramp
    #         start_time=0,            # Start time for constant_timespan or ramped inflow
    #         end_time=200             # End time for constant_timespan or ramped inflow
    # ):
    #     """
    #     Set the boundary conditions for the flow simulation.
    
    #     Args:
    #         waterdepth_boundary (dict, optional): Dictionary of water depth 
    #             boundary conditions {node_index: value}.
    #         inflow_boundary (dict, optional): Dictionary of inflow boundary 
    #             conditions {node_index: value or (initial_rate, peak_rate)}.
    #         critical_depth_boundary (dict, optional): Dictionary of critical 
    #             depth boundary conditions {node_index: value}.
    #         inflow_type (str, optional): Type of inflow ('constant', 'constant_timespan', or 'ramp').
    #         start_time (float, optional): Start time for constant_timespan or ramped inflow.
    #         end_time (float, optional): End time for constant_timespan or ramped inflow.
    #     """
        
    #     if waterdepth_boundary is not None:
    #         self.waterdepth_boundary = waterdepth_boundary
    
    #     if inflow_boundary is not None:
    #         self.inflow_boundary = inflow_boundary
    
    #     if critical_depth_boundary is not None:
    #         self.critical_depth_boundary = critical_depth_boundary
    

    #     self.inflow_type = inflow_type
    #     self.start_time = start_time
    #     self.end_time = end_time
       
            
    def set_stop_conditions(self, flowrate_condition=None, flowrate_threshold=0.98):
        
        if flowrate_condition is not None:
            self.flowrate_condition = flowrate_condition
            self.flowrate_threshold = flowrate_threshold
            self.stop_condition_set = True
        else:
            self.stop_condition_set = False
        

        
    def _initialize_state_variables(self):
        """
        Initialize state variables for the simulation.

        This method copies the current state variables to their respective previous and
        old state variables to prepare for the next iteration of the simulation.
        """
        
        np.copyto(self.Q_old_t, self.Q)
        np.copyto(self.Q_prev_i, self.Q)
        np.copyto(self.y_new, self.y)
        np.copyto(self.y_old_t, self.y)
        np.copyto(self.y_prev_i, self.y)  
        np.copyto(self.a_mid_old_t, self.a_mid) 
        np.copyto(self.dQ_old_t, self.dQ)
         
    def _update_network_state(self):
        """
        Update the network state properties.

        This method updates the state properties of the network by assigning the new values
        to the current state variables.
        """
        
        self.Q = self.Q_new
        self.y = self.y_new
        self.a_mid = self.a_mid_new
        self.dQ = self.dQ_new

    def _dynamic_wave(self):
        """
        Perform the dynamic wave computation for the current time step.

        This method iterates through the dynamic wave computation, updating flow depths,
        velocities, discharge areas, and hydraulic properties at each time step until 
        convergence is achieved or the maximum number of iterations is reached.
    
        The computation includes:
        - Setting flow depths to a minimum value.
        - Retrieving water depths at both ends and the middle of the conduits.
        - Determining conduit state (full or not).
        - Computing surface areas and widths.
        - Calculating discharge areas.
        - Computing velocity and Froude number at conduit center.
        - Calculating upstream weighting factors.
        - Adapting time steps based on Froude and Courant numbers.
        - Computing hydraulic radii.
        - Updating flow components and flow rates.
        - Adjusting flow rates for dry nodes.
        - Printing timestep information.
        - Checking for Picard convergence.
    
        Returns:
            bool: True if the computation converges within the maximum number of iterations,
            False otherwise.
        """
        
        iteration = 0

        while iteration < self.max_iterations:
             
            # Set flow depths to a minimum value
            self.y_new[self.y_new <= self.min_waterdepth] = self.min_waterdepth
            
            y1, y2, y_mid = self._get_water_depths()
            
            h1, h2 = self._get_hydraulic_heads(y1, y2)
            
            # Determine conduit state (i.e. full or not)
            self._compute_conduit_state(y1, y2, y_mid)
            
            # Compute surface areas and surface widths
            (n_surface_a,slot_w1, slot_w2,
             slot_w_mid, w_mid) = self._compute_surface_area(y1, y2, y_mid)
            
            # Compute discharge areas
            a1, a2, self.a_mid_new = self._compute_discharge_areas(y1, y2, y_mid, slot_w_mid)
            
    
            
            # Compute velocity and Frounde number at conduit center 
            v_mid = self.Q_prev_i / self.a_mid_new
            froude = np.abs(v_mid) / np.sqrt(self.gravity *  self.a_mid_new / w_mid)

            # Store to use for adaptive timestep update outside of dynamic_wave
            self._v_mid_last = v_mid
            self._froude_last = froude  
           
            # Compute alpha for upstream weighting
            alpha = self._compute_alpha(froude)
            
            #if self.adaptive_timesteps:
            #    # Compute new step size based on Froude and Courant number
            #    self._compute_new_dt(v_mid, froude)
            
            # Compute hydraulic radii at both ends and middle of conduit
            r1 = self._compute_hydraulic_radius(y1, a1, slot_w1, self.is_full_y1)
            r2 = self._compute_hydraulic_radius(y2, a2, slot_w2, self.is_full_y2)
            r_mid = self._compute_hydraulic_radius(y_mid, self.a_mid_new, slot_w_mid,
                                                   self.is_full_y_mid)
                
            # Compute upstream-weighted hydraulic radii and areas
            r_mid_upwtd = self._compute_upstream_weighted_radii(r1, r2, r_mid, h1, h2, alpha)
            
            a_mid_upwtd = self._compute_upstream_weighted_areas(a1, a2, self.a_mid_new,
                                                                h1, h2, alpha)
            
            # Compute dQ components and new flows
            self._compute_flows(a1, a2, a_mid_upwtd, r_mid_upwtd, r_mid, h1, h2,alpha, v_mid, w_mid)
            
            self._compute_water_depths(n_surface_a)

            self._adjust_flowrates_dry_nodes()
            
            self._print_timestep_info(iteration, froude)
            
            if self._check_picard_convergence():
                return True
            
            # Update iteration state variables 
            np.copyto(self.Q_prev_i, self.Q_new)
            np.copyto(self.y_prev_i, self.y_new)
            
            iteration += 1

        return False  # if max_iterations hit

    def _get_water_depths(self):
        """
        Calculate water depths at both ends and the middle of the conduits.

        This method retrieves the water depths at both ends of the conduits and
        computes the average water depth in the middle.

        Returns:
            tuple: A tuple containing three numpy arrays:
                - y1 (numpy.ndarray): Water depths at the first end of the conduits.
                - y2 (numpy.ndarray): Water depths at the second end of the conduits.
                - y_mid (numpy.ndarray): Average water depths at the middle of the conduits.
        """
    
        y1 = self.y_new[self.n_indices1]
        y2 = self.y_new[self.n_indices2]
        y_mid = (y1 + y2) * 0.5
        
        return y1, y2, y_mid
    
    def _get_hydraulic_heads(self, y1, y2):
        """
        Calculate hydraulic heads at both ends of the conduits.

        This method calculates the hydraulic heads at both ends of the conduits 
        by adding the water depths to the respective node heights.

        Args:
            y1 (numpy.ndarray): Array of water depths at the first end of the conduits.
            y2 (numpy.ndarray): Array of water depths at the second end of the conduits.

        Returns:
            tuple:
                - h1 (numpy.ndarray): Array of hydraulic heads at the first end of the conduits.
                - h2 (numpy.ndarray): Array of hydraulic heads at the second end of the conduits.
    """
        h1 = y1 + self.z1
        h2 = y2 + self.z2
        
        return h1, h2

    def _compute_conduit_state(self, y1, y2, y_mid):
        """
        Determine the pressurization state of the conduits.

        This method sets the pressurization state (True or False) of the water depths
        at both ends and the middle of the conduits.

        Args:
            y1 (numpy.ndarray): Array of water depths at the first end of the conduits.
            y2 (numpy.ndarray): Array of water depths at the second end of the conduits.
            y_mid (numpy.ndarray): Array of water depths at the middle of the conduits.
        """
        
        if self.geometry_channel == True: # Channel examples never pressurized
            self.is_full_y1.fill(False)
            self.is_full_y2.fill(False)
            self.is_full_y_mid.fill(False)
        
        else:
            self.is_full_y1 = (y1 >= self.conduit_diameters)
            self.is_full_y2 = (y2 >= self.conduit_diameters)
            self.is_full_y_mid = (y_mid >= self.conduit_diameters)
            
 
    def _compute_surface_area(self, y1, y2, y_mid):
        """
        Compute the surface areas and widths of conduits.

        This method calculates the surface areas and widths at both ends and the middle
        of the conduits based on the water depths.

        Args:
            y1 (numpy.ndarray): Array of water depths at the first end of the conduits.
            y2 (numpy.ndarray): Array of water depths at the second end of the conduits.
            y_mid (numpy.ndarray): Array of water depths at the middle of the conduits.

        Returns:
            tuple: A tuple containing the following numpy.ndarrays:
                - n_surface_a: Node surface areas.
                - slot_w1: Preissman slot widths at the first end of the conduits.
                - slot_w2: Preissman slot widths at the second end of the conduits.
                - slot_w_mid: Preissman slot widths at the middle of the conduits.
                - w_mid: Surface widths at the middle of the conduits.
        """
        
        # Initialize surface areas and widths
        surface_a1 = np.zeros(self.network.Nt, dtype=float)
        surface_a2 = np.zeros(self.network.Nt, dtype=float)
        n_surface_a = np.zeros(self.network.Np, dtype=float)
        w1 = np.zeros(self.network.Nt, dtype=float)
        w2 = np.zeros(self.network.Nt, dtype=float)
        w_mid = np.zeros(self.network.Nt, dtype=float)
    
        # Initialize Preissmann slot widths
        slot_w1 = np.zeros(self.network.Nt, dtype=float)
        slot_w2 = np.zeros(self.network.Nt, dtype=float)
        slot_w_mid = np.zeros(self.network.Nt, dtype=float)
    
        if np.any(self.is_full_y1):
            slot_w1[self.is_full_y1] = (
                self._compute_slot_width(
                    y1[self.is_full_y1], self.conduit_diameters[self.is_full_y1]
                )
            )
        if np.any(self.is_full_y2):
            slot_w2[self.is_full_y2] = (
                self._compute_slot_width(
                    y2[self.is_full_y2], self.conduit_diameters[self.is_full_y2]
                )
            )
        if np.any(self.is_full_y_mid): 
            slot_w_mid[self.is_full_y_mid] = (
                self._compute_slot_width(
                    y_mid[self.is_full_y_mid], self.conduit_diameters[self.is_full_y_mid]
                )
            )
    
        # Mask for both nodes being wet
        mask1 = (y1 > self.min_waterdepth) & (y2 > self.min_waterdepth)
        
        if np.any(mask1):
            
            # 1.0 unit width for anayltical solutions, 0.12 for laboratory experiment (Delestre)
            if self.geometry_channel == True:
                if self.channel_type == 'finite':
                    w1[mask1] = self.channel_width
                    w2[mask1] = self.channel_width
                    w_mid[mask1] = self.channel_width
                else:
                    w1[mask1] = 1.0
                    w2[mask1] = 1.0
                    w_mid[mask1] = 1.0  # Default for infinite channel
                
                surface_a1[mask1] = (
                    0.5 * (w1[mask1] + w_mid[mask1]) * (self.conduit_lengths[mask1]/2)
                )
                
                surface_a2[mask1] = (
                    0.5 * (w_mid[mask1] + w2[mask1]) * (self.conduit_lengths[mask1]/2)
                )
                
            else:
                # If flow depths above conduit ceiling set width = Preissman slot width
                # otherwise calculate free surface width
                w1[mask1] = np.where(
                    self.is_full_y1[mask1],
                    slot_w1[mask1],
                    2 * np.sqrt(
                        self.conduit_diameters[mask1] * y1[mask1] - y1[mask1]**2
                    )
                )
                
                w2[mask1] = np.where(
                    self.is_full_y2[mask1],
                    slot_w2[mask1],
                    2 * np.sqrt(
                        self.conduit_diameters[mask1] * y2[mask1] - y2[mask1]**2
                        )
                )
                
                w_mid[mask1] = np.where(
                    self.is_full_y_mid[mask1],
                    slot_w_mid[mask1],
                    2 * np.sqrt(
                        self.conduit_diameters[mask1] * y_mid[mask1] - y_mid[mask1]**2
                        )
                )
            
                # Calculate surface areas
                surface_a1[mask1] = (
                    0.5 * (w1[mask1] + w_mid[mask1]) * (self.conduit_lengths[mask1]/2)
                )
                
                surface_a2[mask1] = (
                    0.5 * (w_mid[mask1] + w2[mask1]) * (self.conduit_lengths[mask1]/2)
                )

        # Calculation when y1 and y2 are below LOWER_LIMIT
        mask2 = (y1 <= self.min_waterdepth) & (y2 <= self.min_waterdepth)
        
        if np.any(mask2):
            
            # 1.0 unit width for anayltical solutions, 0.12 for laboratory experiment (Delestre)
            if self.geometry_channel == True:
                if self.channel_type == 'finite':
                    w1[mask2] = self.channel_width
                    w2[mask2] = self.channel_width
                    w_mid[mask2] = self.channel_width
                else:
                    w1[mask2] = 1.0
                    w2[mask2] = 1.0
                    w_mid[mask2] = 1.0  # Default for infinite channel
                    
                surface_a1[mask2] = (
                    0.5 * (w1[mask2] + w_mid[mask2]) * (self.conduit_lengths[mask2]/2)
                )
                
                surface_a2[mask2] = (
                    0.5 * (w_mid[mask2] + w2[mask2]) * (self.conduit_lengths[mask2]/2)
                )
                
            else:
                w1[mask2] = 2 * np.sqrt(
                    self.conduit_diameters[mask2] * y1[mask2] - y1[mask2]**2
                    )
                
                w2[mask2] = 2 * np.sqrt(
                    self.conduit_diameters[mask2] * y2[mask2] - y2[mask2]**2
                    )
                
                w_mid[mask2] = 2 * np.sqrt(
                    self.conduit_diameters[mask2] * y_mid[mask2] - y_mid[mask2]**2
                    )
                
                surface_a1[mask2] = (
                    0.5 * (w1[mask2] + w_mid[mask2]) * (self.conduit_lengths[mask2]/2)
                )
                
                surface_a2[mask2] = (
                    0.5 * (w_mid[mask2] + w2[mask2]) * (self.conduit_lengths[mask2]/2)
                )
        
        # Calculation when only y1 is below LOWER_LIMIT
        # y2 could be pressurized
        mask3 = (y1 <= self.min_waterdepth) & (y2 > self.min_waterdepth)
        
        if np.any(mask3):
            
            # 1.0 unit width for anayltical solutions, 0.12 for laboratory experiment (Delestre)
            if self.geometry_channel == True:
                if self.channel_type == 'finite':
                    w1[mask3] = self.channel_width
                    w2[mask3] = self.channel_width
                    w_mid[mask3] = self.channel_width
                else:
                    w1[mask3] = 1.0
                    w2[mask3] = 1.0
                    w_mid[mask3] = 1.0  # Default for infinite channel

                surface_a1[mask3] = (
                    0.5 * (w1[mask3] + w_mid[mask3]) * (self.conduit_lengths[mask3]/2)
                )
                
                surface_a2[mask3] = (
                    0.5 * (w_mid[mask3] + w2[mask3]) * (self.conduit_lengths[mask3]/2)
                )
                
            else:
                w1[mask3] = 2 * np.sqrt(
                    self.conduit_diameters[mask3] * y1[mask3] - y1[mask3]**2
                    )
                w2[mask3] = np.where(
                    self.is_full_y2[mask3],
                    slot_w2[mask3],
                    2 * np.sqrt(
                        self.conduit_diameters[mask3] * y2[mask3] - y2[mask3]**2
                        )
                )
                
                w_mid[mask3] = np.where(
                    self.is_full_y_mid[mask3],
                    slot_w_mid[mask3],
                    2 * np.sqrt(
                        self.conduit_diameters[mask3] * y_mid[mask3] - y_mid[mask3]**2
                        )
                )
                
                surface_a1[mask3] = (
                    0.5 * (w1[mask3] + w_mid[mask3]) * (self.conduit_lengths[mask3]/2)
                )
                
                surface_a2[mask3] = (
                    0.5 * (w_mid[mask3] + w2[mask3]) * (self.conduit_lengths[mask3]/2)
                )
        
        # Calculation when only y2 is below LOWER_LIMIT
        # y1 could be pressurized
        mask4 = (y1 > self.min_waterdepth) & (y2 <= self.min_waterdepth)
        
        if np.any(mask4):
            
            # 1.0 unit width for anayltical solutions, 0.12 for laboratory experiment (Delestre)
            if self.geometry_channel == True:
                if self.channel_type == 'finite':
                    w1[mask4] = self.channel_width
                    w2[mask4] = self.channel_width
                    w_mid[mask4] = self.channel_width
                else:
                    w1[mask4] = 1.0
                    w2[mask4] = 1.0
                    w_mid[mask4] = 1.0  # Default for infinite channel

                surface_a1[mask4] = (
                    0.5 * (w1[mask4] + w_mid[mask4]) * (self.conduit_lengths[mask4]/2)
                )
                
                surface_a2[mask4] = (
                    0.5 * (w_mid[mask4] + w2[mask4]) * (self.conduit_lengths[mask4]/2)
                )
            
            else:
                w1[mask4] = np.where(
                    self.is_full_y1[mask4],
                    slot_w1[mask4],
                    2 * np.sqrt(self.conduit_diameters[mask4] * y1[mask4] - y1[mask4]**2
                                )
                )
                
                w2[mask4] = 2 * np.sqrt(
                    self.conduit_diameters[mask4] * y2[mask4] - y2[mask4]**2
                    )
                
                w_mid[mask4] = np.where(
                    self.is_full_y_mid[mask4],
                    slot_w_mid[mask4],
                    2 * np.sqrt(
                        self.conduit_diameters[mask4] * y_mid[mask4] - y_mid[mask4]**2
                        )
                )
                
                surface_a1[mask4] = (
                    0.5 * (w1[mask4] + w_mid[mask4]) * (self.conduit_lengths[mask4]/2)
                )
                
                surface_a2[mask4] = (
                    0.5 * (w_mid[mask4] + w2[mask4]) * (self.conduit_lengths[mask4]/2)
                )
    
        # Add contributing surface areas to each node
        np.add.at(n_surface_a, self.n_indices1, surface_a1)
        np.add.at(n_surface_a, self.n_indices2, surface_a2)
        
        return n_surface_a, slot_w1, slot_w2, slot_w_mid, w_mid    
    
    def _compute_slot_width(self, flow_depths, diameters):
        """
        Compute the slot width for given flow depths and diameters.

        This method calculates the slot width based on normalized flow depths. If the
        normalized flow depth (y_norm) is greater than 1.78, the slot width is set to 1%
        of the maximum width. Otherwise, it uses the Sjoberg equation from SWMM.
        
        Args:
            flow_depths (numpy.ndarray): Array of flow depths.
            diameters (numpy.ndarray): Array of conduit diameters.

        Returns:
            numpy.ndarray: Array of slot widths.
        """
        y_norm = flow_depths / diameters

        width_max = diameters
        slot_widths = np.where(
            y_norm > 1.78,
            0.01 * width_max,
            width_max * 0.5423 * np.exp(-np.power(y_norm, 2.4))
        )
        
        return slot_widths
        
    def _compute_discharge_areas(self, y1, y2, y_mid, slot_widths):
        """
        Compute the discharge areas for given water depths and slot widths.

        This method calculates the discharge areas at both ends and the middle of the 
        conduits based on the water depths and slot widths. For channel geometry, a 
        constant width is used, while for other geometries, the areas are computed 
        considering both pressurized and free surface conditions.

        Args:
            y1 (numpy.ndarray): Water depths at the first end of the conduits.
            y2 (numpy.ndarray): Water depths at the second end of the conduits.
            y_mid (numpy.ndarray): Water depths at the middle of the conduits.
            slot_widths (numpy.ndarray): Slot widths for the conduits.

        Returns:
            tuple: Discharge areas at the first end (a1), second end (a2),
                   and middle (self.a_mid_new) of the conduits.
        """
        

        # 1.0 unit width for anayltical solutions, 0.12 for laboratory experiment (Delestre)
        if self.geometry_channel == True:
            if self.channel_type == 'finite':
                a1 = self.channel_width * y1
                a2 = self.channel_width * y2
                self.a_mid_new = self.channel_width * y_mid
            else:
                a1 = 1.0 * y1
                a2 = 1.0 * y2
                self.a_mid_new = 1.0 * y_mid          
        else:  
            
            radii = self.conduit_diameters * 0.5
            theta1 = 2 * np.arccos(np.clip((radii - y1) / radii, -1, 1))
            theta2 = 2 * np.arccos(np.clip((radii - y2) / radii, -1, 1))
            theta_mid = 2 * np.arccos(np.clip((radii - y_mid) / radii, -1, 1))
            
            a1 = np.where(
                self.is_full_y1,
                np.pi * radii**2 + (y1 - self.conduit_diameters) * slot_widths,
                (radii**2 * (theta1 - np.sin(theta1))) / 2
            )
            
            a2 = np.where(
                self.is_full_y2,
                np.pi * radii**2 + (y2 - self.conduit_diameters) * slot_widths,
                (radii**2 * (theta2 - np.sin(theta2))) / 2
            )
            
            self.a_mid_new = np.where(
                self.is_full_y_mid,
                np.pi * radii**2 + (y_mid - self.conduit_diameters) * slot_widths,
                (radii**2 * (theta_mid - np.sin(theta_mid))) / 2
            )
    
        return a1, a2, self.a_mid_new
    
    def _compute_alpha(self, froude_number):
        """
        Compute alpha for upstream weighting based on the Froude number.

        This method calculates the weighting factor alpha used for upstream weighting 
        in flow computations. The value of alpha depends on the Froude number and 
        whether the conduits are pressurized.

        Args:
            froude_number (numpy.ndarray): Array of Froude numbers for the conduits.

        Returns:
            numpy.ndarray: Array of alpha values for the conduits.
        """
       
        alpha = np.zeros(self.network.Nt, dtype=float)

        # Define logical masks based on Froude number ranges
        logical1 = froude_number <= 0.5
        logical2 = np.logical_and(froude_number > 0.5, froude_number < 1.0)
        logical3 = froude_number >= 1.0
        
        # Assign alpha values based on the Froude number ranges
        alpha[logical1] = 1.0
        alpha[logical2] = 2 * (1 - froude_number[logical2])
        alpha[logical3] = 0
        
        # Set zero when conduits are pressurized (no upstream weighting)
        alpha[self.is_full_y_mid] = 0
        
        return alpha
    
    def _compute_new_dt(self, v_mid, froude):
        """
        Compute the new time step size based on the Froude number and velocity.

        This method calculates the new time step size (dt) based on two criteria:
            1. The Froude number and velocity.
            2. The maximum allowable time step based on the change in nodal head over time (dydt)
               per maximum depths (max_depths) of conduits attached to a node.
            
        When open channel flow is simulated max_depths is set to to be 1.0m during the
        intialization of conduit properties.

        Args:
            v_mid (numpy.ndarray): Array of velocities at the middle of conduits.
            froude (numpy.ndarray): Array of Froude numbers for the conduits.

        Raises:
            ValueError: If the computed time step dt is not valid (NaN or Inf).
        """
        
        # First criterion
    
        # Check if any conduit is full and halve the Courant number if so
        if np.any(self.is_full_y_mid):
            effective_courant = self.courant #/ 2
        else:
            effective_courant = self.courant
    
        # Calculate dt_criterion_froude only for valid conduits
        dt_criterion_froude = np.where(
            v_mid != 0,
            1 / np.abs(v_mid) * (froude / (1 + froude)) * effective_courant,
            np.inf
        )

        # Find the minimum dt from the valid conduits
        max_allowable_dt1 = np.nanmin(dt_criterion_froude)
    
        # Second criterion
        # Determine maximum allowable time step based on change in head over time dydt
        max_allowable_dt2 = np.nanmin(self.max_depths / self.dydt)

        # Initialize dt as the smallest value of the criteria           
        self.dt = min(max_allowable_dt1, max_allowable_dt2)
        
        # Ensure dt does not exceed the specified maximum dt
        self.dt = min(self.dt, self.dt_max)
    
        # Check if the computed dt is valid (not NaN or Inf)
        if np.isnan(self.dt) or np.isinf(self.dt):
            raise ValueError("Computed time step dt is not valid. Please check the input parameters.")

                    
    def _compute_hydraulic_radius(self, flow_depths, flow_areas, slot_width, is_full):
        """
        Compute the hydraulic radius of conduits based on flow conditions.

        This method calculates the hydraulic radius for conduits considering both 
        free surface and pressurized flow conditions. It handles cases where open
        channels are considered instead of conduits.

        Args:
            flow_depths (numpy.ndarray): Array of flow depths in the conduits.
            flow_areas (numpy.ndarray): Array of flow areas in the conduits.
            slot_width (numpy.ndarray): Array of Preissmann slot widths.
            is_full (numpy.ndarray): Boolean array indicating if the conduit is full.

        Returns:
            numpy.ndarray: Array of computed hydraulic radii for the conduits.
        """
        
        if self.geometry_channel:
            if self.channel_type == 'infinite':
                hydraulic_radii = flow_depths
            elif self.channel_type == 'finite':
                hydraulic_radii = (self.channel_width * flow_depths) / (self.channel_width + 2 * flow_depths)
            
        else:
            # Calculate hydraulic radius for both free surface and pressurized conditions
            radii = self.conduit_diameters * 0.5
            theta = 2 * np.arccos(np.clip((radii - flow_depths) / radii, -1, 1))
            wetted_perimeter = np.where(is_full, 2 * np.pi * radii + slot_width, radii * theta)
            hydraulic_radii = flow_areas / wetted_perimeter
            
        return hydraulic_radii 
                       
    def _compute_upstream_weighted_radii(self, r1, r2, r_mid, h1, h2, alpha):
        """
        Compute upstream-weighted hydraulic radii for conduits.

        This method calculates the upstream-weighted hydraulic radii for conduits 
        based on the hydraulic heads and alpha values for upstream weighting. It 
        determines which nodes are upstream and applies the weighting accordingly.

        Args:
            r1 (numpy.ndarray): Hydraulic radii at the first end of the conduits.
            r2 (numpy.ndarray): Hydraulic radii at the second end of the conduits.
            r_mid (numpy.ndarray): Hydraulic radii at the middle of the conduits.
            h1 (numpy.ndarray): Hydraulic heads at the first end of the conduits.
            h2 (numpy.ndarray): Hydraulic heads at the second end of the conduits.
            alpha (numpy.ndarray): Alpha values for upstream weighting.

        Returns:
            numpy.ndarray: Array of upstream-weighted hydraulic radii for the conduits.
        """
    
        r_mid_upwtd = np.zeros(self.network.Nt, dtype=float)
        
        # Determine upstream nodes
        is_upstr_n1 = h1 > h2
        
        # Node1 is upstream
        r1_upstr = r1[is_upstr_n1]
        alpha_upstr = alpha[is_upstr_n1]
        r_mid_upstr = r_mid[is_upstr_n1]
        r_mid_upwtd[is_upstr_n1] = (r1_upstr + 
                                    alpha_upstr * (r_mid_upstr - r1_upstr)
                                    )
        
        # Node2 is upstream
        r2_downstr = r2[~is_upstr_n1]
        alpha_downstr = alpha[~is_upstr_n1]
        r_mid_downstr = r_mid[~is_upstr_n1]
        r_mid_upwtd[~is_upstr_n1] = (r2_downstr + 
                                     alpha_downstr * (r_mid_downstr - r2_downstr)
                                     )
        
        return r_mid_upwtd
    
    def _compute_upstream_weighted_areas(self, a1, a2, a_mid, h1, h2, alpha):
        """
        Compute upstream-weighted areas for conduits.

        This method calculates the upstream-weighted areas for conduits based on 
        the hydraulic heads and alpha values for upstream weighting. It determines 
        which nodes are upstream and applies the weighting accordingly.

        Args:
            a1 (numpy.ndarray): Areas at the first end of the conduits.
            a2 (numpy.ndarray): Areas at the second end of the conduits.
            a_mid (numpy.ndarray): Areas at the middle of the conduits.
            h1 (numpy.ndarray): Hydraulic heads at the first end of the conduits.
            h2 (numpy.ndarray): Hydraulic heads at the second end of the conduits.
            alpha (numpy.ndarray): Alpha values for upstream weighting.

        Returns:
            numpy.ndarray: Array of upstream-weighted areas for the conduits.
        """
        
        a_mid_upwtd = np.zeros(self.network.Nt, dtype=float)
        
        # Determine upstream nodes
        is_upstr_n1 = h1 > h2
        
        # Node1 is upstream
        a1_upstr = a1[is_upstr_n1]
        alpha_upstr = alpha[is_upstr_n1]
        a_mid_upstr = a_mid[is_upstr_n1]
        a_mid_upwtd[is_upstr_n1] = (a1_upstr +
                                    alpha_upstr * (a_mid_upstr - a1_upstr)
                                    )
        
        # Node2 is upstream
        a2_downstr = a2[~is_upstr_n1]
        alpha_downstr = alpha[~is_upstr_n1]
        a_mid_downstr = a_mid[~is_upstr_n1]
        a_mid_upwtd[~is_upstr_n1] = (a2_downstr +
                                     alpha_downstr*(a_mid_downstr - a2_downstr)
                                     )
        
        return a_mid_upwtd
                  
    def _compute_flows(self, a1, a2, a_mid_upwtd, r_mid_upwtd, r_mid,
                      h1, h2, alpha, v_mid, w_mid):
        """
        Compute the flow rates in conduits based on various physical parameters.

        This method calculates the flow rates in conduits by considering pressure terms,
        inertial terms (+ correction due to recharge), and friction factors. It uses the Manning
        equation for free surface flows and the Churchill equation for pressurized flows. The
        Darcy-Weisbach equation forms the  foundation for calculating friction losses in both cases.

        Args:
            a1 (numpy.ndarray): Areas at the first end of the conduits.
            a2 (numpy.ndarray): Areas at the second end of the conduits.
            a_mid_upwtd (numpy.ndarray): Upstream-weighted areas at the middle of the conduits.
            r_mid_upwtd (numpy.ndarray): Upstream-weighted hydraulic radii at the middle of the conduits.
            r_mid (numpy.ndarray): Hydraulic radii at the middle of the conduits.
            h1 (numpy.ndarray): Hydraulic heads at the first end of the conduits.
            h2 (numpy.ndarray): Hydraulic heads at the second end of the conduits.
            alpha (numpy.ndarray): Alpha values for upstream weighting.
            v_mid (numpy.ndarray): Velocities at the middle of the conduits.
            w_mid (numpy.ndarray): Water widths at the middle of the conduits.

        Returns:
            None: This method updates the instance attribute `self.Q_new` directly.
        """
                
        # Initialize arrays
        f = np.zeros(self.network.Nt, dtype=float)
        dQ_friction = np.zeros(self.network.Nt, dtype=float)
        q_correction = np.zeros(self.network.Nt, dtype=float)
        
        # Pressure term (upstream weighting)
        dQ_pressure = -self.gravity * a_mid_upwtd * (h2 - h1) / self.conduit_lengths * self.dt
        
        # # Get the indices of nodes with inflow boundary conditions
        # boundary_nodes = set(self.inflow_boundary.keys())  # Convert to set for faster lookup
        
        # # Identify the conduits where at least one node has a boundary condition or source/sink
        # relevant_conduits = np.where(
        #     np.isin(self.n_indices1, list(boundary_nodes)) | np.isin(self.n_indices2, list(boundary_nodes))
        # )[0]
        
        # # Loop only over conduits that have a BC or source/sink term
        # for conduit in relevant_conduits:
        #     n1 = self.n_indices1[conduit]
        #     n2 = self.n_indices2[conduit]
            
        #     # Get boundary conditions for both nodes
        #     inflow_n1 = self.inflow_boundary.get(n1, None)
        #     inflow_n2 = self.inflow_boundary.get(n2, None)
            
        #     # Initialize q for both nodes to zero
        #     q_n1, q_n2 = 0, 0
        
        #     # Check if inflow at node n1 is a source/sink flux or volumetric flow
        #     if isinstance(inflow_n1, tuple):
        #         if inflow_n1[0] == 'flux':
        #             q_n1 = inflow_n1[1] * w_mid[conduit]  # convert to m^2/s
        #         # Volumetric boundary condition, hence no correction is applied
        #         elif inflow_n1[0] == 'volumetric':
        #             q_n1 = 0.0
            
        #     # Check if inflow at node n2 is a source/sink flux or volumetric flow
        #     if isinstance(inflow_n2, tuple):
        #         if inflow_n2[0] == 'flux':
        #             q_n2 = inflow_n2[1] * w_mid[conduit]  # convert to m^2/s
        #         # Volumetric boundary condition, hence no correction is applied
        #         elif inflow_n2[0] == 'volumetric':
        #             q_n2 = 0.0 
            
        #     # Calculate the average q for the conduit
        #     q_avg = (q_n1 + q_n2) / 2
            
        #     # Assign the correction term for this conduit
        #     q_correction[conduit] = q_avg

        # Compute q_correction for conduits affected by inflow BCs
        q_correction = np.zeros(self.network.Nt, dtype=float)

        # Precompute inflow values per node at current time
        # These flows are also computed when computing water depths
        current_time = self.current_timestep * self.dt
        inflow_at_nodes = {}

        for bc in self.boundary_conditions.get("inflow", []):
            value = bc.get_value(current_time)
            for node in bc.target_ids:
                inflow_at_nodes[node] = (bc.bc_type, value)

        # Identify conduits with at least one node having a flux boundary condition
        relevant_conduits = np.where(
            np.isin(self.n_indices1, list(inflow_at_nodes.keys())) |
            np.isin(self.n_indices2, list(inflow_at_nodes.keys()))
        )[0]

        for conduit in relevant_conduits:
            n1 = self.n_indices1[conduit]
            n2 = self.n_indices2[conduit]

            q_n1 = q_n2 = 0.0

            inflow_n1 = inflow_at_nodes.get(n1)
            inflow_n2 = inflow_at_nodes.get(n2)

            if inflow_n1 is not None and inflow_n1[0] == 'flux':
                q_n1 = inflow_n1[1] * w_mid[conduit]

            if inflow_n2 is not None and inflow_n2[0] == 'flux':
                q_n2 = inflow_n2[1] * w_mid[conduit]

            q_correction[conduit] = 0.5 * (q_n1 + q_n2)
                   
                   
        # Inertial terms (alpha is zero when pressurized)
        # Apply the momentum correction term to the inertia term
        dQ_inertia1 = alpha * 2 * v_mid * (self.a_mid_new - self.a_mid_old_t - q_correction * self.dt)
        dQ_inertia2 = alpha * v_mid * v_mid * (a2 - a1) / self.conduit_lengths * self.dt     
        
        # Compute this only for circular conduits (i.e. not for open channel flows)
        if self.geometry_channel == False:
            # Compute Reynolds number for pressurized conduits
            self.Re_conduit[self.is_full_y_mid] = (
                self.rho * np.abs(v_mid[self.is_full_y_mid]) *
                self.conduit_diameters[self.is_full_y_mid] / self.dyn_viscosity
            )
    
            # Compute Reynolds number for free-surface flows
            self.Re_conduit[~self.is_full_y_mid] = (
                self.rho * np.abs(v_mid[~self.is_full_y_mid]) *
                r_mid[~self.is_full_y_mid] / self.dyn_viscosity
            )
            
            # Define masks for flow regimes under pressurized conditions
            laminar_flow_mask = (self.Re_conduit <= 2300) & self.is_full_y_mid
            turbulent_flow_mask = (self.Re_conduit > 2300) & self.is_full_y_mid
    
            # Compute friction factor for laminar flow under pressurized conditions
            # Set to zero if velocities_mid are zero
            f[laminar_flow_mask] = np.where(
                self.Re_conduit[laminar_flow_mask] != 0,
                64 / self.Re_conduit[laminar_flow_mask],
                0.0
            )
    
            # Compute friction factor using the Churchill equation for turbulent flow
            # under pressurized conditions
            if np.any(turbulent_flow_mask):
                  
                C = ((7 / self.Re_conduit[turbulent_flow_mask]) ** 0.9 +
                     0.27 * self.conduit_epsilon[turbulent_flow_mask] /
                     self.conduit_diameters[turbulent_flow_mask])
                A = (-2.457 * np.log(C)) ** 16
                B = (37530 / self.Re_conduit[turbulent_flow_mask]) ** 16
                f[turbulent_flow_mask] = 8 * ((8 / self.Re_conduit[turbulent_flow_mask]) ** 12 +
                                              1 / (A + B) ** 1.5) ** (1 / 12)
                
            # Compute friction term for pressurized conduits using Churchill
            dQ_friction[self.is_full_y_mid] = (
                f[self.is_full_y_mid] * np.abs(v_mid[self.is_full_y_mid]) /
                (8 * r_mid[self.is_full_y_mid]) * self.dt
            )


        # Compute friction term for free-surface flows using equivalent
        # Manning n friction factor. This factor stays constant and is defined
        # for all conduits using f(epsilon, Re->infty)
        # In the case of open channel flow the Manning factor is directly applied via the
        # physical property settings
        dQ_friction[~self.is_full_y_mid] = (
            self.gravity * self.conduit_manning[~self.is_full_y_mid]**2 *
            np.abs(v_mid[~self.is_full_y_mid]) /
            (r_mid_upwtd[~self.is_full_y_mid]**(4/3)) * self.dt
        )
        
        # Compute dQ components and new flows Q_new
        self.Q_new = (self.Q_old_t + dQ_pressure + dQ_inertia1 + dQ_inertia2)/(1 + dQ_friction)
       
        # Update flows using under-relaxation
        self.Q_new = (1.0 - self.w) * self.Q_prev_i + self.w * self.Q_new
        
        # Check for flow rate sign changes to address potential numerical instabilities.
        # Currently not needed, but retained for future debugging.
        # is_sign_change = np.sign(self.Q_new) != np.sign(self.Q_prev_i)
        # if np.any(is_sign_change) == True:
        #     print("is sign change")
        # self.Q_new[is_sign_change] = 1e-9 * np.sign(self.Q_new[is_sign_change])

        return
    

    def _compute_water_depths(self, n_surface_a):
        """
        Compute the water depths at each node.

        This method calculates the change in water depths at each node based on 
        flow rates and boundary conditions, then updates the water depths using 
        under-relaxation. It accounts for positive and negative flows, applies 
        inflow, fixed head, and critical depth boundary conditions.

        Args:
            n_surface_a (numpy.ndarray): Array of surface areas for each node.

        """
        
        # Set dQ to zero as this is summed each iteration
        self.dQ_new.fill(0.0)
        
        is_positive_flow = self.Q_new > 0.0
        is_negative_flow = self.Q_new < 0.0
   
        # For positive flows, subtract from source node and add to target node
        np.add.at(self.dQ_new, self.n_indices1[is_positive_flow],
                  -self.Q_new[is_positive_flow])
        np.add.at(self.dQ_new, self.n_indices2[is_positive_flow],
                  self.Q_new[is_positive_flow])

        # For negative flows, add to source node and subtract from target node
        np.add.at(self.dQ_new, self.n_indices1[is_negative_flow],
                  -self.Q_new[is_negative_flow])
        np.add.at(self.dQ_new, self.n_indices2[is_negative_flow],
                  self.Q_new[is_negative_flow])
 
                    
        # # Apply inflow boundary conditions with node-specific and time-dependent inflows
        # current_time = self.current_timestep * self.dt
        # # Loop through inflow boundary conditions and apply inflows
        # for node_index, inflow_value in self.inflow_boundary.items():
            
        #     # Ensure inflow_value is numerical (flux or volumetric flowrate)
        #     if isinstance(inflow_value, tuple):
        #         # Handle both cases for flux or volumetric inflow
        #         if inflow_value[0] == 'flux':
        #             flux_value = inflow_value[1]
                    
        #             # Get the conduits connected to this node
        #             connected_conduits = np.where(
        #                 (self.n_indices1 == node_index) | (self.n_indices2 == node_index)
        #             )[0]
                    
        #             # Consider half of each connected conduit length
        #             half_conduit_lengths = 0.5 * self.conduit_lengths[connected_conduits]
                    
        #             # For channel geometry only
        #             if self.geometry_channel:
        #                 if self.channel_type == 'infinite':
        #                     # Multiplied by unit width (1.0)
        #                     inflow_value = flux_value * np.sum(half_conduit_lengths) # * 1.0
        #                 else:  # Finite channel
        #                     inflow_value = flux_value * self.channel_width * np.sum(half_conduit_lengths)
        #             else:
        #                 raise ValueError("Flux inputs are not yet handled for non-channel geometries.")
        #         else:
        #             # If the tuple is not a 'flux', assume it's a volumetric flowrate (m^3/s)
        #             inflow_value = inflow_value[1]
        #     else:
        #         # If not a tuple, assume inflow_value is already a float (volumetric flowrate in m^3/s)
        #         inflow_value = float(inflow_value)
    
        #     # Apply inflow based on the type of inflow
        #     if self.inflow_type == 'constant':
        #         # Apply constant inflow rate
        #         self.dQ_new[node_index] += inflow_value
        #     elif self.inflow_type == 'ramp':
        #         # Handle ramped inflow
        #         if isinstance(inflow_value, tuple) and len(inflow_value) == 2:
        #             initial_rate, peak_rate = inflow_value
        #             ramped_inflow = self._time_dependent_flowrate(
        #                 current_time, self.start_time, self.end_time, initial_rate, peak_rate
        #             )
        #             self.dQ_new[node_index] += ramped_inflow
        #         else:
        #             raise ValueError(f"Expected a tuple for ramped inflow at node {node_index}, but got {inflow_value}")
        #     elif self.inflow_type == 'constant_timespan':
        #         # Apply constant inflow rate only within a specific time span
        #         if self.start_time <= current_time <= self.end_time:
        #             self.dQ_new[node_index] += inflow_value
        #         else:
        #             self.dQ_new[node_index] += 0  # No inflow if outside the time range
    
        # Apply time-dependent inflow BCs (new format)
        current_time = self.current_timestep * self.dt

        for bc in self.boundary_conditions.get('inflow', []):
            value = bc.get_value(current_time)

            for node in bc.target_ids:
                # If bc_type is flux convert to volume otherwise assume it is volumetric
                if getattr(bc, 'bc_type', 'volumetric') == 'flux':
                    # Convert to volumetric inflow based on local geometry
                    connected_conduits = np.where(
                        (self.n_indices1 == node) | (self.n_indices2 == node)
                    )[0]
                    half_lengths = 0.5 * self.conduit_lengths[connected_conduits]

                    if self.geometry_channel:
                        if self.channel_type == 'infinite':
                            inflow_volume = value * np.sum(half_lengths)
                        else:
                            inflow_volume = value * self.channel_width * np.sum(half_lengths)
                    else:
                        raise ValueError("Flux inputs are not supported for non-channel geometries.")
                    
                    self.dQ_new[node] += inflow_volume

                else:
                    # Apply directly as volumetric flowrate
                    self.dQ_new[node] += value

        # Compute the change in volume at each node (dV)
        dV = 0.5 * (self.dQ_old_t + self.dQ_new) * self.dt
        
        # Compute change in flow depths and new depths
        dy = dV / n_surface_a
        self.y_new = self.y_old_t + dy
                 
        # Update water depths using under-relaxation
        self.y_new = (1.0 - self.w) * self.y_prev_i + self.w * self.y_new
    
        # Apply fixed water depth BCs (new format)
        for bc in self.boundary_conditions.get('waterdepth', []):
            value = bc.get_value(current_time)
            for node in bc.target_ids:
                self.y_new[node] = value    

        # # Apply fixed head boundary conditions
        # for node_index, waterdepth_value in self.waterdepth_boundary.items():
        #     self.y_new[node_index] = waterdepth_value
        
        # # Apply free outfall condition using critical depth
        # for node_index in self.critical_depth_boundary:
            
        #     # Find all conduits connected to this node
        #     connected_conduits = np.where(
        #         (self.n_indices1 == node_index) | (self.n_indices2 == node_index)
        #     )[0]
        
        #     # Compute the critical depth for each connected conduit
        #     critical_depths = [
        #         self._find_critical_depth(self.Q_new[conduit_index],
        #                                   self.conduit_diameters[conduit_index]
        #                                   )
        #         for conduit_index in connected_conduits
        #     ]   
        
        #     # Assign the maximum critical depth to the node
        #     critical_depth = max(critical_depths) if critical_depths else 0.0
        #     self.y_new[node_index] = critical_depth
    
        # Ensure water depths don't go negative
        self.y_new[self.y_new <= 0.0] = 0.0
        
        # Compute change in water depths
        self.dydt = np.abs(self.y_new - self.y_old_t) / self.dt
                
        return
    
    def _adjust_flowrates_dry_nodes(self):
        """
        Adjust flow rates for nodes with insufficient water depth.

        This method sets the flow rates to a minimum threshold for nodes
        where the water depth is below a specified minimum value. It ensures
        that the flow rates do not exceed a minimum flow rate in either 
        positive or negative direction for these dry nodes.
        
        """
        
        self.Q_new[(self.Q_new > self.min_flowrate) &
                   (self.y_new[self.n_indices1] <= self.min_waterdepth)] = self.min_flowrate
        self.Q_new[(self.Q_new < -self.min_flowrate) &
                   (self.y_new[self.n_indices2] <= self.min_waterdepth)] = -self.min_flowrate
    
    def _print_timestep_info(self, iteration, froude):
        """
        Print information about the current timestep at specified intervals.

        This method prints details about the simulation at specified intervals
        including the new time step size, maximum Froude number, current timestep, 
        current simulation time, iteration count, maximum Reynolds number, and 
        average Reynolds number.

        Args:
            iteration (int): The current iteration count within the timestep.
            froude (ndarray): Array of Froude numbers for the conduits.

        """
    
        # Print information at specified intervals
        if math.fmod(self.current_timestep, self.print_info_interval) == 0:
            print(f'New dt = {self.dt:.2e} Max Froude = {np.max(froude):.2f}')
            print(f'Timestep = {self.current_timestep}, '
                  f'Time = {self.current_time:.1f}, i = {iteration}')
            print(f'Max Re = {np.max(self.Re_conduit):.2f} '
                  f'Avg Re = {np.mean(self.Re_conduit):.2f}\n')


    def _check_picard_convergence(self):
        """
        Check if the Picard iteration has converged.

        This method checks if the Picard iteration has converged by comparing 
        the absolute differences between the new and previous water depths 
        against a specified tolerance.

        Returns:
            bool: True if the iteration has converged, False otherwise.

        """

        # if np.all(np.abs(self.y_new - self.y_prev_i) < self.picard_depth_tol):
        #     return True
        # else:
        #     return False
        
        # Calculate the L2 norm of the difference between the new and previous water depths
        l2_norm_diff = np.linalg.norm(self.y_new - self.y_prev_i)
    
        # Calculate the L2 norm of the new water depths
        l2_norm_new = np.linalg.norm(self.y_new)
    
        # Compute the relative L2 norm
        if l2_norm_new != 0:
            relative_l2_norm = l2_norm_diff / l2_norm_new
        else:
            # Handle the case where the new L2 norm is zero (to avoid division by zero)
            relative_l2_norm = l2_norm_diff
    
        # Check if the relative L2 norm is below the specified tolerance
        if relative_l2_norm < self.picard_depth_tol:
            return True
        else:
            return False
        
    def _compute_error_norms(self):
        """
        Compute the relative L2 and MAD norms between the new state and the old state.
    
        This method calculates the L2 norm and the median absolute deviation (MAD)
        between the new state (`self.y_new`) and the old state (`self.y_old_t`).
        It then computes the relative L2 norm and the relative MAD norm.
    
        If the L2 norm or the MAD of `self.y_new` is zero, the corresponding relative
        norm is set to zero.
    
        Attributes:
            relative_l2_norm (float): The relative L2 norm between `self.y_new` and `self.y_old_t`.
            relative_mad_norm (float): The relative MAD norm between `self.y_new` and `self.y_old_t`.
        """
        
        l2_norm = np.linalg.norm(self.y_new - self.y_old_t)
        if np.linalg.norm(self.y_new) != 0:
            self.relative_l2_norm = l2_norm / np.linalg.norm(self.y_new)
        else:
            self.relative_l2_norm = 0.0
        
        mad = np.median(np.abs(self.y_new - self.y_old_t))
        if np.median(np.abs(self.y_new)) != 0:
            self.relative_mad_norm = mad / np.median(np.abs(self.y_new))
        else:
            self.relative_mad_norm = 0.0
        
    def _check_steady_state_convergence(self):
        """
        Check if the system has reached steady state convergence.
    
        This method checks whether the system has reached steady state convergence
        by evaluating the relative L2 norm and the relative MAD norm, which are
        computed previously using the `_compute_error_norms` method.
    
        The system is considered to have reached steady state convergence if both
        the relative L2 norm and the relative MAD norm are below their respective
        tolerances.
    
        Returns:
            bool: True if the system has converged based on the relative L2 and MAD norms, False otherwise.
        """

        is_l2_converged = (
            (self.relative_mad_norm < self.ss_rel_madtol) and
            (self.relative_l2_norm < self.ss_rel_l2tol)
        )
        
        return is_l2_converged
    
    # Calculation of critical depths. This is computationally inefficient. Probably better to use
    # a lookup table with precomputed values and then interpolate.
    def _flow_area_cdepth(self, depth, diameter):
        """
        Calculate the flow area in a circular conduit at a given depth.

        This method computes the segment area of flow in a circular conduit given 
        the depth of flow and the diameter of the conduit.

        Args:
            depth (float): The depth of flow in the conduit.
            diameter (float): The diameter of the conduit.

        Returns:
            float: The segment area of flow in the conduit.
        """
        
        r = diameter / 2
        theta = 2 * np.arccos((r - depth) / r)
        segment_area = (r**2 / 2) * (theta - np.sin(theta))
        return segment_area

    def _wetted_perimeter_cdepth(self, depth, diameter):
        """
        Calculate the wetted perimeter in a circular conduit at a given depth.

        This method computes the wetted perimeter of flow in a circular conduit 
        given the depth of flow and the diameter of the conduit.

        Args:
            depth (float): The depth of flow in the conduit.
            diameter (float): The diameter of the conduit.

        Returns:
            float: The wetted perimeter of flow in the conduit.
        """
        
        r = diameter / 2
        theta = 2 * np.arccos((r - depth) / r)
        return r * theta

    def _hydraulic_radius_cdepth(self, depth, diameter):
        """
        Calculate the hydraulic radius in a circular conduit at a given depth.

        This method computes the hydraulic radius, which is the ratio of the flow area 
        to the wetted perimeter, for a circular conduit given the depth of flow 
        and the diameter of the conduit.

        Args:
            depth (float): The depth of flow in the conduit.
            diameter (float): The diameter of the conduit.

        Returns:
            float: The hydraulic radius of the flow in the conduit.
        """
        
        area = self._flow_area_cdepth(depth, diameter)
        perimeter = self._wetted_perimeter_cdepth(depth, diameter)
        return area / perimeter

    def _critical_depth(self, depth, Q, g, diameter):
        """
        Calculate the critical depth in a circular conduit.

        This method calculates the critical depth of flow in a circular conduit
        using the given flow rate, gravitational constant, and diameter of the conduit.
        The critical depth is the depth at which the flow is at a specific energy minimum.

        Args:
            depth (float): The depth of flow in the conduit.
            Q (float): The flow rate in the conduit.
            g (float): The gravitational constant.
            diameter (float): The diameter of the conduit.

        Returns:
            float: The computed critical depth value.
        """
        
        perimeter = self._wetted_perimeter_cdepth(depth, diameter)
        area = self._flow_area_cdepth(depth, diameter)
        return (Q**2 * perimeter) / (g * area**3) - 1

    def _find_critical_depth(self, Q, diameter, g=9.81):
        """
        Find the critical depth in a circular conduit.

        This method calculates the critical depth of flow in a circular conduit
        using the given flow rate, gravitational constant, and diameter of the conduit.
        It uses an initial guess for the critical depth and solves the equation using
        a root-finding algorithm.

        Args:
            Q (float): The flow rate in the conduit.
            diameter (float): The diameter of the conduit.
            g (float, optional): The gravitational constant. Defaults to 9.81.

        Returns:
            float: The computed critical depth value.
        """
        
        initial_guess = 1.01 * (Q**2/g)**0.25 / (diameter**0.26) # SWMM 5.1
        critical_depth = optimize.fsolve(self._critical_depth, initial_guess, args=(Q, g, diameter))[0]
        return critical_depth

    
    def _time_dependent_flowrate(self, current_time, start_time, end_time, initial_rate, peak_rate):
        
        """
        Calculate the inflow rate based on the current time.
        
        Args:
            current_time (float): The current time in seconds.
            start_time (float): The time at which the ramp-up starts.
            end_time (float): The time at which the ramp-down ends.
            initial_rate (float): The initial flow rate at the start time.
            peak_rate (float): The peak flow rate during the ramp-up.
            
        Returns:
            float: The inflow rate at the given time.
        """
        # Duration for ramping up
        ramp_up_duration = (end_time - start_time) / 2
        
        # Ramp-up phase
        if start_time <= current_time < start_time + ramp_up_duration:
            return initial_rate + (peak_rate - initial_rate) * \
                   ((current_time - start_time) / ramp_up_duration)
        
        # Ramp-down phase
        elif start_time + ramp_up_duration <= current_time <= end_time:
            return peak_rate - (peak_rate - initial_rate) * \
                   ((current_time - (start_time + ramp_up_duration)) / ramp_up_duration)
        
        # Outside the defined period, return the initial flow rate
        else:
            return initial_rate
   
