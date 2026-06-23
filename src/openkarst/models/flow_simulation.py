#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:56:06 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import numpy as np
import math
from numbers import Real

from termcolor import colored
from typing import Optional, Dict, Any

from openkarst.config.physical_properties import PhysicalProperties
from openkarst.config.geometry_settings import GeometrySettings
from openkarst.config.solver_settings import SolverSettings
from openkarst.config.simulation_settings import SimulationSettings
from openkarst.config.transport_settings import TransportSettings
from openkarst.config.validate_settings import validate_settings
from openkarst.config.apply_settings import apply_settings

from openkarst.io.results_handling import initialize_results_container, store_results
from openkarst.io.observation_recorder import ObservationRecorder

from openkarst.utils.helpers import time_this
from openkarst.utils.logging_config import setup_logging

from openkarst.models.boundary_conditions import (
    BoxBC,
    ConstantBC,
    TimeSeriesBC,
    broadcast_boundary_values,
    normalize_target_ids,
)
from openkarst.models.hydraulics import (
    compute_churchill_friction_factor,
    compute_slot_width,
    compute_upstream_weight_alpha,
)
from openkarst.models.cross_section_geometry import create_cross_section_geometry


class FlowSimulation:
    """
    Simulates free surface and pressurized flow including transport through a karst conduit
    network using the dynamic wave equation.

    This class models the flow of water and transport through a network of conduits, considering both
    free surface and pressurized flow conditions. It utilizes the dynamic wave equation
    to compute the flow rates, water depths, concentrations and other relevant properties over time.

    Attributes:
        GEOMETRY_CHANNEL (int): Indicator for channel geometry. Set to 1 for channel validation.
        logger (Logger): Logger instance for logging simulation information and debugging.
        physical_properties (PhysicalProperties): Object containing the physical properties of the simulation.
        solver_settings (SolverSettings): Object containing the solver settings.
        simulation_settings (SimulationSettings): Object containing the simulation settings.
        transport_settings (TransportSettings): Object containing the transport settings.
        logging_settings (LoggingSettings): Object containing the logger settings.
        network (openpnm.network.GenericNetwork): The OpenPNM network used in the simulation.
        waterdepth_boundary (dict): Dictionary storing water depth boundary conditions.
        inflow_boundary (dict): Dictionary storing inflow boundary conditions.
        critical_depth_boundary (dict): Dictionary storing critical depth boundary conditions.

    Methods:
        __init__(self, openpnm_network, physical_properties=None, solver_settings=None, simulation_settings=None, transport_settings=None):
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
        _accept_picard_iteration(self):
            Accepts the latest Picard iterate for the next iteration.
        _accept_hydraulic_solution(self):
            Accepts the solved hydraulic state for the current timestep.
        _prepare_timestep_state(self):
            Prepares state buffers at the beginning of each timestep.
        _compute_conduit_state(self, y1, y2, y_mid):
            Computes the pressurization state of the conduits.
        _compute_surface_area(self, y1, y2, y_mid):
            Computes the surface areas and widths of the conduits.
        _compute_discharge_areas(self, y1, y2, y_mid, slot_widths):
            Computes the discharge areas of the conduits.
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
        _print_timestep_info(self, n_iterations, converged, froude):
            Prints information about the current timestep.
        _has_picard_converged(self):
            Checks the convergence of the Picard iterations.
        _update_timestep_error_norms(self):
            Computes the L2 error norms.
        _has_reached_steady_state(self):
            Checks the convergence to steady state using the L2 norms.
        __del__(self):
            Destructor for the FlowSimulation class, closes the logger.
    """
    
    #GEOMETRY_CHANNEL = 1     # Geometry = 1 for channel validation
    
    def __init__(self, openpnm_network,
                 physical_properties: Optional[Dict[str, Any]] = None,
                 geometry_settings: Optional[Dict[str, Any]] = None,
                 solver_settings: Optional[Dict[str, Any]] = None,
                 simulation_settings: Optional[Dict[str, Any]] = None,
                 transport_settings: Optional[Dict[str, Any]] = None,
                 logging_settings: Optional[Dict[str, Any]] = None):
        """
        Initializes the FlowSimulation class with provided settings and network.

        Args:
            openpnm_network: The OpenPNM network to be used in the simulation.
            physical_properties (Optional[Dict[str, Any]]): Physical properties settings.
            geometry_settings (Optional[Dict[str, Any]]): Geometry backend settings.
            solver_settings (Optional[Dict[str, Any]]): Solver settings.
            simulation_settings (Optional[Dict[str, Any]]): Simulation settings.
            logging_settings (Optional[Dict[str, Any]]): Logging configuration settings.
        
        Attributes:
            logger: Logger for logging information and debugging.
            physical_properties: Instance of PhysicalProperties with provided or default settings.
            solver_settings: Instance of SolverSettings with provided or default settings.
            simulation_settings: Instance of SimulationSettings with provided or default settings.
            transport_settings: Instance of TransportSettings with provided or default settings.
            network: The OpenPNM network to be used in the simulation.
            waterdepth_boundary (dict): Dictionary for water depth boundary conditions.
            inflow_boundary (dict): Dictionary for inflow boundary conditions.
            critical_depth_boundary (dict): Dictionary for critical depth boundary conditions.
        """
        
        # Set up logger
        self.logger = setup_logging(logging_settings)

        physical_properties = dict(physical_properties) if physical_properties else {}
        geometry_settings = dict(geometry_settings) if geometry_settings else {}
        
        self.physical_properties = (PhysicalProperties(**physical_properties) 
                                    if physical_properties else PhysicalProperties()
                                    )
        self.geometry_settings = (GeometrySettings(**geometry_settings)
                                  if geometry_settings else GeometrySettings()
                                  )
        self.solver_settings = (SolverSettings(**solver_settings)
                                if solver_settings else SolverSettings()
                                )
        self.simulation_settings= (SimulationSettings(**simulation_settings)
                                   if simulation_settings else SimulationSettings()
                                   )
        self.transport_settings= (TransportSettings(**transport_settings)
                                   if transport_settings else TransportSettings()
                                   )

        validate_settings(self.physical_properties,
                          self.geometry_settings,
                          self.solver_settings,
                          self.simulation_settings,
                          self.transport_settings,
                          self.logger
                          )
        
        apply_settings(self,
                       self.physical_properties,
                       self.geometry_settings,
                       self.solver_settings,
                       self.simulation_settings,
                       self.transport_settings,
                       self.logger
                       )
        
        #Get OpenPNM network (this will later come from another class)
        self.network = openpnm_network
        
        # Additional tools
        self.observation_recorder = None

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
        self.logger.info('FlowSimulation initialized with geometry settings: %s',
                         self.geometry_settings)
        self.logger.info('FlowSimulation initialized with solver settings: %s',
                         self.solver_settings)
        self.logger.info('FlowSimulation initialized with simulation settings: %s',
                         self.simulation_settings)
        self.logger.info('FlowSimulation initialized with transport settings: %s',
                         self.transport_settings)
        
        
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

        # Flow arrays
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
        #self.Z = np.full(self.network.Np, self.network['pore.coords'][:, 2], dtype=float)
        self.Z = self.network['pore.coords'][:, 2]
        self.z1 = self.Z[self.n_indices1]
        self.z2 = self.Z[self.n_indices2]

        # Arrays for nodal boundary conditions
        self.bc_inflow_vol_node = np.zeros(self.network.Np, dtype=float) # [m^3/s]
        self.bc_flux_node = np.zeros(self.network.Np, dtype=float) # [m/s]
        self.bc_flux_to_vol_node = np.zeros(self.network.Np, dtype=float) # [m^3/s]
        self.bc_reservoir_exchange_node = np.zeros(self.network.Np, dtype=float) # [m^3/s]
        self.bc_prescribed_y_mask = np.zeros(self.network.Np, dtype=bool) # Dirichlet mask [m] 
        self.bc_prescribed_y_vals = np.zeros(self.network.Np, dtype=float) # Dirichlet values [m] 
        self.bc_Qin_node = np.zeros(self.network.Np, dtype=float) # total external inflow [m^3/s] 


        # Scratch arrays reused in _compute_flows
        self.f = np.zeros(self.network.Nt, dtype=float)             
        self.dQ_friction = np.zeros(self.network.Nt, dtype=float)   
        self.q_correction = np.zeros(self.network.Nt, dtype=float) # momentum correction (flux only currently)
        self.D_eff = np.zeros(self.network.Nt, dtype=float)

         # Transport arrays (only initialize when transport is enabled)
        if self.simulation_settings.enable_transport:
            self.C = np.zeros(self.network.Np, dtype=float) # concentration [kg/m^3]
            self.C_new = np.zeros(self.network.Np, dtype=float)
            self.M = np.zeros(self.network.Np, dtype=float) # mass [kg]
            self.bc_prescribed_C_mask = np.zeros(self.network.Np, dtype=bool) # Dirichlet C
            self.bc_prescribed_C_vals = np.zeros(self.network.Np, dtype=float)
            self.bc_Cin_node = np.zeros(self.network.Np, dtype=float) # concentration of incoming water at inflow nodes
            self.bc_mass_inflow_rate_node  = np.zeros(self.network.Np, dtype=float) # [kg/s] NOTE: currently not used
            self.bc_mass_injection_node = np.zeros(self.network.Np, dtype=float) # [kg/s]

        # Always available for hydraulics; used by transport only if enabled:
        self.V_node = np.zeros(self.network.Np, dtype=float)  # [m^3]
        self._dV_last = np.zeros(self.network.Np, dtype=float)    
        
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
            self.cross_section_geometry = None
            
            # Get Manning coefficient from physical property settings
            self.conduit_manning = np.full(self.network.Nt, self.channel_manning, dtype=float)
            
            self.conduit_lengths = np.asarray(self.network['throat.lengths'], dtype=float)
            
            # Set max_depths to a default value for open channel flow
            # This can affect the calculation of the second adaptive dt criterion
            self.max_depths = np.full(self.network.Np, 1e-8, dtype=float)
            
            self.Re_conduit = np.zeros(self.network.Nt, dtype=float)
        
        else:
                
            self.conduit_diameters = np.asarray(self.network['throat.diameters'], dtype=float)
            self.conduit_lengths = np.asarray(self.network['throat.lengths'], dtype=float)
            self.conduit_epsilon = np.asarray(self.network['throat.epsilon'], dtype=float)
            self.cross_section_geometry = create_cross_section_geometry(
                self.geometry_backend,
                self.conduit_diameters,
                table_points=self.geometry_table_points,
                table_file=self.geometry_table_file,
                scale_by_diameter=self.geometry_scale_by_diameter,
                interpolation_method=self.geometry_interpolation_method,
            )
            self.full_conduit_areas = self.cross_section_geometry.full_area()
            self.full_hydraulic_diameters = (
                self.cross_section_geometry.full_hydraulic_diameter()
            )
             
            # Initialize array to store the maximum depth for each node
            # At each node max_depth is the crown depth of the largest connected conduit
            self.max_depths = np.zeros(self.network.Np, dtype=float)
            
            # Update max_depth based on the connected conduits
            # For nodes connected at n_indices1
            for i, node in enumerate(self.n_indices1):
                self.max_depths[node] = max(
                    self.max_depths[node],
                    self.cross_section_geometry.full_depths[i],
                )
            # For nodes connected at n_indices2
            for i, node in enumerate(self.n_indices2):
                self.max_depths[node] = max(
                    self.max_depths[node],
                    self.cross_section_geometry.full_depths[i],
                )
            
            self.Re_conduit = np.zeros(self.network.Nt, dtype=float)
            
            # Compute equivalent Manning coefficient at f(epsilon, Re->infty)
            # This is the equivalent Manning coefficient used for pressurized conduits
            if self.friction_model == 'hybrid':
                RE_INFTY = 1e7
                f = compute_churchill_friction_factor(
                    RE_INFTY,
                    self.conduit_epsilon,
                    self.conduit_diameters,
                )
                self.conduit_manning = (
                1 / (np.sqrt(8 * self.gravity)) 
                * np.sqrt(f) 
                * (0.5 * self.conduit_diameters)**(1 / 3)
                )
            else:
                self.conduit_manning = np.zeros(self.network.Nt, dtype=float)
        
        # Sum of half-lengths of conduits connected to each node
        self.half_lengths_sum_per_node = 0.5 * (
            np.bincount(self.n_indices1, weights=self.conduit_lengths, minlength=self.network.Np) +
            np.bincount(self.n_indices2, weights=self.conduit_lengths, minlength=self.network.Np)
        )
            
        self.logger.info('Conduit properties initialized')
            
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
        # Check if user wants to save concentration or mass but has not enabled transport
        self._validate_transport_outputs(desired_outputs)

        # Check boundary conditions for overlaps and transport consistency
        self._check_bc_conflicts()

        results_container = initialize_results_container(desired_outputs, self.logger)
        output_interval = desired_outputs.get('output_interval', 1.0)
        next_output_time = output_interval
        
        with time_this('run_simulation'):

            self.convergence_fails = 0
            self.dt = self.dt_init
            self.current_time = 0.0
            self.current_timestep = 0
            self.relative_y_l2_norm = 0.0
            self.relative_Q_l2_norm = 0.0
            self.picard_iterations_last = 0
            self.picard_iterations_total = 0
            
            while True:

                # Prepare timestep snapshot and buffer for Picard iteration
                self._prepare_timestep_state()
               
                # Compute boundary condition values
                # This currently also computes 
                self._cache_hydraulic_bcs()

                # Perform the dynamic wave computation for the current time step
                converged, n_iterations = self._dynamic_wave()
                self.picard_iterations_last = n_iterations
                self.picard_iterations_total += n_iterations
                
                if not converged:
                    print(colored(
                        f'[run_simulation] Not converged at time = '
                        f'{self.current_time:.1f}', 'red'
                    ))
                    self.convergence_fails += 1
                    
                # Accept the solved hydraulic state for this timestep
                self._accept_hydraulic_solution()

                # Compute L2 error norms for each timestep
                self._update_timestep_error_norms()

                self._print_timestep_info(n_iterations, converged, self._froude_last)

                # Compute AD Transport with updated flow field
                # Only if enable_transport is True
                if self.simulation_settings.enable_transport:
                    self._advance_transport()
                
                # Compute new step size based on Froude and Courant number 
                if self.adaptive_timesteps and self.current_timestep > 0:
                    self._compute_new_dt(self._v_mid_last, self._froude_last)

                # Record observation data if recorder is active and it is time
                if self.observation_recorder and self.current_time >= self.observation_recorder.next_record_time:
                    self.observation_recorder.record(self.current_time, self)
                    self.observation_recorder.next_record_time += self.observation_recorder.interval

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
                    if self._has_reached_steady_state() and self.current_timestep > 10:
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
        Set the initial conditions for flow and derive transport volumes.

        This method sets the initial flow rates and water depths for the
        simulation.

        Args:
            initial_Q (array-like): Initial flow rates for the conduits.
            initial_y (array-like): Initial water depths for the nodes.
        """
        
        np.copyto(self.Q, initial_Q)
        np.copyto(self.y, initial_y)


    def set_waterdepth_BC(self, nodes, values, mode='add', extrapolate='hold'):
        """
        Set water depth (Dirichlet) boundary conditions at specified nodes.

        This method assigns constant or time-dependent water depth boundary
        conditions to one or more nodes in the network. Scalars are broadcast
        automatically to all specified nodes.

        Args:
            nodes (int, list of int, or np.ndarray):
                Index or indices of nodes where water depth boundary conditions
                are applied.

            values (float, tuple, list, or np.ndarray):
                Boundary condition definitions. If a single scalar or tuple is
                provided, it is broadcast to all specified nodes. Supported formats:

                * **float** or **int**: constant water depth.
                * **tuple**:
                    - `('timeseries', times, values)`: interpolated time series.
                    - `('box', value, t0, t1 [, value_before=0.0, value_after=0.0])`:
                      constant value during `[t0, t1]`, with optional values before and after.
                * **list** or **1D np.ndarray**: per-node values, one entry per node.
                * **0D np.ndarray** (e.g., `np.array(0.2)`): treated as scalar and broadcast.

            mode (str, optional):
                Defines how new BCs interact with existing ones:
                - `'add'` (default): add new BCs; raises an error if a BC already exists
                  at a node.
                - `'overwrite'`: replace any existing BCs at the given nodes.
                - `'remove'`: remove BCs from the specified nodes.

            extrapolate (str, optional):
                Extrapolation behavior for `'timeseries'` BCs:
                - `'hold'` (default): hold the first/last value constant outside the defined range.
                - `'zero'`: set BC value to zero outside the defined range.

        Raises:
            ValueError: If an unrecognized BC format is provided, if duplicate
                nodes are given in `'add'` mode, or if the number of values
                does not match the number of nodes.
        """

        # Initialize boundary_conditions dictionary if missing
        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}

        if 'waterdepth' not in self.boundary_conditions:
            self.boundary_conditions['waterdepth'] = []

        nodes = normalize_target_ids(nodes)
        values = broadcast_boundary_values(nodes, values)

        # Mode: remove BCs and exit early
        if mode == 'remove':
            self.boundary_conditions['waterdepth'] = [
                bc for bc in self.boundary_conditions['waterdepth']
                if all(n not in nodes for n in bc.target_ids)
            ]
            return

        # Loop over each node-value pair
        for node, val in zip(nodes, values):
            if mode == 'overwrite':
                self.boundary_conditions['waterdepth'] = [
                    bc for bc in self.boundary_conditions['waterdepth']
                    if node not in bc.target_ids
                ]
            elif mode == 'add':
                for bc in self.boundary_conditions['waterdepth']:
                    if node in bc.target_ids:
                        raise ValueError(f"Water depth BC already exists at node {node}. Use mode='overwrite' to replace it.")

            # Create appropriate BC object
            if isinstance(val, Real):
                bc = ConstantBC([node], value=float(val))
            elif isinstance(val, tuple) and val[0] == 'box':
                _, v_during, t0, t1, *rest = val
                v_before = rest[0] if len(rest) > 0 else 0.0
                v_after = rest[1] if len(rest) > 1 else 0.0
                bc = BoxBC([node], v_during, t0, t1, v_before, v_after)
            elif isinstance(val, tuple) and val[0] == 'timeseries':
                _, times, flow_values = val[:3]
                bc = TimeSeriesBC([node], times=times, values=flow_values, extrapolate=extrapolate)
            else:
                raise ValueError(f"Unrecognized value for BC at node {node}: {val}")

            self.boundary_conditions['waterdepth'].append(bc)


    def set_inflow_BC(self, nodes, values, mode='add', inflow_type='volumetric', extrapolate='hold'):
        """
        Set inflow boundary conditions at specified nodes.

        This method assigns constant or time-dependent inflow boundary conditions
        (either volumetric or flux-type) to one or more nodes. Scalars are automatically
        broadcast to all specified nodes.

        Args:
            nodes (int, list of int, or np.ndarray):
                Index or indices of nodes where inflow boundary conditions are applied.

            values (float, tuple, list, or np.ndarray):
                Boundary condition definitions. If a single scalar or tuple is provided,
                it is broadcast to all specified nodes. Supported formats:

                * **float** or **int**: constant inflow.
                * **tuple**:
                    - `('timeseries', times, values)`: interpolated time series.
                    - `('box', value, t0, t1 [, value_before=0.0, value_after=0.0])`:
                      constant inflow during `[t0, t1]`, with optional values before and after.
                * **list** or **1D np.ndarray**: per-node inflow definitions, one entry per node.
                * **0D np.ndarray** (e.g., `np.array(0.2)`): treated as a scalar and broadcast.

            mode (str, optional):
                Defines how new boundary conditions interact with existing ones:
                - `'add'` (default): add new BCs; raises an error if a BC already exists at a node.
                - `'overwrite'`: replace existing BCs at the given nodes.
                - `'remove'`: remove BCs from the specified nodes.

            inflow_type (str, optional):
                Specifies the inflow type:
                - `'volumetric'` (default): total flow rate in m³/s.
                - `'flux'`: area-normalized rate in m/s.

            extrapolate (str, optional):
                Extrapolation behavior for `'timeseries'` BCs:
                - `'hold'` (default): hold the first/last value constant outside the defined range.
                - `'zero'`: set BC value to zero outside the defined range.

        Raises:
            ValueError: If an unrecognized BC format is provided, if duplicate nodes are given in
                `'add'` mode, or if the number of values does not match the number of nodes.
        """
        # Initialize boundary_conditions dictionary if missing
        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}

        if 'inflow' not in self.boundary_conditions:
            self.boundary_conditions['inflow'] = []

        nodes = normalize_target_ids(nodes)
        values = broadcast_boundary_values(nodes, values)

                # Mode: remove BCs and exit early
        if mode == 'remove':
            self.boundary_conditions['inflow'] = [
                bc for bc in self.boundary_conditions['inflow']
                if all(n not in nodes for n in bc.target_ids)
            ]
            return

        # Loop over each node-value pair
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

            # Create appropriate BC object
            if isinstance(val, Real):
                bc = ConstantBC([node], value=float(val), bc_type=inflow_type)
            elif isinstance(val, tuple) and val[0] == 'box':
                _, v_during, t0, t1, *rest = val
                v_before = rest[0] if len(rest) > 0 else 0.0
                v_after = rest[1] if len(rest) > 1 else 0.0
                bc = BoxBC([node], v_during, t0, t1, v_before, v_after, bc_type=inflow_type)
            elif isinstance(val, tuple) and val[0] == 'timeseries':
                _, times, flow_values = val[:3]
                bc = TimeSeriesBC([node], times=times, values=flow_values,
                                bc_type=inflow_type, extrapolate=extrapolate)
            else:
                raise ValueError(f"Unrecognized inflow BC format at node {node}: {val}")

            self.boundary_conditions['inflow'].append(bc)

 
    def set_reservoir_BC(self, nodes, fixed_exchange_rate, mode='add'):
        """
        Set fixed reservoir exchange boundary conditions at specified nodes.

        Positive exchange injects water from the reservoir into a conduit node.
        Negative exchange drains water from an conduit node into the reservoir.
        This minimal test stores fixed volumetric exchange rates as constant boundary
        conditions. Here we can implement a more complex reservoir dynamic then.

        Args:
            nodes (int, list of int, or np.ndarray):
                Index or indices of nodes coupled to the reservoir.

            fixed_exchange_rate (float, list, or np.ndarray):
                Fixed volumetric reservoir-to-node exchange rate in m^3/s.
                Scalars are broadcast automatically to all specified nodes.

            mode (str, optional):
                Defines how new reservoir conditions interact with existing ones:
                - `'add'` (default): add new reservoir BCs; raises an error if one
                  already exists at a node.
                - `'overwrite'`: replace existing reservoir BCs at the given nodes.
                - `'remove'`: remove reservoir BCs from the specified nodes.

        Raises:
            ValueError: If an unrecognized mode or non-scalar exchange rate is
                provided, or if a reservoir BC already exists in `'add'` mode.
        """

        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}

        if 'reservoir' not in self.boundary_conditions:
            self.boundary_conditions['reservoir'] = []

        nodes = normalize_target_ids(nodes)
        fixed_exchange_rate = broadcast_boundary_values(nodes, fixed_exchange_rate)

        if mode == 'remove':
            self.boundary_conditions['reservoir'] = [
                bc for bc in self.boundary_conditions['reservoir']
                if all(n not in nodes for n in bc.target_ids)
            ]
            return

        if mode not in ('add', 'overwrite'):
            raise ValueError("mode must be one of 'add', 'overwrite', or 'remove'.")

        for node, rate in zip(nodes, fixed_exchange_rate):
            if mode == 'overwrite':
                self.boundary_conditions['reservoir'] = [
                    bc for bc in self.boundary_conditions['reservoir']
                    if node not in bc.target_ids
                ]
            elif mode == 'add':
                for bc in self.boundary_conditions['reservoir']:
                    if node in bc.target_ids:
                        raise ValueError(f"Reservoir BC already exists at node {node}. Use mode='overwrite' to replace it.")

            if not isinstance(rate, Real):
                raise ValueError(f"Reservoir fixed_exchange_rate at node {node} must be a scalar value.")

            bc = ConstantBC([node], value=float(rate), bc_type='volumetric')
            self.boundary_conditions['reservoir'].append(bc)


    def set_stop_conditions(self, flowrate_condition=None, flowrate_threshold=0.98):
        
        if flowrate_condition is not None:
            self.flowrate_condition = flowrate_condition
            self.flowrate_threshold = flowrate_threshold
            self.stop_condition_set = True
        else:
            self.stop_condition_set = False
        
    # def set_observation_points(self, nodes, variables, interval=1.0):
    #     self.observation_recorder = ObservationRecorder(nodes, variables, interval)

    def set_observation_points(self, nodes, variables, interval=1.0):
        """Record selected node-based time series during transient simulations.

        Supported variables are:
            - ``'water_depth'``: water depth at each observed node.
            - ``'connected_abs_flowrate'``: sum of absolute flowrates through
              all conduits connected to each observed node.
            - ``'connected_net_flowrate'``: signed net flowrate into each
              observed node from all connected conduits.
            - ``'concentrations'``: concentration at each observed node when
              transport is enabled.
            - ``'mass'``: mass at each observed node when transport is enabled.
        """

        # Check if user wants C and M saved in observation and stop is transport is not enabled
        wants_trans_output = any(v in ("concentrations", "mass") for v in variables)
        if wants_trans_output and not getattr(self.simulation_settings, "enable_transport", False):
            raise ValueError(
                "Observation variables include 'concentrations' and/or 'mass'"
                "but enable_transport=False. Enable transport or remove these variables."
            )
        
        self.observation_recorder = ObservationRecorder(nodes, variables, interval)
    
    def get_observation_dataframe(self):
        if self.observation_recorder:
            return self.observation_recorder.to_dataframe()
        else:
            raise RuntimeError("No observation recorder initialized.")
        
    def _check_bc_conflicts(self):
        """
        Validates that boundary condition assignments are physically consistent.

        This function enforces exclusive behavior and logical consistency between
        hydraulic and transport boundary conditions. It ensures that nodes are not
        assigned multiple incompatible BCs and that transport BCs only exist where
        the corresponding hydraulic BCs are defined.

        Rules enforced:
            1. A node cannot have both a prescribed water depth BC and an inflow BC.
            2. A node cannot have both a prescribed water depth BC and a critical depth BC.
            3. A node cannot have both an inflow BC and a critical depth BC.
            4. Transport BCs are only allowed if transport is enabled.
            5. A node cannot have both inflow concentration BC and water-depth concentration BC.
            6. A node with water depth concentration BC must also have a water depth BC.
            7. A node with inflow concentration BC must also have an inflow BC.
            8. A node cannot have a mass-injection BC together with any concentration BC
            (water depth concentration or inflow concentration) on the same node.
            9. A reservoir node cannot also have water depth, inflow, or critical depth BCs.
            10. A node cannot have duplicate reservoir BCs.

        Raises:
            ValueError: If any conflicting or inconsistent boundary conditions are detected.
        """

        # Collect nodes having a BC type
        # Use sets as they remove duplicates (i.e. nodes with multiple BCs appear only once)
        # Collect node sets
        wd_nodes            = set()
        inflow_nodes        = set()
        crit_nodes          = set()
        wd_conc_nodes       = set()
        inflow_conc_nodes   = set()
        mass_inj_nodes      = set()
        reservoir_nodes     = set()
        reservoir_node_list = []

        if hasattr(self, "boundary_conditions"):
            bcs = self.boundary_conditions

            wd_nodes.update(
                n for bc in bcs.get("waterdepth", []) for n in bc.target_ids
            )
            inflow_nodes.update(
                n for bc in bcs.get("inflow", []) for n in bc.target_ids
            )
            crit_nodes.update(
                n for bc in bcs.get("critical_depth", []) for n in bc.target_ids
            )
            wd_conc_nodes.update(
                n for bc in bcs.get("waterdepth_concentration", []) for n in bc.target_ids
            )
            inflow_conc_nodes.update(
                n for bc in bcs.get("inflow_concentration", []) for n in bc.target_ids
            )
            mass_inj_nodes.update(
                n for bc in bcs.get("mass_injection", []) for n in bc.target_ids
            )
            reservoir_node_list = [
                n for bc in bcs.get("reservoir", []) for n in bc.target_ids
            ]
            reservoir_nodes.update(reservoir_node_list)

        transport_enabled = bool(getattr(self.simulation_settings, "enable_transport", False))

        # Transport disabled: Do NOT allow any transport BCs (including mass injection)
        if not transport_enabled:
            offending = []
            if wd_conc_nodes or inflow_conc_nodes:
                offending.append("inflow_concentration or waterdepth_concentration")
            if mass_inj_nodes:
                offending.append("mass_injection")
            if offending:
                raise ValueError(
                    "Transport BCs are defined (" + ", ".join(offending) + ") "
                    "but transport is disabled. Enable transport or remove these BCs."
                )

        # Hydraulic BC exclusivity
        both = wd_nodes & inflow_nodes
        if both:
            raise ValueError(
                f"Conflict: nodes {sorted(both)} have both prescribed water depth and inflow BCs."
            )

        both = wd_nodes & crit_nodes
        if both:
            raise ValueError(
                f"Conflict: nodes {sorted(both)} have both prescribed water depth and critical depth BCs."
            )

        both = inflow_nodes & crit_nodes
        if both:
            raise ValueError(
                f"Conflict: nodes {sorted(both)} have both inflow and critical depth BCs."
            )

        duplicate_reservoir_nodes = {
            n for n in reservoir_node_list if reservoir_node_list.count(n) > 1
        }
        if duplicate_reservoir_nodes:
            raise ValueError(
                f"Conflict: nodes {sorted(duplicate_reservoir_nodes)} have duplicate reservoir BCs."
            )

        both = reservoir_nodes & wd_nodes
        if both:
            raise ValueError(
                f"Conflict: nodes {sorted(both)} have both reservoir and prescribed water depth BCs."
            )

        both = reservoir_nodes & inflow_nodes
        if both:
            raise ValueError(
                f"Conflict: nodes {sorted(both)} have both reservoir and inflow BCs."
            )

        both = reservoir_nodes & crit_nodes
        if both:
            raise ValueError(
                f"Conflict: nodes {sorted(both)} have both reservoir and critical depth BCs."
            )

        # Transport BC consistency (only if enabled)
        if transport_enabled:
            both = wd_conc_nodes & inflow_conc_nodes
            if both:
                raise ValueError(
                    f"Conflict: nodes {sorted(both)} have both inflow concentration and "
                    "water depth concentration BCs."
                )

            missing = wd_conc_nodes - wd_nodes
            if missing:
                raise ValueError(
                    f"Invalid BC: nodes {sorted(missing)} have a water-depth concentration BC "
                    "but no water-depth BC."
                )

            missing = inflow_conc_nodes - inflow_nodes
            if missing:
                raise ValueError(
                    f"Invalid BC: nodes {sorted(missing)} have an inflow concentration BC "
                    "but no inflow BC."
                )

            # EITHER mass injection OR any concentration BC
            both = mass_inj_nodes & (wd_conc_nodes | inflow_conc_nodes)
            if both:
                raise ValueError(
                    f"Conflict: nodes {sorted(both)} have a mass-injection BC together with a "
                    "concentration BC (inflow_concentration or waterdepth_concentration). "
                    "Use only one transport BC type per node."
                )

        # Passed all tests
        self.logger.info("Boundary condition consistency check passed.")
        
    def _prepare_timestep_state(self):
        """
        Prepare state buffers for the next hydraulic timestep.

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
         
    def _accept_hydraulic_solution(self):
        """
        Accept the solved hydraulic state for the current timestep.

        This method updates the hydraulic state properties by assigning the new values
        to the current state variables.
        """
        
        self.Q = self.Q_new
        self.y = self.y_new
        self.a_mid = self.a_mid_new
        self.dQ = self.dQ_new

        # Transport: Accept the volume change after each timestep 
        self.V_node += self._dV_last
        self.V_node[self.V_node < 0.0] = 0.0  # For safety

    def _accept_picard_iteration(self):
        """
        Accept the latest Picard iterate as the reference for the next iteration.
        """

        np.copyto(self.Q_prev_i, self.Q_new)
        np.copyto(self.y_prev_i, self.y_new)

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
            #v_mid = self.Q_prev_i / self.a_mid_new
            # Slot is only taken into account for free-surface cases
            if self.geometry_channel:
                v_mid = self.Q_prev_i / self.a_mid_new
            else:
                v_mid = np.where(
                    self.is_full_y_mid,
                    self.Q_prev_i / self.full_conduit_areas,
                    self.Q_prev_i / self.a_mid_new
                )

            froude = np.abs(v_mid) / np.sqrt(self.gravity * self.a_mid_new / w_mid)


            # Store to use for adaptive timestep update outside of dynamic_wave
            self._v_mid_last = v_mid
            self._froude_last = froude  
           
            # Compute alpha for upstream weighting
            alpha = compute_upstream_weight_alpha(froude, self.is_full_y_mid)
            
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

            if self._has_picard_converged():
                return True, iteration + 1
            
            # Accept the iteration state for the next Picard iteration
            self._accept_picard_iteration()
            
            iteration += 1

        return False, iteration + 1  # if max_iterations hit

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
            self.is_full_y1 = self.cross_section_geometry.is_full(y1)
            self.is_full_y2 = self.cross_section_geometry.is_full(y2)
            self.is_full_y_mid = self.cross_section_geometry.is_full(y_mid)
            
 
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
                compute_slot_width(
                    y1[self.is_full_y1],
                    self.cross_section_geometry.full_depths[self.is_full_y1],
                )
            )
        if np.any(self.is_full_y2):
            slot_w2[self.is_full_y2] = (
                compute_slot_width(
                    y2[self.is_full_y2],
                    self.cross_section_geometry.full_depths[self.is_full_y2],
                )
            )
        if np.any(self.is_full_y_mid): 
            slot_w_mid[self.is_full_y_mid] = (
                compute_slot_width(
                    y_mid[self.is_full_y_mid],
                    self.cross_section_geometry.full_depths[self.is_full_y_mid],
                )
            )

        if not self.geometry_channel:
            conduit_w1 = self.cross_section_geometry.top_width(y1)
            conduit_w2 = self.cross_section_geometry.top_width(y2)
            conduit_w_mid = self.cross_section_geometry.top_width(y_mid)
    
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
                    conduit_w1[mask1],
                )
                
                w2[mask1] = np.where(
                    self.is_full_y2[mask1],
                    slot_w2[mask1],
                    conduit_w2[mask1],
                )
                
                w_mid[mask1] = np.where(
                    self.is_full_y_mid[mask1],
                    slot_w_mid[mask1],
                    conduit_w_mid[mask1],
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
                w1[mask2] = conduit_w1[mask2]
                w2[mask2] = conduit_w2[mask2]
                w_mid[mask2] = conduit_w_mid[mask2]
                
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
                w1[mask3] = conduit_w1[mask3]
                w2[mask3] = np.where(
                    self.is_full_y2[mask3],
                    slot_w2[mask3],
                    conduit_w2[mask3],
                )
                
                w_mid[mask3] = np.where(
                    self.is_full_y_mid[mask3],
                    slot_w_mid[mask3],
                    conduit_w_mid[mask3],
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
                    conduit_w1[mask4],
                )
                
                w2[mask4] = conduit_w2[mask4]
                
                w_mid[mask4] = np.where(
                    self.is_full_y_mid[mask4],
                    slot_w_mid[mask4],
                    conduit_w_mid[mask4],
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
    
    # # Slot is only taken into account for free-surface cases
    def _compute_discharge_areas(self, y1, y2, y_mid, slot_widths):

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
            a1 = self.cross_section_geometry.area(y1)
            a2 = self.cross_section_geometry.area(y2)
            self.a_mid_new = self.cross_section_geometry.area(y_mid)

        return a1, a2, self.a_mid_new
    
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
 
        # Check if any conduit is full and halve the Courant number
        # Currently not used
        if np.any(self.is_full_y_mid):
            effective_courant = self.courant #/ 2
        else:
            effective_courant = self.courant

        # 1. Courant–Froude criterion
        v_mask = np.abs(v_mid) > 1e-8
        dt_froude = np.full_like(v_mid, np.inf)
        dt_froude[v_mask] = (
            1 / np.abs(v_mid[v_mask]) * (froude[v_mask] / (1 + froude[v_mask])) * effective_courant
        )
        max_dt1 = np.nanmin(dt_froude)

        # 2. Storage (dydt) criterion
        dydt_mask = self.dydt > 1e-10
        dt_dydt = np.full_like(self.dydt, np.inf)
        dt_dydt[dydt_mask] = self.max_depths[dydt_mask] / self.dydt[dydt_mask]
        max_dt2 = np.nanmin(dt_dydt)

        # Choose conservative dt
        dt_new = min(max_dt1, max_dt2)

        # Fallback logic
        if not np.isfinite(dt_new) or dt_new <= 0.0:
            self.logger.warning("Invalid dt computed. Falling back to dt_init.")
            dt_new = self.dt_init

        # Enforce max limit
        #self.dt = min(dt_new, self.dt_max)
        
        # Enforce limits (dt_init is lower bound)
        dt_new = max(dt_new, self.dt_init)
        self.dt = min(dt_new, self.dt_max)

                    
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
            
        # else:
        #     # Calculate hydraulic radius for both free surface and pressurized conditions
        #     radii = self.conduit_diameters * 0.5
        #     theta = 2 * np.arccos(np.clip((radii - flow_depths) / radii, -1, 1))
        #     wetted_perimeter = np.where(is_full, 2 * np.pi * radii + slot_width, radii * theta)
        #     hydraulic_radii = flow_areas / wetted_perimeter

        # Slot is only taken into account for free-surface cases
        else:
            hydraulic_radii = self.cross_section_geometry.hydraulic_radius(
                flow_depths,
                areas=flow_areas,
            )
            
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

        Advances conduit discharge from self.Q_old_t to self.Q_new over one
        time step self.dt by combining a pressure-gradient term (upstream weighted),
        inertial terms with a correction applied only at nodes with flux boundaries,
        and friction losses computed either with Churchill only (pressurized and
        free-surface via D_eff) or with a hybrid approach (Churchill for pressurized,
        Manning for free-surface), depending on self.friction_model.


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

        self.f.fill(0.0)
        self.dQ_friction.fill(0.0)

        # Not needed but easier to read...
        f = self.f
        dQ_friction = self.dQ_friction
        q_correction = self.q_correction
        D_eff = self.D_eff

        # Pressure term (upstream weighting)
        #dQ_pressure = -self.gravity * a_mid_upwtd * (h2 - h1) / self.conduit_lengths * self.dt
        ### 14.4.26, slot not taken into account for pressurized flows
        if self.geometry_channel:
            a_pressure = a_mid_upwtd
        else:
            a_pressure = np.where(self.is_full_y_mid, self.full_conduit_areas, a_mid_upwtd)

        dQ_pressure = (
            -self.gravity * a_pressure * (h2 - h1) / self.conduit_lengths * self.dt
        )
    
        # Add correction term. This is currently only applied for flux BCs
        flux_n1 = self.bc_flux_node[self.n_indices1]
        flux_n2 = self.bc_flux_node[self.n_indices2]
        q_correction[:] = 0.5 * w_mid * (flux_n1 + flux_n2)
                             
        # Inertial terms (alpha is zero when pressurized)
        # Apply the momentum correction term to the inertia term
        dQ_inertia1 = alpha * 2 * v_mid * (self.a_mid_new - self.a_mid_old_t - q_correction * self.dt)
        dQ_inertia2 = alpha * v_mid * v_mid * (a2 - a1) / self.conduit_lengths * self.dt     
        
        # Precompute 
        #abs_vmid = np.abs(v_mid)

        # Case: Open channel geometry
        if self.geometry_channel == True:

            # Approximate Reynolds number using flow depth as hydraulic radius.
            # Notice that r_mid takes into account channel width (finite or infinite)
            D_eff[:] = 4.0 * r_mid
            self.Re_conduit[:] = (self.rho * np.abs(v_mid) * D_eff) / self.dyn_viscosity
   
            # Compute friction term using Manning's equation for free-surface flow
            # Manning n is provided directly via physical properties
            dQ_friction[~self.is_full_y_mid] = (
                self.gravity * self.conduit_manning[~self.is_full_y_mid]**2 *
                np.abs(v_mid[~self.is_full_y_mid]) /
                (r_mid_upwtd[~self.is_full_y_mid]**(4/3)) * self.dt
            )

        # Case: Circular conduits
        else:
            # Only Churchill friction
            if self.friction_model == "churchill":

                # Define effective diameter
                D_eff[self.is_full_y_mid] = self.full_hydraulic_diameters[self.is_full_y_mid]
                D_eff[~self.is_full_y_mid] = 4 * r_mid[~self.is_full_y_mid]

                # Reynolds number using D_eff
                self.Re_conduit[:] = (
                    self.rho * np.abs(v_mid) * D_eff / self.dyn_viscosity
                )

                # Define masks for flow regimes under pressurized conditions
                laminar_flow_mask = self.Re_conduit <= 2300
                turbulent_flow_mask = self.Re_conduit > 2300
        
                # f = 64 / Re (laminar), np.clip avoids division by zero and caps very small Re 
                f[laminar_flow_mask] = 64.0 / np.clip(self.Re_conduit[laminar_flow_mask], 1e-12, None)

                # Turbulent: Churchill equation
                if np.any(turbulent_flow_mask):
                    f[turbulent_flow_mask] = compute_churchill_friction_factor(
                        self.Re_conduit[turbulent_flow_mask],
                        self.conduit_epsilon[turbulent_flow_mask],
                        D_eff[turbulent_flow_mask],
                    )
              
                # Compute friction dQ term for all conduits using Churchill
                dQ_friction[:] = (f * np.abs(v_mid) / (8 * r_mid) * self.dt)

            # Hybrid friction (Churchill + Manning for free-surface flows)
            else:

                D_eff[self.is_full_y_mid]  = self.full_hydraulic_diameters[self.is_full_y_mid]
                D_eff[~self.is_full_y_mid] = 4.0 * r_mid[~self.is_full_y_mid]
                self.Re_conduit[:] = (self.rho * np.abs(v_mid) * D_eff) / self.dyn_viscosity

                # Define masks for flow regimes under pressurized conditions
                laminar_flow_mask = (self.Re_conduit <= 2300) & self.is_full_y_mid
                turbulent_flow_mask = (self.Re_conduit > 2300) & self.is_full_y_mid
        
                # f = 64 / Re (laminar), np.clip avoids division by zero and caps very small Re 
                f[laminar_flow_mask] = 64.0 / np.clip(self.Re_conduit[laminar_flow_mask], 1e-12, None)
        
                # Compute friction factor using the Churchill equation for turbulent flow
                # under pressurized conditions
                if np.any(turbulent_flow_mask):
                    f[turbulent_flow_mask] = compute_churchill_friction_factor(
                        self.Re_conduit[turbulent_flow_mask],
                        self.conduit_epsilon[turbulent_flow_mask],
                        D_eff[turbulent_flow_mask],
                    )
                    
                # Compute friction dQ term for pressurized conduits using Churchill
                dQ_friction[self.is_full_y_mid] = (
                    f[self.is_full_y_mid] * np.abs(v_mid[self.is_full_y_mid]) /
                    (8 * r_mid[self.is_full_y_mid]) * self.dt
                )

                # Compute Manning-based friction dQ for free-surface flow in circular conduits
                dQ_friction[~self.is_full_y_mid] = (
                    self.gravity * self.conduit_manning[~self.is_full_y_mid]**2 *
                    np.abs(v_mid[~self.is_full_y_mid]) /
                    (r_mid_upwtd[~self.is_full_y_mid]**(4/3)) * self.dt
                )
        
        # Compute dQ components and new flows Q_new
        self.Q_new[:] = (self.Q_old_t + dQ_pressure + dQ_inertia1 + dQ_inertia2)/(1 + dQ_friction)
       
        # Update flows using under-relaxation
        self.Q_new[:] = (1.0 - self.w) * self.Q_prev_i + self.w * self.Q_new
        
        # Check for flow rate sign changes to address potential numerical instabilities.
        # Currently not needed, but retained for future debugging.
        #is_sign_change = np.sign(self.Q_new) != np.sign(self.Q_prev_i)
        #if np.any(is_sign_change) == True:
        #     print("is sign change")
        #    self.Q_new[is_sign_change] = 1e-9 * np.sign(self.Q_new[is_sign_change])

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
            
        # Subtract flow from source node (n1) and add to target node (n2); sign handled by weights
        np.add(self.dQ_new,
               -np.bincount(self.n_indices1, weights=self.Q_new, minlength=self.network.Np),
               out=self.dQ_new)
        np.add(self.dQ_new,
               +np.bincount(self.n_indices2, weights=self.Q_new, minlength=self.network.Np),
               out=self.dQ_new)
 
        # Add nodal inflows from BCs (direct volumetric or converted from flux)
        #self.dQ_new += self.bc_inflow_vol_node
        #if isinstance(self.bc_flux_to_vol_node, np.ndarray) or self.bc_flux_to_vol_node != 0.0:
        #    self.dQ_new += self.bc_flux_to_vol_node

        self.dQ_new += self.bc_inflow_vol_node
        self.dQ_new += self.bc_flux_to_vol_node
        self.dQ_new += self.bc_reservoir_exchange_node

        # Compute the change in volume at each node (dV)
        dV = 0.5 * (self.dQ_old_t + self.dQ_new) * self.dt

        # Save change in nodal volume for transport
        self._dV_last = dV
        
        # Compute change in flow depths and new depths
        dy = dV / n_surface_a
        self.y_new = self.y_old_t + dy
                 
        # Update water depths using under-relaxation
        self.y_new[:] = (1.0 - self.w) * self.y_prev_i + self.w * self.y_new
    
        # Enforce water depth (_cache_hydraulic_bcs)
        self.y_new[self.bc_prescribed_y_mask] = self.bc_prescribed_y_vals[self.bc_prescribed_y_mask]    

        # Ensure water depths don't go negative
        self.y_new[self.y_new <= 0.0] = 0.0
        
        # Compute change in water depths
        self.dydt[:] = np.abs(self.y_new - self.y_old_t) / self.dt
                
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
    
    def _print_timestep_info(self, n_iterations, converged, froude):
        """
        Print information about the current timestep at specified intervals.

        This method prints details about the simulation at specified intervals
        including the time step size, maximum Froude number, current timestep,
        current simulation time, Picard iteration count, convergence status,
        cumulative convergence failures, relative L2 changes, maximum Reynolds
        number, and average Reynolds number.

        Args:
            n_iterations (int): Number of Picard iterations used in the timestep.
            converged (bool): Whether the Picard solve converged.
            froude (ndarray): Array of Froude numbers for the conduits.

        """
    
        # Print information at specified intervals
        if math.fmod(self.current_timestep, self.print_info_interval) == 0:
            print(f'Timestep = {self.current_timestep}, '
                  f'Time = {self.current_time:.1f}, dt = {self.dt:.2e}')
            print(f'Converged = {converged}, Picard iterations = {n_iterations}, '
                  f'convergence fails = {self.convergence_fails}')
            print(f'relL2(y) = {self.relative_y_l2_norm:.2e}, '
                  f'relL2(Q) = {self.relative_Q_l2_norm:.2e}')
            print(f'Max Froude = {np.max(froude):.2f}')
            print(f'Max Re = {np.max(self.Re_conduit):.2f} '
                  f'Avg Re = {np.mean(self.Re_conduit):.2f}\n')


    def _cache_hydraulic_bcs(self):
        """Evaluate BC values at current time and cache per-node arrays."""
        t = self.current_time

        # Reset arrays
        self.bc_inflow_vol_node.fill(0.0)
        self.bc_flux_node.fill(0.0)
        self.bc_flux_to_vol_node.fill(0.0)
        self.bc_reservoir_exchange_node.fill(0.0)
        self.bc_prescribed_y_mask.fill(False)
        self.bc_prescribed_y_vals.fill(0.0)

        for bc in self.boundary_conditions.get('inflow', []):
            v = bc.get_value(t)
            if getattr(bc, 'bc_type', 'volumetric') == 'flux':
                self.bc_flux_node[bc.target_ids] += v
            else:
                self.bc_inflow_vol_node[bc.target_ids] += v

        for bc in self.boundary_conditions.get('reservoir', []):
            v = bc.get_value(t)
            self.bc_reservoir_exchange_node[bc.target_ids] += v

        # Dirichlet water depth BCs
        for bc in self.boundary_conditions.get('waterdepth', []):
            v = bc.get_value(t)
            self.bc_prescribed_y_mask[bc.target_ids] = True
            self.bc_prescribed_y_vals[bc.target_ids] = v

        # Precompute flux-to-volume conversion for channels
        if np.any(self.bc_flux_node):
            if self.geometry_channel:
                if self.channel_type == 'infinite':
                    self.bc_flux_to_vol_node[:] = self.bc_flux_node * self.half_lengths_sum_per_node
                else:
                    self.bc_flux_to_vol_node[:] = (self.bc_flux_node * self.channel_width *
                                            self.half_lengths_sum_per_node)
            else:
                raise ValueError("Flux BCs detected but geometry_channel=False."
                                "Use volumetric BCs or implement conversion for new geometry.")
        
        # Total external inflow per node
        self.bc_Qin_node = self.bc_inflow_vol_node + self.bc_flux_to_vol_node

    def _cache_transport_bcs(self, t=None):
        """Evaluate transport BC values at current time and cache per-node arrays."""

        t = self.current_time

        # reset arrays
        self.bc_Cin_node.fill(0.0)
        self.bc_prescribed_C_vals.fill(0.0)
        self.bc_prescribed_C_mask.fill(False)
        self.bc_mass_inflow_rate_node.fill(0.0)
        self.bc_mass_injection_node.fill(0.0)

        # inflow concentration Cin(t)
        for bc in self.boundary_conditions.get('inflow_concentration', []):
            v = bc.get_value(t)
            self.bc_Cin_node[bc.target_ids] = v 

        # Dirichlet concentration at waterdepth nodes
        for bc in self.boundary_conditions.get('waterdepth_concentration', []):
            v = bc.get_value(t)
            self.bc_prescribed_C_vals[bc.target_ids] = v
            self.bc_prescribed_C_mask[bc.target_ids] = True

        # Mass inflow rate from external water inflow
        self.bc_mass_inflow_rate_node[:] = self.bc_Qin_node * self.bc_Cin_node

        # Mass injection (direct source) [kg/s]
        for bc in self.boundary_conditions.get('mass_injection', []):
            mdot = bc.get_value(t)              # kg/s
            self.bc_mass_injection_node[bc.target_ids] = mdot


    def _has_picard_converged(self):
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
        
    def _update_timestep_error_norms(self):
        """
        Update relative error norms between the new and old states.

        This method computes relative L2 norms for both `y` and `Q`, comparing
        the new state (`self.y_new`, `self.Q_new`) with the previous state
        (`self.y_old_t`, `self.Q_old_t`).

        The relative norms are computed by dividing the absolute norm of the
        difference by the norm of the new state. If the norm of the new state
        is zero, the corresponding relative norm is set to 0.0 to avoid division
        by zero.

        Sets:
            self.relative_y_l2_norm (float): Relative L2 norm of `y`.
            self.relative_Q_l2_norm (float): Relative L2 norm of `Q`.
        """
        
        y_l2_norm = np.linalg.norm(self.y_new - self.y_old_t)
        if np.linalg.norm(self.y_new) != 0:
            self.relative_y_l2_norm = y_l2_norm / np.linalg.norm(self.y_new)
        else:
            self.relative_y_l2_norm = 0.0

        Q_l2_norm = np.linalg.norm(self.Q_new - self.Q_old_t)
        if np.linalg.norm(self.Q_new) != 0:
            self.relative_Q_l2_norm = Q_l2_norm / np.linalg.norm(self.Q_new)
        else:
            self.relative_Q_l2_norm = 0.0
        
    def _has_reached_steady_state(self):
        """
        Check whether steady-state convergence has been reached.

        This method evaluates convergence based on the previously computed
        relative L2 norms of water depth (`y`) and discharge (`Q`). The system
        is considered converged when both relative norms are below the steady-
        state tolerance `self.ss_rel_l2tol`.

        The relative norms (`self.relative_y_l2_norm` and
        `self.relative_Q_l2_norm`) are computed by calling
        `_update_timestep_error_norms()`.

        Returns:
            bool: True if both `y` and `Q` satisfy the steady-state convergence
            criterion, False otherwise.
        """

        is_y_converged = (
            self.relative_y_l2_norm < self.ss_rel_l2tol
        )

        is_Q_converged = (
            self.relative_Q_l2_norm < self.ss_rel_l2tol
        )

        return is_y_converged and is_Q_converged
 

    # def _time_dependent_flowrate(self, current_time, start_time, end_time, initial_rate, peak_rate):
        
    #     """
    #     Calculate the inflow rate based on the current time.
        
    #     Args:
    #         current_time (float): The current time in seconds.
    #         start_time (float): The time at which the ramp-up starts.
    #         end_time (float): The time at which the ramp-down ends.
    #         initial_rate (float): The initial flow rate at the start time.
    #         peak_rate (float): The peak flow rate during the ramp-up.
            
    #     Returns:
    #         float: The inflow rate at the given time.
    #     """
    #     # Duration for ramping up
    #     ramp_up_duration = (end_time - start_time) / 2
        
    #     # Ramp-up phase
    #     if start_time <= current_time < start_time + ramp_up_duration:
    #         return initial_rate + (peak_rate - initial_rate) * \
    #                ((current_time - start_time) / ramp_up_duration)
        
    #     # Ramp-down phase
    #     elif start_time + ramp_up_duration <= current_time <= end_time:
    #         return peak_rate - (peak_rate - initial_rate) * \
    #                ((current_time - (start_time + ramp_up_duration)) / ramp_up_duration)
        
    #     # Outside the defined period, return the initial flow rate
    #     else:
    #         return initial_rate
        

    # # New Transport functions
    # def set_initial_concentration(self, C0):
    #     np.copyto(self.C, C0)
    #     # Mass will be set after V_node is initialized (first step) or here if V_node is known.
    #     if np.any(self.V_node):
    #         self.M = self.C * self.V_node

    def set_initial_concentrations(self, C0=0.0):
        """
        Initialize transport state (C, M, V_node) from current hydraulics (y, Q).
        Requires simulation_settings.enable_transport == True.
        """
        # If enable_transport is False raise warning and stop
        if not getattr(self.simulation_settings, "enable_transport", False):
            msg = ("set_initial_concentration() called but enable_transport=False. "
                "Enable transport (solver_settings.enable_transport=True) before initializing concentrations.")
            try:
                self.logger.warning(msg)
            except Exception:
                pass
            raise RuntimeError(msg)

        # Build initial V_node from discharge areas (works for free-surface & pressurized)
        y0   = self.y.copy()
        y1   = y0[self.n_indices1]
        y2   = y0[self.n_indices2]
        y_mid = 0.5*(y1 + y2)

        # Get current pressurization masks 
        self._compute_conduit_state(y1, y2, y_mid)

        # Get slot width & areas
        _, _, _, slot_w_mid, _ = self._compute_surface_area(y1, y2, y_mid)
        A1, A2, A_mid = self._compute_discharge_areas(y1, y2, y_mid, slot_w_mid)

        L = self.conduit_lengths.astype(float)
        V_conduit = (L/6.0) * (A1 + 4.0*A_mid + A2)

        # Distribute half of each conduit volume to its end nodes
        self.V_node[:] = 0.0
        np.add.at(self.V_node, self.n_indices1, 0.5 * V_conduit)
        np.add.at(self.V_node, self.n_indices2, 0.5 * V_conduit)
        self.V_node[self.V_node < 0.0] = 0.0  # safety

        # Set initial concentration and mass
        self.C[:] = C0
        self.M[:] = self.C * self.V_node



    def _validate_transport_outputs(self, desired_outputs):
        wants_transport = any(
            desired_outputs.get(k, False)
            for k in ("concentrations", "mass")
        )
        if wants_transport and not getattr(self.simulation_settings, "enable_transport", False):
            raise ValueError(
                "User requests to save transport fields (e.g., 'concentrations' or 'mass') "
                "but enable_transport=False. Enable transport or remove these outputs."
            )
        
    
    def set_inflow_concentration_BC(self, nodes, values, mode='add', extrapolate='hold'):
        """
        Set inflow concentration boundary conditions at specified nodes.

        This defines the tracer concentration `C_{in}(t)` associated with hydraulic
        inflows (e.g., those defined via `set_inflow_BC`). The mass input rate is computed
        later as `Q_in(node, t) * C_in(node, t)`.

        Args:
            nodes (int, list[int], or np.ndarray):
                Index or indices of nodes where the inflow concentration BC is applied.

            values (float | int | tuple | list | np.ndarray):
                Concentration definitions. If a single scalar or tuple is provided, it is
                broadcast to all specified nodes. Supported formats:

                * float or int: constant concentration (e.g., mg/L or kg/m³).
                * tuple:
                    - `('timeseries', times, values)`: interpolated time series.
                    - `('box', value, t0, t1 [, value_before=0.0, value_after=0.0])`:
                    constant concentration during `[t0, t1]`, with optional values
                    before and after.
                * list or 1D np.ndarray: per-node concentrations, one entry per node.
                * 0D np.ndarray (e.g., `np.array(0.2)`): treated as a scalar and broadcast.

            mode (str, optional):
                How new BCs interact with existing ones:
                - `'add'` (default): add new BCs; raises if a BC already exists at a node.
                - `'overwrite'`: replace any existing BCs at the given nodes.
                - `'remove'`: remove BCs from the specified nodes.

            extrapolate (str, optional):
                Extrapolation behavior for `'timeseries'` BCs:
                - `'hold'` (default): hold first/last value constant outside the defined range.
                - `'zero'`: set value to zero outside the defined range.

        Raises:
            ValueError: If an unrecognized BC format is provided, if a duplicate node
                exists in `'add'` mode, or if the number of per-node values does not
                match the number of nodes.

        Notes:
            This BC contributes a mass source term only when the hydraulic solution yields
            positive external inflow at a node.
        """
                
        # init dict
        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}
        if 'inflow_concentration' not in self.boundary_conditions:
            self.boundary_conditions['inflow_concentration'] = []

        nodes = normalize_target_ids(nodes)
        values = broadcast_boundary_values(nodes, values)

        # modes
        if mode == 'remove':
            self.boundary_conditions['inflow_concentration'] = [
                bc for bc in self.boundary_conditions['inflow_concentration']
                if all(n not in nodes for n in bc.target_ids)
            ]
            return

        for node, val in zip(nodes, values):
            if mode == 'overwrite':
                self.boundary_conditions['inflow_concentration'] = [
                    bc for bc in self.boundary_conditions['inflow_concentration']
                    if node not in bc.target_ids
                ]
            elif mode == 'add':
                for bc in self.boundary_conditions['inflow_concentration']:
                    if node in bc.target_ids:
                        raise ValueError(
                            f"Inflow concentration BC already exists at node {node}. "
                            "Use mode='overwrite' to replace it."
                        )

            # build BC object
            if isinstance(val, Real):
                bc = ConstantBC([node], value=float(val))
            elif isinstance(val, tuple) and val[0] == 'box':
                _, v_during, t0, t1, *rest = val
                v_before = rest[0] if len(rest) > 0 else 0.0
                v_after  = rest[1] if len(rest) > 1 else 0.0
                bc = BoxBC([node], v_during, t0, t1, v_before, v_after)
            elif isinstance(val, tuple) and val[0] == 'timeseries':
                _, times, conc_values = val[:3]
                bc = TimeSeriesBC([node], times=times, values=conc_values, extrapolate=extrapolate)
            else:
                raise ValueError(f"Unrecognized inflow concentration format at node {node}: {val}")

            self.boundary_conditions['inflow_concentration'].append(bc)


    def set_waterdepth_concentration_BC(self, nodes, values, mode='add', extrapolate='hold'):
        """
        Set concentration boundary conditions at nodes with prescribed water depth (Dirichlet head BCs).

        This defines the boundary concentration C_b(t) associated with nodes where the water depth
        is prescribed via `set_waterdepth_BC(...)`. The concentration is applied only when water
        flows into the domain through those nodes (i.e., when external inflow is positive due to
        the hydraulic solution).

        Args:
            nodes (int, list of int, or np.ndarray):
                Index or indices of nodes where the water-depth concentration boundary condition
                is applied.

            values (float, tuple, list, or np.ndarray):
                Concentration definitions. If a single scalar or tuple is provided, it is broadcast
                to all specified nodes. Supported formats:

                * **float** or **int**: constant concentration (e.g., mg/L or kg/m³).
                * **tuple**:
                    - `('timeseries', times, values)`: interpolated time series.
                    - `('box', value, t0, t1 [, value_before=0.0, value_after=0.0])`:
                      constant concentration during `[t0, t1]`, with optional values before and after.
                * **list** or **1D np.ndarray**: per-node concentrations, one entry per node.
                * **0D np.ndarray** (e.g., `np.array(0.2)`): treated as a scalar and broadcast.

            mode (str, optional):
                Defines how new BCs interact with existing ones:
                - `'add'` (default): add new BCs; raises an error if a BC already exists at a node.
                - `'overwrite'`: replace any existing BCs at the given nodes.
                - `'remove'`: remove BCs from the specified nodes.

            extrapolate (str, optional):
                Extrapolation behavior for `'timeseries'` BCs:
                - `'hold'` (default): hold the first/last value constant outside the defined range.
                - `'zero'`: set BC value to zero outside the defined range.

        Raises:
            ValueError: If an unrecognized BC format is provided, if duplicate nodes are given in
                `'add'` mode, or if the number of values does not match the number of nodes.

        Notes:
            The actual inflow or outflow at these nodes is determined by the hydraulic solution.
            This boundary condition contributes a mass inflow term only when the local discharge
            corresponds to inflow (Q_in > 0).
        """
        # init dict
        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}
        if 'waterdepth_concentration' not in self.boundary_conditions:
            self.boundary_conditions['waterdepth_concentration'] = []

        nodes = normalize_target_ids(nodes)
        values = broadcast_boundary_values(nodes, values)

        # modes
        if mode == 'remove':
            self.boundary_conditions['waterdepth_concentration'] = [
                bc for bc in self.boundary_conditions['waterdepth_concentration']
                if all(n not in nodes for n in bc.target_ids)
            ]
            return

        for node, val in zip(nodes, values):
            if mode == 'overwrite':
                self.boundary_conditions['waterdepth_concentration'] = [
                    bc for bc in self.boundary_conditions['waterdepth_concentration']
                    if node not in bc.target_ids
                ]
            elif mode == 'add':
                for bc in self.boundary_conditions['waterdepth_concentration']:
                    if node in bc.target_ids:
                        raise ValueError(
                            f"Water-depth concentration BC already exists at node {node}. "
                            "Use mode='overwrite' to replace it."
                        )

            # build BC object
            if isinstance(val, Real):
                bc = ConstantBC([node], value=float(val))
            elif isinstance(val, tuple) and val[0] == 'box':
                _, v_during, t0, t1, *rest = val
                v_before = rest[0] if len(rest) > 0 else 0.0
                v_after  = rest[1] if len(rest) > 1 else 0.0
                bc = BoxBC([node], v_during, t0, t1, v_before, v_after)
            elif isinstance(val, tuple) and val[0] == 'timeseries':
                _, times, conc_values = val[:3]
                bc = TimeSeriesBC([node], times=times, values=conc_values, extrapolate=extrapolate)
            else:
                raise ValueError(
                    f"Unrecognized water-depth concentration format at node {node}: {val}"
                )

            self.boundary_conditions['waterdepth_concentration'].append(bc)


    def set_mass_injection_BC(self, nodes, values, mode='add', extrapolate='hold'):
        """
        Set mass-injection boundary conditions at nodes.

        This defines a mass-injection rate ṁ(t) [kg/s] applied at the specified nodes.
        The injected mass is added as an external source term during transport.

        Args:
            nodes (int, list of int, or np.ndarray):
                Index or indices of nodes where the mass-injection BC is applied.

            values (float, tuple, list, or np.ndarray):
                Mass-injection definitions. If a single scalar or tuple is provided, it
                is broadcast to all specified nodes. Supported formats:

                * float or int:
                    Constant mass-injection rate in kg/s.

                * tuple:
                    - ('timeseries', times, rates):
                        Piecewise-linear time series, with 'times' (array-like) and
                        'rates' (array-like, kg/s). Outside the defined range the
                        behavior follows 'extrapolate'.
                    - ('box', rate, t0, t1 [, rate_before=0.0, rate_after=0.0]):
                        Constant rate 'rate' (kg/s) during [t0, t1]. Optional
                        'rate_before' and 'rate_after' (kg/s) set values outside
                        the window.

                * list or 1D np.ndarray:
                    Per-node specifications (one entry per node).

                * 0D np.ndarray (e.g., np.array(0.2)):
                    Treated as a scalar and broadcast.

            mode (str, optional):
                Behavior when BCs already exist at the given nodes:
                - 'add' (default): add new BCs; raises if a BC exists at a node.
                - 'overwrite': replace any existing BCs at the given nodes.
                - 'remove': remove BCs from the specified nodes.

            extrapolate (str, optional):
                Extrapolation behavior for 'timeseries' BCs:
                - 'hold' (default): hold the first/last value constant outside range.
                - 'zero': set to zero outside range.

        Raises:
            ValueError: If an unrecognized BC format is provided, if duplicate nodes
                are given in 'add' mode, or if the number of values does not match
                the number of nodes.
        """
        # init dict
        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}
        if 'mass_injection' not in self.boundary_conditions:
            self.boundary_conditions['mass_injection'] = []

        nodes = normalize_target_ids(nodes)
        values = broadcast_boundary_values(nodes, values)

        # Modes
        if mode == 'remove':
            self.boundary_conditions['mass_injection'] = [
                bc for bc in self.boundary_conditions['mass_injection']
                if all(n not in nodes for n in bc.target_ids)
            ]
            return

        for node, val in zip(nodes, values):
            if mode == 'overwrite':
                self.boundary_conditions['mass_injection'] = [
                    bc for bc in self.boundary_conditions['mass_injection']
                    if node not in bc.target_ids
                ]
            elif mode == 'add':
                for bc in self.boundary_conditions['mass_injection']:
                    if node in bc.target_ids:
                        raise ValueError(
                            f"Mass-injection BC already exists at node {node}. "
                            "Use mode='overwrite' to replace it."
                        )

            # Build BC object
            if isinstance(val, Real):
                # Constant rate (kg/s)
                bc = ConstantBC([node], value=float(val))
            elif isinstance(val, tuple) and val[0] == 'box':
                _, rate, t0, t1, *rest = val
                r_before = rest[0] if len(rest) > 0 else 0.0
                r_after  = rest[1] if len(rest) > 1 else 0.0
                bc = BoxBC([node], rate, t0, t1, r_before, r_after)
            elif isinstance(val, tuple) and val[0] == 'timeseries':
                # ('timeseries', times, rates)
                _, times, rates = val[:3]
                bc = TimeSeriesBC(
                    [node],
                    times=np.asarray(times, dtype=float),
                    values=np.asarray(rates, dtype=float),
                    extrapolate=extrapolate
                )

            else:
                raise ValueError(f"Unrecognized mass-injection format at node {node}: {val}")

            self.boundary_conditions['mass_injection'].append(bc)


    def _advance_transport(self):
        """
        Preliminary AD transport implementation.
        """
        import numpy as np

        # Geometry & hydraulics of conduits
        i = self.n_indices1
        j = self.n_indices2
        L  = self.conduit_lengths
        Qe = self.Q_new.copy()
        Ae = self.a_mid
        ve = Qe / np.maximum(Ae, 1e-30)
        De = self.molecular_diffusivity + self.alpha_l * np.abs(ve)

        # CFL substepping
        dt_adv  = np.min(L / np.maximum(np.abs(ve), 1e-12))
        dt_diff = np.min(L**2 / np.maximum(2.0 * De, 1e-30))
        dt_lim  = self.transport_cfl * min(dt_adv, dt_diff)
        n_sub   = max(1, int(np.ceil(self.dt / np.maximum(dt_lim, 1e-12))))
        dt_s    = self.dt / n_sub

        # Update time-dependent BC values (probably not required, need to check...)
        self._cache_transport_bcs(self.current_time)

        # Dirichlet concentration (from water-depth concentration BCs)
        dirichlet_C_mask = self.bc_prescribed_C_mask    # (Np,)
        Cb_node          = self.bc_prescribed_C_vals    # (Np,)

        # Flags per conduit: does the endpoint have Dirichlet-BC?
        di = dirichlet_C_mask[i]
        dj = dirichlet_C_mask[j]

        # Node volumes
        Vn = self.V_node

        # Flux scheme (hardcoded for now, may allow user choice later on)
        use_sg = bool(getattr(self, "use_scharfetter", True))

        # Bernoulli for SG
        def _bernoulli(z):
            z = np.asarray(z, dtype=float)
            out = np.empty_like(z)
            small = np.abs(z) < 1e-6
            out[small]  = 1.0 - 0.5 * z[small]
            out[~small] = z[~small] / np.expm1(z[~small])
            return out

        for _ in range(n_sub):
            # Current concentrations from mass
            Cn = np.where(Vn > 0.0, self.M / Vn, 0.0)

            # Dirichlet: hard set concentration at nodes before computing fluxes
            if np.any(dirichlet_C_mask):
                self.M[dirichlet_C_mask] = Cb_node[dirichlet_C_mask] * Vn[dirichlet_C_mask]
                Cn[dirichlet_C_mask]     = Cb_node[dirichlet_C_mask]

            if use_sg:
                # ----------------------------
                # Scharfetter–Gummel flux
                # ----------------------------
                Ci = Cn[i].copy()
                Cj = Cn[j].copy()
                if np.any(di):
                    Ci[di] = Cb_node[i[di]]
                if np.any(dj):
                    Cj[dj] = Cb_node[j[dj]]

                P  = ve * L / np.maximum(De, 1e-30)
                Bp = _bernoulli(P)
                Bm = _bernoulli(-P)

                # Mass flux from i -> j (positive adds to j, removes from i)
                F = - (De * Ae / np.maximum(L, 1e-30)) * (Bp * Cj - Bm * Ci)

            else:
                # --------------------------------------------
                # Upwind advection + centered diffusion
                # --------------------------------------------
                upstream_from_i = (Qe > 0.0)

                # Advection: substitute Cb on the upstream Dirichlet end only
                Ci_adv = Cn[i].copy()
                Cj_adv = Cn[j].copy()
                if np.any(di):
                    mask = di &  upstream_from_i
                    Ci_adv[mask] = Cb_node[i[mask]]
                if np.any(dj):
                    mask = dj & (~upstream_from_i)
                    Cj_adv[mask] = Cb_node[j[mask]]

                Cup  = np.where(upstream_from_i, Ci_adv, Cj_adv)
                Fadv = Qe * Cup

                # Diffusion: full-length gradient (is this correct at nodes with degree 1 and BC?)
                G = (Cn[j] - Cn[i]) / np.maximum(L, 1e-30)
                Fdiff = - (Ae * De) * G

                F = Fadv + Fdiff

            # Assembly of nodal mass rates
            net_rate = np.zeros(self.network.Np, dtype=float)
            np.add.at(net_rate, i, -F)
            np.add.at(net_rate, j, +F)

            # Add mass injection from direct sources
            net_rate += self.bc_mass_injection_node

            # Currently NO Qin*Cin or other source terms (i.e. self.bc_mass_inflow_rate_node ...)
            # ...
            
            # Update mass (avoid negatives, not sure if this is good...)
            self.M = np.maximum(0.0, self.M + dt_s * net_rate)

        # Final concentration field
        self.C = self.M / np.maximum(Vn, 1e-20)
