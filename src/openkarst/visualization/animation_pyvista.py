#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 08:47:02 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

import numpy as np
import pyvista as pv

def animate_network(cn_geometry, Q_history, y_history, t_history,
                    update_interval,
                    conduit_plotradius,
                    bar_plotradius,
                    node_plotsize,
                    depthscaling,
                    fig_width,
                    fig_height,
                    zoom_factor,
                    background_color,
                    isometric_view,
                    create_animation,
                    filename):
    """
    Animate the network using PyVista.
    
    This function visualizes the flow rates and water depths in a network over time using PyVista.
    It allows for both static and animated visualizations, as well as saving animations to an mpeg
    file. Saving the animation requires a properly set up path to FFMPEG in main()
    
    Args:
        cn_geometry (openpnm.network.GenericNetwork): The pore network object.
        Q_history (np.ndarray): Array of flow rates history.
        y_history (np.ndarray): Array of water depths history.
        t_history (np.ndarray): Array of time steps history.
        update_interval (int): Interval between updates.
        conduit_plotradius (float): Radius for the conduit plots.
        bar_plotradius (float): Radius for the bar plots.
        node_plotsize (float): Size of the node plots.
        depthscaling (float): Scaling factor for the depth.
        fig_width (int): Width of the figure.
        fig_height (int): Height of the figure.
        zoom_factor (float): Zoom factor for the camera.
        background_color (str): Background color for the plot.
        isometric_view (bool): If True, use isometric view.
        create_animation (bool): If True, create an animation.
        filename (str): Filename for the animation.
    
    Returns:
        None
    """
    
    # This right now gives xz plane view (z is the direction of water depth)
    # Allow user to show different plane or isometric 3D view!!!
    
    colorscheme = 'black' if background_color == 'white' else 'white'
    
    # Get min and max for colormap limits
    Q_min, Q_max = np.min(Q_history), np.max(Q_history)
    depth_min, depth_max = np.min(y_history), np.max(y_history)
    
    # Extract the coordinates and connectivity
    coords = cn_geometry['pore.coords']
    conns = cn_geometry['throat.conns']
    conduit_radii = cn_geometry['throat.diameters']/2.0
    
    n_indices1, n_indices2 = cn_geometry.conns.T
    max_depths = np.zeros(cn_geometry.Np)
    # Update max_depth based on the connected conduits
    # For nodes connected at n_indices1
    for i, node in enumerate(n_indices1):
        max_depths[node] = max(max_depths[node], cn_geometry['throat.diameters'][i])
    # For nodes connected at n_indices2
    for i, node in enumerate(n_indices2):
        max_depths[node] = max(max_depths[node], cn_geometry['throat.diameters'][i])
        
    # Calculate the bounding box of the coordinates
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

    # Calculate the center of the bounding box
    center = [(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2]
    
    window_width = fig_width
    window_height = fig_height

    # Ensure window height and width are a multiple of 16
    window_width = (window_width // 16) * 16
    window_height = (window_height // 16) * 16
    
    # Zoom factor determines whitespace, couldnt figure out yet how to always properly set this
    # Seems to change depending on window size as well...
    max_dim = max(x_max - x_min, z_max - z_min)
    camera_distance = max_dim * zoom_factor
    camera_position = [center[0], center[1] - camera_distance, center[2]]
    focal_point = center
    view_up = [0, 0, 1]
   

    # Create a PyVista plotter
    plotter = pv.Plotter(window_size=[window_width, window_height])
    plotter.background_color=background_color

    # Set the camera position
    if isometric_view == True:
        plotter.view_isometric()
    else:
        plotter.camera.position = camera_position
        plotter.camera.focal_point = focal_point
        plotter.camera.up = view_up
    
    
    # Create lines
    lines = []
    for start, end in conns:
        lines.append([2, start, end])  # Two points on each line

    # Flatten the lines list
    lines = np.hstack(lines)
    poly_lines = pv.PolyData()
    poly_lines.points = coords
    poly_lines.lines = lines
    
    # Create a separate PolyData for points
    poly_points = pv.PolyData(coords)

    # Add initial flow data to the cells (conduits)
    poly_lines.cell_data["flow"] = Q_history[0]

    # Add initial water depths data to the nodes
    poly_points.point_data["depths"] = y_history[0]

    # Create vertical bars for water depths
    def create_depth_bars(coords, depths, depthscaling):
        bars = []
        for i, (x, y, z) in enumerate(coords):
            depth = depths[i]
            bars.append([x, y, z])
            bars.append([x, y, z + depth * depthscaling])
        return np.array(bars)

    initial_bars = create_depth_bars(coords, y_history[0], depthscaling)
    bar_cells = np.arange(len(initial_bars)).reshape(-1, 2)

    poly_bars = pv.PolyData()
    poly_bars.points = initial_bars
    poly_bars.lines = np.hstack([[2] + cell.tolist() for cell in bar_cells])
    poly_bars["depths"] = np.repeat(y_history[0], 2)
    
    cmap_flows = "RdBu_r"
    cmap_depths = "RdBu_r"

    # Plot the network and bars
    conduits = poly_lines.tube(conduit_plotradius)
    plotter.add_mesh(conduits,
                     scalars="flow",
                     cmap=cmap_flows,
                     line_width=3,
                     clim=[Q_min, Q_max],
                     scalar_bar_args={'title': 'Volumetric flow rate', 'color': colorscheme})
    
    # Create lines for conduit ceiling
    def create_ceiling_lines(coords, conns, shift, scaling):
        ceiling_coords = coords.copy()
        ceiling_coords[:, 2] += shift * scaling
        ceiling_lines = []
        for start, end in conns:
            ceiling_lines.append([2, start, end])
        ceiling_lines = np.hstack(ceiling_lines)
        return ceiling_coords, ceiling_lines

    ceiling_coords, ceiling_lines = create_ceiling_lines(coords, conns, max_depths, depthscaling)
    poly_ceiling_lines = pv.PolyData()
    poly_ceiling_lines.points = ceiling_coords
    poly_ceiling_lines.lines = ceiling_lines

    shifted_conduits = poly_ceiling_lines.tube(conduit_plotradius)
    plotter.add_mesh(shifted_conduits,
                     color='blue',
                     line_width=3,
                     name="shifted_conduits")
    
    time_actor = plotter.add_text(f"Time: {t_history[0]:.2f}",
                                  position="upper_left", font_size=12, color=colorscheme)
    
    current_frame = 0 # To store frame for use in depthscaling slider
    bars_visible = True
    depth_bars_actor = None

    def update_frame(frame, depthscaling):
        nonlocal depth_bars_actor, time_actor
        
        poly_lines.cell_data["flow"][:] = Q_history[frame]
        poly_points.point_data["depths"][:] = y_history[frame]
        
        # Update conduits
        new_conduits = poly_lines.tube(conduit_plotradius)
        plotter.add_mesh(new_conduits,
                         scalars="flow",
                         cmap=cmap_flows,
                         line_width=3,
                         clim=[Q_min, Q_max],
                         name="conduits",
                         scalar_bar_args={'title': 'Volumetric flow rate', 'color': colorscheme})
        
        # # Update points
        # plotter.add_mesh(poly_points,
        #                  scalars="depths",
        #                  cmap=cmap_depths,
        #                  point_size=node_plotsize,
        #                  render_points_as_spheres=True,
        #                  clim=[depth_min, depth_max],
        #                  name="points",
        #                  scalar_bar_args={'title': 'Water depth (m)'})
        
        # Update shifted blue lines
        ceiling_coords, ceiling_lines = create_ceiling_lines(coords, conns, max_depths, depthscaling)
        poly_ceiling_lines.points = ceiling_coords
        new_ceiling_conduits = poly_ceiling_lines.tube(0.05)
        plotter.add_mesh(new_ceiling_conduits,
                         color='gray',
                         line_width=1,
                         name="shifted_conduits")
        
        if bars_visible:
            if depth_bars_actor is not None:
                plotter.remove_actor(depth_bars_actor)
            bars = create_depth_bars(coords, y_history[frame], depthscaling)
            poly_bars.points = bars
            poly_bars["depths"] = np.repeat(y_history[frame], 2)
            new_depth_bars = poly_bars.tube(bar_plotradius)
            depth_bars_actor = plotter.add_mesh(new_depth_bars,
                                                scalars="depths",
                                                cmap=cmap_depths,
                                                clim=[depth_min, depth_max],
                                                name="depth_bars",
                                                scalar_bar_args={'title': 'Water depth (m)', 'color': colorscheme})
        else:
            if depth_bars_actor is not None:
                plotter.remove_actor(depth_bars_actor)
                depth_bars_actor = None
                
        plotter.remove_actor(time_actor)
        time_actor = plotter.add_text(f"Time: {t_history[frame]:.2f}",
                                      position="upper_left", font_size=12, color=colorscheme)
                
        
       
    # Create an animation
    if create_animation:
        plotter.open_movie(filename)
    
        n_frames = len(t_history)
        for frame in range(n_frames):
            if frame % update_interval == 0:
                update_frame(frame, depthscaling)
            plotter.write_frame()  # Write the frame to the movie file
        
        plotter.camera_position = (camera_position, focal_point, view_up)
        plotter.enable_parallel_projection()
        #plotter.show(auto_close=True)
        plotter.view_isometric
        plotter.close()  # Close the movie file
    
    # Re-create the plotter for the interactive slider
    plotter = pv.Plotter(window_size=[window_width, window_height])
    plotter.background_color = background_color
    
    # Set the camera position
    if isometric_view == True:
        plotter.view_isometric()
    else:
        plotter.camera.position = camera_position
        plotter.camera.focal_point = focal_point
        plotter.camera.up = view_up

    
    # Plot the initial state again
    plotter.add_mesh(conduits,
                     scalars="flow",
                     cmap=cmap_flows,
                     line_width=3,
                     clim=[Q_min, Q_max],
                     scalar_bar_args={'title': 'Volumetric flow rate', 'color': colorscheme})
    
    # Plot the initial shifted blue lines
    plotter.add_mesh(shifted_conduits,
                     color='blue',
                     line_width=3,
                     name="shifted_conduits")
   
    # plotter.add_mesh(poly_points,
    #                  scalars="depths",
    #                  cmap=cmap_depths,
    #                  point_size=node_plotsize,
    #                  render_points_as_spheres=True,
    #                  clim=[depth_min, depth_max],
    #                  scalar_bar_args={'title': 'Water depth (m)'})
    
    time_actor = plotter.add_text(f"Time: {t_history[0]:.2f}",
                                  position="upper_left", font_size=12, color=colorscheme)
        
    # Add the slider for depthscaling
    def depthscaling_slider_callback(value):
        nonlocal depthscaling, current_frame
        depthscaling = value
        update_frame(current_frame, depthscaling)
        plotter.render()
    
    plotter.add_slider_widget(callback=depthscaling_slider_callback,
                              rng=[1, 100], value=depthscaling,
                              title='Depth Scaling', title_color=colorscheme, color=colorscheme,
                              interaction_event='always', style="modern", pointa=(.6, .9), pointb=(.9, .9))
    
    # Update the current frame when the time slider is moved
    def time_slider_callback(value):
        global current_frame
        current_frame = int(value)
        update_frame(current_frame, depthscaling)
        plotter.render()
    
    plotter.add_slider_widget(callback=time_slider_callback,
                              rng=[0, len(t_history) - 1], value=0,
                              title='Time step', title_color=colorscheme, color=colorscheme,
                              interaction_event='always', style="modern", pointa=(.1, .9), pointb=(.6, .9))
    
    def toggle_bars(value):
        nonlocal bars_visible
        bars_visible = value
        update_frame(current_frame, depthscaling)
        plotter.render()
        
    plotter.add_checkbox_button_widget(callback=toggle_bars, value=True, size=30, border_size=2, color_off='grey', color_on='green', position=(20, 20))
    plotter.add_text("Toggle Depth Bars", position=(60, 20), color=colorscheme, font_size=12)
    
    plotter.show()
    
