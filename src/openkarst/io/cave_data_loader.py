#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:22:54 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es
"""

from importlib import resources
from os import PathLike

import networkx as nx
import openpnm as op
import numpy as np


_BUILTIN_CAVES = {
    "seefeldhoehle": {
        "nodes": "seefeldhoehle_nodes.csv",
        "edges": "seefeldhoehle_edges.csv",
        "diameters": "seefeldhoehle_diameters.csv",
    },
}


def _available_caves() -> str:
    return ", ".join(sorted(_BUILTIN_CAVES))


def _resolve_builtin_cave_files(cave: str):
    cave_key = cave.lower()
    try:
        cave_files = _BUILTIN_CAVES[cave_key]
    except KeyError:
        raise ValueError(
            f"Unknown cave '{cave}'. Available caves: {_available_caves()}."
        ) from None

    data_dir = resources.files("openkarst").joinpath("cave_data", cave_key)
    return (
        data_dir.joinpath(cave_files["nodes"]),
        data_dir.joinpath(cave_files["edges"]),
        data_dir.joinpath(cave_files["diameters"]),
    )


def _open_text(file):
    if hasattr(file, "open"):
        return file.open("r", encoding="utf-8")
    return open(file, "r", encoding="utf-8")


def load_cave_data(
    nodes_file: str | PathLike[str] | None = None,
    edges_file: str | PathLike[str] | None = None,
    diameters_file: str | PathLike[str] | None = None,
    *,
    cave: str | None = None,
):
    """
    Load cave network data from CSV files and create an OpenPNM geometry object.

    This function reads node coordinates, edge connections, and diameters from
    CSV files and constructs a NetworkX graph with the data. It then
    converts the graph into an OpenPNM geometry object with assigned conduit lengths
    and diameters.

    Args:
        nodes_file (str | os.PathLike | None): Path to the CSV file containing
            node coordinates. Required unless ``cave`` is provided.
        edges_file (str | os.PathLike | None): Path to the CSV file containing
            edge connections. Required unless ``cave`` is provided.
        diameters_file (str | os.PathLike | None): Path to the CSV file
            containing node diameters. Required unless ``cave`` is provided.
        cave (str | None): Name of a bundled cave dataset to load. Available
            caves: seefeldhoehle.

    Returns:
        openpnm.network.GenericNetwork: An OpenPNM geometry object representing
            the network with assigned conduit lengths and diameters.
    """
    if cave is not None:
        if any(file is not None for file in (nodes_file, edges_file, diameters_file)):
            raise ValueError(
                "Pass either cave='seefeldhoehle' or nodes_file, edges_file, "
                "and diameters_file, not both."
            )
        nodes_file, edges_file, diameters_file = _resolve_builtin_cave_files(cave)
    elif any(file is None for file in (nodes_file, edges_file, diameters_file)):
        raise ValueError(
            "Pass either cave='seefeldhoehle' or nodes_file, edges_file, "
            "and diameters_file."
        )

    G = nx.Graph()
    node_diameters = {}

    # Load nodes and their coordinates from the file, skipping the header
    with _open_text(nodes_file) as file:
        next(file)  # Skip the header line
        for line in file:
            node_id, x, y, z = line.strip().split(';')
            G.add_node(int(node_id), coords=[float(x), float(y), float(z)])

    # Load edges from the file, skipping the header
    with _open_text(edges_file) as file:
        next(file)  # Skip the header line
        for line in file:
            node_a, node_b = map(int, line.strip().split(';'))
            G.add_edge(node_a, node_b)

    # Load diameters from the file, skipping the header
    with _open_text(diameters_file) as file:
        next(file)  # Skip the header line
        for line in file:
            node_id, cswidth, csheight = line.strip().split(';')
            average_diameter = (float(cswidth) + float(csheight)) / 2
            node_diameters[int(node_id)] = average_diameter

    #Check if the data follows openkarst requirements
    #Verify that node start at zero and is consecutive/ does not contain gaps (0,1,2,3...)
    if min(G.nodes()) != 0 or max(G.nodes()) + 1 != len(G.nodes()) :
        print("Node numbering is not correct for openKARST.\n Please process the data as in the following example: https://github.com/ERC-Karst/KNdata-public/blob/main/notebooks/3.Load_and_prep_for_openkarst.ipynb")

    #Verify is every node has a diameter
    if len(G.nodes) > len(node_diameters):
        print("Some diameters are missing. \n Please process the data as in the following example: https://github.com/ERC-Karst/KNdata-public/blob/main/notebooks/3.Load_and_prep_for_openkarst.ipynb")

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
