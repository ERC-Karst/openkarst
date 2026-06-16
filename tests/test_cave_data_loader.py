import numpy as np

from openkarst.io import load_cave_data


def test_load_cave_data_returns_openpnm_geometry(tmp_path):
    nodes_file = tmp_path / "nodes.csv"
    edges_file = tmp_path / "edges.csv"
    diameters_file = tmp_path / "diameters.csv"

    nodes_file.write_text("id;x;y;z\n0;0;0;0\n1;3;4;0\n2;3;4;3\n")
    edges_file.write_text("from_id;to_id\n0;1\n1;2\n")
    diameters_file.write_text("id;cswidth;csheight\n0;1;1\n1;3;3\n2;4;4\n")

    geometry = load_cave_data(nodes_file, edges_file, diameters_file)

    assert geometry.Np == 3
    assert geometry.Nt == 2
    np.testing.assert_allclose(np.sort(geometry["throat.lengths"]), [3.0, 5.0])
    np.testing.assert_allclose(np.sort(geometry["throat.diameters"]), [2.0, 3.5])
