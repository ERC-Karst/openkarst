import numpy as np
import pytest

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


def test_load_cave_data_loads_builtin_seefeldhoehle():
    geometry = load_cave_data(cave="seefeldhoehle")

    assert geometry.Np == 471
    assert geometry.Nt == 622
    assert np.all(geometry["throat.lengths"] > 0)
    assert np.all(geometry["throat.diameters"] > 0)


def test_load_cave_data_rejects_unknown_builtin_cave():
    with pytest.raises(ValueError, match="Unknown cave"):
        load_cave_data(cave="unknown")


def test_load_cave_data_rejects_mixed_builtin_and_file_args(tmp_path):
    with pytest.raises(ValueError, match="Pass either"):
        load_cave_data(nodes_file=tmp_path / "nodes.csv", cave="seefeldhoehle")
