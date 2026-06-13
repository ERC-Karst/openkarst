import numpy as np
import openpnm as op

from openkarst.models import FlowSimulation
from openkarst.network_generation import compute_conduit_lengths


def _small_network():
    geometry = op.network.Cubic(shape=[5, 1, 1], connectivity=6, spacing=1.0)
    geometry = compute_conduit_lengths(geometry)
    geometry["throat.epsilon"] = 0.03
    geometry["throat.diameters"] = 1.0
    return geometry


def _small_flow_simulation(
    tmp_path,
    geometry_backend="circular_analytical",
    table_points=None,
):
    geometry_settings = {"backend": geometry_backend}
    if table_points is not None:
        geometry_settings["table_points"] = table_points

    return FlowSimulation(
        _small_network(),
        geometry_settings=geometry_settings,
        solver_settings={
            "max_iterations": 20,
            "picard_depth_tol": 1e-7,
        },
        simulation_settings={
            "adaptive_timesteps": False,
            "dt_init": 0.1,
            "dt_max": 0.1,
            "t_max": 0.2,
            "print_info_interval": 1000,
        },
        logging_settings={
            "base_dir": str(tmp_path),
            "log_file": "simulation.log",
        },
    )


def _small_flow_simulation_with_default_geometry_settings(tmp_path):
    return FlowSimulation(
        _small_network(),
        solver_settings={
            "max_iterations": 20,
            "picard_depth_tol": 1e-7,
        },
        simulation_settings={
            "adaptive_timesteps": False,
            "dt_init": 0.1,
            "dt_max": 0.1,
            "t_max": 0.2,
            "print_info_interval": 1000,
        },
        logging_settings={
            "base_dir": str(tmp_path),
            "log_file": "simulation.log",
        },
    )


def test_flow_simulation_runs_on_small_linear_network(tmp_path):
    flow = _small_flow_simulation(tmp_path)
    geometry = flow.network
    flow.set_initial_conditions(
        initial_Q=np.zeros(geometry.Nt, dtype=float),
        initial_y=np.full(geometry.Np, 0.01, dtype=float),
    )
    flow.set_inflow_BC(nodes=0, values=0.001)
    flow.set_waterdepth_BC(nodes=4, values=0.01)

    results = flow.run_simulation(
        desired_outputs={
            "output_interval": 0.1,
            "time": True,
            "flowrates": True,
            "water_depths": True,
        }
    )

    assert set(results) == {"time", "flowrates", "water_depths"}
    assert results["time"].shape[0] >= 1
    assert results["flowrates"].shape[1] == geometry.Nt
    assert results["water_depths"].shape[1] == geometry.Np
    assert np.isfinite(results["flowrates"]).all()
    assert np.isfinite(results["water_depths"]).all()


def test_flow_simulation_defaults_to_analytical_geometry_backend(tmp_path):
    flow = _small_flow_simulation_with_default_geometry_settings(tmp_path)

    assert flow.geometry_backend == "circular_analytical"
    assert flow.geometry_table_points == 1000


def test_flow_simulation_applies_geometry_table_points(tmp_path):
    flow = _small_flow_simulation(
        tmp_path,
        geometry_backend="circular_tabulated",
        table_points=1234,
    )

    assert flow.geometry_table_points == 1234
    assert flow.cross_section_geometry.n_points == 1234


def test_flow_simulation_tabulated_circular_matches_analytical(tmp_path):
    def run_backend(backend):
        flow = _small_flow_simulation(tmp_path, geometry_backend=backend)
        geometry = flow.network
        flow.set_initial_conditions(
            initial_Q=np.zeros(geometry.Nt, dtype=float),
            initial_y=np.full(geometry.Np, 0.01, dtype=float),
        )
        flow.set_inflow_BC(nodes=0, values=0.001)
        flow.set_waterdepth_BC(nodes=4, values=0.01)
        return flow.run_simulation(
            desired_outputs={
                "output_interval": 0.1,
                "time": True,
                "flowrates": True,
                "water_depths": True,
            }
        )

    analytical = run_backend("circular_analytical")
    tabulated = run_backend("circular_tabulated")

    np.testing.assert_allclose(tabulated["time"], analytical["time"])
    np.testing.assert_allclose(
        tabulated["flowrates"],
        analytical["flowrates"],
        rtol=2e-3,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        tabulated["water_depths"],
        analytical["water_depths"],
        rtol=2e-3,
        atol=1e-9,
    )


def test_scalar_flux_inflow_bc_preserves_flux_type(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    flow.set_inflow_BC(nodes=0, values=1e-5, inflow_type="flux")

    bc = flow.boundary_conditions["inflow"][0]
    assert bc.bc_type == "flux"
    assert bc.get_value(0.0) == 1e-5


def test_box_flux_inflow_bc_preserves_flux_type_for_all_nodes(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    flow.set_inflow_BC(
        nodes=[0, 1],
        values=("box", 1e-5, 5.0, 125.0),
        inflow_type="flux",
    )

    bcs = flow.boundary_conditions["inflow"]
    assert [bc.target_ids for bc in bcs] == [[0], [1]]
    assert all(bc.bc_type == "flux" for bc in bcs)
    assert all(bc.get_value(6.0) == 1e-5 for bc in bcs)
