import numpy as np
import openpnm as op

from openkarst.models import FlowSimulation
from openkarst.network_generation import compute_conduit_lengths


def _small_flow_simulation(tmp_path):
    geometry = op.network.Cubic(shape=[5, 1, 1], connectivity=6, spacing=1.0)
    geometry = compute_conduit_lengths(geometry)
    geometry["throat.epsilon"] = 0.03
    geometry["throat.diameters"] = 1.0

    return FlowSimulation(
        geometry,
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
