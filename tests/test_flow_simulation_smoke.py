import numpy as np
import openpnm as op

from openkarst.models import FlowSimulation
from openkarst.network_generation import compute_conduit_lengths


def test_flow_simulation_runs_on_small_linear_network(tmp_path):
    geometry = op.network.Cubic(shape=[5, 1, 1], connectivity=6, spacing=1.0)
    geometry = compute_conduit_lengths(geometry)
    geometry["throat.epsilon"] = 0.03
    geometry["throat.diameters"] = 1.0

    flow = FlowSimulation(
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
