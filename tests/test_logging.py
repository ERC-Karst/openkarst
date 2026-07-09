import numpy as np
import openpnm as op

from openkarst.models import FlowSimulation
from openkarst.network_generation import compute_conduit_lengths
from openkarst.utils.logging_config import setup_logging


def test_flow_simulation_writes_quiet_normal_log(tmp_path):
    network = op.network.Cubic(shape=[3, 1, 1], connectivity=6, spacing=1.0)
    compute_conduit_lengths(network)
    network["throat.diameters"] = 1.0
    network["throat.epsilon"] = 0.03

    flow = FlowSimulation(
        network,
        simulation_settings={
            "adaptive_timesteps": False,
            "dt_init": 0.1,
            "dt_max": 0.1,
            "t_max": 0.2,
            "print_info_interval": 1,
        },
        logging_settings={
            "base_dir": str(tmp_path),
            "log_file": "simulation.log",
        },
    )
    flow.set_initial_conditions(np.zeros(network.Nt), np.full(network.Np, 0.01))
    flow.set_inflow_BC(nodes=[0], values=0.01)
    flow.set_waterdepth_BC(nodes=[2], values=0.01)
    flow.set_observation_points(
        nodes=[0, 2],
        variables=["water_depth", "connected_net_flowrate"],
        interval=0.1,
    )

    flow.run_simulation({
        "output_interval": 0.1,
        "time": True,
        "water_depths": True,
    })

    log_text = (tmp_path / "logs" / "simulation.log").read_text()

    assert " | INFO    | FlowSimulation | " in log_text
    assert "Settings: physical=PhysicalProperties" not in log_text
    assert "  physical.water_density=1000.0" in log_text
    assert "  simulation.dt_init=0.1" in log_text
    assert "Network summary:" in log_text
    assert (
        "Boundary condition configured: type=inflow, mode=add, nodes=[0], "
        "value_type=constant, value=0.01"
    ) in log_text
    assert "Observation recorder added: name=observation_0, nodes=[0, 2], interval=0.1" in log_text
    assert "  variable=water_depth" in log_text
    assert "  variable=connected_net_flowrate" in log_text
    assert "Requested outputs:" in log_text
    assert "  output_interval=0.1" in log_text
    assert "  time=True" in log_text
    assert "outputs=[" not in log_text
    assert "Run finished: stop_reason=t_max" in log_text
    assert "Convergence: failures=" in log_text
    assert "Results stored: count=2" in log_text
    assert "Results stored: keys=" not in log_text
    assert log_text.endswith("\n\n")
    assert "Timestep =" not in log_text
    assert "Picard iterations =" not in log_text


def test_setup_logging_uses_timestamped_file_when_log_file_is_omitted(tmp_path):
    logger = setup_logging({"base_dir": str(tmp_path)})
    logger.info("first run")
    first_handlers = list(logger.handlers)
    for handler in first_handlers:
        handler.flush()

    logger = setup_logging({"base_dir": str(tmp_path)})
    logger.info("second run")
    for handler in logger.handlers:
        handler.flush()

    log_files = sorted((tmp_path / "logs").glob("simulation_*.log"))
    assert len(log_files) == 2
    assert "first run" in log_files[0].read_text()
    assert "second run" in log_files[1].read_text()


def test_setup_logging_appends_when_log_file_is_explicit(tmp_path):
    logger = setup_logging({
        "base_dir": str(tmp_path),
        "log_file": "simulation.log",
    })
    logger.info("first run")
    for handler in logger.handlers:
        handler.flush()

    logger = setup_logging({
        "base_dir": str(tmp_path),
        "log_file": "simulation.log",
    })
    logger.info("second run")
    for handler in logger.handlers:
        handler.flush()

    log_files = sorted((tmp_path / "logs").glob("*.log"))
    assert [path.name for path in log_files] == ["simulation.log"]
    log_text = log_files[0].read_text()
    assert "first run" in log_text
    assert "second run" in log_text
