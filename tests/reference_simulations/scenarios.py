"""Small deterministic hydraulic reference simulations.

These scenarios are intentionally compact and avoid transport, which is not
part of the released reference behavior yet.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "openkarst-matplotlib-cache"),
)

import numpy as np
import openpnm as op

from openkarst.models import FlowSimulation
from openkarst.network_generation import compute_conduit_lengths


REFERENCE_OUTPUT_KEYS = (
    "time",
    "flowrates",
    "velocities",
    "water_depths",
    "reynolds_numbers",
    "y_l2_norms",
    "Q_l2_norms",
    "picard_iterations",
    "picard_iterations_total",
)

RESERVOIR_OUTPUT_KEYS = (
    "reservoir_nodes",
    "reservoir_water_depths",
    "reservoir_heads",
    "reservoir_storage",
    "reservoir_exchange",
    "reservoir_recharge",
)


@dataclass(frozen=True)
class ReferenceScenario:
    """Definition of a deterministic reference simulation."""

    name: str
    description: str
    runner: Callable[[Path], dict[str, np.ndarray]]


def reference_network():
    """Return the small conduit network shared by the reference simulations."""

    network = op.network.Cubic(shape=[5, 1, 1], connectivity=6, spacing=1.0)
    network = compute_conduit_lengths(network)
    network["throat.epsilon"] = 0.03
    network["throat.diameters"] = 1.0
    return network


def reference_flow(
    log_dir: Path,
    *,
    t_max: float = 0.6,
    max_iterations: int = 40,
):
    """Create a deterministic non-transport flow simulation."""

    return FlowSimulation(
        reference_network(),
        solver_settings={
            "parallelization": False,
            "max_iterations": max_iterations,
            "picard_depth_tol": 1e-9,
        },
        simulation_settings={
            "adaptive_timesteps": False,
            "dt_init": 0.1,
            "dt_max": 0.1,
            "t_max": t_max,
            "print_info_interval": 100000,
            "enable_transport": False,
        },
        logging_settings={
            "base_dir": str(log_dir),
            "log_file": "reference_simulation.log",
        },
    )


def requested_outputs(*extra_keys: str):
    """Return the standard output request for reference simulations."""

    outputs = {"output_interval": 0.1}
    outputs.update({key: True for key in REFERENCE_OUTPUT_KEYS})
    outputs.update({key: True for key in extra_keys})
    return outputs


def run_linear_closed_conduit(log_dir: Path):
    flow = reference_flow(log_dir)
    network = flow.network
    flow.set_initial_conditions(
        initial_Q=np.zeros(network.Nt, dtype=float),
        initial_y=np.full(network.Np, 0.01, dtype=float),
    )
    flow.set_inflow_BC(nodes=0, values=0.001)
    flow.set_waterdepth_BC(nodes=4, values=0.01)
    return flow.run_simulation(requested_outputs())


def run_timeseries_inflow(log_dir: Path):
    flow = reference_flow(log_dir)
    network = flow.network
    flow.set_initial_conditions(
        initial_Q=np.zeros(network.Nt, dtype=float),
        initial_y=np.full(network.Np, 0.012, dtype=float),
    )
    flow.set_inflow_BC(
        nodes=0,
        values=(
            "timeseries",
            np.array([0.0, 0.2, 0.4, 0.6]),
            np.array([0.0005, 0.0015, 0.0008, 0.0012]),
        ),
    )
    flow.set_waterdepth_BC(nodes=4, values=0.012)
    return flow.run_simulation(requested_outputs())


def run_spring_outflow(log_dir: Path):
    flow = reference_flow(log_dir)
    network = flow.network
    flow.set_initial_conditions(
        initial_Q=np.zeros(network.Nt, dtype=float),
        initial_y=np.full(network.Np, 0.08, dtype=float),
    )
    flow.set_inflow_BC(nodes=0, values=0.0002)
    flow.set_spring_BC(
        nodes=4,
        outlet_elevation=flow.Z[4] + 0.03,
        coefficient=0.001,
        exponent=1.0,
    )
    return flow.run_simulation(requested_outputs())


def run_reservoir_exchange(log_dir: Path):
    flow = reference_flow(log_dir)
    network = flow.network
    flow.set_initial_conditions(
        initial_Q=np.zeros(network.Nt, dtype=float),
        initial_y=np.full(network.Np, 0.01, dtype=float),
    )
    flow.add_reservoir(
        node=0,
        area=1000.0,
        specific_yield=0.1,
        initial_water_depth=2.0,
        conductance=1e-4,
        recharge=0.001,
    )
    flow.set_waterdepth_BC(nodes=4, values=0.01)
    return flow.run_simulation(requested_outputs(*RESERVOIR_OUTPUT_KEYS))


REFERENCE_SCENARIOS = (
    ReferenceScenario(
        name="linear_closed_conduit_v1",
        description="Five-node closed conduit with fixed inflow and downstream water depth.",
        runner=run_linear_closed_conduit,
    ),
    ReferenceScenario(
        name="timeseries_inflow_v1",
        description="Five-node closed conduit with time-series upstream inflow.",
        runner=run_timeseries_inflow,
    ),
    ReferenceScenario(
        name="spring_outflow_v1",
        description="Five-node closed conduit with an upstream inflow and downstream spring outflow.",
        runner=run_spring_outflow,
    ),
    ReferenceScenario(
        name="reservoir_exchange_v1",
        description="Five-node closed conduit coupled to one stateful upstream reservoir.",
        runner=run_reservoir_exchange,
    ),
)


SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in REFERENCE_SCENARIOS}
