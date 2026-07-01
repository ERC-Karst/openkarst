import numpy as np
import openpnm as op
import pytest

from openkarst.models import FlowSimulation, UnconfinedReservoir
from openkarst.network_generation import compute_conduit_lengths


def _small_network():
    geometry = op.network.Cubic(shape=[5, 1, 1], connectivity=6, spacing=1.0)
    geometry = compute_conduit_lengths(geometry)
    geometry["throat.epsilon"] = 0.03
    geometry["throat.diameters"] = 1.0
    return geometry


def _write_normalized_circular_table(path, n_points=2001):
    eta = np.linspace(0.0, 1.0, n_points)
    width_norm = 2.0 * np.sqrt(np.maximum(eta - eta**2, 0.0))

    width_norm[0] = 0.0
    width_norm[-1] = 0.0

    table = np.column_stack((eta, width_norm))
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="eta,width_norm",
        comments="",
    )


def _small_flow_simulation(
    tmp_path,
    geometry_backend="circular_analytical",
    table_points=None,
    table_file=None,
    scale_by_diameter=True,
    interpolation_method=None,
):
    geometry_settings = {"backend": geometry_backend}
    if table_points is not None:
        geometry_settings["table_points"] = table_points
    if interpolation_method is not None:
        geometry_settings["interpolation_method"] = interpolation_method
    if table_file is not None:
        geometry_settings["table_file"] = str(table_file)
        geometry_settings["scale_by_diameter"] = scale_by_diameter

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
            "picard_iterations": True,
            "picard_iterations_total": True,
        }
    )

    assert set(results) == {
        "time",
        "flowrates",
        "water_depths",
        "picard_iterations",
        "picard_iterations_total",
    }
    assert results["time"].shape[0] >= 1
    assert results["flowrates"].shape[1] == geometry.Nt
    assert results["water_depths"].shape[1] == geometry.Np
    assert results["picard_iterations"].shape == results["time"].shape
    assert results["picard_iterations_total"].shape == results["time"].shape
    assert np.all(np.diff(results["picard_iterations_total"]) >= 0)
    assert np.isfinite(results["flowrates"]).all()
    assert np.isfinite(results["water_depths"]).all()


def test_flow_simulation_defaults_to_analytical_geometry_backend(tmp_path):
    flow = _small_flow_simulation_with_default_geometry_settings(tmp_path)

    assert flow.settings.geometry.backend == "circular_analytical"
    assert flow.settings.geometry.table_points == 100
    assert flow.settings.geometry.interpolation_method == "pchip"
    assert not hasattr(flow, "geometry_backend")


def test_flow_simulation_applies_geometry_table_points(tmp_path):
    flow = _small_flow_simulation(
        tmp_path,
        geometry_backend="circular_tabulated",
        table_points=1234,
    )

    assert flow.settings.geometry.table_points == 1234
    assert flow.cross_section_geometry.n_points == 1234


def test_flow_simulation_applies_geometry_interpolation_method(tmp_path):
    flow = _small_flow_simulation(
        tmp_path,
        geometry_backend="circular_tabulated",
        interpolation_method="linear",
    )

    assert flow.settings.geometry.interpolation_method == "linear"
    assert flow.cross_section_geometry.interpolation_method == "linear"


def test_flow_simulation_tabulated_circular_matches_analytical(tmp_path):
    def run_backend(backend):
        table_points = 1001 if backend == "circular_tabulated" else None
        flow = _small_flow_simulation(
            tmp_path,
            geometry_backend=backend,
            table_points=table_points,
        )
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


def test_flow_simulation_user_tabulated_csv_matches_analytical(tmp_path):
    table_file = tmp_path / "normalized_circle.csv"
    _write_normalized_circular_table(table_file, n_points=50001)

    def run_backend(backend, table_file=None):
        flow = _small_flow_simulation(
            tmp_path,
            geometry_backend=backend,
            table_file=table_file,
        )
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
    tabulated = run_backend("tabulated", table_file=table_file)

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


def test_spring_bc_stores_power_law_and_computes_outflow(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    flow.set_spring_BC(
        nodes=[3, 4],
        outlet_elevation=[flow.Z[3] + 0.2, flow.Z[4] + 0.5],
        coefficient=0.01,
        exponent=1.0,
    )

    bcs = flow.boundary_conditions["spring"]
    assert [bc.target_ids for bc in bcs] == [[3], [4]]

    flow.y_prev_i[3] = 0.5
    flow.y_prev_i[4] = 0.3
    flow._compute_spring_outflows()

    expected = np.zeros(flow.network.Np, dtype=float)
    expected[3] = 0.003
    np.testing.assert_allclose(flow.bc_spring_outflow_node, expected)


def test_spring_bc_overwrite_and_remove(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    flow.set_spring_BC(
        nodes=0,
        outlet_elevation=flow.Z[0],
        coefficient=0.001,
    )

    with pytest.raises(ValueError, match="Spring BC already exists"):
        flow.set_spring_BC(
            nodes=0,
            outlet_elevation=flow.Z[0],
            coefficient=0.002,
        )

    flow.set_spring_BC(
        nodes=0,
        outlet_elevation=flow.Z[0],
        coefficient=0.002,
        mode="overwrite",
    )
    bcs = flow.boundary_conditions["spring"]
    assert len(bcs) == 1
    assert bcs[0].target_ids == [0]
    assert bcs[0].coefficient == 0.002

    flow.set_spring_BC(
        nodes=0,
        outlet_elevation=flow.Z[0],
        mode="remove",
    )
    assert flow.boundary_conditions["spring"] == []


def test_spring_outflow_is_subtracted_from_node_balance(tmp_path):
    flow = _small_flow_simulation(tmp_path)
    flow.dt = 1.0
    flow.Q_new.fill(0.0)
    flow.dQ_old_t.fill(0.0)
    flow.y_old_t.fill(1.0)
    flow.y_prev_i.fill(1.0)
    flow.set_spring_BC(
        nodes=2,
        outlet_elevation=flow.Z[2] + 0.5,
        coefficient=0.02,
    )

    flow._compute_water_depths(np.ones(flow.network.Np, dtype=float))

    assert flow.bc_spring_outflow_node[2] == pytest.approx(0.01)
    assert flow.dQ_new[2] == pytest.approx(-0.01)


def test_spring_and_waterdepth_same_node_conflict(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    flow.set_spring_BC(
        nodes=0,
        outlet_elevation=flow.Z[0],
        coefficient=0.001,
    )
    flow.set_waterdepth_BC(nodes=0, values=0.01)

    with pytest.raises(ValueError, match="spring and prescribed water depth"):
        flow._check_bc_conflicts()


def test_reservoir_bc_stores_fixed_exchange_and_caches(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    flow.set_reservoir_BC(nodes=[0, 1], fixed_exchange_rate=0.001)

    bcs = flow.boundary_conditions["reservoir"]
    assert [bc.target_ids for bc in bcs] == [[0], [1]]
    assert all(bc.bc_type == "volumetric" for bc in bcs)
    assert all(bc.get_value(0.0) == 0.001 for bc in bcs)

    flow.current_time = 0.0
    flow._cache_hydraulic_bcs()
    expected = np.zeros(flow.network.Np, dtype=float)
    expected[[0, 1]] = 0.001
    np.testing.assert_allclose(flow.bc_reservoir_exchange_node, expected)


def test_add_reservoir_returns_and_registers_stateful_object(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    reservoir = flow.add_reservoir(
        node=0,
        area=1000.0,
        specific_yield=0.1,
        initial_water_depth=2.0,
        conductance=1e-4,
        recharge=0.001,
    )

    assert isinstance(reservoir, UnconfinedReservoir)
    assert flow.reservoirs == [reservoir]
    assert reservoir.node == 0
    assert reservoir.base_elevation == flow.Z[0]
    assert reservoir.reservoir_water_depth == 2.0
    assert reservoir.get_hydraulic_head() == flow.Z[0] + 2.0
    assert reservoir.get_storage() == 200.0
    assert reservoir._get_recharge_value(12.0) == 0.001
    assert reservoir.last_recharge_rate == 0.001
    assert not hasattr(reservoir, "exchange_history")
    assert not hasattr(reservoir, "water_depth_history")


def test_add_reservoir_accepts_timeseries_recharge(tmp_path):
    flow = _small_flow_simulation(tmp_path)
    times = np.array([0.0, 10.0, 20.0])
    recharge_rates = np.array([0.0, 0.001, 0.003])

    reservoir = flow.add_reservoir(
        node=0,
        area=1000.0,
        specific_yield=0.1,
        initial_water_depth=2.0,
        conductance=1e-4,
        recharge=("timeseries", times, recharge_rates),
        recharge_extrapolate="zero",
    )

    np.testing.assert_allclose(reservoir.time, times)
    np.testing.assert_allclose(reservoir.recharge, recharge_rates)
    assert reservoir.recharge_extrapolate == "zero"
    assert reservoir._get_recharge_value(-1.0) == 0.0
    assert reservoir._get_recharge_value(5.0) == 0.0005
    assert reservoir._get_recharge_value(25.0) == 0.0


def test_add_reservoir_rejects_invalid_recharge_format(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    with pytest.raises(ValueError, match="Reservoir recharge"):
        flow.add_reservoir(
            node=0,
            area=1000.0,
            specific_yield=0.1,
            initial_water_depth=2.0,
            conductance=1e-4,
            recharge=("box", 0.001, 0.0, 10.0),
        )


def test_add_reservoir_requires_one_unique_valid_node(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    with pytest.raises(ValueError, match="exactly one node"):
        flow.add_reservoir(
            node=[0, 1],
            area=1000.0,
            specific_yield=0.1,
            initial_water_depth=2.0,
            conductance=1e-4,
        )

    flow.add_reservoir(
        node=0,
        area=1000.0,
        specific_yield=0.1,
        initial_water_depth=2.0,
        conductance=1e-4,
    )
    with pytest.raises(ValueError, match="already exists"):
        flow.add_reservoir(
            node=0,
            area=1000.0,
            specific_yield=0.1,
            initial_water_depth=2.0,
            conductance=1e-4,
        )


def test_stateful_reservoir_exchange_is_cached_and_advanced(tmp_path, monkeypatch):
    flow = _small_flow_simulation(tmp_path)
    reservoir = flow.add_reservoir(
        node=1,
        area=1000.0,
        specific_yield=0.1,
        initial_water_depth=2.0,
        conductance=1e-4,
    )
    flow.dt = 0.25
    flow.current_time = 0.0
    flow.y_old_t[1] = 0.75

    exchange_calls = []
    advance_calls = []

    def compute_exchange(connected_node_water_depth, dt):
        exchange_calls.append((connected_node_water_depth, dt))
        return 0.003

    def advance(exchange_rate, dt, t_start):
        advance_calls.append((exchange_rate, dt, t_start))

    monkeypatch.setattr(reservoir, "compute_exchange", compute_exchange)
    monkeypatch.setattr(reservoir, "advance", advance)

    flow._cache_hydraulic_bcs()
    flow._advance_reservoirs()

    assert exchange_calls == [(0.75, 0.25)]
    assert advance_calls == [(0.003, 0.25, 0.0)]
    assert reservoir.last_exchange_rate == 0.003
    assert flow.bc_reservoir_exchange_node[1] == 0.003


def test_stateful_reservoir_conflicts_with_hydraulic_bc(tmp_path):
    flow = _small_flow_simulation(tmp_path)
    flow.add_reservoir(
        node=0,
        area=1000.0,
        specific_yield=0.1,
        initial_water_depth=2.0,
        conductance=1e-4,
    )
    flow.set_inflow_BC(nodes=0, values=0.001)

    with pytest.raises(ValueError, match="reservoir and inflow"):
        flow._check_bc_conflicts()


def test_reservoir_observation_variables_require_reservoir_nodes(tmp_path):
    flow = _small_flow_simulation(tmp_path)
    flow.add_reservoir(
        node=0,
        area=1000.0,
        specific_yield=0.1,
        initial_water_depth=2.0,
        conductance=1e-4,
    )

    flow.set_observation_points(
        nodes=0,
        variables=["reservoir_storage", "reservoir_exchange"],
    )

    with pytest.raises(ValueError, match="without reservoirs"):
        flow.set_observation_points(
            nodes=[0, 1],
            variables=["reservoir_storage"],
        )


def test_multiple_observation_recorders_return_one_combined_dataframe(tmp_path):
    flow = _small_flow_simulation(tmp_path)
    flow.add_reservoir(
        node=0,
        area=1000.0,
        specific_yield=0.1,
        initial_water_depth=2.0,
        conductance=1e-4,
    )

    flow.set_observation_points(
        nodes=[0, 1],
        variables=["water_depth"],
        name="nodes",
    )
    flow.set_observation_points(
        nodes=0,
        variables=["reservoir_storage"],
        name="reservoir",
    )

    flow.observation_recorders[0].records = [
        {"time": 0.0, "node": 0, "water_depth": 0.1},
        {"time": 0.0, "node": 1, "water_depth": 0.2},
    ]
    flow.observation_recorders[1].records = [
        {"time": 0.0, "node": 0, "reservoir_storage": 200.0},
    ]

    obs_df = flow.get_observation_dataframe()

    assert list(obs_df.columns) == [
        "time",
        "node",
        "water_depth",
        "reservoir_storage",
    ]
    node_0 = obs_df.loc[obs_df["node"] == 0].iloc[0]
    node_1 = obs_df.loc[obs_df["node"] == 1].iloc[0]
    assert node_0["water_depth"] == 0.1
    assert node_0["reservoir_storage"] == 200.0
    assert node_1["water_depth"] == 0.2
    assert np.isnan(node_1["reservoir_storage"])

    observation_dataframes = flow.get_observation_dataframes()
    assert set(observation_dataframes) == {"nodes", "reservoir"}
    assert len(observation_dataframes["nodes"]) == 2
    assert len(observation_dataframes["reservoir"]) == 1


def test_observation_recorder_names_must_be_unique(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    with pytest.raises(ValueError, match="non-empty string"):
        flow.set_observation_points(
            nodes=0,
            variables=["water_depth"],
            name="",
        )

    flow.set_observation_points(
        nodes=0,
        variables=["water_depth"],
        name="same",
    )

    with pytest.raises(ValueError, match="already exists"):
        flow.set_observation_points(
            nodes=1,
            variables=["water_depth"],
            name="same",
        )


def test_reservoir_outputs_can_be_requested_from_run_simulation(tmp_path):
    flow = _small_flow_simulation(tmp_path)
    geometry = flow.network
    flow.set_initial_conditions(
        initial_Q=np.zeros(geometry.Nt, dtype=float),
        initial_y=np.full(geometry.Np, 0.01, dtype=float),
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

    results = flow.run_simulation(
        desired_outputs={
            "output_interval": 0.1,
            "time": True,
            "reservoir_nodes": True,
            "reservoir_water_depths": True,
            "reservoir_heads": True,
            "reservoir_storage": True,
            "reservoir_exchange": True,
            "reservoir_recharge": True,
        }
    )

    assert results["reservoir_nodes"].shape[1] == 1
    assert results["reservoir_nodes"][0, 0] == 0
    assert results["reservoir_water_depths"].shape == results["reservoir_heads"].shape
    assert results["reservoir_storage"].shape == results["reservoir_heads"].shape
    assert results["reservoir_exchange"].shape == results["reservoir_heads"].shape
    assert results["reservoir_recharge"].shape == results["reservoir_heads"].shape
    assert np.isfinite(results["reservoir_storage"]).all()


def test_duplicate_reservoir_bc_raises_unless_overwritten_or_removed(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    flow.set_reservoir_BC(nodes=0, fixed_exchange_rate=0.001)

    with pytest.raises(ValueError, match="Reservoir BC already exists"):
        flow.set_reservoir_BC(nodes=0, fixed_exchange_rate=0.002)

    flow.set_reservoir_BC(nodes=0, fixed_exchange_rate=0.002, mode="overwrite")
    bcs = flow.boundary_conditions["reservoir"]
    assert len(bcs) == 1
    assert bcs[0].target_ids == [0]
    assert bcs[0].get_value(0.0) == 0.002

    flow.set_reservoir_BC(nodes=0, fixed_exchange_rate=0.0, mode="remove")
    assert flow.boundary_conditions["reservoir"] == []


def test_reservoir_and_inflow_same_node_conflict(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    flow.set_reservoir_BC(nodes=0, fixed_exchange_rate=0.001)
    flow.set_inflow_BC(nodes=0, values=0.001)

    with pytest.raises(ValueError, match="reservoir and inflow"):
        flow._check_bc_conflicts()


def test_reservoir_and_waterdepth_same_node_conflict(tmp_path):
    flow = _small_flow_simulation(tmp_path)

    flow.set_reservoir_BC(nodes=0, fixed_exchange_rate=0.001)
    flow.set_waterdepth_BC(nodes=0, values=0.01)

    with pytest.raises(ValueError, match="reservoir and prescribed water depth"):
        flow._check_bc_conflicts()


def test_positive_fixed_reservoir_exchange_matches_equivalent_inflow(tmp_path):
    def run_with_source(source_kind):
        flow = _small_flow_simulation(tmp_path / source_kind)
        geometry = flow.network
        flow.set_initial_conditions(
            initial_Q=np.zeros(geometry.Nt, dtype=float),
            initial_y=np.full(geometry.Np, 0.01, dtype=float),
        )
        if source_kind == "inflow":
            flow.set_inflow_BC(nodes=0, values=0.001)
        else:
            flow.set_reservoir_BC(nodes=0, fixed_exchange_rate=0.001)
        flow.set_waterdepth_BC(nodes=4, values=0.01)
        return flow.run_simulation(
            desired_outputs={
                "output_interval": 0.1,
                "time": True,
                "flowrates": True,
                "water_depths": True,
            }
        )

    inflow = run_with_source("inflow")
    reservoir = run_with_source("reservoir")

    np.testing.assert_allclose(reservoir["time"], inflow["time"])
    np.testing.assert_allclose(reservoir["flowrates"], inflow["flowrates"])
    np.testing.assert_allclose(reservoir["water_depths"], inflow["water_depths"])
