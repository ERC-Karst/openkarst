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
    assert reservoir.get_hydraulic_head() == flow.Z[0] + 2.0
    assert reservoir.get_storage() == 200.0


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

    def compute_exchange(node_water_depth, dt):
        exchange_calls.append((node_water_depth, dt))
        return 0.003

    def advance(exchange_rate, dt):
        advance_calls.append((exchange_rate, dt))

    monkeypatch.setattr(reservoir, "compute_exchange", compute_exchange)
    monkeypatch.setattr(reservoir, "advance", advance)

    flow._cache_hydraulic_bcs()
    flow._advance_reservoirs()

    assert exchange_calls == [(0.75, 0.25)]
    assert advance_calls == [(0.003, 0.25)]
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
