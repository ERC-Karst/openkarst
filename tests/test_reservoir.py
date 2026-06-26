import pytest

from openkarst.models import UnconfinedReservoir


def _reservoir(**kwargs):
    defaults = {
        "node": 0,
        "base_elevation": 10.0,
        "area": 100.0,
        "specific_yield": 0.2,
        "initial_water_depth": 2.0,
        "conductance": 0.5,
    }
    defaults.update(kwargs)
    return UnconfinedReservoir(**defaults)


def test_compute_exchange_uses_connected_node_water_depth_difference():
    reservoir = _reservoir()

    exchange_rate = reservoir.compute_exchange(
        connected_node_water_depth=1.5,
        dt=1.0,
    )

    assert exchange_rate == pytest.approx(0.25)


def test_compute_exchange_limits_positive_rate_by_available_storage():
    reservoir = _reservoir(
        area=1.0,
        specific_yield=1.0,
        initial_water_depth=0.1,
        conductance=10.0,
    )

    exchange_rate = reservoir.compute_exchange(
        connected_node_water_depth=0.0,
        dt=2.0,
    )

    assert exchange_rate == pytest.approx(0.05)


def test_compute_exchange_does_not_storage_limit_negative_rate():
    reservoir = _reservoir(
        area=1.0,
        specific_yield=1.0,
        initial_water_depth=0.1,
        conductance=10.0,
    )

    exchange_rate = reservoir.compute_exchange(
        connected_node_water_depth=1.0,
        dt=2.0,
    )

    assert exchange_rate == pytest.approx(-9.0)


def test_compute_exchange_rejects_invalid_inputs():
    reservoir = _reservoir()

    with pytest.raises(ValueError, match="dt"):
        reservoir.compute_exchange(
            connected_node_water_depth=1.0,
            dt=0.0,
        )

    with pytest.raises(ValueError, match="connected_node_water_depth"):
        reservoir.compute_exchange(
            connected_node_water_depth=float("nan"),
            dt=1.0,
        )
