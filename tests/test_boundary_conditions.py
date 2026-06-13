import numpy as np
import pytest

from openkarst.models.boundary_conditions import (
    BoxBC,
    ConstantBC,
    TimeSeriesBC,
    broadcast_boundary_values,
    normalize_target_ids,
)


def test_normalize_target_ids_accepts_scalar_and_sequences():
    assert normalize_target_ids(2) == [2]
    assert normalize_target_ids(np.int64(3)) == [3]
    assert normalize_target_ids([1, 2]) == [1, 2]
    assert normalize_target_ids(np.array([4, 5])) == [4, 5]


def test_broadcast_boundary_values_broadcasts_scalar_tuple_and_zero_dimensional_array():
    target_ids = [1, 2]

    assert broadcast_boundary_values(target_ids, 0.5) == [0.5, 0.5]
    assert broadcast_boundary_values(target_ids, ("box", 1.0, 0.0, 1.0)) == [
        ("box", 1.0, 0.0, 1.0),
        ("box", 1.0, 0.0, 1.0),
    ]
    assert broadcast_boundary_values(target_ids, np.array(0.2)) == [0.2, 0.2]


def test_broadcast_boundary_values_validates_per_target_values():
    assert broadcast_boundary_values([1, 2], [0.1, 0.2]) == [0.1, 0.2]
    assert broadcast_boundary_values([1, 2], np.array([0.1, 0.2])) == [0.1, 0.2]

    with pytest.raises(ValueError, match="Length mismatch"):
        broadcast_boundary_values([1, 2], [0.1])


def test_constant_boundary_condition_returns_same_value():
    bc = ConstantBC(target_ids=[1, 2], value=3.5, bc_type="volumetric")

    assert bc.target_ids == [1, 2]
    assert bc.bc_type == "volumetric"
    assert bc.get_value(0.0) == 3.5
    assert bc.get_value(10.0) == 3.5


def test_box_boundary_condition_uses_before_during_and_after_values():
    bc = BoxBC(
        target_ids=[0],
        v_during=2.0,
        t_start=5.0,
        t_end=10.0,
        v_before=0.1,
        v_after=0.2,
    )

    assert bc.get_value(4.9) == 0.1
    assert bc.get_value(5.0) == 2.0
    assert bc.get_value(10.0) == 2.0
    assert bc.get_value(10.1) == 0.2


def test_time_series_boundary_condition_interpolates_and_holds_edges():
    bc = TimeSeriesBC(target_ids=[0], times=[0.0, 10.0], values=[1.0, 3.0])

    assert bc.get_value(-1.0) == 1.0
    assert bc.get_value(5.0) == 2.0
    assert bc.get_value(11.0) == 3.0


def test_time_series_boundary_condition_can_zero_extrapolate():
    bc = TimeSeriesBC(
        target_ids=[0],
        times=[0.0, 10.0],
        values=[1.0, 3.0],
        extrapolate="zero",
    )

    assert bc.get_value(-1.0) == 0.0
    assert bc.get_value(11.0) == 0.0


def test_boundary_conditions_reject_invalid_options():
    with pytest.raises(ValueError, match="Invalid bc_type"):
        ConstantBC(target_ids=[0], value=1.0, bc_type="invalid")

    with pytest.raises(ValueError, match="Invalid extrapolate"):
        TimeSeriesBC(target_ids=[0], times=[0.0, 1.0], values=[1.0, 2.0], extrapolate="invalid")
