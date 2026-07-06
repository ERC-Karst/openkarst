import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.reference_simulations.scenarios import REFERENCE_SCENARIOS


REFERENCE_DIR = Path(__file__).parent / "reference_simulations"
EXACT_OUTPUT_KEYS = {
    "picard_iterations",
    "picard_iterations_total",
    "reservoir_nodes",
}


def _assert_reference_close(name, actual, expected):
    if name in EXACT_OUTPUT_KEYS:
        np.testing.assert_array_equal(actual, expected)
    elif name == "time":
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("scenario", REFERENCE_SCENARIOS, ids=lambda scenario: scenario.name)
def test_reference_simulation_matches_stored_output(scenario):
    reference_file = REFERENCE_DIR / f"{scenario.name}.npz"
    assert reference_file.is_file(), (
        f"Missing reference output {reference_file}. Run "
        "python tests/reference_simulations/regenerate_reference_outputs.py"
    )

    with tempfile.TemporaryDirectory(prefix=f"openkarst-{scenario.name}-") as tmp:
        actual = scenario.runner(Path(tmp))

    with np.load(reference_file) as expected:
        assert set(actual) == set(expected.files)
        for key in expected.files:
            _assert_reference_close(key, actual[key], expected[key])

