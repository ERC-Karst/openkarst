"""Regenerate stored reference simulation outputs.

Run this script only when an intentional model behavior change should update
the reference outputs:

    python tests/reference_simulations/regenerate_reference_outputs.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.reference_simulations.scenarios import REFERENCE_SCENARIOS  # noqa: E402


REFERENCE_DIR = Path(__file__).resolve().parent


def main():
    manifest = {
        "name": "openKARST reference simulations",
        "generated_on": date.today().isoformat(),
        "generator": "tests/reference_simulations/regenerate_reference_outputs.py",
        "transport_included": False,
        "description": (
            "Small deterministic hydraulic simulations used to detect "
            "unintended behavioral changes during refactoring."
        ),
        "scenarios": [],
    }

    for scenario in REFERENCE_SCENARIOS:
        output_file = REFERENCE_DIR / f"{scenario.name}.npz"
        with tempfile.TemporaryDirectory(prefix=f"openkarst-{scenario.name}-") as tmp:
            with open(Path(tmp) / "simulation_stdout.txt", "w", encoding="utf-8") as stdout:
                with redirect_stdout(stdout):
                    results = scenario.runner(Path(tmp))

        np.savez_compressed(output_file, **results)
        manifest["scenarios"].append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "file": output_file.name,
                "outputs": sorted(results),
            }
        )
        print(f"Wrote {output_file}")

    manifest_file = REFERENCE_DIR / "reference_manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_file}")


if __name__ == "__main__":
    main()

