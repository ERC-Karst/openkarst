# Reference Simulations

Reference simulations are small, deterministic hydraulic scenarios with stored
reference outputs. They are used to detect unintended behavioral changes during
refactoring.

Transport is intentionally excluded until it becomes part of the released
behavior.

Regenerate the stored outputs only when a behavior change is deliberate:

```bash
python tests/reference_simulations/regenerate_reference_outputs.py
```

