# Installation

openKARST requires Python 3.11 or newer. A separate environment manager is not
required. However, using Python's built-in `venv` module is recommended because
it keeps openKARST and its dependencies isolated from other Python projects.

## Requirements

- Python 3.11 or newer
- `pip`
- Git, when installing directly from GitHub or cloning the repository

## Create a virtual environment

=== "Unix/macOS"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows PowerShell"

    ```powershell
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

Once activated, `python` and `pip` refer to this isolated environment. You can
deactivate it later with `deactivate`.

## Install from GitHub

```bash
python -m pip install --upgrade pip
python -m pip install "openkarst @ git+https://github.com/ERC-Karst/openkarst.git"
```

This installs the current openKARST package and its required dependencies into
the active environment.

## Install from a local clone

Use this method when you want a local copy of the examples or source files:

```bash
git clone https://github.com/ERC-Karst/openkarst.git
cd openkarst
python -m pip install .
```

The regular installation above is appropriate for most users. An editable
installation is only needed when actively developing openKARST itself.
