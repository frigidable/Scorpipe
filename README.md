# Scorpipe

Pipeline for processing long-slit spectroscopic data from SCORPIO instruments (BTA).
It provides night inspection, automatic configuration, bias/neon calibration, and an
interactive LineID GUI for wavelength solutions.

For detailed usage, see [MANUAL.md](MANUAL.md).

## Installation

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -e .
```

Optional extras:

```bash
# GUI for LineID and embedded PDF atlas
pip install -e ".[gui]"

# SciPy for improved peak detection
pip install -e ".[science]"
```

## Quick start

Inspect a night and generate a config for an object:

```bash
scorpio-pipe inspect --data-dir /path/to/night
scorpio-pipe run --data-dir /path/to/night --object "NGC2146sat" --work-dir work/ngc2146sat
```

Run the workflow tasks with `doit` (GUI will open for `lineid_prepare`):

```bash
# Recommended: set CONFIG
export CONFIG=$(pwd)/work/ngc2146sat/config.yaml

doit -f workflow/dodo.py list
doit -f workflow/dodo.py lineid_prepare
```

Alternatively, pass the config directly:

```bash
doit -f workflow/dodo.py config=work/ngc2146sat/config.yaml lineid_prepare
```

## CLI entry points

- `scorpio-pipe` — CLI pipeline commands
- `scorpio-ui` — GUI launcher (requires `.[gui]`)
