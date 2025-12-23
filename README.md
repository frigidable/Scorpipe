# Scorpipe (v4.6)

Pipeline for processing long-slit spectroscopic data from SCORPIO instruments (BTA).
It provides night inspection, automatic configuration, bias/neon calibration, and an
interactive LineID GUI for wavelength solutions.

For detailed usage, see [MANUAL.md](MANUAL.md).

## What's new in v4.6

- **Project UI**: frame table with filters + quick FITS preview; one-click "use this setup".
- **Config UI**: "Diff" button against last saved YAML.
- **Run UI**: per-step Outputs panel (expected products + existence).
- **Toolbar**: "Plan" dialog showing what will run/skip in resume mode.
- **Windows installer pipeline**: GitHub Actions now publishes a **Release asset** containing `setup.exe` (and fixes the Inno Setup output path).
- **Docs**: clear distinction between **source zip** (Code → Download ZIP) and **Windows release zip** (Releases → Scorpipe-Windows-x64.zip).

## Installation

### Windows (recommended)

Important:

- **"Code → Download ZIP" is source code** and does **NOT** contain `setup.exe`.
- To install without Python, open **GitHub Releases** and download **`Scorpipe-Windows-x64.zip`** (Release asset). It contains `setup.exe`.

Then run `setup.exe` and launch **Scorpipe** from Start Menu.

See **INSTALL.md** for details.

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

You can validate environment and config:

```bash
scorpio-pipe doctor
scorpio-pipe validate --config work/<dd_mm_yyyy>/config.yaml
```


Inspect a night and generate a config for an object:

```bash
scorpio-pipe inspect --data-dir /path/to/night
scorpio-pipe run --data-dir /path/to/night --object "NGC2146sat"
# to run non-interactive steps immediately:
# scorpio-pipe run --data-dir /path/to/night --object "NGC2146sat" --execute
# (work dir will be inferred as work/<dd_mm_yyyy> from nightlog; or specify explicitly)
# scorpio-pipe run --data-dir /path/to/night --object "NGC2146sat" --work-dir work/16_12_2025
```

Run the workflow tasks with `doit` (GUI will open for `lineid_prepare`):

```bash
# Recommended: set CONFIG
export CONFIG=$(pwd)/work/<dd_mm_yyyy>/config.yaml

doit -f workflow/dodo.py list
doit -f workflow/dodo.py lineid_prepare
```

Alternatively, pass the config directly:

```bash
doit -f workflow/dodo.py config=work/<dd_mm_yyyy>/config.yaml lineid_prepare
```

## Entry points

- `scorpio-pipe` — CLI pipeline commands
- `scorpipe` / `scorpio-ui` — GUI launcher (requires `.[gui]`)
