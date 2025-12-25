# Changelog

## v5.5.0

- LineID/Wavesol: реализованы все кнопки управления библиотекой пар (Use/Copy/Save/Open/Export).
- LineID/Wavesol: исправлена «мертвая» выпадашка наборов пар — теперь список корректно обновляется из built-in + user library.
- Cosmics: k приведён к правильному смыслу (порог σ/MAD), исправлена синхронизация QDoubleSpinBox с конфигом.

## v5.4.0

- Sky subtraction: добавлен интерактивный выбор регионов object/sky на линейризованном кадре (ROI) и сохранение в конфиг.
- Stage-state: улучшена индикация Dirty/Up-to-date и причины перепрогона (manifest/stage_state.json).

## v5.3.0

### New long-slit science steps (core v5.x)
- **Linearize (2D λ-map):** linearizes cleaned object frames onto a common linear wavelength grid using the 2D dispersion solution (``lambda_map.fits``).
  - Outputs a **linearized summed object frame** (WCS: Å/pix) and optional per-exposure rectified frames.
  - Adds first-pass **variance** estimation (GAIN from FITS header, fallback defaults) and carries a mask plane (reserved for cosmic/badpix).
- **Sky subtraction (Kelson-style, baseline v1):** runs on the linearized summed frame.
  - User defines **object** and **sky** regions on the 2D frame (interactive GUI, stored in config).
  - Produces ``sky_model.fits`` and ``obj_sky_sub.fits`` + QC plots/metrics for residuals in sky regions.
- **1D extraction:** sums rows in the chosen object aperture to produce a 1D spectrum (flux(λ), var(λ), mask(λ)).

### Reliability fixes required for v5.x work
- Fixed stage execution wiring (GUI runner → actual stage functions) so Linearize/Sky/Extract run end-to-end.
- Normalized stage naming via aliases (e.g. ``sky_sub`` → ``sky``) to keep backward compatibility in UI/workflow.

## v5.2.0

### Foundation for v5.x
- Added initial stage scaffolding and parameter schema for: ``linearize``, ``sky``, ``extract1d``.
- Introduced stage provenance helpers (pipeline/package version, git commit) for FITS/JSON products.

## v4.13.1

### UI hotfix
- Fixed a startup crash in Project & Data page: incorrect QFormLayout.addRow() signature (was passing QHBoxLayout as the label).

## v4.9

### Windows build fixes (critical)
- Fixed `tools/build_ui_exe.ps1` so PyInstaller **always finds resources** (`src/scorpio_pipe/resources`) even when the generated `.spec` is written under `build\\pyinstaller`.
  - Uses an **absolute** `--add-data` input path (prevents resolution relative to the `.spec` directory).
  - Added strict exit-code checks (no more "Done" after a failed build).
- `scripts/windows/setup.bat` now supports `--installer` mode:
  - builds `scorpipe.exe` and then builds `packaging/windows/Output/setup.exe` via Inno Setup.
- `packaging/windows/build.ps1` now fails fast on non-zero exit codes from both PyInstaller and ISCC.

## v4.8

### Packaging / Setup reliability
- Fixed `setup.bat` to be **robust on Windows**:
  - Prefers `py -3` launcher (avoids the Microsoft Store python alias).
  - Resolves project root to an absolute path (spaces-safe) and uses **absolute paths** for build scripts.
  - Uses the venv interpreter for pip/run and **fails fast** on build errors.
- `tools/build_ui_exe.ps1` now supports `-PythonPath` and builds via `python -m PyInstaller` for consistent interpreter selection.

### GitHub Releases
- Windows release workflow now triggers on:
  - pushed tags (`v*` and `*.*`), and
  - **published GitHub Releases**.
- The workflow builds and uploads Release assets:
  - `setup.exe`
  - `Scorpipe-Windows-x64.zip` (contains `setup.exe` + `INSTALL.md`)

## v4.7

### UI / UX
- Added an always-visible **Inspector** dock (right side) with 3 tabs:
  - **Outputs** — expected products for the current step + existence checks.
  - **Plan** — RUN/SKIP decisions in *Resume* / *Force* modes.
  - **QC Alerts** — compact warnings/errors summary + quick-open.
- Added a richer **Status bar** with quick access to:
  - Data directory, Work directory, current config, QC report.
- Added keyboard shortcuts:
  - **Ctrl+I** Inspect dataset
  - **Ctrl+S** Save config
  - **Ctrl+R** Run all steps
  - **Ctrl+P** Run plan
  - **Ctrl+Q** QC viewer
- Dataset Overview now supports **multi-select** + **batch** actions:
  - Batch: build configs (per object)
  - Batch: run (non-interactive steps only; prepares LineID)

### Packaging / Release
- Version bump: pipeline **4.7**, python package **0.2.7**.
- Clarified the distribution model in docs: `setup.exe` is produced by the Windows build (GitHub Actions / local Windows build), and is not present in source archives.
