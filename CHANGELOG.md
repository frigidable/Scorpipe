## v5.40.20

### Fixes
- CI (Windows Release): generate SBOM for the portable Windows ZIP by extracting it first and scanning the resulting directory (avoids Syft failures when scanning the ZIP directly).

## v5.40.19

### Fixes
- CI (Windows Release): replaced `softprops/action-gh-release` with `gh release upload` to avoid intermittent/permission-related 404s when fetching the third-party action; release assets are now uploaded using the official GitHub CLI.

## v5.40.18

### Fixes
- Packaging/CLI: added `scorpio_pipe.__main__` so `python -m scorpio_pipe` works (defaults to `version` when invoked without arguments).

## v5.40.17

### Fixes
- Sky Subtraction: when legacy aliases `*_skysub.fits` are written in the Sky stage, prefer copying an existing rectified product from Linearize (if present) so downstream stages/tests get a proper wavelength grid.
- Sky Subtraction (Kelson RAW): flexure estimation is now best-effort; failures no longer prevent writing `*_skysub_raw.fits`.
- Stack2D: restored stable top-level return keys (`stack2d_fits`, `stacked2d_fits`, `coverage_png`, `qc_png`) in the function result for downstream convenience/tests.

## v5.40.16

### Fixes
- Sky Subtraction (Kelson RAW): fixed missing outputs in some CI environments by using a compatibility wrapper around the B-spline design matrix and avoiding sparse broadcasting assumptions during weighted least-squares.

## v5.40.14

### Fixes
- Fixed JSON manifest writing to support small dataclasses/paths/numpy scalars in metadata (prevents crashes when `done.json` contains `ROISelection`).
- Fixed FITS header keyword warnings by compacting MASK schema cards to 8-character FITS-safe keywords (no more Astropy `VerifyWarning`).

## v5.40.13

### Fixes
- Fixed Python 3.12 test collection crash caused by an accidental duplicate `@dataclass(frozen=True)` decorator on `ROISelection` in `sky_geometry.py`.

## v5.40.7

### P2 — Workspace validator + StageRegistry regressions
- Added `scorpio_pipe.run_validate.validate_run_dir()` to validate `workspace/<night>/<obj>_<disperser>_<run_id>` layout and `run.json` passport schema/consistency.
- GUI: opening a run now validates the workspace; if folder name and `run.json` diverge, a "Fix run.json" button is shown (no silent guessing).
- Runner: `run_sequence()` ensures minimal layout and validates the run passport before executing stages.
- StageRegistry: canonical stage dir for Arc Line ID is now `07_arc_line_id` (legacy `07_arclineid`/`07_lineid` are still recognized).
- Tests: added P2 regression tests for StageRegistry order and workspace validation.

## v5.40.4

### P1-E — Frame Stacking: strict contract + robust combine + η(λ) variance calibration
- Stack2D input contract hardened: consumes **only** `10_linearize/*_skysub.fits` (no legacy fallbacks in runner/GUI).
- Compatibility checks: validates **shape**, **wavelength grid**, and **SCI unit** (with consistent per-second normalization via EXPTIME / TEXPS).
- Robust stacking: adds **invvar_huber** (default) and **invvar_clip** methods, with masked/fatal-bit aware coverage and REJECTED bookkeeping.
- Variance stability: optional **VAR floor** from a low percentile of sky VAR (sub-sampled, ROI-aware when available).
- η(λ) calibration: estimates column-wise variance mismatch in sky windows (MAD-based), smooths, clamps, writes `eta_lambda.fits`, and stamps headers (`ETAAPPL`, etc.).
- Reporting: writes canonical `stack_done.json` (and keeps `stack2d_done.json` + `done.json`), including inputs, checks, runtime, metrics, and QC flags.
- Outputs: writes canonical `11_stack/stack2d.fits` (and keeps `stacked2d.fits`).

## v5.40.3

### P1-D — Linearization: full Sky→Linearize transfer + reporting contract
- Linearize: fixed per-run reporting to include a full P1-D schema (`inputs/grid/resampling/delta_lambda/roi/cleanup/metrics/outputs/flags`) and a detailed `per_exposure` array.
- Linearize: fixed a runtime bug in coverage bookkeeping (`no_coverage_fracs` append typo).
- Residual cleanup diagnostics: AUTO reports both output metrics and (when rejected) candidate-after metrics, without mislabeling whether cleanup was applied.

## v5.40.2

### P1-C — Sky Subtraction (RAW) ROI-aware + SciPy spline core
- Dependencies: promoted **SciPy** to a core dependency (no more "optional science" install needed for key spectral steps).
- Sky Subtraction (Kelson RAW): model fit now uses SciPy's cubic B-spline design matrix + sparse weighted least-squares (robust sigma-clipping), improving numerical stability and accuracy.
- Reports: `sky_done.json` is now aligned with the standard `done.json` QC contract (`qc.flags`, `qc.max_severity`, `status=ok|warn|fail`) and includes P1-C fields (ROI provenance, flexure summary, metrics, outputs).

## v5.40.1

### P1-B — Wavelength Solution artifacts for Kelson + Linearization
- Wavesolution: writes `rectification_model.json` next to `lambda_map.fits` (unit/reference, sha256 signature, and VAR/MASK policies).
- Linearize: prefers `rectification_model.json` to locate/validate the lambda-map and records hashes + policies in `done.json`.
- QC/UI: product registry + stage contract now include the rectification model as a first-class artifact.

## v5.40.0

### P0 — QC gate + done.json contract + MEF I/O
- Runner: added a **QC gate** that stops downstream stages when any upstream stage emitted QC flags with severity **ERROR/FATAL** in its `done.json`. The GUI shows a blocking dialog with details and an explicit **override** button; CLI adds `--override-qc-gate`. Any override is recorded in `manifest/stage_state.json`.
- Linearize / Sky Subtraction / Stack2D: now write canonical `done.json` (and keep legacy `*_done.json`) and embed `qc.flags` + `qc.max_severity` for reproducible gating.
- MEF contract: `write_sci_var_mask(...)` now supports an optional `COV` extension (exposure coverage); `validate_sci_var_mask()` understands it. Mask bits remain consistent across stages.
- Sky Subtraction: fixed a crash in output bookkeeping (missing `done.json` path).

## v5.39.4

### GUI — finish Wavesol + Linearize refactor (DoD)
- Wavesol: parameter pane is now sectioned (Core / Edge crop / Power model / Chebyshev model / Trace / Power fit / Chebyshev fit).
- Wavesol: model2d gating is done via enable/disable (no hiding) → stable layout; also fixed "dirty on open" by removing init pending.
- Linearize: parameter pane is now sectioned (Core / Geometry / Diagnostics, plus Advanced Outputs) with checkbox text removed.
- Params: default-icon hover now shows Default + (when available) Usually from centralized metadata.
- Metadata: expanded `param_metadata.py` (Wavesol + SuperNeon-related Wavesol keys), used as UI source of truth.

## v5.39.3

### GUI — Linearize: units + defaults + tooltip polish
- Linearize: parameter rows now use schema-backed cfg_path, so the default-icon reset works reliably and tooltips show the default value.
- Units are rendered consistently as `Name [unit]` (e.g. Δλ [Å], λ min/max [Å], crop [px]).
- Default-icon hover text is now human-friendly for schema defaults like `None`/empty/bool (shows `auto`, `empty`, `on/off` instead of `None/True/False`).

## v5.39.2

### GUI — layout stability, less visual noise
- Stage pages: Outputs are shown in a detached tool window (toolbar action) — no in-page Outputs panes/drawers, so the main layout does not shift.
- Setup page: removed inline help/"?" buttons; help is now via tooltip on parameter labels.

## v5.38.2

### P1 — Stack2D as a first-class stage + strict product naming
- Stack2D is now an explicit stage everywhere: pipeline/UI order is **Sky → Stack2D → Extract1D**, with products in the canonical `products/NN_stack2d/` stage dir.
- Stack2D input selection is now deterministic: it validates that a sky-subtracted product exists for every science `raw_stem` from `frames.obj` (and fails fast with a clear list of missing stems).
- Naming policy hardened: stages no longer write compatibility aliases (e.g. `*_lin.fits`). Legacy names remain supported for reading via the resolver.
- Extract1D: keeps the strict requirement that Stack2D must be complete in the pipeline flow; adds an **opt-in** manual fallback to a Sky frame via `extract1d.allow_sky_fallback`.

## v5.38.1

### Block 1 — Canonical workspace paths + backward compatibility
- Stage Registry (single source of truth): introduced `stage_registry.py` with 1:1 mapping **GUI ↔ stage_id ↔ products/NN_slug**.
- Canonical workspace paths: added `workspace_paths.py` helpers (`products_dir`, `stage_dir`, `per_exp_dir`) and a universal **new → legacy** reader resolver (`resolve_input_path`, `legacy_candidates`).
- Minimal workspace layout: `ensure_work_layout()` now creates only `raw/`, `products/`, `manifest/` by default (no auto-created legacy `calib/` / `report/`).
- Backward compatibility: reading stages now prefer canonical `products/NN_*` but can fall back to legacy locations without writing there (unless legacy dirs already exist).

## v5.37.10

- QC report: `build_qc_report()` now returns a mapping with output paths (`{"json": ..., "html": ...}`) while remaining backward-compatible as a Path-like HTML return value.

## v5.37.9

- QC report: fixed product-key mismatches, now correctly aggregates wavesolution/linearize/manifest signatures into one JSON+HTML page.
- Calibration done markers: added `used_signature` alias (keeps `frame_signature`) for consistency with QC tooling.

## v5.37.8

- QC report: unify calibration + wavesolution + linearize metrics into a single JSON/HTML “one page of truth” and add red-flag alerts (unit missing, signature mismatch, low coverage, RMS regression, etc.).
- SuperNeon: if bias subtraction is enabled, require a valid superbias master (fail-fast, no silent skip).
- Cosmics (laplacian mode): expose niter and edge protection parameters.

## v5.37.7
- Cosmics: added explicit `cosmics.method` = `current|laplacian`. The Laplacian mode uses a SciPy-based Laplacian detector and writes per-kind `qc.json` with mask statistics (requires `scorpio-pipe[science]`).

## v5.37.6
- Variance model: centralized GAIN/RDNOISE handling and made variance propagation physically consistent through flatfielding and linearize.

## v5.37.5
- Linearize: added EXPTIME normalization (ADU/s), sigma-clipped robust stacking across exposures, and expanded `linearize_qc.json`.

## v5.37.4
- SuperNeon now produces shift/QC machine-readable artifacts and strict FrameSignature validation.
## v5.37.3

### P0 — FrameSignature: strict calibration compatibility (ROI / binning / readout)
- Introduced **FrameSignature** (shape + binning + ROI/window + readout) and added it to the reproducibility **manifest**.
- Superbias and superflat now require **strictly compatible inputs**: no silent skipping, no implicit pad/crop.
- Builders write `*_done.json` markers with the signature used to improve transparency and debugging.

## v5.36.4

### GitHub / Release process hardening
- **Dependabot**: configuration is now expected at `.github/dependabot.yml` (GitHub standard path).
- **Issue templates**: cleaned up to keep only YAML issue forms; removed duplicate legacy templates.
- **PR template**: fixed Markdown escaping so checklists and headings render correctly.
- **Docs**: added `docs/RELEASING.md`; refreshed `docs/RUNBOOK.md` and `docs/AUDIT.md` titles to avoid stale version tags.
- **Repo meta**: added `CONTRIBUTING.md` and `SECURITY.md`.
- **CI**: added a **build_check** job (build sdist/wheel, install from wheel, smoke) to prevent broken releases.

## v5.36.2

### Cosmics (step 2.1) + memmap-safe stack2d
- **Cosmics**: `method=auto` now uses an L.A.Cosmic-like single-frame detector (`la_cosmic`) when only one exposure is available; keeps 2-frame diff for N=2 and stack-MAD for N≥3.
- **Cosmics**: added optional **line protection** heuristics for long-slit sky/emission lines to reduce the risk of “eating” real spectral structure.
- **Cosmics**: ensured FITS I/O for science frames uses the smart loader (`read_image_smart`) and stores per-frame mask FITS as `uint8` to avoid FITS scaling surprises.
- **Stack2D**: switched MEF loading to `memmap="auto"` to avoid Astropy strict-memmap failures when MASK extensions carry scaling keywords (BZERO/BSCALE/BLANK).

## v5.36.1

### Stage contract + QC metrics skeleton + FITS I/O hardening
- **Stage contract**: added a lightweight `StageContract` registry describing per-stage inputs/outputs/metrics (used by tests and QC).
- **QC skeleton**: added a persistent `products/metrics.json` store that is updated after each stage run/skip/fail; added automatic mirroring of `work/qc` into `products/qc`.
- **FITS I/O**: unified GUI preview loading on a single smart loader (`open_fits_smart` / `read_image_smart`) with explicit scaling diagnostics (BZERO/BSCALE/BLANK).
- **NumPy 2.0 compatibility**: removed strict `copy=False` dtype conversions that could crash FITS display and some numeric paths.

## v5.30.0

### Hotfixes: cosmics diff + Sky ROI serialization
- **Cosmics**: fixed an off-by-one bug in the fast boxcar local-mean computation that produced `(ny-1, nx-1)` arrays and caused broadcasting crashes during 2-frame cleaning.
- **Sky ROI**: added `ROI.to_dict()` so Sky ROI selection can be saved back into the config without crashing the GUI.

## v5.29.0

### Fixes: LineID GUI restore + robust cosmics geometry + GUI logging
- **LineID GUI**: the **"Open LineID GUI"** action now actually opens the interactive LineID dialog (v4.13.2 behavior) and allows creating/updating `hand_pairs.txt`.
- **Cosmics**: fixed crashes when input frames differ by ~1 pixel in geometry (e.g. `(1040, 4112)` vs `(1039, 4111)`). The stage now crops science frames (and superbias) to the common overlap and proceeds.
- **GUI**: fixed crash `AttributeError: 'LauncherWindow' object has no attribute '_log_warn'` (Sky ROI selection now logs warnings correctly).

## v5.27.0

### Critical fixes: LineID + cosmics for small N + work-dir resolver
- **resolve_work_dir**: removed remaining fragile imports; all stages now import the canonical resolver from `scorpio_pipe.paths` (eliminates the frequent `NameError: resolve_work_dir is not defined`).
- **LineID GUI**:
  - `lineid_prepare` now returns the actual artifact paths (template/auto/report) so the log no longer shows `LineID wrote: None`.
  - Added a **wait cursor** while preparing LineID artifacts to reduce the “frozen UI” feel.
- **Cosmics**: new robust behavior when you have *very few science frames*:
  - For **N=2**, automatic fallback to a **two-frame difference** detector (replaces spikes using the other exposure).
  - For **N=1**, fallback to a **Laplacian high-pass** detector with local-mean replacement.
  - Keeps the original stack-MAD method for **N>=3**.

## v5.26.0

### GUI + stage robustness sweep
- **Extract 1D**: fixed a latent crash when running the stage (wrong import of a removed module); stage now uses the canonical work-dir resolver.
- **FITS preview**: more robust image-HDU discovery + richer diagnostics (includes `HDUList.info()` on failure). Preview now defaults to `memmap=False` for stability and only falls back to `memmap=True` when needed.
- **Frames windows**: browsers now recognize `*.fts` alongside `.fits/.fit` when scanning stage output folders.
- **QC viewer**: opens maximized by default (consistent with other windows).
- **Packaging**: aligned `pyproject.toml` and internal version strings to avoid mismatched release metadata.

## v5.25.0

### Hotfixes: runner stability + FITS viewer + window behavior
- **Cosmics**: fixed crash `NameError: _load_cfg_any is not defined` (stage now normalizes config input like other stages).
- **LineID GUI**: fixed crash `NameError: resolve_work_dir is not defined` and corrected wavesolution directory resolution.
- **FITS preview**: fixed a common SCORPIO case where scaled images (BZERO/BSCALE/BLANK) failed with memmap=True but were incorrectly reported as “No image HDU…”; viewer now retries reliably with `memmap=False`.
- **Windows**: main window and major dialogs now open **maximized** by default.

## v5.24.0

### GUI: Basic/Advanced tabs + scrolling for all stages
- Unified all stage parameter panels to the same UX as *Wavelength solution*:
  - **Basic / Advanced** tabs everywhere (Calib, Cosmics, Linearize, Sky, Extract 1D).
  - Each tab content is **scrollable**, so long parameter lists never get cut off.
- Fixed a serious UI bug where the **Cosmics Basic** box was created but not added to the layout (parameters could be missing).

## v5.23.0

### GUI: fixed "Save/Apply/Run does nothing" regressions
- **Hardened YAML sync and stage control syncing**: the GUI no longer crashes silently if the user edits YAML values as strings or leaves a numeric field half-typed.
- **Config save** now uses a reliable target path:
  - prefers the path from the *Project* page (if set),
  - falls back to *work dir* when available,
  - otherwise asks for a file via **Save As…**.
- **Better feedback**: errors are shown via message boxes + log panel; successful Save/Apply shows a short status-bar message.
- **Apply now force-commits active spinbox edits** (interprets text before applying) to avoid the common Qt pitfall where the value is not yet converted.

## v5.22.0

### Frames: FITS preview now works for SCORPIO headers + DS9-like basics
- **Fixed a hard FITS preview failure** in *Frames* for files containing scaling keywords
  (BZERO/BSCALE/BLANK): the viewer now auto-falls back to `memmap=False` when Astropy
  cannot memory-map scaled images.
- Upgraded the preview widget to a more DS9-like baseline:
  - **Zoom** (mouse wheel) and **pan** (drag with left mouse button)
  - **Value under cursor** readout (x, y, value)
  - Stretch controls: colormap, percentile cut, gamma, **scale** (linear/log/sqrt/asinh), **invert**
  - Optional **downsample** for very large frames to keep GUI responsive

## v5.21.0

### Critical fixes (runner + products/QC)
- **GUI runner**: fixed multiple hard crashes caused by outdated task wrappers and path handling:
  - removed wrong positional arguments and keyword-only violations,
  - aligned calls with current stage APIs (superbias/superflat/superneon/lineid_prepare/etc.),
  - made stage input hashing use the canonical work layout and disperser-aware wavesol directory,
  - added a missing `qc_report` task to the task registry.
- **Products registry**: restored missing helpers and products required by runner/QC/tests:
  - reintroduced `group_by_stage()`, `products_for_task()`, `task_is_complete()`,
  - added missing products (superneon PNG/FITS, peaks candidates CSV, lineid outputs, cosmics summary, flatfield done, etc.),
  - added `Product.size()` used by QC report.
- **QC paths**: timings + QC alerts now prefer canonical `work/qc/*` with fallback to legacy `work/report/*`.
- **Calibrations**: superbias/superflat now write to canonical `work/calibs/*` and mirror to legacy `work/calib/*`.
- **LineID prepare** now writes outputs into `work/wavesol/<disperser_slug>/` (legacy flat layout still supported).
- Tests: synthetic smoke test is skipped cleanly when `astropy` is not installed.

## v5.20.0

## v5.19.0

### Sky subtraction (Kelson-style, per exposure in (λ,y))
- **Per-exposure Kelson-style sky modeling** on *linearized* frames (λ,y): B-splines along λ + smooth (polynomial) dependence of scale/offset with y.
- ROI **object/sky**: headless from config and optional **interactive Qt picker** (if enabled).
- New artifacts:
  - ``products/sky/per_exp/<tag>_sky_model.fits``
  - ``products/sky/per_exp/<tag>_skysub.fits`` (+ legacy alias ``<tag>_sky_sub.fits``)
  - ``products/sky/qc_sky.json`` + per-exposure diagnostic PNGs (spectrum fit + residual map).

### Linearize (MEF I/O + strict masks)
- **Fixed mask propagation** bug (undefined MASK_* constants) and aligned to ``maskbits`` semantics.
- Input science frames can now be **MEF (SCI/VAR/MASK)**; VAR/MASK are carried through linearization when present.
- Per-exposure outputs now use the canonical name ``*_rectified.fits`` (+ legacy ``*_lin.fits`` copy).

### Products/QC plumbing
- Product registry corrected (``products/spec`` dir) and augmented with JSON artifacts (e.g. ``sky_sub_done.json``, ``stack2d_done.json``, ``extract1d_done.json``) so QC report can pick them up.
- Products manifest now groups **per-exposure** artifacts by tag and includes FITS/PNG/JSON files.

# Changelog

## v5.40.7

P2 (HARDENING / QA / REGRESSIONS): "ремень безопасности".

- Workspace/run validator: `validate_run_dir()` — проверка layout `workspace/<night>/<obj>_<disperser>_<run_id>` и `run.json` (schema=1), с понятными сообщениями об ошибках.
- GUI: при открытии run выполняется валидация; при несогласованности `run.json` ↔ имя папки доступна кнопка "Fix run.json".
- Runner: `run_sequence()` гарантирует базовый layout (`ensure_work_layout`) и валидирует run перед запуском.
- StageRegistry: каноническое имя стадии `07_arc_line_id`.
- Тесты P2: фиксация 12 стадий и порядка, + регрессионные проверки `validate_run_dir()`.

## v5.16.0

Цель релиза: **довести P1-фичи до “production”** — удобный выбор λ-окон в GUI, автоматический выбор единиц (Å/px) и максимально гладкая Δλ(y).

### GUI: интерактивный выбор λ-окон
- `ui/lambda_windows_dialog.py`: новый диалог **Pick λ-windows** (SpanSelector по спектру), возвращает окна в **Å** (если есть линейный WCS) или в **pix**.
- `ui/launcher_window.py`: на странице **Sky → Advanced** добавлены:
  - блок **Flexure correction (Δλ)**: включение, `mode full|windows`, `windows_unit auto|A|pix`, кнопка **Pick…**, параметры Δλ(y).
  - блок **Stack2D options**: sigma-clip/maxiter и **y-alignment** с тем же механизмом λ-окон.

### Flexure / Δλ(y): максимальная гладкость и стабильность
- `stages/sky_sub.py`:
  - `y_poly_deg` по умолчанию **1** (самый стабильный тренд),
  - добавлено сглаживание измеренных точек `y_smooth_bins` + робастный fit с sigma-clipping (`y_sigma_clip`, `y_fit_maxiter`).
  - режим **windows** теперь поддерживает **Å и pix** (через `windows_unit`, `windows_A`, `windows_pix`).

### Bugfix: mask shifting API
- `shift_utils.py`: `shift2d_subpix_x_mask()` теперь принимает алиас `no_coverage_bit` (GUI/sky_sub больше не падают при Δλ(y)).

### Metadata
- Обновлены `HISTORY` cards в продуктах стадий до `v5.16`.


## v5.15.0

Цель релиза: **научное качество P1** — «дотянуть до результата» на серии long-slit: гибкая коррекция дрейфа, более стабильное выравнивание по y, корректная маска насыщения.

### Flexure / Δλ correction: y-dependent + windows
- `stages/sky_sub.py`: расширена опциональная коррекция flexure:
  - добавлен режим **Δλ(y)**: измерение сдвига по λ в нескольких y-бинах по sky-регионам и аппроксимация **полиномом** по y;
  - добавлен режим **windows** для кросс-корреляции (использовать только заданные диапазоны λ, чтобы максимизировать S/N по линиям);
  - сохраняются продукты QC: `*_flexure_ycurve.csv` + `*_flexure_ycurve.png` (если включено).
- Новые параметры (Advanced) в `sky.flexure`:
  - `y_dependent` (bool), `y_step`, `y_bin`, `y_poly_deg`, `min_score`,
  - `mode: full|windows`, `windows_A`,
  - `save_curve`, `save_curve_png`.

### Y-alignment before stacking: full-range or windows
- `stages/stack2d.py`: y-выравнивание теперь может строить пространственный профиль:
  - по **всему** диапазону λ (как раньше),
  - либо по выбранным **окнам λ** (`mode: windows`), чтобы выравнивание держалось за линии/полосы с максимальным S/N.

### Saturation mask propagation
- `stages/linearize.py`: добавлена **маска насыщения** (опционально):
  - уровень насыщения берётся из конфига или оценивается из FITS header (эвристики для 16-bit unsigned),
  - маска распространяется в `rect_mask` как отдельный бит (`MASK_SAT`).

### Mask bits
- Устранён конфликт битов: `stack2d` перенес `MASK_CLIPPED` на следующий бит (не пересекается с `MASK_SAT`).

### Tests
- `tests/test_subpixel_alignment.py`: добавлены тесты для новых 2D subpixel шифтеров (`shift2d_subpix_x*`).

## v5.14.0

Цель релиза: **научное качество P1** — аккуратные поправки на дрейф и выравнивание серии с *минимальным* доп. ресемплингом + расширение QC.

### Flexure / Δλ correction (optional)
- `stages/sky_sub.py`: добавлена **опциональная** коррекция глобального сдвига по λ *per exposure* по кросс-корреляции 1D спектра неба в sky-регионах:
  - ищется **субпиксельный** сдвиг (parabola refinement вокруг лучшего целочисленного лага) в пределах `max_shift_pix` (по умолчанию 5), 
  - применяется **до** построения Kelson-модели (чтобы модель работала стабильнее на серии),
  - края, появившиеся из-за сдвига, маскируются как `MASK_NO_COVERAGE`.
- Параметры:
  - `sky.flexure_enabled` (Basic) / либо `sky.flexure: {enabled, max_shift_pix}` (Advanced).

### Y-alignment before stacking (optional)
- `stages/stack2d.py`: добавлено **опциональное** выравнивание кадров по y *перед* stacking:
  - оценивается **субпиксельный** сдвиг по пространственному профилю (коллапс по λ с акцентом на положительный поток),
  - применяется при чтении чанков через **линейную интерполяцию по y** (без повторной линеаризации по λ),
  - формируется список `y_offsets` для QC.
- Параметры:
  - `stack2d.y_align_enabled` / либо `stack2d.y_align: {enabled, max_shift_pix}`.

### QC improvements
- `qc_report.py`: агрегируются новые метрики:
  - статистики по `flexure_shift_pix/A` (median, p90_abs),
  - статистики по `y_shift_pix` (median, p90_abs).

### Schema
- `schema.py`: добавлены ключи по умолчанию для flexure и y-align (обратная совместимость сохранена, `extra="allow"` остаётся).

## v5.13.0

Цель релиза: **стабильный GUI + согласование UI ↔ научные стадии (v5.x P0)** без ломания совместимости.

### Critical / GUI
- Исправлен крэш при старте GUI: добавлен недостающий helper `_small_note()` в `LauncherWindow` (ошибка `AttributeError: ... _small_note`).
- `ui/__init__.py`: теперь **не валится в headless/CI** без PySide6 (lazy/try-import). Это позволяет запускать smoke-check и импортировать пайплайн как библиотеку.

### Sky subtraction (Kelson, per exposure) — GUI wiring
- Убраны дублирующиеся чекбоксы на странице Sky (они приводили к «двойным» виджетам и потере сигналов).
- Исправлена привязка параметра `maxiter` (раньше писалось в несуществующий ключ конфигурации).
- Кнопка **Run: Sky subtraction** теперь, при включённом `Stack after sky`, автоматически запускает `stack2d` (как ты просил: `sky_sub + stack2d`).

### Extract 1D — GUI ↔ stage alignment
- Переработана страница **Extract 1D** под реальную стадию `extract1d`:
  - режимы `boxcar` (Basic) и `optimal` (Advanced) + сохранены legacy `sum/mean` ради обратной совместимости;
  - добавлены основные параметры апертуры и trace;
  - исправлено отображение outputs: продукты Extraction теперь показываются как `products/spec/...`.

### Outputs / Products
- OutputsPanel на страницах Linearize/Extract теперь смотрит на правильные группы продуктов (`lin` и `spec`), чтобы UI реально показывал наличие FITS/PNG/JSON.

## v5.12.0 (intermediate)

Фиксируем прогресс, чтобы его не потерять, и закрываем критические «бомбы», из-за которых GUI мог не запускаться.

### Critical / GUI stability
- Исправлен крэш на старте (PyInstaller/MEI): добавлен `get_provenance()` в `scorpio_pipe.version` (его импортировал `stage_state.py`).
- `ui/pipeline_runner.py`: добавлены отсутствующие GUI-обёртки `load_context()`, `run_lineid_prepare()`, `run_wavesolution()` и поддержка `RunContext` в `run_sequence()`.
- Добавлены маленькие отсутствующие модули, на которые ссылались стадии: `logging_utils.py`, `provenance.py`.

### Long-slit science (ядро — промежуточно)
- `Sky` получил **per-exposure** режим (обработка `lin/per_exp/*.fits`) + опцию `stack_after` (стэкинг внутри Sky).
  - По умолчанию `save_per_exp_model = false` (как ты просил: сохранять модель неба только при включённой галочке).
  - Добавлено сохранение 1D спектра неба (`*_sky_spectrum.csv/.json`) при включении `save_spectrum_1d`.
- Добавлена стадия `stack2d` (variance-weighting + опциональный sigma-clip) — пока как «первый надёжный вариант».

### Products
- Начата стандартизация: **canonical outputs** пишутся в `work_dir/products/...`, при этом для обратной совместимости копируются в старые пути (`work_dir/sky`, `work_dir/spec`).

### Config/schema
- В `schema.py` добавлены новые флаги Sky (`per_exposure`, `stack_after`, `save_per_exp_model`, `save_spectrum_1d`) и блок `Stack2DBlock`.

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
  - builds `scorpipe.exe` and then builds `packaging/windows/Output/ScorpioPipe-Setup-x64-<версия>.exe` via Inno Setup.
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
  - `ScorpioPipe-Setup-x64-<версия>.exe`
  - `Scorpipe-Windows-x64-<версия>.zip` (contains `ScorpioPipe-Setup-x64-<версия>.exe` + `INSTALL.md`)

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
- Clarified the distribution model in docs: `ScorpioPipe-Setup-x64-<версия>.exe` is produced by the Windows build (GitHub Actions / local Windows build), and is not present in source archives.
