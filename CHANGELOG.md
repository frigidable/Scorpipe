## 8.0.1

- feat(p0-semantics): froze calibration-stack semantics by synchronizing `contracts/calibration.py`, calibrated-product validators, and spec docs so 1D/2D status claims are explicit and machine-checked. In particular, 2D calibrated products may no longer claim `ABSOLUTE_TRUSTED`.
- feat(p0-metadata): tightened `FrameMeta` provenance handling with explicit provenance helpers (`header` / `derived` / `fallback` / `missing` / `untracked`) and made standard association / sensfunc logic downgrade trust when atmospheric metadata is weak instead of silently overclaiming.
- feat(p0-reference-context): strengthened reference-context metadata for standards / extinction / telluric resources with explicit logical family, unit family, interpolation assumptions, source names, checksum semantics, and richer lineage surfaced into calibration QC.
- feat(p0-calibration-systematics): promoted `calibration_systematics.fits` to a stricter first-class contract with semantic headers plus explicit `CALSYS_TOTAL` quadrature bookkeeping, and carried separated calibration uncertainty summaries into downstream science outputs/reports.
- feat(p0-api-catalog-sync): fixed observer-layer drift by repairing `pipeline.public_api` run-root resolution, registering `science_lines_preview_png` in stage/product catalogs, broadening observer cards, and hardening sync tests across catalog/public-api/dashboard behavior.
- test(p0-hardening): added regression coverage for calibration status legality, provenance-aware standard-association downgrade, reference metadata semantics, and catalog/public-api consistency.
- docs(p0): updated `FLUX_CALIBRATION_CONTRACT.md`, `CALIBRATION_STACK_v1.md`, and `SCIENCE_LAYER_v1.md` to match the hardened v7.70 behavior.

## 7.65.0

- feat(science-layer): implemented P3.1 by adding a dedicated `science/` domain (`lines.py`, `reddening.py`, `kinematics.py`, `report.py`) so downstream science interpretation is no longer folded into reduction/calibration stages.
- feat(line-measure): implemented P3.2 with a real `line_measure` stage that measures calibrated 1D lines, propagates both `err_stat` and `err_cal`, writes machine-readable CSV/JSON products, and emits a preview plot.
- feat(science-report): implemented P3.3 with a real `science_report` stage that renders compact observer-facing Markdown/HTML summaries from calibrated products, QC, measured lines, and provenance.
- feat(observer-api): implemented P3.4 by extending `pipeline.public_api` with UI-safe summaries for sensfunc / telluric / fluxcal / science report, plus observer dashboard builders and recipe runners. Added lightweight `ui/observer_cards.py` helpers for viewer-side rendering.
- feat(long-comparison): implemented P3.5 by adding a formal comparison harness under `src/scorpio_pipe/compare/longslit.py`, separating metric-bundle comparison against LONG from the reduction stages themselves.
- test(observer-layer): implemented P3.6 with new regression coverage for line-measurement uncertainty propagation, report rendering, recipe definitions, public dashboard summaries, and LONG comparison metrics.
- feat(recipes): implemented P3.7 with recipe-level workflows (`quicklook_science`, `full_reduction`, `standard_star_reduction`, `relative_flux_calibration`, `full_science_calibration_report`) under `pipeline/recipes.py`.
- docs(observer-layer): updated architecture notes and added `SCIENCE_LAYER_v1.md` plus `LONG_COMPARISON_HARNESS_v1.md`.

## 7.60.0

- feat(sensfunc): implemented P2.1 with a first stable response-model builder in `calib/sensfunc.py`, including wavelength-convention normalization, extinction correction, conservative feature/telluric masking, smooth response fitting, QC JSON, and preview output.
- feat(telluric): implemented P2.2 with an explicitly separate telluric layer in `calib/telluric.py` / `stages/telluric_model.py`, producing inspectable transmission, confidence, QC, and preview products.
- feat(fluxcal): implemented P2.3 by replacing the `fluxcal_apply` placeholder with a real application stage that reads sensfunc/telluric products, assigns semantic calibration status, and writes calibrated 1D/2D outputs plus preview/QC artifacts.
- feat(calibration-systematics): implemented P2.4 by writing `calibration_systematics.fits` as a companion error layer with separate relative / telluric / absolute systematic components instead of hiding them inside statistical variance.
- feat(validators): implemented P2.5 by adding contract validators for sensfunc products, telluric products, and calibration-systematics outputs, alongside catalog wiring for the new preview/QC artifacts.
- test(calibration-stack): implemented P2.6 with new synthetic / contract coverage for wavelength-convention round-trips, telluric-window behavior, semantic status assignment, and calibration-stack FITS validators.
- docs(calibration-stack): updated architecture docs and added `docs/spec/CALIBRATION_STACK_v1.md` to describe the first stable P2 stack.

## 7.55.0

- feat(standard-prepare): implemented P1.1 by adding a real `standard_prepare` stage that selects standard / telluric-standard frames, validates science association, prepares a rectified standard input, and emits `standard_1d.fits`, `standard_trace.json`, `standard_qc.json`, and a machine-readable preparation summary.
- refactor(extract-core): implemented P1.2 by introducing a shared extraction facade under `scorpio_pipe.extract` so the new calibration workflow reuses the same trace / aperture / boxcar mechanics as science `extract1d` instead of forking a copy-paste branch.
- feat(standard-policy): implemented P1.3 by making wide-box extraction the default primary standard product, with optional optimal extraction kept diagnostic-only and explicit aperture-growth / flux-capture diagnostics recorded in QC.
- feat(standard-qc): implemented P1.4 by adding standard-specific diagnostics for association status, centroid drift, masked fraction, edge truncation, contamination proxy, trace smoothness, and a `safe_for` classification (`response_only`, `approx_absolute`, `absolute_candidate`, etc.).
- feat(public-api): implemented P1.5 by exposing `run_standard_prepare(...)` and `read_standard_prepare_summary(...)` through `pipeline.public_api`, plus UI-safe dataclasses for selected-standard summaries.
- docs(spec): added `docs/spec/STANDARD_PREPARE_CONTRACT.md` and updated `docs/ARCHITECTURE.md` to document the flux-first standard workflow and its public boundary.
- test(p1): added regression coverage for standard aperture policy helpers and for the new public-API summary loader.

## 7.50.0

- docs(architecture): implemented P0.8 by documenting the new calibration/science domain in `docs/ARCHITECTURE.md`, including the frozen boundaries, stage flow, catalog vocabulary, and semantic calibrated-output layer.
- docs(spec): added `docs/spec/FLUX_CALIBRATION_CONTRACT.md` as the normative contract for calibrated products, covering `CALSTAT`, `FLUXUNIT`, `REFCTID`, `SENSFUNC`, `TELLURIC`, `WAVEREF`, and flux-unit semantics.
- docs(contract-law): clarified that calibrated outputs are governed by explicit machine-readable semantics on top of the existing data-model contract, not by filenames or log conventions alone.

## 7.49.0

- feat(contracts): implemented P0.7 by adding calibrated-product semantic validation for `CALSTAT`, `FLUXUNIT`, `REFCTID`, `SENSFUNC`, and telluric-aware `TELLURIC` references.
- feat(units): tightened calibrated-output unit rules so flux-calibrated products must declare physical flux-like units and reject silent drift back to detector-count units (`electron`/`ADU`).
- test(contracts): added regression coverage for calibrated MEF/SPEC1D products, including rejection of telluric-aware outputs without `TELLURIC` and rejection of non-flux calibrated units.

## 7.48.0

- feat(product-catalog): implemented P0.6 by registering first-class calibration/science-domain products for standards, sensitivity functions, tellurics, flux-calibrated outputs, calibration systematics, line measurements, and science reports.
- feat(stage-catalog): wired the new stages to canonical product keys, including both current placeholder plan artifacts and future stable output paths.
- test(product-catalog): added regression coverage that the new calibration-domain product keys are present in `ProductCatalog`.

## 7.47.0

- feat(stage-catalog): implemented P0.5 by registering the new calibration/science-domain tasks in `StageCatalog`: `standard_prepare`, `sensfunc`, `telluric_model`, `fluxcal_apply`, `line_measure`, `science_report`.
- feat(workflow): extended the doit DAG so the new tasks participate in canonical catalog order and preserve explicit dependencies from `standard_prepare` through `science_report`.
- feat(entrypoints): added thin placeholder entrypoints for the new tasks so the catalog vocabulary is executable from day one without leaking ad-hoc task tables or legacy wrappers.

## 7.46.0

- feat(reference-context): implemented P0.4 by extending `refs/context.py` with first-class logical references for standards, extinction curves, telluric masks, and telluric templates.
- feat(resources): packaged baseline calibration-domain resources under `resources/standards`, `resources/extinction`, and `resources/telluric`, with normalized metadata (`units`, `wavelength_convention`, `sampling`) captured into `reference_context.json`.
- test(reference-context): added regression coverage that the new logical references are present and that changing a tracked standard table deterministically changes `context_id`.

## 7.45.0

- feat(calib-association): implemented P0.3 by replacing the placeholder association shim with a machine-readable standard/telluric compatibility policy (`INCOMPATIBLE`, `RELATIVE_RESPONSE_ONLY`, `RELATIVE_TELLURIC_ONLY`, `ABSOLUTE_APPROX`, `ABSOLUTE_TRUSTED`).
- feat(calib-association): added downgrade diagnostics for shape, slit-width, time, and airmass mismatches, so standard-star matching can explicitly distinguish response-only vs approximate/trusted absolute calibration use.
- test(calib-association): added regression coverage for trusted, approximate, telluric-only, and incompatible association outcomes.

## 7.44.0

- feat(metadata): implemented P0.2 by extending `FrameMeta` with calibration-grade fields (`airmass`, `zenith_distance_deg`, `parallactic_angle_deg`, `ra_deg`, `dec_deg`, `utc_mid_exposure`, `observatory`, `site_name`, `seeing_fwhm_arcsec`).
- feat(metadata-provenance): added explicit provenance for the new metadata fields (`header`, `derived`, `missing`) and conservative derivations for `utc_mid_exposure` and `zenith_distance_deg` when they can be computed from trustworthy inputs.
- test(metadata): added regression coverage for header-driven parsing, RA/DEC normalization, derived mid-exposure UTC, and preservation of missing calibration metadata as `None` without fake precision.

## 7.43.0

- feat(project-manifest): implemented P0.1 by extending canonical manifest roles with `STANDARD_FRAMES` and `TELLURIC_STANDARD_FRAMES`.
- feat(project-manifest-ui): manifest dialog now exposes dedicated GUI tabs for standards and telluric standards, so the new roles survive GUI/YAML round-trips.
- test(project-manifest): added regression coverage for defaults, legacy-shape coercion, canonical serialization, and `apply_project_manifest_to_cfg(...)` mapping to `frames.standard` / `frames.telluric_standard`.

## 7.42.11

- fix(engine/tests): align StageSpec contract-kind coverage with calibration contracts (`calib_bias_triplet`, `calib_flat`, `calib_neon`) and remove the stale regression in `test_p0_a_engine_unification`.
- perf(baselines): add versioned perf baselines under `benchmarks/baselines/{microbench,golden_time_budget}/` with `main.json` rolling track plus per-version snapshots.
- perf(ci): teach `scorpio_pipe.perf.microbench` and `scorpio_pipe.perf.golden_time_budget` to resolve versioned baselines and support advisory `--soft-fail` mode for PR checks.
- ci(perf): run perf workflows on PRs in warning-only mode and keep weekly/workflow_dispatch runs as hard regression gates.

## 7.42.10

- PERF-74.2: added reproducible runtime performance profiles via `runtime.performance_mode` with two supported modes: `full` (scientific maximum) and `fast` (rapid-look / reduced-iteration profile).
- New resolver `scorpio_pipe.perf.profiles.apply_performance_profile(...)` applies the chosen profile before planning, stage hashing, execution, provenance capture, and QC generation, so `fast -> full` reruns become explicit and deterministic.
- Fast mode now coarsens/shortens selected heavy-stage knobs (Wave2D tracing/iterations, Kelson sky iterations and flexure sampling, linearize/stack robust passes, cosmics iteration counts) and records every applied override in `runtime.performance_profile`.
- Provenance now captures the performance mode in `effective_config.yaml` and `manifest/run_signature.json`, while `qc_report.json` surfaces the selected mode/profile and emits a visible `QC_FAST_MODE_RUN` reminder for rapid-look runs.

## 7.42.9

- PERF-74.1: introduced stage-level execution budgets via `StageContext.resources` (`max_rss_mb`, `n_jobs`, `blas_threads`) with config resolution from `runtime`, top-level `resources`, and per-stage `resources` / `perf` overrides.
- Added `scorpio_pipe.pipeline.resources` to normalize budgets, cap BLAS/OpenMP threads for a stage, and translate tight RAM budgets into safer tiling/streaming knobs (`sky.perf.strip_rows/strip_workers`, `stack2d.chunk_rows`, `linearize/cosmics/superneon` job caps).
- Engine-created `StageContext` objects now carry the resolved budget, and stage metrics/timings include the applied resource-budget metadata for post-run diagnostics.

## 7.42.8

- PERF-72.2: promoted the FITS header/meta LRU cache to the single shared source for repeated scan-style callers, with a combined `get_header_and_frame_meta(...)` accessor keyed by `(path, mtime_ns, size)`.
- `inspect`, `preflight`, and `dataset builder` now fetch header fields and `FrameMeta` through one shared cached snapshot per file instead of separate header/meta lookup paths.
- Launcher/frame-browser setup inference now uses the same cache for representative FITS headers, so repeated work-dir/run suggestions on the same inspected night no longer reopen FITS headers.
- Added cache diagnostics (`hits/misses/disk_loads/evictions`) and regression coverage to ensure repeated `inspect/preflight/builder` passes stay on the in-memory metadata path.

## 7.42.7

- PERF-72.1: artifact index migrated to append-only `manifest/artifact_index.jsonl` (schema v2), so post-stage index updates are now O(1) appends instead of full snapshot rewrites.
- `load_artifact_index(...)` now reads the incremental JSONL format and still falls back to legacy `artifact_index.json` for backward compatibility; first write bootstraps JSONL from an existing legacy snapshot once.
- StageFramesWindow cache invalidation now tracks the canonical artifact-index storage path (`jsonl` preferred, legacy `json` fallback), preserving the fast UI path after the storage migration.

## 7.42.6

- PERF-71.2: freeze watchdog dumps are now full forensic snapshots: main-thread stack, active Scorpipe QRunnable jobs (`id/name/age`), queued-log backlog size (`lines/chars`), and the last 50 profiled UI actions plus currently in-flight profiled actions.
- Added heuristic “freeze reason hints” to the dump so common causes (matplotlib redraw, synchronous FITS I/O, filesystem walk, worker backlog, log flood) are surfaced immediately next to the stack trace.
- `slow_action_profiler` now keeps an in-memory ring buffer of recent UI actions for post-mortem debugging, and `qt_log` exposes backlog-size diagnostics for the watchdog.

## 7.42.5

- PERF-71.1: UI async worker now keeps a global job registry with pool/queue diagnostics (`running`, `queued`, `cancelled`, `stale`) exposed to the freeze watchdog snapshot.
- Added best-effort queue limiting (`SCORPIPE_UI_QUEUE_LIMIT`, default 200) with latest-wins deduplication and queued-job eviction to prevent click-storm degradation.
- Wired dedupe keys into the hottest UI producers (`Inspect`, Sky ROI, Manual Cosmics, FITS preview, Stage scan`) so repeated clicks collapse instead of flooding the pool queue.

## 7.42.1

- PERF-70.1: GUI `Inspect dataset` now runs via the shared async worker pool (`QThreadPool`) instead of synchronously in the click handler, so large nights no longer freeze the UI.
- Added live inspect feedback in the GUI (`Inspecting…`, found/opened FITS counter, current file) plus a user-cancel path wired through cooperative cancellation in `inspect_dataset(...)`.
- Added regression coverage for inspect progress/cancellation and an integration-style GUI test that verifies the launcher remains responsive while a mocked slow inspect job is running.

## 7.42.0

- PERF-60.2: golden weekly workflow now checks soft per-stage time budgets from `manifest/timings.jsonl` / `manifest/metrics.json` via `scorpio_pipe.perf.golden_time_budget`, and uploads a comparison JSON artifact.
- Golden mini-night artifacts now include `manifest/timings.jsonl` and `manifest/metrics.json` for time-budget comparison/debugging.

## 7.41.0

- PERF-60.1: added a lightweight microbench suite for hot kernels (`linearize_core`, `sky_fit_core`, `cosmics_core`, `stack_accumulator`) with JSON results, JSON baseline, and per-benchmark tolerance gates.
- Added weekly/manual GitHub Actions workflow `perf-microbench.yml` that runs the suite, compares against `benchmarks/microbench_baseline.json`, uploads artifacts, and fails on meaningful regressions.

## 7.40.0

- PERF-50.2: multiprocessing backends now force BLAS/OpenMP thread caps (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, etc.) during child-process spawn and again inside workers, preventing oversubscription like `16 processes × 16 BLAS threads`.
- The direct process-pool path in `sky_sub` now uses the same guard, so all current multiprocessing entry points inherit the same protection.

## 7.39.0

- PERF-50.1: added a shared stage-level `parallel_map(...)` adapter with `thread` / `process` / best-effort `loky` backends and conservative worker caps (`min(8, cpu_count)`).
- `linearize` now routes per-exposure rectification through the shared parallel helper (default backend: `thread`), so independent frames can be processed concurrently while preserving ordered outputs.
- `cosmics` now uses the same helper for per-frame output emission in `stack_mad` / `two_frame_diff`, removing stage-local ad-hoc loops and centralizing backend selection in `cosmics.perf`.

## 7.38.0

- PERF-40.5: accelerated Extract1D optimal extraction with vectorized per-column aperture math and optional numba backend (`extract1d.optimal.backend=numba` or `SCORPIPE_EXTRACT1D_OPT_BACKEND=numba`) with graceful NumPy fallback.

## 7.37.0

- PERF-40.4: Stack2D now uses streaming accumulation per Y-chunk instead of building full `(nexp, y, x)` cubes in RAM. The default mean / invvar / Huber paths keep only block-sized accumulators in memory, stabilizing RAM usage on long exposure series.
- Added a regression test guarding against reintroducing `np.stack`-based full-cube stacking in Stack2D.

## 7.36.0

- PERF-40.3: `cosmics` теперь умеет ограничивать детекцию ROI (`cosmics.roi`) и оставляет остальной кадр нетронутым, что резко уменьшает объём вычислений на больших сериях.
- Для масок космиков добавлен fast-path dilation через `scipy.ndimage.binary_dilation` (с автоматическим fallback на legacy NumPy-реализацию), а ROI и `mode_profile` теперь отражаются в `summary.json` / `qc.json`.

## 7.35.0

- PERF-40.2: sky subtraction full-frame model evaluation is now tiled by Y-strips with best-effort process-pool parallelism and per-strip basis reuse for large frames.

## [7.34.0] - 2026-03-06

### Changed
- PERF-40.1: `linearize` теперь строит и LRU-кеширует resampling/mapping grid по `(FrameSignature, lambda_map_sha256, shape, wave0, dw, nlam)` и переиспользует его на сериях кадров одной конфигурации вместо повторного пересчёта overlap-grid на каждом кадре.
- В `linearize` добавлена диагностика `payload["resampling"]["grid_cache"]` (`enabled/hit/frame_signature/lambda_map_key`) для воспроизводимого контроля попаданий в кеш.


## [7.33.0] - 2026-03-06

### Changed
- PERF-30.3: добавлен LRU-кеш FITS header/meta по ключу `(path, mtime_ns, size)`; `preflight`, `inspect/frame browser` и `dataset builder` теперь переиспользуют кэш и не открывают одни и те же FITS тысячи раз при повторных операциях по одной ночи.


## [7.32.0] - 2026-03-06

### Changed
- PERF-30.2: `manifest/metrics.json` теперь пишется атомарно через `atomic_write_json` вместо прямого `write_text`, поэтому файл не остаётся частично записанным при crash/kill и UI не натыкается на битый JSON.


## [7.31.0] - 2026-03-06

### Changed
- PERF-30.1: `manifest/timings.json` переведён в append-only формат `manifest/timings.jsonl` (одна запись = одна строка). Запись делается через безопасный append (flush+fsync best-effort), поэтому тайминги не замедляют пайплайн даже на длинных сессиях. Чтение поддерживает legacy `timings.json` как fallback.


## [7.30.0] - 2026-03-05

### Changed
- PERF-20.2: UI — StageFramesWindow и OutputsPanel теперь предпочитают `manifest/artifact_index.json` для списков файлов/артефактов и не запускают дисковый обход/глоб-поиск при кликах по поздним стадиям. `scan_tree` остаётся только fallback, когда индекс отсутствует.


## [7.29.0] - 2026-03-05

### Changed
- PERF-20.1: Engine теперь пишет `manifest/artifact_index.json` после каждой стадии (индекс файлов по продуктам: `files[]` + `keyed{product_key:path}`), чтобы UI не делал `os.walk` по stage_dir.
- StageFramesWindow теперь предпочитает `artifact_index.json` для вкладок со списком файлов и использует скан по диску только как fallback (если индекс отсутствует/не содержит нужной стадии).


## [7.28.0] - 2026-03-05

### Changed
- PERF-10.5: UI log backpressure — при большом backlog очередь начинает дропать DEBUG (а при экстремальном backlog — INFO), и периодически вставляет схлопнутую строку вида “N lines skipped”.
- Добавлен UI-фильтр уровня логов (INFO/WARN/ERROR) через контекстное меню log панели (фильтр также поднимает level UI-handler, чтобы снизить нагрузку).


## [7.27.0] - 2026-02-24

### Changed
- PERF-10.4: UI — добавлен UiRunCache (plan_sequence + stage_status/done.json snapshot) с инвалидацией по mtime `manifest/stage_state.json` и `manifest/metrics.json`. Run Plan dialog теперь использует кеш и не пересчитывает план заново на каждый клик.


## [7.26.0] - 2026-02-24

### Changed
- PERF-10.3: FITS preview — добавлен “quick peek” (header + strided sampling через `HDU.section`, без чтения всего SCI) и LRU-кеш превью по ключу `(path, mtime, size)` (файл `manifest/ui_perf.jsonl` по-прежнему помогает ловить медленные apply/render).


## [7.25.0] - 2026-02-24

### Changed
- PERF-10.2: StageFramesWindow — scan jobs теперь дедуплицированы “по вкладке”: хранится один job на tab (tab->job), при refresh/close jobs отменяются и отписываются (disconnect), после завершения ссылки удаляются для корректного GC.


## [7.24.0] - 2026-02-24

### Changed
- PERF-10.1: UI async_worker переведён с модели “QThread per job” на `QThreadPool.globalInstance()` + `QRunnable` с лимитом потоков `min(8, cpu_count)` (override: `SCORPIPE_UI_MAX_THREADS`).
- Добавлены `job_id` и best-effort `cancel()/mark_stale()` (stale jobs не эмитят сигналы; если ещё не стартовали — выходят сразу).
- StageFramesWindow больше не копит хэндлы сканов: предыдущие in-flight scans помечаются stale при refresh/close, хэндлы очищаются по завершению.


## [7.23.0] - 2026-02-24

### Changed
- PERF-00.3: добавлен UI slow-action profiler `profile_ui_action(name)` — логирует всё, что >200ms (duration + stack) и пишет JSONL в `manifest/ui_perf.jsonl`.
- Профилирование включено на ключевых UI-действиях: открытие StageFramesWindow, загрузка FITS preview (apply/render), открытие QC Viewer и рендер QC-артефактов, Outputs (refresh), расчёт Run plan (refresh).

## [7.22.0] - 2026-02-24

### Changed
- PERF-00.2: добавлен UI Freeze Watchdog (в debug/advanced режиме). При зависании event loop (>~1.25s без heartbeat) пишет `ui_freeze_dump.txt` в папку логов (stack главного треда + stacks всех Python threads + snapshot активных QThreads + backlog UI-логов).
- Watchdog включён по умолчанию в dev (не frozen) режиме; в frozen builds включается через env: `SCORPIPE_UI_WATCHDOG=1` (или `SCORPIPE_UI_DEBUG=1`) / CLI: `--watchdog`.

## [7.21.0] - 2026-02-24

### Changed
- PERF-00.1: добавлены per-stage resource metrics (CPU/RSS/IO) через psutil: пишутся в `manifest/timings.json` (через `extra`, начиная с 7.31.0 — `manifest/timings.jsonl`) и в `manifest/metrics.json` (через `qc.metrics_store.update_after_stage`).
- Engine теперь пишет timing-строки также для skipped/blocked/cancelled стадий (0s, с флагами в `extra`).

## [7.20.0] - 2026-02-23

### Changed
- P0-ARCH-11: добавлен weekly/manual workflow `.github/workflows/golden.yml` (pytest -m golden) с публикацией артефактов mini-night (qc_report + ключевые продукты).
- CI (pull_request) ускорен: fast тесты исключают `golden` (pytest -m "not smoke and not golden"); pytest marker `golden` зарегистрирован.
- Golden E2E тест теперь опционально сохраняет артефакты в `SCORPIPE_CI_ARTIFACTS_DIR` (для upload-artifact в golden workflow).

## [7.19.0] - 2026-02-23

### Changed
- P0-ARCH-10: добавлен dependency lock (requirements.lock) + sha256 в run manifest (manifest.json: dependency_lock {path, sha256, tool, created_utc}).
- Добавлен полный deps snapshot `manifest/deps_snapshot.json` (importlib.metadata; name/version) и включён в export science package.
- ProductCatalog/StageCatalog расширены продуктами `dependency_lock` и `deps_snapshot` для boundary enforcement.

## [7.18.0] - 2026-02-23

### Changed
- P0-ARCH-09: введены строгие контракт-kind для калибровок: `calib_bias_triplet`, `calib_flat`, `calib_neon` (StageCatalog + engine boundary enforcement).
- Добавлены stage-level валидаторы в `scorpio_pipe.contracts.validators`: ловят битые калибровки на своей стадии с понятными кодами (например `bias_var_shape_mismatch`, `flat_has_nan`, `neon_empty`).
- Исправлен ProductCatalog: `master_bias.fits` больше не ошибочно валидируется как MEF (`required_headers_contract` → `none`), так как superbias использует отдельные файлы SCI/VAR/DQ.

## [7.17.0] - 2026-02-23

### Changed
- P0-ARCH-08: workflow/dodo.py теперь дериватив StageCatalog: канонический порядок задач берётся из каталога и защищает инвариант wavesolution → sky → linearize (исправлена инверсия SKY↔LINEARIZE в doit DAG; stack2d зависит от linearize).
- Добавлен архитектурный тест `tests/test_workflow_matches_catalog.py`: doit-граф не может противоречить StageCatalog order и ловит регрессии в PR.

## [7.16.0] - 2026-02-21

### Changed
- P0-ARCH-07: стадии больше не вычисляют `stage_dir` из `work_dir` для записи: `sky/flatfield/cosmics/linearize/stack2d/extract1d` требуют явный `out_dir` (engine передаёт `ctx.stage_dir`).
- Убраны stage-level "fail done.json" guards из `sky_sub.run_sky_sub`, `extract1d.run_extract1d`, `linearize.run_linearize`, `stack2d.run_stack2d`: при исключении источник истины — `engine._upsert_stage_done_json`.
- `flatfield` больше не автособирает `superbias/superflat` (и не пишет в чужие stage dirs): теперь жёстко требует продукты от стадий `superbias` и `superflat`.
- `build_superbias/build_superflat` принимают `stage_dir=...` (обычно `ctx.stage_dir`) и пишут *_done.json маркеры строго в него; legacy-mirrors выполняются только при явном `allow_legacy_mirrors=True`.
- Убраны legacy-write копии продуктов из стадий (например в `work_dir/lin` и `work_dir/products/spec`).

## [7.15.0] - 2026-02-21

### Changed
- P0-ARCH-06.2: добавлен каркас Golden-nights: `tests/golden/mini_night.py` + E2E тест `tests/test_golden_mini_night_e2e.py`.
- Sky stage теперь резолвит `lambda_map.fits` через `wavesol_paths.wavesol_dir()` (поддержка disperser-aware `08_wavesol/<slug>/...`).

## [7.14.0] - 2026-02-21

### Changed
- P0-ARCH-06.1: добавлен CI-enforcement архитектурных импорт-правил через `grimp` + pytest тест `tests/test_arch_no_forbidden_imports.py`.
- Введён файл правил `tools/arch_imports.yml`.
- UI больше не импортирует `scorpio_pipe.stages.*` напрямую: добавлен стабильный `scorpio_pipe.pipeline.public_api` (lazy wrappers).
- pipeline больше не импортирует `scorpio_pipe.ui.*`: session-store перенесён в `scorpio_pipe.pipeline.session_store` (UI путь оставлен как re-export).

## [7.13.0] - 2026-02-21

### Changed
- P0-ARCH-05.1: введён порт `InstrumentAdapter` (`scorpio_pipe.ports.instrument`) и lazy-реестр инструментов (`scorpio_pipe.catalog.instruments`) с адаптерами для SCORPIO/SCORPIO-2.
- Header parsing теперь диспатчится через реестр адаптеров (без прямых SCORPIO-ветвлений в preflight/metadata). Добавление нового инструмента = новый адаптер + запись в реестр.

## [7.12.0] - 2026-02-21

### Changed
- P0-ARCH-04.2: формализованы стабильные top-level namespaces конфигурации: добавлены `io` и `qc` (в pydantic-схеме), а также правило "namespace discipline".
- `find_unknown_keys()` больше не эвристика по вложенным stage-параметрам: теперь unknown-key check выводится из pydantic-схемы и работает как архитектурный страж (строгий top-level + строгие `io`/`qc`).
- Валидация поддерживает обратную совместимость: legacy path-ключи (`work_dir`, `data_dir`, `resources_dir`, `dataset_manifest_path`, ...) зеркалируются в `io.*` для TypedConfig.

## [7.11.0] - 2026-02-21

### Changed
- P0-ARCH-04.1: введён единый typed-конфиг (pydantic) поверх текущего dict: модуль `scorpio_pipe.typed_config` + `config.load_typed_config(...)` / `load_typed_config_any(...)` возвращают `(TypedConfig, raw_dict)`.
- Добавлен обязательный pre-run validation pipeline (engine gate): `schema_validate` + `mode_profile_validate` + `contracts_precheck` выполняются в `pipeline.engine.run_sequence()` *до* запуска стадий.
- Критические enum-ошибки (например `project.wave_reference=UNKNOWN`) теперь ловятся fail-fast с человеческим сообщением (CLI и GUI).

## [7.10.0] - 2026-02-21

### Changed
- P0-ARCH-03.3: введена retention policy как часть Workspace: `Workspace.cleanup(policy)` и реализация `FSWorkspace.cleanup(...)` с отчётом `manifest/cleanup_report.json`.
- Добавлены политики: `all` (не чистить), `canonical` (оставить canonical publish-продукты из ProductCatalog (без коллекций) + qc + provenance), `none` (агрессивная чистка тяжёлых промежуточных по whitelist).
- CLI: добавлен `scorpio-pipe cleanup` (для любого существующего run_root) и опции `scorpio-pipe run --retain {all,canonical,none}` / `--cleanup` (shortcut на canonical).


## [7.9.0] - 2026-02-21

### Changed
- P0-ARCH-03.2: введена атомарная запись FITS: `scorpio_pipe.io.atomic_fits.atomic_write_fits` (temp в той же директории + `os.replace`).
- Все записи FITS в `io/mef.py`, стадиях и GUI-инструментах переведены с `fits.writeto(..., overwrite=True)` / `.writeto(..., overwrite=True)` на `atomic_write_fits`, чтобы kill/краш не оставлял полу-FITS и не ломал resume.

## [7.8.0] - 2026-02-21

### Changed
- P0-ARCH-03.1: введён порт Workspace (:mod:`scorpio_pipe.ports.workspace`) и файловый адаптер FSWorkspace (:mod:`scorpio_pipe.adapters.workspace_fs`).
- Реализована гарантия "single writer" через `run.lock` в `run_root`: CLI `cmd_run` и GUI runner теперь берут lock перед записью результатов и снимают его в `finally`.
- Добавлена диагностика lock-файла (pid/host/time/user/argv) и явный CLI override: `scorpio-pipe run --override-lock` (UNSAFE).
- Исправлена потенциальная циклическая зависимость импорта: `scorpio_pipe.catalog.__init__` теперь использует lazy `__getattr__` и не тянет `products`/`workspace_paths` на старте.

## [7.7.0] - 2026-02-21

### Changed
- P0-ARCH-02.2: StageCatalog теперь хранит `entrypoint` (строка вида `module:callable`) для каждой pipeline-задачи.
- Engine больше не содержит обязательные `_task_*`-прослойки: исполнение идёт через lazy-resolve entrypoints (`importlib`) с кешированием.
- Добавлены lightweight entrypoint-модули для стадий и сервисных задач (manifest/qc_report/navigator) для ускорения старта GUI/CLI и снижения связности.

## [7.6.0] - 2026-02-21

### Changed
- P0-ARCH-02.1: введены `StageContext` и `StageResult` (:mod:`scorpio_pipe.pipeline.context`).
- Engine теперь создаёт `StageContext` и передаёт его в задачи (через публичный контракт stage entrypoint `run(ctx) -> StageResult`).
- Калибровочные стадии, которые ранее игнорировали `out_dir` (`superbias`, `superflat`, `superneon`, `wavesolution`), теперь выполняются в *engine-controlled* `run_root` (больше нет скрытого `work_dir` внутри стадии).
- `wavesol_dir()` и `ProductCatalog` получили параметр `run_root` для корректного резолва путей при временных workspace и UI/CLI overrides.
- `build_superbias/build_superflat` поддерживают `allow_legacy_mirrors=False` для строгих границ I/O (без записи в `calibs/` и прочие legacy-локации).

### Tests
- Добавлен тест на возможность вызвать stage entrypoint с временным workspace (без зависимости от cfg.work_dir).

## [7.5.0] - 2026-02-20

### Changed
- P0-ARCH-01.2: введён единый источник истины по продуктам: :mod:`scorpio_pipe.catalog.products` (ProductCatalog + ProductSpec).
- `products.py` переведён в режим deprecated shim и читает продукты из ProductCatalog.
- `stage_contracts.validate_contracts()` валидирует outputs через ProductCatalog (а не через ad-hoc list_products()).
- `qc_report` и `export_package` разрешают пути продуктов через ProductCatalog (включая disperser-aware wavesol subdir).
- Граница стадий: engine валидирует *ровно* список `(stage_key, product_keys)` из StageCatalog через ProductCatalog.
- StageCatalog расширен ключевыми коллекционными продуктами для строгой валидации (flatfield/cosmics/sky/linearize/stack2d).

### Tests
- Добавлены тесты ProductCatalog: покрытие ключей + экспорт пакета с wavesol subdir.

## [7.4.0] - 2026-02-20

### Changed
- P0-ARCH-01.1: введён единый источник истины по стадиям: :mod:`scorpio_pipe.catalog.stages` (StageCatalog).
- `stage_contracts` теперь полностью генерируется из StageCatalog (включая legacy aliases).
- Engine `TASKS` расширяется алиасами из StageCatalog (исключает дрейф; добавлен legacy алиас `wavesolution2d`).
- GUI (ui/pipeline_runner) получает список стадий/спеки напрямую из StageCatalog.
- CLI валидирует и канонизирует список задач через StageCatalog.

### Fixed
- CLI: исправлена критическая ошибка отступов в `cmd_run` (autoconfig).

### Tests
- Добавлен регрессионный тест на `wavesolution2d` алиас (StageCatalog + stage_registry).

## [7.0.0] - 2026-02-20

### Added
- P0-A1: preflight ядро: детерминированный ранний скан raw-night директории.
  Новые артефакты в run_root:
  - `manifest/night_profile.json`
  - `manifest/preflight_report.json` (OK/WARN/ERROR + причины)
  - `quarantine/*.json` + `manifest/quarantine_index.json`
  Preflight использует строгий контракт ключей (strict=True) для обязательных
  метаданных и помечает мягкие/опциональные проблемы как WARN.

### Changed
- `manifest` task теперь запускает preflight автоматически перед записью `manifest/manifest.json`.
- QC viewer (GUI) отображает новые JSON-артефакты preflight.

### Tests
- Добавлен интеграционный тест на детерминизм preflight JSON.

## [6.0.28-pre1] - 2026-01-30

### Fixed
- P0-B7: engine boundary validation now matches the real sky preview product name (`preview.fits`).

### Changed
- P0-B10: reference_context no longer hardcodes a historical lamp default ("Ne"); it uses `lamp_contract.resolve_lamp_type()` so the long-slit default is HeNeAr.
- P0-B10: reference_context additionally tracks all built-in package reference resources (CSV/YAML/PDF/JSON) excluding documentation-only files, so any shipped-resource change updates `context_id` and triggers reruns.
- P0-B11: linearize forbids legacy directory search for `lambda_map*.fits` when a validated `dataset_manifest.json` is present; legacy discovery remains available only without a manifest, with a WARN surfaced in QC.

### Tests
- Add regression test for sky preview glob coverage.
- Add regression test enforcing "no directory search" for linearize when `dataset_manifest.json` exists.

## [6.0.27] - 2026-01-19

### Added
- P0-A8: documentation “law” for execution semantics: `docs/EXECUTION_CONTRACT.md` (resume/force, QC gate, done.json, stage_state).

### Changed
- P0-A7: `STAGE_SPECS` now declares outputs for **all** canonical tasks (100% coverage) and includes optional `output_keys` aligned with `scorpio_pipe.products`.
- CI test now enforces `set(CANONICAL_TASKS) == set(STAGE_SPECS)` and requires non-empty `StageSpec.validate_globs` for every canonical task.

## [6.0.26] - 2026-01-18

### Added
- P0-B6: unit tests for product contract validator (missing HDU, dtype, variance sanity, shape mismatch, mandatory header keys).
- P0-B6: variance propagation unit tests (multiply/scale, divide incl. divisor-variance, weighted-mean stacking, contract cleanup).
- P0-B6: FrameMeta policy unit tests (required missing key error, optional-missing recorded, fallback provenance recorded).
- P0-B6: dataset_manifest deterministic-JSON integration test (input order independent).
- P0-B6: QC report now includes a non-fatal `compliance` block summarizing core contract wiring (data model version, stage-contract registry, reference_context presence, spot product validations).

### Changed
- Dataset manifest builder now sorts input records and pool/frame_index ordering to make `dataset_manifest.json` deterministic independent of filesystem scan order.
- Smoke end-to-end test now creates a minimal `reference_context.json`, writes a contract-compliant `lambda_map.fits`, and asserts `SCORPDMV` + `compliance`/`reference_context` presence in the QC report.

## [6.0.25] - 2026-01-18

### Fixed
- Flatfield stage now guarantees a MASK plane in ``*_flat.fits`` MEF products even when upstream inputs lacked any DQ/mask information (writes an all-zero mask).

### Added
- P0-B5 (stage 2): engine boundary-contract enforcement expanded to MEF products from `flatfield`, `sky`, and `linearize` using explicit `STAGE_SPECS.validate_globs`.
- `StageSpec.contract_kind` now supports `lambda_map`, allowing `wavesolution` contract validation to be expressed declaratively in `STAGE_SPECS`.

### Changed
- Engine boundary-contract validation is now driven by `STAGE_SPECS` (less duplication; easier to expand coverage safely).

### Tests
- Add regression test asserting `flatfield.apply_flat()` always writes a MEF with a `MASK` extension.

## [6.0.24] - 2026-01-18

### Fixed
- P0-B5 (stage 1): engine boundary-contract validation now runs against the actual stage output directory (``out_dir``), so contract violations are no longer silently skipped.

### Added
- P0-B5 (stage 1): new contract validator for wavesolution ``lambda_map.fits`` (minimal header+data requirements) and engine enforcement after `wavesolution`.

### Changed
- P0-B5 (stage 1): boundary-contract enforcement is intentionally limited to high-value product stages: `stack2d`, `extract1d`, and `wavesolution` (coverage expands in later migration stages).
- Dataset frame classifier now imports the canonical `FrameMeta` from `scorpio_pipe.metadata` (instrument shim remains for backward compatibility).

### Tests
- Add regression tests for `lambda_map.fits` contract validation and for engine boundary validation operating on `out_dir`.

## [6.0.23] - 2026-01-18

### Added
- P0-B4: dataset_manifest schema v3 (`scorpio-pipe.dataset-manifest.v3`) now records per-match `match_reason` and `qc_deltas` for calibration associations.
- P0-B4: new manifest validator (`scorpio_pipe.calib.manifest_schema`) with strict schema/version checks; CLI `dataset-manifest` validates the on-disk JSON after writing.
- P0-B4: QC report block `calibration_associations` summarizing selected calibrations per science_set, suboptimal matches, and warning counts.

### Changed
- P0-B4: dataset-manifest builder emits `CALIB_SUBOPTIMAL_MATCH` warnings when a physically compatible but QC-suboptimal calibration is selected (e.g., readout mismatch allowed, spectral-range mismatch, time-window fallback).
- P0-B4: wavesolution and superneon stages prefer calibration associations from dataset_manifest when present; they do not perform any directory search.

### Tests
- Add regression tests asserting `CALIB_SUBOPTIMAL_MATCH` + `match_reason/qc_deltas` for suboptimal flat/arc readout mismatch selections.

## [6.0.22] - 2026-01-18

### Added
- P0-B3: CRDS-lite reference context (`manifest/reference_context.json`) capturing resolved reference resources (linelists/atlases/...) with SHA256 hashes and a stable `context_id`.
- P0-B3: `context_id` is recorded into per-stage `done.json` and surfaced in the QC report (JSON + HTML).

### Changed
- Stage hashing now includes `reference_context_id`, so changes in any tracked reference force a re-run of affected stages.

## [6.0.21] - 2026-01-18

### Added
- P0-B2: metadata contract layer (`scorpio_pipe.metadata`) now provides the canonical `FrameMeta` with per-field `meta_provenance`, `meta_missing_optional`, and `meta_fallback_used`.
- P0-B2: `FrameMeta.from_header(...)` constructor supporting missing-key policy and explicit fallback sources.

### Changed
- Dataset manifest scanning uses only `FrameMeta` for normalized keys (including spectral range) and emits QA warnings when optional keys are missing or fallbacks are used.
- Calibration compatibility QC (GAIN/RATE/RDNOISE) prefers normalized values from `FrameMeta` instead of ad-hoc raw header lookups.
- `scorpio_pipe.calib.compat` is fixed as a stable import path and stages now import compat checks from it.

## [6.0.20] - 2026-01-18

### Added
- P0-B: introduced `scorpio_pipe/contracts` (data model, units, variance, maskbits, validators) as a stable, testable contract layer.
- P0-B: introduced stable import shims `scorpio_pipe.metadata`, `scorpio_pipe.refs`, and `scorpio_pipe.calib` (thin wrappers over existing implementations).

### Changed
- MEF writer now stamps `SCORPDMV=1` and sets per-extension `BUNIT` on `SCI` and the corresponding squared `BUNIT` on `VAR` (with auto-inference from legacy `SCORPUM`).
- `extract1d` now stamps `SCORPDMV` into `spec1d.fits` primary header.
- Pipeline runner maps product contract violations to `done.json` `error_code=CONTRACT`.

### Tests
- Updated boundary contract tests to assert P0-B1 stamps (`SCORPDMV`, per-extension `BUNIT`) and to match the new noise-provenance validation order.

## [6.0.19] - 2026-01-18

### Fixed
- GUI startup hang on Windows ("Загрузка интерфейса..." forever): main window pages are now constructed lazily (on-demand) instead of all at once during startup, preventing long freezes / apparent hangs in frozen builds.
- GUI startup robustness: `launcher_app` now shows a crash dialog with full traceback and the GUI log path if the main window fails to import/construct.

## [6.0.18] - 2026-01-18

### Added
- P0-A: core execution engine moved to `scorpio_pipe.pipeline.engine` as the single source of truth for stage execution semantics (resume/force hashing, QC gate blocking, boundary contract validation, done.json/stage_state/provenance updates).
- P0-A: `workflow/dodo.py` is now a pure adapter: each doit task delegates execution to the engine, eliminating UI/CLI/doit divergence.

### Changed
- `scorpio_pipe.ui.pipeline_runner` is now a backward-compatible shim re-exporting the core engine API plus a small set of GUI helpers.
- CLI execution (`scorpio-pipe run --execute`) now imports the engine directly.

### Fixed
- Runner provenance: `done.json` upsert no longer references an undefined `effective_config` variable.
- `done.json` now records the expected `stage_hash` even for `skipped/blocked/cancelled` statuses (best-effort), improving traceability.

### Tests
- Add regression tests ensuring `workflow/dodo.py` does not execute stages directly and that it produces engine-standard artifacts (`manifest/stage_state.json`, per-task `done.json`).

## [6.0.17] - 2026-01-17

### Fixed
- Windows installer / PyInstaller GUI startup: show a splash screen and defer heavy imports until after Qt is initialized, avoiding the "process in Task Manager but no window" symptom on slow machines / first start.
- GUI startup diagnostics: timestamped progress logs in per-user `scorpipe_gui.log` and periodic faulthandler stack dumps during startup (automatically cancelled once the main window is shown).
- Frozen build safety: added `multiprocessing.freeze_support()` to GUI entrypoints to prevent rare spawn recursion / deadlocks in packaged builds.

## [6.0.16] - 2026-01-17

### Fixed
- `sky_sub`: removed a local `make_flag` import that shadowed the module-level symbol and could raise `UnboundLocalError` in some failure paths.
- `extract1d`: ensured upstream degradation markers (`QADEGRD`, `SKYOK`, `SKYMD`, `PROV*`) propagate into `spec1d.fits` primary header (regression in P0-L test).
- `sky_sub` geometry config: treat numeric 0/0.0 as valid values (e.g. `thresh_sigma: 0.0`) instead of falling back to defaults.
- Tests: `P0-J` pass-through assertion now compares against the float32-rounded values actually written to the input FITS.

## [6.0.15] - 2026-01-17

### Fixed
- `stack2d`: fixed `NameError: _get_ext` by using the local MEF helper `_get_mask_data()` (with a safe zero-mask fallback for legacy/synthetic inputs).

## [6.0.14] - 2026-01-17

### Fixed
- `sky_sub`: removed a local `fits` import that shadowed the module-level `astropy.io.fits` and could raise `UnboundLocalError` during preview writing (smoke chain regression).

## [6.0.13] - 2026-01-16

### Added
- P0-K: **Absolute exclude** now applies to the dataset-manifest builder: if `data_dir/project_manifest.yaml` exists, its `exclude` is applied **before** building pools/matches; excluded frames are recorded in `dataset_manifest.json` as `excluded_summary`.
- P0-K: stages that consume `dataset_manifest.json` (e.g. flatfield) apply a final safety filter by `exclude` and emit `MANIFEST_EXCLUDE_APPLIED` (WARN) if anything is removed.
- P0-L: downstream stages propagate sky-degradation: `stack2d` and `extract1d` carry `QADEGRD=1` when upstream sky pass-through happened and add `UPSTREAM_SKY_PASSTHROUGH` (WARN) to their `done.json`.
- P0-L: `stack2d.reject_if_mask_bits` (default empty) allows operator-controlled rejection, e.g. `[SKYMODEL_FAIL]`.
- P0-M: lamp contract for wavelength calibration: explicit `lamp_type` (`HeNeAr|Ne|Unknown`) with provenance (`config|header|default|none`), defaulting to **HeNeAr** for long-slit SCORPIO/SCORPIO-2; the chosen line-list is recorded in wavesolution `done.json`.
- P0-N: wavesolution validates configuration **per setup** (setup_signature). Multiple distinct setups in one run are rejected in strict mode with a clear explanation.

### Changed
- P0-M: `OBJECT=NEON` is treated as a weak label (not sufficient evidence for `lamp_type=Ne`), preventing silent HeNeAr-vs-Ne systematics.

### Fixed
- P0-M: align lamp/linelist helper API used by `wavesolution` (`resolve_lamp_type(..., arc_path=..., instrument_hint=...)` and `resolve_linelist_csv_path`).
- P0-L: QC report now emits a high-visibility warning alert `QC_SKY_SUB_SKIPPED` when any stage reports sky pass-through.

### Tests
- Add regression tests for lamp contract defaulting and for extract1d propagation of upstream sky pass-through.

## [6.0.12] - 2026-01-16

### Added
- P0-J: `sky.failure_policy: soft|strict` (default **soft**) to control whether sky subtraction failures abort the run.
- P0-J (soft): degradable sky-fit failures now emit `QADEGRD=1`, set the MASK bit `SKYMODEL_FAIL`, and write a clearly marked **pass-through** product (`*_skysub_raw.fits`) with QC flag `SKY_SUB_PASSTHROUGH` (WARN).

### Fixed
- P0-J: `NO_SKY_WINDOWS` and related geometry errors are downgraded to WARN in `sky.failure_policy=soft`, so extended objects / small sky windows no longer make the stage (and night run) fail.

### Tests
- Add a synthetic regression test ensuring soft-mode sky pass-through produces a WARN-level stage and correct flags/headers/mask bits.

## [6.0.11] - 2026-01-16

### Added
- P0-I: project manifest search order is now **night-level first**: `data_dir/project_manifest.yaml` → `work_dir/...` → `config_dir/...`.
- P0-I: `project_manifest.yaml` schema **v1.1** adds a **global exclude** block (`exclude: {files, globs}`) and role-level `exclude` lists.
- P0-I GUI: Project Manifest dialog adds **Exclude selected / Restore** controls and a **Suggested SKY candidates** panel (conservative hints; requires explicit "Accept → SKY_FRAMES").

### Changed
- Excluded frames are removed from all role lists before downstream products are built (bias/flat/sky template/stack).

## [6.0.10] - 2026-01-16

### Added
- P0-H: calibration compatibility contract is now applied to **ARCs** in the wavesolution stage. QC-only mismatches emit the same flags as flats: `CALIB_ROT_MISMATCH`, `CALIB_SLITPOS_MISMATCH`, `CALIB_READOUT_DIFF`.
- P0-H: `sky_scale_raw` now filters `SKY_FRAMES` by must-match configuration (instrument/mode/disperser/slit/binning/geometry). Incompatible sky frames are rejected with `SKY_TEMPLATE_INCOMPATIBLE` (WARN) and listed in the stage metadata.

## [6.0.9] - 2026-01-16

### Added
- P0-G ARC matching: arcs can now **prefer same readout but allow** gain/rate mismatch (NODE remains strict), mirroring the flat policy.
- Dataset manifest now records transparent arc readout-selection provenance in `matches[].arc_meta`: `readout_policy`, `science_readout`, `selected_readout`, `selected_readout_match`, `n_hard_arc`, `n_same_readout_arc`, `selection_reason`.

### Changed
- Dataset builder defaults `arc_allow_readout_diff=True` for long-slit SCORPIO/SCORPIO-2 (can be overridden).

### Fixed
- Nights with science in Normal readout but arc/flat in Fast/low gain no longer silently end up “without an arc”; a deterministic fallback is used and explicitly flagged.

## [6.0.8] - 2026-01-16

### Fixed
- Dataset manifest builder now actually attaches the P0-F flat readout selection provenance into `matches[].flat_meta` (previously computed but not written).

## [6.0.7] - 2026-01-15

### Added
- P0-F calibration compatibility contract: **must-match** (geometry) vs **QC-only** (orientation/position) fields.
- QC-only mismatches now emit explicit flags: `CALIB_ROT_MISMATCH`, `CALIB_SLITPOS_MISMATCH`, `CALIB_READOUT_DIFF`.
- Flatfield stage writes canonical `done.json` (in addition to legacy `flatfield_done.json`) and carries QC flags into the report.
- Dataset manifest now records transparent readout-selection reasoning for flats (policy + selected/science readout + rationale).

### Changed
- Flat matching now **prefers same readout** for flats when gain/rate mismatches are allowed, but falls back deterministically when none exist.
- QC report aggregation includes `flatfield` stage flags.

### Fixed
- `ROTANGLE` and small `SLITPOS` differences no longer abort flat application; they are surfaced as QC warnings instead.

## [6.0.6] - 2026-01-15

### Added
- P0-E MasterFlat (long-slit): per-science-set MasterFlat built strictly from `dataset_manifest.json` associations.
- MasterFlat preprocessing is **readout-aware per input flat**: each flat is bias-subtracted using a MasterBias selected by the flat's own readout signature.
- MasterFlat products are MEF FITS with `SCI`+`VAR`+`MASK` (PRIMARY=SCI for backward compatibility).
- Flat application includes flat variance contribution when available: `Var(S/F)=Var(S)/F^2+S^2 Var(F)/F^4`.
- Flatfield stage can run in `use_manifest` mode and writes `masterflats` mapping + per-set QC stats into `flatfield_done.json`.

### Changed
- Dataset manifest builder now records `flat_ids` (full flat set) in `matches` while keeping `flat_id` as the representative "best" flat for backward compatibility.

### Fixed
- Flatfield stage now propagates flat MASK when present.

## [6.0.5] - 2026-01-14

### Added
- P0-D noise contract: compact FITS noise keywords stamped on electron-standardized products: `GAIN`, `RDNOISE`, `RN_SRC`, `NOISRC`.
- ReadNoisePolicy with strict provenance: read-noise is resolved from override/header/config/bias-estimate/docs/fallback (never silently guessed).
- MasterBias now estimates read-noise from bias pairs and stores it in products (`RNADU`, `RNPAIR`) and in `master_bias_index.json` (schema v2, `noise` block).
- Sky subtraction adds an explicit sky-model variance term when residuals exceed predicted VAR (`SKYVADD`, `SKYVMOD`, `SKYMSIG`, `SKYVSRC`).

### Fixed
- Flat-field stage: bias VAR is added exactly once; invalid flat pixels are flagged as `BADPIX` and do not produce NaNs in outputs.

## [6.0.3] - 2026-01-13

### Added
- P0-C2 VAR contract foundation: new `scorpio_pipe.variance_model` with CCD variance estimation + deterministic propagation helpers (add/sub/divide/scale/weighted-mean) and a robust reduced-chi^2 sanity metric.

### Changed
- MASK schema upgraded to **v2** (still `uint16`): adds bits `OUTSIDE_SLIT`, `INVALID_WAVELENGTH`, `SKYMODEL_FAIL`. Schema is written using compact <=8-char FITS keywords (`SCORPMKV`, `SCORPMB0..SCORPMB9`).
- Boundary contract now requires declared MASK schema cards and enforces schema version match (prevents silent drift in bit meanings).

## [6.0.2] - 2026-01-12

### Added
- P0-B1 Dataset Builder: generates explicit `dataset_manifest.json` with `science_sets`, calibration pools, deterministic `matches`, and `warnings` (no silent "closest enough" calibration selection).
- P0-B2 Strict calibration matching: must-match keys (instrument/geometry/readout and for flats/arcs spectro), plus soft-match selection by |Δt| with SPERANGE/SLITPOS tie-breakers.
- CLI: `scorpio-pipe dataset-manifest --data-dir <RAW_NIGHT>` to generate the manifest (supports `--hash`, `--no-frames`, `--no-recursive`, `--strict`).
- Tests: unit coverage for must-match and soft-match selection via the manifest builder.

## [6.0.1] - 2026-01-12

### Added
- P0-A2 deterministic frame classifier (`scorpio_pipe.dataset.classify`): `FrameClass`, `classify_frame()`, and long-slit mode guardrail `is_longslit_mode()`.
- Tests for frame classification using the real SCORPIO-1/2 header samples.

## [6.0.0] - 2026-01-12

### Added
- P0-A1 Header Contract: unified metadata structure `FrameMeta` and strict header parsers for **SCORPIO-1** and **SCORPIO-2** (`scorpio_pipe/instruments/*`).
- Header normalization utilities (string cleanup, non-standard date parsing, stable slit/binning keys).
- Tests for SCORPIO headers: `tests/test_headers_sc1.py`, `tests/test_headers_sc2.py`.

### Changed
- `inspect` now prefers contract-normalized meta for stable grouping (slit width, binning, disperser, instrument id), while keeping best-effort behavior for dataset browsing.
- `calib_compat` now derives matching keys via the Header Contract first (fails loudly on contract violations), with a legacy fallback for non-SCORPIO headers.

## [5.40.34] - 2026-01-03

## 5.40.40

- Исправлен запуск GUI в windowed/.exe сборках: добавлен bootstrap с логом и показом ошибок.
- Для Windows сборок отключён UPX (меньше ложных срабатываний антивируса).
- Окно лаунчера принудительно активируется при старте.


### Fixed
- Workflow (doit): added `flatfield` task and wired dependencies (`sky_sub` depends on `flatfield`; `cosmics` depends on `superbias` when bias subtraction is enabled; `superflat` depends on `superbias`).
- Calibration master resolution: `_resolve_superbias_path` / `_resolve_superflat_path` now check canonical stage dirs and legacy `calibs/` + `calib/` fallbacks.
- SuperNeon: robust superbias resolution for GUI/CLI runs (no assumption about `work_dir/calibs`).
- LineID: added `waveref` metadata header to the default neon line list and enforced consistency with `wavesol.waveref` to prevent silent systematic wavelength shifts.
- Docs: corrected stage order (sky subtraction happens before linearization) and updated quick-start steps.

## [5.40.32] - 2026-01-03

### Fixed
- Stage hashing: stage state hash now includes **string and boolean** config values (not only numerics), preventing silent cache hits when users change non-numeric settings (e.g. method names/policies).
- Stage hashing: input frame paths are now resolved via `config_dir`/`data_dir` before hashing, so cache correctness no longer depends on current working directory.
- Flat → Sky wiring: Sky Subtraction now prefers flat-fielded per-exposure frames (`04_flat/obj/*_flat.fits`) when available, otherwise falls back to cosmics-cleaned or raw frames.
- Sky variance model: removed dangerous hidden defaults (`gain=1`, `RN=5e-`) and now resolves gain/read-noise from overrides/header/instrument defaults consistently.
- Noise model: fixed `estimate_variance_auto()` returning an incorrect tuple for ADU inputs.
- Linearize flexure correction: when applying Sky-derived `delta_lambda`, the lambda-map is now shifted consistently along with the output grid (prevents sign errors and wrong binning).
- Pipeline hash deps: Sky-stage hash now tracks the actual inputs (`05_cosmics/obj/clean` and/or flat-fielded frames), not a non-existent `05_cosmics/clean` directory.

## [5.40.31] - 2026-01-03

### Fixed
- Calib: fixed `superbias` stage crash caused by logging/header stamping using undefined `combine`/`sigma_clip` (now resolved from `calib.bias_*` settings deterministically).
- Flat-fielding: restored missing imports (`ensure_compatible_calib`, `ensure_electron_units`) so the stage runs and enforces strict calibration/unit contracts.
- Stack2D: made SCI/VAR/MASK validation a hard gate; stacking now fails fast if NaNs or negative variance appear in unmasked pixels.
- Sky subtraction: default is now **strict** about `lambda_map` shape; mismatch becomes an ERROR and stops the stage (legacy implicit Y-repeat/crop is available only via `sky.lambda_map.allow_shape_adjust: true`).

### Changed
- QC gate: bypassing ERROR blockers now requires explicit opt-in `qc.allow_override_errors: true` to prevent accidental propagation of bad products.

## [5.40.25] - 2026-01-01

### Added
- CI: split tests into a fast pytest suite and a separate synthetic smoke-test suite (`-m smoke`). The smoke job uploads `qc_report.json` and `ui/navigator/data.json` as CI artifacts (always uploaded, even on failures).
- UI/QC: added a **Recommended actions** panel in the inspector, mapping QC flag codes to conservative next steps (no auto-run).
- P3: implemented optional Sky-frame scaling (Skycorr-like) when explicit sky frames are provided; writes metrics and safe no-gain flags.
- P3: added optional optimal extraction (Horne-like) with honest variance propagation; writes a per-λ profile diagnostic.
- P3: added explicit η(λ) diagnostics in Stack2D (robust sigma before/after η) with QC flagging when calibration looks wrong.
- P3: added `export-package` to bundle spec1d + QC + navigator into a single zip for archival/sharing.

## [5.40.24] - 2026-01-01

### Fixed
- UI: eliminated sporadic PySide6 ownership/use-after-delete crashes by avoiding parented Q*Layout constructors and adding defensive layout re-creation in stage pages (Cosmics and others).
- UI: removed duplicate `lay.addWidget(left, 1)` blocks that could cause unpredictable layout behavior.

## v5.40.23

### Fixes
- Packaging: migrated license file declaration from deprecated `tool.setuptools.license-files` to `project.license-files` to avoid upcoming setuptools build breakage.
- CLI: ensured diagnostic output renders literally under Rich markup (error codes/messages like `frames[obj]` are no longer eaten as markup). This also fixes the Doctor hint displaying `scorpio-pipe[gui]` correctly.

## v5.40.22

### Fixes
- GUI (PySide6): fixed a crash in the Calibration page where `QObject.findChildren()` was called with a tuple of widget types (unsupported by PySide6). Advanced-tab emptiness detection now queries per-type.

## v5.40.21

### Fixes
- CI (Windows Release): fixed `gh release upload` failing on the `publish` job with `fatal: not a git repository` by adding an explicit checkout and passing `--repo`.

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
