# Architecture

## Layers and frozen boundaries

Scorpipe is organized in layers. The key goal is to keep dependencies flowing **one way** and to prevent “layer punching” over time.

**Frozen public boundary (P0-ARCH-12):**

- **UI and CLI must import pipeline functionality only via** `scorpio_pipe.pipeline.public_api`.
- UI must not import `scorpio_pipe.stages.*` directly.
- The engine/pipeline must not depend on UI.

These rules are enforced in CI by `tests/test_arch_no_forbidden_imports.py` using `tools/arch_imports.yml`.

### Layer map

**Application layer**
- `scorpio_pipe.ui.*` (PySide GUI)
- `scorpio_pipe.cli` / `scorpio_pipe.__main__`

Allowed dependencies:
- `scorpio_pipe.pipeline.public_api`
- “pure” utilities (paths, schema, small helpers)

**Pipeline layer**
- `scorpio_pipe.pipeline.engine` (execution semantics)
- `scorpio_pipe.pipeline.context` (StageContext/StageResult + small API types)
- `scorpio_pipe.pipeline.session_store` (UI session persistence)

Allowed dependencies:
- `catalog/*`, `contracts/*`, `ports/*`, `adapters/*`, `io/*`, `qc/*`, `utils/*`, `stages/*`

**Catalog / Contracts / Ports / Adapters**
- `scorpio_pipe.catalog.*` — single source of truth for stages/products
- `scorpio_pipe.contracts.*` — contract shapes
- `scorpio_pipe.ports.*` — interfaces (keep pure)
- `scorpio_pipe.adapters.*` — concrete implementations

**Stage implementations**
- `scorpio_pipe.stages.*`

Stages are *plugins* executed by the engine via **lazy entrypoints** declared in StageCatalog.


## Calibration/science domain (P0 flux-calibration preparation)

Starting with the P0 calibration-domain expansion, the architecture explicitly knows
about standard stars, telluric standards, response functions, flux-calibrated
products, and science-side derived products **even before** the full scientific
implementation is complete.

This domain is intentionally split across frozen boundaries:

- `metadata/frame_meta.py` — frame-level observing metadata needed for calibration
  policy (`airmass`, `zenith_distance_deg`, `parallactic_angle_deg`, `ra_deg`,
  `dec_deg`, `utc_mid_exposure`, site/seeing fields, and provenance for each).
- `calib/association.py` — machine-readable policy for standard/science
  compatibility (`INCOMPATIBLE`, `RELATIVE_RESPONSE_ONLY`,
  `RELATIVE_TELLURIC_ONLY`, `ABSOLUTE_APPROX`, `ABSOLUTE_TRUSTED`).
- `refs/context.py` + packaged resources — immutable reference-context snapshots
  for standard libraries, extinction curves, telluric masks, and telluric
  templates.
- `catalog/stages.py` — canonical vocabulary of the calibration/science stages:
  `standard_prepare`, `sensfunc`, `telluric_model`, `fluxcal_apply`,
  `line_measure`, `science_report`.
- `catalog/products.py` — canonical vocabulary of the corresponding artifacts
  (`standard_1d.fits`, `sensfunc_table.fits`, `telluric_model.fits`,
  `fluxcal_spec1d.fits`, `fluxcal_2d.fits`, `science_lines.csv`,
  `science_report.md`, etc.).
- `contracts/calibration.py` + `contracts/validators.py` — semantic law for
  calibrated outputs (`CALSTAT`, `FLUXUNIT`, `REFCTID`, `SENSFUNC`, `TELLURIC`,
  plus wavelength-reference and unit checks).

The important architectural rule is that these concepts are now part of the
**catalog/contract/reference** layer, not hidden inside future stage
implementations. That keeps the pipeline vocabulary stable while the numerical
algorithms evolve.

### Calibration-domain stage flow

The intended high-level graph is:

`linearize -> standard_prepare -> {sensfunc, telluric_model}`

`extract1d + sensfunc + telluric_model -> fluxcal_apply -> line_measure -> science_report`

Notes:

- `standard_prepare` is the standard-star-specific preparation boundary. It now reuses the same extraction kernels as `extract1d`, but with a distinct policy layer: wide-box flux-first extraction is primary, while optimal extraction is diagnostic-only.
- `sensfunc` and `telluric_model` consume standard-domain products and reference
  context, not ad-hoc globals.
- `fluxcal_apply` is the first place where products become semantically
  calibrated and therefore must satisfy the calibrated-product contract.
- `line_measure` and `science_report` are first-class pipeline stages, not
  post-hoc side scripts.

### Semantic contract for calibrated outputs

A calibrated product is not defined only by its filename. It must carry a
machine-readable semantic stamp in FITS headers:

- `CALSTAT` — calibration status
- `FLUXUNIT` — declared flux unit
- `REFCTID` — reference-context id used when building the calibration
- `SENSFUNC` — applied sensitivity-function reference
- `TELLURIC` — applied telluric reference (required for telluric-aware statuses)
- `WAVEREF` — air/vacuum wavelength convention

See `docs/spec/FLUX_CALIBRATION_CONTRACT.md` for the normative calibrated-output
contract.


## Science / observer layer (P3)

P3 freezes one more boundary above calibration: **science interpretation and
observer-facing summaries must not be mixed into the reduction/calibration
stages themselves**.

The dedicated science layer now lives under `src/scorpio_pipe/science/` and is
split into small interoperable modules:

- `science/lines.py` — line measurement, continuum handling, statistical +
  calibration uncertainty propagation;
- `science/reddening.py` — compact reddening helpers (e.g. Balmer decrement);
- `science/kinematics.py` — velocity / linewidth conversions;
- `science/report.py` — observer/science summary rendering.

Operational stages consume this layer without owning it:

- `line_measure` measures lines from calibrated spectra and preserves both
  `err_stat` and `err_cal`.
- `science_report` renders the final observer-facing summary from calibrated
  products, QC, provenance, and measured lines.

This keeps Scorpipe architecturally honest: reduction, calibration, science, and
observer presentation remain separate layers, but with explicit machine-readable
handoffs.

### Observer-facing wrappers

The public API now exposes UI-safe summary wrappers for:

- standard association / standard_prepare;
- sensfunc;
- telluric model;
- flux-calibration status;
- science report / detected lines;
- recipe-level workflows.

The GUI is therefore expected to consume **public summaries**, not raw internal
JSON blobs or direct stage internals.

### Recipe-level workflows

P3 also adds first-class recipe definitions. Recipes are not a replacement for
StageCatalog; they are a convenience layer mapping observer intents onto stable
canonical task sequences.

Examples:

- quicklook science;
- full reduction;
- standard-star reduction;
- relative flux calibration;
- full science + calibration + report.

### LONG comparison harness

A separate comparison harness lives outside the reduction stages. LONG is treated
as an **operational benchmark**, not as a truth oracle. The harness compares
wavelength, sky, extraction, response-calibration, and science-output metrics by
numbers, while synthetic truth and repeatability remain independent guardrails.

## Public API

The public API for orchestration is:

- `scorpio_pipe.pipeline.public_api`

It intentionally re-exports:
- StageCatalog data for UI lists/help (`STAGE_CATALOG`, `CANONICAL_TASKS`, `TASKS`, `STAGE_SPECS`)
- Engine wrappers (`plan_sequence`, `run_sequence`, `run_sequence_locked`)
- Calibration-domain UI helpers (`run_standard_prepare`, `read_standard_prepare_summary`)
- Context types (`StageContext`, `StageResult`, `CancelToken`, `StageSpec`, `PlanItem`)
- Session store helpers (`load_session`, `save_session`, …)

If UI/CLI needs a new pipeline capability, add it to `pipeline/public_api.py` (prefer wrappers and lazy imports).


## How to add a stage

### 1) Implement the stage entrypoint

Create a module under `scorpio_pipe/stages/` with a callable entrypoint (usually `run`).

Guidelines:
- Prefer signature: `run(*, cfg: Mapping[str, Any], run_root: Path, stage_dir: Path, cancel_token: CancelToken | None = None, **kwargs)`.
- Or accept a `StageContext` if the stage is already migrated to the context object.
- **Write outputs only under** `ctx.stage_dir` / `stage_dir` (unless explicitly handled by engine policies like `manifest/`, `qc/`, `ui/`).
- Return a `StageResult` when possible (engine will record status/products/QC uniformly).

### 2) Register the stage in StageCatalog

Add a `StageInfo(...)` entry to `scorpio_pipe/catalog/stages.py`.

Minimum required fields:
- `key`: canonical task name
- `label`: human label
- `aliases`: alternative names (optional)
- `entrypoint`: **string**, e.g. `"scorpio_pipe.stages.my_stage:run"`
- `done_dir_policy`: where `done.json` lives (`stage`, `manifest`, `qc`, `ui`, …)
- `contract_kind`: output kind (`none`, `mef`, `spec1d`, `lambda_map`, …)
- `validate_globs`: glob patterns for outputs that must exist / be validated
- `product_keys`: product registry keys written by the stage
- (optional) `stage_key`, `stage_slug`: if it maps to a numbered GUI stage directory

**Important:** StageCatalog is the **single source of truth**. Do not create new per-stage tables elsewhere.

### 3) Wire products (if needed)

If you introduce new product keys, add them to the product catalog (`scorpio_pipe/catalog/products.py`) and ensure:
- naming convention is stable
- output paths are deterministic

### 4) Add/adjust tests

At minimum:
- `tests/test_p0_arch_stage_catalog.py` should still pass
- add a small unit test for the new stage entrypoint (import + basic I/O contract)
- if you add a strict product kind, add boundary validation tests

### 5) Document changes

Update:
- `CHANGELOG.md` (user-visible behavior)
- any relevant docs under `docs/`


## Enforcement

- Direct import rules live in `tools/arch_imports.yml` and are enforced by `tests/test_arch_no_forbidden_imports.py`.
- If a rule blocks needed functionality, the fix is usually to extend `pipeline/public_api.py`, not to add exceptions.


## P2 implementation status

The calibration-domain placeholders registered in P0 are now backed by real core logic:

- `sensfunc` builds a smooth instrumental-response model from the extracted standard, reference table, extinction curve, and conservative masks.
- `telluric_model` produces an explicit telluric transmission layer with per-wavelength confidence rather than folding atmospheric absorption into the sensfunc.
- `fluxcal_apply` applies response, telluric, and semantic calibration-status assignment while preserving calibration systematics in a separate companion FITS product.

This keeps the contract rule intact: *response*, *telluric*, *application*, and *systematics* remain separate artifacts and separate semantic layers.
