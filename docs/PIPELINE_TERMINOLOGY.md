# Scorpipe pipeline terminology and presentation contract

Status: **proposed, backward-compatible display-name migration (v1)**.

This specification governs labels and navigation terminology. It does **not**
declare that a computation has been validated for scientific use, and it does
not modify the reduction order, the algorithms, or the quality-control gate.

## Four levels of terminology

- **Workspace:** a set of observations, configuration, and reduction products.
- **Section:** a navigation grouping; **never** an executable stage or dependency.
- **GUI stage:** an existing numbered workspace page. Its stable key, number, and
  directory slug are unchanged; the display label is allowed to evolve.
- **Pipeline task:** a canonical engine operation (such as `superbias` or
  `wavesolution`), optionally used in multiple recipes. A task can have no
  dedicated GUI page. Aliases resolve to stable canonical task identifiers.

Labels live in `src/scorpio_pipe/catalog/stages.py`; the display-only grouping
and help text live in `src/scorpio_pipe/catalog/presentation.py`.
Call `gui_stage_presentation()` or `task_presentation()` when a human-readable
name, descriptive help, and section assignment are needed. Never reconstruct a
task name by lowercasing its label.

## Navigation sections

| Order | Section key | Section label | Scope |
| --- | --- | --- | --- |
| 1 | `observations` | Observations & Setup | Observation inspection, manifests, setup |
| 2 | `detector` | Detector Processing | Master detector calibrations, artifacts |
| 3 | `spectral` | Spectral Calibration | Arc frames, line identification, wavelength/LSF |
| 4 | `science` | Science Reduction | Sky, resampling, coaddition, extraction |
| 5 | `spectrophotometry` | Spectrophotometric Calibration | Standards, response, telluric transmission |
| 6 | `analysis` | Analysis & Quality Control | Measurements, QA, navigation, exports |

The section ordering is designed for navigation, **not** for scheduling.
For example, standard-star reduction is a parallel calibration branch and
quality checks must be performed wherever scientifically required.

## Existing numbered GUI stages

| Stable key | Previous label | New display label | Existing directory |
| --- | --- | --- | --- |
| `project` | Project | Observations | `01_project` |
| `setup` | Setup | Reduction Setup | `02_setup` |
| `biascorr` | Bias Correction | Bias Correction | `03_bias` |
| `flatfield` | Flat-Fielding | Flat-Field Correction | `04_flat` |
| `cosmics` | Cosmics Cleaning | Cosmic-Ray Rejection | `05_cosmics` |
| `superneon` | Superneon | Arc Frame Combination | `06_superneon` |
| `arclineid` | Arc Line ID | Arc Line Identification | `07_arc_line_id` |
| `wavesol` | Wavelength Solution | 2D Wavelength Calibration | `08_wavesol` |
| `sky` | Sky Subtraction | Sky Modeling & Subtraction | `09_sky` |
| `linearize` | Linearization | Spectral Resampling | `10_linearize` |
| `stack2d` | Frame Stacking | 2D Spectral Coaddition | `11_stack` |
| `extract1d` | Object Extraction | 1D Spectral Extraction | `12_extract` |

The legacy workspace numbering is a **directory compatibility contract**.
It is **not** necessarily identical to GUI construction indices or the
optimal scientific processing sequence.

## Canonical engine task terminology

| Stable task key | New display label | Section |
| --- | --- | --- |
| `manifest` | Observation Manifest | observations |
| `superbias` | Master Bias Construction | detector |
| `superflat` | Master Flat Construction | detector |
| `flatfield` | Flat-Field Correction | detector |
| `cosmics` | Cosmic-Ray Rejection | detector |
| `superneon` | Arc Frame Combination | spectral |
| `slit_sanity` | Slit Geometry Check | spectral |
| `geometry_prior` | Geometry Constraints | spectral |
| `lineid_prepare` | Arc Line Preparation | spectral |
| `wavesolution` | 2D Wavelength Calibration | spectral |
| `lsf_model` | LSF Characterization | spectral |
| `sky` | Sky Modeling & Subtraction | science |
| `sky_refine` | Sky Residual Refinement | science |
| `linearize` | Spectral Resampling | science |
| `stack2d` | 2D Spectral Coaddition | science |
| `extract1d` | 1D Spectral Extraction | science |
| `standard_prepare` | Standard Star Reduction | spectrophotometry |
| `sensfunc` | Sensitivity Function | spectrophotometry |
| `telluric_model` | Telluric Transmission Model | spectrophotometry |
| `fluxcal_apply` | Flux Calibration | spectrophotometry |
| `publication_ready` | Science Product Export | analysis |
| `kinematics_prep` | Kinematics Preparation | analysis |
| `line_measure` | Emission-Line Measurements | analysis |
| `science_report` | Science Report | analysis |
| `qc_report` | Quality Control Report | analysis |
| `observer_dashboard` | Reduction Overview | analysis |
| `navigator` | Reduction Navigator | analysis |

The GUI-only `project` and `setup` entries retain their labels in the
canonical catalog, but are not engine tasks.

## Scientific wording constraints

- **Wavelength Calibration** names the operation;
  **wavelength solution** names its computed `lambda(x, y)` product.
- **Spectral Resampling** accurately describes the current wavelength-grid
  transform. Do not claim full *2D spectral rectification* unless spatial
  distortion is explicitly corrected and validated.
- **Arc Frame Combination** is independent of the calibration-lamp species.
  Legacy identifiers such as `superneon` must remain unchanged for compatibility.
- **Cosmic-Ray Rejection** must not imply that a masked event was safely
  interpolated; the scientific masks and QA retain their separate meaning.
- **Science Product Export** deliberately replaces **Publication ready**:
  exporting a file must not imply science qualification or publication suitability.
- **Completed** (execution) and **Science qualified** (validation) are
  distinct statuses; rename work must not merge them.

## Rollout and invariants

1. Introduce this terminology and presentation layer. Keep the 12 legacy
   pages and all 27 engine tasks; change *labels only* in the canonical catalog.
2. Add tests ensuring all current stage/task keys have precisely one UI
   section and a nonempty description, and ensure legacy paths are unchanged.
3. In a separate PR, migrate GUI navigation to a key-based model, never a
   magic list-row index; retain lazy page construction.
4. In separate scientific/engine PRs, remove any dependency of QC or planning
   on GUI ordering and replace competing hard-coded Run All/Run Plan lists
   with the execution engine's canonical recipes.
5. Introduce expandable six-section navigation only after the underlying
   key-based mapping and scientific QC are independently tested.
6. Compare representative real and synthetic reductions before release. In
   a label-only PR, numerical FITS SCI/VAR/MASK and wavelength arrays should
   remain unchanged; changes to scientific code require separate qualification.

**Compatibility guardrails:** do not rename config keys, task identifiers,
aliases, `done.json` semantics, output globs, FITS HDUs, stage slugs, or
existing numerical algorithms in a presentation-only PR.
