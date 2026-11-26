# BTA 2024-12-06 — P0 default manifest must not erase science frames

## Real-data evidence

First instrument-night validation on Windows Scorpipe 8.0.5, source
`C:\Users\frigi\Desktop\s241206` with 306 FITS. Night Inspector
identified science, bias, flats and neon. The N2146sat run was created
under `C:\Users\frigi\Documents\PhD\ScorpipeWork\06_12_2024\N2146sat_VPHG1200B_01`.

The run creation generated this *unconfigured* manifest template:

```yaml
schema: scorpio-pipe.project-manifest.v1.2
roles:
  OBJECT_FRAMES: {files: [], globs: []}
  ARCS: {files: [], globs: []}
  FLATS: {files: [], globs: []}
  BIAS: {files: [], globs: []}
```

All eight actual roles were empty. After loading config, the GUI's
effective `frames.obj`, `frames.neon`, `frames.bias` and
`frames.flat` were empty and a Preflight ERROR followed (78 excluded
tickets). This caused a potentially dangerous false absence of source
data and blocked the first CCD stage.

## Root cause

`autocfg.build_autoconfig` uses the inspect table and *does* derive
science/calibration candidates. `ensure_work_layout` then creates
an all-empty project role template. Before this patch,
`apply_project_manifest_to_cfg` interpreted the mere presence of
each role key as an explicit instruction to overwrite non-empty
`frames` lists with `[]`.

The metadata `roles_present` reflected role keys rather than any
actual configured provenance. It could wrongly suggest the role
inventory was complete.

## New resolution semantics

- An unconfigured project manifest containing no assigned role
  `files`, `globs` or role-level exclusions is **not an override**.
  AutoConfig or manually supplied config frame lists survive.
- A curated manifest with at least one assigned role *does* override
  all explicitly present role keys; its other explicit empty roles
  retain their intended meaning. Omitted roles preserve the config.
- An intentionally all-empty explicit role manifest is supported via
  `role_policy: explicit` and may correctly fail Preflight.
- A manifest containing only global exclusions preserves config
  role selection while applying the exclusions. Metadata reports
  `frames_source: config+manifest_exclude`.
- `_project_manifest.role_selection_status` distinguishes
  `template_unconfigured` from `explicit`;
  `roles_applied` enumerates actual overridden keys; an unconfigured
  template has `roles_present: []`, avoiding false gating.
- Project manifest GUI defaults to the selected **run directory**,
  not the read-only source `data_dir`; shared night-level manifest
  selection requires explicit `project_manifest_path`.

This changes role-selection semantics only; it does **not** automatically
approve gain-mismatched flats/arcs, substitute missing calibration
frames or turn a Preflight ERROR into PASS.

## Qualification

Regression fixtures cover the selected seven N2146sat object frames
(600s plus six 1200s), a neon calibration mislabeled in OBJECT,
one compatible bias, one candidate flat, explicit role curation,
global exclusion, intentional empty explicit manifests, and source
folder immutability. The Qt dialog test checks run-local default.

**Windows validation required:** read the raw on-disk `config.yaml`
separately from `load_config()` to distinguish a temporary
manifest override from an already-saved empty frame list. If
`config.yaml` itself contains empty lists, regenerate the config
through Night Inspector and review the resulting candidates.
Never use `Override ERROR` to force a science stage.

## Other science gates

Before Bias Correction, the frame inventory needs CCD calibration
readout compatibility, date/proximity, sky/flat/arc assignment
validation. Flat-field disabled, wavelength reference unknown and
absolute `/tmp/longsrc` path in embedded LONG provenance remain
separate audit findings and do not become safe due to this repair.
