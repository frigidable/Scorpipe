# P0 field fix — first BTA GUI run: air wavelength reference and calibration paths

On 2026-11-04, the first independent GUI real-data run of SCORPIO 8.0.5
on the 306-FITS BTA night `s241206` reached Preflight but failed **before**
running Manifest or Superbias.

## Blocking defects seen in GUI evidence

1. `AutoConfig.to_yaml_text()` emitted
   `project.wave_reference: UNKNOWN`, while the typed schema
   permits only `air` or `vacuum`. The live run consequently raised
   `ConfigValidationError` before any computation.
2. `load_config()` selected `project_root/03_bias/superbias.fits`
   and `project_root/04_flat/superflat.fits` if neither target existed
   under the new run directory. It used **file existence to resolve an
   output path**. In the GUI the absolute paths pointed into the Git
   worktree, not the intended run. This risks cross-run contamination.

## Corrective contract

* Bundled optical He-Ne-Ar catalogs `resources/neon_lines.csv` and
  `resources/henear_lines.csv` explicitly declare `waveref: air`.
  Auto-generated setups therefore use `project.wave_reference: air`.
  This is a convention of the adopted calibration atlas, **not** an
  assertion that SCORPIO photons have an intrinsic air/vacuum reference.
  Explicit `vacuum` is still permitted; selecting it requires a
  physically consistent atlas and wavelength pipeline.
* Relative `calib.superbias_path` and `calib.superflat_path` are
  always resolved from `work_dir`, independent of whether the
  products exist. Explicit absolute paths stay absolute.
  Explicit legacy `work/...` project-root-relative paths retain their
  legacy behavior. No source FITS or calibration files are moved.

## Independent gates still open

The current observation's Preflight selected **85** frames:
7 SCI, 41 bias, 17 neon, 3 flat, 17 sunsky. All 85 have optional
metadata warnings. There are 4 header-contract ERROR frames in
the full 306-FITS night (night-wide audit; not assigned to this run).
This patch **does not validate role compatibility**, particularly
readout/gain, timing and physically credible SUNSKY and NEON choices.

A GUI run must remain blocked at scientific acceptance until those
assignments are independently audited. The workdir must be outside
the Git checkout. Old configs have persisted wrong **absolute** paths
and are not repaired by changing source code: create a *new* run
or deliberately correct/revalidate their YAML; do not assume Reload
changes already serialized absolute path values.

## Validation

Synthetic-only tests cover three SCORPIO identifiers, typed pre-run
schema, generation before output exists, unrelated project-root
file contamination, stable reload after file creation, legacy explicit
project paths and explicit vacuum preservation. No observational
qualification or science-output variance release is claimed.
