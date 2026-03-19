# LONG parity benchmark corpus

This directory is the P0 backbone for comparing **Scorpipe** against **LONG**.

It contains three layers:

- `cases/registry.json` — the canonical list of benchmark slots that must be covered before calling a release observatory-ready;
- `cases/*.case.json` — per-case manifests with metric rules and pointers to comparison bundles;
- `fixtures/` — tiny public fixtures used by CI to validate the harness itself.

## Important honesty rule

The public repository only ships a **synthetic fixture** plus the corpus schema.
Real SCORPIO long-slit nights are typically too large and/or private to bundle into git.

That means:

- CI validates the harness, schema, reports, and release gates;
- real LONG-parity certification still requires filling the registry with private or separately hosted nights.

## Case manifest contract

A case manifest is a JSON object with at least:

- `case_id`
- `title`
- `category`
- `source_type`
- `description`
- `observer_notes`
- `scorpipe_bundle`
- `long_bundle`
- `metrics`

Metric keys use dotted paths into the observer bundles, for example:

- `wavelength_solution.rms`
- `sky_subtraction.sky_rms`
- `extraction.snr`
- `science_outputs.n_detected_lines`

Rules currently support:

- `prefer: lower`
- `prefer: higher`
- `prefer: equal`
- optional `tolerance_abs`
- optional `tolerance_rel`

## Suggested real-case mix

The registry intentionally reserves slots for at least these night types:

1. clean SCORPIO-1 reference night;
2. clean SCORPIO-2 reference night;
3. weak/ambiguous neon lines;
4. strong skyline residuals;
5. weak object extraction;
6. flexure-sensitive setup;
7. curved slit / geometry-stressed setup;
8. mixed calibration metadata / slit mismatch;
9. extended-object sky-window stress case;
10. repeatability case with repeated exposures.

## Public operational baseline

Alongside the synthetic LONG fixture, the repo now ships `cases/golden_mini_night.case.json`.
It compares the weekly golden run against a committed observer-bundle baseline (wave/sky/extract only).
This is **not** scientific certification; it is an operational stability check that complements the LONG harness.

## Filling private real-case slots quickly

Use the CLI helper below when you have a completed Scorpipe run and the matching LONG observer bundle:

```bash
scorpio-pipe prepare-long-case \
  --config work/<night>/<run>/config.yaml \
  --long-bundle /private/long/<case>/long_bundle.json \
  --registry benchmarks/long_parity/cases/registry.json \
  --case-id skyline_rich_residuals \
  --title "Skyline residual stress case" \
  --category sky
```

That command will:

- export a Scorpipe observer bundle for the chosen run;
- write/update `cases/<case_id>.case.json`;
- upsert the corresponding registry slot from `pending-private-data` to `ready`.

This does **not** fake certification. It only removes the mechanical work of keeping manifests, bundles, and the registry in sync.


## Mining existing LONG setup parameters

The extracted LONG tree already contains useful setup priors:

- anchor tables like `VPHG1200@540.txt` (`px -> λ`);
- curated line tables like `lines_tab_VPHG1200@540.tab`;
- slit-width helper routines such as `check_slitwidth.pro` / `slit_width_estimation.pro`.

You can summarise those priors into a machine-readable pack with:

```bash
scorpio-pipe inspect-long-params \
  --long-root /path/to/LONG \
  --output bench_work/long_setup_summary
```

## Bootstrapping night candidates from an observing log

When you have a BTA/SCORPIO night log but have not exported the final bundles yet, generate a candidate pack first:

```bash
scorpio-pipe bootstrap-night-cases \
  --nightlog /path/to/s251216.txt \
  --long-root /path/to/LONG \
  --header-text /path/to/ngc2784_science.txt \
  --output bench_work/night_bootstrap \
  --write-case-drafts
```

This writes:

- `night_bootstrap.json` / `night_bootstrap.md` with suggested registry slots;
- optional bootstrap-only case manifests in `case_drafts/`;
- registry updates marked as `bootstrap-only` so certification stays honest and does not count them as populated real cases yet.

That means you can pre-fill the evidence trail **before** attaching the real Scorpipe/LONG observer bundles.
