# Science Layer v1

This specification defines the first stable observer/science layer above the
reduction + calibration stack.

## Principles

1. Science interpretation does not live inside reduction stages.
2. Measured science products must be machine-readable.
3. Uncertainties must keep statistical and calibration components separate.
4. Observer-facing reports must summarize trust, not hide it.

## `line_measure`

Expected outputs:

- `science_lines.csv`
- `science_lines.json`
- preview plot(s)

Each measured line should report at least:

- integrated flux;
- equivalent width;
- centroid;
- sigma / FWHM;
- local continuum;
- `err_stat`;
- `err_cal`;
- component calibration terms (`err_cal_rel`, `err_cal_tell`, `err_cal_abs`) when available.

`line_measure` must treat calibration-systematics as a first-class input rather than silently folding them into `VAR`.

## `science_report`

Expected outputs:

- `science_report.md`
- `science_report.html`

The report should summarize:

- calibrated spectrum preview;
- calibration status;
- key measured lines;
- warnings / degraded-confidence flags;
- provenance of standard / sensfunc / telluric layers.

## Recipes

Recipes are observer-level convenience workflows built from canonical tasks.
They are intentionally declarative and should remain import-light.

## Comparison harness

Formal comparison against LONG belongs to a separate harness and must use
quantitative metrics rather than visual-only judgments.
