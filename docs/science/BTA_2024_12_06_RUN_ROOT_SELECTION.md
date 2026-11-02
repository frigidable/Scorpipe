# Real BTA night test: Run folder picked from wrong object/configuration

Observed on `s241206` (306 FITS, two dates 2024-12-06/07):
the user selected `N2146sat / VPHG1200B / slit 1.2 / bin 1x2`,
but the Setup `Run folder` field still showed a suffix `unknown_01`.
This was unsafe to use as the root of a scientific reduction.

## Three independent root causes

1. `_representative_header_from_inspect_df` selected the **first obj
frame from the whole night**, not the configured N2146sat observation.
For mixed multi-target nights that can create runs under the wrong
object, and could accidentally select a different disperser.
2. `run_root.disperser_key_from_header` and the run-passport signature
did not recognize the SCORPIO-1 `DISPERSE` FITS keyword, defaulting to
`unknown`.
3. `run_root.night_date_from_header` attempted only `DATE-OBS`
when present, not `DATE` if DATE-OBS failed to parse. SCORPIO-1 writes
`DATE-OBS=2024/06/12` (non-ISO year/day/month) alongside
`DATE=2024-12-06T15:55:05`. The old code could use *today's date*
as a silent fallback. Bias from `2024-12-07T04:14:18` legitimately
belongs to the prior observing night.

## Implemented contract

The GUI picks the representative header from `kind=obj` rows that
match **all four selected science dimensions**: object, disperser,
slit and CCD binning. A matching science FITS is mandatory, with a
cross-check against its physical `OBJECT` header. It never substitutes
another object's first SCI frame or a neon frame.

Automatic `Run folder` suggestions update when the selected object
or disperser or workspace changes, while user-edited run paths are
preserved. The selected slit value is included in the run passport
without editing the raw FITS. Errors clear a stale *auto-generated*
path; initializing a run without a matching science frame fails.

`run_root` now recognizes `DISPERSE` and independently parses the
unambiguous `DATE` header when `DATE-OBS` is unparseable. If neither
date is valid, it fails closed instead of generating a path for today.

Expected default output under a user-selected workspace:

```
C:\Users\frigi\Documents\PhD\ScorpipeWork\06_12_2024\N2146sat_VPHG1200B_01\
```

No original BTA FITS are modified, and no previous run directories
are rewritten automatically. This PR is stacked on #67, which is in
turn stacked on #66, to preserve GUI inspection lifecycle repairs.

Tests cover the actual multi-target/multi-instrument frame table,
neon exclusion, selected-science matching, unsupported selection,
DATE-OBS/DATE precedence, after-midnight night assignment,
SCORPIO disperser, auto run increments, and fail-closed missing date.

**Windows/manual gate:** screenshot of generated full absolute
`Run folder` before pressing Initialize run; verify that `Workspace
root` is not inside the raw FITS directory or any Git checkout.
