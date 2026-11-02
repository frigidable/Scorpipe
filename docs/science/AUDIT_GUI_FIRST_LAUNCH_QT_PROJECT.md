# GUI field observation: initial Setup placeholder instead of Project

**BTA field test:** Scorpipe 8.0.5 on Windows, data
`C:\Users\frigi\Desktop\s241206`, 306 FITS of 2024-12-06/07.

On fresh launch, step `01 Project` was highlighted but the main area
showed `Loading... 02 Setup` indefinitely. Clicking the Project step
made the correct page appear, indicating the page constructor itself
was functional, but initial selected-row / QStackedWidget current-widget
synchronization was not. This is distinct from the QProgressDialog
spurious cancel issue in Draft #66.

## Fix and limitations

- Project page is now built synchronously during launcher construction
  rather than queued using `QTimer.singleShot(0)`. All expensive
  science-stage UI pages remain lazy.
- Explicitly set `page_project` as the active QStackedWidget child
  before the window is shown.
- Navigate between stages via stored QWidget identity, not positional
  widget indices. The existing Flat/Cosmics page swap is covered.
- Materialize the lazy Setup page before `Use this setup` from the
  Frames Browser or object double-click from the Project overview.
  Previously those callbacks could silently fail on the first click.
- A new Qt offscreen test validates immediate visibility and navigation.
- Headless tests run the production navigation method using deliberately
  scrambled widget ordering.

**Dependency:** Draft #66 handles successful Night Inspector completion
misreported as user cancellation. This PR is stacked on that branch
and must be retargeted after #66 is merged.

## CI note

GitHub workflow `ci.yml` is restricted to PRs targeting `main`.
The combined #66 + #67 branch was temporarily directed to `main`
for Linux CI on this exact commit and will be restored to the #66
parent after qualification. Full Qt GUI behavior still needs a
Windows offscreen/manual launch check: Linux CI installs `[dev]`, not
`[gui]`. Do not equate headless CI with on-screen verification.


## Second Windows field-test finding: Inspect-before-Setup AttributeError

Reproduction on the actual PR #67 worktree:
`C:\Users\frigi\Documents\GitHub\Scorpipe-PR67-QA`, the same
306-frame BTA night `s241206`. Project renders properly (the initial
permanent Setup placeholder is fixed). Inspect reads all 306 FITS and
finds the 306-row nightlog, but the completion callback raises

```
AttributeError: 'LauncherWindow' object has no attribute 'combo_object'
launcher_window.py, _done -> _populate_objects_from_inspect
```

The lazy Setup page creates `combo_object` only when the user opens
Setup. The Inspect completion callback was wrongly accessing those
controls before they existed. Its outer try/except consequently marked
a successful Inspect as failed and left the Project overview empty.

The fix records InspectResult in memory, **skips only the Setup-selector
synchronization** while Setup is not built, and still refreshes the
Project overview and marks Inspect successful. When Setup is first
materialized, the previously completed InspectResult is replayed into
its object, disperser, slit and binning selectors. No automatic
navigation to Setup is performed. Frames Browser remains usable and
independent; `s22800606.fts` was correctly classified as `neon`.

Two headless contract tests execute the actual production methods
against a missing Setup control and a subsequently constructed page.
An offscreen PySide6 regression uses a synthetic N2146sat science
frame and a neon frame. Windows manual qualification must verify
that the Project indicator turns green, counts render and the
`Inspect dataset` button becomes available again, without
`AttributeError` in Log; then opening Setup must populate the
selector without re-scanning the files.

This is a real GUI lifecycle defect discovered by testing, not
a failure of FITS reading or scientific reduction.

## Exact-head CI audit

The combined Draft #66 + #67 branch is tested against `main` for the
GitHub Actions workflow, then retargeted to #66. Verify CI for the
specific head commit containing the Inspect-before-Setup regression,
not the earlier initial-Project-only commit. This CI does not replace
Windows GUI testing on actual BTA FITS.
