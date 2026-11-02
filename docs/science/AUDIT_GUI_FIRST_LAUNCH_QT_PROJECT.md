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
