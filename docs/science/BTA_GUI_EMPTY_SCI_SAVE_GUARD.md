# GUI safeguard: do not save zero SCI over a detected BTA science selection

The first real 306-FITS BTA night found 7 valid N2146sat science candidates
but saved `config.yaml` with `frames.obj: []` after the default empty
project manifest incorrectly masked selections and the editable YAML was
saved back to disk.

Patch #71 corrects the unconfigured manifest application.
This additional GUI safeguard checks, **immediately before Save**, that
the editable YAML contains a non-empty `frames.obj` whenever the
current Night Inspector table can produce science exposures matching
the selected object's disperser, slit and binning.

When a user tries to save `frames.obj: []` for N2146sat (7 matched
science frames), the UI refuses and logs a specific recovery action:
regenerate the config via `Create new config` from Setup, then inspect
the role inventory. The guard **does not write files, fill missing
calibrations, or override user provenance**.

If inspection is absent, or the object has no matching science
candidates, the guard does not interfere with configuration editing.
Power users who deliberately need empty SCI lists can manage YAML
outside this GUI after separate validation.

Unit regressions execute the actual LauncherWindow method in headless
Linux CI; an AST ordering test makes sure the guard happens before
`config.yaml` can be written. A Windows GUI check should show that
saving empty SCI is blocked and saving 7 valid frames is allowed.

No source FITS is altered.
