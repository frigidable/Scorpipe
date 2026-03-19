# Observatory release gates

A release may only be called **LONG-level** or **observatory-ready** when all of the following are true.

## Required benchmark evidence

- The LONG parity registry exists and is up to date.
- At least 8 real benchmark cases are populated.
- Each populated case has machine-readable Scorpipe and LONG observer bundles.
- The generated comparison reports are archived with the release candidate.

## Required quality gates

- No failing metric in the mandatory LONG parity cases.
- No regression in weekly golden mini-night QC/time budgets.
- No uncategorized `ERROR` severity stage flag in the release candidate.
- QC report includes operational failure classes where applicable.

## Required operator artifacts

- runbook updated for any newly introduced failure class or recovery path;
- observer-facing markdown/html comparison report attached to the release qualification package;
- known limitations listed explicitly.

## Honest wording rule

Do **not** call a build observatory-ready if only the public synthetic fixture has passed.
That fixture only validates the comparison machinery itself.


## Required certification package

A release-qualification bundle should be written with `scorpio-pipe certify-long-parity` and should include:

- machine-readable per-case reports for all populated manifests;
- golden weekly/manual evidence (`golden_time_budget_compare.json` or stricter equivalent);
- explicit known limitations;
- observer sign-off (for a true observatory-ready claim).
