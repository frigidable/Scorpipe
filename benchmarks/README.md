# Perf baselines

Scorpipe stores performance baselines as versioned JSON artifacts under:

- `benchmarks/baselines/microbench/`
- `benchmarks/baselines/golden_time_budget/`

Recommended tracks:

- `main.json` — rolling baseline used by CI comparisons
- `<version>.json` — snapshot kept with the release (for example `7.42.11.json`)

Compatibility files are still kept at:

- `benchmarks/microbench_baseline.json`
- `tests/golden/mini_night_time_budget.json`

Weekly workflows use hard regression gates. PR workflows run the same comparisons in advisory mode and emit warnings instead of failing the check.


## LONG parity baselines

Observer-level LONG parity fixtures and case manifests live under `benchmarks/long_parity/`.
Public git only ships the schema + synthetic fixture; real observatory certification requires filling the registry with private or separately hosted nights.
