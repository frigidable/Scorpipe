# LONG parity report — Synthetic mini-night harness sanity check

- Case ID: `mini_night_synthetic`
- Category: `ci-fixture`
- Source type: `synthetic_fixture`
- Status: **PASS**
- Metrics passed: 5 / 5
- Description: Tiny public fixture used to validate the LONG comparison harness, markdown report rendering, and CI gating behaviour.
- Observer notes: This is not a scientific parity claim. It only guarantees that the benchmark/reporting machinery works and that comparison rules stay stable in CI.

| Metric | Current | LONG | Status | Rule | Notes |
|---|---:|---:|---|---|---|
| Extracted S/N | 25 | 21 | PASS | higher | higher is better than reference |
| Response smoothness | 0.02 | 0.03 | PASS | lower | within absolute tolerance 0.02 |
| Detected science lines | 5 | 4 | PASS | higher | higher is better than reference |
| Sky residual RMS | 1.1 | 1.3 | PASS | lower | within absolute tolerance 0.35 |
| Wave RMS [A] | 0.12 | 0.18 | PASS | lower | lower is better than reference |
