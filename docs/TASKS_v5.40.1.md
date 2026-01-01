# v5.40.1 — tasks (P1-B)

## Implemented in this release

- [x] Wavesolution: write `rectification_model.json` next to `lambda_map.fits`.
- [x] Wavesolution: include VAR/MASK propagation policies in the model (explicit, versioned).
- [x] wavesol_dir resolver: treat `rectification_model.json` as a canonical presence marker.
- [x] Linearize: prefer `rectification_model.json` and verify `lambda_map.sha256` when available.
- [x] Linearize: record `lambda_map_sha256`, `rectification_model_sha256`, `var_policy`, `mask_policy` in `done.json`.
- [x] Product registry: add `rectification_model_json` as a stable product.
- [x] Stage contracts: include `rectification_model_json` in Wavesolution outputs.
- [x] Changelog + version bump to 5.40.1.

## Follow-ups (next small, well-scoped steps)

- [ ] Sky Subtraction (Kelson RAW): optionally consume `rectification_model.json` and support sky-line-aware modeling.
- [ ] Add a small unit test: create a dummy `rectification_model.json` + `lambda_map.fits`, verify Linearize refuses on sha256 mismatch.
- [ ] Extend model schema with an *optional* precomputed sparse weight map (`rectification_weights.npz`) once performance becomes limiting.
