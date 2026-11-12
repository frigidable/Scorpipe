# Windows Release vNext — nonpublishing acceptance run

**Scope:** engineering acceptance of the pinned Windows Release Actions in PR #213. This is **not** permission to publish, tag or merge #180/#16 to `main`. SCIENCE HOLD remains.

## Automated, nonpublishing PR validation (preferred)

This PR adds a narrowly scoped `pull_request` event for **vNext-targeted changes to the Windows release workflow, Windows packaging or release-safety tests**. On the actual GitHub-hosted runners, the original `build_check` and `build` Windows jobs run unmodified. A separate read-only `artifact_contract` job downloads the same artifact via `actions/download-artifact@v8`, verifies the upload/archive round-trip, SHA-256 manifest, installer, and ZIP integrity. The existing `publish` job is constrained to **`github.event_name == 'release'`** and MUST remain **skipped** for the PR run.

For acceptance, inspect the **Windows Release** workflow associated with the precise PR #213 HEAD, not merely the Linux `CI` workflow. Demand `build_check=success`, `build=success`, `artifact_contract=success`, and `publish=skipped`. Any failure or missing run is a HOLD. This proves a Windows package/artifact path, not signing, publishing, generated SBOM, or live attestations: those actions remain in the intentionally skipped `publish` job.

If the PR workflow does not start (e.g. restricted Actions settings), use the controlled `workflow_dispatch` route below as a fallback; note that it does **not** run `artifact_contract` or exercise download-artifact.

## Pre-run gate

1. In GitHub, open **Actions → Windows Release → Run workflow**.
2. Select **`integration/windows-release-actions-vnext-20261112`** while PR #213 is in Draft. After integration, use **`integration/scorpipe-vnext-20261111`**. Confirm the selected ref points at the reviewed PR/merge HEAD before submitting.
3. Launch **workflow_dispatch only**. **Do not** create or publish a GitHub Release, tag a new release, merge #16, or manually bypass the `publish` job condition. Do not use production observation files.

## Expected job behavior

| Job | Expected | Reason |
|---|---|---|
| `build_check` | **success** | Build wheel and sdist on `windows-latest`; install wheel and smoke-test CLI |
| `build` | **success** | Build actual installer and portable ZIP; produce `SHA256SUMS.txt` and upload `windows-release-assets` |
| `artifact_contract` | **success on PR; skipped on manual dispatch** | Read-only archive, ZIP and SHA-256 round-trip verification (PR only) |
| `publish` | **skipped** | Workflow job condition is `github.event_name == 'release'`; `workflow_dispatch` MUST NOT publish |

The repository must have **no new published release, published assets or tag** as a result of this dry-run. Workflow status may appear as success with `publish` skipped. Any run in which `publish` starts on `workflow_dispatch` is an immediate release-safety failure.

## Artifact verification

Download the workflow artifact **`windows-release-assets`** from that exact run and inspect:

- `ScorpioPipe-Setup-x64-<version>.exe` exists, is nonempty and has a version matching `src/scorpio_pipe/version.py`.
- `Scorpipe-Windows-x64-<version>.zip` exists, opens without errors and contains the portable bundle.
- `SHA256SUMS.txt` exists; every listed filename exists and its SHA-256 digest recomputes exactly.
- Artifact upload succeeded using **`actions/upload-artifact@v7` with `archive: true`**. The receiver `download-artifact@v8` is **not exercised** because the `publish` job is intentionally skipped. Do not claim end-to-end upload/download coverage from this dry-run alone.

The `publish` job also generates SBOM, attestation and release assets. These actions are intentionally **not exercised** in the safe PR/manual checks. Their inputs and path contracts are statically guarded by `tests/test_windows_release_workflow_safety.py`, but that is **not runtime proof**. An isolated, nonpublishing attestation test must precede any actual release when science gates eventually permit it.

## Required evidence before accepting PR #213

Record the exact workflow URL, selected ref, exact commit SHA, Windows runner result, installer/portable/archive checksums, `publish=skipped` status and any log warnings in the PR. Keep PR #213 **Draft / runtime HOLD** until this evidence passes. Do not certify publication, signing, SBOM provenance or scientific uncertainty from successful packaging alone.

If the workflow fails during `Resolve build version` or installer creation, diagnose the **actual Windows PowerShell logs** and fix that separately; never weaken the publish condition to force a green result.
