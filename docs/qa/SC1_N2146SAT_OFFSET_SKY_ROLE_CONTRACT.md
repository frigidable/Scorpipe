# NGC2146sat: observer-confirmed OFFSET SKY and ON-TARGET role contract

**Engineering status:** isolated loader verification only; observational scientific release **HOLD**.

## Evidence already established — do not rediscover

In historical [#178](https://github.com/frigidable/Scorpipe/pull/178), the
observer confirmed `s22800604.fts` as an intentional 600 s OFF-TARGET SKY
exposure. Its offset field is observed after IMG03. The six later
`s22800607.fts` … `s22800612.fts` are the 1200 s ON-TARGET exposures,
following IMG05 and ARC06. The observer-confirmed role overrides the
ambiguous header `IMAGETYP=obj`; it does **not** independently prove clean
sky slit regions or constant night-sky emission.

The previously saved `N2146sat_VPHG1200B_01/config.yaml` lists **seven**
files in `frames.obj` and no `frames.sky`. This is a genuine stale science
role contract, even though its earlier header audit and whole-file SHA-256
readability inventory passed. Keep that saved config and all original
RAW, `03_bias`, `_91`–`_96`, QC and previous run files **unchanged**.

The read-only inventory executed on the observer's Windows workstation
2026-11-13 has source manifest SHA-256:

`be6966c73b7f04aebdbca6739d5d38a279379933b2cf7c80ae567dd65f458765`

This manifest hash describes the observed bytes, not independently
authenticated observatory provenance.

## New isolated plan (never a stage run)

`scripts/qa/sc1_offset_sky_role_plan.py` takes the saved config, the
*actual exported* `20261113_003014_082_raw_inventory.json`, and a **new,
nonexisting output directory outside both RAW and all prior work roots**.
It verifies 32 selected paths and every whole-file SHA-256 against the
inventory *again*, and refuses changed input bytes, duplicate selections,
unexpected source role lists, or output collisions.

A new isolated `config.yaml` and `project_manifest.yaml` are created.
Their **explicit roles** are:

- `OBJECT_FRAMES`: exactly `s22800607.fts` … `s22800612.fts`.
- `SKY_FRAMES`: exactly `s22800604.fts` alone.
- The global `exclude` list stays **empty**. Do not globally exclude OBJ04:
  it must remain available as SKY.
- Other calibrated input selections (20 BIAS, 3 FLAT, 2 NEON) are preserved
  verbatim as the original raw source paths; no fast-readout BIAS is
  invented or substituted.

The isolated `config.yaml` pins `project_manifest_path` to that new
manifest **explicitly**. Without that pin, the raw data directory's own
manifest (if any) takes priority. After writing, the script invokes the
real `scorpio_pipe.config.load_config`, which may create bookkeeping
directories *only under the newly isolated root*. Its resolved
`_project_manifest` must say source=manifest and exactly two applied roles;
the resulting `frames.obj` and `frames.sky` must match the verified
six + one paths, with no overlap. The unchanged BIAS/FLAT/NEON lists and
isolation of `work_dir` are checked too.

The script writes an explicitly **nonqualifying**
`role_contract.json`. There is NO engine invocation and NO permission to
perform science reduction, stack, sky subtraction or overwrite past stages.
An invalid explicitly supplied manifest now fails closed in the common
config loader rather than silently reverting to seven OBJ.

## Windows PowerShell 5.1 (after approved integration HEAD is pinned)

Run from an existing verified QA worktree, with the project's Python
environment on `PYTHONPATH`:

```powershell
$qaCode = "C:\Users\frigi\Documents\GitHub\Scorpipe-SC1-QA-vNext"
$config = "C:\Users\frigi\Documents\PhD\ScorpipeWork\06_12_2024\N2146sat_VPHG1200B_01\config.yaml"
$inventory = "C:\Users\frigi\Documents\Scorpipe-QA\N2146sat_20241206\20261113_003014_082_raw_inventory.json"
$isolated = "C:\Users\frigi\Documents\Scorpipe-QA\N2146sat_20241206\role_contract_isolated_20261113"
$py = "C:\Users\frigi\Documents\GitHub\Scorpipe\.venv\Scripts\python.exe"

$env:PYTHONPATH = "$qaCode\src;$qaCode\scripts\qa"
& $py "$qaCode\scripts\qa\sc1_offset_sky_role_plan.py" --saved-config $config --raw-inventory $inventory --output-dir $isolated
if ($LASTEXITCODE -ne 0) { throw "SC1 isolated role plan failed" }
```

Choose a **fresh** `$isolated` name for every attempt; never remove
existing results to make it pass. A CI-tested branch/HEAD must be confirmed
before executing locally. The previously validated 32-FITS header gate
intentionally expects seven historical `frames.obj`; do **not** invoke it
on the corrected six-OBJECT/one-SKY isolated plan as if it were a first-build
check — use the explicit loader verification instead.

## Remaining physical gates

OBJ04's intent is known; its spatial cleanliness is **not** approved.
The saved line-profile/flexure studies in [#178](https://github.com/frigidable/Scorpipe/pull/178)
and physical sideband/BIAS evidence in [#174](https://github.com/frigidable/Scorpipe/pull/174)
must be reused. Do not multiply the 600 s OFF SKY by two and subtract
from every 1200 s ON TARGET without contemporaneous temporal/scaling
validation. A mechanical ROTANGLE change is not a measured slit PA, and
Stack2D pointing QC must not be relaxed. Wave2D/LSF calibration and
cross-exposure MasterBias covariance remain SCIENCE HOLD.

Central [#180](https://github.com/frigidable/Scorpipe/pull/180) remains
Draft; `main`, raw observations and existing reduction products are untouched.
