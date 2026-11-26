# P1.2 velocity correction audit

> **Provenance note (integration, 2026-11-11):** This is the author's original
> Windows local audit of the uncommitted P1.2 patch relative to
> `d145f171d057ce56516e0ae755a3afe9a9d80c95`. Its references to
> 59 new velocity tests and local Windows results describe that local working
> tree; the uploaded patch contained **four tracked files, not those untracked
> tests**. PR #180 receives a separately reviewed subset of regressions and
> must pass its **own** CI before declaring equivalent coverage. This document
> is historical evidence, not a production E2E certification.
>
> **SCIENCE HOLD:** compute-only; no automatic wavelength-grid application,
> no physical sky-line or LSF approval, no changes to old FITS.


Baseline: `d145f171d057ce56516e0ae755a3afe9a9d80c95`.
Scope: standalone correction utility, configuration diagnostics, one spectral
metadata default, and synthetic tests. No correction is connected to a stage.

## Findings and actual usage

At baseline `apply_barycentric_to_wavelength_grid` divided both grid parameters
by `1 + vcorr/c`. Astropy defines a positive correction to be applied in the
opposite direction: multiply the vacuum wavelength. For +30 km/s at 5000 A the
old result was 4999.499703921445 A, versus 5000.500346142797 A required.
Round-trip alone cannot detect this error.

Other confirmed defects: numeric RA <=24 was multiplied by 15 unless RADECSYS
was FK5 (a coordinate reference system does not specify angular units); invalid
custom locations silently became BTA; timestamps were taken at exposure start;
DATE (file creation time) could substitute for observation time; invalid factors
silently returned an unchanged grid. Some errors escaped the result interface.

Repository-wide symbol/configuration search found no production caller/import
of either velocity helper. `autocfg.py` writes OFF; `schema.py` formerly accepted
arbitrary correction strings without a consumer. Validation now accepts only
None/OFF and rejects requests for unimplemented automatic application. This
diagnostic operates through existing schema/typed-config/doctor validation;
it is not a new engine hook. The engine, stack2d and extraction do not call the
utility. No historical product is invalidated or rewritten by this change.

`linearize._write_mef` formerly copied SPECSYS to a missing SSYSOBS. It now
defaults SSYSOBS independently to TOPOCENT, preserving explicit values and the
existing SPECSYS. No wavelength calculation, resampling, or SCI/VAR/MASK
propagation is changed by this metadata fix.

Fixtures `test_headers_sc1.py` and `test_headers_sc2.py` establish SCORPIO dates
as YYYY/DD/MM with UTC in UT, distinct from local TIME-OBS. They contain no
numeric target RA unit evidence. `test_p0_b6_framemeta_policy_unit.py` has
colon-separated RA `12:30:00`, interpreted as 187.5 degrees. No real FITS data
or run directory was processed during the audit.

## Scientific and input contracts

For optical radial velocity v = cz:

```
F = 1 + vcorr/c, c = 299792458 m/s exactly
1 + z_out = (1 + z_in) F
v_out = v_in + vcorr + v_in vcorr/c
lambda_out = lambda_in F
inverse factor = 1/F, not 1 - vcorr/c
```

Astropy computes both explicit kinds, barycentric (BARYCENT) and heliocentric
(HELIOCEN); the latter omits some precision terms. Legacy function names,
v_bary_mps and specsyst_in/out fields remain, with output frame authoritative
for either kind. The value is optical correction in m/s, not radio velocity,
relativistic velocity or an intrinsic source velocity.

Numeric RA requires `ra_unit` or RAUNIT: deg or hourangle (documented aliases
accepted); no magnitude/FK5 heuristic. Colon sexagesimal RA defaults to hours.
DEC defaults to degrees; other DEC units are refused. Angles must be finite and
in range. Missing RADESYS and EQUINOX means ICRS, consistent with FITS defaults;
explicit FK5 is transformed using its Julian EQUINOX (default J2000). Unsupported
frames and an EQUINOX without RADESYS are refused, not relabelled as ICRS.

UTC only (TIMESYS defaults to UTC). MJD-AVG/DATE-AVG provide a representative
midpoint; otherwise MJD-OBS/DATE-OBS plus EXPTIME/2 are required. An explicit
`time_reference='mid'` declares OBS already at midpoint. Duplicate timestamps
must agree within 1 ms, including an average versus a derivable midpoint.
Date-only DATE-OBS needs a UT clock; DATE-AVG must include its own clock and
cannot borrow observation-start UT. Local TIME-OBS is not substituted. SCORPIO
slash dates require an explicit SCORPIO-1/2 instrument. No DATE/MJD fallback.
Missing, contradictory and invalid metadata produce ok=False, diagnostic text,
and NaN correction/factor. Zero EXPTIME is an explicit instantaneous observation.

The default BTA coordinates remain exactly as in the original code, matching
SAO's published longitude 41d26m30s E, latitude 43d39m12s N, altitude 2070 m.
Explicit terrestrial geodetic strings and scalar EarthLocation are accepted;
malformed/nonfinite/out-of-range values fail. No silent BTA substitution.
The published altitude is above sea level, not a precision WGS84 survey height;
this site representation does not warrant cm/s absolute accuracy claims.

Computation is offline with builtin ERFA ephemerides and actual bundled IERS-B.
No synthetic Earth orientation or degraded extrapolation is allowed. Dates
outside the installed table fail, so future observations may need an explicit,
reviewed data refresh. Result provenance includes Astropy/IERS data versions,
time, location, actual transformed ICRS coordinates with their explicit frame,
optical definition and ephemeris. Original FK5 coordinates are not mislabelled
as ICRS in the result. Tests pin
the same real data source and use independent direct Astropy calls.

The grid helper accepts finite scalar floats, a positive dimensionless factor,
positive origin and nonzero step. Quantity inputs are refused rather than
silently discarding units. It scales vacuum coordinates only, with no flux
resampling. Air/vacuum conversion is separate and generally nonlinear for an
air grid; air input is refused. Corrected/non-topocentric input frames and an
explicit already_applied flag reject a repeat. A bare tuple has no persistent
history: legacy callers must propagate frame/state themselves. It is not a
general reference-frame graph and cannot infer whether anonymous numbers were
previously corrected. There are no stage callers that could double-apply it.

## FITS and minimal provenance

WCS Paper III: SPECSYS is the frame of the spectral-axis coordinates; SSYSOBS
is the frame constant over the non-spectral coordinates (default TOPOCENT).
VELOSYS is a relative radial velocity in m/s, not a boolean "correction done"
or an arbitrary storage slot for Astropy's optical correction redshift times c.
WAVEREF is Scorpipe's air/vacuum convention, independent of all these frames.
MJD-OBS/DATE-OBS describe observation start; AVG keywords are representative time.

Compute-only returns an immutable result with applied=False and never writes
SPECSYS, SSYSOBS, VELOSYS, WAVEREF, arrays or files. The minimal current provenance
is this result, serialized separately if the caller chooses; no new FITS cards
are introduced. A requested correction is rejected by config until application
exists; a computed correction is not proof of application.

A future integration task must atomically transform the real spectral
coordinates and record input/output frame, optical F and applied state with
time/location/ephemeris lineage. Only then may it set SPECSYS. SSYSOBS remains
the observer frame; VELOSYS needs its own physical WCS-consistent semantics.
It must handle WCS and wavelength arrays consistently, idempotency across
cache/resume, per-exposure stacking and frame propagation into extraction.
Do not add repeated resampling merely to change coordinate labels. This is a
separate architectural task, not implemented by P1.2.

## Independent references and verification

Decimal evaluation with exact SI c gives the following absolute grid references:

| vcorr (km/s) | corrected 5000 A |
|---|---|
| 0 | 5000.000000000000 |
| +30 | 5000.500346142797 |
| -30 | 4999.499653857203 |

These tests failed on the old division for both nonzero signs before the fix;
they also verify the cross term in optical velocity and reciprocal inversion.
Initial regression tests against baseline: 33 failed, 7 passed (including new
API contracts absent from baseline). Two additional metadata tests reproduced
the old schema/SSYSOBS defects before their fixes.

Published Astropy Keck example (2016-06-04T00:00:00 UTC, ICRS RA 4.88375 deg,
DEC 35.0436389 deg, lon -155.4783 deg, lat 19.8283 deg, height 4160 m):
barycentric 20077.135 m/s; heliocentric 20070.039 m/s. Literal references use a
0.05 m/s tolerance for published rounding/data differences, never a Scorpipe
helper as an oracle. A second reference is computed directly by Astropy at
ICRS (10 deg,20 deg), BTA, MJD UTC 59000+300/86400, for both kinds, at 1e-6 m/s
agreement. Synthetic compute-only FITS tests check exact unchanged file bytes,
header cards and arrays.

Environment: Windows 11 AMD64, CPython 3.12.10, Astropy 8.0.2,
astropy-iers-data 0.2026.10.19.0.52.1, builtin ephemeris. No dependency changes.
Final verification: 59 new velocity tests, 153 targeted/contract tests passed;
fast suite 349 passed, 11 skipped, 2 deselected, and exactly the two baseline
Windows path failures below (baseline: 290 passed, same 2 failures). Smoke:
1 passed, 1 skipped. Golden/full suite was not rerun. Ruff check/format,
pip check and git diff --check passed. Commands:

```
python -m pytest -q tests/test_barycentric.py tests/test_p2_synthetic_calibration_core.py tests/test_p2_reg_020_stage_input_contracts.py tests/test_p2_syn_linearize_var_stat.py tests/test_headers_sc1.py tests/test_headers_sc2.py tests/test_p0_b6_framemeta_policy_unit.py --basetemp <new-workspace-temp>
python -m pytest -q -m "not smoke and not golden" --basetemp <new-workspace-temp>
python -m pytest -q -m smoke --basetemp <new-workspace-temp>
```

Windows
sandbox runs must direct both pytest basetemp and Python tempfile.tempdir to
ignored workspace storage; some tests create TemporaryDirectory independently
of pytest. The two baseline fast failures are slash assumptions in
test_workflow_manifest_task_delegates_to_engine and
test_p0_k_resolve_global_exclude_expands_files_and_globs. They are out of scope.

## NGC 2784

Applying the old inverse factor instead of the forward factor could in
principle produce an error approximately -2 vcorr (tens of km/s, up to about
60 km/s for an Earth orbital correction). However, there is no production
caller of this helper in this checkout, and no proof that it was applied to
the historical NGC 2784 data. Night-log fixtures mentioning NGC 2784 do not
establish such provenance. Do not attribute the observed historical velocity
discrepancy to this bug without the exact historical code/configuration,
spectral coordinate products, observation metadata and application evidence.

## Sources

- https://docs.astropy.org/en/stable/coordinates/velocities.html
- https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.radial_velocity_correction
- https://www.sao.ru/Doc-en/Telescopes/bta/descrip.html
- Greisen et al. 2006, WCS Paper III, sections 7 and 10:
  https://www.greenbankobservatory.org/~fghigo/gbtdoc/wcs_Fits_paper3.pdf