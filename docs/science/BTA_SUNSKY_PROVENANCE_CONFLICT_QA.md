# SCORPIO-1 BTA field QA: SUNSKY versus explicit FITS calibration role

During first read-only inspection of the 306 FITS from `s241206`,
AutoConfig selected 26 SUNSKY candidates. Comparing their actual
`IMAGETYP` headers revealed 17 `obj`, 3 `neon`, and 6 `flat`.
The 9 calibrations cannot be assumed to be scientifically
interchangeable with the other 17 apparent sky frames.

The original inspector overrode FITS classification with the observing
nightlog, then unconditionally changed any object named `sunsky` to
`kind=sunsky`. This could hide true arc/flat semantics and could feed
calibration-lamp/flat pixels into a sky-reduction stage as a homogeneous
sky group.

## Resolution

* Preserve `header_imagetyp`, `header_kind`, `nightlog_kind`,
  `kind_source`, `role_conflict`, and `role_conflict_reason` columns.
* An unambiguous explicit calibration `IMAGETYP=neon/flat/bias`
  contradicting the observer's log or SUNSKY name produces
  `kind=unresolved`. The frame remains visible in the night inspector,
  but is **not** assigned automatically to arc, flat, sky, sunsky or
  object roles. It needs human resolution via explicit project manifest
  assignment.
* Non-conflicting frames remain unchanged. For example,
  `s22800606` has `OBJECT=N2146sat`, `IMAGETYP=neon` and
  nightlog `neon`: it remains an arc.
* A frame labelled SUNSKY in the nightlog but `IMAGETYP=obj`
  remains SUNSKY: `obj` is not strong physical evidence of a
  calibration conflict.
* Add explicit `IMAGETYP=neon` / `IMAGETYP=flat` recognition to
  the basic header-only classifier. Previously the neon type was
  recognized only from OBJECT text or IMAGETYP containing `ARC`.

The independent synthetic test reproduces
`s22800407` (`neon` versus SUNSKY),
`s22800408` (`flat` versus SUNSKY),
`s22800401` (unambiguous SUNSKY),
`s22800606` (unambiguous target neon), and
`s22800607` (target SCI).

This is a conservative classification decision, **not** a claim that
the original observer's nightlog is incorrect. The actual photons,
readout mode and intended calibration purpose require an observer
decision before the 9 disputed frames can be used.

It does not authorize cross-gain bias subtraction, combine long-separated
neon sets, or replace the science preflight requirements. The raw FITS
and observing log remain immutable.

## Windows acceptance check

Re-run Inspect on the 306 files using an exact head containing this PR.
Expect original 306 frame count, nine candidates marked `unresolved`,
and `N2146sat` source science/adjacent arcs preserved. Review
`role_conflict_reason` for those nine, and manually assign their roles
only after confirming their actual instrument function.
