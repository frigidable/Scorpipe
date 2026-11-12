# Stack2D rejected-input provenance versus Extract1D usability

The Stack2D stage can mark a final combined 2D pixel with `REJECTED`
because one or more input exposures were rejected or Huber-downweighted.
This bit is **informational for the accepted coadded pixel**, not necessarily
a cause to reject the science signal. It differs from a single-exposure
`REJECTED` bit, which is fatal.

## Safe interpretation

Extract1D treats Stack2D REJECTED as informational **only** when:

- the input is recognized as Stack2D and declares `STKMETH`;
- the product contains an integer `COV` map of matching shape;
- the `COV` map has no negative entries and every `COV=0` pixel
  is also flagged `NO_COVERAGE` in the input 2D MASK.

When these conditions are not established, REJECTED remains fatal.
The 2D Stack2D MASK and COV remain unchanged.

Only the REJECTED bit is dropped from 1D TRACE/FIXED *usability*
masks for these verified coadds. NO_COVERAGE and the existing fatal bits
remain fatal. A missing contributor or bad variance does not become valid
merely by clearing REJECTED.

## Deliberate limitations

- This is a safe subset of the broader 1D DQ provenance issue #99.
  COSMIC/BADPIX and other excluded aperture-input flags still leak into
  1D MASK and may cause LineMeasure to discard a partially valid channel.
  A future schema update must separate `MASK_ANY_INPUT` from
  `MASK_FATAL_OUTPUT` without silently reinterpreting existing products.
- Huber weight fraction is not itself input coverage; `COV` counts
  positive contributing exposures but does not quantify effective N.
- This change does not repair coadd variance under data-dependent Huber
  selection or shared MasterBias uncertainty (PR #115 / #117).
- Old stacks without trusted provenance remain conservative (potential
  signal loss) until reprocessed; no unverified compatibility guessing.
