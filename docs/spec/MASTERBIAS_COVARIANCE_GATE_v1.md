# Shared MasterBias covariance — P0 provenance and numerical oracle

## Confirmed defect and current limitations

When exposures are calibrated using the same MasterBias estimator, their
calibration errors are correlated. The fixed-weight variance calculated in
PR #122 assumes independent input exposures and therefore underestimates
the common-bias contribution when it is already included in each input VAR.

For independent image noise v_i and a single calibration error b with
variance V_B, fixed normalized weights alpha_i give:

  VAR = sum_i(alpha_i**2 * v_i) + (sum_i alpha_i)**2 * V_B.

For multiple independent calibration groups, and a declared scalar response
s_i to each group's bias error:

  VAR = sum_i(alpha_i**2 * v_i)
      + sum_g [sum_i_in_g(alpha_i * s_i)]**2 * V_B,g.

This formula is implemented in a *reference oracle* in
scorpio_pipe.calib.bias_covariance, with deterministic test cases and an
independent joint Gaussian Monte Carlo. It is **NOT YET called by
Stack2D**, because the necessary independent noise and calibration
components are not presently available after FlatField, SkySub and
Linearize. Applying a guessed correction to total VAR could double count
or misrepresent common calibration uncertainty.

## Implemented provenance and qualification

- FlatField stamps BVARINC=True when it adds indexed MasterBias estimator
  variance to a pre-existing VAR plane, as well as when it estimated VAR.
- Stack2D logs SB_GID and BVARINC for each admitted exposure.
- A multi-group or unknown-group stack does not inherit the first input's
  SB_GID as though it described the entire stack.
- Stack2D emits FITS: SBGRPN (# known groups), SBGMISS (# unknown group),
  SBGSTAT (SHARED, DISTINCT, SINGLE or UNKNOWN), SBCOVQ=UNMODEL.
- For confirmed shared groups, Stack2D emits a WARN flag
  SHARED_MASTERBIAS_COVARIANCE_UNMODELLED.
- Extract1D carries these flags, plus Stack2D STKVARQ and STKXCVQ, into
  its published SPEC1D header; Fluxcal can preserve them as provenance.
- Actual science SCI/VAR/MASK are **unchanged** by this patch.

## Why issue #117 remains OPEN

MasterBias must be decomposed from detector noise *before* FlatField,
carried through flat-field division, sky estimation and spectral
rectification. A resampling operator can mix detector columns, requiring
A C_bias A^T rather than only per-pixel diagonal V_B. Sky subtraction may
also partially cancel a common additive bias component; flat and sky
errors are not generally independent. Positive Huber weights are data
derived; the reference formula is conditional on fixed weights.

Acceptance will require a versioned component-variance contract,
operator-level propagation and calibrated coverage tests over complete
synthetic observing nights, then independent real SCORPIO comparisons.

Do not equate passing pytest with validation of final scientific S/N.
