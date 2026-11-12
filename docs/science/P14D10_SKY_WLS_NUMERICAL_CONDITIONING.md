# P1.4d10 — synthetic SVD vs normal-equation LU numeric audit

**Status:** opt-in read-only diagnostic; no science VAR changes, no actual
observations. Parent: independently Windows-validated PR #156.

## Methodological issue

`FrozenSkyWLS` builds the conditional residual projector from
`(Xfit.T @ W @ Xfit)^(-1)` via sparse LU on the normal equations.
Forming the normal matrix approximately **squares** the spectral
condition number `kappa(sqrt(W) Xfit)`. A formally nonsingular
factorization is therefore insufficient evidence that forward
and adjoint covariance actions are accurate near degeneracy.

## Independent numerical oracle

`calib.sky_wls_conditioning.audit_frozen_wls_numerics` densifies only
the **bounded weighted fitted-pixel design** and computes its
singular-value decomposition, independently of `FrozenSkyWLS`'s
LU factorization.

For reproducible Gaussian native-CCD directions, it compares:
- forward residual action `R @ v` (LU versus independent SVD);
- fitted model response `(I-R) @ v` (separate relative error, which
  cannot be hidden by the direct pixel noise);
- transpose/adjoint action `R.T @ u` (LU versus independent SVD);
- the duality identity `u.T @ R @ v = (R.T @ u).T @ v`.

The report records the singular values' effective rank, whitened
design condition number, estimated squared normal-equation condition,
maximal numerical discrepancies, probe seed, thresholds and
`science_uncertainty_qualified=false`.

`STABLE` requires full numerical rank, condition below a
user-specified conservative threshold, and forward/model/adjoint/
duality discrepancies below the requested numerical tolerance.
`ILL_CONDITIONED` and `NUMERIC_MISMATCH` prevent endorsement;
`RANK_DEFICIENT` is reported where an unregularized input operator
can be evaluated but SVD finds deficient numerical rank.

The default `max_condition=1e5` is an engineering diagnostic
threshold, **not** an observation-calibrated or instrument-specific
physics limit. A high condition number motivates a better solver,
reparameterization or basis/geometry correction, not an automatic
reduction of the formal statistical variance.

## Resource and science restrictions

The routine refuses to operate above 300,000 dense fit matrix entries
or 100,000 native CCD pixels. It does not construct a dense CCD by CCD
covariance matrix. Ridge-regularized operators are rejected, since
this SVD reference models the unregularized weighted least-squares
projector only.

This audit is designed for synthetic and already captured small
frozen geometry. It does not describe selection-dependent clipping
derivatives, uncertain calibration noise maps, correlated MasterFlat
errors, Poisson likelihoods, resampling or final 1-D statistics.
Production `SCI/VAR/MASK` and uncertainty flags remain unchanged.

Numerical linear algebra reference:
Higham, N. J. (2002), *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., SIAM, chapters on least squares and conditioning.
