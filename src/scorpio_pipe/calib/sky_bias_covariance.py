"""Frozen weighted sky-fit residual operator and shared-bias covariance.

Opt-in mathematical diagnostic. Not attached to the production SkySub runner.
Fixed training support, weights, basis and output geometry are REQUIRED.
The full covariance for data-selected clipping weights is not computed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


class FrozenSkyWLS:
    """R = I - Xall (Xfit.T W Xfit + ridge I)^-1 Xfit.T W S.

    S extracts observed sky pixels by stable, unique flattened indices.
    No dense output-pixel x output-pixel matrix is constructed.
    """

    def __init__(
        self,
        design: sparse.spmatrix | np.ndarray,
        fit_indices: np.ndarray,
        fit_weights: np.ndarray,
        *,
        ridge: float = 0.0,
    ) -> None:
        x = sparse.csr_matrix(design, dtype=np.float64)
        if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x.data).all():
            raise ValueError("Sky design must be nonempty and finite")
        index = np.asarray(fit_indices)
        if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
            raise ValueError("Sky fit indices must be a one-dimensional integer array")
        if (index.size == 0 or np.any(index < 0) or np.any(index >= x.shape[0])
                or np.unique(index).size != index.size):
            raise ValueError("Sky fit indices must be nonempty, unique, and within geometry")
        w = np.asarray(fit_weights, dtype=np.float64)
        if (w.shape != index.shape or not np.isfinite(w).all()
                or np.any(w <= 0)):
            raise ValueError("Sky fit weights must be positive finite samples")
        regularizer = float(ridge)
        if not np.isfinite(regularizer) or regularizer < 0:
            raise ValueError("Sky fit ridge must be finite and nonnegative")
        if index.size < x.shape[1] and regularizer == 0:
            raise ValueError("Underdetermined frozen sky fit without ridge")
        fit = x[index].tocsr()
        normal = (fit.T @ fit.multiply(w[:, None])).tocsc()
        if regularizer:
            normal = normal + sparse.eye(x.shape[1], format="csc") * regularizer
        try:
            lu = splu(normal)
        except RuntimeError as exc:
            raise ValueError("Sky fit normal matrix is singular") from exc
        self.design = x
        self.fit_indices = index.astype(np.intp, copy=True)
        self.fit_weights = w.copy()
        self.fit_design = fit
        self.ridge = regularizer
        self._factor = lu

    @property
    def n_pixels(self) -> int:
        return int(self.design.shape[0])

    def _vector(self, values: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (self.n_pixels,) or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be a finite full-grid sky vector")
        return arr

    def apply(self, perturbation: np.ndarray) -> np.ndarray:
        """Forward derivative of sky-subtracted data w.r.t. input data."""
        x = self._vector(perturbation, "Sky perturbation")
        rhs = self.fit_design.T @ (self.fit_weights * x[self.fit_indices])
        coeff = self._factor.solve(np.asarray(rhs).ravel())
        return x - np.asarray(self.design @ coeff).ravel()

    def adjoint(self, measurement: np.ndarray) -> np.ndarray:
        """Pull an output functional back to pre-sky detector pixels."""
        u = self._vector(measurement, "Sky output functional")
        rhs = self.design.T @ u
        coeff = self._factor.solve(np.asarray(rhs).ravel(), trans="T")
        contribution = self.fit_weights * np.asarray(self.fit_design @ coeff).ravel()
        result = u.copy()
        np.add.at(result, self.fit_indices, -contribution)
        return result


def frozen_kelson_bspline_operator(
    wavelength: np.ndarray,
    y_normalized_rows: np.ndarray,
    *,
    knots: np.ndarray,
    degree: int,
    y_order: int,
    final_fit_mask: np.ndarray,
    variance: np.ndarray,
) -> FrozenSkyWLS:
    """Build WLS operator from frozen B-spline geometry and FINAL sky selection.

    The SkySub production fit currently uses scipy.sparse.linalg.lsqr,
    data-derived clipping and possibly sky_scale_raw. This diagnostic models
    the converged, frozen fixed-weight least-squares solution only. The
    exact production coefficients and clipping mask are NOT auto-captured.
    """
    from scipy.sparse import diags, hstack
    from scorpio_pipe.stages.sky_sub import _bspline_design_matrix

    lam = np.asarray(wavelength, dtype=np.float64)
    if lam.ndim != 2 or min(lam.shape) == 0 or not np.isfinite(lam).all():
        raise ValueError("Sky wavelength map must be finite nonempty 2-D")
    ny, nx = lam.shape
    yn = np.asarray(y_normalized_rows, dtype=np.float64)
    if yn.shape != (ny,) or not np.isfinite(yn).all():
        raise ValueError("Normalized sky y-coordinate must have one value per row")
    mask = np.asarray(final_fit_mask)
    if mask.shape != lam.shape or mask.dtype != np.dtype(bool):
        raise ValueError("Frozen final sky fit mask must be Boolean 2-D geometry")
    var = np.asarray(variance, dtype=np.float64)
    if var.shape != lam.shape:
        raise ValueError("Frozen sky fit variance must match CCD geometry")
    good = np.flatnonzero(mask.ravel())
    if good.size == 0 or np.any(~np.isfinite(var.ravel()[good]) | (var.ravel()[good] <= 0)):
        raise ValueError("Positive finite sky variance required at fitted pixels")
    if int(y_order) not in (0, 1):
        raise ValueError("Only actual SkySub y_order=0/1 is supported")
    basis = _bspline_design_matrix(
        lam.ravel(), np.asarray(knots, dtype=np.float64), int(degree),
    )
    if int(y_order) == 1:
        ys = np.repeat(yn, nx)
        basis = hstack(
            [basis, diags(ys, format="csr") @ basis], format="csr"
        )
    return FrozenSkyWLS(basis, good, 1.0 / var.ravel()[good])


def covariance_of_frozen_sky_grouped(
    sky_operators: Sequence[FrozenSkyWLS],
    left_functionals: Sequence[np.ndarray],
    right_functionals: Sequence[np.ndarray],
    independent_variances: Sequence[np.ndarray],
    bias_sensitivities: Sequence[np.ndarray],
    group_ids: Sequence[str],
    masterbias_variances: Mapping[str, np.ndarray],
) -> tuple[float, float, float]:
    """Exact fixed-fit conditional covariance, excluding nonlinear sky selection.

    Each exposure has its OWN fitted sky residual operator R_e. A common
    detector-pixel MasterBias error field b_g is seen by all exposures in
    group g through (R_e @ diag(sensitivity_e)) b_g. For a 1-D functional
    u_e, back-projected coefficients are sensitivity_e * R_e.T @ u_e.
    """
    n = len(sky_operators)
    if n == 0 or any(len(item) != n for item in (
        left_functionals, right_functionals,
        independent_variances, bias_sensitivities, group_ids,
    )):
        raise ValueError("One operator, functional, variance, sensitivity and group per exposure")
    ids = [str(g).strip() for g in group_ids]
    if not all(ids) or set(ids) != set(masterbias_variances):
        raise ValueError("Explicit complete shared MasterBias group mapping required")
    independent = 0.0
    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for op, left, right, v_ind, sensitivity, gid in zip(
        sky_operators, left_functionals, right_functionals,
        independent_variances, bias_sensitivities, ids,
    ):
        u = op.adjoint(left)
        v = op.adjoint(right)
        var = np.asarray(v_ind, dtype=np.float64)
        s = np.asarray(sensitivity, dtype=np.float64)
        if (var.shape != (op.n_pixels,) or s.shape != var.shape
                or not np.isfinite(var).all() or np.any(var < 0)
                or not np.isfinite(s).all()):
            raise ValueError("Pre-sky independent variance/sensitivity geometry invalid")
        independent += float(np.dot(var, u * v))
        if gid not in grouped:
            grouped[gid] = (np.zeros_like(u), np.zeros_like(v))
        gleft, gright = grouped[gid]
        if gleft.shape != u.shape:
            raise ValueError("Same-group MasterBias CCD geometry mismatch")
        gleft += s * u
        gright += s * v
    shared = 0.0
    for gid, (gleft, gright) in grouped.items():
        bvar = np.asarray(masterbias_variances[gid], dtype=np.float64)
        if (bvar.shape != gleft.shape or not np.isfinite(bvar).all()
                or np.any(bvar < 0)):
            raise ValueError("MasterBias group variance must match CCD geometry")
        shared += float(np.dot(bvar, gleft * gright))
    total = float(independent + shared)
    if not np.isfinite(total) or not np.isfinite(shared):
        raise ValueError("Nonfinite frozen-sky covariance")
    return float(independent), float(shared), total
