"""Bounded, read-only bridge: captured Kelson sky fit + real FlatField noise.

This diagnostic binds actual same-run FITS to a *frozen* WLS operator and
evaluates the group-shared MasterBias term. VREST is an as-if-independent
component ONLY: shared MasterFlat, detector covariance, clipping/LSQR
iteration derivatives, and all downstream stages are NOT included.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.calib.sky_bias_covariance import (
    FrozenSkyWLS,
    covariance_of_frozen_sky_grouped,
)
from scorpio_pipe.calib.sky_fit_capture import load_frozen_kelson_fit
from scorpio_pipe.calib.sky_fit_lineage import audit_frozen_kelson_fit_lineage
from scorpio_pipe.calib.sky_masterbias_source_audit import (
    audit_sky_masterbias_group_sources,
)


@dataclass(frozen=True)
class CapturedSkyBiasConditionalBasis:
    """Authenticated real-run input planes for a *conditional*, limited oracle."""

    operators: tuple[FrozenSkyWLS, ...]
    vrest_if_independent: tuple[np.ndarray, ...]
    signed_bias_sensitivities: tuple[np.ndarray, ...]
    group_ids: tuple[str, ...]
    masterbias_variances: dict[str, np.ndarray]
    ccd_shapes: tuple[tuple[int, int], ...]
    status: str = "HOLD_CONDITIONAL_SOURCE_NOISE_NOT_FULL_COVARIANCE"
    science_uncertainty_qualified: bool = False
    production_variance_release_allowed: bool = False


def load_captured_sky_bias_conditional_basis(
    lineage_manifests: Sequence[Path],
    *,
    max_pixels_per_exposure: int = 131_072,
    max_exposures: int = 8,
) -> CapturedSkyBiasConditionalBasis:
    """Bind actual captured FITS into an explicitly resource-bounded oracle.

    Requires the selected input to be FlatField SCI/VAR with the complete
    physically validated component schema. A source with no components
    fails closed rather than assuming a component from the total VAR.

    Default cap intentionally excludes ordinary large science CCD frames:
    a production-scale sparse WLS fit needs a separate memory/time budget.
    """
    if (type(max_pixels_per_exposure) is not int or max_pixels_per_exposure <= 0
            or type(max_exposures) is not int or max_exposures <= 0):
        raise ValueError("Conditional sky-bias bridge needs positive integer resource budgets")
    audit = audit_sky_masterbias_group_sources(
        lineage_manifests,
        max_pixels_per_exposure=max_pixels_per_exposure,
        max_exposures=max_exposures,
    )
    if (audit["n_sources_without_verified_component_planes"] != 0
            or audit["status"] != "HOLD_PRE_SKY_VARIANCE_COMPONENTS_PRESENT_BUT_NOT_PROPAGATED"):
        raise ValueError("Conditional sky-bias bridge requires complete source FlatField components")
    ops: list[FrozenSkyWLS] = []
    independent_assumption: list[np.ndarray] = []
    sensitivities: list[np.ndarray] = []
    ids: list[str] = []
    maps: dict[str, np.ndarray] = {}
    shapes: list[tuple[int, int]] = []
    for source_entry, manifest in zip(audit["source_entries"], lineage_manifests):
        linked = audit_frozen_kelson_fit_lineage(
            Path(manifest), check_fit_against_products=True,
        )
        path = Path(linked["selected_sky_input"])
        with fits.open(path, memmap=False) as hdul:
            var = np.asarray(hdul["VAR"].data, dtype=np.float64)
            vr = np.asarray(hdul["VREST"].data, dtype=np.float64).copy()
            sens = np.asarray(hdul["BSENS"].data, dtype=np.float64).copy()
            bvar = np.asarray(hdul["BVARADU"].data, dtype=np.float64).copy()
        if any(x.shape != var.shape for x in (vr, sens, bvar)):
            raise ValueError("Conditional input variance or bias response geometry mismatch")
        if var.size > max_pixels_per_exposure:
            raise ValueError("Frozen source exceeds bridge's explicit sparse WLS pixel budget")
        with np.load(linked["fit_archive"], allow_pickle=False) as fit:
            selected = np.asarray(fit["fit_indices"])
            selection_var = np.asarray(fit["fit_variance"], dtype=np.float64)
            lam_shape = np.asarray(fit["lambda_map"]).shape
        if var.shape != lam_shape:
            raise ValueError("Kelson fit and actual selected FlatField CCD geometries differ")
        # Critical: ensure the actual fit did not use a new noise estimate,
        # implicit gain scaling or a different input VAR from the source.
        if not np.allclose(
            var.ravel()[selected], selection_var, rtol=5e-7, atol=5e-7,
        ):
            raise ValueError("Actual Kelson final fit variance differs from selected source VAR")
        op, _coefficients = load_frozen_kelson_fit(Path(linked["fit_archive"]))
        if op.n_pixels != var.size:
            raise ValueError("Captured Kelson conditional operator/source geometry mismatch")
        gid = source_entry["masterbias_group"]
        if not isinstance(gid, str) or not gid:
            raise ValueError("Missing original MasterBias calibration group")
        if gid in maps:
            if not np.array_equal(maps[gid], bvar.ravel()):
                raise ValueError("Shared group has inconsistent MasterBias source map")
        else:
            maps[gid] = bvar.ravel().copy()
        ops.append(op)
        independent_assumption.append(vr.ravel())
        sensitivities.append(sens.ravel())
        ids.append(gid)
        shapes.append(var.shape)
    return CapturedSkyBiasConditionalBasis(
        operators=tuple(ops),
        vrest_if_independent=tuple(independent_assumption),
        signed_bias_sensitivities=tuple(sensitivities),
        group_ids=tuple(ids),
        masterbias_variances=maps,
        ccd_shapes=tuple(shapes),
    )


def evaluate_captured_sky_bias_conditional_pair(
    basis: CapturedSkyBiasConditionalBasis,
    left_ccd_functionals: Sequence[np.ndarray],
    right_ccd_functionals: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Conditional *sky-output* covariance, not COUNT_V1/Stack2D/1D.

    A user must supply functionals on SkySub's true CCD output geometry.
    VREST is treated independent for the numerical oracle; its physical
    independence from other exposures has NOT been established.
    """
    if not isinstance(basis, CapturedSkyBiasConditionalBasis):
        raise ValueError("Only the audited captured SkySub basis is accepted")
    if basis.science_uncertainty_qualified or basis.production_variance_release_allowed:
        raise ValueError("Captured sky basis must never be science-qualified")
    if (len(left_ccd_functionals) != len(basis.operators)
            or len(right_ccd_functionals) != len(basis.operators)):
        raise ValueError("One complete CCD functional required per captured exposure")
    left, right = [], []
    for i, (a, b, shape) in enumerate(zip(
        left_ccd_functionals, right_ccd_functionals, basis.ccd_shapes,
    )):
        x = np.asarray(a, dtype=np.float64)
        y = np.asarray(b, dtype=np.float64)
        if (x.shape != shape or y.shape != shape
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError(f"Conditional CCD functional geometry/values invalid: exposure {i}")
        left.append(x.ravel())
        right.append(y.ravel())
    vrest_if_independent, shared_bias, total = covariance_of_frozen_sky_grouped(
        basis.operators,
        left,
        right,
        basis.vrest_if_independent,
        basis.signed_bias_sensitivities,
        basis.group_ids,
        basis.masterbias_variances,
    )
    return {
        "status": "HOLD_CONDITIONAL_SOURCE_NOISE_NOT_FULL_COVARIANCE",
        "vrest_as_if_independent_term": vrest_if_independent,
        "signed_shared_MasterBias_term": shared_bias,
        "conditional_sum_under_unverified_independence": total,
        "n_exposures": len(basis.operators),
        "masterbias_group_ids": list(basis.group_ids),
        "frozen_sky_model": "WLS_NORMAL_EQUATIONS_NOT_FINITE_LSQR_DERIVATIVE",
        "downstream_COUNT_V1_Stack2D_Extract1D": "NOT_BOUND",
        "shared_MasterFlat_and_VREST_off_diagonal": "NOT_MODELLED",
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
