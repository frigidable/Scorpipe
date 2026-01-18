"""Variance propagation contract.

This is a thin contract layer over :mod:`scorpio_pipe.variance_model`.

Keeping it in :mod:`scorpio_pipe.contracts` makes downstream code depend on a
stable "law" path, even if internal implementations move.
"""

from __future__ import annotations

from scorpio_pipe.variance_model import (  # noqa: F401
    VarQC,
    ensure_var_contract,
    estimate_ccd_variance,
    median_reduced_chi2,
    propagate_add,
    propagate_divide,
    propagate_multiply,
    propagate_sub,
    propagate_weighted_mean,
)
