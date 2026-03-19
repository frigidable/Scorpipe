from __future__ import annotations

import pytest

from scorpio_pipe.contracts.calibration import (
    CalibrationProductKind,
    CalibrationStatus,
    calibration_statuses_for_product_kind,
    coerce_calibration_status,
    validate_calibration_claim,
)


def test_mef2d_cannot_claim_absolute_trusted() -> None:
    with pytest.raises(ValueError):
        validate_calibration_claim(CalibrationStatus.ABSOLUTE_TRUSTED.value, product_kind=CalibrationProductKind.MEF2D)


def test_spec1d_may_claim_absolute_trusted() -> None:
    got = validate_calibration_claim(CalibrationStatus.ABSOLUTE_TRUSTED.value, product_kind=CalibrationProductKind.SPEC1D)
    assert got == CalibrationStatus.ABSOLUTE_TRUSTED.value


def test_coerce_mef2d_trusted_to_approx() -> None:
    assert coerce_calibration_status(CalibrationStatus.ABSOLUTE_TRUSTED.value, product_kind=CalibrationProductKind.MEF2D) == CalibrationStatus.ABSOLUTE_APPROX.value
