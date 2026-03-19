from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scorpio_pipe.calib.association import (
    StandardCompatibilityStatus,
    assess_standard_association,
)
from scorpio_pipe.metadata.frame_meta import FrameMeta, ReadoutKey


def _meta(**kwargs) -> FrameMeta:
    base = dict(
        instrument="SCORPIO2",
        mode="SPECTRA",
        imagetyp="obj",
        disperser="VPHG1200@540",
        slit_width_arcsec=1.0,
        slit_pos=None,
        binning_x=1,
        binning_y=1,
        naxis1=2048,
        naxis2=2048,
        readout_key=ReadoutKey(node="0", rate=100.0, gain=1.0, rdnoise=3.0),
        date_time_utc=datetime(2026, 1, 1, 20, 0, 0, tzinfo=timezone.utc),
        object_name="target",
        frame_id="F0001",
        exptime=300.0,
        detector="",
        slitmask="",
        sperange="4000-7000",
        airmass=1.10,
        zenith_distance_deg=24.62,
        parallactic_angle_deg=None,
        ra_deg=180.0,
        dec_deg=20.0,
        utc_mid_exposure=datetime(2026, 1, 1, 20, 2, 30, tzinfo=timezone.utc),
        observatory="BTA",
        site_name="BTA",
        seeing_fwhm_arcsec=1.5,
    )
    base.update(kwargs)
    return FrameMeta(**base)


def test_standard_association_absolute_trusted_when_geometry_and_conditions_match() -> None:
    science = _meta(object_name="NGC1234")
    standard = _meta(object_name="BD+28 4211", imagetyp="obj")

    rep = assess_standard_association(science, standard, standard_role="standard")

    assert rep.status == StandardCompatibilityStatus.ABSOLUTE_TRUSTED
    assert rep.is_shape_compatible is True
    assert rep.is_absolute_compatible is True
    assert rep.downgrade_reasons == ()


def test_standard_association_downgrades_to_absolute_approx_on_moderate_mismatch() -> None:
    science = _meta(object_name="NGC1234", airmass=1.10)
    standard = _meta(
        object_name="BD+28 4211",
        date_time_utc=science.date_time_utc + timedelta(hours=3),
        airmass=1.32,
    )

    rep = assess_standard_association(science, standard, standard_role="standard")

    assert rep.status == StandardCompatibilityStatus.ABSOLUTE_APPROX
    assert rep.is_absolute_compatible is True
    assert "time_mismatch" in rep.downgrade_reasons
    assert "airmass_mismatch" in rep.downgrade_reasons


def test_standard_association_telluric_role_stays_relative_telluric_only() -> None:
    science = _meta(object_name="NGC1234")
    telluric = _meta(
        object_name="HR 1234",
        slit_width_arcsec=1.3,
        date_time_utc=science.date_time_utc + timedelta(hours=7),
        airmass=1.7,
    )

    rep = assess_standard_association(science, telluric, standard_role="telluric_standard")

    assert rep.status == StandardCompatibilityStatus.RELATIVE_TELLURIC_ONLY
    assert rep.is_shape_compatible is True
    assert rep.is_absolute_compatible is False
    assert "slit_width_mismatch" in rep.downgrade_reasons


def test_standard_association_rejects_shape_mismatch() -> None:
    science = _meta(object_name="NGC1234")
    standard = _meta(object_name="BD+28 4211", disperser="VPHG940@600")

    rep = assess_standard_association(science, standard, standard_role="standard")

    assert rep.status == StandardCompatibilityStatus.INCOMPATIBLE
    assert rep.is_shape_compatible is False
    assert rep.is_absolute_compatible is False
    assert "disperser_mismatch" in rep.downgrade_reasons
