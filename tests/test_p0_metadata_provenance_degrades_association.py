from __future__ import annotations

from datetime import datetime, timezone

from scorpio_pipe.calib.association import StandardCompatibilityStatus, assess_standard_association
from scorpio_pipe.metadata.frame_meta import FrameMeta, ReadoutKey


def _meta(**kwargs) -> FrameMeta:
    base = dict(
        instrument='SCORPIO2',
        mode='SPECTRA',
        imagetyp='obj',
        disperser='VPHG1200@540',
        slit_width_arcsec=1.0,
        slit_pos=None,
        binning_x=1,
        binning_y=1,
        naxis1=2048,
        naxis2=2048,
        readout_key=ReadoutKey(node='0', rate=100.0, gain=1.0, rdnoise=3.0),
        date_time_utc=datetime(2026, 1, 1, 20, 0, 0, tzinfo=timezone.utc),
        object_name='x',
        frame_id='F1',
        exptime=300.0,
        detector='',
        slitmask='',
        sperange='4000-7000',
        airmass=1.1,
        zenith_distance_deg=24.6,
        parallactic_angle_deg=None,
        ra_deg=180.0,
        dec_deg=20.0,
        utc_mid_exposure=datetime(2026, 1, 1, 20, 2, 30, tzinfo=timezone.utc),
        observatory='BTA',
        site_name='BTA',
        seeing_fwhm_arcsec=1.2,
        meta_provenance={'airmass': 'header:AIRMASS', 'site_name': 'header:SITENAME', 'observatory': 'header:OBSERVAT'},
    )
    base.update(kwargs)
    return FrameMeta(**base)


def test_fallback_airmass_prevents_absolute_trusted() -> None:
    science = _meta()
    standard = _meta(meta_provenance={'airmass': 'fallback:config', 'site_name': 'header:SITENAME', 'observatory': 'header:OBSERVAT'})
    rep = assess_standard_association(science, standard, standard_role='standard')
    assert rep.status != StandardCompatibilityStatus.ABSOLUTE_TRUSTED
    assert 'airmass_provenance_degraded' in rep.downgrade_reasons
