from scorpio_pipe.long_setup_priors import get_long_setup_prior
from scorpio_pipe.operational_presets import apply_setup_preset_defaults


def test_long_setup_prior_lookup_and_autocfg_defaults():
    prior = get_long_setup_prior('VPHG1200@540')
    assert prior is not None
    assert 3880 <= prior.lambda_min_A <= 3900
    assert 0.7 <= prior.median_dispersion_A_per_px <= 1.0

    cfg = {
        'instrument': {'name': 'SCORPIO-2', 'disperser': 'VPHG1200@540', 'slit': '1.0', 'mode': 'Spectra'},
        'frames': {'__setup__': {'instrument': 'SCORPIO-2', 'disperser': 'VPHG1200@540', 'slit': '1.0', 'mode': 'Spectra'}},
    }
    out = apply_setup_preset_defaults(cfg)
    assert out['wavesol']['pair_tol_A'] == prior.suggested_pair_tol_A
    assert out['wavesol']['expected_dispersion_A_per_px'] == prior.median_dispersion_A_per_px
    assert out['linearize']['lambda_min_A'] == prior.trimmed_lambda_min_A
    assert out['linearize']['lambda_max_A'] == prior.trimmed_lambda_max_A
    assert out['lsf_model']['expected_fwhm_A_1arcsec'] == prior.empirical_resolution_fwhm_A_1arcsec
