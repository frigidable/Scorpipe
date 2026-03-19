from __future__ import annotations


def test_product_catalog_registers_p2_products() -> None:
    from scorpio_pipe.catalog.products import CATALOG

    expected = {
        'sensfunc_qc_json',
        'telluric_preview_png',
        'fluxcal_preview_png',
        'calibration_systematics_fits',
        'sky_refine_done',
        'sky_residual_windows_json',
        'sky_observer_pack_md',
    }
    assert expected.issubset(set(CATALOG.keys()))


def test_stage_catalog_p2_products_are_declared() -> None:
    from scorpio_pipe.catalog.stages import CATALOG

    assert 'sensfunc_qc_json' in set(CATALOG.get_task('sensfunc').product_keys)
    assert 'telluric_preview_png' in set(CATALOG.get_task('telluric_model').product_keys)
    assert 'fluxcal_preview_png' in set(CATALOG.get_task('fluxcal_apply').product_keys)
    assert 'sky_refine_done' in set(CATALOG.get_task('sky_refine').product_keys)
    assert 'sky_residual_windows_json' in set(CATALOG.get_task('sky_refine').product_keys)
