from __future__ import annotations


def test_line_measure_preview_is_registered_consistently() -> None:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCTS
    from scorpio_pipe.catalog.stages import CATALOG as STAGES

    assert 'science_lines_preview_png' in set(PRODUCTS.keys())
    assert 'science_lines_preview_png' in set(STAGES.get_task('line_measure').product_keys)


def test_public_api_dashboard_reader_can_reference_line_preview_product() -> None:
    from scorpio_pipe.pipeline.public_api import build_observer_dashboard
    dash = build_observer_dashboard({'work_dir': '.', 'frames': {}})
    assert isinstance(dash, dict)
