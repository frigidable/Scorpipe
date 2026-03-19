from __future__ import annotations

from scorpio_pipe.pipeline.recipes import RECIPES, get_recipe, recipe_tasks


def test_full_science_recipe_contains_science_tail() -> None:
    r = get_recipe('full_science_calibration_report')
    assert r.tasks[-4:] == ('line_measure', 'science_report', 'qc_report', 'observer_dashboard')
    assert 'standard_prepare' in r.tasks
    assert 'fluxcal_apply' in r.tasks


def test_recipe_tasks_known() -> None:
    tasks = recipe_tasks('quicklook_science')
    assert tasks[0] == 'manifest'
    assert 'lsf_model' in tasks
    assert 'publication_ready' in tasks
    assert 'kinematics_prep' in tasks
    assert 'relative_flux_calibration' in RECIPES
