from __future__ import annotations

from scorpio_pipe.catalog.presentation import (
    GUI_STAGE_HELP,
    GUI_STAGE_SECTIONS,
    SECTIONS,
    TASK_HELP,
    TASK_SECTIONS,
    get_section,
    gui_stage_presentation,
    gui_stages_in_section,
    task_presentation,
)
from scorpio_pipe.catalog.stages import CATALOG
from scorpio_pipe.stage_registry import REGISTRY


def test_all_legacy_gui_stages_have_presentation_metadata() -> None:
    expected = {s.key for s in CATALOG.gui_stages()}
    assert set(GUI_STAGE_SECTIONS) == expected
    assert set(GUI_STAGE_HELP) == expected
    for stage in CATALOG.gui_stages():
        item = gui_stage_presentation(stage.key)
        assert item.key == stage.key
        assert item.label == stage.label
        assert item.description.strip()
        assert get_section(item.section_key).label.strip()


def test_all_engine_tasks_have_presentation_metadata() -> None:
    expected = set(CATALOG.task_keys())
    assert len(expected) == 27
    assert set(TASK_SECTIONS) == expected
    assert set(TASK_HELP) == expected
    for task in CATALOG.task_infos():
        item = task_presentation(task.key)
        assert item.key == task.key
        assert item.label == task.label
        assert item.description.strip()
        assert get_section(item.section_key).label.strip()
        if task.stage_key is not None:
            assert item.section_key == GUI_STAGE_SECTIONS[task.stage_key]


def test_sections_are_nonexecuting_navigation_groups() -> None:
    assert tuple(s.key for s in SECTIONS) == (
        "observations",
        "detector",
        "spectral",
        "science",
        "spectrophotometry",
        "analysis",
    )
    all_keys: list[str] = []
    for section in SECTIONS:
        group = gui_stages_in_section(section.key)
        assert all(item.section_key == section.key for item in group)
        all_keys.extend(item.key for item in group)
    assert len(all_keys) == len(set(all_keys)) == len(CATALOG.gui_stages())
    assert set(all_keys) == {s.key for s in CATALOG.gui_stages()}
    assert set(s.key for s in SECTIONS).isdisjoint(set(CATALOG.task_keys()))


def test_renaming_does_not_change_workspace_directory_contract() -> None:
    assert [s.dir_name for s in REGISTRY.all()] == [
        "01_project",
        "02_setup",
        "03_bias",
        "04_flat",
        "05_cosmics",
        "06_superneon",
        "07_arc_line_id",
        "08_wavesol",
        "09_sky",
        "10_linearize",
        "11_stack",
        "12_extract",
    ]


def test_display_labels_do_not_replace_identifiers_or_aliases() -> None:
    assert CATALOG.canonical_task_name("biascorr") == "superbias"
    assert CATALOG.canonical_task_name("arclineid") == "lineid_prepare"
    assert CATALOG.get_gui_stage("wavesolution").key == "wavesol"
    assert gui_stage_presentation("biascorr").label == "Bias Correction"
    assert task_presentation("biascorr").label == "Master Bias Construction"
    assert gui_stage_presentation("linearize").label == "Spectral Resampling"
    assert task_presentation("publication_ready").label == "Science Product Export"


def test_presentation_does_not_change_recipe_identity_or_order() -> None:
    from scorpio_pipe.pipeline.recipes import recipe_tasks

    # Pin the existing vNext recipe, not the obsolete historical #175 base.
    # Display labels cannot introduce a new biascorr task or reorder execution.
    assert recipe_tasks("full_reduction")[:4] == (
        "manifest", "superbias", "superflat", "cosmics",
    )
    assert recipe_tasks("full_reduction")[-2:] == (
        "qc_report", "observer_dashboard",
    )
