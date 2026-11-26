"""Prevent duplicate canonical stages and incorrect science recipe ordering.

Flatfield may consume Cosmics CLEAN products. The first execution must not
silently differ from a rerun solely because Flatfield preceded Cosmics.
"""
from __future__ import annotations

import pytest

from scorpio_pipe.catalog.stages import CATALOG as STAGE_CATALOG
from scorpio_pipe.pipeline.recipes import RECIPES, recipe_tasks


DEPENDENCIES_IF_BOTH_PRESENT = (
    ("superbias", "superflat"),
    ("superbias", "cosmics"),
    ("superflat", "flatfield"),
    ("cosmics", "flatfield"),
    ("flatfield", "sky"),
    ("wavesolution", "sky"),
    ("sky", "sky_refine"),
    ("sky_refine", "linearize"),
    ("linearize", "stack2d"),
    ("stack2d", "extract1d"),
)


@pytest.mark.parametrize("recipe_name", sorted(RECIPES))
def test_unique_canonical_tasks(recipe_name: str) -> None:
    canonical = [
        STAGE_CATALOG.canonical_task_name(t)
        for t in recipe_tasks(recipe_name)
    ]
    assert len(canonical) == len(set(canonical)), (
        f"{recipe_name} repeats canonical stages: {canonical}"
    )


@pytest.mark.parametrize("recipe_name", sorted(RECIPES))
def test_prerequisites_before_consumers(recipe_name: str) -> None:
    canonical = [
        STAGE_CATALOG.canonical_task_name(t)
        for t in recipe_tasks(recipe_name)
    ]
    positions = {task: i for i, task in enumerate(canonical)}
    for before, after in DEPENDENCIES_IF_BOTH_PRESENT:
        if before in positions and after in positions:
            assert positions[before] < positions[after], (
                f"{recipe_name}: {before} must precede {after}"
            )


@pytest.mark.parametrize(
    "recipe_name", ["full_reduction", "full_science_calibration_report"]
)
def test_full_science_cosmics_precedes_flatfield(recipe_name: str) -> None:
    tasks = [STAGE_CATALOG.canonical_task_name(t) for t in recipe_tasks(recipe_name)]
    assert tasks.index("cosmics") < tasks.index("flatfield")
