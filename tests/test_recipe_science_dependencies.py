"""Science-dependency invariants for observer recipes.

Flatfield opportunistically reads an existing Cosmics CLEAN product.
If Flatfield runs before Cosmics, first-run output can differ from reruns
on an unchanged raw dataset. These tests protect the explicit execution
order, without claiming to solve all cross-run provenance requirements.
"""

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
def test_recipe_contains_each_canonical_stage_at_most_once(recipe_name):
    tasks = recipe_tasks(recipe_name)
    canonical = [STAGE_CATALOG.canonical_task_name(t) for t in tasks]
    assert len(canonical) == len(set(canonical)), (
        f"{recipe_name}: duplicate canonical stage or alias: {canonical}"
    )


@pytest.mark.parametrize("recipe_name", sorted(RECIPES))
def test_recipe_respects_order_of_present_prerequisites(recipe_name):
    canonical = [
        STAGE_CATALOG.canonical_task_name(t)
        for t in recipe_tasks(recipe_name)
    ]
    position = {task: index for index, task in enumerate(canonical)}
    for prerequisite, downstream in DEPENDENCIES_IF_BOTH_PRESENT:
        if prerequisite in position and downstream in position:
            assert position[prerequisite] < position[downstream], (
                f"{recipe_name}: {prerequisite} must run before {downstream}"
            )


@pytest.mark.parametrize(
    "recipe_name", ["full_reduction", "full_science_calibration_report"]
)
def test_full_science_recipes_produce_clean_before_flat(recipe_name):
    canonical = [
        STAGE_CATALOG.canonical_task_name(t)
        for t in recipe_tasks(recipe_name)
    ]
    assert canonical.index("cosmics") < canonical.index("flatfield")
