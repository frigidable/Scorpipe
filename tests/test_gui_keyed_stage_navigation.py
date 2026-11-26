from __future__ import annotations

import pytest

from scorpio_pipe.catalog.stages import CATALOG
from scorpio_pipe.ui.stage_navigation import build_page_index


def test_page_index_uses_keys_not_sidebar_order() -> None:
    keys = [s.key for s in CATALOG.gui_stages()]
    # Existing pages are intentionally constructed in a different order from
    # the canonical scientific stage registry.
    pages = {i: (key, f"page_{key}", None) for i, key in enumerate(reversed(keys))}
    result = build_page_index(pages, keys)
    assert result["project"] == 11
    assert result["extract1d"] == 0
    assert result["flatfield"] != keys.index("flatfield")


def test_sidebar_reordering_does_not_change_stage_page_mapping() -> None:
    pages = {
        0: ("project", "page_project", None),
        1: ("cosmics", "page_cosmics", None),
        2: ("flatfield", "page_flat", None),
    }
    a = build_page_index(pages, ["project", "flatfield", "cosmics"])
    b = build_page_index(pages, ["cosmics", "project", "flatfield"])
    assert a == b == {"project": 0, "cosmics": 1, "flatfield": 2}


@pytest.mark.parametrize(
    ("pages", "keys"),
    [
        ({0: ("project", "", None)}, ["project", "setup"]),
        ({0: ("project", "", None), 1: ("setup", "", None)}, ["project"]),
        ({0: ("project", "", None), 1: ("project", "", None)}, ["project"]),
        ({0: ("project", "", None), 2: ("setup", "", None)}, ["project", "setup"]),
        ({-1: ("project", "", None)}, ["project"]),
        ({0: ("project", "", None)}, ["project", "project"]),
    ],
)
def test_mapping_fails_closed_on_invalid_definitions(pages, keys) -> None:
    with pytest.raises(ValueError):
        build_page_index(pages, keys)
