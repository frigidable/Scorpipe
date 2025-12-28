"""Stage registry (single source of truth for stage ids, labels and directories).

Goal
----
We keep *one* canonical mapping:

    GUI ↔ stage_key ↔ stage_id ↔ products/NN_slug

The registry is intentionally small and stable. If you add/reorder stages,
update this module and only then update callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StageSpec:
    """Single stage definition.

    Fields
    ------
    id
        Integer stage id. Directory prefix uses two digits (NN_).
    key
        Stable internal key used by the runner and config (e.g. "linearize").
    slug
        Short directory slug (e.g. "lin").
    label
        Human-facing label for GUI.
    dir_name
        Canonical directory name under work_dir/products.
    """

    id: int
    key: str
    slug: str
    label: str
    dir_name: str


def _mk(id_: int, key: str, slug: str, label: str) -> StageSpec:
    return StageSpec(id=id_, key=key, slug=slug, label=label, dir_name=f"{id_:02d}_{slug}")


# NOTE: Keep ids stable. The canonical directory is products/NN_slug.
# This is the *only* place where the order and ids are defined.
STAGES: tuple[StageSpec, ...] = (
    _mk(0, "manifest", "manifest", "Manifest"),
    _mk(1, "superbias", "superbias", "SuperBias"),
    _mk(2, "superflat", "superflat", "SuperFlat"),
    _mk(3, "flatfield", "flatfield", "Flatfield"),
    _mk(4, "cosmics", "cosmics", "Cosmics"),
    _mk(5, "superneon", "superneon", "SuperNeon"),
    _mk(6, "lineid_prepare", "lineid", "LineID prepare"),
    _mk(7, "wavesolution", "wavesol", "Wavelength solution"),
    _mk(8, "linearize", "lin", "Linearize"),
    _mk(9, "sky", "sky", "Sky subtraction"),
    # Keep numeric id stable; use explicit slug to avoid ambiguity.
    _mk(10, "stack2d", "stack2d", "Stack2D"),
    _mk(11, "extract1d", "spec", "Extract1D"),
    _mk(12, "qc_report", "qc", "QC report"),
)


class StageRegistry:
    """Lookup helpers for stage metadata."""

    def __init__(self, stages: Iterable[StageSpec] = STAGES):
        self._stages = tuple(stages)
        self._by_key = {s.key: s for s in self._stages}
        self._by_id = {s.id: s for s in self._stages}

    def all(self) -> tuple[StageSpec, ...]:
        return self._stages

    def get(self, key: str) -> StageSpec:
        k = (key or "").strip().lower()
        if k not in self._by_key:
            raise KeyError(f"Unknown stage key: {key!r}")
        return self._by_key[k]

    def by_id(self, id_: int) -> StageSpec:
        if int(id_) not in self._by_id:
            raise KeyError(f"Unknown stage id: {id_!r}")
        return self._by_id[int(id_)]

    def dir_name(self, key: str) -> str:
        return self.get(key).dir_name

    def label(self, key: str) -> str:
        return self.get(key).label

    def stage_id(self, key: str) -> int:
        return int(self.get(key).id)

    def keys(self) -> tuple[str, ...]:
        return tuple(s.key for s in self._stages)


# Public singleton.
REGISTRY = StageRegistry(STAGES)
