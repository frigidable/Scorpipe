"""Stage registry (single source of truth).

This module defines the *only* canonical list of GUI/pipeline stages.

Contract (v5.39.1)
------------------
- Stage numbering is fixed: 01..12.
- Canonical stage output directories live directly under the workspace/run root:

    run_root/NN_slug/

- UI-only stages (01..02) are displayed in the GUI but never produce outputs.

Stage order change
------------------
v5.39.1 moves **Sky Subtraction** *before* **Linearization**:

    ... 08_wavesol -> 09_sky -> 10_linearize -> 11_stack -> 12_extract

Backward compatibility
----------------------
Older runs (v5.38.x) used:

    09_sky -> 10_linearize

We keep stage *keys* stable ("sky", "linearize"), and :func:`scorpio_pipe.workspace_paths.stage_dir`
is responsible for resolving legacy directories when opening older runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class StageDef:
    """Single stage definition."""

    id: int
    key: str
    slug: str
    label: str
    ui_only: bool = False

    @property
    def dir_name(self) -> str:
        return f"{int(self.id):02d}_{self.slug}"

    @property
    def title(self) -> str:
        """Human title with numbering: ``NN Label``."""

        return f"{int(self.id):02d} {self.label}"


# -----------------------------------------------------------------------------
# Canonical stage table (DO NOT reorder without a migration plan).
# Labels MUST match GUI strings exactly.
# -----------------------------------------------------------------------------
STAGES: tuple[StageDef, ...] = (
    StageDef(1, "project", "project", "Project", ui_only=True),
    StageDef(2, "setup", "setup", "Setup", ui_only=True),
    StageDef(3, "biascorr", "bias", "Bias Correction"),
    StageDef(4, "flatfield", "flat", "Flat-Fielding"),
    StageDef(5, "cosmics", "cosmics", "Cosmics Cleaning"),
    StageDef(6, "superneon", "superneon", "Superneon"),
    StageDef(7, "arclineid", "arc_line_id", "Arc Line ID"),
    StageDef(8, "wavesol", "wavesol", "Wavelength Solution"),
    StageDef(9, "sky", "sky", "Sky Subtraction"),
    StageDef(10, "linearize", "linearize", "Linearization"),
    StageDef(11, "stack2d", "stack", "Frame Stacking"),
    StageDef(12, "extract1d", "extract", "Object Extraction"),
)


# Compatibility aliases (old keys / task names -> canonical stage keys).
#
# Important: this maps *tasks/sections* to stages, not legacy directory names.
ALIASES: dict[str, str] = {
    # UI-only placeholders
    "project": "project",
    "setup": "setup",
    # Calibs / flats
    "superbias": "biascorr",
    "bias": "biascorr",
    "biascorr": "biascorr",
    "superflat": "flatfield",
    "flat": "flatfield",
    "flatfield": "flatfield",
    # Others
    "cosmics": "cosmics",
    "superneon": "superneon",
    "lineid": "arclineid",
    "lineid_prepare": "arclineid",
    "arclineid": "arclineid",
    "arc_line_id": "arclineid",
    "wavesolution": "wavesol",
    "wavesol": "wavesol",
    "wave": "wavesol",
    # Unified sky
    "sky": "sky",
    "sky_sub": "sky",
    # Downstream
    "lin": "linearize",
    "linearize": "linearize",
    "stack": "stack2d",
    "stack2d": "stack2d",
    "extract": "extract1d",
    "extract1d": "extract1d",
}


class StageRegistry:
    """Lookup helpers for stage metadata."""

    def __init__(self, stages: Iterable[StageDef] = STAGES) -> None:
        self._stages = tuple(stages)
        self._by_key = {s.key: s for s in self._stages}
        self._by_id = {int(s.id): s for s in self._stages}

    def resolve_key(self, key: str) -> str:
        k = str(key or "").strip().lower()
        if not k:
            raise KeyError("stage key is empty")
        return ALIASES.get(k, k)

    def get(self, key: str) -> StageDef:
        k = self.resolve_key(key)
        if k not in self._by_key:
            raise KeyError(f"Unknown stage key: {key!r} (resolved={k!r})")
        return self._by_key[k]

    def by_id(self, id_: int) -> StageDef:
        i = int(id_)
        if i not in self._by_id:
            raise KeyError(f"Unknown stage id: {id_!r}")
        return self._by_id[i]

    def all(self, *, include_ui_only: bool = True) -> tuple[StageDef, ...]:
        if include_ui_only:
            return self._stages
        return tuple(s for s in self._stages if not s.ui_only)

    def iter(self, *, include_ui_only: bool = True) -> Iterator[StageDef]:
        yield from self.all(include_ui_only=include_ui_only)

    def dir_name(self, key: str) -> str:
        return self.get(key).dir_name

    def title(self, key: str) -> str:
        return self.get(key).title

    def label(self, key: str) -> str:
        return self.get(key).label

    def stage_id(self, key: str) -> int:
        return int(self.get(key).id)


REGISTRY = StageRegistry(STAGES)


def get_stage(key: str) -> StageDef:
    return REGISTRY.get(key)


def iter_stages(*, include_ui_only: bool = True) -> Iterator[StageDef]:
    yield from REGISTRY.iter(include_ui_only=include_ui_only)


def iter_pipeline_stages() -> Iterator[StageDef]:
    """Iterate stages that are executed by the pipeline."""

    yield from iter_stages(include_ui_only=False)


def iter_gui_stages() -> Iterator[StageDef]:
    """Iterate stages shown in the GUI list (incl. UI-only placeholders)."""

    yield from iter_stages(include_ui_only=True)
