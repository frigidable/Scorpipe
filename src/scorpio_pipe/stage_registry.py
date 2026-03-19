"""Stage registry (deprecated shim).

Single source of truth moved to :mod:`scorpio_pipe.catalog.stages`.

This module is kept for backward compatibility: it re-exports the same public
API (StageDef/STAGES/ALIASES/StageRegistry/REGISTRY) but reads data from the
catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from scorpio_pipe.catalog.stages import CATALOG


@dataclass(frozen=True)
class StageDef:
    """Single GUI stage definition."""

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
        return f"{int(self.id):02d} {self.label}"


# Canonical stage table (derived from StageCatalog)
STAGES: tuple[StageDef, ...] = tuple(
    StageDef(s.id, s.key, s.slug, s.label, ui_only=bool(s.ui_only))
    for s in CATALOG.gui_stages()
)


# Compatibility aliases (task names / shorthands -> canonical GUI stage keys).
ALIASES: dict[str, str] = CATALOG.stage_key_aliases()


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
