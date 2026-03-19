"""Stage contract registry (deprecated shim).

Single source of truth moved to :mod:`scorpio_pipe.catalog.stages`.

This module is kept for backward compatibility: it re-exports the same public
API (StageContract/CONTRACTS/list_stage_contracts/validate_contracts) but reads
contracts from the catalog.

P0-ARCH-01.1: contracts are derived *entirely* from StageCatalog (including
legacy task aliases), eliminating a third "table of truth".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from scorpio_pipe.catalog.stages import CATALOG
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG


@dataclass(frozen=True)
class StageContract:
    stage: str
    title: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]  # product keys
    metrics: tuple[str, ...] = ()
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "title": self.title,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "metrics": list(self.metrics),
            "notes": self.notes,
        }


def _mk(stage_key: str) -> StageContract:
    # get_task() accepts aliases; it returns the canonical task info.
    t = CATALOG.get_task(stage_key)
    return StageContract(
        stage=stage_key,
        title=str(t.label),
        inputs=tuple(t.inputs or ()),
        outputs=tuple(t.product_keys or ()),
        metrics=tuple(t.metrics_keys or ()),
        notes=t.notes,
    )


def _build_contracts() -> dict[str, StageContract]:
    out: dict[str, StageContract] = {}

    # Include canonical tasks + all catalog-declared aliases.
    # This keeps backwards compatibility (e.g. bias_combine/masterbias/wavesol).
    for t in CATALOG.task_infos():
        keys = [t.key, *list(t.aliases or ())]
        for k in keys:
            if k in out:
                # Should not happen (StageCatalog enforces uniqueness), but keep
                # this ultra-defensive to avoid import-time explosions.
                continue
            out[k] = _mk(k)

    return out


CONTRACTS: dict[str, StageContract] = _build_contracts()


def list_stage_contracts() -> dict[str, StageContract]:
    return dict(CONTRACTS)


def validate_contracts(cfg: Mapping[str, Any]) -> None:
    """Validate contracts against the current product registry."""

    product_keys = set(PRODUCT_CATALOG.keys())
    for st, c in CONTRACTS.items():
        if st != c.stage:
            raise ValueError(f"Contract key '{st}' != contract.stage '{c.stage}'")
        missing = [k for k in c.outputs if k not in product_keys]
        if missing:
            raise ValueError(
                f"Contract '{st}' refers to unknown product keys: {missing}"
            )
