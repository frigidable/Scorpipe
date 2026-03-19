"""Pipeline execution context objects.

P0-ARCH-02.1
------------
The engine owns I/O boundaries. A stage receives :class:`StageContext` and may
write only under ``ctx.stage_dir`` (except a small set of explicitly permitted
global canonical folders such as ``manifest/`` handled by the engine).

Stages return :class:`StageResult` so the engine can record status, products
and QC artifacts in a uniform way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Literal, Mapping

from scorpio_pipe.pipeline.resources import StageResources


StageStatus = Literal["ok", "warn", "fail", "blocked", "skipped"]


class CancelToken:
    """Cooperative cancellation token."""

    def __init__(self) -> None:
        self._ev = Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()


@dataclass
class StageContext:
    """Execution context passed into a stage entrypoint."""

    cfg: Mapping[str, Any]
    run_root: Path
    stage_key: str
    stage_dir: Path

    # Ports for future refactors (kept generic for now)
    workspace: Any = None
    resources: StageResources | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    cancel_token: CancelToken | None = None

    def ensure_stage_dir(self) -> Path:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        return self.stage_dir

    def path(self, *parts: str) -> Path:
        """Convenience: build a path under the stage directory."""

        return (self.stage_dir / Path(*parts)).resolve()


@dataclass
class StageResult:
    """Uniform stage result object returned to the engine."""

    status: StageStatus = "ok"
    products: dict[str, str] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "products": dict(self.products),
            "qc": dict(self.qc),
            "artifacts": list(self.artifacts),
            "messages": list(self.messages),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    @classmethod
    def ok(cls, **kwargs: Any) -> "StageResult":
        return cls(status="ok", **kwargs)

    @classmethod
    def warn(cls, **kwargs: Any) -> "StageResult":
        return cls(status="warn", **kwargs)

    @classmethod
    def skipped(cls, **kwargs: Any) -> "StageResult":
        return cls(status="skipped", **kwargs)

    @classmethod
    def blocked(cls, **kwargs: Any) -> "StageResult":
        return cls(status="blocked", **kwargs)

    @classmethod
    def fail(
        cls,
        *,
        code: str | None = None,
        message: str | None = None,
        **kwargs: Any,
    ) -> "StageResult":
        return cls(status="fail", error_code=code, error_message=message, **kwargs)



@dataclass(frozen=True)
class PlanItem:
    """Planning output item for :func:`plan_sequence`.

    The engine returns an ordered list of actions (run/skip) with reasons.
    This is a UI/CLI-friendly object and is part of the frozen public API.
    """

    task: str
    action: str  # 'run'|'skip'
    reason: str


@dataclass(frozen=True)
class StageSpec:
    """Lightweight stage/task specification.

    This is intentionally non-invasive: stage implementations do not depend on it.
    The engine may use it for deterministic validation/provenance and UI/help text.

    ``validate_globs`` serves two roles:

    1) Execution contract: declares which on-disk artifacts count as this task's outputs.
    2) Boundary validation: for strict product kinds (MEF/spec1d/lambda_map) the engine
       validates every matched file after a successful task.
    """

    key: str
    title: str
    contract_kind: str = 'none'  # 'mef'|'spec1d'|'lambda_map'|'calib_bias_triplet'|'calib_flat'|'calib_neon'|'none'
    validate_globs: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()
    hash_inputs_policy: str = 'default'
