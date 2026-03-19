"""Stage entrypoint: lineid_prepare.

Artifacts live in the disperser-aware wavesol directory.

P0-ARCH-02.2: lazy entrypoint (string-resolved by the engine).
"""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.lineid_auto_backup import prepare_lineid

    # Prefer the engine-controlled run_root when supported.
    try:
        out = prepare_lineid(dict(ctx.cfg), run_root=Path(ctx.run_root))
    except TypeError:
        out = prepare_lineid(dict(ctx.cfg))

    prods: dict[str, str] = {}
    # Best-effort: expose the files this stage creates.
    try:
        if isinstance(out, dict):
            for k, v in out.items():
                if v is not None:
                    p = Path(str(v))
                    if p.exists():
                        prods[f"lineid_{k}"] = str(p)
    except Exception:
        pass

    artifacts = [str(Path(v)) for v in out.values()] if isinstance(out, dict) else []
    return StageResult.ok(products=prods, artifacts=artifacts)
