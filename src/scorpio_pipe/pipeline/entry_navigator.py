"""Stage entrypoint: navigator (static HTML UI).

P0-ARCH-02.2: lazy entrypoint (string-resolved by the engine).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scorpio_pipe.pipeline.context import StageContext, StageResult


def run(ctx: StageContext, **_: Any) -> StageResult:
    from scorpio_pipe.work_layout import ensure_work_layout
    from scorpio_pipe.navigator import build_navigator

    out_dir = Path(ctx.run_root)
    _ = ensure_work_layout(out_dir)

    p = build_navigator(out_dir)
    return StageResult.ok(artifacts=[str(Path(p))])
