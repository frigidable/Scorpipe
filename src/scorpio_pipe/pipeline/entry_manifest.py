"""Stage entrypoint: manifest.

P0-ARCH-02.2: used as a lazy entrypoint (string-resolved by the engine).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scorpio_pipe.pipeline.context import StageContext, StageResult


def run(
    ctx: StageContext,
    *,
    config_path: Path | None = None,
    preflight_override: bool = False,
    preflight_override_reason: str | None = None,
    preflight_override_source: str | None = None,
    **_: Any,
) -> StageResult:
    from scorpio_pipe.work_layout import ensure_work_layout
    from scorpio_pipe.manifest import write_manifest

    out_dir = Path(ctx.run_root)
    _ = ensure_work_layout(out_dir)

    # Preflight is an engine gate, but keep a defensive local check for
    # direct entrypoint use and clearer errors.
    from scorpio_pipe.preflight import ensure_preflight

    pf = ensure_preflight(dict(ctx.cfg), out_dir)
    if str((pf or {}).get("status") or "").upper() == "ERROR":
        if bool(preflight_override):
            try:
                from scorpio_pipe.preflight import write_preflight_override

                write_preflight_override(
                    out_dir,
                    status="ERROR",
                    reason=preflight_override_reason,
                    source=preflight_override_source,
                )
            except Exception:
                pass
        else:
            raise RuntimeError(
                "Preflight gate blocked run (status=ERROR). "
                f"See: {Path(out_dir) / 'manifest' / 'preflight_report.json'}"
            )

    p = write_manifest(
        out_path=Path(out_dir) / "manifest" / "manifest.json",
        cfg=dict(ctx.cfg),
        cfg_path=config_path,
    )

    # Legacy mirrors: write only if legacy roots already exist (do not create them).
    try:
        qc_legacy = Path(out_dir) / "qc"
        if qc_legacy.exists():
            write_manifest(
                out_path=qc_legacy / "manifest.json",
                cfg=dict(ctx.cfg),
                cfg_path=config_path,
            )
    except Exception:
        pass

    try:
        rep_legacy = Path(out_dir) / "report"
        if rep_legacy.exists():
            write_manifest(
                out_path=rep_legacy / "manifest.json",
                cfg=dict(ctx.cfg),
                cfg_path=config_path,
            )
    except Exception:
        pass

    manifest_dir = Path(p).parent

    return StageResult.ok(
        products={
            "manifest": str(Path(p)),
            "dependency_lock": str(manifest_dir / "requirements.lock"),
            "deps_snapshot": str(manifest_dir / "deps_snapshot.json"),
        },
        artifacts=[str(Path(p)), str(manifest_dir / "requirements.lock"), str(manifest_dir / "deps_snapshot.json")],
    )
