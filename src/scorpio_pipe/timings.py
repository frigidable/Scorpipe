from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def timings_file(*, work_dir: str | Path) -> Path:
    wd = Path(work_dir).expanduser().resolve()
    return wd / "qc" / "timings.json"


def _read_list(p: Path) -> list[dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def append_timing(
    *,
    work_dir: str | Path,
    stage: str,
    seconds: float,
    ok: bool = True,
    extra: dict[str, Any] | None = None,
) -> Path:
    p = timings_file(work_dir=work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_list(p)
    rows.append(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "stage": str(stage),
            "seconds": float(seconds),
            "ok": bool(ok),
            "extra": extra or {},
        }
    )
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    # legacy mirror
    try:
        legacy = Path(work_dir).expanduser().resolve() / "report"
        legacy.mkdir(parents=True, exist_ok=True)
        (legacy / "timings.json").write_text(
            p.read_text(encoding="utf-8"), encoding="utf-8"
        )
    except Exception:
        pass
    return p


@contextmanager
def timed_stage(
    *,
    work_dir: str | Path,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Context manager that records timings to qc/timings.json (legacy mirror: report/timings.json).

    It never raises on write failures: timing must not break the pipeline.
    """

    t0 = time.perf_counter()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        dt = time.perf_counter() - t0
        try:
            append_timing(
                work_dir=work_dir, stage=stage, seconds=dt, ok=ok, extra=extra
            )
        except Exception:
            pass
