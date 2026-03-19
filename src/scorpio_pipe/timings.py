from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def timings_file(*, work_dir: str | Path) -> Path:
    """Canonical timings file.

    PERF-30.1: stage timings are stored as **append-only JSONL**.
    One record = one line, so writing timings stays O(1) even for long sessions.
    """

    wd = Path(work_dir).expanduser().resolve()
    return wd / "manifest" / "timings.jsonl"


def timings_file_legacy_json(*, work_dir: str | Path) -> Path:
    """Legacy timings list file (pre v7.31.0)."""

    wd = Path(work_dir).expanduser().resolve()
    return wd / "manifest" / "timings.json"


def _iter_timings_jsonl(p: Path) -> Iterator[dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row
    except Exception:
        return


def iter_timings(*, work_dir: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate timing records.

    Supports both the new JSONL file and the legacy JSON list file.
    """

    p = timings_file(work_dir=work_dir)
    if p.exists():
        yield from _iter_timings_jsonl(p)
        return

    # fallback: legacy list
    p0 = timings_file_legacy_json(work_dir=work_dir)
    try:
        if p0.exists():
            rows = json.loads(p0.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                for r in rows:
                    if isinstance(r, dict):
                        yield r
    except Exception:
        return


def append_timing(
    *,
    work_dir: str | Path,
    stage: str,
    seconds: float,
    ok: bool = True,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append a timing record to manifest/timings.jsonl (best-effort).

    The write is append-only and uses flush+fsync best-effort to reduce the chance
    of losing the tail on crashes.
    """

    p = timings_file(work_dir=work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "seconds": float(seconds),
        "ok": bool(ok),
        "extra": extra or {},
    }
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
        try:
            f.flush()
        except Exception:
            pass
        try:
            os.fsync(f.fileno())
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
    """Context manager that records timings to manifest/timings.jsonl.

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
