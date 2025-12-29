from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkLayout:
    work_dir: Path

    # Paths kept for backward compatibility with internal tests/callers.
    # v5.38.4 no longer creates these directories automatically.
    raw: Path

    # Machine-readable bookkeeping.
    manifest: Path

    # Legacy/compatibility roots (not created by default).
    calibs: Path
    science: Path
    qc: Path
    calib_legacy: Path
    report_legacy: Path


def _write_default_config(path: Path) -> None:
    if path.exists():
        return
    # Minimal, human-editable template.
    path.write_text(
        """# Scorpio Pipe workspace configuration\n\n# This file is intentionally minimal at workspace creation time.\n# The GUI will populate it when you create a new config.\n\nwork_dir: .\nframes: {}\n""",
        encoding="utf-8",
    )


def ensure_work_layout(work_dir: str | Path) -> WorkLayout:
    """Create (if needed) a minimal workspace root.

    New layout (v5.38.4)
    --------------------
    - Always create: ``manifest/`` and ``config.yaml``.
    - Do NOT create: ``raw/`` and ``products/``.
    - Stage output directories live directly under the workspace root as
      ``NN_slug/`` and are created **lazily** by each stage when it runs.

    Returns
    -------
    WorkLayout
        Resolved absolute paths.
    """

    wd = Path(work_dir).expanduser().resolve()

    raw = wd / "raw"  # not created automatically
    manifest = wd / "manifest"

    # Legacy/compatibility paths (not created here)
    calibs = wd / "calibs"
    science = wd / "science"
    qc = wd / "qc"
    calib_legacy = wd / "calib"
    report_legacy = wd / "report"

    manifest.mkdir(parents=True, exist_ok=True)
    _write_default_config(wd / "config.yaml")

    return WorkLayout(
        work_dir=wd,
        raw=raw,
        manifest=manifest,
        calibs=calibs,
        science=science,
        qc=qc,
        calib_legacy=calib_legacy,
        report_legacy=report_legacy,
    )
