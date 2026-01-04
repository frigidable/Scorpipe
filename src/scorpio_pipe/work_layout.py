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
    - Always create: ``manifest/``, ``qc/``, ``ui/navigator/``, ``ui/history/`` and ``config.yaml``.
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
    # Lightweight UI/QC roots are part of the run layout contract.
    qc.mkdir(parents=True, exist_ok=True)
    (wd / "ui" / "navigator").mkdir(parents=True, exist_ok=True)
    (wd / "ui" / "history").mkdir(parents=True, exist_ok=True)
    _write_default_config(wd / "config.yaml")

    # Project-level role manifest (P1): explicit source of truth for frame roles.
    # We only create a minimal template if the file does not exist.
    try:
        from scorpio_pipe.project_manifest import write_default_project_manifest

        write_default_project_manifest(wd / "project_manifest.yaml")
    except Exception:
        pass

    # Ensure the run passport exists (best-effort; it may be created later
    # when a representative header is available).
    try:
        from scorpio_pipe.run_passport import ensure_run_passport

        ensure_run_passport(wd)
    except Exception:
        pass

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
