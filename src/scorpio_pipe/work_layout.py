from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkLayout:
    work_dir: Path
    raw: Path
    products: Path
    manifest: Path

    # Legacy/compatibility roots. These are *not* created by default.
    calibs: Path
    science: Path
    qc: Path

    # legacy compatibility
    calib_legacy: Path
    report_legacy: Path


def ensure_work_layout(work_dir: str | Path) -> WorkLayout:
    """Create (if needed) the *minimal* workspace root.

    Canonical (v5.38+):
      raw/ products/ manifest/

    We intentionally do *not* create legacy directories (calib/, report/) nor
    older roots (calibs/, science/, qc/) here. Stages may create additional
    directories when they actually need to write outputs.

    Returns
    -------
    WorkLayout
        Resolved absolute paths.
    """
    wd = Path(work_dir).expanduser().resolve()
    raw = wd / "raw"
    products = wd / "products"
    manifest = wd / "manifest"

    # Legacy/compatibility paths (not created here)
    calibs = wd / "calibs"
    science = wd / "science"
    qc = wd / "qc"

    calib_legacy = wd / "calib"
    report_legacy = wd / "report"

    for p in [raw, products, manifest]:
        p.mkdir(parents=True, exist_ok=True)

    return WorkLayout(
        work_dir=wd,
        raw=raw,
        products=products,
        manifest=manifest,
        calibs=calibs,
        science=science,
        qc=qc,
        calib_legacy=calib_legacy,
        report_legacy=report_legacy,
    )
