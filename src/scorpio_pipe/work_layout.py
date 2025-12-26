from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkLayout:
    work_dir: Path
    raw: Path
    calibs: Path
    science: Path
    products: Path
    qc: Path

    # legacy compatibility
    calib_legacy: Path
    report_legacy: Path


def ensure_work_layout(work_dir: str | Path) -> WorkLayout:
    """Create (if needed) standard subdirectories inside work_dir.

    Standard (v5.17+):
      raw/ calibs/ science/ products/ qc/

    Legacy:
      calib/ report/ are still created to keep older code/UI stable.

    Returns
    -------
    WorkLayout
        Resolved absolute paths.
    """
    wd = Path(work_dir).expanduser().resolve()
    raw = wd / "raw"
    calibs = wd / "calibs"
    science = wd / "science"
    products = wd / "products"
    qc = wd / "qc"

    # v5.36+: canonical "products/qc" mirror for QC artifacts.
    products_qc = products / "qc"

    calib_legacy = wd / "calib"
    report_legacy = wd / "report"

    for p in [raw, calibs, science, products, qc, products_qc, calib_legacy, report_legacy]:
        p.mkdir(parents=True, exist_ok=True)

    return WorkLayout(
        work_dir=wd,
        raw=raw,
        calibs=calibs,
        science=science,
        products=products,
        qc=qc,
        calib_legacy=calib_legacy,
        report_legacy=report_legacy,
    )
