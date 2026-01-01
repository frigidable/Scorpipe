
from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from scorpio_pipe.version import __version__
from scorpio_pipe.work_layout import stage_dir


def _iter_files(base: Path) -> Iterable[Path]:
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    out: list[Path] = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            out.append(p)
    return out


def export_science_package(
    work_dir: Path,
    out_zip: Path | None = None,
    *,
    include_html: bool = True,
    include_png: bool = True,
    overwrite: bool = True,
) -> Path:
    """Export a compact, shareable result ZIP.

    Contents (when available):
      - spec1d.fits (and auxiliary 1D outputs)
      - qc/qc_report.json
      - ui/navigator/index.html + data.json (+ assets in the navigator dir)
      - key PNG quicklooks (coverage/stack/spec/etc.)

    The function is intentionally conservative: missing optional files are skipped,
    but if spec1d is absent we still create the zip (QC-only package can be useful).
    """
    work_dir = Path(work_dir).expanduser().resolve()
    if out_zip is None:
        out_zip = work_dir / "science_package.zip"
    out_zip = Path(out_zip).expanduser().resolve()

    if out_zip.exists() and not overwrite:
        raise FileExistsError(f"{out_zip} exists (use overwrite=True)")

    # Collect core deliverables.
    files: list[tuple[Path, str]] = []

    # spec1d
    try:
        ext_dir = stage_dir(work_dir, "extract")
        for p in sorted(ext_dir.glob("spec1d*.fits")):
            files.append((p, str(p.relative_to(work_dir))))
    except Exception:
        # Fallback: search anywhere under work_dir
        for p in sorted(work_dir.rglob("spec1d*.fits")):
            files.append((p, str(p.relative_to(work_dir))))

    # QC report
    qc_json = work_dir / "qc" / "qc_report.json"
    if qc_json.exists():
        files.append((qc_json, str(qc_json.relative_to(work_dir))))

    # Navigator HTML bundle
    if include_html:
        nav_dir = work_dir / "ui" / "navigator"
        for p in _iter_files(nav_dir):
            files.append((p, str(p.relative_to(work_dir))))

    # Key PNG quicklooks
    if include_png:
        png_names = {
            "coverage.png",
            "stack2d.png",
            "eta_lambda.png",
            "spec1d_trace.png",
            "spec1d_fixed.png",
            "qc_summary.png",
            "cosmics_coverage.png",
        }
        for p in sorted(work_dir.rglob("*.png")):
            if p.name in png_names:
                files.append((p, str(p.relative_to(work_dir))))

        # Also include a small curated set from common stage dirs (first 2 PNG each).
        stage_png_dirs = [
            work_dir / "plots",
            work_dir / "qc",
            stage_dir(work_dir, "stack") / "plots",
            stage_dir(work_dir, "extract") / "plots",
        ]
        seen = {a for _, a in files}
        for d in stage_png_dirs:
            if not d.exists():
                continue
            cnt = 0
            for p in sorted(d.glob("*.png")):
                arc = str(p.relative_to(work_dir))
                if arc in seen:
                    continue
                files.append((p, arc))
                seen.add(arc)
                cnt += 1
                if cnt >= 2:
                    break

    # Remove duplicates, keep first occurrence.
    uniq: dict[str, Path] = {}
    for p, arc in files:
        if arc not in uniq and p.exists():
            uniq[arc] = p

    manifest = {
        "pipeline_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work_dir),
        "files": sorted(list(uniq.keys())),
    }

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for arc, p in uniq.items():
            zf.write(p, arcname=arc)

    return out_zip

