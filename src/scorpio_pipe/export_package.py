from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from scorpio_pipe.version import __version__


def export_science_package(
    work_dir: Path,
    out_zip: Path | None = None,
    *,
    include_html: bool = True,
    include_png: bool = True,
    overwrite: bool = True,
) -> Path:
    """Export a compact, shareable result ZIP.

    P0-ARCH-01.2: products are resolved via :class:`ProductCatalog`.

    Contents (when available):
      - spec1d.fits (+ trace.json)
      - stacked2d.fits / stack2d.fits (+ eta_lambda.fits)
      - lambda_map.fits
      - qc/qc_report.json + index.html
      - ui/navigator/ bundle
      - key PNG quicklooks (coverage/spec/etc.)

    Missing optional files are skipped.
    """

    work_dir = Path(work_dir).expanduser().resolve()
    if out_zip is None:
        out_zip = work_dir / "science_package.zip"
    out_zip = Path(out_zip).expanduser().resolve()

    if out_zip.exists() and not overwrite:
        raise FileExistsError(f"{out_zip} exists (use overwrite=True)")

    # Best-effort load config (needed for disperser-aware wavesol paths).
    cfg: dict = {
        "work_dir": str(work_dir),
        "work_dir_abs": str(work_dir),
        "config_dir": str(work_dir),
        "config_path": str(work_dir / "config.yaml"),
    }
    try:
        from scorpio_pipe.config import load_config

        cfg_path = work_dir / "config.yaml"
        if cfg_path.is_file():
            cfg = load_config(cfg_path)
    except Exception:
        pass

    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

    # Curated default export set.
    keys: list[str] = [
        # Science deliverables
        "spec1d_fits",
        "trace_json",
        "stacked2d_fits",
        "stack2d_fits",
        "eta_lambda_fits",
        "lambda_map",
        # QC + manifests
        "qc_json",
        "products_manifest",
        "manifest",
        "dependency_lock",
        "deps_snapshot",
    ]

    if include_html:
        keys.extend([
            "qc_html",
            "navigator_dir",
        ])

    if include_png:
        keys.extend([
            "coverage_png",
            "spec1d_png",
            "lin_preview_png",
            "superneon_png",
            "wavesolution_1d_png",
        ])

    # Collect files.
    files: list[tuple[Path, str]] = []
    for p, arc in PRODUCT_CATALOG.iter_export_files(cfg, product_keys=tuple(keys), include_dirs=True):
        files.append((p, arc))

    # Remove duplicates, keep first occurrence.
    uniq: dict[str, Path] = {}
    for p, arc in files:
        if arc not in uniq and p.exists() and p.is_file():
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
