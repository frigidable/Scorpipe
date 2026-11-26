"""Batch read-only numerical audit for actual captured Kelson SkySub nights.

Usage:
    python -m scorpio_pipe.calib.sky_fit_batch /path/to/09_sky

Always prints deterministic JSON. Exit 0 means every *Kelson* sky product
found has a complete matching captured fit and passes numerical fidelity.
Exit 1 means incomplete/failed or no Kelson products (fail closed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from astropy.io import fits

from scorpio_pipe.calib.sky_fit_audit import audit_frozen_kelson_products


def audit_sky_directory(sky_directory: str | Path) -> dict[str, Any]:
    root = Path(sky_directory)
    if not root.is_dir():
        raise ValueError(f"Sky stage directory not found: {root}")

    matches = sorted(root.glob("*_skysub_raw.fits"))
    if not matches:
        raise ValueError(f"No raw SkySub FITS science products found: {root}")
    items: list[dict[str, Any]] = []
    seen = set()
    for sci_path in matches:
        stem = sci_path.name.removesuffix("_skysub_raw.fits")
        if stem in seen:
            raise ValueError(f"Duplicate SkySub stem: {stem}")
        seen.add(stem)
        model_path = root / f"{stem}_skymodel_raw.fits"
        item: dict[str, Any] = {
            "stem": stem,
            "status": "FAIL",
            "science_fits": sci_path.name,
            "model_fits": model_path.name,
        }
        try:
            with fits.open(sci_path, memmap=False) as hdul:
                hdr = hdul[0].header
                method = str(hdr.get("SKYMD") or "").strip().lower()
                pointer = str(hdr.get("SKYFRZP") or "").strip()
                version = int(hdr.get("SKYFRZV") or 0)
            item["method"] = method
            if method in {"sky_scale_raw", "passthrough"}:
                item["status"] = "SKIPPED_OTHER_METHOD"
            elif method != "kelson_raw":
                raise ValueError("Unknown or missing SkySub science method")
            else:
                # FITS header values are untrusted path strings. Only one
                # sibling archive file, never an absolute/parent path, is
                # legal for science provenance.
                if (not pointer or Path(pointer).name != pointer
                        or "/" in pointer or "\\" in pointer
                        or pointer in (".", "..")
                        or not pointer.endswith("_kelson_frozen_fit_v1.npz")
                        or version != 1):
                    raise ValueError("No valid current Kelson capture linkage")
                if not model_path.is_file():
                    raise ValueError("Corresponding sky model FITS missing")
                archive_path = root / pointer
                if not archive_path.is_file():
                    raise ValueError("Current frozen Kelson fit NPZ missing")
                item["archive"] = pointer
                item["audit"] = audit_frozen_kelson_products(
                    archive_path, model_path, sci_path,
                )
                item["status"] = "PASS"
        except Exception as exc:
            item["status"] = "FAIL"
            item["error"] = f"{type(exc).__name__}: {exc}"
        items.append(item)

    fitted = [i for i in items if i.get("method") == "kelson_raw"]
    passed = sum(i["status"] == "PASS" for i in fitted)
    failed = sum(i["status"] == "FAIL" for i in items)
    return {
        "schema": "KELSON_FROZEN_SKY_FIDELITY_BATCH_V1",
        "status": "PASS" if fitted and failed == 0 and passed == len(fitted) else "FAIL",
        "sky_directory": str(root),
        "n_sky_products": len(items),
        "n_kelson": len(fitted),
        "n_passed": passed,
        "n_failed": failed,
        "n_other_methods": len(items) - len(fitted),
        "science_uncertainty_qualified": False,
        "per_exposure": items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only archive-to-FITS fidelity audit of a real Kelson sky stage."
    )
    parser.add_argument("sky_directory", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_sky_directory(args.sky_directory)
    except Exception as exc:
        report = {
            "schema": "KELSON_FROZEN_SKY_FIDELITY_BATCH_V1",
            "status": "FAIL",
            "sky_directory": str(args.sky_directory),
            "error": f"{type(exc).__name__}: {exc}",
            "science_uncertainty_qualified": False,
        }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
