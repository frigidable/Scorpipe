#!/usr/bin/env python3
"""Read-only evidence inventory for an existing SCORPIO-1 Scorpipe run.

Reports stage markers and existing FITS headers. The report is written only to
the explicit output path. It does not rerun reductions or change source files.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HEADER_KEYS = (
    "INSTRUME", "OBJECT", "NODE", "RATE", "GAIN", "BINNING", "BIASSUB",
    "BVARINC", "SB_GID", "DEGRAD", "DEGRRSN", "FSIGRDO", "FSIGBIN",
    "NBIAS", "NFLAT", "NCOMBINE", "MF_AXIS", "MF_SMW", "SF_METH",
    "FLATCOR", "SB_PATH", "WAVEREF", "BUNIT", "BIASSEC", "TRIMSEC",
)


def inspect(config: Path, max_fits_per_stage: int = 300) -> dict:
    import yaml

    cfg = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError("Expected config mapping")
    wd = Path(str(cfg.get("work_dir") or config.parent)).expanduser()
    if not wd.is_absolute():
        wd = (config.parent / wd).resolve()
    if not wd.is_dir():
        raise FileNotFoundError(f"Configured work_dir unavailable: {wd}")

    from astropy.io import fits

    stages = []
    for folder in sorted(wd.iterdir()):
        if not folder.is_dir() or not re.match(r"^\d+_[\w-]+$", folder.name):
            continue
        markers = []
        for p in sorted(set(folder.glob("done.json")) | set(folder.glob("*_done.json"))):
            try:
                d = json.loads(p.read_text(encoding="utf-8-sig"))
                markers.append({
                    "file": str(p.relative_to(wd)), "status": d.get("status"),
                    "enabled": d.get("enabled"), "n_outputs": d.get("n_outputs"),
                    "flags": [str(x.get("code")) for x in (d.get("flags") or [])
                              if isinstance(x, dict)],
                })
            except Exception as exc:
                markers.append({"file": str(p.relative_to(wd)),
                                "read_error": str(exc)})
        file_paths = sorted(p for p in folder.rglob("*") if p.is_file()
                            and p.suffix.lower() in {".fits", ".fts", ".fit"})
        fits_rows = []
        for p in file_paths[:max_fits_per_stage]:
            entry = {"file": str(p.relative_to(wd)),
                     "size_bytes": p.stat().st_size,
                     "modified_utc": datetime.fromtimestamp(
                         p.stat().st_mtime, tz=timezone.utc
                     ).isoformat()}
            try:
                with fits.open(p, memmap=False, do_not_scale_image_data=True,
                               ignore_missing_end=True) as hdul:
                    h = hdul[0].header
                    entry["shape"] = list(reversed(
                        [int(h.get(f"NAXIS{i}", 0))
                         for i in range(1, int(h.get("NAXIS", 0))+1)]
                    ))
                    entry["header"] = {k: str(h[k]) for k in HEADER_KEYS if k in h}
                    entry["extensions"] = [str(x.header.get("EXTNAME", "PRIMARY"))
                                           for x in hdul]
            except Exception as exc:
                entry["read_error"] = str(exc)
            fits_rows.append(entry)
        stages.append({
            "stage": folder.name, "markers": markers,
            "n_fits": len(file_paths), "truncated": len(file_paths)>max_fits_per_stage,
            "fits_products": fits_rows,
        })
    return {
        "status": "READ_ONLY_INVENTORY_NOT_SCIENCE_VALIDATION",
        "config_file": str(config), "work_dir": str(wd),
        "flatfield_enabled": bool((cfg.get("flatfield") or {}).get("enabled", False)),
        "bias_policy": str((cfg.get("calib") or {}).get("bias_policy", "degraded")),
        "superneon_bias_sub": bool((cfg.get("superneon") or {}).get("bias_sub", True)),
        "stage_count": len(stages), "stages": stages,
        "limitations": [
            "A done marker does not prove scientific validity.",
            "Existing files can be stale relative to current config.",
            "No raw FITS, config, or pipeline code are modified.",
            "Only stage folders and existing FITS headers are scanned.",
        ],
    }


def selftest():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as t:
        root = Path(t)
        conf = root / "config.yaml"
        conf.write_text("work_dir: .\nflatfield:\n  enabled: false\n",
                        encoding="utf-8")
        stage = root / "04_flatfield"
        stage.mkdir()
        (stage / "flatfield_done.json").write_text(
            '{"enabled":false,"status":"skipped"}', encoding="utf-8")
        result = inspect(conf)
        assert result["stage_count"] == 1
        assert result["stages"][0]["markers"][0]["status"] == "skipped"
        assert not result["flatfield_enabled"]
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 3:
        config = Path(sys.argv[1]).expanduser().resolve()
        out = Path(sys.argv[2]).expanduser().resolve()
        result = inspect(config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print("WORK_ROOT:", result["work_dir"])
        print("STAGES:", result["stage_count"])
        for stage in result["stages"]:
            markers = ", ".join(
                f"{Path(m['file']).name}={m.get('status','?')}"
                for m in stage["markers"]
            ) or "none"
            print(f"  {stage['stage']}: {stage['n_fits']} FITS; markers {markers}")
        print("OUTPUT:", out)
        print("READ-ONLY RUNTIME INVENTORY PASSED")
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml output.json")
