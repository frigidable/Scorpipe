from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json

import yaml

from scorpio_pipe.refs.store import resolve_reference
from scorpio_pipe.wavesol_paths import get_selected_disperser, slugify_disperser, wavesol_dir


@dataclass(frozen=True)
class CuratedLinePreset:
    disperser: str
    slug: str
    waveref: str
    unit: str
    lines: tuple[dict[str, Any], ...]
    source: str
    path: Path

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def n_safe_auto(self) -> int:
        return sum(1 for row in self.lines if _as_bool(row.get("safe_for_auto", False)))


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    return s in {"1", "y", "yes", "true", "t"}


def _resource_rel_for_disperser(disperser: str | None) -> str | None:
    slug = slugify_disperser(disperser or "")
    if not slug or slug == "default":
        return None
    return f"lines/{slug}.yaml"


def get_disperser_slug(cfg: dict[str, Any]) -> str:
    return slugify_disperser(get_selected_disperser(cfg))


def load_curated_line_preset(
    cfg: dict[str, Any],
    *,
    disperser: str | None = None,
    work_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> CuratedLinePreset | None:
    disp = str(disperser or get_selected_disperser(cfg) or "").strip()
    rel = _resource_rel_for_disperser(disp)
    if not rel:
        return None
    try:
        rr = resolve_reference(
            rel,
            resources_dir=cfg.get("resources_dir"),
            work_dir=work_dir,
            config_dir=config_dir or cfg.get("config_dir"),
            project_root=project_root or cfg.get("project_root") or cfg.get("data_dir"),
            allow_package=True,
        )
    except Exception:
        return None
    try:
        raw = yaml.safe_load(Path(rr.resolved_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    lines = raw.get("lines") if isinstance(raw, dict) else None
    if not isinstance(lines, list) or not lines:
        return None

    norm: list[dict[str, Any]] = []
    for row in lines:
        if not isinstance(row, dict):
            continue
        try:
            wav = float(row.get("wavelength"))
        except Exception:
            continue
        out = dict(row)
        out["wavelength"] = float(wav)
        out["priority"] = int(row.get("priority", 0) or 0)
        try:
            out["intensity"] = float(row.get("intensity", row.get("strength_tier", 1.0)) or 1.0)
        except Exception:
            out["intensity"] = 1.0
        out["safe_for_auto"] = _as_bool(row.get("safe_for_auto", False))
        out["blend_class"] = str(row.get("blend_class", "unknown") or "unknown")
        out["tag"] = str(row.get("tag", "") or "")
        if row.get("exclusion_reason") is not None:
            out["exclusion_reason"] = str(row.get("exclusion_reason") or "")
        norm.append(out)
    if not norm:
        return None

    return CuratedLinePreset(
        disperser=str(raw.get("disperser") or disp or ""),
        slug=slugify_disperser(str(raw.get("disperser") or disp or "")),
        waveref=str(raw.get("waveref") or "air").strip().lower() or "air",
        unit=str(raw.get("unit") or "angstrom").strip().lower() or "angstrom",
        lines=tuple(sorted(norm, key=lambda r: float(r.get("wavelength", 0.0)))),
        source=str(raw.get("source") or "curated preset"),
        path=Path(rr.resolved_path),
    )


def materialize_curated_linelist_csv(
    cfg: dict[str, Any],
    *,
    disperser: str | None = None,
    run_root: str | Path | None = None,
    outdir: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    preset = load_curated_line_preset(
        cfg,
        disperser=disperser,
        work_dir=run_root,
        config_dir=cfg.get("config_dir"),
        project_root=cfg.get("project_root") or cfg.get("data_dir"),
    )
    if preset is None:
        return None, {}

    dest_dir = Path(outdir).expanduser().resolve() if outdir is not None else wavesol_dir(cfg, run_root=run_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"curated_linelist_{preset.slug}.csv"
    with dest.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# waveref: {preset.waveref}\n")
        f.write(f"# unit: {preset.unit}\n")
        f.write(f"# source: {preset.source}\n")
        f.write(f"# disperser: {preset.disperser}\n")
        w = csv.writer(f)
        w.writerow(["wavelength", "intensity", "priority", "safe_for_auto", "blend_class", "tag", "exclusion_reason"])
        for row in preset.lines:
            w.writerow([
                f"{float(row['wavelength']):.6f}",
                f"{float(row.get('intensity', 1.0)):.6f}",
                int(row.get("priority", 0) or 0),
                1 if _as_bool(row.get("safe_for_auto", False)) else 0,
                str(row.get("blend_class", "unknown") or "unknown"),
                str(row.get("tag", "") or ""),
                str(row.get("exclusion_reason", "") or ""),
            ])

    meta = {
        "disperser": preset.disperser,
        "slug": preset.slug,
        "waveref": preset.waveref,
        "unit": preset.unit,
        "source": preset.source,
        "preset_path": str(preset.path),
        "csv_path": str(dest),
        "n_lines": int(preset.n_lines),
        "n_safe_auto": int(preset.n_safe_auto),
    }
    try:
        (dest_dir / "line_priors_used.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return dest, meta
