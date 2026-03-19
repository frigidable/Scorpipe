from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scorpio_pipe.calib.common import save_json


_SETUP_NAME_RE = re.compile(r"^(VPHG|GR)\S+", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _norm_setup_name(name: str) -> str:
    s = str(name or "").strip()
    s = s.replace("VPGH", "VPHG")
    return s


def _parse_anchor_file(path: str | Path) -> list[tuple[float, float]]:
    p = Path(path)
    anchors: list[tuple[float, float]] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        nums = _NUM_RE.findall(line)
        if len(nums) < 2:
            continue
        try:
            x = float(nums[0])
            lam = float(nums[1])
        except Exception:
            continue
        # Heuristic: anchor tables are px, wavelength pairs.
        if 0 <= x <= 10000 and 1000 <= lam <= 20000:
            anchors.append((x, lam))
    anchors.sort(key=lambda t: t[0])
    return anchors


def _parse_lines_tab(path: str | Path) -> list[dict[str, float]]:
    p = Path(path)
    out: list[dict[str, float]] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cols = s.split()
        if len(cols) < 1:
            continue
        try:
            lam = float(cols[0])
        except Exception:
            continue
        quality = None
        weight = None
        if len(cols) > 1:
            try:
                quality = float(cols[1])
            except Exception:
                quality = None
        if len(cols) > 2:
            try:
                weight = float(cols[2])
            except Exception:
                weight = None
        row: dict[str, float] = {"wavelength_A": lam}
        if quality is not None:
            row["quality"] = quality
        if weight is not None:
            row["weight"] = weight
        out.append(row)
    out.sort(key=lambda r: r.get("wavelength_A", 0.0))
    return out


def _is_anchor_filename(path: Path) -> bool:
    name = path.name
    if name.lower().startswith("lines_tab_"):
        return False
    if name.startswith("."):
        return False
    stem = _norm_setup_name(path.stem)
    return bool(_SETUP_NAME_RE.match(stem))


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    if n % 2:
        return float(vals[n // 2])
    return float((vals[n // 2 - 1] + vals[n // 2]) / 2.0)


def _summarise_anchors(anchors: list[tuple[float, float]]) -> dict[str, Any]:
    if not anchors:
        return {}
    px = [a[0] for a in anchors]
    lam = [a[1] for a in anchors]
    disp: list[float] = []
    for (x0, l0), (x1, l1) in zip(anchors[:-1], anchors[1:]):
        dx = x1 - x0
        if dx == 0:
            continue
        disp.append((l1 - l0) / dx)
    return {
        "n_anchor_points": int(len(anchors)),
        "pixel_min": float(min(px)),
        "pixel_max": float(max(px)),
        "lambda_min_A": float(min(lam)),
        "lambda_max_A": float(max(lam)),
        "median_dispersion_A_per_px": _median(disp),
        "dispersion_samples_A_per_px": [round(v, 6) for v in disp[:16]],
        "anchors_preview": [[float(x), float(y)] for x, y in anchors[:8]],
    }


def _summarise_lines(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {}
    lam = [float(r["wavelength_A"]) for r in rows if "wavelength_A" in r]
    weights = [float(r["weight"]) for r in rows if "weight" in r]
    strongest = sorted(
        [r for r in rows if "weight" in r],
        key=lambda r: (float(r.get("weight") or 0.0), float(r.get("quality") or 0.0)),
        reverse=True,
    )[:8]
    return {
        "line_table_count": int(len(rows)),
        "line_lambda_min_A": float(min(lam)),
        "line_lambda_max_A": float(max(lam)),
        "median_line_weight": _median(weights) if weights else None,
        "strongest_lines_A": [float(r["wavelength_A"]) for r in strongest],
    }


def discover_long_setups(long_root: str | Path) -> dict[str, Any]:
    root = Path(long_root).expanduser().resolve()
    txt_files = sorted(p for p in root.glob("*.txt") if _is_anchor_filename(p))
    setups: list[dict[str, Any]] = []
    for txt in txt_files:
        setup_name = _norm_setup_name(txt.stem)
        anchors = _parse_anchor_file(txt)
        if len(anchors) < 2:
            continue
        line_tab = root / f"lines_tab_{setup_name}.tab"
        line_rows = _parse_lines_tab(line_tab) if line_tab.exists() else []
        row: dict[str, Any] = {
            "setup": setup_name,
            "anchor_file": str(txt),
            **_summarise_anchors(anchors),
        }
        if line_tab.exists():
            row["line_table_file"] = str(line_tab)
            row.update(_summarise_lines(line_rows))
        setups.append(row)

    setups.sort(key=lambda r: str(r.get("setup") or ""))
    return {
        "schema": "scorpio_pipe.long_setup_summary",
        "schema_version": 1,
        "long_root": str(root),
        "n_setups": int(len(setups)),
        "setups": setups,
    }


def write_long_setup_summary(long_root: str | Path, output: str | Path) -> Path:
    payload = discover_long_setups(long_root)
    return save_json(Path(output).expanduser().resolve(), payload)


def render_long_setup_summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LONG setup summary",
        "",
        f"- LONG root: `{payload.get('long_root')}`",
        f"- Setups discovered: {int(payload.get('n_setups') or 0)}",
        "",
    ]
    for row in list(payload.get("setups") or []):
        lines.append(f"## {row.get('setup')}")
        lines.append("")
        lines.append(f"- Coverage: {row.get('lambda_min_A')} .. {row.get('lambda_max_A')} Å")
        lines.append(f"- Median dispersion: {row.get('median_dispersion_A_per_px')} Å/px")
        lines.append(f"- Anchor points: {row.get('n_anchor_points')}")
        if row.get("line_table_count"):
            lines.append(f"- LONG line-table entries: {row.get('line_table_count')}")
            strong = row.get("strongest_lines_A") or []
            if strong:
                lines.append("- Strong lines: " + ", ".join(f"{float(v):.2f}" for v in strong[:8]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_long_setup_summary_markdown(long_root: str | Path, output: str | Path) -> Path:
    payload = discover_long_setups(long_root)
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_long_setup_summary_markdown(payload), encoding="utf-8")
    return out
