from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


def safe_slug(s: str, *, max_len: int = 80) -> str:
    """Convert a label to a filesystem-friendly slug.

    Keeps letters/digits, replaces everything else with underscores,
    collapses repeats and trims.
    """
    s = (s or "").strip()
    if not s:
        return ""

    # Replace separators/whitespace
    s = s.replace("/", " ").replace("\\", " ")
    s = re.sub(r"\s+", " ", s)

    # Keep unicode letters and digits. Everything else -> underscore.
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    slug = re.sub(r"_+", "_", slug).strip("_")

    if not slug:
        slug = "item"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


@dataclass(frozen=True)
class RunSignature:
    object_name: str
    disperser: str = ""
    slit: str = ""
    binning: str = ""

    def key(self) -> str:
        parts = [
            safe_slug(self.object_name) or "obj",
            safe_slug(self.disperser) or "nodisp",
            safe_slug(self.slit) or "noslit",
            safe_slug(self.binning) or "nobin",
        ]
        return "__".join(parts)


def signature_from_yaml(cfg_path: Path) -> RunSignature | None:
    """Extract the run signature from an existing config.yaml."""
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        obj = str(cfg.get("object", "") or "").strip()
        frames = cfg.get("frames") or {}
        setup = frames.get("__setup__", {}) if isinstance(frames, dict) else {}
        if not isinstance(setup, dict):
            setup = {}
        disp = str(setup.get("disperser", "") or "").strip()
        slit = str(setup.get("slit", "") or "").strip()
        binning = str(setup.get("binning", "") or "").strip()
        if not obj:
            return None
        return RunSignature(obj, disp, slit, binning)
    except Exception:
        return None


def _is_effectively_empty(dir_path: Path) -> bool:
    if not dir_path.exists():
        return True
    ignore = {"README.md", "README.txt", ".gitkeep"}
    for p in dir_path.iterdir():
        if p.name in ignore:
            continue
        return False
    return True


def pick_smart_run_dir(
    base_dir: Path,
    sig: RunSignature,
    *,
    prefer_flat: bool = True,
) -> Path:
    """Pick a collision-free output directory inside the night folder.

    Rules:
      1) If base_dir is empty -> base_dir (single-run, no extra nesting).
      2) If base_dir has config.yaml with the same signature -> base_dir (resume).
      3) Otherwise create a subfolder under base_dir to avoid collisions:
         - if prefer_flat: <object>/<disperser>
         - else: <object>__<disperser>
      4) If that folder exists and is used by someone else -> append _2, _3, ...
    """
    base_dir = Path(base_dir).expanduser()

    if _is_effectively_empty(base_dir):
        return base_dir

    cfg0 = base_dir / "config.yaml"
    if cfg0.is_file():
        s0 = signature_from_yaml(cfg0)
        if s0 is not None and s0.key() == sig.key():
            return base_dir

    obj_slug = safe_slug(sig.object_name) or "obj"
    disp_slug = safe_slug(sig.disperser) or "nodisp"

    if prefer_flat:
        cand = base_dir / obj_slug / disp_slug
    else:
        cand = base_dir / f"{obj_slug}__{disp_slug}"

    if _is_effectively_empty(cand):
        return cand
    cfg1 = cand / "config.yaml"
    if cfg1.is_file():
        s1 = signature_from_yaml(cfg1)
        if s1 is not None and s1.key() == sig.key():
            return cand

    for i in range(2, 1000):
        if prefer_flat:
            cand_i = base_dir / obj_slug / f"{disp_slug}_{i}"
        else:
            cand_i = base_dir / f"{obj_slug}__{disp_slug}_{i}"
        if _is_effectively_empty(cand_i):
            return cand_i
        cfg_i = cand_i / "config.yaml"
        if cfg_i.is_file():
            s_i = signature_from_yaml(cfg_i)
            if s_i is not None and s_i.key() == sig.key():
                return cand_i

    return cand.with_name(cand.name + "_overflow")
