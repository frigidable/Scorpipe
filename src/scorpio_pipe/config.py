from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import logging

from scorpio_pipe.work_layout import ensure_work_layout


def _norm_path_str(p: str) -> str:
    """Normalize a path string for cross-platform YAML.

    We always normalize path separators to forward slashes when serializing/
    round-tripping configs. Python (and Astropy) accept forward slashes on
    Windows, while POSIX treats backslashes as literal characters.
    """
    return str(p).replace("\\", "/")


def resolve_path(p: str | Path, *, base_dir: Path) -> Path:
    pp = Path(_norm_path_str(str(p))).expanduser()
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


def _find_project_root(start: Path) -> Path | None:
    """Best-effort project root detection.

    We need a stable base directory for resolving paths like "work/run1".
    The config file itself is typically located *inside* work_dir
    (e.g. work/run1/config.yaml), so resolving work_dir relative to config_dir
    produces duplicated segments.

    Strategy: walk up and look for pyproject.toml (dev checkout).
    If not found, return None and fall back to config_dir.
    """
    start = start.resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").is_file():
            return p
    return None


def load_config(cfg_path: str | Path) -> dict[str, Any]:
    """Load YAML config + resolve relative paths.

    Adds:
      - config_path (absolute)
      - config_dir (absolute)
      - work_dir_abs (absolute)

    Does NOT change the YAML structure itself (keeps it as plain dict) so that
    existing stages stay compatible.
    """
    cfg_path = Path(cfg_path).expanduser().resolve()
    cfg_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    project_root = _find_project_root(cfg_dir) or cfg_dir
    cfg["project_root"] = str(project_root)

    cfg["config_path"] = str(cfg_path)
    cfg["config_dir"] = str(cfg_dir)

    # data_dir: resolve relative to config file directory (natural expectation)
    if cfg.get("data_dir"):
        cfg["data_dir"] = str(resolve_path(cfg["data_dir"], base_dir=cfg_dir))

    # work_dir: resolve robustly.
    # - "." means "use the directory of this config"
    # - paths starting with "work" are resolved relative to project_root
    # - everything else falls back to config_dir
    if cfg.get("work_dir"):
        wd_raw = _norm_path_str(str(cfg["work_dir"]).strip())
        if wd_raw in {".", "./", ".\\"}:
            wd = cfg_dir.resolve()
        else:
            wd_rel = Path(wd_raw)
            base = project_root if (wd_rel.parts and wd_rel.parts[0].lower() == "work") else cfg_dir
            wd = resolve_path(wd_rel, base_dir=base)
        cfg["work_dir"] = str(wd)
        cfg["work_dir_abs"] = str(wd)

    # resolve calib paths robustly
    wd = Path(cfg.get("work_dir", cfg_dir))
    # v5.17+: ensure standard work layout (raw/calibs/science/products/qc)
    ensure_work_layout(wd)
    calib = cfg.get("calib") or {}
    if isinstance(calib, dict):
        if calib.get("superbias_path"):
            sb_raw = Path(_norm_path_str(str(calib["superbias_path"])) )
            if sb_raw.is_absolute():
                sb = sb_raw
            else:
                # Prefer:
                # 1) inside work_dir (calib/superbias.fits)
                # 2) relative to project root (work/run1/calib/superbias.fits)
                cand1 = (wd / sb_raw).resolve()
                cand2 = (Path(project_root) / sb_raw).resolve()
                sb = cand1 if cand1.exists() else cand2
            calib["superbias_path"] = str(sb)
        if calib.get("superflat_path"):
            sf_raw = Path(_norm_path_str(str(calib["superflat_path"])) )
            if sf_raw.is_absolute():
                sf = sf_raw
            else:
                # Prefer:
                # 1) inside work_dir (calib/superflat.fits)
                # 2) relative to project root (work/run1/calib/superflat.fits)
                cand1 = (wd / sf_raw).resolve()
                cand2 = (Path(project_root) / sf_raw).resolve()
                sf = cand1 if cand1.exists() else cand2
            calib["superflat_path"] = str(sf)
        cfg["calib"] = calib

    # resolve frame paths: if relative, treat as relative to data_dir (preferred)
    frames = cfg.get("frames") or {}
    if isinstance(frames, dict):
        data_dir = Path(cfg.get("data_dir", cfg_dir))
        for k, v in list(frames.items()):
            if not isinstance(v, list):
                continue
            frames[k] = [str(resolve_path(x, base_dir=data_dir)) for x in v]
        cfg["frames"] = frames
        setup = cfg.get("frames", {}).get("__setup__")
        if isinstance(setup, dict):
            cfg["setup"] = setup
        import re

        def _deep_update(dst: dict, src: dict) -> dict:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _deep_update(dst[k], v)
                else:
                    dst[k] = v
            return dst

        profiles = cfg.get("profiles")
        setup = cfg.get("setup")
        if isinstance(profiles, dict) and isinstance(setup, dict):
            tag = "|".join(str(setup.get(k, "")) for k in ("mode", "disperser", "slit", "shape"))
            tag_l = tag.lower()
            applied = []
            for key, overrides in profiles.items():
                if not isinstance(overrides, dict):
                    continue
                if str(key).startswith("re:"):
                    if re.search(str(key)[3:], tag, flags=re.I):
                        _deep_update(cfg, overrides)
                        applied.append(str(key))
                else:
                    if str(key).lower() in tag_l:
                        _deep_update(cfg, overrides)
                        applied.append(str(key))
            if applied:
                cfg["_profiles_applied"] = applied

    return cfg


def load_config_any(cfg: Any) -> dict[str, Any]:
    """Load config from path/dict/RunContext-like objects.

    Many stages accept either a path to YAML, or an already-loaded config dict.
    This helper keeps that behavior consistent.
    """
    if isinstance(cfg, (str, Path)):
        return load_config(cfg)
    if isinstance(cfg, dict):
        return cfg

    # Common wrappers: RunContext (ui.pipeline_runner) or similar
    if hasattr(cfg, 'cfg'):
        v = getattr(cfg, 'cfg')
        if isinstance(v, dict):
            return v
    for attr in ('cfg_path', 'config_path'):
        if hasattr(cfg, attr):
            try:
                return load_config(getattr(cfg, attr))
            except Exception as e:
                logging.getLogger(__name__).warning("Failed to load config from %s=%r: %s", attr, getattr(cfg, attr), e)

    raise TypeError(f'Unsupported config type: {type(cfg)}')


def _normalize_cfg_paths_for_yaml(obj: Any):
    """Recursively normalize path-like strings for YAML portability.

    Policy: whenever we write YAML, we convert backslashes to forward slashes
    in *all* strings. This is safe in Python on Windows and avoids hard-to-debug
    portability issues when configs are moved between OSes.
    """
    if isinstance(obj, dict):
        return {k: _normalize_cfg_paths_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_cfg_paths_for_yaml(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_cfg_paths_for_yaml(v) for v in obj)
    if isinstance(obj, str):
        return obj.replace('\\', '/')
    return obj


def write_config(cfg: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        cfg_norm = _normalize_cfg_paths_for_yaml(cfg)
        yaml.safe_dump(cfg_norm, f, sort_keys=False, allow_unicode=True)
