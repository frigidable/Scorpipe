from __future__ import annotations

"""Project-level frame role manifest.

This module implements **BL-P1-PROJ-010**: a user-editable YAML file that
explicitly declares which raw frames play which roles (obj/sky/arc/flat/bias).

Rationale
---------
On many SCORPIO data sets, *sky frames are not reliably marked in headers/logs*.
The pipeline therefore needs an explicit "source of truth" that the user can
edit once and keep reproducible.

File
----
By default we look for ``project_manifest.yaml`` in the run/work directory.

Minimal schema (human-friendly)
------------------------------

roles:
  OBJECT_FRAMES:
    files: ["raw/obj1.fits", ...]
    globs: ["raw/*NGC*.fits"]
  SKY_FRAMES:
    files: []
    globs: []
  ARCS:
    files: []
    globs: []
  FLATS:
    files: []
    globs: []
  BIAS:
    files: []
    globs: []

Optional grouping placeholders are allowed but not interpreted yet:

groups:
  night_YYYYMMDD:
    roles: { ... }
active_group: night_YYYYMMDD

Notes
-----
- Paths may be absolute or relative. Relative paths are resolved against
  ``data_dir`` (preferred), then the manifest directory.
- The manifest only overwrites roles that are explicitly present.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


ROLE_TO_FRAMES_KEY: dict[str, str] = {
    "OBJECT_FRAMES": "obj",
    "SKY_FRAMES": "sky",
    "ARCS": "neon",
    "FLATS": "flat",
    "BIAS": "bias",
    # optional
    "SUNSKY_FRAMES": "sunsky",
    "SUNSKY": "sunsky",
}

FRAMES_KEY_TO_ROLE: dict[str, str] = {v: k for k, v in ROLE_TO_FRAMES_KEY.items() if k in {"OBJECT_FRAMES","SKY_FRAMES","ARCS","FLATS","BIAS","SUNSKY_FRAMES","SUNSKY"}}

DEFAULT_MANIFEST_NAME = "project_manifest.yaml"


def default_project_manifest_dict() -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ("OBJECT_FRAMES", "SKY_FRAMES", "ARCS", "FLATS", "BIAS"):
        roles[role] = {"files": [], "globs": []}
    # keep optional sunsky for SCORPIO flats variants
    roles["SUNSKY_FRAMES"] = {"files": [], "globs": []}
    return {
        "schema": "scorpio-pipe.project-manifest.v1",
        "roles": roles,
        "groups": {},
        "active_group": None,
    }


def write_default_project_manifest(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(
            yaml.safe_dump(default_project_manifest_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return p


def find_project_manifest(*, work_dir: Path | None, config_dir: Path | None) -> Path | None:
    """Find manifest by convention.

    Priority:
      1) work_dir/project_manifest.yaml
      2) config_dir/project_manifest.yaml
    """

    if work_dir is not None:
        cand = (Path(work_dir) / DEFAULT_MANIFEST_NAME)
        if cand.is_file():
            return cand
    if config_dir is not None:
        cand = (Path(config_dir) / DEFAULT_MANIFEST_NAME)
        if cand.is_file():
            return cand
    return None


def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/").strip()


def _resolve_one(p: str | Path, *, data_dir: Path, manifest_dir: Path) -> Path:
    pp = Path(_norm_path(str(p))).expanduser()
    if pp.is_absolute():
        return pp.resolve()
    # Prefer data_dir (raw frames live there)
    cand = (data_dir / pp).resolve()
    if cand.exists():
        return cand
    return (manifest_dir / pp).resolve()


def _expand_globs(patterns: list[str], *, data_dir: Path, manifest_dir: Path) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        pat_s = _norm_path(str(pat))
        base = data_dir
        # allow patterns anchored to manifest dir with "./".
        if pat_s.startswith("./"):
            base = manifest_dir
            pat_s = pat_s[2:]
        for p in sorted(base.glob(pat_s)):
            if p.is_file():
                out.append(p.resolve())
    return out


def _parse_role_entry(v: Any) -> tuple[list[str], list[str]]:
    """Return (files, globs) from a role entry.

    Accepts:
      - list[str]  -> treated as files
      - {files: [...], globs: [...]}
    """
    if v is None:
        return [], []
    if isinstance(v, list):
        return [str(x) for x in v], []
    if isinstance(v, dict):
        files = v.get("files", [])
        globs = v.get("globs", [])
        if isinstance(files, str):
            files = [files]
        if isinstance(globs, str):
            globs = [globs]
        return [str(x) for x in (files or [])], [str(x) for x in (globs or [])]
    return [], []


def load_project_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(d, dict):
        raise ValueError("project_manifest.yaml must be a mapping")
    return d


def roles_present(manifest: dict[str, Any]) -> set[str]:
    roles = manifest.get("roles")
    if not isinstance(roles, dict):
        return set()
    return {str(k) for k in roles.keys()}


def apply_project_manifest_to_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply project_manifest.yaml onto cfg.frames.

    - Only roles explicitly present in the manifest are applied.
    - Adds cfg['_project_manifest'] metadata so stages/UI can gate features.

    Returns cfg (mutated for convenience).
    """

    # best-effort paths
    work_dir = Path(str(cfg.get("work_dir", ""))).expanduser() if cfg.get("work_dir") else None
    config_dir = Path(str(cfg.get("config_dir", ""))).expanduser() if cfg.get("config_dir") else None

    if work_dir is not None and not work_dir.is_absolute() and config_dir is not None:
        work_dir = (config_dir / work_dir).resolve()
    if config_dir is not None:
        config_dir = config_dir.resolve()

    manifest_path = None
    if cfg.get("project_manifest_path"):
        manifest_path = Path(str(cfg["project_manifest_path"])).expanduser()
        if not manifest_path.is_absolute() and work_dir is not None:
            manifest_path = (work_dir / manifest_path).resolve()
    else:
        manifest_path = find_project_manifest(work_dir=work_dir, config_dir=config_dir)

    meta: dict[str, Any] = {
        "has_manifest": False,
        "path": None,
        "roles_present": [],
        "has_sky_frames": False,
        "n_sky_frames": 0,
        "frames_source": "config",
    }

    if manifest_path is None or not Path(manifest_path).is_file():
        cfg["_project_manifest"] = meta
        return cfg

    manifest_path = Path(manifest_path).resolve()
    manifest_dir = manifest_path.parent
    manifest = load_project_manifest(manifest_path)

    # choose group if requested
    active_group = manifest.get("active_group")
    if active_group and isinstance(manifest.get("groups"), dict):
        g = manifest.get("groups", {}).get(active_group)
        if isinstance(g, dict):
            # group may override roles
            if isinstance(g.get("roles"), dict):
                manifest = dict(manifest)
                manifest["roles"] = g["roles"]

    roles = manifest.get("roles")
    if not isinstance(roles, dict):
        roles = {}

    # data_dir resolution
    data_dir = Path(str(cfg.get("data_dir") or (config_dir or manifest_dir))).expanduser().resolve()

    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    if not isinstance(frames, dict):
        frames = {}

    applied_roles: list[str] = []

    for role, frames_key in ROLE_TO_FRAMES_KEY.items():
        if role not in roles:
            continue
        files_s, globs_s = _parse_role_entry(roles.get(role))
        files = [_resolve_one(x, data_dir=data_dir, manifest_dir=manifest_dir) for x in files_s]
        globbed = _expand_globs(globs_s, data_dir=data_dir, manifest_dir=manifest_dir)
        # de-duplicate preserving order
        out_paths: list[str] = []
        seen: set[str] = set()
        for p in [*files, *globbed]:
            ps = str(p)
            if ps not in seen:
                seen.add(ps)
                out_paths.append(ps)
        frames[frames_key] = out_paths
        applied_roles.append(role)

    cfg["frames"] = frames

    sky_list = frames.get("sky") if isinstance(frames.get("sky"), list) else []

    meta.update(
        {
            "has_manifest": True,
            "path": str(manifest_path),
            "roles_present": sorted(list(roles_present(manifest))),
            "has_sky_frames": bool(sky_list),
            "n_sky_frames": int(len(sky_list)) if isinstance(sky_list, list) else 0,
            "frames_source": "manifest",
        }
    )

    cfg["_project_manifest"] = meta
    return cfg
