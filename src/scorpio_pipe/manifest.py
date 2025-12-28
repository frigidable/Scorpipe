from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scorpio_pipe.frame_signature import FrameSignature
from scorpio_pipe.resource_utils import resolve_resource_maybe
from scorpio_pipe.version import PIPELINE_VERSION, __version__



def _signature_summary(paths: list[str]) -> dict[str, object]:
    """Compute a compact signature summary for a list of FITS frames.

    We intentionally treat missing metadata as 'unknown' and only enforce strict
    equality when both sides provide a value.
    """
    p_list = [Path(p) for p in paths]
    if not p_list:
        return {"signature": None, "consistent": True, "mismatches": []}

    sig0 = None
    mismatches: list[dict[str, object]] = []
    for p in p_list:
        try:
            sig = FrameSignature.from_path(p)
        except Exception as e:
            mismatches.append({"path": str(p), "error": f"read_failed: {e}"})
            continue
        if sig0 is None:
            sig0 = sig
            continue
        if not sig.is_compatible_with(sig0):
            mismatches.append({"path": str(p), "signature": sig.to_dict(), "diff": sig.diff(sig0)})

    return {
        "signature": sig0.to_dict() if sig0 is not None else None,
        "consistent": len(mismatches) == 0,
        "mismatches": mismatches[:10],
    }


def _md5_file(path: Path) -> str | None:
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _pkg_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    # keep this defensive: missing optional deps shouldn't break manifest
    for name in (
        "numpy",
        "astropy",
        "pandas",
        "yaml",
        "pydantic",
        "matplotlib",
        "scipy",
    ):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            out[name] = str(ver)
        except Exception:
            continue
    return out


@dataclass(frozen=True)
class Manifest:
    payload: dict

    def to_json(self) -> str:
        return json.dumps(self.payload, indent=2, ensure_ascii=False)


def build_manifest(*, cfg: dict, cfg_path: str | Path | None = None) -> Manifest:
    """Build a reproducibility manifest.

    The manifest is intentionally stable and JSON-serializable.
    """

    now = datetime.now(timezone.utc)
    cfg_path_p = Path(cfg_path).expanduser().resolve() if cfg_path is not None else None

    work_dir = (
        Path(str(cfg.get("work_dir", ""))).expanduser().resolve()
        if cfg.get("work_dir")
        else None
    )
    config_dir = (
        Path(str(cfg.get("config_dir", ""))).expanduser().resolve()
        if cfg.get("config_dir")
        else None
    )
    project_root = (
        Path(str(cfg.get("project_root", ""))).expanduser().resolve()
        if cfg.get("project_root")
        else None
    )

    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    frames_summary: dict[str, object] = {}
    if isinstance(frames, dict):
        for k, v in frames.items():
            if k == "__setup__":
                continue
            if isinstance(v, list):
                sig = _signature_summary([str(x) for x in v])
                frames_summary[k] = {
                    "n": len(v),
                    "sample": [str(x) for x in v[:5]],
                    "frame_signature": sig.get("signature"),
                    "signature_consistent": bool(sig.get("consistent", True)),
                    "signature_mismatches": sig.get("mismatches", []),
                }

    # resources used by interactive lineid
    wcfg = cfg.get("wavesol") if isinstance(cfg.get("wavesol"), dict) else {}
    neon_lines = (wcfg or {}).get("neon_lines_csv", "neon_lines.csv")
    atlas_pdf = (wcfg or {}).get("atlas_pdf", "HeNeAr_atlas.pdf")

    neon_res = resolve_resource_maybe(
        neon_lines,
        work_dir=work_dir,
        config_dir=config_dir,
        project_root=project_root,
    )
    atlas_res = resolve_resource_maybe(
        atlas_pdf,
        work_dir=work_dir,
        config_dir=config_dir,
        project_root=project_root,
    )

    payload: dict[str, object] = {
        "pipeline": {
            "pipeline_version": PIPELINE_VERSION,
            "package_version": __version__,
        },
        "created_utc": now.isoformat(),
        "platform": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": _pkg_versions(),
        "paths": {
            "config_path": str(cfg_path_p)
            if cfg_path_p
            else str(cfg.get("config_path", "")),
            "work_dir": str(work_dir) if work_dir else str(cfg.get("work_dir", "")),
            "data_dir": str(cfg.get("data_dir", "")),
            "project_root": str(project_root)
            if project_root
            else str(cfg.get("project_root", "")),
        },
        "setup": cfg.get("setup", cfg.get("frames", {}).get("__setup__", {})),
        "frames": frames_summary,
        "resources": {
            "neon_lines_csv": {
                "requested": str(neon_lines),
                "resolved": str(neon_res.path) if neon_res else None,
                "source": neon_res.source if neon_res else None,
            },
            "atlas_pdf": {
                "requested": str(atlas_pdf),
                "resolved": str(atlas_res.path) if atlas_res else None,
                "source": atlas_res.source if atlas_res else None,
            },
        },
        "config": {
            "md5": _md5_file(cfg_path_p)
            if cfg_path_p and cfg_path_p.exists()
            else None,
            "profiles_applied": cfg.get("_profiles_applied"),
        },
    }

    return Manifest(payload=payload)


def write_manifest(
    *, out_path: str | Path, cfg: dict, cfg_path: str | Path | None = None
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = build_manifest(cfg=cfg, cfg_path=cfg_path)
    out_path.write_text(m.to_json(), encoding="utf-8")
    return out_path
