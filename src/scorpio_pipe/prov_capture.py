"""Provenance capture utilities (P0-PROV-003).

This module is intentionally self-contained and defensive:
it must *never* crash a science run by failing to capture provenance.

What we capture
---------------
* SHA256 of each referenced raw FITS file.
* Normalized header hash (SHA256) for each raw FITS file.
* Effective (expanded) config snapshot as YAML.
* Environment snapshot: pipeline/package versions, python, OS and dependency versions.

Design notes
------------
* Hashing large datasets can be expensive. We store size/mtime alongside hashes
  and reuse cached entries when possible.
* Header normalization removes obviously volatile cards (DATE, CHECKSUM, DATASUM, etc.)
  and ignores COMMENT/HISTORY.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _volatile_header_keys() -> set[str]:
    # Obvious volatile / bookkeeping keys.
    return {
        "DATE",
        "DATE-OBS",
        "UTC",
        "UT",
        "MJD",
        "MJD-OBS",
        "JD",
        "JD-OBS",
        "CHECKSUM",
        "DATASUM",
        "FILENAME",
        "FILE",
        "ORIGIN",
    }


def normalized_header_items(header: Any) -> list[tuple[str, str]]:
    """Return a deterministic list of (KEY, VALUE) pairs suitable for hashing."""

    try:
        # astropy Header-like
        cards = list(getattr(header, "cards", []))
    except Exception:
        cards = []

    volatile = _volatile_header_keys()
    out: list[tuple[str, str]] = []
    for c in cards:
        try:
            key = str(getattr(c, "keyword", "")).strip().upper()
            if not key or key in {"COMMENT", "HISTORY"}:
                continue
            if key in volatile:
                continue
            val = getattr(c, "value", None)
            # Normalize to a stable string representation.
            if val is None:
                sval = ""
            elif isinstance(val, (int, float, bool)):
                sval = str(val)
            else:
                sval = str(val).strip()
            out.append((key, sval))
        except Exception:
            continue

    # Deterministic order.
    out.sort(key=lambda kv: kv[0])
    return out


def header_sha256_fits(path: str | Path) -> str | None:
    """Compute normalized header SHA256 for a FITS file.

    Returns None if astropy is unavailable or the header cannot be read.
    """

    try:
        from astropy.io import fits  # type: ignore
    except Exception:
        return None

    p = Path(path)
    try:
        hdr = fits.getheader(p, 0)
    except Exception:
        try:
            # Try first extension if primary is empty.
            hdr = fits.getheader(p, 1)
        except Exception:
            return None

    items = normalized_header_items(hdr)
    # Hash a canonical JSON encoding.
    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pkg_versions_full() -> dict[str, str]:
    """Best-effort dependency versions using importlib.metadata."""

    out: dict[str, str] = {}
    try:
        from importlib.metadata import distributions  # type: ignore

        for d in distributions():
            try:
                name = (d.metadata.get("Name") or "").strip()
                ver = (d.version or "").strip()
                if name and ver:
                    out[name] = ver
            except Exception:
                continue
    except Exception:
        # Fall back to a small set.
        for name in ("numpy", "astropy", "pandas", "pyyaml", "pydantic", "scipy", "matplotlib"):
            try:
                mod = __import__(name)
                out[name] = str(getattr(mod, "__version__", "unknown"))
            except Exception:
                continue
    return out


def capture_environment() -> dict[str, Any]:
    from scorpio_pipe.version import PIPELINE_VERSION, __version__

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": {"pipeline_version": PIPELINE_VERSION, "package_version": __version__},
        "python": sys.version.replace("\n", " "),
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cwd": os.getcwd(),
        "dependencies": _pkg_versions_full(),
    }


@dataclass(frozen=True)
class FileHash:
    path: str
    sha256: str
    size: int
    mtime: float
    header_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": int(self.size),
            "mtime": float(self.mtime),
            "header_sha256": self.header_sha256,
        }


def hash_files(
    paths: Iterable[str | Path],
    *,
    cache: dict[str, Any] | None = None,
    compute_header_hash: bool = True,
) -> list[dict[str, Any]]:
    """Compute hashes for files with optional cache reuse."""

    cache = cache or {}
    out: list[dict[str, Any]] = []
    for x in paths:
        p = Path(x)
        key = str(p)
        try:
            st = p.stat()
        except Exception:
            continue

        cached = cache.get(key)
        if isinstance(cached, dict):
            try:
                if int(cached.get("size", -1)) == int(st.st_size) and float(
                    cached.get("mtime", -1.0)
                ) == float(st.st_mtime):
                    out.append(dict(cached))
                    continue
            except Exception:
                pass

        sh = sha256_file(p)
        hh = header_sha256_fits(p) if compute_header_hash else None
        out.append(
            FileHash(
                path=key,
                sha256=sh,
                size=int(st.st_size),
                mtime=float(st.st_mtime),
                header_sha256=hh,
            ).as_dict()
        )

    return out


def write_effective_config_yaml(out_path: str | Path, cfg: dict[str, Any]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        txt = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    except Exception:
        # JSON fallback
        txt = json.dumps(cfg, indent=2, ensure_ascii=False)
    out_path.write_text(txt, encoding="utf-8")
    return out_path


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


def ensure_run_provenance(
    run_root: str | Path,
    cfg: dict[str, Any],
    *,
    raw_paths: Iterable[str | Path] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, str]:
    """Ensure provenance artifacts exist under run_root/manifest.

    This function is **best-effort**: it must never crash the pipeline.

    Parameters
    ----------
    run_root
        Workspace/run root.
    cfg
        Effective (fully expanded) config mapping.
    raw_paths
        Optional explicit list of raw FITS paths. If omitted, the function tries
        to collect them from ``cfg['frames']``.
    config_path
        Optional original config path (for audit only).

    Returns
    -------
    dict
        Mapping of well-known keys to absolute file paths (strings).

    Notes
    -----
    Keys used by the GUI runner (compat):
      - raw_hashes_json
      - environment_json
      - effective_config_yaml

    Additional keys:
      - raw_index_csv (human-friendly table)
      - run_signature_json (compact fingerprint)
    """

    run_root = Path(run_root)
    man = run_root / "manifest"
    man.mkdir(parents=True, exist_ok=True)

    out_p: dict[str, Path] = {}

    # Effective config
    cfg_path = man / "effective_config.yaml"
    out_p["effective_config"] = write_effective_config_yaml(cfg_path, cfg)

    # Environment
    env_p = man / "environment.json"
    try:
        atomic_write_json(env_p, capture_environment())
    except Exception:
        pass
    out_p["environment"] = env_p

    # Raw hashes
    raw_p = man / "raw_hashes.json"
    cache: dict[str, Any] = {}
    if raw_p.exists():
        try:
            cache_payload = json.loads(raw_p.read_text(encoding="utf-8"))
            if isinstance(cache_payload, dict) and isinstance(cache_payload.get("files"), list):
                for it in cache_payload.get("files", []):
                    if isinstance(it, dict) and isinstance(it.get("path"), str):
                        cache[it["path"]] = it
        except Exception:
            cache = {}

    # Determine raw paths if not passed explicitly.
    if raw_paths is None:
        frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
        raw_list: list[str] = []
        if isinstance(frames, dict):
            for k, v in frames.items():
                if k == "__setup__":
                    continue
                if isinstance(v, list):
                    raw_list.extend([str(x) for x in v if isinstance(x, (str, Path))])
        raw_paths = raw_list

    try:
        files = hash_files(raw_paths, cache=cache, compute_header_hash=True)
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_files": int(len(files)),
            "files": files,
            "config_path": str(config_path) if config_path else None,
        }
        atomic_write_json(raw_p, payload)
    except Exception:
        pass
    out_p["raw_hashes"] = raw_p

    # Bonus: raw index CSV for quick audit.
    try:
        import csv

        idx_p = man / "raw_index.csv"
        with idx_p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "sha256", "header_sha256", "size", "mtime"])
            cache_map = load_raw_hash_cache(raw_p)
            for path_str, meta in sorted(cache_map.items()):
                pp = Path(path_str)
                w.writerow([
                    path_str,
                    meta.get("sha256"),
                    meta.get("header_sha256"),
                    meta.get("size"),
                    meta.get("mtime"),
                ])
        out_p["raw_index"] = idx_p
    except Exception:
        pass

    # Bonus: compact run signature.
    try:
        sig_p = man / "run_signature.json"
        def _sha(pth: Path) -> str | None:
            try:
                return sha256_file(pth)
            except Exception:
                return None
        sig = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "effective_config_sha256": _sha(cfg_path),
            "raw_hashes_sha256": _sha(raw_p),
            "environment_sha256": _sha(env_p),
            "config_path": str(config_path) if config_path else None,
        }
        atomic_write_json(sig_p, sig)
        out_p["run_signature"] = sig_p
    except Exception:
        pass

    # Return string paths with legacy key names.
    out: dict[str, str] = {
        "effective_config_yaml": str(out_p.get("effective_config", cfg_path)),
        "environment_json": str(out_p.get("environment", env_p)),
        "raw_hashes_json": str(out_p.get("raw_hashes", raw_p)),
    }
    if out_p.get("raw_index"):
        out["raw_index_csv"] = str(out_p["raw_index"])
    if out_p.get("run_signature"):
        out["run_signature_json"] = str(out_p["run_signature"])
    return out


def load_raw_hash_cache(raw_hashes_json: str | Path) -> dict[str, Any]:
    """Load raw hash cache into mapping path->hash dict."""

    p = Path(raw_hashes_json)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        cache: dict[str, Any] = {}
        for it in payload.get("files", []) if isinstance(payload, dict) else []:
            if isinstance(it, dict) and isinstance(it.get("path"), str):
                cache[it["path"]] = it
        return cache
    except Exception:
        return {}

# -----------------------------------------------------------------------------
# Compatibility helpers (GUI runner expects these names)
# -----------------------------------------------------------------------------


def load_hash_cache(raw_hashes_json: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for :func:`load_raw_hash_cache`."""

    return load_raw_hash_cache(raw_hashes_json)



def compute_input_hashes(
    input_paths: Iterable[str | Path],
    *,
    raw_cache: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute hashes for stage inputs (best-effort).

    For raw FITS that appear in ``raw_cache`` (loaded from ``raw_hashes.json``),
    we reuse the cached sha256 + normalized header hash. For other inputs we
    compute sha256 of the file contents (no header normalization).

    Returns a list of dicts suitable for embedding into done.json.
    """

    cache = raw_cache or {}
    out: list[dict[str, Any]] = []
    for ip in input_paths or []:
        try:
            p = Path(str(ip))
        except Exception:
            continue
        if not p.exists() or not p.is_file():
            out.append({"path": str(p), "exists": False})
            continue
        key = str(p)
        if key in cache and isinstance(cache[key], dict):
            meta = dict(cache[key])
            meta["path"] = key
            meta["kind"] = meta.get("kind") or "raw"
            meta["exists"] = True
            out.append(meta)
            continue
        # Generic file hash (content sha256 only).
        try:
            h = sha256_file(p)
        except Exception:
            h = None
        try:
            st = p.stat()
            size = int(st.st_size)
            mtime = float(st.st_mtime)
        except Exception:
            size = None
            mtime = None
        out.append({
            "path": key,
            "kind": "file",
            "exists": True,
            "sha256": h,
            "size": size,
            "mtime": mtime,
        })
    return out
