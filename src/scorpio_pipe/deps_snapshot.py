"""Dependency snapshot + lock helpers (P0-ARCH-10).

Goals
-----
* Provide a deterministic *full* dependency snapshot for provenance/QC.
* Provide a deterministic lock file (requirements.lock) that can be hashed and
  packaged alongside results.

Design
------
We intentionally avoid third-party dependencies. The lock is generated from
``importlib.metadata`` distributions available in the current environment.

This does not replace a repo-level resolver lock (uv.lock / pip-tools), but it
makes every run/export self-contained and explainable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_name(name: str) -> str:
    # PEP 503-ish normalization (good enough for our purpose)
    return str(name or "").strip().lower().replace("_", "-")


def collect_packages() -> list[dict[str, str]]:
    """Return a deterministic list of installed packages with versions."""

    pkgs: dict[str, str] = {}
    try:
        from importlib.metadata import distributions  # type: ignore

        for d in distributions():
            try:
                raw = (d.metadata.get("Name") or "").strip()
                ver = (d.version or "").strip()
                if not raw or not ver:
                    continue
                key = _norm_name(raw)
                # Prefer the first occurrence for a normalized key to keep stable.
                pkgs.setdefault(key, ver)
            except Exception:
                continue
    except Exception:
        # Defensive fallback: at least capture core stack.
        for mod_name in ("numpy", "astropy", "pandas", "pyyaml", "pydantic", "scipy", "matplotlib"):
            try:
                mod = __import__(mod_name)
                ver = str(getattr(mod, "__version__", "unknown"))
                pkgs[_norm_name(mod_name)] = ver
            except Exception:
                continue

    out = [{"name": n, "version": v} for n, v in sorted(pkgs.items(), key=lambda kv: kv[0])]
    return out


def build_deps_snapshot() -> dict[str, Any]:
    """Build a full dependency snapshot payload."""

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": collect_packages(),
    }


def write_deps_snapshot_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = build_deps_snapshot()
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _iter_lock_lines(pkgs: Iterable[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for it in pkgs:
        try:
            name = _norm_name(it.get("name", ""))
            ver = str(it.get("version", "")).strip()
            if not name or not ver:
                continue
            lines.append(f"{name}=={ver}")
        except Exception:
            continue
    # Deduplicate, deterministic order
    uniq = sorted(set(lines), key=lambda s: s)
    return uniq


def write_requirements_lock(
    path: str | Path,
    *,
    packages: Iterable[dict[str, str]] | None = None,
    header: dict[str, str] | None = None,
) -> str:
    """Write a deterministic requirements.lock and return its sha256."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if packages is None:
        packages = collect_packages()

    meta = header or {}
    created = meta.get("created_utc") or datetime.now(timezone.utc).isoformat()

    out_lines: list[str] = []
    out_lines.append(f"# requirements.lock")
    out_lines.append(f"# created_utc: {created}")
    if meta.get("python"):
        out_lines.append(f"# python: {meta['python']}")
    if meta.get("platform"):
        out_lines.append(f"# platform: {meta['platform']}")
    if meta.get("pipeline_version"):
        out_lines.append(f"# pipeline_version: {meta['pipeline_version']}")
    out_lines.append("#")

    out_lines.extend(_iter_lock_lines(packages))
    data = ("\n".join(out_lines) + "\n").encode("utf-8")
    p.write_bytes(data)
    return sha256_bytes(data)
