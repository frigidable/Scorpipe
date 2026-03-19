from __future__ import annotations

from pathlib import Path
from typing import Iterable

from scorpio_pipe.version import __version__


_BASELINE_DIRS: dict[str, str] = {
    "microbench": "benchmarks/baselines/microbench",
    "golden_time_budget": "benchmarks/baselines/golden_time_budget",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def normalize_baseline_tag(tag: str | None) -> str:
    s = str(tag or "current").strip()
    if not s or s.lower() == "current":
        return str(__version__)
    return s


def baseline_dir(kind: str, *, root: str | Path | None = None) -> Path:
    key = str(kind or "").strip().lower()
    rel = _BASELINE_DIRS.get(key)
    if rel is None:
        known = ", ".join(sorted(_BASELINE_DIRS))
        raise KeyError(f"Unknown baseline kind {kind!r}; expected one of: {known}")
    base_root = Path(root) if root is not None else repo_root()
    return (base_root / rel).resolve()


def list_baseline_tags(kind: str, *, root: str | Path | None = None) -> list[str]:
    d = baseline_dir(kind, root=root)
    if not d.exists():
        return []
    out: list[str] = []
    for p in sorted(d.glob("*.json")):
        out.append(p.stem)
    return out


def candidate_paths(kind: str, *, version: str | None = None, root: str | Path | None = None, fallback_to_main: bool = True) -> list[Path]:
    d = baseline_dir(kind, root=root)
    tag = normalize_baseline_tag(version)
    out = [d / f"{tag}.json"]
    if fallback_to_main and tag != "main":
        out.append(d / "main.json")
    return out


def resolve_baseline_path(kind: str, *, version: str | None = None, root: str | Path | None = None, fallback_to_main: bool = True) -> Path:
    tried: list[Path] = []
    for p in candidate_paths(kind, version=version, root=root, fallback_to_main=fallback_to_main):
        tried.append(p)
        if p.exists():
            return p
    tried_s = ", ".join(str(x) for x in tried)
    raise FileNotFoundError(f"No baseline found for kind={kind!r}, version={normalize_baseline_tag(version)!r}; tried: {tried_s}")
