from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.resources as ir

from scorpio_pipe.wavesol_paths import slugify_disperser


@dataclass(frozen=True)
class BuiltinPairs:
    slug: str
    title: str
    path: Path


def list_builtin_pairs() -> list[BuiltinPairs]:
    """List all built-in hand-pair sets bundled with the package.

    Files live in `scorpio_pipe/resources/pairs/*.txt`.
    """
    out: list[BuiltinPairs] = []

    try:
        base = ir.files("scorpio_pipe").joinpath("resources").joinpath("pairs")
        if not base.is_dir():
            return out
        for p in sorted(base.iterdir()):
            if p.is_file() and p.name.lower().endswith(".txt"):
                slug = p.stem
                out.append(BuiltinPairs(slug=slug, title=slug, path=Path(p)))
    except Exception:
        # If resources aren't packaged (running from source), fall back to filesystem.
        here = Path(__file__).resolve().parent
        base = here / "resources" / "pairs"
        if base.is_dir():
            for p in sorted(base.glob("*.txt")):
                out.append(BuiltinPairs(slug=p.stem, title=p.stem, path=p))

    return out


def find_builtin_pairs_for_disperser(disperser: str | None) -> BuiltinPairs | None:
    slug = slugify_disperser(disperser)
    for it in list_builtin_pairs():
        if it.slug == slug:
            return it
    return None
