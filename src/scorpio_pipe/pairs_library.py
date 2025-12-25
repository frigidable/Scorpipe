from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import re
import shutil
import zipfile

from scorpio_pipe.wavesol_paths import slugify_disperser
from scorpio_pipe.app_paths import user_data_root


ENV_USER_PAIRS_DIR = "SCORPIO_PIPE_USER_PAIRS_DIR"


@dataclass(frozen=True)
class PairSet:
    """A pair list file plus metadata."""

    label: str
    path: Path
    origin: str  # "builtin" | "user"


def _setup_from_cfg_or_setup(arg: object) -> dict:
    """Accept either full cfg dict or just a __setup__ dict."""
    if not isinstance(arg, dict):
        return {}
    if "frames" in arg and isinstance(arg.get("frames"), dict):
        setup = arg.get("frames", {}).get("__setup__", {})
        if isinstance(setup, dict):
            return setup
    setup = arg.get("__setup__")
    if isinstance(setup, dict):
        return setup
    return arg


def _setup_fields(setup: dict) -> tuple[str, str, str, str]:
    inst = str(setup.get("instrument") or setup.get("instr") or setup.get("instrume") or "unknown")
    disp = str(setup.get("disperser") or setup.get("grism") or setup.get("grism_name") or "unknown")
    slit = str(setup.get("slit") or setup.get("slit_width") or "unknown")
    binning = str(setup.get("binning") or setup.get("bin") or "unknown")
    return inst, disp, slit, binning


def _builtin_slug_for(arg: object) -> str:
    if isinstance(arg, str):
        return slugify_disperser(arg)
    setup = _setup_from_cfg_or_setup(arg)
    _, disp, _, _ = _setup_fields(setup)
    return slugify_disperser(disp)


def _user_slug_for(arg: object) -> str:
    # User library is keyed by (instrument, disperser, slit, binning)
    if isinstance(arg, str):
        return slugify_disperser(arg)
    setup = _setup_from_cfg_or_setup(arg)
    inst, disp, slit, binning = _setup_fields(setup)
    key = f"{inst}__{disp}__slit{slit}__bin{binning}"
    return slugify_disperser(key)


def builtin_pairs_root() -> Path:
    """Built-in pairs shipped with the package (may be empty)."""
    return Path(__file__).resolve().parent / "resources" / "pairs"


def user_pairs_root() -> Path:
    """Persistent user library root.

    Default: ~/.scorpio_pipe/pairs
    Override: set env var SCORPIO_PIPE_USER_PAIRS_DIR
    """
    env = (os.environ.get(ENV_USER_PAIRS_DIR) or "").strip()
    if env:
        return Path(env).expanduser()
    return user_data_root("Scorpipe") / "pairs"


def ensure_user_pairs_root() -> Path:
    root = user_pairs_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_pair_sets(setup_or_disperser: str | dict) -> list[PairSet]:
    """List available pair sets.

    - Built-in sets are keyed by disperser.
    - User sets are keyed by (instrument, disperser, slit, binning) so they don't collide.

    For backward compatibility we also search the legacy user folder keyed only by disperser.
    """
    builtin_slug = _builtin_slug_for(setup_or_disperser)
    user_slug = _user_slug_for(setup_or_disperser)

    out: list[PairSet] = []
    seen: set[Path] = set()

    bdir = builtin_pairs_root() / builtin_slug
    if bdir.exists():
        for p in sorted(bdir.glob("*.txt")):
            if p not in seen:
                out.append(PairSet(label=p.stem, path=p, origin="builtin"))
                seen.add(p)

    roots = [ensure_user_pairs_root() / user_slug]
    if user_slug != builtin_slug:
        roots.append(ensure_user_pairs_root() / builtin_slug)  # legacy
    for udir in roots:
        if udir.exists():
            for p in sorted(udir.glob("*.txt")):
                if p not in seen:
                    out.append(PairSet(label=p.stem, path=p, origin="user"))
                    seen.add(p)

    return out


def find_builtin_pairs_for_disperser(disperser: str) -> Path | None:
    """Return any built-in pair file for the disperser (first match), if present."""
    for s in list_pair_sets(disperser):
        if s.origin == "builtin":
            return s.path
    return None


def _sanitize_label(label: str) -> str:
    label = (label or "").strip()
    if not label:
        return "pairs"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return label[:64]


def save_user_pair_set(setup_or_disperser: str | dict, source: Path, label: str | None = None) -> Path:
    """Copy a pair file into the user library and return destination path."""
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))

    slug = _user_slug_for(setup_or_disperser)
    root = ensure_user_pairs_root() / slug
    root.mkdir(parents=True, exist_ok=True)

    if label is None:
        label = source.stem
    dest = root / f"{_sanitize_label(label)}.txt"
    shutil.copy2(source, dest)
    return dest


def copy_pair_set_to_workdir(disperser: str, work_dir: Path, pair_file: Path, filename: str = "hand_pairs.txt") -> Path:
    """Copy a selected pair file into the workdir wavesol/<grism>/ folder."""
    work_dir = Path(work_dir).expanduser().resolve()
    pair_file = Path(pair_file).expanduser().resolve()
    if not pair_file.exists():
        raise FileNotFoundError(str(pair_file))

    slug = slugify_disperser(disperser or "unknown")

    ddir = work_dir / "wavesol" / slug
    ddir.mkdir(parents=True, exist_ok=True)
    dest = ddir / filename
    shutil.copy2(pair_file, dest)
    return dest

# --------------------------- export helpers ---------------------------

def export_pair_set(pair_file: Path, dest_path: Path) -> Path:
    """Export a single pair set file to an arbitrary path (copy)."""
    pair_file = Path(pair_file).expanduser().resolve()
    if not pair_file.exists():
        raise FileNotFoundError(str(pair_file))
    dest_path = Path(dest_path).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pair_file, dest_path)
    return dest_path


def export_user_library_zip(dest_zip: Path, *, include_builtin: bool = False) -> Path:
    """Export the pair library into a zip archive."""
    dest_zip = Path(dest_zip).expanduser().resolve()
    dest_zip.parent.mkdir(parents=True, exist_ok=True)

    user_root = ensure_user_pairs_root()
    builtin_root = builtin_pairs_root()

    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if user_root.exists():
            for p in user_root.rglob("*.txt"):
                arc = Path("user") / p.relative_to(user_root)
                z.write(p, str(arc))

        if include_builtin and builtin_root.exists():
            for p in builtin_root.rglob("*.txt"):
                arc = Path("builtin") / p.relative_to(builtin_root)
                z.write(p, str(arc))

        # small manifest for convenience
        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "user_root": str(user_root),
            "include_builtin": bool(include_builtin),
        }
        z.writestr("MANIFEST.json", __import__("json").dumps(meta, indent=2))

    return dest_zip


def export_user_library_folder(dest_dir: Path, *, include_builtin: bool = False) -> Path:
    """Export the pair library into a folder."""
    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    user_root = ensure_user_pairs_root()
    if user_root.exists():
        tgt = dest_dir / "user"
        if tgt.exists():
            shutil.rmtree(tgt)
        shutil.copytree(user_root, tgt)

    if include_builtin:
        builtin_root = builtin_pairs_root()
        if builtin_root.exists():
            tgt = dest_dir / "builtin"
            if tgt.exists():
                shutil.rmtree(tgt)
            shutil.copytree(builtin_root, tgt)

    return dest_dir
