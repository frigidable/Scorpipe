"""Workspace / run directory validation.

P2 contract ("belt of safety")
----------------------------
The GUI and the runner must not "guess" which run is open.

This module validates:
* run directory layout: ``workspace/<night>/<obj>_<disperser>_<run_id>/``
* presence and schema of ``run.json`` (passport)
* consistency between folder name and passport fields
* legacy layout detection (stage dirs under ``products/``)

The validator is intentionally strict in runner/CI contexts (``strict=True``)
and can be used in a softer "warn + offer fix" mode in the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


class RunLayoutError(RuntimeError):
    """Raised when run directory layout or passport is invalid."""


@dataclass(frozen=True)
class RunValidation:
    run_root: Path
    night_dir: str
    night_date: str
    object_name: str
    disperser: str
    run_id: str
    legacy_stage_layout: bool
    warnings: tuple[str, ...] = ()
    mismatches: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.mismatches


_RE_NIGHT = re.compile(r"^(\d{2})_(\d{2})_(\d{4})$")


def _parse_night_dir_name(night_dir: str) -> str:
    """Return ISO date string for ``DD_MM_YYYY``."""
    m = _RE_NIGHT.match(str(night_dir or "").strip())
    if not m:
        raise RunLayoutError(
            f"Invalid night folder '{night_dir}'. Expected DD_MM_YYYY (e.g. 30_12_2025)."
        )
    dd, mm, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= dd <= 31 and 1 <= mm <= 12):
        raise RunLayoutError(
            f"Invalid night folder '{night_dir}'. Day/month out of range."
        )
    return f"{yyyy:04d}-{mm:02d}-{dd:02d}"


def _parse_run_dir_name(name: str) -> tuple[str, str, str]:
    """Parse ``<obj>_<disperser>_<run_id>``.

    Object name may contain underscores; parsing is done from the right.
    """
    s = str(name or "").strip()
    parts = s.split("_")
    if len(parts) < 3:
        raise RunLayoutError(
            f"Invalid run folder '{name}'. Expected <obj>_<disperser>_<NN>."
        )
    run_id = parts[-1]
    disp = parts[-2]
    obj = "_".join(parts[:-2])
    if not (len(run_id) == 2 and run_id.isdigit()):
        raise RunLayoutError(
            f"Invalid run_id suffix in '{name}'. Expected two digits, e.g. _01."
        )
    if not obj or not disp:
        raise RunLayoutError(
            f"Invalid run folder '{name}'. Object/disperser tokens are empty."
        )
    return obj, disp, run_id


def _require_fields(passport: dict[str, Any], keys: list[str]) -> None:
    missing = [k for k in keys if k not in passport or passport.get(k) in (None, "")]
    if missing:
        raise RunLayoutError(
            f"run.json is missing required fields: {missing}."
        )


def validate_run_dir(run_root: str | Path, *, strict: bool = True) -> RunValidation:
    """Validate run directory layout and ``run.json``.

    Parameters
    ----------
    run_root
        Path to run folder.
    strict
        If True, inconsistencies raise :class:`RunLayoutError`.
        If False, mismatches become warnings and are returned in ``mismatches``.
    """

    run_root = Path(run_root).expanduser().resolve()
    if not run_root.exists() or not run_root.is_dir():
        raise RunLayoutError(f"Run folder does not exist: {run_root}")

    night_dir = run_root.parent.name
    night_date = _parse_night_dir_name(night_dir)
    obj, disp, run_id = _parse_run_dir_name(run_root.name)

    # Sanitizer contract (Windows-safe names): forbidden characters are replaced by '_'.
    # We treat a mismatch as a warning so existing runs can be opened but users are nudged.
    warnings: list[str] = []
    try:
        from scorpio_pipe.workdir import safe_slug

        obj_s = safe_slug(obj)
        disp_s = safe_slug(disp)
        if obj_s != obj:
            warnings.append(
                f"Object token contains unsupported characters. Suggested: '{obj_s}'."
            )
        if disp_s != disp:
            warnings.append(
                f"Disperser token contains unsupported characters. Suggested: '{disp_s}'."
            )
    except Exception:
        pass

    # run.json schema
    from scorpio_pipe.run_passport import passport_path, read_run_passport

    p = passport_path(run_root)
    passport = read_run_passport(run_root)
    if passport is None:
        raise RunLayoutError(f"Missing or unreadable run.json: {p}")
    if int(passport.get("schema") or 0) != 1:
        raise RunLayoutError(
            f"Unsupported run.json schema={passport.get('schema')!r} (expected 1)."
        )

    _require_fields(
        passport,
        ["night_date", "object", "disperser", "run_id", "created_at", "pipeline_version"],
    )

    mismatches: list[str] = []

    # Required layout roots (P0/P1 contract).
    required_dirs = [
        run_root / "manifest",
        run_root / "qc",
        run_root / "ui" / "navigator",
        run_root / "ui" / "history",
    ]
    missing_dirs = [d for d in required_dirs if not d.exists()]
    if missing_dirs:
        msg = "Missing required workspace subfolders: " + ", ".join(
            str(d.relative_to(run_root)) for d in missing_dirs
        )
        if strict:
            raise RunLayoutError(msg)
        warnings.append(msg)

    def _cmp(field: str, expected: str) -> None:
        got = str(passport.get(field) or "")
        if got != expected:
            mismatches.append(
                f"{field} mismatch: folder='{expected}' vs run.json='{got}'"
            )

    _cmp("night_date", night_date)
    _cmp("object", obj)
    _cmp("disperser", disp)
    _cmp("run_id", run_id)

    # Minimal stage layout checks: either stage dirs live under run root
    # or legacy 'products/' layout is present.
    legacy_stage_layout = False
    try:
        from scorpio_pipe.workspace_migrate import detect_legacy_stage_layout

        legacy_stage_layout = bool(detect_legacy_stage_layout(run_root))
        if legacy_stage_layout:
            warnings.append(
                "Legacy stage layout detected (stage dirs under products/). "
                "Consider migrating this run via the GUI." 
            )
    except Exception:
        pass

    # Stage directory naming sanity check.
    try:
        from scorpio_pipe.stage_registry import REGISTRY

        expected = {s.dir_name for s in REGISTRY.all()}

        def _stage_dirs(root: Path) -> set[str]:
            out: set[str] = set()
            for d in root.iterdir():
                if d.is_dir() and re.match(r"^\d{2}_", d.name):
                    out.add(d.name)
            return out

        actual = _stage_dirs(run_root)
        if legacy_stage_layout:
            prod = run_root / "products"
            if prod.exists() and prod.is_dir():
                actual |= _stage_dirs(prod)

        unexpected = sorted(actual - expected)
        if unexpected:
            warnings.append(
                "Unexpected stage directories detected: "
                + ", ".join(unexpected)
                + ". (This may indicate a typo or an old layout.)"
            )
    except Exception:
        pass

    if mismatches and strict:
        raise RunLayoutError("; ".join(mismatches))
    if mismatches and not strict:
        warnings.append(
            "run.json fields do not match folder naming. Use 'Fix run.json' to repair."
        )

    return RunValidation(
        run_root=run_root,
        night_dir=night_dir,
        night_date=night_date,
        object_name=obj,
        disperser=disp,
        run_id=run_id,
        legacy_stage_layout=legacy_stage_layout,
        warnings=tuple(warnings),
        mismatches=tuple(mismatches),
    )
