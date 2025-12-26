from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scorpio_pipe.schema import find_unknown_keys, schema_validate


@dataclass
class ValidationIssue:
    level: str  # 'error' | 'warning'
    code: str
    message: str
    hint: str | None = None


@dataclass
class ValidationReport:
    ok: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]


def _get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_abs(p: str | Path, base_dir: Path) -> Path:
    pp = Path(str(p)).expanduser()
    if not pp.is_absolute():
        pp = (base_dir / pp).resolve()
    return pp


def validate_config(
    cfg_or_path: dict[str, Any] | str | Path,
    *,
    base_dir: Path | None = None,
    strict_paths: bool = True,
) -> ValidationReport:
    """Validate config.

    Parameters
    ----------
    cfg_or_path : dict | path
        Either a loaded config dict or a path to config.yaml.
    base_dir : Path, optional
        Base dir for resolving relative paths. If cfg_or_path is a path, defaults
        to its parent.
    strict_paths : bool
        If False, missing files are reported as warnings (useful for early GUI
        preview / partially-built work dirs).

    Returns
    -------
    ValidationReport
        Pragmatic validation report used by both GUI and CLI.
    """

    cfg_path: Path | None = None
    cfg: dict[str, Any]

    if isinstance(cfg_or_path, dict):
        cfg = cfg_or_path
        base_dir = (base_dir or Path.cwd()).resolve()
    else:
        cfg_path = Path(cfg_or_path).expanduser().resolve()
        from scorpio_pipe.config import load_config

        cfg = load_config(cfg_path)
        base_dir = (base_dir or cfg_path.parent).resolve()

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    # --- schema (typos / types) ---
    sch = schema_validate(cfg)
    for it in sch.errors:
        errors.append(ValidationIssue("error", it.code, it.message, it.hint))
    for it in sch.warnings:
        warnings.append(ValidationIssue("warning", it.code, it.message, it.hint))

    # --- unknown keys (fail-fast) ---
    strict_unknown = bool(cfg.get("strict_unknown", True))
    if strict_unknown:
        unknown = find_unknown_keys(cfg)
        if unknown:
            msg = "; ".join(
                f"{sec}: {', '.join(keys)}" for sec, keys in unknown.items()
            )
            errors.append(
                ValidationIssue(
                    "error",
                    "UNKNOWN_KEYS",
                    f"Unknown config keys: {msg}",
                    "Fix typos or disable strict_unknown",
                )
            )

    # --- basic paths ---
    data_dir = _get(cfg, "data_dir")
    work_dir = _get(cfg, "work_dir")

    if not data_dir:
        errors.append(
            ValidationIssue(
                "error", "DATA_DIR", "data_dir is missing", "Select the night folder"
            )
        )
    else:
        p = _as_abs(str(data_dir), base_dir)
        if not p.exists():
            (errors if strict_paths else warnings).append(
                ValidationIssue(
                    "error" if strict_paths else "warning",
                    "DATA_DIR",
                    f"data_dir does not exist: {p}",
                    "Fix the path",
                )
            )

    if not work_dir:
        errors.append(
            ValidationIssue(
                "error", "WORK_DIR", "work_dir is missing", "Choose a work directory"
            )
        )
    else:
        try:
            _as_abs(str(work_dir), base_dir)
        except Exception:
            errors.append(
                ValidationIssue(
                    "error", "WORK_DIR", f"Bad work_dir: {work_dir}", "Fix the path"
                )
            )

    # --- frames ---
    frames = _get(cfg, "frames", {}) or {}
    if not isinstance(frames, dict):
        errors.append(
            ValidationIssue(
                "error", "FRAMES", "frames must be a mapping", "Recreate config"
            )
        )
        frames = {}

    for kind in ("bias", "flat", "neon", "obj", "sky", "sunsky"):
        v = frames.get(kind)
        if v is None:
            continue
        if not isinstance(v, list):
            errors.append(
                ValidationIssue(
                    "error", "FRAMES", f"frames.{kind} must be a list", "Fix YAML"
                )
            )
            continue
        for fp in v:
            try:
                p = _as_abs(str(fp), base_dir)
                if not p.exists():
                    (errors if strict_paths else warnings).append(
                        ValidationIssue(
                            "error" if strict_paths else "warning",
                            "MISSING_FRAME",
                            f"Missing file: {kind}: {p}",
                            "If you moved the night folder, rebuild config",
                        )
                    )
            except Exception:
                warnings.append(
                    ValidationIssue(
                        "warning",
                        "BAD_FRAME",
                        f"Bad frame path in {kind}: {fp}",
                        "Fix YAML",
                    )
                )

    ok = len(errors) == 0
    return ValidationReport(ok=ok, errors=errors, warnings=warnings)
