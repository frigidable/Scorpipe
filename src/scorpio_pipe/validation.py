from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any

from scorpio_pipe.config import load_config
from scorpio_pipe.resource_utils import resolve_resource_maybe
from scorpio_pipe.schema import schema_validate


Level = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    level: Level
    code: str
    message: str
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
        }


@dataclass
class ValidationReport:
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


def validate_config(cfg: dict | str | Path, *, strict_paths: bool = False) -> ValidationReport:
    """Validate configuration for early, user-friendly failures."""

    cfg_path: Path | None = None
    if isinstance(cfg, (str, Path)):
        cfg_path = Path(cfg).expanduser().resolve()
        cfg_d = load_config(cfg_path)
    elif isinstance(cfg, dict):
        cfg_d = cfg
    else:
        raise TypeError(f"Unsupported config type: {type(cfg)}")

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    def err(code: str, message: str, hint: str | None = None) -> None:
        errors.append(ValidationIssue("error", code, message, hint))

    def warn(code: str, message: str, hint: str | None = None) -> None:
        warnings.append(ValidationIssue("warning", code, message, hint))

    # Pydantic schema/type checks (backward compatible)
    sch = schema_validate(cfg_d)
    for i in sch.errors:
        err(i.code, i.message, i.hint)
    for i in sch.warnings:
        warn(i.code, i.message, i.hint)

    # Required top-level
    if not str(cfg_d.get("work_dir", "")).strip():
        err("CFG_WORK_DIR", "config.work_dir is missing")
    if not str(cfg_d.get("data_dir", "")).strip():
        err("CFG_DATA_DIR", "config.data_dir is missing")

    work_dir = Path(str(cfg_d.get("work_dir", "."))).expanduser().resolve()
    config_dir = Path(str(cfg_d.get("config_dir", work_dir))).expanduser().resolve()
    project_root = Path(str(cfg_d.get("project_root", config_dir))).expanduser().resolve()

    # Frames
    frames = cfg_d.get("frames")
    if not isinstance(frames, dict):
        err("CFG_FRAMES", "config.frames must be a mapping")
        frames = {}

    # Minimal expectations (soft)
    if "obj" not in frames or not isinstance(frames.get("obj"), list) or len(frames.get("obj", [])) == 0:
        warn("CFG_NO_OBJ", "No science frames listed in frames.obj")

    # Validate paths
    def _check_list(key: str) -> None:
        v = frames.get(key)
        if v is None:
            return
        if not isinstance(v, list):
            err("CFG_FRAME_LIST", f"frames.{key} must be a list")
            return
        missing = []
        for raw in v:
            try:
                p = Path(str(raw))
            except Exception:
                missing.append(str(raw))
                continue
            if not p.exists():
                missing.append(str(p))
        if missing:
            msg = f"{len(missing)} file(s) in frames.{key} do not exist"
            if strict_paths:
                err("CFG_MISSING_FILES", msg, hint="Fix paths or re-run inspect/autoconfig")
            else:
                warn("CFG_MISSING_FILES", msg, hint="Some stages may fail later")

    for k in ("bias", "flat", "neon", "obj", "sky"):
        _check_list(k)

    # Resource checks
    wcfg = cfg_d.get("wavesol") if isinstance(cfg_d.get("wavesol"), dict) else {}
    neon_lines = (wcfg or {}).get("neon_lines_csv", "neon_lines.csv")
    atlas_pdf = (wcfg or {}).get("atlas_pdf", "HeNeAr_atlas.pdf")

    neon_res = resolve_resource_maybe(
        neon_lines,
        work_dir=work_dir,
        config_dir=config_dir,
        project_root=project_root,
        allow_package=True,
    )
    if neon_res is None:
        err(
            "CFG_NEON_LINES",
            f"Neon line list not found: {neon_lines}",
            hint="Put neon_lines.csv near work_dir/config.yaml, or rely on packaged resource in v4.1+",
        )

    atlas_res = resolve_resource_maybe(
        atlas_pdf,
        work_dir=work_dir,
        config_dir=config_dir,
        project_root=project_root,
        allow_package=True,
    )
    if atlas_res is None:
        warn(
            "CFG_ATLAS_PDF",
            f"Atlas PDF not found: {atlas_pdf}",
            hint="Optional: used for interactive line identification",
        )

    # Hygiene
    if cfg_path is not None:
        if "\\" in str(cfg_path):
            warn("CFG_PATH_SLASH", "Config path contains backslashes (Windows).", hint="This is OK, but keep paths normalized if running on Linux/WSL")

    return ValidationReport(errors=errors, warnings=warnings)
