"""Typed configuration facade (P0-ARCH-04.1).

The pipeline historically operates on a plain dict loaded from YAML.
For backward compatibility this remains true for stage implementations.

This module provides a *typed* pydantic model (``TypedConfig``) plus
pre-run validations that must fail-fast before any heavy computation.

Validation pipeline (hard gate):
  1) schema_validate (types + critical enums)
  2) mode_profile_validate (setup consistency)
  3) contracts_precheck (catalog consistency)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.schema import ConfigSchema, schema_validate


TypedConfig = ConfigSchema


@dataclass(frozen=True)
class ConfigIssue:
    level: str  # 'error' | 'warning'
    code: str
    message: str
    hint: str | None = None


class ConfigValidationError(RuntimeError):
    """Raised when config validation fails before stage execution."""

    def __init__(self, *, issues: list[ConfigIssue], config_path: str | None = None):
        self.issues = list(issues)
        self.config_path = str(config_path) if config_path else None
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        hdr = "Config validation failed (before running any stage)."
        if self.config_path:
            hdr += f"\nConfig: {self.config_path}"

        errs = [x for x in self.issues if x.level == "error"]
        warns = [x for x in self.issues if x.level == "warning"]

        lines: list[str] = [hdr]
        if errs:
            lines.append("\nErrors:")
            for it in errs[:20]:
                lines.append(f" - [{it.code}] {it.message}")
                if it.hint:
                    lines.append(f"   hint: {it.hint}")
            if len(errs) > 20:
                lines.append(f" - … ({len(errs) - 20} more)")

        if warns:
            lines.append("\nWarnings:")
            for it in warns[:10]:
                lines.append(f" - [{it.code}] {it.message}")
                if it.hint:
                    lines.append(f"   hint: {it.hint}")
            if len(warns) > 10:
                lines.append(f" - … ({len(warns) - 10} more)")

        return "\n".join(lines)


def _get_setup(cfg: dict[str, Any]) -> dict[str, Any] | None:
    setup = cfg.get("setup")
    if isinstance(setup, dict) and setup:
        return setup
    frames = cfg.get("frames")
    if isinstance(frames, dict):
        v = frames.get("__setup__")
        if isinstance(v, dict) and v:
            return v
    return None


def mode_profile_validate(cfg: dict[str, Any]) -> list[ConfigIssue]:
    """Lightweight consistency checks for setup/mode/profile selection."""

    issues: list[ConfigIssue] = []
    setup = _get_setup(cfg)
    if setup is None:
        issues.append(
            ConfigIssue(
                "error",
                "SETUP_MISSING",
                "Missing frames.__setup__/setup block; cannot determine instrument/mode/disperser.",
                "Rebuild config via GUI (Inspect → Autoconfig) or add frames.__setup__.",
            )
        )
        return issues

    # Minimal required fields for spectroscopy runs.
    required = ("mode", "disperser", "slit", "binning")
    missing = [k for k in required if not str(setup.get(k, "") or "").strip()]
    if missing:
        issues.append(
            ConfigIssue(
                "error",
                "SETUP_INCOMPLETE",
                f"setup is missing required fields: {', '.join(missing)}",
                "Fill setup fields in GUI or regenerate config.",
            )
        )

    runtime = cfg.get("runtime")
    if isinstance(runtime, dict):
        perf_mode = str(runtime.get("performance_mode") or "full").strip().lower()
        if perf_mode and perf_mode not in {"full", "fast"}:
            issues.append(
                ConfigIssue(
                    "error",
                    "PERFORMANCE_MODE_INVALID",
                    f"runtime.performance_mode must be 'fast' or 'full' (got {perf_mode!r})",
                    "Use fast for rapid-look runs and full for publication/science reruns.",
                )
            )

    # If an explicit instrument block exists, ensure it is not contradictory.
    inst = cfg.get("instrument")
    if isinstance(inst, dict):
        for k in ("mode", "disperser", "slit", "binning", "shape"):
            a = str(inst.get(k, "") or "").strip()
            b = str(setup.get(k, "") or "").strip()
            if a and b and a != b:
                issues.append(
                    ConfigIssue(
                        "warning",
                        "SETUP_MISMATCH",
                        f"instrument.{k}={a!r} differs from setup.{k}={b!r}",
                        "Prefer keeping them consistent (usually setup is the source of truth).",
                    )
                )

    return issues


def contracts_precheck(cfg: dict[str, Any]) -> list[ConfigIssue]:
    """Catalog consistency checks (products/contracts) before execution."""

    issues: list[ConfigIssue] = []
    try:
        from scorpio_pipe.catalog.stages import CATALOG as STAGE_CATALOG
        from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    except Exception as e:
        issues.append(
            ConfigIssue(
                "error",
                "CATALOG_IMPORT",
                f"Failed to import catalogs: {e}",
                "Reinstall package or report a bug.",
            )
        )
        return issues

    missing: list[str] = []
    for t in STAGE_CATALOG.task_infos():
        for pk in (t.product_keys or ()):  # type: ignore[truthy-bool]
            try:
                PRODUCT_CATALOG.get_spec(pk)
            except Exception:
                missing.append(f"{t.key}:{pk}")

    if missing:
        issues.append(
            ConfigIssue(
                "error",
                "MISSING_PRODUCT_SPEC",
                "StageCatalog refers to product keys not present in ProductCatalog: "
                + ", ".join(missing[:20])
                + (" …" if len(missing) > 20 else ""),
                "Fix catalog definitions (developer error).",
            )
        )

    return issues


def validate_pre_run(cfg: dict[str, Any]) -> list[ConfigIssue]:
    """Run the full pre-run validation pipeline."""

    issues: list[ConfigIssue] = []

    # 1) schema/type/enum validation
    sch = schema_validate(cfg)
    for it in sch.errors:
        issues.append(ConfigIssue("error", it.code, it.message, it.hint or None))
    for it in sch.warnings:
        issues.append(ConfigIssue("warning", it.code, it.message, it.hint or None))

    # 2) setup/mode/profile checks
    issues.extend(mode_profile_validate(cfg))

    # 3) catalog/contracts sanity checks
    issues.extend(contracts_precheck(cfg))

    return issues


def build_typed_config(cfg: dict[str, Any], *, config_path: str | Path | None = None) -> TypedConfig:
    """Return TypedConfig or raise ConfigValidationError."""

    issues = validate_pre_run(cfg)
    errors = [x for x in issues if x.level == "error"]
    if errors:
        raise ConfigValidationError(
            issues=issues,
            config_path=str(config_path or cfg.get("config_path") or "") or None,
        )

    # At this point schema is OK; the model_validate should succeed.
    return TypedConfig.model_validate(cfg)
