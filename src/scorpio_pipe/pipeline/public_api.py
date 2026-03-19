"""Frozen public API boundary for application layers (UI/CLI).

P0-ARCH-12 — Hard freeze of public boundaries
-------------------------------------------
The rule is simple:

- **UI and CLI code must not import pipeline internals directly.**
- They import **only** :mod:`scorpio_pipe.pipeline.public_api`.

This module stays **small, stable, and import-light**:
- It re-exports catalog *data* (StageCatalog) for UI lists/help text.
- It provides *thin wrappers* that lazy-import the heavy execution engine.
- It hosts a few GUI helper wrappers that must not pull stage modules on import.

If you need a new capability for UI/CLI, add it here (or add a tiny helper
module under ``scorpio_pipe.pipeline`` and re-export it here).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from scorpio_pipe.catalog.stages import (
    CATALOG as STAGE_CATALOG,
    GuiStageDef,
    StageCatalog,
    StageInfo,
    iter_gui_stages as _iter_gui_stages,
)
from scorpio_pipe.config import load_config_any
from scorpio_pipe.pipeline.context import CancelToken, PlanItem, StageContext, StageResult, StageSpec  # noqa: F401
from scorpio_pipe.pipeline.session_store import (  # noqa: F401
    load_session,
    save_session,
    session_path,
    snapshot,
    update_stage,
)
from scorpio_pipe.wavesol_paths import wavesol_dir


# -----------------------------------------------------------------------------
# Catalog-facing exports (lightweight)
# -----------------------------------------------------------------------------


def get_stage_catalog() -> StageCatalog:
    """Return the canonical :class:`~scorpio_pipe.catalog.stages.StageCatalog` instance."""

    return STAGE_CATALOG


def get_task(name: str) -> StageInfo:
    """Return :class:`~scorpio_pipe.catalog.stages.StageInfo` for a task (aliases accepted)."""

    return STAGE_CATALOG.get_task(name)


def get_gui_stage(key: str) -> GuiStageDef:
    """Return GUI stage definition (task/stage aliases accepted)."""

    return STAGE_CATALOG.get_gui_stage(key)



def iter_gui_stages() -> Iterator[GuiStageDef]:
    """Iterate GUI stages (incl. UI-only placeholders)."""

    yield from _iter_gui_stages()


CANONICAL_TASKS: tuple[str, ...] = tuple(STAGE_CATALOG.task_keys())

_LEGACY_STAGE_SPEC_KINDS = {"none", "mef", "spec1d", "lambda_map", "calib_bias_triplet", "calib_flat", "calib_neon"}


def _legacy_stage_spec_kind(task) -> str:
    kind = str(getattr(task, "contract_kind", "none") or "none").strip().lower()
    if kind in _LEGACY_STAGE_SPEC_KINDS:
        return kind
    if kind in {"standard_prepare", "sensfunc", "telluric_model", "line_measure", "science_report"}:
        return "spec1d"
    if kind in {"fluxcal_apply"}:
        return "mef"
    return "none"


STAGE_SPECS: dict[str, StageSpec] = {
    t.key: StageSpec(
        key=t.key,
        title=str(t.label),
        validate_globs=t.validate_globs,
        contract_kind=_legacy_stage_spec_kind(t),
        output_keys=t.product_keys,
    )
    for t in STAGE_CATALOG.task_infos()
}


def canonical_task_name(name: str) -> str:
    """Resolve aliases to a canonical task key (StageCatalog is the source of truth)."""

    return STAGE_CATALOG.canonical_task_name(name)


# Canonical task -> entrypoint string.
TASK_ENTRYPOINTS: dict[str, str] = {
    t.key: str(t.entrypoint or '').strip() for t in STAGE_CATALOG.task_infos()
}

# Canonical task + aliases -> entrypoint string (UI help text / membership checks).
class _TaskEntry(str):
    pass

TASKS: dict[str, _TaskEntry] = {}
for _t in STAGE_CATALOG.task_infos():
    _ep = str(_t.entrypoint or '').strip()
    if not _ep:
        continue
    TASKS[_t.key] = _TaskEntry(_ep)
    for _a in (_t.aliases or ()):  # type: ignore[truthy-bool]
        aa = str(_a).strip()
        if aa:
            TASKS[aa] = _TaskEntry(_ep)


def validate_and_canonicalize_tasks(task_names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Validate task names against StageCatalog and return (canon_tasks, bad_tasks)."""

    bad: list[str] = []
    canon_tasks: list[str] = []
    seen: set[str] = set()

    for raw in task_names:
        try:
            # get_task() accepts aliases; raises KeyError for unknown names.
            _ = STAGE_CATALOG.get_task(raw)
            t = STAGE_CATALOG.canonical_task_name(raw)
        except Exception:
            bad.append(str(raw))
            continue
        if t not in seen:
            seen.add(t)
            canon_tasks.append(t)

    return canon_tasks, bad


# -----------------------------------------------------------------------------
# UI-friendly run context (lightweight)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunContext:
    """Small wrapper used by the GUI (kept in the frozen public API)."""

    cfg_path: Path
    cfg: dict[str, Any]


def load_context(cfg_path: str | Path) -> RunContext:
    """Load config from disk and return a GUI-friendly context."""

    p = Path(cfg_path)
    cfg = load_config_any(p)
    return RunContext(cfg_path=p, cfg=cfg)


# -----------------------------------------------------------------------------
# Engine wrappers (lazy import)
# -----------------------------------------------------------------------------


def plan_sequence(
    cfg_or_path: Mapping[str, Any] | str | Path | RunContext,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    config_path: Path | None = None,
) -> list[PlanItem]:
    from scorpio_pipe.pipeline.engine import plan_sequence as _plan_sequence  # lazy

    return _plan_sequence(
        cfg_or_path,
        task_names,
        resume=resume,
        force=force,
        config_path=config_path,
    )


def run_sequence(
    cfg_or_path: Mapping[str, Any] | str | Path | RunContext,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    qc_override: bool = False,
    preflight_override: bool = False,
    preflight_override_reason: str | None = None,
    preflight_override_source: str | None = None,
    cancel_token: CancelToken | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    from scorpio_pipe.pipeline.engine import run_sequence as _run_sequence  # lazy

    return _run_sequence(
        cfg_or_path,
        task_names,
        resume=resume,
        force=force,
        qc_override=qc_override,
        preflight_override=preflight_override,
        preflight_override_reason=preflight_override_reason,
        preflight_override_source=preflight_override_source,
        cancel_token=cancel_token,
        config_path=config_path,
    )


def run_sequence_locked(
    cfg_or_path: Mapping[str, Any] | str | Path | RunContext,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    qc_override: bool = False,
    preflight_override: bool = False,
    preflight_override_reason: str | None = None,
    preflight_override_source: str | None = None,
    cancel_token: CancelToken | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run sequence under a workspace lock (single-writer guarantee)."""

    from scorpio_pipe.adapters.workspace_fs import FSWorkspace  # lazy
    from scorpio_pipe.paths import resolve_work_dir  # lazy

    # Best-effort: load cfg to derive run_root before entering the engine.
    cfg_for_lock: dict[str, Any]
    if isinstance(cfg_or_path, RunContext):
        cfg_for_lock = load_config_any(cfg_or_path.cfg_path)
    elif isinstance(cfg_or_path, dict):
        cfg_for_lock = dict(cfg_or_path)
    else:
        cfg_for_lock = load_config_any(cfg_or_path)

    run_root = Path(resolve_work_dir(cfg_for_lock))

    with FSWorkspace(run_root).lock():
        return run_sequence(
            cfg_or_path,
            task_names,
            resume=resume,
            force=force,
            qc_override=qc_override,
            preflight_override=preflight_override,
            preflight_override_reason=preflight_override_reason,
            preflight_override_source=preflight_override_source,
            cancel_token=cancel_token,
            config_path=config_path,
        )


def run_one(
    cfg_or_path: Mapping[str, Any] | str | Path | RunContext,
    task_name: str,
    *,
    resume: bool = True,
    force: bool = False,
    qc_override: bool = False,
) -> None:
    run_sequence(cfg_or_path, [task_name], resume=resume, force=force, qc_override=qc_override)


def done_dir_for_task(run_root: Path, task: str) -> Path:
    from scorpio_pipe.pipeline.engine import done_dir_for_task as _done_dir_for_task  # lazy

    return _done_dir_for_task(run_root, task)


def done_json_path_for_task(run_root: Path, task: str) -> Path:
    from scorpio_pipe.pipeline.engine import done_json_path_for_task as _done_json_path_for_task  # lazy

    return _done_json_path_for_task(run_root, task)


# -----------------------------------------------------------------------------
# Convenience helpers used by the GUI workflow
# -----------------------------------------------------------------------------


def run_lineid_prepare(ctx: RunContext) -> dict[str, Path]:
    """Run (or skip) ``lineid_prepare`` and return expected output paths."""

    run_sequence_locked(ctx, ["lineid_prepare"], resume=True, force=False, config_path=ctx.cfg_path)

    outdir = wavesol_dir(ctx.cfg)
    return {
        "template": (outdir / "manual_pairs_template.csv"),
        "auto": (outdir / "manual_pairs_auto.csv"),
        "report": (outdir / "lineid_report.txt"),
    }


def run_wavesolution(ctx: RunContext) -> dict[str, Any]:
    """Run (or skip) ``wavesolution`` and return per-task results."""

    return run_sequence_locked(ctx, ["wavesolution"], resume=True, force=False, config_path=ctx.cfg_path)


# -----------------------------------------------------------------------------
# GUI helper wrappers (keep stage imports LAZY)
# -----------------------------------------------------------------------------


def boxcar_mean2d_masked(*args: Any, **kwargs: Any):
    """Wrapper for the cosmic-mask smoothing helper used by the GUI."""

    from scorpio_pipe.stages.cosmics import _boxcar_mean2d_masked  # local import: lazy

    return _boxcar_mean2d_masked(*args, **kwargs)


def robust_polyfit_1d(*args: Any, **kwargs: Any):
    """Wrapper for robust 1D polynomial fit used in interactive tools."""

    from scorpio_pipe.stages.wavesolution import robust_polyfit_1d  # local import: lazy

    return robust_polyfit_1d(*args, **kwargs)


def robust_polyfit_2d_power(*args: Any, **kwargs: Any):
    from scorpio_pipe.stages.wavesolution import robust_polyfit_2d_power  # local import: lazy

    return robust_polyfit_2d_power(*args, **kwargs)


def robust_polyfit_2d_cheb(*args: Any, **kwargs: Any):
    from scorpio_pipe.stages.wavesolution import robust_polyfit_2d_cheb  # local import: lazy

    return robust_polyfit_2d_cheb(*args, **kwargs)


def polyval2d_power(*args: Any, **kwargs: Any):
    from scorpio_pipe.stages.wavesolution import polyval2d_power  # local import: lazy

    return polyval2d_power(*args, **kwargs)


def polyval2d_cheb(*args: Any, **kwargs: Any):
    from scorpio_pipe.stages.wavesolution import polyval2d_cheb  # local import: lazy

    return polyval2d_cheb(*args, **kwargs)


def prepare_lineid(*args: Any, **kwargs: Any):
    """Prepare line-identification products for the GUI LineID workflow."""

    from scorpio_pipe.stages.lineid_gui import prepare_lineid  # local import: lazy

    return prepare_lineid(*args, **kwargs)


@dataclass(frozen=True)
class StandardSelectionSummary:
    path: str
    role: str
    object_name: str
    association_status: str | None
    safe_for: str | None
    downgrade_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class StandardPrepareSummary:
    selected: StandardSelectionSummary | None
    max_severity: str = "OK"
    flags: tuple[dict[str, Any], ...] = ()
    qc_path: str | None = None
    trace_path: str | None = None
    spectrum_path: str | None = None


def run_standard_prepare(ctx: RunContext) -> dict[str, Any]:
    """Run (or skip) ``standard_prepare`` and return per-task results."""

    return run_sequence_locked(ctx, ["standard_prepare"], resume=True, force=False, config_path=ctx.cfg_path)



def _cfg_and_run_root(cfg_or_path: Mapping[str, Any] | str | Path | RunContext) -> tuple[dict[str, Any], Path]:
    if isinstance(cfg_or_path, RunContext):
        cfg = load_config_any(cfg_or_path.cfg_path)
    elif isinstance(cfg_or_path, dict):
        cfg = dict(cfg_or_path)
    else:
        cfg = load_config_any(cfg_or_path)
    run_root = Path(cfg.get("work_dir") or ".")
    return cfg, run_root


def read_standard_prepare_summary(cfg_or_path: Mapping[str, Any] | str | Path | RunContext) -> StandardPrepareSummary | None:
    """Load a UI-safe summary of ``standard_prepare`` outputs when available."""

    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    import json

    cfg, run_root = _cfg_and_run_root(cfg_or_path)

    try:
        qc_path = PRODUCT_CATALOG.best_path(cfg, "standard_qc_json", run_root=run_root)
    except Exception:
        return None
    if not qc_path.exists():
        return None
    payload = json.loads(qc_path.read_text(encoding="utf-8"))
    sel = payload.get("selected_standard") if isinstance(payload, dict) else None
    assoc = payload.get("association") if isinstance(payload, dict) else None
    selected = None
    if isinstance(sel, dict):
        selected = StandardSelectionSummary(
            path=str(sel.get("path") or ""),
            role=str(sel.get("role") or ""),
            object_name=str(sel.get("object_name") or ""),
            association_status=(str(assoc.get("status")) if isinstance(assoc, dict) and assoc.get("status") is not None else None),
            safe_for=(str(payload.get("safe_for")) if payload.get("safe_for") is not None else None),
            downgrade_reasons=tuple(str(x) for x in (assoc.get("downgrade_reasons") or [])) if isinstance(assoc, dict) else (),
        )
    try:
        trace_path = str(PRODUCT_CATALOG.best_path(cfg, "standard_trace_json", run_root=run_root))
    except Exception:
        trace_path = None
    try:
        spec_path = str(PRODUCT_CATALOG.best_path(cfg, "standard_1d_fits", run_root=run_root))
    except Exception:
        spec_path = None
    return StandardPrepareSummary(
        selected=selected,
        max_severity=str(payload.get("max_severity") or "OK"),
        flags=tuple(dict(x) for x in (payload.get("flags") or [])),
        qc_path=str(qc_path),
        trace_path=trace_path,
        spectrum_path=spec_path,
    )


@dataclass(frozen=True)
class CalibrationStatusSummary:
    status: str
    reference_context_id: str | None = None
    preview_path: str | None = None
    qc_path: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensfuncSummary:
    waveref: str | None
    preview_path: str | None = None
    qc_path: str | None = None
    model_path: str | None = None
    median_residual: float | None = None
    extinction_applied: bool | None = None
    airmass_provenance: str | None = None


@dataclass(frozen=True)
class TelluricSummary:
    coverage_fraction: float | None = None
    median_confidence: float | None = None
    preview_path: str | None = None
    qc_path: str | None = None


@dataclass(frozen=True)
class ScienceReportSummary:
    calibration: CalibrationStatusSummary | None = None
    n_detected_lines: int = 0
    report_md_path: str | None = None
    report_html_path: str | None = None
    line_preview_path: str | None = None
    warnings: tuple[str, ...] = ()
    calibration_uncertainty_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class ObserverStageSummary:
    status: str | None = None
    qc_path: str | None = None
    preview_path: str | None = None
    metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class ObserverDashboardSummary:
    trust_score: dict[str, Any]
    recovery_plan: dict[str, Any]
    setup_preset: dict[str, Any]
    dashboard_json_path: str | None = None
    dashboard_md_path: str | None = None
    dashboard_html_path: str | None = None


def run_line_measure(ctx: RunContext) -> dict[str, Any]:
    return run_sequence_locked(ctx, ["line_measure"], resume=True, force=False, config_path=ctx.cfg_path)


def run_science_report(ctx: RunContext) -> dict[str, Any]:
    return run_sequence_locked(ctx, ["science_report"], resume=True, force=False, config_path=ctx.cfg_path)


def run_recipe(ctx: RunContext, recipe_key: str) -> dict[str, Any]:
    from scorpio_pipe.pipeline.recipes import recipe_tasks

    return run_sequence_locked(ctx, list(recipe_tasks(recipe_key)), resume=True, force=False, config_path=ctx.cfg_path)


def list_recipes() -> tuple[dict[str, Any], ...]:
    from scorpio_pipe.pipeline.recipes import iter_recipes

    return tuple({"key": r.key, "label": r.label, "tasks": r.tasks, "description": r.description} for r in iter_recipes())


def _read_json(path: Path | None) -> dict[str, Any]:
    import json
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def read_fluxcal_summary(cfg_or_path: Mapping[str, Any] | str | Path | RunContext) -> CalibrationStatusSummary | None:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    cfg, run_root = _cfg_and_run_root(cfg_or_path)
    try:
        qc_path = PRODUCT_CATALOG.best_path(cfg, 'fluxcal_qc_json', run_root=run_root)
    except Exception:
        return None
    payload = _read_json(qc_path)
    if not payload:
        return None
    try:
        preview_path = str(PRODUCT_CATALOG.best_path(cfg, 'fluxcal_preview_png', run_root=run_root))
    except Exception:
        preview_path = None
    warnings = []
    for fl in payload.get('flags', []) or []:
        msg = str((fl or {}).get('message') or (fl or {}).get('code') or '').strip()
        if msg:
            warnings.append(msg)
    return CalibrationStatusSummary(
        status=str(payload.get('status') or payload.get('calibration_status') or 'UNKNOWN'),
        reference_context_id=(str(payload.get('reference_context_id')) if payload.get('reference_context_id') is not None else None),
        preview_path=preview_path,
        qc_path=str(qc_path),
        warnings=tuple(warnings),
    )


def read_sensfunc_summary(cfg_or_path: Mapping[str, Any] | str | Path | RunContext) -> SensfuncSummary | None:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    cfg, run_root = _cfg_and_run_root(cfg_or_path)
    try:
        qc_path = PRODUCT_CATALOG.best_path(cfg, 'sensfunc_qc_json', run_root=run_root)
    except Exception:
        return None
    payload = _read_json(qc_path)
    if not payload:
        return None
    try:
        preview_path = str(PRODUCT_CATALOG.best_path(cfg, 'sensfunc_preview_png', run_root=run_root))
    except Exception:
        preview_path = None
    try:
        model_path = str(PRODUCT_CATALOG.best_path(cfg, 'sensfunc_model_json', run_root=run_root))
    except Exception:
        model_path = None
    fit_meta = payload.get('fit_meta') or {}
    return SensfuncSummary(
        waveref=(str(payload.get('waveref')) if payload.get('waveref') is not None else None),
        preview_path=preview_path,
        qc_path=str(qc_path),
        model_path=model_path,
        median_residual=(float(fit_meta.get('mad_residual')) if isinstance(fit_meta, dict) and fit_meta.get('mad_residual') is not None else None),
        extinction_applied=(bool(payload.get('extinction_applied')) if payload.get('extinction_applied') is not None else None),
        airmass_provenance=(str(payload.get('airmass_provenance')) if payload.get('airmass_provenance') is not None else None),
    )


def read_telluric_summary(cfg_or_path: Mapping[str, Any] | str | Path | RunContext) -> TelluricSummary | None:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    cfg, run_root = _cfg_and_run_root(cfg_or_path)
    try:
        qc_path = PRODUCT_CATALOG.best_path(cfg, 'telluric_qc_json', run_root=run_root)
    except Exception:
        return None
    payload = _read_json(qc_path)
    if not payload:
        return None
    try:
        preview_path = str(PRODUCT_CATALOG.best_path(cfg, 'telluric_preview_png', run_root=run_root))
    except Exception:
        preview_path = None
    return TelluricSummary(
        coverage_fraction=(float(payload.get('coverage_fraction')) if payload.get('coverage_fraction') is not None else None),
        median_confidence=(float(payload.get('median_confidence')) if payload.get('median_confidence') is not None else None),
        preview_path=preview_path,
        qc_path=str(qc_path),
    )


def read_science_report_summary(cfg_or_path: Mapping[str, Any] | str | Path | RunContext) -> ScienceReportSummary | None:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    cfg, run_root = _cfg_and_run_root(cfg_or_path)
    calibration = read_fluxcal_summary(cfg)
    try:
        lines_path = PRODUCT_CATALOG.best_path(cfg, 'science_lines_json', run_root=run_root)
    except Exception:
        lines_path = None
    lines_payload = _read_json(lines_path)
    if calibration is None and not lines_payload:
        return None
    try:
        md_path = str(PRODUCT_CATALOG.best_path(cfg, 'science_report_md', run_root=run_root))
    except Exception:
        md_path = None
    try:
        html_path = str(PRODUCT_CATALOG.best_path(cfg, 'science_report_html', run_root=run_root))
    except Exception:
        html_path = None
    try:
        line_preview = str(PRODUCT_CATALOG.best_path(cfg, 'science_lines_preview_png', run_root=run_root))
    except Exception:
        line_preview = None
    warnings = tuple(str(w) for w in ((lines_payload.get('warnings') or []) if isinstance(lines_payload, dict) else []))
    n_detected = sum(1 for r in (lines_payload.get('lines') or []) if bool(r.get('detected'))) if isinstance(lines_payload, dict) else 0
    return ScienceReportSummary(calibration=calibration, n_detected_lines=int(n_detected), report_md_path=md_path, report_html_path=html_path, line_preview_path=line_preview, warnings=warnings, calibration_uncertainty_summary=(dict(lines_payload.get('calibration_uncertainty_summary') or {}) if isinstance(lines_payload, dict) else None))


def read_stage_json_summary(cfg_or_path: Mapping[str, Any] | str | Path | RunContext, product_key: str, *, preview_key: str | None = None) -> ObserverStageSummary | None:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    cfg, run_root = _cfg_and_run_root(cfg_or_path)
    try:
        qc_path = PRODUCT_CATALOG.best_path(cfg, product_key, run_root=run_root)
    except Exception:
        return None
    payload = _read_json(qc_path)
    if not payload:
        return None
    preview_path = None
    if preview_key:
        try:
            preview_path = str(PRODUCT_CATALOG.best_path(cfg, preview_key, run_root=run_root))
        except Exception:
            preview_path = None
    metrics = None
    if isinstance(payload.get('metrics'), dict):
        metrics = dict(payload.get('metrics') or {})
    elif isinstance(payload.get('diagnostics'), dict):
        metrics = dict(payload.get('diagnostics') or {})
    return ObserverStageSummary(status=str(payload.get('status') or payload.get('max_severity') or payload.get('stage') or ''), qc_path=str(qc_path), preview_path=preview_path, metrics=metrics)


def read_observer_dashboard_summary(cfg_or_path: Mapping[str, Any] | str | Path | RunContext) -> ObserverDashboardSummary | None:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    from scorpio_pipe.observer_dashboard import build_observer_dashboard as _build_observer_dashboard
    cfg, run_root = _cfg_and_run_root(cfg_or_path)
    payload = _build_observer_dashboard(cfg, run_root=run_root)
    try:
        json_path = str(PRODUCT_CATALOG.best_path(cfg, 'observer_dashboard_json', run_root=run_root))
    except Exception:
        json_path = None
    try:
        md_path = str(PRODUCT_CATALOG.best_path(cfg, 'observer_dashboard_md', run_root=run_root))
    except Exception:
        md_path = None
    try:
        html_path = str(PRODUCT_CATALOG.best_path(cfg, 'observer_dashboard_html', run_root=run_root))
    except Exception:
        html_path = None
    return ObserverDashboardSummary(
        trust_score=dict(payload.get('trust_score') or {}),
        recovery_plan=dict(payload.get('recovery_plan') or {}),
        setup_preset=dict(payload.get('setup_preset') or {}),
        dashboard_json_path=json_path,
        dashboard_md_path=md_path,
        dashboard_html_path=html_path,
    )


def build_observer_dashboard(cfg_or_path: Mapping[str, Any] | str | Path | RunContext) -> dict[str, Any]:
    from scorpio_pipe.observer_dashboard import build_observer_dashboard as _build_observer_dashboard

    legacy = {
        'standard_prepare': read_standard_prepare_summary(cfg_or_path),
        'sensfunc': read_sensfunc_summary(cfg_or_path),
        'telluric': read_telluric_summary(cfg_or_path),
        'fluxcal': read_fluxcal_summary(cfg_or_path),
        'science_report': read_science_report_summary(cfg_or_path),
    }
    payload = _build_observer_dashboard(cfg_or_path)
    legacy.update({
        'slit_sanity': read_stage_json_summary(cfg_or_path, 'slit_sanity_json'),
        'geometry': read_stage_json_summary(cfg_or_path, 'geometry_qc_json'),
        'wavesolution_stage': read_stage_json_summary(cfg_or_path, 'wave_done_json', preview_key='line_id_overview_png'),
        'sky_refine': read_stage_json_summary(cfg_or_path, 'qc_sky_refine_json'),
        'extraction': read_stage_json_summary(cfg_or_path, 'extraction_qc_json', preview_key='extraction_diagnostics_png'),
        'observer_dashboard': read_observer_dashboard_summary(cfg_or_path),
        'trust_score': dict(payload.get('trust_score') or {}),
        'recovery_plan': dict(payload.get('recovery_plan') or {}),
        'setup_preset': dict(payload.get('setup_preset') or {}),
    })
    return legacy


__all__ = [
    # Catalog
    "GuiStageDef",
    "StageInfo",
    "StageCatalog",
    "STAGE_CATALOG",
    "get_stage_catalog",
    "get_task",
    "get_gui_stage",
    "iter_gui_stages",
    "CANONICAL_TASKS",
    "STAGE_SPECS",
    "TASK_ENTRYPOINTS",
    "TASKS",
    "canonical_task_name",
    "validate_and_canonicalize_tasks",
    # Context / types
    "CancelToken",
    "StageContext",
    "StageResult",
    "PlanItem",
    "StageSpec",
    # Session store
    "load_session",
    "save_session",
    "session_path",
    "snapshot",
    "update_stage",
    # Engine wrappers
    "RunContext",
    "load_context",
    "plan_sequence",
    "run_sequence",
    "run_sequence_locked",
    "run_one",
    "done_dir_for_task",
    "done_json_path_for_task",
    "run_lineid_prepare",
    "run_wavesolution",
    "run_standard_prepare",
    "StandardSelectionSummary",
    "StandardPrepareSummary",
    "read_standard_prepare_summary",
    "CalibrationStatusSummary",
    "SensfuncSummary",
    "TelluricSummary",
    "ScienceReportSummary",
    "ObserverStageSummary",
    "ObserverDashboardSummary",
    "run_line_measure",
    "run_science_report",
    "run_recipe",
    "list_recipes",
    "read_fluxcal_summary",
    "read_sensfunc_summary",
    "read_telluric_summary",
    "read_science_report_summary",
    "read_stage_json_summary",
    "read_observer_dashboard_summary",
    "build_observer_dashboard",
    # GUI helpers
    "boxcar_mean2d_masked",
    "robust_polyfit_1d",
    "robust_polyfit_2d_power",
    "robust_polyfit_2d_cheb",
    "polyval2d_power",
    "polyval2d_cheb",
    "prepare_lineid",
]
