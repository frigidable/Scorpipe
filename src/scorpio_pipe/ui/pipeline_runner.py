"""Pipeline runner used by the GUI.

Goals:
- Keep the UI responsive (runner can be executed in a QThread).
- Provide deterministic skip/re-run logic:
  a stage is skipped only if required products exist AND the stage hash matches.
- Record stage state to ``work_dir/manifest/stage_state.json``.
"""

from __future__ import annotations

import inspect
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterable

from scorpio_pipe.config import load_config_any
from scorpio_pipe.products import task_is_complete
from scorpio_pipe.stage_state import compute_stage_hash, is_stage_up_to_date, record_stage_result
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.wavesol_paths import wavesol_dir
from scorpio_pipe.work_layout import ensure_work_layout

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    """Lightweight wrapper used by the GUI.

    The launcher window keeps a context object around and passes it back to
    ``run_sequence`` / per-stage helpers.
    """

    cfg_path: Path
    cfg: dict[str, Any]


def load_context(cfg_path: str | Path) -> RunContext:
    """Load config from disk and return a GUI-friendly context."""

    p = Path(cfg_path)
    cfg = load_config_any(p)
    return RunContext(cfg_path=p, cfg=cfg)


def run_lineid_prepare(ctx: RunContext) -> dict[str, Path]:
    """Compatibility wrapper for the GUI.

    Runs (or skips) ``lineid_prepare`` and returns the expected output paths.
    """

    run_sequence(ctx, ["lineid_prepare"], resume=True, force=False, config_path=ctx.cfg_path)

    # Even if the stage is skipped (Up-to-date), the GUI wants to know
    # where the artifacts are.
    outdir = wavesol_dir(ctx.cfg)
    return {
        "template": (outdir / "manual_pairs_template.csv"),
        "auto": (outdir / "manual_pairs_auto.csv"),
        "report": (outdir / "lineid_report.txt"),
    }


def run_wavesolution(ctx: RunContext) -> dict[str, Any]:
    """Compatibility wrapper for the GUI.

    Returns the per-task results dict from :func:`run_sequence` so the UI can show outputs
    without crashing when a stage returns ``None``.
    """

    return run_sequence(ctx, ["wavesolution"], resume=True, force=False, config_path=ctx.cfg_path)


@dataclass(frozen=True)
class PlanItem:
    task: str
    action: str  # 'run'|'skip'
    reason: str


class CancelToken:
    """Cooperative cancellation token.

    Current implementation checks cancellation *between* tasks.
    Stage implementations may optionally accept a ``cancel_token`` parameter for
    finer-grained cancellation.
    """

    def __init__(self) -> None:
        self._ev = Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()


# --- Task registry ---

TaskFn = Callable[..., Any]


def _call_maybe_with_cancel(fn: TaskFn, *, cancel_token: CancelToken | None = None, **kwargs: Any) -> Any:
    """Call a task function, passing only supported keyword arguments.

    The GUI runner tends to call tasks with a superset of kwargs
    (e.g. ``config_path``), but not every task needs/accepts them.
    Filtering prevents hard failures like:
    ``TypeError: ... got an unexpected keyword argument 'config_path'``.
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters

        # Cooperative cancellation is optional per-task.
        if cancel_token is not None and "cancel_token" in params:
            kwargs["cancel_token"] = cancel_token

        # If the function does NOT accept **kwargs, filter extras.
        accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not accepts_varkw:
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        # Ultra-defensive; never block execution due to introspection issues.
        pass
    return fn(**kwargs)


def _task_manifest(cfg: dict[str, Any], out_dir: Path, *, config_path: Path | None = None, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.manifest import write_manifest

    layout = ensure_work_layout(out_dir)
    # Canonical location (work/qc). Keep a legacy mirror for older tooling.
    p = write_manifest(out_path=layout.qc / "manifest.json", cfg=cfg, cfg_path=config_path)
    try:
        write_manifest(out_path=layout.report_legacy / "manifest.json", cfg=cfg, cfg_path=config_path)
    except Exception:
        # Legacy mirror should never break science.
        pass
    return p


def _task_superbias(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.calib import build_superbias

    _ = out_dir
    return build_superbias(cfg)


def _task_superflat(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.calib import build_superflat

    _ = out_dir
    return build_superflat(cfg)


def _task_flatfield(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.flatfield import run_flatfield

    return run_flatfield(cfg=cfg, out_dir=Path(out_dir) / "flatfield")


def _task_cosmics(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.cosmics import clean_cosmics

    return clean_cosmics(cfg, out_dir=Path(out_dir) / "cosmics")


def _task_superneon(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.superneon import build_superneon

    _ = out_dir
    res = build_superneon(cfg)
    return Path(res.superneon_fits)


def _task_lineid_prepare(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> dict[str, Path]:
    from scorpio_pipe.stages.lineid_auto_backup import prepare_lineid

    _ = out_dir
    _ = cancel_token
    return prepare_lineid(cfg)


def _task_wavesolution(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> dict[str, Any]:
    # build_wavesolution writes into the work_dir-derived wavesolution directory
    # (it does not need out_dir, but we keep the signature consistent for the runner).
    from scorpio_pipe.stages.wavesolution import build_wavesolution

    _ = out_dir
    _ = cancel_token  # currently unused inside wavesolution
    res = build_wavesolution(cfg)
    return res.as_dict()


def _task_qc_report(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.qc_report import build_qc_report

    _ = cancel_token
    layout = ensure_work_layout(out_dir)
    res = build_qc_report(cfg, out_dir=layout.qc)
    # Prefer canonical html output
    return Path(res.get("qc_html", layout.qc / "index.html"))


def _task_linearize(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.linearize import run_linearize

    res = run_linearize(cfg, out_dir=Path(out_dir) / "products" / "lin", cancel_token=cancel_token)
    return Path(res.get("products", {}).get("preview_fits", Path(out_dir) / "products" / "lin" / "lin_preview.fits"))


def _task_sky(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.sky_sub import run_sky_sub

    _ = cancel_token
    # v5.12+: canonical output lives in products/sky (legacy is mirrored).
    run_sky_sub(cfg, out_dir=Path(out_dir) / "products" / "sky")
    return Path(out_dir) / "products" / "sky" / "sky_sub_done.json"


def _task_stack2d(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.stack2d import run_stack2d

    _ = cancel_token
    wd = resolve_work_dir(cfg)
    per_dir = wd / "products" / "sky" / "per_exp"
    if not per_dir.exists():
        per_dir = wd / "sky" / "per_exp"
    inputs = sorted(per_dir.glob("*_sky_sub.fits"))
    if not inputs:
        raise FileNotFoundError(f"No per-exposure sky-subtracted frames found in {per_dir}")
    out_p = Path(out_dir) / "products" / "stack"
    res = run_stack2d(cfg, inputs=inputs, out_dir=out_p)
    return Path(res.get("stacked2d", out_p / "stacked2d.fits"))


def _task_extract1d(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.extract1d import run_extract1d

    _ = cancel_token
    res = run_extract1d(cfg, out_dir=Path(out_dir) / "products" / "spec")
    return Path(res.get("spec1d", Path(out_dir) / "products" / "spec" / "spec1d.fits"))


TASKS: dict[str, TaskFn] = {
    # canonical names
    "manifest": _task_manifest,
    "superbias": _task_superbias,
    "superflat": _task_superflat,
    "flatfield": _task_flatfield,
    "cosmics": _task_cosmics,
    "superneon": _task_superneon,
    "lineid_prepare": _task_lineid_prepare,
    "wavesolution": _task_wavesolution,
    "linearize": _task_linearize,
    "sky": _task_sky,
    "stack2d": _task_stack2d,
    "extract1d": _task_extract1d,
    "qc_report": _task_qc_report,
    # aliases (backward compatibility)
    "wavesol": _task_wavesolution,
    "wavesolution2d": _task_wavesolution,
    "sky_sub": _task_sky,
    "stack": _task_stack2d,
    "qc": _task_qc_report,
}


def canonical_task_name(name: str) -> str:
    n = (name or "").strip().lower()
    # tolerate old naming
    aliases = {
        "wavesol": "wavesolution",
        "wavesol2d": "wavesolution",
        "wavelength_solution": "wavesolution",
        "lineid": "lineid_prepare",
        "qc": "qc_report",
    }
    return aliases.get(n, n)


def _stage_cfg_for_hash(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    # map task -> config section
    sec_map = {
        "manifest": {},
        "superbias": cfg.get("calib", {}),
        "superflat": cfg.get("calib", {}),
        "flatfield": cfg.get("flatfield", {}),
        "cosmics": cfg.get("cosmics", {}),
        # superneon depends on a few config sections; keep only numeric-ish values anyway
        "superneon": {
            "calib": cfg.get("calib", {}),
            "wavesol": cfg.get("wavesol", {}),
            "superneon": cfg.get("superneon", {}),
        },
        "lineid_prepare": {
            "wavesol": cfg.get("wavesol", {}),
            "superneon": cfg.get("superneon", {}),
        },
        "wavesolution": cfg.get("wavesol", {}),
        "qc_report": cfg.get("qc", {}),
        "linearize": cfg.get("linearize", {}),
        "sky": cfg.get("sky", {}),
        "sky_sub": cfg.get("sky", {}),
        "stack2d": cfg.get("stack2d", {}),
        "stack": cfg.get("stack2d", {}),
        "extract1d": cfg.get("extract1d", {}),
    }
    sec = sec_map.get(task, cfg.get(task, {}))
    return sec if isinstance(sec, dict) else {}


def _input_paths_for_hash(cfg: dict[str, Any], task: str, out_dir: Path) -> list[Path]:
    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}

    def _frame_list(key: str) -> list[Path]:
        v = frames.get(key, []) if isinstance(frames, dict) else []
        return [Path(str(x)) for x in v] if isinstance(v, list) else []

    wd = Path(out_dir)
    layout = ensure_work_layout(wd)
    wsol = wavesol_dir(cfg)

    def _first_existing(*cands: Path) -> Path:
        for c in cands:
            try:
                if c and Path(c).exists():
                    return Path(c)
            except Exception:
                pass
        return Path(cands[0])

    # Key calibration paths (prefer config override, then canonical, then legacy).
    calib_cfg = cfg.get("calib", {}) if isinstance(cfg.get("calib"), dict) else {}

    def _resolve_cfg_path(v: Any) -> Path | None:
        if not v:
            return None
        p = Path(str(v))
        if p.is_absolute():
            return p
        # relative to work_dir by convention
        return (wd / p).resolve()

    sb_cfg = _resolve_cfg_path(calib_cfg.get("superbias_path"))
    sf_cfg = _resolve_cfg_path(calib_cfg.get("superflat_path"))
    sb = sb_cfg or _first_existing(layout.calibs / "superbias.fits", layout.calib_legacy / "superbias.fits")
    sf = sf_cfg or _first_existing(layout.calibs / "superflat.fits", layout.calib_legacy / "superflat.fits")

    if task == "manifest":
        paths: list[Path] = []
        for _k, v in frames.items() if isinstance(frames, dict) else []:
            if isinstance(v, list):
                paths.extend(Path(str(x)) for x in v)
        return paths

    if task == "superbias":
        return _frame_list("bias")
    if task == "superflat":
        return _frame_list("flat")
    if task == "flatfield":
        return _frame_list("obj") + [sf, sb]
    if task == "cosmics":
        return _frame_list("obj") + [sb]
    if task == "superneon":
        return _frame_list("neon") + ([sb] if sb else [])
    if task == "lineid_prepare":
        return [wsol / "superneon.fits"]
    if task == "wavesolution":
        return [wsol / "peaks_candidates.csv", wsol / "hand_pairs.txt"]
    if task == "linearize":
        return _frame_list("obj") + [wsol / "lambda_map.fits"]
    if task in ("sky", "sky_sub"):
        p = layout.products / "lin" / "lin_preview.fits"
        legacy = wd / "lin" / "obj_sum_lin.fits"
        return [_first_existing(p, legacy)]
    if task in ("stack2d", "stack"):
        per_exp = layout.products / "sky" / "per_exp"
        inputs = [layout.products / "sky" / "sky_sub_done.json"]
        # include per-exp FITS signatures if directory exists
        if per_exp.is_dir():
            inputs.extend(sorted(per_exp.glob("*.fits")))
        else:
            inputs.append(per_exp)
        return inputs
    if task == "extract1d":
        return [layout.products / "stack" / "stacked2d.fits"]
    if task in ("qc_report", "qc"):
        # QC report should refresh when the stage state changes (proxy for upstream outputs).
        return [wd / "manifest" / "stage_state.json"]

    return []


def plan_sequence(
    cfg_or_path: dict[str, Any] | str | Path,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    config_path: Path | None = None,
) -> list[PlanItem]:
    cfg = load_config_any(cfg_or_path) if not isinstance(cfg_or_path, dict) else cfg_or_path
    work_dir = resolve_work_dir(cfg)

    plan: list[PlanItem] = []
    for raw in task_names:
        t = canonical_task_name(raw)
        if t not in TASKS:
            plan.append(PlanItem(task=t, action="run", reason="Unknown task (will fail)"))
            continue

        complete = task_is_complete(cfg, t)
        expected_hash = compute_stage_hash(
            stage=t,
            stage_cfg=_stage_cfg_for_hash(cfg, t),
            input_paths=_input_paths_for_hash(cfg, t, work_dir),
        )
        up_to_date = complete and is_stage_up_to_date(work_dir, t, expected_hash)
        if resume and (not force) and up_to_date:
            plan.append(PlanItem(task=t, action="skip", reason="Up-to-date"))
        else:
            if not complete:
                reason = "Missing products"
            elif not is_stage_up_to_date(work_dir, t, expected_hash):
                reason = "Dirty (params/inputs changed)"
            else:
                reason = "Forced"
            plan.append(PlanItem(task=t, action="run", reason=reason))
    return plan


def run_sequence(
    cfg_or_path: dict[str, Any] | str | Path | RunContext,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    cancel_token: CancelToken | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if isinstance(cfg_or_path, RunContext):
        cfg = cfg_or_path.cfg
        if config_path is None:
            config_path = cfg_or_path.cfg_path
    else:
        cfg = load_config_any(cfg_or_path) if not isinstance(cfg_or_path, dict) else cfg_or_path

    work_dir = resolve_work_dir(cfg)
    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    plan = plan_sequence(cfg, task_names, resume=resume, force=force, config_path=config_path)
    for it in plan:
        if cancel_token is not None and cancel_token.cancelled:
            log.warning("Cancelled before task %s", it.task)
            record_stage_result(out_dir, it.task, status="cancelled", stage_hash=None, message="Cancelled", trace=None)
            break

        t = it.task
        if it.action == "skip":
            log.info("Skip %s (%s)", t, it.reason)
            results[t] = None
            try:
                from scorpio_pipe.qc.metrics_store import update_after_stage

                update_after_stage(cfg, stage=t, status="skipped", stage_hash=None)
            except Exception:
                pass
            continue

        fn = TASKS.get(t)
        if fn is None:
            raise KeyError(f"Unknown task: {t}")

        stage_hash = compute_stage_hash(
            stage=t,
            stage_cfg=_stage_cfg_for_hash(cfg, t),
            input_paths=_input_paths_for_hash(cfg, t, out_dir),
        )

        log.info("Run %s...", t)
        try:
            res = _call_maybe_with_cancel(fn, cfg=cfg, out_dir=out_dir, config_path=config_path, cancel_token=cancel_token)
            results[t] = res
            record_stage_result(out_dir, t, status="ok", stage_hash=stage_hash, message=None, trace=None)
            try:
                from scorpio_pipe.qc.metrics_store import mirror_qc_to_products, update_after_stage

                stage_metrics = None
                if isinstance(res, dict):
                    stage_metrics = {}
                    for k, v in res.items():
                        # Store numeric scalars (+ a few small categorical values), but skip file paths.
                        if k.endswith(("_fits", "_png", "_csv", "_json", "_txt", "_dir")):
                            continue
                        if isinstance(v, (int, float, bool)):
                            stage_metrics[k] = v
                        elif isinstance(v, str) and k in {"model2d_kind"}:
                            stage_metrics[k] = v
                update_after_stage(cfg, stage=t, status="ok", stage_hash=stage_hash, stage_metrics=stage_metrics)
                if t in ("manifest", "qc_report"):
                    mirror_qc_to_products(out_dir)
            except Exception:
                pass
        except Exception as e:
            tb = traceback.format_exc()
            log.error("Task %s failed: %s", t, e, exc_info=True)
            record_stage_result(out_dir, t, status="failed", stage_hash=stage_hash, message=str(e), trace=tb)
            try:
                from scorpio_pipe.qc.metrics_store import update_after_stage

                update_after_stage(cfg, stage=t, status="failed", stage_hash=stage_hash, stage_metrics={"error": str(e)})
            except Exception:
                pass
            raise

    return results


def run_one(cfg_or_path: dict[str, Any] | str | Path, task_name: str, *, resume: bool = True, force: bool = False) -> None:
    run_sequence(cfg_or_path, [task_name], resume=resume, force=force)
