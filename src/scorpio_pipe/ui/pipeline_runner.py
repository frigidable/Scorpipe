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
from scorpio_pipe.products import products_for_task, task_is_complete
from scorpio_pipe.stage_state import compute_stage_hash, is_stage_up_to_date, load_stage_state, record_stage_result
from scorpio_pipe.wavesol_paths import resolve_work_dir

log = logging.getLogger(__name__)


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
    try:
        sig = inspect.signature(fn)
        if cancel_token is not None and "cancel_token" in sig.parameters:
            kwargs["cancel_token"] = cancel_token
    except Exception:
        # ultra-defensive; never block execution due to introspection issues
        pass
    return fn(**kwargs)


def _task_manifest(cfg: dict[str, Any], out_dir: Path, *, config_path: Path | None = None, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.manifest import write_manifest

    return write_manifest(out_path=out_dir / "report" / "manifest.json", cfg=cfg, cfg_path=config_path)


def _task_superbias(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.calib import build_superbias

    return build_superbias(cfg, out_dir)


def _task_superflat(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    # One source of truth: stages.flatfield.build_superflat
    from scorpio_pipe.stages.flatfield import build_superflat

    return build_superflat(cfg=cfg, out_dir=out_dir)


def _task_flatfield(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.flatfield import run_flatfield

    return run_flatfield(cfg=cfg, out_dir=out_dir)


def _task_cosmics(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.cosmics import clean_cosmics

    return clean_cosmics(cfg, out_dir)


def _task_superneon(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.superneon import build_superneon

    return build_superneon(cfg, out_dir)


def _task_lineid_prepare(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> None:
    from scorpio_pipe.stages.lineid_auto_backup import prepare_lineid

    prepare_lineid(cfg, out_dir)


def _task_wavesolution(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> None:
    # build_wavesolution writes into the work_dir-derived wavesolution directory
    # (it does not need out_dir, but we keep the signature consistent for the runner).
    from scorpio_pipe.stages.wavesolution import build_wavesolution

    _ = cancel_token  # currently unused inside wavesolution
    build_wavesolution(cfg)


def _task_linearize(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.linearize import run_linearize

    res = run_linearize(cfg, out_dir=out_dir, cancel_token=cancel_token)
    # payload contains products.sum_fits
    return Path(res.get("products", {}).get("sum_fits", out_dir / "obj_sum_lin.fits"))


def _task_sky(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.sky_sub import run_sky_sub

    _ = cancel_token
    res = run_sky_sub(cfg, out_dir=out_dir)
    return Path(res.get("sky_sub", out_dir / "obj_sky_sub.fits"))


def _task_extract1d(cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None) -> Path:
    from scorpio_pipe.stages.extract1d import extract_1d

    _ = cancel_token
    res = extract_1d(cfg, out_dir=out_dir)
    return Path(res.get("spectrum_1d", out_dir / "spec1d.fits"))


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
    "extract1d": _task_extract1d,
    # aliases (backward compatibility)
    "wavesol": _task_wavesolution,
    "wavesolution2d": _task_wavesolution,
    "sky_sub": _task_sky,
}


def canonical_task_name(name: str) -> str:
    n = (name or "").strip().lower()
    # tolerate old naming
    aliases = {
        "wavesol": "wavesolution",
        "wavesol2d": "wavesolution",
        "wavelength_solution": "wavesolution",
        "lineid": "lineid_prepare",
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
        "superneon": cfg.get("calib", {}),
        "lineid_prepare": cfg.get("wavesol", {}),
        "wavesolution": cfg.get("wavesol", {}),
        "linearize": cfg.get("linearize", {}),
        "sky": cfg.get("sky", {}),
        "sky_sub": cfg.get("sky", {}),
        "extract1d": cfg.get("extract1d", {}),
    }
    sec = sec_map.get(task, cfg.get(task, {}))
    return sec if isinstance(sec, dict) else {}


def _input_paths_for_hash(cfg: dict[str, Any], task: str, out_dir: Path) -> list[Path]:
    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    def _frame_list(key: str) -> list[Path]:
        v = frames.get(key, []) if isinstance(frames, dict) else []
        return [Path(str(x)) for x in v] if isinstance(v, list) else []

    # products from previous stages
    prod_by_key = {p.key: p.path for p in products_for_task(cfg, task)}

    if task == "manifest":
        paths: list[Path] = []
        for k, v in frames.items() if isinstance(frames, dict) else []:
            if isinstance(v, list):
                paths.extend(Path(str(x)) for x in v)
        return paths

    if task == "superbias":
        return _frame_list("bias")
    if task == "superflat":
        return _frame_list("flat") + [prod_by_key.get("superbias", out_dir / "calib" / "superbias.fits")]
    if task == "flatfield":
        return _frame_list("obj") + [out_dir / "calib" / "superflat.fits"]
    if task == "cosmics":
        return _frame_list("obj") + [out_dir / "calib" / "superbias.fits", out_dir / "calib" / "superflat.fits"]
    if task == "superneon":
        return _frame_list("neon")
    if task == "lineid_prepare":
        return [out_dir / "calib" / "superneon.fits"]
    if task == "wavesolution":
        return [out_dir / "wavesol" / "peaks_candidates.json", out_dir / "wavesol" / "hand_pairs.yaml"]
    if task == "linearize":
        return _frame_list("obj") + [out_dir / "wavesol" / "lambda_map.fits"]
    if task in ("sky", "sky_sub"):
        return [out_dir / "lin" / "obj_sum_lin.fits"]
    if task == "extract1d":
        return [out_dir / "sky" / "obj_sky_sub.fits"]

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
    cfg_or_path: dict[str, Any] | str | Path,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    cancel_token: CancelToken | None = None,
    config_path: Path | None = None,
) -> None:
    cfg = load_config_any(cfg_or_path) if not isinstance(cfg_or_path, dict) else cfg_or_path

    work_dir = resolve_work_dir(cfg)
    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_sequence(cfg, task_names, resume=resume, force=force, config_path=config_path)
    for it in plan:
        if cancel_token is not None and cancel_token.cancelled:
            log.warning("Cancelled before task %s", it.task)
            record_stage_result(out_dir, it.task, status="cancelled", stage_hash=None, message="Cancelled", trace=None)
            break

        t = it.task
        if it.action == "skip":
            log.info("Skip %s (%s)", t, it.reason)
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
            _call_maybe_with_cancel(fn, cfg=cfg, out_dir=out_dir, config_path=config_path, cancel_token=cancel_token)
            record_stage_result(out_dir, t, status="ok", stage_hash=stage_hash, message=None, trace=None)
        except Exception as e:
            tb = traceback.format_exc()
            log.error("Task %s failed: %s", t, e, exc_info=True)
            record_stage_result(out_dir, t, status="failed", stage_hash=stage_hash, message=str(e), trace=tb)
            raise


def run_one(cfg_or_path: dict[str, Any] | str | Path, task_name: str, *, resume: bool = True, force: bool = False) -> None:
    run_sequence(cfg_or_path, [task_name], resume=resume, force=force)
