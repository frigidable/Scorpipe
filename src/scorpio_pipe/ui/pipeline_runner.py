"""Pipeline runner used by the GUI.

Goals:
- Keep the UI responsive (runner can be executed in a QThread).
- Provide deterministic skip/re-run logic:
  a stage is skipped only if required products exist AND the stage hash matches.
- Record stage state to ``work_dir/manifest/stage_state.json``.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterable

from scorpio_pipe.config import load_config_any
from scorpio_pipe.products import task_is_complete
from scorpio_pipe.stage_state import (
    compute_stage_hash,
    is_stage_up_to_date,
    record_stage_result,
)
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.wavesol_paths import wavesol_dir
from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.workspace_paths import stage_dir, per_exp_dir, resolve_input_path
from scorpio_pipe.stage_registry import REGISTRY

# P0 provenance + boundary contract enforcement
from scorpio_pipe.boundary_contract import (
    ProductContractError,
    validate_mef_product,
    validate_spec1d_product,
)
from scorpio_pipe.io.done_json import write_done_json
from scorpio_pipe.prov_capture import (
    compute_input_hashes,
    ensure_run_provenance,
    load_hash_cache,
)
from scorpio_pipe.run_passport import ensure_run_passport

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

    run_sequence(
        ctx, ["lineid_prepare"], resume=True, force=False, config_path=ctx.cfg_path
    )

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

    return run_sequence(
        ctx, ["wavesolution"], resume=True, force=False, config_path=ctx.cfg_path
    )


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


def _call_maybe_with_cancel(
    fn: TaskFn, *, cancel_token: CancelToken | None = None, **kwargs: Any
) -> Any:
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
        accepts_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if not accepts_varkw:
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        # Ultra-defensive; never block execution due to introspection issues.
        pass
    return fn(**kwargs)


def _task_manifest(
    cfg: dict[str, Any],
    out_dir: Path,
    *,
    config_path: Path | None = None,
    cancel_token: CancelToken | None = None,
) -> Path:
    from scorpio_pipe.manifest import write_manifest

    _ = ensure_work_layout(out_dir)

    # Canonical location (run_root/manifest/manifest.json).
    p = write_manifest(
        out_path=Path(out_dir) / "manifest" / "manifest.json",
        cfg=cfg,
        cfg_path=config_path,
    )

    # Legacy mirrors: write only if legacy roots already exist (do not create them).
    try:
        qc_legacy = Path(out_dir) / "qc"
        if qc_legacy.exists():
            write_manifest(
                out_path=qc_legacy / "manifest.json",
                cfg=cfg,
                cfg_path=config_path,
            )
    except Exception:
        pass

    try:
        rep_legacy = Path(out_dir) / "report"
        if rep_legacy.exists():
            write_manifest(
                out_path=rep_legacy / "manifest.json",
                cfg=cfg,
                cfg_path=config_path,
            )
    except Exception:
        pass

    return p


def _task_superbias(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    from scorpio_pipe.stages.calib import build_superbias

    _ = out_dir
    return build_superbias(cfg)


def _task_superflat(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    from scorpio_pipe.stages.calib import build_superflat

    _ = out_dir
    return build_superflat(cfg)


def _task_flatfield(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    from scorpio_pipe.stages.flatfield import run_flatfield

    return run_flatfield(cfg=cfg, out_dir=stage_dir(out_dir, "flatfield"))


def _task_cosmics(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    from scorpio_pipe.stages.cosmics import clean_cosmics

    return clean_cosmics(cfg, out_dir=stage_dir(out_dir, "cosmics"))


def _task_superneon(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    from scorpio_pipe.stages.superneon import build_superneon

    _ = out_dir
    res = build_superneon(cfg)
    return Path(res.superneon_fits)


def _task_lineid_prepare(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> dict[str, Path]:
    from scorpio_pipe.stages.lineid_auto_backup import prepare_lineid

    _ = out_dir
    _ = cancel_token
    return prepare_lineid(cfg)


def _task_wavesolution(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> dict[str, Any]:
    # build_wavesolution writes into the work_dir-derived wavesolution directory
    # (it does not need out_dir, but we keep the signature consistent for the runner).
    from scorpio_pipe.stages.wavesolution import build_wavesolution

    _ = out_dir
    _ = cancel_token  # currently unused inside wavesolution
    res = build_wavesolution(cfg)
    return res.as_dict()


def _task_qc_report(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    from scorpio_pipe.qc_report import build_qc_report

    _ = cancel_token
    _ = ensure_work_layout(out_dir)
    # Canonical: manifest/qc_report.json + work_dir/index.html
    res = build_qc_report(cfg)
    # build_qc_report historically returned Path; keep compatibility if it returns a dict
    if isinstance(res, dict):
        return Path(res.get("qc_html", Path(out_dir) / "index.html"))
    return Path(res)


def _task_navigator(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    """Build static HTML navigator (ui/navigator).

    This is lightweight and safe to run at any time.
    """

    _ = cancel_token
    _ = ensure_work_layout(out_dir)
    from scorpio_pipe.navigator import build_navigator

    return build_navigator(out_dir)


def _task_linearize(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    from scorpio_pipe.stages.linearize import run_linearize

    out_p = stage_dir(out_dir, "linearize")
    res = run_linearize(cfg, out_dir=out_p, cancel_token=cancel_token)
    return Path(res.get("products", {}).get("preview_fits", out_p / "lin_preview.fits"))


def _task_sky(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    from scorpio_pipe.stages.sky_sub import run_sky_sub

    _ = cancel_token
    out_p = stage_dir(out_dir, "sky")
    run_sky_sub(cfg, out_dir=out_p)
    return out_p / "sky_done.json"


def _task_stack2d(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    """Stack rectified, sky-subtracted frames.

    Contract (P1-E / v5.40.4)
    -------------------------
    Stack2D consumes **only** rectified, sky-subtracted frames produced by
    Linearization:

      10_linearize/<stem>_skysub.fits

    No hidden fallbacks to other stage folders are allowed.
    """

    from scorpio_pipe.stages.stack2d import run_stack2d
    from scorpio_pipe.product_naming import sky_sub_fits_name
    from scorpio_pipe.workspace_paths import extract_stem_short

    _ = cancel_token
    wd = resolve_work_dir(cfg)

    lin_stage = stage_dir(wd, "linearize")

    # Prefer an explicit science exposure list from config.
    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    obj_list = frames.get("obj") if isinstance(frames.get("obj"), list) else []
    stems = [
        extract_stem_short(x)
        for x in obj_list
        if isinstance(x, str) and x.strip()
    ]

    def _find_rectified_skysub(stem: str) -> Path:
        # Hard contract: only 10_linearize/<stem>_skysub.fits
        return lin_stage / sky_sub_fits_name(stem)

    if stems:
        inputs: list[Path] = []
        missing: list[str] = []
        for stem in stems:
            p = _find_rectified_skysub(stem)
            if p.exists():
                inputs.append(p)
            else:
                missing.append(stem)
        if missing:
            raise FileNotFoundError(
                "Stack2D: missing rectified sky-subtracted inputs for stems: "
                + ", ".join(missing)
                + ". Expected 10_linearize/<stem>_skysub.fits. Run Linearization first."
            )
    else:
        # Scan only the Linearization stage root (non-recursive).
        inputs = sorted(p for p in lin_stage.glob("*_skysub.fits") if p.is_file())

        if not inputs:
            raise FileNotFoundError(
                f"No rectified sky-subtracted frames found in {lin_stage}. Expected 10_linearize/*_skysub.fits. Run Linearization first."
            )

    out_p = stage_dir(out_dir, "stack2d")
    res = run_stack2d(cfg, inputs=inputs, out_dir=out_p)
    # Canonical new name is stack2d.fits; keep legacy stacked2d.fits as well.
    return Path(res.get("stack2d", res.get("stacked2d", out_p / "stack2d.fits")))


def _task_extract1d(
    cfg: dict[str, Any], out_dir: Path, *, cancel_token: CancelToken | None = None
) -> Path:
    from scorpio_pipe.stages.extract1d import run_extract1d

    _ = cancel_token
    out_p = stage_dir(out_dir, "extract1d")
    res = run_extract1d(cfg, out_dir=out_p)
    return Path(res.get("spec1d", out_p / "spec1d.fits"))


TASKS: dict[str, TaskFn] = {
    # canonical names
    "manifest": _task_manifest,
    "superbias": _task_superbias,
    # P0-C1 naming (alias to superbias task)
    "bias_combine": _task_superbias,
    "masterbias": _task_superbias,
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
    "navigator": _task_navigator,
    # aliases (backward compatibility)
    "wavesol": _task_wavesolution,
    "wavesolution2d": _task_wavesolution,
    "sky_sub": _task_sky,
    "stack": _task_stack2d,
    "qc": _task_qc_report,
    "nav": _task_navigator,
}


def _done_dir_for_task(run_root: Path, task: str) -> Path:
    """Return directory where task-scoped done.json should live.

    We keep done.json in a task-specific folder to avoid collisions.
    For tasks that conceptually belong to a stage folder (e.g. superbias -> biascorr),
    we reuse the stage directory.
    """

    if task == "manifest":
        return Path(run_root) / "manifest"
    if task == "navigator":
        return Path(run_root) / "ui" / "navigator"
    if task == "qc_report":
        return Path(run_root) / "qc"
    if task == "superbias":
        return stage_dir(run_root, "biascorr")
    if task == "superflat":
        return stage_dir(run_root, "flatfield")

    # Most tasks map 1:1 to a stage key (stage registry).
    try:
        REGISTRY.get(task)
        return stage_dir(run_root, task)
    except Exception:
        pass

    # Fallback: a folder named after the task.
    return Path(run_root) / task


def _collect_output_paths(res: Any) -> list[Path]:
    """Best-effort flatten of stage return value into a list of file paths."""

    out: list[Path] = []

    def _add(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, (str, Path)):
            p = Path(str(v))
            out.append(p)
            return
        if isinstance(v, dict):
            for vv in v.values():
                _add(vv)
            return
        if isinstance(v, (list, tuple, set)):
            for vv in v:
                _add(vv)

    _add(res)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def _validate_task_products(task: str, run_root: Path) -> None:
    """Validate boundary contract for task outputs (hard-fail).

    We validate only canonical *product* tasks that output a standard 2D MEF
    or 1D spectrum product.
    """

    if task == "sky":
        st = stage_dir(run_root, "sky")
        for p in sorted(st.glob("*_skysub_raw.fits")):
            validate_mef_product(p, stage=task)
        for p in sorted(st.glob("*_skymodel_raw.fits")):
            validate_mef_product(p, stage=task)
        return

    if task == "linearize":
        st = stage_dir(run_root, "linearize")
        for pat in ("*_rectified.fits", "*_skysub.fits", "*_skymodel.fits"):
            for p in sorted(st.glob(pat)):
                validate_mef_product(p, stage=task)
        # Optional preview
        prev = st / "lin_preview.fits"
        if prev.is_file():
            validate_mef_product(prev, stage=task)
        return

    if task == "stack2d":
        st = stage_dir(run_root, "stack2d")
        p = st / "stack2d.fits"
        if p.is_file():
            validate_mef_product(p, stage=task)
        return

    if task == "extract1d":
        st = stage_dir(run_root, "extract1d")
        # Most common output
        for p in sorted(st.glob("spec1d*.fits")):
            validate_spec1d_product(p, stage=task)
        return



def canonical_task_name(name: str) -> str:
    n = (name or "").strip().lower()
    # tolerate old naming
    aliases = {
        # P0-C1 naming
        "bias_combine": "superbias",
        "masterbias": "superbias",
        "wavesol": "wavesolution",
        "wavesol2d": "wavesolution",
        "wavelength_solution": "wavesolution",
        "lineid": "lineid_prepare",
        "qc": "qc_report",
        "nav": "navigator",
        "navigator": "navigator",
    }
    return aliases.get(n, n)


def _task_label(task: str) -> str:
    """Human-friendly label for logs.

    If the task maps to a stage, returns e.g. ``10 Sky Subtraction [sky]``.
    Otherwise returns the raw task name.
    """

    t = (task or "").strip().lower()
    try:
        st = REGISTRY.get(t)
        return f"{st.title} [{t}]"
    except Exception:
        return t


def _stage_cfg_for_hash(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    # map task -> config section
    sec_map = {
        "manifest": {},
        "superbias": cfg.get("calib", {}),
        "bias_combine": cfg.get("calib", {}),
        "masterbias": cfg.get("calib", {}),
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

    # Resolve frame paths the same way the stages do (config_dir first, then data_dir).
    cfg_dir = Path(str(cfg.get("config_dir", "."))).expanduser().resolve()
    data_dir_raw = str(cfg.get("data_dir", "") or "").strip()
    data_dir = Path(data_dir_raw).expanduser() if data_dir_raw else None
    if data_dir is not None:
        try:
            data_dir = data_dir.resolve()
        except Exception:
            pass

    def _resolve_frame_path(x: Any) -> Path:
        p = Path(str(x)).expanduser()
        if p.is_absolute():
            return p
        cand = (cfg_dir / p)
        if cand.exists():
            return cand.resolve()
        if data_dir is not None:
            cand2 = (data_dir / p)
            if cand2.exists():
                return cand2.resolve()
        # Keep deterministic even if missing.
        return cand.resolve()

    def _frame_list(key: str) -> list[Path]:
        v = frames.get(key, []) if isinstance(frames, dict) else []
        return [_resolve_frame_path(x) for x in v] if isinstance(v, list) else []

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
    sb = sb_cfg or resolve_input_path(
        "superbias_fits", wd, "superbias", relpath="superbias.fits"
    )
    sf = sf_cfg or resolve_input_path(
        "superflat_fits", wd, "superflat", relpath="superflat.fits"
    )

    if task == "manifest":
        paths: list[Path] = []
        for _k, v in frames.items() if isinstance(frames, dict) else []:
            if isinstance(v, list):
                paths.extend(_resolve_frame_path(x) for x in v)
        return paths

    if task == "superbias":
        return _frame_list("bias")
    if task == "superflat":
        return _frame_list("flat")
    if task == "flatfield":
        # Flat-fielding prefers cosmics-cleaned frames (if present) to avoid
        # having the flat correction interpolate across cosmic hits.
        paths: list[Path] = []
        obj = _frame_list("obj")
        cos_clean_dir = stage_dir(wd, "cosmics") / "obj" / "clean"
        for p in obj:
            stem = p.stem
            cand = cos_clean_dir / f"{stem}_clean.fits"
            paths.append(cand if cand.exists() else p)
        paths.extend([sf, sb])
        return paths
    if task == "cosmics":
        return _frame_list("obj") + [sb]
    if task == "superneon":
        return _frame_list("neon") + ([sb] if sb else [])
    if task == "lineid_prepare":
        return [wsol / "superneon.fits"]
    if task == "wavesolution":
        return [wsol / "peaks_candidates.csv", wsol / "hand_pairs.txt"]
    if task == "linearize":
        lin_inputs: list[Path] = []
        lin_inputs.extend(_frame_list("obj"))
        lin_inputs.append(wsol / "lambda_map.fits")
        sky_stage = stage_dir(wd, "sky")
        if sky_stage.exists():
            raw_products = sorted(p for p in sky_stage.glob("*_skysub_raw.fits") if p.is_file())
            if raw_products:
                lin_inputs.extend(raw_products)
            else:
                lin_inputs.append(sky_stage / "sky_done.json")
        else:
            lin_inputs.append(sky_stage)
        return lin_inputs
    if task in ("sky", "sky_sub"):
        # Sky runs in RAW geometry; it depends on the wavelength solution and
        # the *actual* per-exposure inputs it will consume.
        sky_inputs: list[Path] = []
        obj = _frame_list("obj")
        cos_clean_dir = stage_dir(wd, "cosmics") / "obj" / "clean"
        flat_dir = stage_dir(wd, "flatfield") / "obj"

        def _best_obj_input(raw_p: Path) -> Path:
            stem = raw_p.stem
            # Prefer flat-fielded frame if present
            cand_flat = flat_dir / f"{stem}_flat.fits"
            if cand_flat.exists():
                return cand_flat
            # Else cosmics-cleaned
            cand_clean = cos_clean_dir / f"{stem}_clean.fits"
            if cand_clean.exists():
                return cand_clean
            return raw_p

        if obj:
            sky_inputs.extend(_best_obj_input(p) for p in obj)
        else:
            # If frames.obj is not configured, mimic stage behavior: consume all
            # available pre-products.
            if flat_dir.exists():
                sky_inputs.extend(sorted(p for p in flat_dir.glob("*_flat.fits") if p.is_file()))
            elif cos_clean_dir.exists():
                sky_inputs.extend(sorted(p for p in cos_clean_dir.glob("*_clean.fits") if p.is_file()))

        sky_inputs.append(wsol / "lambda_map.fits")
        return sky_inputs
    if task in ("stack2d", "stack"):
        # Stack2D depends on rectified sky-subtracted frames produced by Linearization.
        inputs: list[Path] = []
        lin_stage = stage_dir(wd, "linearize")
        try:
            if lin_stage.exists():
                inputs.extend(sorted(p for p in lin_stage.glob("*_skysub.fits") if p.is_file()))
                if not inputs:
                    inputs.extend(sorted(p for p in lin_stage.rglob("*_skysub.fits") if p.is_file()))
        except Exception:
            pass
        if not inputs:
            legacy_dirs = [wd / "products" / "lin" / "per_exp", wd / "lin" / "per_exp"]
            for d in legacy_dirs:
                if d.exists():
                    inputs.extend(sorted(p for p in d.glob("*_skysub.fits") if p.is_file()))
                    if inputs:
                        break
        if not inputs:
            inputs.append(lin_stage)
        return inputs
    if task == "extract1d":
        return [
            resolve_input_path(
                "stacked2d_fits",
                wd,
                "stack2d",
                relpath="stacked2d.fits",
                extra_candidates=[wd / "products" / "stack" / "stacked2d.fits", wd / "stack" / "stacked2d.fits"],
            )
        ]
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
    cfg = (
        load_config_any(cfg_or_path)
        if not isinstance(cfg_or_path, dict)
        else cfg_or_path
    )
    work_dir = resolve_work_dir(cfg)

    plan: list[PlanItem] = []
    for raw in task_names:
        t = canonical_task_name(raw)
        if t not in TASKS:
            plan.append(
                PlanItem(task=t, action="run", reason="Unknown task (will fail)")
            )
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
    qc_override: bool = False,
    cancel_token: CancelToken | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if isinstance(cfg_or_path, RunContext):
        cfg = cfg_or_path.cfg
        if config_path is None:
            config_path = cfg_or_path.cfg_path
    else:
        cfg = (
            load_config_any(cfg_or_path)
            if not isinstance(cfg_or_path, dict)
            else cfg_or_path
        )

    work_dir = resolve_work_dir(cfg)
    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # P2 safety belt: ensure minimal layout + validate workspace/run passport.
    try:
        from scorpio_pipe.work_layout import ensure_work_layout
        from scorpio_pipe.run_validate import validate_run_dir

        ensure_work_layout(out_dir)
        # P0-PROV-002: run passport (run.json) must exist even if we crash later.
        try:
            ensure_run_passport(out_dir)
        except Exception:
            # Never block a run on passport capture.
            log.debug("Failed to ensure run passport", exc_info=True)
        validate_run_dir(out_dir, strict=True)
    except Exception as e:
        raise RuntimeError(
            f"Invalid run folder layout: {out_dir}\n{e}"
        ) from e

    # P0-PROV-003: capture a publication-grade provenance bundle.
    prov_paths: dict[str, str] = {}
    hash_cache = {}
    try:
        prov_paths = ensure_run_provenance(out_dir, cfg, config_path=config_path)
        raw_hashes_path = Path(prov_paths.get("raw_hashes_json", out_dir / "manifest" / "raw_hashes.json"))
        hash_cache = load_hash_cache(raw_hashes_path)
    except Exception:
        # Must never crash the pipeline.
        log.debug("Failed to capture run provenance", exc_info=True)

    # ------------------------------------------------------------------
    # P0-PROV-002: done.json must exist for every stage (ok/skip/fail).
    # We also enrich any stage-written done.json with runner-level metadata.
    # ------------------------------------------------------------------

    def _deep_merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def _atomic_write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(path)



    def _upsert_stage_done_json(
        task: str,
        *,
        status: str,
        stage_hash: str | None,
        started_utc: str | None = None,
        finished_utc: str | None = None,
        duration_s: float | None = None,
        reason: str | None = None,
        res_obj: Any | None = None,
        err_code: str | None = None,
        err_msg: str | None = None,
        trace: str | None = None,
    ) -> None:
        """Ensure a task-scoped done.json exists and enrich it with runner provenance.

        Must never raise (best-effort); failure to write done.json must not break the run.
        """

        def _worst_status(a: str | None, b: str | None) -> str:
            order = {
                "ok": 0,
                "warn": 1,
                "skipped": 2,
                "cancelled": 3,
                "blocked": 4,
                "fail": 5,
            }
            aa = str(a or "ok").strip().lower()
            bb = str(b or "ok").strip().lower()
            aa = {"failed": "fail", "skip": "skipped", "canceled": "cancelled"}.get(aa, aa)
            bb = {"failed": "fail", "skip": "skipped", "canceled": "cancelled"}.get(bb, bb)
            return aa if order.get(aa, 0) >= order.get(bb, 0) else bb

        try:
            ddir = _done_dir_for_task(out_dir, task)
            ddir.mkdir(parents=True, exist_ok=True)
            done_p = ddir / "done.json"

            # Base structure if the task did not create its own done.json.
            if not done_p.exists():
                try:
                    write_done_json(
                        stage=task,
                        stage_dir=ddir,
                        status=status,
                        error_code=err_code,
                        error_message=err_msg,
                    )
                except Exception:
                    _atomic_write_json(
                        done_p,
                        {
                            "stage": task,
                            "status": status,
                            "error_code": err_code,
                            "error_message": err_msg,
                        },
                    )

            # Read existing payload.
            existing: dict = {}
            try:
                existing = json.loads(done_p.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            except Exception:
                existing = {}

            merged_status = _worst_status(existing.get("status"), status)

            # Provenance: input hashes.
            input_paths = _input_paths_for_hash(cfg, task, out_dir)
            try:
                input_hashes = compute_input_hashes(input_paths, raw_cache=hash_cache)
            except Exception:
                input_hashes = []

            # Outputs: best-effort list.
            try:
                out_paths = _collect_output_paths(res_obj) if res_obj is not None else []
            except Exception:
                out_paths = []

            outputs_list = [str(p) for p in out_paths]

            # Bonus: product sanity metrics (SCORPNAN counter + NO_COVERAGE fraction).
            products_sanity: dict[str, Any] = {}
            try:
                from astropy.io import fits  # type: ignore
                import numpy as np  # type: ignore
                from scorpio_pipe.maskbits import NO_COVERAGE

                per: list[dict[str, Any]] = []
                scorpnan_total = 0
                cov_fracs: list[float] = []

                for p in out_paths:
                    try:
                        p = Path(str(p))
                        if not p.exists() or p.suffix.lower() != ".fits":
                            continue
                        with fits.open(p, memmap=False) as hdul:
                            hdr0 = hdul[0].header
                            scorpnan = int(hdr0.get("SCORPNAN", 0) or 0)
                            scorpnan_total += scorpnan

                            no_cov: float | None = None

                            # 2D MEF product
                            if "MASK" in hdul:
                                m = hdul["MASK"].data
                                if m is not None:
                                    mm = m.astype(np.uint16, copy=False)
                                    no_cov = float(np.mean((mm & np.uint16(NO_COVERAGE)) != 0))

                            # 1D product tables
                            elif "SPEC_TRACE" in hdul:
                                fracs: list[float] = []
                                try:
                                    mt = hdul["SPEC_TRACE"].data["MASK_TRACE"]
                                    if mt is not None:
                                        fracs.append(
                                            float(
                                                np.mean(
                                                    (mt.astype(np.uint16) & np.uint16(NO_COVERAGE)) != 0
                                                )
                                            )
                                        )
                                except Exception:
                                    pass
                                try:
                                    mf = hdul["SPEC_FIXED"].data["MASK_FIXED"]
                                    if mf is not None:
                                        fracs.append(
                                            float(
                                                np.mean(
                                                    (mf.astype(np.uint16) & np.uint16(NO_COVERAGE)) != 0
                                                )
                                            )
                                        )
                                except Exception:
                                    pass
                                if fracs:
                                    no_cov = float(max(fracs))

                            if no_cov is not None:
                                cov_fracs.append(float(no_cov))

                            per.append(
                                {
                                    "path": str(p),
                                    "scorpnan": scorpnan,
                                    "no_coverage_frac": no_cov,
                                }
                            )
                    except Exception:
                        continue

                if per:
                    products_sanity = {
                        "scorpnan_total": int(scorpnan_total),
                        "no_coverage_frac_max": float(max(cov_fracs)) if cov_fracs else None,
                        "no_coverage_frac_mean": float(sum(cov_fracs) / len(cov_fracs)) if cov_fracs else None,
                        "products": per,
                    }
            except Exception:
                products_sanity = {}

            additions: dict[str, Any] = {
                "stage": task,
                "status": merged_status,
                "stage_hash": stage_hash,
                "runner": {
                    "started_utc": started_utc,
                    "finished_utc": finished_utc,
                    "duration_s": duration_s,
                    "reason": reason,
                    "traceback": trace,
                },
                "error_code": err_code,
                "error_message": err_msg,
                "input_hashes": input_hashes,
                "effective_config": effective_config if isinstance(effective_config, dict) else {},
                "outputs_list": outputs_list,
                "provenance_bundle": prov_paths if isinstance(prov_paths, dict) else {},
            }
            if products_sanity:
                additions.setdefault("metrics", {})
                if isinstance(additions["metrics"], dict):
                    additions["metrics"].setdefault("products_sanity", products_sanity)

            merged = _deep_merge(existing, additions)
            merged["status"] = merged_status
            _atomic_write_json(done_p, merged)
        except Exception:
            # Must never fail the run
            log.debug("Failed to upsert done.json for %s", task, exc_info=True)
    results: dict[str, Any] = {}

    plan = plan_sequence(
        cfg, task_names, resume=resume, force=force, config_path=config_path
    )
    for it in plan:
        if cancel_token is not None and cancel_token.cancelled:
            log.warning("Cancelled before %s", _task_label(it.task))
            _upsert_stage_done_json(it.task, status="cancelled", stage_hash=None, reason="Cancelled", started_utc=datetime.now(timezone.utc).isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(), duration_s=0.0)
            record_stage_result(
                out_dir,
                it.task,
                status="cancelled",
                stage_hash=None,
                message="Cancelled",
                trace=None,
            )
            break

        t = it.task
        if it.action == "skip":
            log.info("Skip %s (%s)", _task_label(t), it.reason)
            results[t] = None
            try:
                from scorpio_pipe.qc.metrics_store import update_after_stage

                update_after_stage(cfg, stage=t, status="skipped", stage_hash=None)
            except Exception:
                pass
            _upsert_stage_done_json(t, status="skipped", stage_hash=None, reason=str(it.reason or "skipped"), started_utc=datetime.now(timezone.utc).isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(), duration_s=0.0)
            continue

        fn = TASKS.get(t)
        if fn is None:
            raise KeyError(f"Unknown task: {t}")

        # QC gate: block downstream execution if any upstream stage wrote ERROR/FATAL flags.
        try:
            from scorpio_pipe.qc.gate import QCGateError, check_qc_gate

            check_qc_gate(cfg, task=t, allow_override=bool(qc_override))
        except QCGateError as ge:
            _upsert_stage_done_json(t, status="blocked", stage_hash=None, reason=ge.summary(), started_utc=datetime.now(timezone.utc).isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(), duration_s=0.0, err_code="QC_GATE", err_msg=ge.summary())
            record_stage_result(
                out_dir,
                t,
                status="blocked",
                stage_hash=None,
                message=ge.summary(),
                trace=None,
                meta={"qc_override": bool(qc_override)},
            )
            raise
        except Exception:
            # Never let gating crash the runner; it should only gate when data exists.
            pass

        stage_hash = compute_stage_hash(
            stage=t,
            stage_cfg=_stage_cfg_for_hash(cfg, t),
            input_paths=_input_paths_for_hash(cfg, t, out_dir),
        )

        log.info("Run %s...", _task_label(t))
        t_start = time.time()
        started_utc = datetime.now(timezone.utc).isoformat()
        try:
            res = _call_maybe_with_cancel(
                fn,
                cfg=cfg,
                out_dir=out_dir,
                config_path=config_path,
                cancel_token=cancel_token,
            )
            results[t] = res
            # Hard-fail boundary contract enforcement for produced products
            try:
                _validate_task_products(t, out_dir)
            except ProductContractError as ce:
                raise
            except Exception as ce:
                raise ProductContractError(stage=t, path="", code="CONTRACT", message=str(ce)) from ce

            finished_utc = datetime.now(timezone.utc).isoformat()
            duration_s = float(max(0.0, time.time() - t_start))
            _upsert_stage_done_json(
                t,
                status="ok",
                stage_hash=stage_hash,
                started_utc=started_utc,
                finished_utc=finished_utc,
                duration_s=duration_s,
                res_obj=res,
            )

            record_stage_result(
                out_dir,
                t,
                status="ok",
                stage_hash=stage_hash,
                message=None,
                trace=None,
                meta={"qc_override": bool(qc_override)} if qc_override else None,
            )
            try:
                # UI session + history snapshot (P1-G UI-020).
                try:
                    from scorpio_pipe.ui.session_store import snapshot, update_stage
                    from scorpio_pipe.workspace_paths import stage_dir

                    done_name_map = {
                        "sky": "sky_done.json",
                        "linearize": "linearize_done.json",
                        "stack2d": "stack_done.json",
                        "extract1d": "extract_done.json",
                    }
                    cfg_section_map = {
                        "sky": "sky",
                        "linearize": "linearize",
                        "stack2d": "stack2d",
                        "extract1d": "extract1d",
                    }
                    done_path = None
                    if t in done_name_map:
                        dp = stage_dir(out_dir, t) / done_name_map[t]
                        if dp.is_file():
                            done_path = dp
                    sect = cfg_section_map.get(t)
                    stage_cfg_obj = None
                    if sect and isinstance(cfg, dict):
                        v = cfg.get(sect)
                        if isinstance(v, dict):
                            stage_cfg_obj = v

                    update_stage(
                        out_dir,
                        t,
                        cfg_section=stage_cfg_obj,
                        done_path=str(done_path) if done_path else None,
                        status="ok",
                    )
                    snapshot(out_dir, reason=f"stage_{t}_ok")
                except Exception:
                    pass

                from scorpio_pipe.qc.metrics_store import (
                    mirror_qc_to_products,
                    update_after_stage,
                )

                stage_metrics = None
                if isinstance(res, dict):
                    stage_metrics = {}
                    for k, v in res.items():
                        # Store numeric scalars (+ a few small categorical values), but skip file paths.
                        if k.endswith(
                            ("_fits", "_png", "_csv", "_json", "_txt", "_dir")
                        ):
                            continue
                        if isinstance(v, (int, float, bool)):
                            stage_metrics[k] = v
                        elif isinstance(v, str) and k in {"model2d_kind"}:
                            stage_metrics[k] = v
                update_after_stage(
                    cfg,
                    stage=t,
                    status="ok",
                    stage_hash=stage_hash,
                    stage_metrics=stage_metrics,
                )
                if t in ("manifest", "qc_report"):
                    mirror_qc_to_products(out_dir)
            except Exception:
                pass
        except Exception as e:
            tb = traceback.format_exc()
            log.error("Task %s failed: %s", t, e, exc_info=True)
            finished_utc = datetime.now(timezone.utc).isoformat()
            duration_s = float(max(0.0, time.time() - t_start)) if "t_start" in locals() else None
            _upsert_stage_done_json(
                t,
                status="fail",
                stage_hash=stage_hash if "stage_hash" in locals() else None,
                started_utc=started_utc if "started_utc" in locals() else None,
                finished_utc=finished_utc,
                duration_s=duration_s,
                reason=str(e),
                res_obj=results.get(t),
                err_code=type(e).__name__,
                err_msg=str(e),
                trace=tb,
            )
            record_stage_result(
                out_dir,
                t,
                status="failed",
                stage_hash=stage_hash,
                message=str(e),
                trace=tb,
                meta={"qc_override": bool(qc_override)} if qc_override else None,
            )
            try:
                # UI session + history snapshot (P1-G UI-020).
                try:
                    from scorpio_pipe.ui.session_store import snapshot, update_stage

                    update_stage(out_dir, t, status="failed", message=str(e))
                    snapshot(out_dir, reason=f"stage_{t}_failed")
                except Exception:
                    pass

                from scorpio_pipe.qc.metrics_store import update_after_stage

                update_after_stage(
                    cfg,
                    stage=t,
                    status="failed",
                    stage_hash=stage_hash,
                    stage_metrics={"error": str(e)},
                )
            except Exception:
                pass
            raise

    return results


def run_one(
    cfg_or_path: dict[str, Any] | str | Path,
    task_name: str,
    *,
    resume: bool = True,
    force: bool = False,
    qc_override: bool = False,
) -> None:
    run_sequence(cfg_or_path, [task_name], resume=resume, force=force, qc_override=qc_override)
