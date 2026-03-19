"""Core pipeline execution engine.

This module is the *single source of truth* for execution semantics across:
- CLI (scorpio-pipe run --execute)
- GUI (via scorpio_pipe.ui.pipeline_runner shim)
- workflow/doit (workflow/dodo.py adapter)

Invariants enforced here:
- Deterministic resume/force (stage_state + stage hash)
- QC gate as the only downstream blocker
- Boundary contract validation for products
- done.json + stage_state are always written (ok/skip/blocked/fail/cancelled)
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import traceback
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from scorpio_pipe.config import load_config_any
from scorpio_pipe.typed_config import build_typed_config
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
from scorpio_pipe.catalog.stages import CATALOG as STAGE_CATALOG
from scorpio_pipe.pipeline.context import CancelToken, StageContext, StageResult, PlanItem, StageSpec
from scorpio_pipe.pipeline.resources import resolve_stage_resources
from scorpio_pipe.perf.profiles import apply_performance_profile, compact_performance_profile_summary
# P0 provenance + boundary contract enforcement
from scorpio_pipe.boundary_contract import (
    ProductContractError,
    validate_lambda_map_product,
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
from scorpio_pipe.artifact_index import update_after_stage as _artifact_index_update_after_stage

log = logging.getLogger(__name__)



# Optional runtime dependency for PERF-00.1 resource metrics.
try:  # pragma: no cover
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


def _mb(x: float | int | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x) / (1024.0 * 1024.0)
    except Exception:
        return None


class _StageResourceMonitor:
    '''Best-effort per-stage process resource monitor.

    Uses psutil when available. Never raises.

    Metrics:
      - rss_mb, rss_peak_mb
      - io_read_mb, io_write_mb (if available)
      - cpu_user_s, cpu_system_s, cpu_total_s (if available)
    '''

    def __init__(self, *, sample_interval_s: float = 0.10) -> None:
        self._sample_interval_s = float(sample_interval_s)
        self._proc = None
        self._stop = threading.Event()
        self._thr = None
        self._started = False
        self._stopped = False

        self._rss_peak = None
        self._cpu0 = None
        self._io0 = None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if psutil is None:
            return
        try:
            self._proc = psutil.Process()  # type: ignore[attr-defined]
            try:
                mi = self._proc.memory_info()
                self._rss_peak = getattr(mi, 'rss', None)
            except Exception:
                self._rss_peak = None
            try:
                self._cpu0 = self._proc.cpu_times()
            except Exception:
                self._cpu0 = None
            try:
                self._io0 = self._proc.io_counters()
            except Exception:
                self._io0 = None

            def _poll() -> None:
                while not self._stop.is_set():
                    try:
                        mi2 = self._proc.memory_info() if self._proc is not None else None
                        rss = getattr(mi2, 'rss', None) if mi2 is not None else None
                        if rss is not None:
                            if self._rss_peak is None or rss > self._rss_peak:
                                self._rss_peak = rss
                    except Exception:
                        pass
                    self._stop.wait(self._sample_interval_s)

            self._thr = threading.Thread(target=_poll, name='scorpipe_rss_poll', daemon=True)
            self._thr.start()
        except Exception:
            self._proc = None

    def stop(self) -> dict[str, float | None]:
        if self._stopped:
            return {}
        self._stopped = True

        if psutil is None or self._proc is None:
            return {
                'rss_mb': None,
                'rss_peak_mb': None,
                'io_read_mb': None,
                'io_write_mb': None,
                'cpu_user_s': None,
                'cpu_system_s': None,
                'cpu_total_s': None,
            }

        try:
            self._stop.set()
            if self._thr is not None:
                try:
                    self._thr.join(timeout=1.0)
                except Exception:
                    pass

            # RSS now
            rss_now = None
            try:
                mi = self._proc.memory_info()
                rss_now = getattr(mi, 'rss', None)
            except Exception:
                rss_now = None

            # CPU delta
            cpu_user = cpu_sys = cpu_total = None
            try:
                c1 = self._proc.cpu_times()
                if self._cpu0 is not None and c1 is not None:
                    cpu_user = float(getattr(c1, 'user', 0.0) - getattr(self._cpu0, 'user', 0.0))
                    cpu_sys = float(getattr(c1, 'system', 0.0) - getattr(self._cpu0, 'system', 0.0))
                    cpu_total = float(cpu_user + cpu_sys)
            except Exception:
                pass

            # IO delta
            io_r = io_w = None
            try:
                io1 = self._proc.io_counters()
                if self._io0 is not None and io1 is not None:
                    io_r = _mb(getattr(io1, 'read_bytes', 0) - getattr(self._io0, 'read_bytes', 0))
                    io_w = _mb(getattr(io1, 'write_bytes', 0) - getattr(self._io0, 'write_bytes', 0))
            except Exception:
                pass

            return {
                'rss_mb': _mb(rss_now),
                'rss_peak_mb': _mb(self._rss_peak),
                'io_read_mb': io_r,
                'io_write_mb': io_w,
                'cpu_user_s': cpu_user,
                'cpu_system_s': cpu_sys,
                'cpu_total_s': cpu_total,
            }
        except Exception:
            return {}



# Canonical task list and task specs are derived from StageCatalog (single source of truth).
CANONICAL_TASKS: tuple[str, ...] = tuple(STAGE_CATALOG.task_keys())

STAGE_SPECS: dict[str, StageSpec] = {
    t.key: StageSpec(
        key=t.key,
        title=str(t.label),
        validate_globs=t.validate_globs,
        contract_kind=t.contract_kind,
        output_keys=t.product_keys,
    )
    for t in STAGE_CATALOG.task_infos()
}


# --- Task registry (lazy entrypoints) ---

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


def _parse_entrypoint(entrypoint: str) -> tuple[str, str]:
    """Parse entrypoint string.

    Supported forms:
      - ``some.module:callable`` (preferred)
      - ``some.module.callable`` (fallback)
    """

    s = str(entrypoint or '').strip()
    if not s:
        raise ValueError('Empty entrypoint')
    if ':' in s:
        mod, attr = s.split(':', 1)
        return mod.strip(), attr.strip()
    if '.' in s:
        mod, attr = s.rsplit('.', 1)
        return mod.strip(), attr.strip()
    raise ValueError(f'Invalid entrypoint format: {entrypoint!r}')


# Canonical task -> entrypoint string (source of truth lives in StageCatalog).
TASK_ENTRYPOINTS: dict[str, str] = {
    t.key: str(t.entrypoint or '').strip()
    for t in STAGE_CATALOG.task_infos()
}

# Compatibility mapping: canonical keys + aliases -> entrypoint string.
# This is used only for membership checks / help text; execution resolves
# canonical names via StageCatalog first.
TASKS: dict[str, str] = {}
for _t in STAGE_CATALOG.task_infos():
    _ep = str(_t.entrypoint or '').strip()
    if _ep:
        TASKS[_t.key] = _ep
        for _a in (_t.aliases or ()):  # type: ignore[truthy-bool]
            aa = str(_a).strip()
            if aa:
                TASKS[aa] = _ep


# Resolver cache (entrypoint string -> callable).
_ENTRYPOINT_CACHE: dict[str, TaskFn] = {}


def resolve_entrypoint(entrypoint: str) -> TaskFn:
    """Resolve and cache entrypoint callable via importlib."""

    ep = str(entrypoint or '').strip()
    if not ep:
        raise ValueError('Empty entrypoint')
    fn = _ENTRYPOINT_CACHE.get(ep)
    if fn is not None:
        return fn

    mod, attr = _parse_entrypoint(ep)
    from importlib import import_module

    m = import_module(mod)
    obj = getattr(m, attr)
    if not callable(obj):
        raise TypeError(f'Entrypoint is not callable: {ep!r}')
    _ENTRYPOINT_CACHE[ep] = obj
    return obj


def task_callable(task: str) -> TaskFn:
    """Return callable for a *canonical* task name (lazy import)."""

    t = canonical_task_name(task)
    ep = TASK_ENTRYPOINTS.get(t)
    if not ep:
        raise KeyError(f'No entrypoint for task: {t!r}')
    return resolve_entrypoint(ep)

def _done_dir_for_task(run_root: Path, task: str) -> Path:
    """Return directory where task-scoped done.json should live.

    The policy comes from StageCatalog (single source of truth).
    Legacy runs are supported by falling back to run_root if the
    special folder does not exist.
    """

    t = STAGE_CATALOG.get_task(task)
    policy = (t.done_dir_policy or "").strip().lower()

    if policy == "manifest":
        d = Path(run_root) / "manifest"
        return d if d.exists() else Path(run_root)

    if policy == "qc":
        d = Path(run_root) / "qc"
        return d if d.exists() else Path(run_root)

    if policy == "ui":
        d = Path(run_root) / "ui" / "navigator"
        return d if d.exists() else Path(run_root)

    if policy in ("stage", "ui_only"):
        stage_key = (t.stage_key or t.key).strip().lower()
        return stage_dir(run_root, stage_key)

    # Fallback: a folder named after the canonical task.
    return Path(run_root) / t.key


# Public helpers (used by CLI/UI/doit adapters)

def done_dir_for_task(run_root: Path, task: str) -> Path:
    """Return the task directory that holds *done.json* for this task.

    The returned directory is always based on the *canonical* task name.
    """

    return _done_dir_for_task(Path(run_root), canonical_task_name(task))


def done_json_path_for_task(run_root: Path, task: str) -> Path:
    """Return the done.json path for a task (canonicalized)."""

    return done_dir_for_task(run_root, task) / "done.json"


def _make_stage_context(
    cfg: dict[str, Any],
    *,
    run_root: Path,
    task: str,
    cancel_token: CancelToken | None,
) -> StageContext:
    """Create a StageContext for a task.

    P0-ARCH-02.1: stages must not derive their own work_dir.
    """

    t = canonical_task_name(task)
    try:
        ti = STAGE_CATALOG.get_task(t)
    except Exception:
        ti = None

    # Most stages write into their own stage_dir. Some legacy calibration flows
    # (SuperNeon/LineID/Wavesolution) write into the disperser-aware wavesol dir.
    if t in ("superneon", "lineid_prepare", "wavesolution"):
        sd = wavesol_dir(cfg, run_root=Path(run_root))
    else:
        stage_key = str((getattr(ti, "stage_key", None) or t)).strip().lower()
        sd = stage_dir(run_root, stage_key)

    perf_summary = compact_performance_profile_summary(cfg)
    perf_metrics = {
        "performance_mode": perf_summary.get("performance_mode"),
        "performance_profile_overrides": perf_summary.get("n_overrides"),
    }

    return StageContext(
        cfg=cfg,
        run_root=Path(run_root),
        stage_key=t,
        stage_dir=Path(sd),
        workspace=None,
        resources=resolve_stage_resources(cfg, t),
        metrics=perf_metrics,
        cancel_token=cancel_token,
    )


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


def _validate_task_products(cfg: dict[str, Any], task: str, out_dir: Path) -> None:
    """Validate boundary contract for *this* task outputs (hard-fail).

    P0-ARCH-01.2: validation is driven by (stage_key, product_keys) rather than
    ad-hoc globs scattered across modules.

    Parameters
    ----------
    cfg
        Pipeline config.
    task
        Canonical task key.
    out_dir
        Directory of the stage outputs (``run_root/NN_slug``).
    """

    # Keep existing gating semantics: only enforce when the stage declares a
    # non-trivial contract kind.
    tinfo = None
    try:
        tinfo = STAGE_CATALOG.get_task(task)
    except Exception:
        tinfo = None
    if tinfo is None:
        return

    kind = str(getattr(tinfo, "contract_kind", "none") or "none").strip().lower()
    if kind in ("", "none"):
        return

    

    # Stage-level contract kinds (cross-product invariants).
    # These run *in addition* to per-product validation.
    if kind == "calib_bias_triplet":
        from scorpio_pipe.contracts.validators import validate_calib_bias_triplet

        validate_calib_bias_triplet(out_dir, stage=str(tinfo.key))
    elif kind == "calib_flat":
        from scorpio_pipe.contracts.validators import validate_calib_flat

        validate_calib_flat(out_dir, stage=str(tinfo.key))
    elif kind == "calib_neon":
        from scorpio_pipe.contracts.validators import validate_calib_neon

        validate_calib_neon(out_dir, stage=str(tinfo.key))

# Validate exactly the declared product keys for this stage.
    try:
        from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

        PRODUCT_CATALOG.validate_products(
            cfg,
            stage_key=str(tinfo.key),
            product_keys=tuple(tinfo.product_keys or ()),
            run_root=Path(out_dir),
        )
    except Exception:
        # Raise into the engine wrapper (it will wrap non-ProductContractError into ProductContractError)
        raise



def canonical_task_name(name: str) -> str:
    return STAGE_CATALOG.canonical_task_name(name)


def _stage_label(task: str) -> str:
    """Human-friendly label for logs.

    Uses StageCatalog for stage numbering when available.
    """

    t = canonical_task_name(task)
    try:
        label = STAGE_CATALOG.task_label(t)
    except Exception:
        label = t
    return f"{label} [{t}]"



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
    wsol = wavesol_dir(cfg, run_root=wd)

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
    cfg = apply_performance_profile(cfg)
    work_dir = resolve_work_dir(cfg)

    # P0-B3: reference context is part of provenance and affects stage hashing.
    # This is a hard contract: missing/unhashable references must fail.
    try:
        from scorpio_pipe.refs.context import ensure_reference_context

        resources_dir = cfg.get("resources_dir") if isinstance(cfg, dict) else None
        _ref_ctx = ensure_reference_context(cfg, resources_dir=resources_dir)
        ref_context_id = str(_ref_ctx.get("context_id") or "")
    except Exception:
        raise

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
            reference_context_id=ref_context_id,
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
    cfg_or_path: Any,
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
    # Accept config dicts, YAML paths, and lightweight wrappers that expose
    # `cfg` and/or `cfg_path` attributes (e.g. GUI RunContext).
    if config_path is None:
        for attr in ('cfg_path', 'config_path'):
            if hasattr(cfg_or_path, attr):
                v = getattr(cfg_or_path, attr)
                try:
                    if isinstance(v, (str, Path)) and str(v).strip():
                        config_path = Path(v)
                        break
                except Exception:
                    pass

    cfg = (
        load_config_any(cfg_or_path)
        if not isinstance(cfg_or_path, dict)
        else cfg_or_path
    )
    cfg = apply_performance_profile(cfg)

    # ------------------------------------------------------------------
    # P0-ARCH-04.1: fail-fast typed config validation.
    # Must happen before any heavy computation / stage execution.
    # ------------------------------------------------------------------
    _ = build_typed_config(
        cfg,
        config_path=str(
            config_path
            or cfg.get("config_path")
            or (str(cfg_or_path) if isinstance(cfg_or_path, (str, Path)) else "")
        )
        or None,
    )

    work_dir = resolve_work_dir(cfg)
    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # P0-B3: reference context is part of provenance and affects stage hashes.
    # The context must exist before planning/running any stage.
    ref_ctx: dict[str, Any] | None = None
    ref_context_id: str = ""

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

    # ------------------------------------------------------------------
    # P0-A2: Preflight is a mandatory *engine gate* (single law).
    # Any entry point that reaches run_sequence must pass this check.
    # ------------------------------------------------------------------
    try:
        from scorpio_pipe.preflight import ensure_preflight

        def _record_preflight_override(pf_status: str) -> None:
            """Record explicit preflight override in provenance.

            Must be best-effort only: never block execution.
            """

            try:
                from scorpio_pipe.preflight import write_preflight_override

                write_preflight_override(
                    out_dir,
                    status=pf_status,
                    reason=preflight_override_reason,
                    source=preflight_override_source,
                )
            except Exception:
                pass

        pf = ensure_preflight(cfg, out_dir)
        if str((pf or {}).get("status") or "").upper() == "ERROR":
            if bool(preflight_override):
                _record_preflight_override("ERROR")
            else:
                raise RuntimeError(
                    "Preflight gate blocked run (status=ERROR). "
                    f"See: {out_dir / 'manifest' / 'preflight_report.json'}"
                )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            "Preflight gate failed unexpectedly. "
            f"See: {out_dir / 'manifest' / 'preflight_report.json'}\n{e}"
        ) from e

    # P0-B3: create/update manifest/reference_context.json (hard contract).
    try:
        from scorpio_pipe.refs.context import ensure_reference_context

        resources_dir = cfg.get("resources_dir") if isinstance(cfg, dict) else None
        ref_ctx = ensure_reference_context(cfg, resources_dir=resources_dir)
        ref_context_id = str(ref_ctx.get("context_id") or "")
    except Exception as e:
        raise RuntimeError(f"Reference context build failed: {e}") from e

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
                "effective_config": _stage_cfg_for_hash(cfg, task) if isinstance(cfg, dict) else {},
                "outputs_list": outputs_list,
                "provenance_bundle": prov_paths if isinstance(prov_paths, dict) else {},
                "reference_context": {
                    "context_id": ref_context_id,
                    "path": "manifest/reference_context.json",
                },
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

    def _update_artifact_index(task: str, *, stage_products: dict[str, str] | None = None) -> None:
        """Best-effort: upsert manifest/artifact_index.json after a stage.

        PERF-20.1: engine writes a product-based index to avoid UI os.walk scans.
        Must never raise.
        """

        try:
            _artifact_index_update_after_stage(dict(cfg), work_dir=out_dir, stage=str(task), stage_products=stage_products)
        except Exception:
            pass
    def _expected_hash(task: str) -> str:
        return compute_stage_hash(
            stage=task,
            stage_cfg=_stage_cfg_for_hash(cfg, task),
            input_paths=_input_paths_for_hash(cfg, task, out_dir),
            reference_context_id=ref_context_id,
        )

    for it in plan:
        if cancel_token is not None and cancel_token.cancelled:
            log.warning("Cancelled before %s", _stage_label(it.task))
            _upsert_stage_done_json(it.task, status="cancelled", stage_hash=_expected_hash(it.task), reason="Cancelled", started_utc=datetime.now(timezone.utc).isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(), duration_s=0.0)
            record_stage_result(
                out_dir,
                it.task,
                status="cancelled",
                stage_hash=None,
                message="Cancelled",
                trace=None,
            )
            _update_artifact_index(it.task)
            try:
                from scorpio_pipe.qc.metrics_store import update_after_stage
                update_after_stage(cfg, stage=it.task, status="cancelled", stage_hash=_expected_hash(it.task))
            except Exception:
                pass
            try:
                from scorpio_pipe.timings import append_timing
                append_timing(work_dir=out_dir, stage=it.task, seconds=0.0, ok=False, extra={"cancelled": True})
            except Exception:
                pass
            break

        t = it.task
        if it.action == "skip":
            log.info("Skip %s (%s)", _stage_label(t), it.reason)
            results[t] = None
            try:
                from scorpio_pipe.qc.metrics_store import update_after_stage

                update_after_stage(cfg, stage=t, status="skipped", stage_hash=_expected_hash(t))
            except Exception:
                pass
            _update_artifact_index(t)
            _upsert_stage_done_json(t, status="skipped", stage_hash=_expected_hash(t), reason=str(it.reason or "skipped"), started_utc=datetime.now(timezone.utc).isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(), duration_s=0.0)
            try:
                from scorpio_pipe.timings import append_timing
                append_timing(work_dir=out_dir, stage=t, seconds=0.0, ok=True, extra={"skipped": True})
            except Exception:
                pass
            continue

        fn = task_callable(t)

        # QC gate: block downstream execution if any upstream stage wrote ERROR/FATAL flags.
        try:
            from scorpio_pipe.qc.gate import QCGateError, check_qc_gate

            check_qc_gate(cfg, task=t, allow_override=bool(qc_override))
        except QCGateError as ge:
            _upsert_stage_done_json(t, status="blocked", stage_hash=_expected_hash(t), reason=ge.summary(), started_utc=datetime.now(timezone.utc).isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(), duration_s=0.0, err_code="QC_GATE", err_msg=ge.summary())
            record_stage_result(
                out_dir,
                t,
                status="blocked",
                stage_hash=None,
                message=ge.summary(),
                trace=None,
                meta={"qc_override": bool(qc_override)},
            )
            _update_artifact_index(t)
            try:
                from scorpio_pipe.qc.metrics_store import update_after_stage
                update_after_stage(cfg, stage=t, status="blocked", stage_hash=_expected_hash(t), stage_metrics={"reason": ge.summary()})
            except Exception:
                pass
            try:
                from scorpio_pipe.timings import append_timing
                append_timing(work_dir=out_dir, stage=t, seconds=0.0, ok=False, extra={"blocked": True, "reason": ge.summary()})
            except Exception:
                pass
            raise
        except Exception:
            # Never let gating crash the runner; it should only gate when data exists.
            pass

        stage_hash = compute_stage_hash(
            stage=t,
            stage_cfg=_stage_cfg_for_hash(cfg, t),
            input_paths=_input_paths_for_hash(cfg, t, out_dir),
            reference_context_id=ref_context_id,
        )

        log.info("Run %s...", _stage_label(t))
        t_start = time.time()
        started_utc = datetime.now(timezone.utc).isoformat()

        # PERF-00.1: record per-stage resource metrics (CPU/RSS/IO) in timings + metrics.json.
        timing_extra: dict[str, Any] = {}
        stage_resources: dict[str, Any] = {}
        ctx_metrics: dict[str, Any] = {}
        res: Any = None
        res_for_json: Any = None

        try:
            from scorpio_pipe.timings import timed_stage

            with timed_stage(work_dir=out_dir, stage=t, extra=timing_extra):
                mon = _StageResourceMonitor()
                mon.start()
                try:
                    ctx = _make_stage_context(cfg, run_root=out_dir, task=t, cancel_token=cancel_token)
                    if isinstance(getattr(ctx, "metrics", None), dict):
                        ctx_metrics = dict(ctx.metrics)
                        if ctx_metrics:
                            timing_extra.update(ctx_metrics)
                    res = _call_maybe_with_cancel(
                        fn,
                        ctx=ctx,
                        cfg=cfg,
                        out_dir=out_dir,
                        config_path=config_path,
                        cancel_token=cancel_token,
                        preflight_override=bool(preflight_override),
                        preflight_override_reason=preflight_override_reason,
                        preflight_override_source=preflight_override_source,
                    )
                    if isinstance(getattr(ctx, "metrics", None), dict):
                        ctx_metrics = dict(ctx.metrics)
                        if ctx_metrics:
                            timing_extra.update(ctx_metrics)
                    # Normalize StageResult into a JSON-friendly dict.
                    res_for_json = res.as_dict() if isinstance(res, StageResult) else res
                    results[t] = res_for_json

                    # If a stage returns an explicit failure, convert it into an exception.
                    if isinstance(res, StageResult) and str(res.status) == "fail":
                        raise RuntimeError(
                            f"Stage reported failure: {res.error_code or 'FAIL'}: {res.error_message or ''}".strip()
                        )

                    # Hard-fail boundary contract enforcement for produced products.
                    try:
                        _validate_task_products(cfg, t, out_dir)
                    except ProductContractError as ce:
                        raise
                    except Exception as ce:
                        raise ProductContractError(stage=t, path="", code="CONTRACT", message=str(ce)) from ce
                finally:
                    stage_resources = mon.stop() or {}
                    if stage_resources:
                        timing_extra.update(stage_resources)

            finished_utc = datetime.now(timezone.utc).isoformat()
            duration_s = float(max(0.0, time.time() - t_start))
            done_status = "ok"
            if isinstance(res, StageResult) and str(res.status) == "warn":
                done_status = "warn"
            _upsert_stage_done_json(
                t,
                status=done_status,
                stage_hash=stage_hash,
                started_utc=started_utc,
                finished_utc=finished_utc,
                duration_s=duration_s,
                res_obj=res_for_json,
            )

            meta = {"qc_override": bool(qc_override)} if qc_override else None
            if isinstance(res, StageResult) and str(res.status) == "warn":
                meta = dict(meta or {})
                meta["warn"] = True
            record_stage_result(
                out_dir,
                t,
                status="ok" if done_status in ("ok", "warn") else str(done_status),
                stage_hash=stage_hash,
                message=None,
                trace=None,
                meta=meta,
            )
            # PERF-20.1: upsert artifact_index.json using stage-reported products when possible.
            try:
                stage_prods = res.products if isinstance(res, StageResult) else None
            except Exception:
                stage_prods = None
            _update_artifact_index(t, stage_products=stage_prods)
            try:
                # UI session + history snapshot (P1-G UI-020).
                try:
                    from scorpio_pipe.pipeline.session_store import snapshot, update_stage
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
                        done_json_rel=str(done_path) if done_path else None,
                        status="ok",
                    )
                    snapshot(out_dir, reason=f"stage_{t}_ok")
                except Exception:
                    pass

                from scorpio_pipe.qc.metrics_store import (
                    mirror_qc_to_products,
                    update_after_stage,
                )

                # Stage metrics: merge stage-returned scalar metrics (if any) with resource metrics.
                stage_metrics: dict[str, Any] = {}
                if isinstance(res_for_json, dict):
                    for k, v in res_for_json.items():
                        # Store numeric scalars (+ a few small categorical values), but skip file paths.
                        if str(k).endswith(("_fits", "_png", "_csv", "_json", "_txt", "_dir")):
                            continue
                        if isinstance(v, (int, float, bool)):
                            stage_metrics[k] = v
                        elif isinstance(v, str) and k in {"model2d_kind"}:
                            stage_metrics[k] = v

                if ctx_metrics:
                    stage_metrics.update(ctx_metrics)
                if stage_resources:
                    stage_metrics.update(stage_resources)

                update_after_stage(
                    cfg,
                    stage=t,
                    status="ok",
                    stage_hash=stage_hash,
                    stage_metrics=stage_metrics or None,
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
                err_code=("CONTRACT" if isinstance(e, ProductContractError) else type(e).__name__),
                err_msg=(f"{e.code}: {e.message}" if isinstance(e, ProductContractError) else str(e)),
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
            # Even on failure, index whatever exists on disk (best-effort).
            try:
                stage_prods = res.products if isinstance(res, StageResult) else None
            except Exception:
                stage_prods = None
            _update_artifact_index(t, stage_products=stage_prods)
            try:
                # UI session + history snapshot (P1-G UI-020).
                try:
                    from scorpio_pipe.pipeline.session_store import snapshot, update_stage

                    update_stage(out_dir, t, status="failed", message=str(e))
                    snapshot(out_dir, reason=f"stage_{t}_failed")
                except Exception:
                    pass

                from scorpio_pipe.qc.metrics_store import update_after_stage

                stage_m: dict[str, Any] = {"error": str(e)}
                if ctx_metrics:
                    stage_m.update(ctx_metrics)
                if stage_resources:
                    stage_m.update(stage_resources)

                update_after_stage(
                    cfg,
                    stage=t,
                    status="failed",
                    stage_hash=stage_hash,
                    stage_metrics=stage_m,
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
