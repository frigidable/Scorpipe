from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

import os

from doit import get_var

from scorpio_pipe.config import load_config
from scorpio_pipe.manifest import write_manifest
from scorpio_pipe.log import setup_logging
from scorpio_pipe.wavesol_paths import resolve_work_dir

setup_logging(os.environ.get("SCORPIO_LOG_LEVEL"))


def _project_root() -> Path:
    # .../scorpio_pipe/workflow/dodo.py -> .../scorpio_pipe
    return Path(__file__).resolve().parents[1]


def _resolve_from_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_project_root() / p).resolve()


def _touch(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        path.write_text(f"created {time.ctime()}\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _timed(stage: str, work_dir: Path, fn) -> None:
    """Run a task action and append timing into report/timings.json.

    Timings must be best-effort and never break the workflow.
    """
    try:
        from scorpio_pipe.timings import timed_stage

        with timed_stage(work_dir=work_dir, stage=stage):
            fn()
    except Exception:
        # If timings infra fails for any reason, still run the task.
        fn()


_CFG_CACHE: dict | None = None


def _cfg_path() -> Path:
    """Resolve config path.

    Priority:
      1) doit var:  config=path.yaml
      2) env var:   CONFIG=path.yaml

    If the path is relative, it's interpreted relative to project root
    (directory above `workflow/`).
    """
    raw = (get_var("config") or os.environ.get("CONFIG") or "").strip()
    if not raw:
        raise RuntimeError(
            "Config is required. Example (PowerShell):\n"
            "  $env:CONFIG=(Resolve-Path .\\work\\run1\\config.yaml).Path\n"
            "  .\\.venv\\Scripts\\doit.exe -f workflow/dodo.py list\n\n"
            "Or:  .\\.venv\\Scripts\\doit.exe -f workflow/dodo.py config=work/<dd_mm_yyyy>/config.yaml list"
        )

    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_project_root() / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return p


def _load_cfg() -> dict:
    global _CFG_CACHE
    if _CFG_CACHE is None:
        _CFG_CACHE = load_config(_cfg_path())
    return _CFG_CACHE

def task_manifest():
    """
    Общий manifest для воспроизводимости: что было выбрано и где work_dir.
    """
    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    out = work_dir / "report" / "manifest.json"

    def _action():
        _timed("manifest", work_dir, lambda: write_manifest(out_path=out, cfg=cfg, cfg_path=_cfg_path()))

    return {
        "actions": [_action],
        "file_dep": [_cfg_path()],
        "targets": [out],
        "clean": True,
    }

def task_qc_report():
    """Lightweight QC summary (JSON + HTML index)."""
    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    out_html = work_dir / "report" / "index.html"
    out_json = work_dir / "report" / "qc_report.json"

    def _action():
        def _run():
            from scorpio_pipe.qc_report import build_qc_report
            build_qc_report(cfg, out_dir=out_html.parent)

        _timed("qc_report", work_dir, _run)

    return {
        "actions": [_action],
        "file_dep": [ _cfg_path(), work_dir / "report" / "manifest.json" ],
        "targets": [out_html, out_json],
        "task_dep": ["manifest"],
        "clean": True,
    }

def task_superbias():
    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    out = work_dir / "calib" / "superbias.fits"

    bias_list = cfg["frames"].get("bias", [])

    def _action():
        def _run():
            from scorpio_pipe.stages.calib import build_superbias
            # передаём путь к YAML-конфигу, который задан через $env:CONFIG
            build_superbias(_cfg_path(), out_path=out)

        _timed("superbias", work_dir, _run)

    return {
        "actions": [_action],
        "file_dep": [_cfg_path()] + [Path(p) for p in bias_list],
        "targets": [out],
        "clean": True,
    }


def task_superflat():
    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    out = work_dir / "calib" / "superflat.fits"

    flat_list = cfg["frames"].get("flat", [])

    def _action():
        def _run():
            from scorpio_pipe.stages.calib import build_superflat
            # передаём путь к YAML-конфигу, который задан через $env:CONFIG
            build_superflat(_cfg_path(), out_path=out)

        _timed("superflat", work_dir, _run)

    return {
        "actions": [_action],
        "file_dep": [_cfg_path()] + [Path(p) for p in flat_list],
        "targets": [out],
        "clean": True,
    }


def task_cosmics():
    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    out = work_dir / "cosmics" / "summary.json"

    obj_list = cfg["frames"].get("obj", [])
    sky_list = cfg["frames"].get("sky", [])
    file_dep = [_cfg_path()] + [Path(p) for p in obj_list + sky_list]

    def _action():
        def _run():
            from scorpio_pipe.stages.cosmics import clean_cosmics
            clean_cosmics(_cfg_path(), out_dir=out.parent)

        _timed("cosmics", work_dir, _run)

    return {
        "actions": [_action],
        "file_dep": file_dep,
        "targets": [out],
        "clean": True,
    }


def task_superneon():
    """Build stacked super-neon + candidates."""
    def _action():
        cfg = _load_cfg()
        def _run():
            from scorpio_pipe.stages.superneon import build_superneon
            build_superneon(cfg)

        _timed("superneon", resolve_work_dir(cfg), _run)

    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    from scorpio_pipe.wavesol_paths import wavesol_dir as _wavesol_dir
    outdir = _wavesol_dir(cfg)
    neon_list = cfg["frames"].get("neon", [])

    # bias subtraction in superneon is optional
    superneon_cfg = (cfg.get("superneon") or {}) if isinstance(cfg.get("superneon"), dict) else {}
    bias_sub = bool(superneon_cfg.get("bias_sub", True))

    superbias = Path(cfg.get("calib", {}).get("superbias_path") or (work_dir / "calib" / "superbias.fits"))
    targets = [
        outdir / "superneon.fits",
        outdir / "superneon.png",
        outdir / "peaks_candidates.csv",
    ]

    file_dep = [_cfg_path()] + [Path(p) for p in neon_list]
    task_dep = []
    if bias_sub:
        file_dep.append(superbias)
        task_dep.append("superbias")

    return {
        "actions": [_action],
        "file_dep": file_dep,
        "targets": targets,
        "task_dep": task_dep,
        "clean": True,
    }

def task_lineid_prepare():
    cfg = _load_cfg()
    w = resolve_work_dir(cfg)
    from scorpio_pipe.wavesol_paths import wavesol_dir as _wavesol_dir
    wsol_dir = _wavesol_dir(cfg)

    superneon_fits = wsol_dir / "superneon.fits"
    peaks_csv = wsol_dir / "peaks_candidates.csv"
    hand_file = wsol_dir / "hand_pairs.txt"   # итог ручной привязки

    # neon_lines.csv: resolve from work_dir/config_dir/project_root or packaged resource
    from scorpio_pipe.resource_utils import resolve_resource
    lines_csv = cfg.get("wavesol", {}).get("neon_lines_csv", "neon_lines.csv")
    lines_path = resolve_resource(lines_csv, work_dir=w, config_dir=Path(cfg.get("config_dir", w)), project_root=Path(cfg.get("project_root", _project_root())), allow_package=True).path

    def _action():
        def _run():
            from scorpio_pipe.stages.lineid import prepare_lineid
            prepare_lineid(
                cfg,
                superneon_fits=superneon_fits,
                peaks_candidates_csv=peaks_csv,
                hand_file=hand_file,
                neon_lines_csv=lines_path,
                y_half=int(cfg.get("wavesol", {}).get("y_half", 20)),
            )

        _timed("lineid_prepare", w, _run)

    return {
        "actions": [_action],
        "file_dep": [_cfg_path(), superneon_fits, peaks_csv, lines_path],
        "targets": [hand_file],
        "verbosity": 2,
    }


def task_wavesol():
    cfg = _load_cfg()
    from scorpio_pipe.stages.wavesolution import build_wavesolution
    from scorpio_pipe.wavesol_paths import resolve_work_dir
    from scorpio_pipe.wavesol_paths import wavesol_dir

    outdir = wavesol_dir(cfg)
    superneon_fits = outdir / "superneon.fits"
    hand_pairs = outdir / "hand_pairs.txt"

    def _action():
        _timed("wavesolution", resolve_work_dir(cfg), lambda: build_wavesolution(cfg))

    return {
        "actions": [_action],
        "file_dep": [_cfg_path(), superneon_fits, hand_pairs],
        "targets": [
            outdir / "wavesolution_1d.png",
            outdir / "wavesolution_1d.json",
            outdir / "residuals_1d.csv",
            outdir / "wavesolution_2d.json",
            outdir / "residuals_2d.csv",
            outdir / "lambda_map.fits",
            outdir / "wavelength_matrix.png",
            outdir / "residuals_2d.png",
        ],
        "task_dep": ["superneon", "lineid_prepare"],
        "clean": True,
    }


def task_sky_sub():
    """Kelson-style sky subtraction (per exposure by default)."""
    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    out_dir = work_dir / "products" / "sky"
    done = out_dir / "sky_sub_done.json"

    def _action():
        from scorpio_pipe.stages.sky_sub import run_sky_sub

        _timed("sky_sub", work_dir, lambda: run_sky_sub(cfg, out_dir=out_dir))

    return {
        "actions": [_action],
        # Depends on linearize preview and/or per-exposure products
        "file_dep": [_cfg_path(), work_dir / "products" / "lin" / "linearize_done.json"],
        "targets": [done],
        "task_dep": ["linearize"],
        "clean": True,
    }


def task_wavelength_solution():
    # alias, чтобы читалось красиво в графе зависимостей
    return {"actions": None, "task_dep": ["wavesol"]}


def task_linearize():
    """Rectify per-pixel spectra to a common linear wavelength grid (per exposure)."""
    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    out_dir = work_dir / "products" / "lin"
    done = out_dir / "linearize_done.json"

    def _action():
        from scorpio_pipe.stages.linearize import run_linearize

        _timed("linearize", work_dir, lambda: run_linearize(cfg, out_dir=out_dir))

    # wavesol provides the lambda_map; cosmics provides cleaned frames (optional)
    from scorpio_pipe.wavesol_paths import wavesol_dir

    return {
        "actions": [_action],
        "file_dep": [_cfg_path(), wavesol_dir(cfg) / "lambda_map.fits"],
        "targets": [done, out_dir / "lin_preview.fits", out_dir / "lin_preview.png"],
        "task_dep": ["wavesol", "cosmics"],
        "clean": True,
    }


def _collect_sky_sub_frames(cfg: dict[str, Any]) -> list[Path]:
    wd = resolve_work_dir(cfg)
    cand_dirs = [
        wd / "products" / "sky" / "per_exp",
        wd / "sky" / "per_exp",
    ]
    for d in cand_dirs:
        if d.exists():
            frames = sorted(d.glob("*_sky_sub.fits"))
            if frames:
                return frames
    return []


def task_stack2d():
    """Stack per-exposure sky-subtracted rectified frames into a single 2D product."""
    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    out_dir = work_dir / "products" / "stack"
    done = out_dir / "stack2d_done.json"

    def _action():
        from scorpio_pipe.stages.stack2d import run_stack2d

        def _run():
            inputs = _collect_sky_sub_frames(cfg)
            return run_stack2d(cfg, inputs=inputs, out_dir=out_dir)

        _timed("stack2d", work_dir, _run)

    return {
        "actions": [_action],
        "file_dep": [_cfg_path(), work_dir / "products" / "sky" / "sky_sub_done.json"],
        "targets": [done, out_dir / "stacked2d.fits", out_dir / "coverage.png"],
        "task_dep": ["sky_sub"],
        "clean": True,
    }


def task_stack():
    """Legacy alias for stack2d."""
    return {"actions": None, "task_dep": ["stack2d"]}


def task_extract1d():
    """1D extraction (boxcar/optimal) from the stacked 2D product."""
    cfg = _load_cfg()
    work_dir = resolve_work_dir(cfg)
    out_dir = work_dir / "products" / "spec"
    done = out_dir / "extract1d_done.json"

    def _action():
        from scorpio_pipe.stages.extract1d import run_extract1d

        _timed("extract1d", work_dir, lambda: run_extract1d(cfg, out_dir=out_dir))

    return {
        "actions": [_action],
        "file_dep": [_cfg_path(), work_dir / "products" / "stack" / "stacked2d.fits"],
        "targets": [done, out_dir / "spec1d.fits", out_dir / "spec1d.png"],
        "task_dep": ["stack2d"],
        "clean": True,
    }


def task_run_all():
    return {"actions": None, "task_dep": ["manifest", "extract1d"]}
