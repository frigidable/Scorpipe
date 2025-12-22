from __future__ import annotations

from pathlib import Path
import json
import time

import os

from doit import get_var

from scorpio_pipe.config import load_config
from scorpio_pipe.log import setup_logging

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
            "Or:  .\\.venv\\Scripts\\doit.exe -f workflow/dodo.py config=work/run1/config.yaml list"
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
    work_dir = Path(cfg["work_dir"])
    out = work_dir / "report" / "manifest.json"

    return {
        "actions": [(_touch, (out, cfg))],
        "file_dep": [_cfg_path()],
        "targets": [out],
        "clean": True,
    }


def task_superbias():
    cfg = _load_cfg()
    work_dir = Path(cfg["work_dir"])
    out = work_dir / "calib" / "superbias.fits"

    bias_list = cfg["frames"].get("bias", [])

    def _action():
        from scorpio_pipe.stages.calib import build_superbias
        # передаём путь к YAML-конфигу, который задан через $env:CONFIG
        build_superbias(_cfg_path(), out_path=out)

    return {
        "actions": [_action],
        "file_dep": [_cfg_path()] + [Path(p) for p in bias_list],
        "targets": [out],
        "clean": True,
    }


def task_superneon():
    """Build stacked super-neon + candidates."""
    def _action():
        cfg = _load_cfg()
        from scorpio_pipe.stages.superneon import build_superneon
        build_superneon(cfg)

    cfg = _load_cfg()
    work_dir = Path(cfg["work_dir"])
    neon_list = cfg["frames"].get("neon", [])

    # bias subtraction in superneon is optional
    superneon_cfg = (cfg.get("superneon") or {}) if isinstance(cfg.get("superneon"), dict) else {}
    bias_sub = bool(superneon_cfg.get("bias_sub", True))

    superbias = Path(cfg.get("calib", {}).get("superbias_path") or (work_dir / "calib" / "superbias.fits"))
    targets = [
        work_dir / "wavesol" / "superneon.fits",
        work_dir / "wavesol" / "superneon.png",
        work_dir / "wavesol" / "peaks_candidates.csv",
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
    w = Path(cfg["work_dir"])
    wavesol_dir = w / "wavesol"

    superneon_fits = wavesol_dir / "superneon.fits"
    peaks_csv = wavesol_dir / "peaks_candidates.csv"
    hand_file = wavesol_dir / "hand_pairs.txt"   # итог ручной привязки

    # neon_lines.csv у тебя лежит в корне проекта: C:\Users\frigi\Desktop\scorpio_pipe\neon_lines.csv
    # разрешаем и относительный вариант:
    lines_csv = cfg.get("wavesol", {}).get("neon_lines_csv", "neon_lines.csv")
    lines_path = Path(str(lines_csv))
    if not lines_path.is_absolute():
        candidates = [w / lines_path, Path(cfg.get("config_dir", w)) / lines_path, _resolve_from_root(lines_path)]
        for c in candidates:
            if c.exists():
                lines_path = c
                break

    def _action():
        from scorpio_pipe.stages.lineid import prepare_lineid
        prepare_lineid(
            cfg,
            superneon_fits=superneon_fits,
            peaks_candidates_csv=peaks_csv,
            hand_file=hand_file,
            neon_lines_csv=lines_path,
            y_half=int(cfg.get("wavesol", {}).get("y_half", 20)),
        )

    return {
        "actions": [_action],
        "file_dep": [Path(cfg["config_path"]), superneon_fits, peaks_csv, lines_path],
        "targets": [hand_file],
        "verbosity": 2,
    }


def task_wavesol():
    cfg = _load_cfg()
    work_dir = Path(cfg["work_dir"])
    out = work_dir / "wavesol" / "lambda_map.fits"

    neon_list = cfg["frames"].get("neon", [])
    superbias = Path(cfg.get("calib", {}).get("superbias_path") or (work_dir / "calib" / "superbias.fits"))

    return {
        "actions": [(_touch, (out, {"stage": "wavesol", "n_neon": len(neon_list)}))],
        "file_dep": [Path(cfg["config_path"]), superbias] + [Path(p) for p in neon_list],
        "targets": [out],
        "task_dep": ["superbias"],
        "clean": True,
    }


def task_sky_sub():
    cfg = _load_cfg()
    work_dir = Path(cfg["work_dir"])
    out = work_dir / "sky" / "sky_sub_done.json"

    obj_list = cfg["frames"].get("obj", [])
    sky_list = cfg["frames"].get("sky", [])
    superbias = Path(cfg.get("calib", {}).get("superbias_path") or (work_dir / "calib" / "superbias.fits"))
    lambda_map = work_dir / "wavesol" / "lambda_map.fits"

    return {
        "actions": [(_touch, (out, {"stage": "sky_sub", "n_obj": len(obj_list), "n_sky": len(sky_list)}))],
        "file_dep": [Path(cfg["config_path"]), superbias, lambda_map] + [Path(p) for p in obj_list + sky_list],
        "targets": [out],
        "task_dep": ["wavelength_solution"],
        "clean": True,
    }


def task_wavelength_solution():
    # alias, чтобы читалось красиво в графе зависимостей
    return {"actions": None, "task_dep": ["wavesol"]}


def task_linearize():
    cfg = _load_cfg()
    work_dir = Path(cfg["work_dir"])
    out = work_dir / "lin" / "linearize_done.json"

    sky_done = work_dir / "sky" / "sky_sub_done.json"
    lambda_map = work_dir / "wavesol" / "lambda_map.fits"

    return {
        "actions": [(_touch, (out, {"stage": "linearize"}))],
        "file_dep": [sky_done, lambda_map],
        "targets": [out],
        "task_dep": ["sky_sub"],
        "clean": True,
    }


def task_stack():
    cfg = _load_cfg()
    work_dir = Path(cfg["work_dir"])
    out = work_dir / "stack" / "stack_done.json"

    lin_done = work_dir / "lin" / "linearize_done.json"

    return {
        "actions": [(_touch, (out, {"stage": "stack"}))],
        "file_dep": [lin_done],
        "targets": [out],
        "task_dep": ["linearize"],
        "clean": True,
    }


def task_extract1d():
    cfg = _load_cfg()
    work_dir = Path(cfg["work_dir"])
    out = work_dir / "spec" / "spectrum_1d.fits"

    stack_done = work_dir / "stack" / "stack_done.json"

    return {
        "actions": [(_touch, (out, {"stage": "extract1d"}))],
        "file_dep": [stack_done],
        "targets": [out],
        "task_dep": ["stack"],
        "clean": True,
    }


def task_run_all():
    return {"actions": None, "task_dep": ["manifest", "extract1d"]}
