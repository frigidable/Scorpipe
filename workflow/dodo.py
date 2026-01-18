"""Doit workflow adapter for scorpio-pipe.

Contract (P0-A)
---------------
This file must NOT implement its own execution semantics.

All stage execution, skip/re-run logic, QC gating, done.json/stage_state updates
and boundary-contract validation are delegated to the core engine:

    scorpio_pipe.pipeline.engine

Doit remains responsible only for ergonomics (DAG dependencies, CLI sugar).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

# doit is an optional dependency (workflow is used only when installed).
try:
    from doit import get_var  # type: ignore
except Exception:  # pragma: no cover
    def get_var(name: str):  # type: ignore
        return None


from scorpio_pipe.log import setup_logging
from scorpio_pipe.config import load_config_any
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.pipeline.engine import (
    canonical_task_name,
    done_json_path_for_task,
    run_sequence,
)


setup_logging()

_CFG_CACHE: dict[str, Any] | None = None


def _cfg_path() -> Path:
    """Return config path from doit variable or environment.

    Supported:
      - doit: `doit CONFIG=path/to/config.yaml <task>`
      - env:  `CONFIG=path/to/config.yaml doit <task>`
    """

    v = get_var("CONFIG") or os.environ.get("CONFIG") or "config.yaml"
    return Path(str(v)).expanduser().resolve()


def _load_cfg() -> dict[str, Any]:
    global _CFG_CACHE
    if _CFG_CACHE is None:
        _CFG_CACHE = load_config_any(_cfg_path())
    return _CFG_CACHE


def _work_dir() -> Path:
    return resolve_work_dir(_load_cfg())


def _bool_var(name: str, default: bool = False) -> bool:
    raw = get_var(name)
    if raw is None:
        raw = os.environ.get(name.upper())
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _engine_run(task_names: Iterable[str]) -> dict[str, Any]:
    cfg_path = _cfg_path()
    return run_sequence(
        cfg_path,
        list(task_names),
        resume=_bool_var("resume", True),
        force=_bool_var("force", False),
        qc_override=_bool_var("qc_override", False),
        config_path=cfg_path,
    )


def _doit_task(name: str, *, task_dep: list[str] | None = None) -> dict[str, Any]:
    canon = canonical_task_name(name)
    wd = _work_dir()
    target = done_json_path_for_task(wd, canon)

    return {
        "actions": [(lambda n=name: _engine_run([n]))],
        # Doit should not decide up-to-date based on file mtimes;
        # engine handles resume/force deterministically via stage_state hashes.
        "uptodate": [False],
        "targets": [str(target)],
        "task_dep": task_dep or [],
    }


# --- Canonical tasks ---

def task_manifest():
    return _doit_task("manifest")


def task_superbias():
    return _doit_task("superbias", task_dep=["manifest"])


def task_superflat():
    return _doit_task("superflat", task_dep=["superbias"])


def task_cosmics():
    return _doit_task("cosmics", task_dep=["superbias"])


def task_flatfield():
    return _doit_task("flatfield", task_dep=["superflat"])


def task_superneon():
    return _doit_task("superneon", task_dep=["manifest"])


def task_lineid_prepare():
    return _doit_task("lineid_prepare", task_dep=["superneon"])


def task_wavesolution():
    return _doit_task("wavesolution", task_dep=["lineid_prepare"])


def task_linearize():
    return _doit_task("linearize", task_dep=["wavesolution", "flatfield", "cosmics"])


def task_sky():
    return _doit_task("sky", task_dep=["linearize"])


def task_stack2d():
    return _doit_task("stack2d", task_dep=["sky"])


def task_extract1d():
    return _doit_task("extract1d", task_dep=["stack2d"])


def task_qc_report():
    return _doit_task("qc_report", task_dep=["extract1d", "wavesolution", "manifest"])


def task_navigator():
    return _doit_task("navigator", task_dep=["qc_report"])


# --- Aliases (backward compatibility) ---

def task_wavesol():
    return _doit_task("wavesol", task_dep=["lineid_prepare"])


def task_wavelength_solution():
    # Historical name used by some notebooks
    return _doit_task("wavelength_solution", task_dep=["lineid_prepare"])


def task_sky_sub():
    return _doit_task("sky_sub", task_dep=["linearize"])


def task_stack():
    return _doit_task("stack", task_dep=["sky"])


def task_run_all():
    """Convenience task: run the full default chain via the engine."""

    chain = [
        "manifest",
        "superbias",
        "superflat",
        "cosmics",
        "flatfield",
        "superneon",
        "lineid_prepare",
        "wavesolution",
        "linearize",
        "sky",
        "stack2d",
        "extract1d",
        "qc_report",
        "navigator",
    ]

    wd = _work_dir()
    target = done_json_path_for_task(wd, "navigator")
    return {
        "actions": [(lambda: _engine_run(chain))],
        "uptodate": [False],
        "targets": [str(target)],
        "task_dep": [],
    }
