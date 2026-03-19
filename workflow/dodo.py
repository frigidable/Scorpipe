"""Doit workflow adapter for scorpio-pipe.

Contract (P0-A)
---------------
This file must NOT implement its own execution semantics.

All stage execution, skip/re-run logic, QC gating, done.json/stage_state updates
and boundary-contract validation are delegated to the core engine:

    scorpio_pipe.pipeline.engine

Doit remains responsible only for ergonomics (DAG dependencies, CLI sugar).

P0-ARCH-08
----------
The Doit DAG must be a *derivative* of :mod:`scorpio_pipe.catalog.stages`.

- Task keys and canonical order come from StageCatalog.
- Any explicit ``task_dep`` edges must not contradict StageCatalog order.
- Critical scientific invariants (e.g. Kelson sky subtraction before rectification)
  are enforced in the DAG.
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
from scorpio_pipe.catalog.stages import CATALOG as STAGE_CATALOG


setup_logging()

_CFG_CACHE: dict[str, Any] | None = None

# ----------------------------------------------------------------------------
# StageCatalog-derived ordering and dependency rules
# ----------------------------------------------------------------------------

# Canonical order of engine tasks (single source of truth)
CATALOG_ORDER: list[str] = list(STAGE_CATALOG.task_keys())
CATALOG_INDEX: dict[str, int] = {k: i for i, k in enumerate(CATALOG_ORDER)}


# Minimal scientific dependencies.
# This mapping is intentionally small; it is validated against StageCatalog.
_BASE_DEPS: dict[str, list[str]] = {
    "manifest": [],
    "superbias": ["manifest"],
    "superflat": ["superbias"],
    "flatfield": ["superflat"],
    "cosmics": ["superbias"],
    "superneon": ["manifest"],
    "slit_sanity": ["flatfield", "superneon"],
    "geometry_prior": ["superneon"],
    "lineid_prepare": ["superneon", "geometry_prior"],
    "wavesolution": ["lineid_prepare", "slit_sanity"],
    "lsf_model": ["wavesolution"],
    # Critical invariant: Kelson sky subtraction must run on RAW geometry,
    # before linearize/rectification.
    "sky": ["wavesolution", "cosmics", "flatfield"],
    "sky_refine": ["sky"],
    "linearize": ["sky_refine"],
    "stack2d": ["linearize"],
    "extract1d": ["stack2d"],
    "publication_ready": ["stack2d", "lsf_model"],
    "kinematics_prep": ["extract1d", "lsf_model"],
    "standard_prepare": ["linearize"],
    "sensfunc": ["standard_prepare"],
    "telluric_model": ["standard_prepare"],
    "fluxcal_apply": ["extract1d", "sensfunc", "telluric_model"],
    "line_measure": ["fluxcal_apply"],
    "science_report": ["line_measure"],
    "qc_report": ["science_report", "wavesolution", "manifest"],
    "observer_dashboard": ["qc_report"],
    "navigator": ["observer_dashboard"],
}


def _validate_deps_against_catalog(deps: dict[str, list[str]]) -> None:
    # 1) all tasks must exist
    missing = [k for k in deps.keys() if k not in CATALOG_INDEX]
    if missing:
        raise RuntimeError(f"workflow/dodo.py: deps contain unknown tasks: {missing}")

    # 2) all dependencies must exist and be ordered
    for task, ds in deps.items():
        ti = CATALOG_INDEX[task]
        for d in ds:
            if d not in CATALOG_INDEX:
                raise RuntimeError(
                    f"workflow/dodo.py: task {task!r} depends on unknown {d!r}"
                )
            di = CATALOG_INDEX[d]
            if di >= ti:
                raise RuntimeError(
                    "workflow/dodo.py: dependency violates StageCatalog order: "
                    f"{d!r} (idx={di}) -> {task!r} (idx={ti}). "
                    "Fix StageCatalog order or workflow deps."
                )

    # 3) Explicit invariant checks (helpful diagnostics)
    if "sky" in deps.get("linearize", []) and "linearize" in deps.get("sky", []):
        raise RuntimeError("workflow/dodo.py: cycle sky <-> linearize")


def _derive_task_deps() -> dict[str, list[str]]:
    # Default empty deps for every catalog task.
    deps: dict[str, list[str]] = {k: [] for k in CATALOG_ORDER}

    for k, ds in _BASE_DEPS.items():
        ck = canonical_task_name(k)
        cdeps = [canonical_task_name(d) for d in ds]
        # De-dup + stable order by catalog
        uniq: list[str] = []
        for d in cdeps:
            if d not in uniq:
                uniq.append(d)
        deps[ck] = sorted(uniq, key=lambda x: CATALOG_INDEX.get(x, 10**9))

    _validate_deps_against_catalog(deps)
    return deps


TASK_DEPS: dict[str, list[str]] = _derive_task_deps()

# Convenience constant used by task_run_all and tests.
RUN_ALL_CHAIN: list[str] = list(CATALOG_ORDER)


def get_task_deps() -> dict[str, list[str]]:
    """Return canonical doit task dependencies (for tests/diagnostics)."""

    return {k: list(v) for k, v in TASK_DEPS.items()}


def get_run_all_chain() -> list[str]:
    return list(RUN_ALL_CHAIN)


def _deps_for(name: str) -> list[str]:
    canon = canonical_task_name(name)
    return list(TASK_DEPS.get(canon, []))


# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# Canonical tasks (keys are derived from StageCatalog)
# ----------------------------------------------------------------------------


def task_manifest():
    return _doit_task("manifest", task_dep=_deps_for("manifest"))


def task_superbias():
    return _doit_task("superbias", task_dep=_deps_for("superbias"))


def task_superflat():
    return _doit_task("superflat", task_dep=_deps_for("superflat"))


def task_cosmics():
    return _doit_task("cosmics", task_dep=_deps_for("cosmics"))


def task_flatfield():
    return _doit_task("flatfield", task_dep=_deps_for("flatfield"))


def task_superneon():
    return _doit_task("superneon", task_dep=_deps_for("superneon"))


def task_lineid_prepare():
    return _doit_task("lineid_prepare", task_dep=_deps_for("lineid_prepare"))


def task_wavesolution():
    return _doit_task("wavesolution", task_dep=_deps_for("wavesolution"))


def task_sky():
    return _doit_task("sky", task_dep=_deps_for("sky"))


def task_linearize():
    return _doit_task("linearize", task_dep=_deps_for("linearize"))


def task_stack2d():
    return _doit_task("stack2d", task_dep=_deps_for("stack2d"))


def task_extract1d():
    return _doit_task("extract1d", task_dep=_deps_for("extract1d"))


def task_lsf_model():
    return _doit_task("lsf_model", task_dep=_deps_for("lsf_model"))


def task_publication_ready():
    return _doit_task("publication_ready", task_dep=_deps_for("publication_ready"))


def task_kinematics_prep():
    return _doit_task("kinematics_prep", task_dep=_deps_for("kinematics_prep"))


def task_qc_report():
    return _doit_task("qc_report", task_dep=_deps_for("qc_report"))


def task_observer_dashboard():
    return _doit_task("observer_dashboard", task_dep=_deps_for("observer_dashboard"))


def task_navigator():
    return _doit_task("navigator", task_dep=_deps_for("navigator"))


# ----------------------------------------------------------------------------
# Aliases (backward compatibility)
# ----------------------------------------------------------------------------


def task_wavesol():
    return _doit_task("wavesol", task_dep=_deps_for("wavesol"))


def task_wavelength_solution():
    # Historical name used by some notebooks
    return _doit_task("wavelength_solution", task_dep=_deps_for("wavelength_solution"))


def task_sky_sub():
    return _doit_task("sky_sub", task_dep=_deps_for("sky_sub"))


def task_stack():
    return _doit_task("stack", task_dep=_deps_for("stack"))


def task_run_all():
    """Convenience task: run the full default chain via the engine."""

    chain = list(RUN_ALL_CHAIN)
    wd = _work_dir()
    target = done_json_path_for_task(wd, chain[-1])
    return {
        "actions": [(lambda: _engine_run(chain))],
        "uptodate": [False],
        "targets": [str(target)],
        "task_dep": [],
    }
