from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from scorpio_pipe.config import load_config
from scorpio_pipe.manifest import write_manifest
from scorpio_pipe.validation import validate_config
from scorpio_pipe.qc_report import build_qc_report


log = logging.getLogger("scorpio")


def _touch(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        path.write_text(f"created {time.ctime()}\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(frozen=True)
class RunContext:
    cfg_path: Path
    cfg: dict
    work_dir: Path


def load_context(cfg_path: str | Path) -> RunContext:
    cfg_path = Path(cfg_path).expanduser().resolve()
    cfg = load_config(cfg_path)

    # Early validation (fail-fast with human-friendly message)
    rep = validate_config(cfg, strict_paths=False)
    if not rep.ok:
        msgs = [f"[{i.code}] {i.message}" for i in rep.errors]
        raise ValueError("Invalid config\n" + "\n".join(msgs))

    work_dir = Path(cfg["work_dir"]).expanduser().resolve()
    return RunContext(cfg_path=cfg_path, cfg=cfg, work_dir=work_dir)


def run_manifest(ctx: RunContext) -> Path:
    out = ctx.work_dir / "report" / "manifest.json"
    log.info("Write manifest → %s", out)
    write_manifest(out_path=out, cfg=ctx.cfg, cfg_path=ctx.cfg_path)
    return out



def run_qc_report(ctx: RunContext) -> Path:
    out = ctx.work_dir / "report" / "index.html"
    log.info("Build QC report → %s", out)
    return build_qc_report(ctx.cfg, out_dir=out.parent)


def run_superbias(ctx: RunContext) -> Path:
    from scorpio_pipe.stages.calib import build_superbias

    out = ctx.work_dir / "calib" / "superbias.fits"
    log.info("Build superbias → %s", out)
    return build_superbias(ctx.cfg_path, out_path=out)


def run_superneon(ctx: RunContext) -> list[Path]:
    from scorpio_pipe.stages.superneon import build_superneon

    log.info("Build superneon (stack + peaks)")
    build_superneon(ctx.cfg)

    # expected outputs (for UI convenience)
    from scorpio_pipe.wavesol_paths import wavesol_dir

    wavesol_dir = wavesol_dir(ctx.cfg)
    outs = [
        wavesol_dir / "superneon.fits",
        wavesol_dir / "superneon.png",
        wavesol_dir / "peaks_candidates.csv",
    ]
    return outs


def run_cosmics(ctx: RunContext) -> Path:
    from scorpio_pipe.stages.cosmics import clean_cosmics

    out = ctx.work_dir / "cosmics" / "summary.json"
    log.info("Clean cosmics → %s", out)
    return clean_cosmics(ctx.cfg, out_dir=out.parent)



def run_flatfield(ctx: RunContext) -> Path:
    from scorpio_pipe.stages.flatfield import run_flatfield as _run_flatfield

    out_dir = ctx.work_dir / "flatfield"
    log.info("Flat-fielding → %s", out_dir)
    return _run_flatfield(ctx.cfg, out_dir=out_dir)
def run_lineid_prepare(ctx: RunContext) -> Path:
    """Open the interactive LineID GUI and write hand_pairs.txt."""

    from scorpio_pipe.stages.lineid import prepare_lineid

    from scorpio_pipe.wavesol_paths import wavesol_dir

    w = ctx.work_dir
    wavesol_dir = wavesol_dir(ctx.cfg)

    superneon_fits = wavesol_dir / "superneon.fits"
    peaks_csv = wavesol_dir / "peaks_candidates.csv"
    # Allow selecting an alternative pairs file in config.
    wcfg = (ctx.cfg.get("wavesol", {}) or {}) if isinstance(ctx.cfg.get("wavesol"), dict) else {}
    hp_raw = str(wcfg.get("hand_pairs_path", "") or "").strip()
    if hp_raw:
        hp = Path(hp_raw)
        hand_file = hp if hp.is_absolute() else (ctx.work_dir / hp).resolve()
    else:
        hand_file = wavesol_dir / "hand_pairs.txt"

    if not superneon_fits.exists():
        raise FileNotFoundError(f"Missing: {superneon_fits} (run superneon first)")
    if not peaks_csv.exists():
        raise FileNotFoundError(f"Missing: {peaks_csv} (run superneon first)")

    lines_csv = (ctx.cfg.get("wavesol", {}) or {}).get("neon_lines_csv", "neon_lines.csv")
    y_half = int((ctx.cfg.get("wavesol", {}) or {}).get("y_half", 20))

    log.info("LineID GUI → will write %s", hand_file)
    prepare_lineid(
        ctx.cfg,
        superneon_fits=superneon_fits,
        peaks_candidates_csv=peaks_csv,
        hand_file=hand_file,
        neon_lines_csv=lines_csv,
        y_half=y_half,
    )
    # Convenience for legacy tools/QC: keep a copy at wavesol/hand_pairs.txt
    try:
        legacy = wavesol_dir / "hand_pairs.txt"
        if legacy.resolve() != hand_file.resolve():
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text(hand_file.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    except Exception:
        pass
    return hand_file



def run_wavesolution(ctx: RunContext) -> dict[str, str]:
    """Build 1D+2D dispersion solution (after LineID)."""
    from scorpio_pipe.stages.wavesolution import build_wavesolution

    out = build_wavesolution(ctx.cfg)
    # return JSON-serializable mapping for UI/QC
    return {k: str(v) for k, v in out.__dict__.items()}

TASKS: dict[str, callable] = {
    "manifest": run_manifest,
    "qc_report": run_qc_report,
    "superbias": run_superbias,
    "cosmics": run_cosmics,
    "flatfield": run_flatfield,
    "superneon": run_superneon,
    "lineid_prepare": run_lineid_prepare,
    "wavesolution": run_wavesolution,
}


def run_sequence(
    cfg_path: str | Path | RunContext,
    task_names: list[str],
    *,
    resume: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Run a sequence of pipeline steps.

    Parameters
    ----------
    cfg_path : path | RunContext
        A path to config.yaml or a pre-built RunContext.
    task_names : list[str]
        Task names in execution order.
    resume : bool
        If True, skip a task when its expected products already exist.
    force : bool
        If True, never skip tasks (even if products exist).
    """

    from scorpio_pipe.timings import append_timing, timed_stage
    from scorpio_pipe.products import products_for_task, task_is_complete

    ctx = cfg_path if isinstance(cfg_path, RunContext) else load_context(cfg_path)
    outputs: dict[str, object] = {}

    for name in task_names:
        fn = TASKS.get(name)
        if fn is None:
            raise ValueError(f"Unknown task: {name}")

        # optional stages
        if name == "flatfield" and not ctx.cfg.get("flatfield", {}).get("enabled", False):
            append_timing(
                work_dir=ctx.work_dir,
                stage=name,
                seconds=0.0,
                ok=True,
                extra={"skipped": True, "reason": "flatfield disabled"},
            )
            continue

        if resume and not force and task_is_complete(ctx.cfg, name):
            ps = products_for_task(ctx.cfg, name)
            try:
                append_timing(
                    work_dir=ctx.work_dir,
                    stage=name,
                    seconds=0.0,
                    ok=True,
                    extra={
                        "skipped": True,
                        "reason": "products already exist",
                        "products": [str(p.path) for p in ps],
                    },
                )
            except Exception:
                pass
            outputs[name] = {"skipped": True, "reason": "products already exist"}
            continue

        with timed_stage(work_dir=ctx.work_dir, stage=name):
            outputs[name] = fn(ctx)

    return outputs
