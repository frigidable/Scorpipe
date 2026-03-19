from __future__ import annotations

import os
import sys
import types
from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext
from scorpio_pipe.pipeline.engine import _make_stage_context
from scorpio_pipe.pipeline.resources import StageResources, apply_stage_resources_to_cfg
from scorpio_pipe.workspace_paths import stage_dir


def test_make_stage_context_resolves_stage_resource_budget(tmp_path: Path) -> None:
    cfg = {
        "runtime": {"n_jobs": 8, "blas_threads": 4, "max_rss_mb": 8192},
        "sky": {"resources": {"n_jobs": 2, "max_rss_mb": 2048}},
    }
    ctx = _make_stage_context(cfg, run_root=tmp_path / "run", task="sky", cancel_token=None)
    assert ctx.resources == StageResources(max_rss_mb=2048, n_jobs=2, blas_threads=4)


def test_apply_stage_resources_to_cfg_caps_heavy_stage_parallelism() -> None:
    cfg = {"linearize": {"perf": {"n_jobs": 7}}}
    out, metrics = apply_stage_resources_to_cfg(
        cfg,
        stage_key="linearize",
        budget=StageResources(max_rss_mb=1024, n_jobs=3, blas_threads=2),
    )
    assert out["linearize"]["perf"]["n_jobs"] == 1
    assert metrics["resource_budget_applied_n_jobs"] == 1
    assert cfg["linearize"]["perf"]["n_jobs"] == 7  # original cfg unchanged


def test_entry_linearize_applies_budget_and_blas_guard(tmp_path: Path, monkeypatch) -> None:
    from scorpio_pipe.stages.entry_linearize import run

    captured: dict[str, object] = {}

    def fake_run_linearize(cfg, out_dir=None, *, cancel_token=None):
        captured["cfg"] = cfg
        captured["out_dir"] = out_dir
        captured["omp"] = os.environ.get("OMP_NUM_THREADS")
        return {}

    fake_mod = types.ModuleType("scorpio_pipe.stages.linearize")
    fake_mod.run_linearize = fake_run_linearize
    monkeypatch.setitem(sys.modules, "scorpio_pipe.stages.linearize", fake_mod)

    ctx = StageContext(
        cfg={"linearize": {"perf": {"n_jobs": 9}}},
        run_root=tmp_path / "run",
        stage_key="linearize",
        stage_dir=tmp_path / "run" / "10_linearize",
        resources=StageResources(max_rss_mb=1536, n_jobs=4, blas_threads=2),
        metrics={},
    )
    run(ctx)

    cfg_eff = captured["cfg"]
    assert isinstance(cfg_eff, dict)
    assert cfg_eff["linearize"]["perf"]["n_jobs"] == 1
    assert captured["omp"] == "2"
    assert ctx.metrics["resource_budget_blas_threads"] == 2
    assert ctx.metrics["resource_budget_applied_n_jobs"] == 1


def test_entry_sky_and_stack2d_switch_to_tighter_chunking(tmp_path: Path, monkeypatch) -> None:
    from scorpio_pipe.stages.entry_sky import run as run_sky
    from scorpio_pipe.stages.entry_stack2d import run as run_stack2d

    sky_captured: dict[str, object] = {}
    stack_captured: dict[str, object] = {}

    def fake_run_sky_sub(cfg, *, out_dir=None):
        sky_captured["cfg"] = cfg
        sky_captured["omp"] = os.environ.get("OMP_NUM_THREADS")
        return {}

    def fake_run_stack2d(cfg, *, inputs, out_dir=None):
        stack_captured["cfg"] = cfg
        stack_captured["inputs"] = list(inputs)
        stack_captured["omp"] = os.environ.get("OMP_NUM_THREADS")
        return {}

    fake_sky = types.ModuleType("scorpio_pipe.stages.sky_sub")
    fake_sky.run_sky_sub = fake_run_sky_sub
    monkeypatch.setitem(sys.modules, "scorpio_pipe.stages.sky_sub", fake_sky)
    fake_stack = types.ModuleType("scorpio_pipe.stages.stack2d")
    fake_stack.run_stack2d = fake_run_stack2d
    monkeypatch.setitem(sys.modules, "scorpio_pipe.stages.stack2d", fake_stack)

    # Sky budget: force serial strip evaluation + smaller strip rows.
    sky_ctx = StageContext(
        cfg={"sky": {"perf": {"strip_workers": 8, "strip_rows": 256}}},
        run_root=tmp_path / "run",
        stage_key="sky",
        stage_dir=tmp_path / "run" / "20_sky",
        resources=StageResources(max_rss_mb=1500, n_jobs=8, blas_threads=3),
        metrics={},
    )
    run_sky(sky_ctx)
    sky_cfg = sky_captured["cfg"]
    assert isinstance(sky_cfg, dict)
    assert sky_cfg["sky"]["perf"]["strip_workers"] == 1
    assert sky_cfg["sky"]["perf"]["strip_rows"] == 32
    assert sky_cfg["sky"]["perf"]["parallel_strips"] is False
    assert sky_captured["omp"] == "3"

    # Stack2D budget: keep streaming but reduce chunk_rows.
    run_root = tmp_path / "run_stack"
    lin_dir = stage_dir(run_root, "linearize")
    lin_dir.mkdir(parents=True, exist_ok=True)
    input_path = lin_dir / "obj1_skysub.fits"
    input_path.write_bytes(b"dummy")

    stack_ctx = StageContext(
        cfg={"frames": {"obj": [str(tmp_path / "raw" / "obj1.fits")]}, "stack2d": {"chunk_rows": 256}},
        run_root=run_root,
        stage_key="stack2d",
        stage_dir=stage_dir(run_root, "stack2d"),
        resources=StageResources(max_rss_mb=1700, n_jobs=6, blas_threads=4),
        metrics={},
    )
    run_stack2d(stack_ctx)
    stack_cfg = stack_captured["cfg"]
    assert isinstance(stack_cfg, dict)
    assert stack_cfg["stack2d"]["chunk_rows"] == 64
    assert stack_ctx.metrics["resource_budget_streaming_mode"] == "chunked"
    assert stack_captured["inputs"] == [input_path]
    assert stack_captured["omp"] == "4"


def test_apply_stage_resources_to_cfg_covers_geometry_slit_and_sky_refine() -> None:
    cfg = {
        "wavesol": {"geometry_prior": {"block_height": 16}},
        "slit_sanity": {"max_candidate_frames": 12},
        "sky_refine": {"detect_snr": 4, "top_n_summary": 10},
    }
    geom_cfg, geom_metrics = apply_stage_resources_to_cfg(cfg, stage_key="geometry_prior", budget=StageResources(max_rss_mb=1500, n_jobs=2, blas_threads=1))
    slit_cfg, slit_metrics = apply_stage_resources_to_cfg(cfg, stage_key="slit_sanity", budget=StageResources(max_rss_mb=1500, n_jobs=2, blas_threads=1))
    refine_cfg, refine_metrics = apply_stage_resources_to_cfg(cfg, stage_key="sky_refine", budget=StageResources(max_rss_mb=1500, n_jobs=2, blas_threads=1))

    assert geom_cfg["wavesol"]["geometry_prior"]["block_height"] == 48
    assert geom_metrics["resource_budget_geometry_block_height"] == 48
    assert slit_cfg["slit_sanity"]["max_candidate_frames"] == 4
    assert slit_metrics["resource_budget_slit_max_candidate_frames"] == 4
    assert refine_cfg["sky_refine"]["detect_snr"] == 5
    assert refine_cfg["sky_refine"]["top_n_summary"] == 4
    assert refine_metrics["resource_budget_sky_refine_top_n_summary"] == 4
