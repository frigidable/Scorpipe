from __future__ import annotations

from pathlib import Path

from scorpio_pipe.stage_registry import REGISTRY, iter_gui_stages
from scorpio_pipe.workspace_paths import stage_dir


def test_stage_registry_has_exact_12_and_order() -> None:
    stages = list(REGISTRY.all())
    assert len(stages) == 12

    # Canonical directory names (P2 contract)
    assert [s.dir_name for s in stages] == [
        "01_project",
        "02_setup",
        "03_bias",
        "04_flat",
        "05_cosmics",
        "06_superneon",
        "07_arc_line_id",
        "08_wavesol",
        "09_sky",
        "10_linearize",
        "11_stack",
        "12_extract",
    ]


def test_gui_order_follows_registry_single_source_of_truth() -> None:
    gui = list(iter_gui_stages())
    assert [s.key for s in gui] == [s.key for s in REGISTRY.all()]


def test_stage_dir_is_nn_plus_slug() -> None:
    run_root = Path("/tmp/workspace/31_12_2025/ngc2146_VPHG1200@540_01")
    assert stage_dir(run_root, "project").name == "01_project"
    assert stage_dir(run_root, "setup").name == "02_setup"
    assert stage_dir(run_root, "biascorr").name == "03_bias"
    assert stage_dir(run_root, "flatfield").name == "04_flat"
    assert stage_dir(run_root, "cosmics").name == "05_cosmics"
    assert stage_dir(run_root, "superneon").name == "06_superneon"
    # renamed slug (compat key preserved)
    assert stage_dir(run_root, "arclineid").name == "07_arc_line_id"
    assert stage_dir(run_root, "wavesol").name == "08_wavesol"
    assert stage_dir(run_root, "sky").name == "09_sky"
    assert stage_dir(run_root, "linearize").name == "10_linearize"
    assert stage_dir(run_root, "stack2d").name == "11_stack"
    assert stage_dir(run_root, "extract1d").name == "12_extract"
