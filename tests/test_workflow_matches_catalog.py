from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_dodo_module():
    repo_root = Path(__file__).resolve().parents[1]
    dodo_path = repo_root / "workflow" / "dodo.py"
    assert dodo_path.exists(), f"Missing workflow/dodo.py at {dodo_path}"

    spec = importlib.util.spec_from_file_location("scorpipe_workflow_dodo", str(dodo_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_doit_graph_respects_stagecatalog_order():
    from scorpio_pipe.catalog.stages import CATALOG as STAGE_CATALOG

    order = list(STAGE_CATALOG.task_keys())
    idx = {k: i for i, k in enumerate(order)}

    dodo = _load_dodo_module()
    deps = dodo.get_task_deps()

    # Must cover at least all canonical tasks
    assert set(order).issubset(set(deps.keys()))

    # Every edge must respect StageCatalog ordering
    for task, ds in deps.items():
        assert task in idx, f"workflow has task not in catalog: {task!r}"
        for d in ds:
            assert d in idx, f"workflow depends on task not in catalog: {d!r}"
            assert idx[d] < idx[task], (
                f"workflow violates StageCatalog order: {d} -> {task} "
                f"(idx {idx[d]} -> {idx[task]})"
            )

    # Explicit science invariant: Kelson sky must precede refine, which must precede linearize.
    assert "sky_refine" in deps["linearize"], "linearize must depend on sky_refine when residual cleanup is enabled"
    assert "sky" in deps["sky_refine"], "sky_refine must depend on sky"
    assert "linearize" not in deps["sky"], "sky must not depend on linearize"

    # Convenience chain must match catalog order
    assert list(dodo.get_run_all_chain()) == order
