from __future__ import annotations


def test_run_sequence_enforces_preflight_gate(monkeypatch, tmp_path):
    """P0-A2: Any entry into run_sequence must pass preflight gate."""

    # Build a minimally valid run layout expected by validate_run_dir(strict=True).
    night = tmp_path / "30_12_2025"
    run_root = night / "OBJ_VPHG1200_01"
    run_root.mkdir(parents=True, exist_ok=True)

    cfg = {
        "work_dir": str(run_root),
        "config_dir": str(tmp_path),
        # Some callers provide the workspace root here; keep it consistent.
        "workspace": {"root": str(run_root)},
    }

    import scorpio_pipe.preflight as pf

    def _fake_ensure_preflight(_cfg, _run_root, **_kw):
        # Ensure the report path exists so error messages are stable.
        from pathlib import Path

        rr = Path(_run_root)
        (rr / "manifest").mkdir(parents=True, exist_ok=True)
        (rr / "manifest" / "preflight_report.json").write_text(
            '{"schema":"scorpio-pipe.preflight-report.v1","status":"ERROR","summary":{},"reasons":[]}'
            "\n",
            encoding="utf-8",
        )
        return {"status": "ERROR"}

    monkeypatch.setattr(pf, "ensure_preflight", _fake_ensure_preflight)

    from scorpio_pipe.pipeline.engine import run_sequence

    import pytest

    with pytest.raises(RuntimeError):
        run_sequence(cfg, ["manifest"], resume=False, force=False)
