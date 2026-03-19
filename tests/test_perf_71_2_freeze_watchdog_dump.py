from __future__ import annotations

from pathlib import Path


def test_recent_ui_actions_tracks_begin_end_and_inflight() -> None:
    from scorpio_pipe.ui import slow_action_profiler as sap

    sap._reset_recent_ui_actions_for_tests()

    with sap.profile_ui_action("alpha", threshold_ms=999999):
        pass

    active = sap.get_active_ui_actions()
    assert active == []

    events = sap.get_recent_ui_actions(limit=10)
    assert any(ev["name"] == "alpha" and ev["phase"] == "begin" for ev in events)
    assert any(ev["name"] == "alpha" and ev["phase"] == "end" for ev in events)

    cm = sap.profile_ui_action("beta", threshold_ms=999999)
    cm.__enter__()
    try:
        active = sap.get_active_ui_actions(limit=10)
        assert any(ev["name"] == "beta" and ev["phase"] == "begin" for ev in active)

        events = sap.get_recent_ui_actions(limit=10)
        assert any(ev["name"] == "beta" and ev["phase"] == "begin" for ev in events)
        assert not any(ev["name"] == "beta" and ev["phase"] == "end" for ev in events)
    finally:
        cm.__exit__(None, None, None)

    active = sap.get_active_ui_actions(limit=10)
    assert not any(ev["name"] == "beta" for ev in active)
    events = sap.get_recent_ui_actions(limit=10)
    assert any(ev["name"] == "beta" and ev["phase"] == "end" for ev in events)


def test_freeze_watchdog_dump_contains_stack_jobs_backlog_and_recent_actions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import sys
    import types

    from scorpio_pipe.ui import slow_action_profiler as sap
    from scorpio_pipe.ui.freeze_watchdog import FreezeWatchdog

    sap._reset_recent_ui_actions_for_tests()

    fake_async_worker = types.ModuleType("scorpio_pipe.ui.async_worker")
    fake_async_worker.snapshot_active_qthreads = lambda: [
        "QThreadPool max=4 active=1",
        "job_registry active=2 running=1 queued=1 cancelled_active=0 stale_active=0 queue_limit=200 deduped_total=3 queue_evicted_total=1",
        "job id=41 name='inspect_dataset' key='inspect:/data' age_s=2.50 running=True queued=False stale=False cancelled=False",
        "job id=42 name='fits_preview' key='preview:/frame.fits' age_s=1.20 running=False queued=True stale=False cancelled=False",
    ]
    fake_qt_log = types.ModuleType("scorpio_pipe.ui.qt_log")
    fake_qt_log.get_backlog_snapshot = lambda: {"lines": 123, "chars": 4567}

    monkeypatch.setitem(sys.modules, "scorpio_pipe.ui.async_worker", fake_async_worker)
    monkeypatch.setitem(sys.modules, "scorpio_pipe.ui.qt_log", fake_qt_log)

    cm = sap.profile_ui_action("Open heavy ROI dialog", threshold_ms=999999)
    cm.__enter__()
    try:
        wd = FreezeWatchdog(dump_path=tmp_path / "ui_freeze_dump.txt")
        wd._qt_threadpool_active = 1
        wd._qt_threadpool_max = 4
        wd._write_dump(stall_seconds=2.75)
    finally:
        cm.__exit__(None, None, None)

    text = (tmp_path / "ui_freeze_dump.txt").read_text(encoding="utf-8")
    assert "-- Freeze reason hints --" in text
    assert "-- Main thread stack --" in text
    assert "test_freeze_watchdog_dump_contains_stack_jobs_backlog_and_recent_actions" in text
    assert "ui_log_backlog: lines=123 chars=4567" in text
    assert "-- Active QRunnable jobs (Scorpipe-managed) --" in text
    assert "job id=41 name='inspect_dataset'" in text
    assert "job id=42 name='fits_preview'" in text
    assert "-- UI actions currently in flight --" in text
    assert "Open heavy ROI dialog" in text
    assert "-- Last 50 profiled UI actions --" in text
