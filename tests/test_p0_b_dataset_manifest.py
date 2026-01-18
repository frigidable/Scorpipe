from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scorpio_pipe.dataset.builder import FrameRecord, build_dataset_manifest_from_records
from scorpio_pipe.dataset.classify import classify_frame
from scorpio_pipe.instruments import FrameMeta, ReadoutKey


def dt0() -> datetime:
    return datetime(2026, 1, 12, 0, 0, 0, tzinfo=timezone.utc)


def mk_meta(
    *,
    imagetyp: str,
    when: datetime,
    object_name: str = "OBJ1",
    instrument: str = "SCORPIO1",
    mode: str = "Spectra",
    disperser: str = "VPHG1200",
    slit_width_arcsec: float = 1.0,
    slit_pos: float = 0.0,
    bin_x: int = 1,
    bin_y: int = 2,
    naxis1: int = 2048,
    naxis2: int = 4096,
    node: str = "A",
    rate: float = 145.0,
    gain: float = 0.90,
) -> FrameMeta:
    return FrameMeta(
        instrument=instrument,
        mode=mode,
        imagetyp=imagetyp,
        disperser=disperser,
        slit_width_arcsec=slit_width_arcsec,
        slit_pos=slit_pos,
        binning_x=bin_x,
        binning_y=bin_y,
        naxis1=naxis1,
        naxis2=naxis2,
        readout_key=ReadoutKey(node=node, rate=rate, gain=gain),
        date_time_utc=when,
        object_name=object_name,
    )


def rec(frame_id: str, meta: FrameMeta, *, sperange: str | None = None) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        path=f"{frame_id}.fits",
        meta=meta,
        frame_class=classify_frame(meta),
        sperange=sperange,
        sha256=None,
        size_bytes=None,
    )


def _single_match(man):
    assert len(man.science_sets) == 1
    assert len(man.matches) == 1
    return man.matches[0]


def test_p0_b_bias_must_match_readout_overrides_time_closeness():
    t0 = dt0()

    # Science series: mid-time at t0 + 30 min
    r_s1 = rec("SCI1", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=0)))
    r_s2 = rec("SCI2", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=60)))

    # Biases: closer one has wrong gain, further one is correct.
    r_b_good = rec("BIAS_OK", mk_meta(imagetyp="bias", when=t0 - timedelta(hours=2), gain=0.90))
    r_b_bad = rec("BIAS_BAD", mk_meta(imagetyp="bias", when=t0 - timedelta(hours=1), gain=1.10))

    # Provide at least one valid flat/arc to avoid unrelated hard-match failures.
    r_f = rec("FLAT1", mk_meta(imagetyp="flat", when=t0 - timedelta(minutes=10)))
    r_a = rec("ARC1", mk_meta(imagetyp="neon", when=t0 - timedelta(minutes=5)))

    man = build_dataset_manifest_from_records(
        [r_s1, r_s2, r_b_good, r_b_bad, r_f, r_a],
        pipeline_version="6.0.2",
        data_dir="/data/12_01_2026",
    )

    m = _single_match(man)
    assert m.bias_id == "BIAS_OK"
    assert m.flat_id == "FLAT1"
    assert m.flat_ids is not None
    assert set(m.flat_ids) == {"FLAT1"}
    assert m.arc_id == "ARC1"

    codes = {w.code for w in (man.warnings or [])}
    assert "BIAS_NO_HARD_MATCH" not in codes
    assert "BIAS_POOL_EMPTY" not in codes


def test_p0_b_no_hard_match_bias_emits_error_and_none():
    t0 = dt0()

    r_s1 = rec("SCI1", mk_meta(imagetyp="obj", when=t0))
    r_s2 = rec("SCI2", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=60)))

    # Only wrong-gain bias exists.
    r_b_bad = rec("BIAS_BAD", mk_meta(imagetyp="bias", when=t0 - timedelta(hours=1), gain=1.10))

    # Valid flat/arc exist.
    r_f = rec("FLAT1", mk_meta(imagetyp="flat", when=t0 - timedelta(minutes=10)))
    r_a = rec("ARC1", mk_meta(imagetyp="neon", when=t0 - timedelta(minutes=5)))

    man = build_dataset_manifest_from_records(
        [r_s1, r_s2, r_b_bad, r_f, r_a],
        pipeline_version="6.0.2",
    )

    m = _single_match(man)
    assert m.bias_id is None

    # Must emit explicit ERROR.
    w = [w for w in (man.warnings or []) if w.code == "BIAS_NO_HARD_MATCH"]
    assert len(w) == 1
    assert w[0].severity == "ERROR"


def test_p0_b_soft_match_picks_nearest_valid_flat_and_rejects_wrong_disperser():
    t0 = dt0()

    # Science mid-time at t0 + 30 min
    r_s1 = rec("SCI1", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=0)), sperange="400-700")
    r_s2 = rec("SCI2", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=60)), sperange="400-700")

    # Valid bias/arc so we focus on flat selection.
    r_b = rec("BIAS_OK", mk_meta(imagetyp="bias", when=t0 - timedelta(hours=2)))
    r_a = rec("ARC1", mk_meta(imagetyp="neon", when=t0 - timedelta(minutes=5)))

    # Flats: one earlier (20 min away), one later (10 min away) -> choose later.
    r_f_early = rec("FLAT_EARLY", mk_meta(imagetyp="flat", when=t0 + timedelta(minutes=10)))
    r_f_late = rec("FLAT_LATE", mk_meta(imagetyp="flat", when=t0 + timedelta(minutes=40)))

    # Wrong disperser flat exactly at mid-time: must be rejected by hard-match.
    r_f_wrong = rec(
        "FLAT_WRONG_DISP",
        mk_meta(imagetyp="flat", when=t0 + timedelta(minutes=30), disperser="VPHG550"),
    )

    man = build_dataset_manifest_from_records(
        [r_s1, r_s2, r_b, r_a, r_f_early, r_f_late, r_f_wrong],
        pipeline_version="6.0.2",
    )

    m = _single_match(man)
    assert m.flat_id == "FLAT_LATE"  # representative closest in time
    assert m.flat_ids is not None
    assert set(m.flat_ids) == {"FLAT_EARLY", "FLAT_LATE"}
    assert m.flat_meta is not None
    assert m.flat_meta.abs_dt_s is not None
    # 10 minutes = 600 seconds
    assert abs(m.flat_meta.abs_dt_s - 600.0) < 1e-6


def test_p0_f_manifest_flat_readout_mismatch_reason_is_recorded_in_flat_meta():
    t0 = dt0()

    # Science readout: gain/rate = 0.90/145
    r_s1 = rec("SCI1", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=0), gain=0.90, rate=145.0))
    r_s2 = rec("SCI2", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=60), gain=0.90, rate=145.0))

    # Provide valid bias/arc.
    r_b = rec("BIAS_OK", mk_meta(imagetyp="bias", when=t0 - timedelta(hours=2), gain=0.90, rate=145.0))
    r_a = rec("ARC1", mk_meta(imagetyp="neon", when=t0 - timedelta(minutes=5), gain=0.90, rate=145.0))

    # Only flat available has *different* gain/rate (node matches), which is allowed for flats.
    r_f = rec("FLAT_MISMATCH", mk_meta(imagetyp="flat", when=t0 + timedelta(minutes=10), gain=1.10, rate=185.0))

    man = build_dataset_manifest_from_records(
        [r_s1, r_s2, r_b, r_a, r_f],
        pipeline_version="6.0.8",
        data_dir="/data/12_01_2026",
        flat_allow_readout_diff=True,
    )

    m = _single_match(man)
    assert m.flat_id == "FLAT_MISMATCH"
    assert m.flat_meta is not None

    fm = m.flat_meta.model_dump()
    assert fm.get("readout_policy") == "prefer_same_readout_but_allow"
    assert fm.get("selected_readout_match") is False
    assert fm.get("n_same_readout_flat") == 0
    assert "selection_reason" in fm

    codes = {w.code for w in (man.warnings or [])}
    assert "FLAT_READOUT_MISMATCH_ALLOWED" in codes



def test_p0_g_manifest_arc_readout_mismatch_reason_is_recorded_in_arc_meta():
    t0 = dt0()

    # Science readout: gain/rate = 0.90/145
    r_s1 = rec("SCI1", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=0), gain=0.90, rate=145.0))
    r_s2 = rec("SCI2", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=60), gain=0.90, rate=145.0))

    # Provide valid bias/flat.
    r_b = rec("BIAS_OK", mk_meta(imagetyp="bias", when=t0 - timedelta(hours=2), gain=0.90, rate=145.0))
    r_f = rec("FLAT1", mk_meta(imagetyp="flat", when=t0 - timedelta(minutes=10), gain=0.90, rate=145.0))

    # Only arc available has *different* gain/rate (node matches), which is allowed for arcs in P0-G.
    r_a = rec("ARC_MISMATCH", mk_meta(imagetyp="neon", when=t0 - timedelta(minutes=5), gain=1.10, rate=185.0))

    man = build_dataset_manifest_from_records(
        [r_s1, r_s2, r_b, r_f, r_a],
        pipeline_version="6.0.9",
        data_dir="/data/12_01_2026",
        arc_allow_readout_diff=True,
    )

    m = _single_match(man)
    assert m.arc_id == "ARC_MISMATCH"
    assert m.arc_meta is not None

    am = m.arc_meta.model_dump()
    assert am.get("readout_policy") == "prefer_same_readout_but_allow"
    assert am.get("selected_readout_match") is False
    assert am.get("n_same_readout_arc") == 0
    assert "selection_reason" in am

    codes = {w.code for w in (man.warnings or [])}
    assert "ARC_READOUT_MISMATCH_ALLOWED" in codes


def test_p0_g_arc_prefers_same_readout_when_available_over_time_closeness():
    t0 = dt0()

    # Science readout: gain/rate = 0.90/145, mid-time is t0+30m
    r_s1 = rec("SCI1", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=0), gain=0.90, rate=145.0))
    r_s2 = rec("SCI2", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=60), gain=0.90, rate=145.0))

    # Provide valid bias/flat.
    r_b = rec("BIAS_OK", mk_meta(imagetyp="bias", when=t0 - timedelta(hours=2), gain=0.90, rate=145.0))
    r_f = rec("FLAT1", mk_meta(imagetyp="flat", when=t0 - timedelta(minutes=10), gain=0.90, rate=145.0))

    # Arc very close in time but wrong readout, and a further arc with matching readout.
    r_a_close_bad = rec(
        "ARC_CLOSE_BAD",
        mk_meta(imagetyp="neon", when=t0 + timedelta(minutes=30), gain=1.10, rate=185.0),
    )
    r_a_same = rec(
        "ARC_SAME",
        mk_meta(imagetyp="neon", when=t0 + timedelta(minutes=20), gain=0.90, rate=145.0),
    )

    man = build_dataset_manifest_from_records(
        [r_s1, r_s2, r_b, r_f, r_a_close_bad, r_a_same],
        pipeline_version="6.0.9",
        arc_allow_readout_diff=True,
    )

    m = _single_match(man)
    assert m.arc_id == "ARC_SAME"  # prefer same readout when available
    assert m.arc_meta is not None

    am = m.arc_meta.model_dump()
    assert am.get("selected_readout_match") is True
    assert am.get("n_same_readout_arc") >= 1
    assert "selection_reason" in am

    codes = {w.code for w in (man.warnings or [])}
    assert "ARC_READOUT_MISMATCH_ALLOWED" not in codes


def test_p0_b4_suboptimal_match_warning_and_meta_fields_for_flat_readout_mismatch():
    t0 = dt0()

    r_s1 = rec("SCI1", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=0), gain=0.90, rate=145.0))
    r_s2 = rec("SCI2", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=60), gain=0.90, rate=145.0))

    r_b = rec("BIAS_OK", mk_meta(imagetyp="bias", when=t0 - timedelta(hours=2), gain=0.90, rate=145.0))
    r_a = rec("ARC1", mk_meta(imagetyp="neon", when=t0 - timedelta(minutes=5), gain=0.90, rate=145.0))

    # Only flat has different readout; allowed.
    r_f = rec("FLAT_MISMATCH", mk_meta(imagetyp="flat", when=t0 + timedelta(minutes=10), gain=1.10, rate=185.0))

    man = build_dataset_manifest_from_records(
        [r_s1, r_s2, r_b, r_a, r_f],
        pipeline_version="6.0.23",
        flat_allow_readout_diff=True,
    )

    m = _single_match(man)
    assert m.flat_id == "FLAT_MISMATCH"
    assert m.flat_meta is not None
    assert m.flat_meta.match_reason is not None
    assert isinstance(m.flat_meta.qc_deltas, dict)
    assert m.flat_meta.qc_deltas.get("selected_readout_match") is False

    codes = {w.code for w in (man.warnings or [])}
    assert "CALIB_SUBOPTIMAL_MATCH" in codes



def test_p0_b4_suboptimal_match_warning_and_meta_fields_for_arc_readout_mismatch():
    t0 = dt0()

    r_s1 = rec("SCI1", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=0), gain=0.90, rate=145.0))
    r_s2 = rec("SCI2", mk_meta(imagetyp="obj", when=t0 + timedelta(minutes=60), gain=0.90, rate=145.0))

    r_b = rec("BIAS_OK", mk_meta(imagetyp="bias", when=t0 - timedelta(hours=2), gain=0.90, rate=145.0))
    r_f = rec("FLAT1", mk_meta(imagetyp="flat", when=t0 - timedelta(minutes=10), gain=0.90, rate=145.0))

    # Only arc has different readout; allowed.
    r_a = rec("ARC_MISMATCH", mk_meta(imagetyp="neon", when=t0 - timedelta(minutes=5), gain=1.10, rate=185.0))

    man = build_dataset_manifest_from_records(
        [r_s1, r_s2, r_b, r_f, r_a],
        pipeline_version="6.0.23",
        arc_allow_readout_diff=True,
    )

    m = _single_match(man)
    assert m.arc_id == "ARC_MISMATCH"
    assert m.arc_meta is not None
    assert m.arc_meta.match_reason is not None
    assert isinstance(m.arc_meta.qc_deltas, dict)
    assert m.arc_meta.qc_deltas.get("selected_readout_match") is False

    codes = {w.code for w in (man.warnings or [])}
    assert "CALIB_SUBOPTIMAL_MATCH" in codes
