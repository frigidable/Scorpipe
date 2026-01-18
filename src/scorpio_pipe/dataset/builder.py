from __future__ import annotations

"""Dataset manifest builder (P0-B1).

Public API
----------
``build_dataset_manifest``
    Scan a raw-night directory of FITS files (Astropy required at runtime) and
    write ``dataset_manifest.json``.

``build_dataset_manifest_from_records``
    Build a manifest from pre-parsed records (no Astropy dependency); this is
    used by unit tests and can be used by external tooling.

The builder is conservative: it never "guesses" incompatible calibrations.
All ambiguity and missing-calibration situations are made explicit in the
manifest warnings.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scorpio_pipe.dataset.classify import FrameClass, classify_frame, is_longslit_mode
from scorpio_pipe.dataset.manifest import (
    CalibrationEntry,
    CalibrationPools,
    ConfigKeyModel,
    DatasetManifest,
    FrameIndexEntry,
    GeometryKeyModel,
    ManifestWarning,
    MatchEntry,
    MatchSelectionMeta,
    ReadoutKeyModel,
    ScienceSet,
    SpectroKeyModel,
)
from scorpio_pipe.dataset.match import (
    Candidate,
    ConfigKey,
    GeometryKey,
    SpectroKey,
    hard_compatible,
    make_readout_key,
    select_best,
    select_flat_set,
)
from scorpio_pipe.metadata import HeaderContractError, FrameMeta, parse_frame_meta


_FITS_SUFFIXES = (".fits", ".fit", ".fts", ".fz")


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _norm_object(s: str) -> str:
    return "".join(ch for ch in (s or "").strip().upper() if ch.isalnum() or ch in ("-", "_"))


def _read_file_sha256(p: Path, *, chunk: int = 1024 * 1024) -> str:
    h = sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _guess_night_id(data_dir: Path, records: Sequence["FrameRecord"]) -> str | None:
    # Convenience: infer dd_mm_yyyy from directory name.
    name = data_dir.name
    import re

    if re.match(r"^\d{2}_\d{2}_\d{4}$", name):
        return name

    # Otherwise attempt to use the most common DATE-OBS from records.
    try:
        if records:
            # Date part in UTC
            vals = [r.meta.date_time_utc.date().isoformat() for r in records if r.meta and r.meta.date_time_utc]
            if vals:
                from collections import Counter

                d = Counter(vals).most_common(1)[0][0]
                yyyy, mm, dd = d.split("-")
                return f"{dd}_{mm}_{yyyy}"
    except Exception:
        pass
    return None


@dataclass(frozen=True)
class FrameRecord:
    """Internal representation of a scanned frame."""

    frame_id: str
    path: str  # POSIX, relative to data_dir where possible
    meta: FrameMeta
    frame_class: FrameClass
    sperange: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None

    @property
    def is_longslit(self) -> bool:
        return is_longslit_mode(self.meta)


def scan_fits_directory(
    data_dir: Path,
    *,
    recursive: bool = True,
    max_files: int | None = None,
    include_hashes: bool = False,
    exclude_paths: Sequence[str] | None = None,
) -> tuple[list[FrameRecord], list[ManifestWarning]]:
    """Scan FITS files under *data_dir* and return FrameRecord list.

    Requires Astropy at runtime (imported lazily).
    """

    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Astropy is required to scan FITS headers. Install with: pip install astropy"
        ) from e

    data_dir = Path(data_dir).expanduser().resolve()
    ex_set: set[str] = set()
    if exclude_paths:
        for x in exclude_paths:
            try:
                p = Path(str(x)).expanduser()
                p_res = p.resolve() if p.is_absolute() else (data_dir / p).resolve()
                ex_set.add(str(p_res))
            except Exception:
                continue
    paths: list[Path] = []
    if recursive:
        for suf in _FITS_SUFFIXES:
            paths.extend(sorted(data_dir.rglob(f"*{suf}")))
    else:
        for suf in _FITS_SUFFIXES:
            paths.extend(sorted(data_dir.glob(f"*{suf}")))

    if max_files is not None:
        paths = paths[: max(0, int(max_files))]

    recs: list[FrameRecord] = []
    warns: list[ManifestWarning] = []

    included_idx = 0
    for p in paths:
        # Exclude at scan stage for reproducibility (keeps frame_id stable for included frames only).
        if ex_set and str(p.resolve()) in ex_set:
            continue
        included_idx += 1
        frame_id = f"F{included_idx:05d}"
        rel = None
        try:
            rel = p.relative_to(data_dir)
            rel_s = rel.as_posix()
        except Exception:
            rel_s = p.as_posix()

        try:
            hdr = fits.getheader(p, 0, memmap=False)
        except Exception as e:
            warns.append(
                ManifestWarning(
                    code="FITS_OPEN_ERROR",
                    severity="ERROR",
                    message="Failed to open FITS header.",
                    context={"path": rel_s, "error": str(e)},
                )
            )
            continue

        try:
            meta = parse_frame_meta(hdr)
            # FrameMeta is the single source of truth; stamp deterministic frame_id here.
            meta = meta.with_frame_id(frame_id)
        except HeaderContractError as e:
            warns.append(
                ManifestWarning(
                    code="HEADER_CONTRACT_ERROR",
                    severity="ERROR",
                    message=str(e),
                    context={"path": rel_s},
                )
            )
            continue
        except Exception as e:
            warns.append(
                ManifestWarning(
                    code="META_PARSE_ERROR",
                    severity="ERROR",
                    message="Unexpected error while parsing header.",
                    context={"path": rel_s, "error": str(e)},
                )
            )
            continue

        # P0-B2 QA: record optional-missing and fallback usage explicitly.
        if getattr(meta, "meta_missing_optional", None):
            warns.append(
                ManifestWarning(
                    code="META_MISSING_OPTIONAL",
                    severity="WARN",
                    message="Some optional metadata keys are missing.",
                    context={"path": rel_s, "missing": list(meta.meta_missing_optional)},
                )
            )
        if getattr(meta, "meta_fallback_used", None):
            warns.append(
                ManifestWarning(
                    code="META_FALLBACK_USED",
                    severity="WARN",
                    message="Metadata fallback values were applied.",
                    context={"path": rel_s, "fallback": dict(meta.meta_fallback_used)},
                )
            )
        fc = classify_frame(meta)
        size = None
        try:
            size = p.stat().st_size
        except Exception:
            size = None

        sh = _read_file_sha256(p) if include_hashes else None
        sper = meta.sperange

        recs.append(
            FrameRecord(
                frame_id=frame_id,
                path=rel_s,
                meta=meta,
                frame_class=fc,
                sperange=sper,
                sha256=sh,
                size_bytes=size,
            )
        )

    return recs, warns


def _geometry_key(meta: FrameMeta) -> GeometryKey:
    return GeometryKey(
        naxis1=int(meta.naxis1),
        naxis2=int(meta.naxis2),
        bin_x=int(meta.binning_x),
        bin_y=int(meta.binning_y),
    )


def _spectro_key(meta: FrameMeta) -> SpectroKey:
    return SpectroKey(
        mode=str(meta.mode or "").strip(),
        disperser=str(meta.disperser or "").strip(),
        slit_width_key=str(meta.slit_width_key),
        slit_width_arcsec=float(meta.slit_width_arcsec),
    )


def _config_key(meta: FrameMeta) -> ConfigKey:
    return ConfigKey(
        instrument=str(meta.instrument),
        geometry=_geometry_key(meta),
        readout=make_readout_key(meta.readout_key.node, meta.readout_key.rate, meta.readout_key.gain),
        spectro=_spectro_key(meta),
    )


def build_dataset_manifest_from_records(
    records: Sequence[FrameRecord],
    *,
    pipeline_version: str,
    data_dir: str | None = None,
    night_id: str | None = None,
    include_frame_index: bool = True,
    flat_time_window_s: float | None = 14400.0,
    flat_allow_readout_diff: bool = True,
    arc_allow_readout_diff: bool | None = None,
) -> DatasetManifest:
    """Build a dataset manifest from records.

    This function has **no Astropy dependency** and can be unit-tested.
    """

    # Deterministic ordering: avoid filesystem-dependent scan order leaking into
    # dataset_manifest.json (P0-B6).
    records = sorted(list(records), key=lambda r: (str(r.frame_id), str(r.path)))

    if night_id is None and data_dir is not None:
        night_id = _guess_night_id(Path(data_dir), records)

    scan_warnings: list[ManifestWarning] = []

    # Build calibration pools
    pools = CalibrationPools()
    calib_index: dict[str, CalibrationEntry] = {}

    def _mk_readout_model(meta: FrameMeta) -> ReadoutKeyModel:
        r = make_readout_key(meta.readout_key.node, meta.readout_key.rate, meta.readout_key.gain)
        return ReadoutKeyModel(node=r.node, rate=r.rate, gain=r.gain)

    def _mk_geom_model(meta: FrameMeta) -> GeometryKeyModel:
        g = _geometry_key(meta)
        return GeometryKeyModel(naxis1=g.naxis1, naxis2=g.naxis2, bin_x=g.bin_x, bin_y=g.bin_y)

    def _mk_spec_model(meta: FrameMeta) -> SpectroKeyModel:
        s = _spectro_key(meta)
        return SpectroKeyModel(mode=s.mode or "", disperser=s.disperser, slit_width_arcsec=s.slit_width_arcsec, slit_width_key=s.slit_width_key)

    # Optional full frame index (for debugging / GUI)
    frame_index: list[FrameIndexEntry] = []

    for r in records:
        meta = r.meta
        g = _mk_geom_model(meta)
        ro = _mk_readout_model(meta)
        spec: SpectroKeyModel | None = None
        if r.frame_class in (FrameClass.ARC, FrameClass.FLAT, FrameClass.SCIENCE):
            spec = _mk_spec_model(meta)

        if include_frame_index:
            frame_index.append(
                FrameIndexEntry(
                    frame_id=r.frame_id,
                    path=r.path,
                    kind=str(r.frame_class.value),
                    instrument=str(meta.instrument),
                    date_time_utc=_iso(meta.date_time_utc),
                    object=str(meta.object_name or ""),
                    mode=str(meta.mode or ""),
                    disperser=str(meta.disperser or ""),
                    slit_width_key=str(meta.slit_width_key),
                    slit_pos=meta.slit_pos,
                    sperange=r.sperange,
                    geometry=g,
                    readout=ro,
                    spectro=spec,
                    sha256=r.sha256,
                    size_bytes=r.size_bytes,
                )
            )

        # Calibration pools
        if r.frame_class == FrameClass.BIAS:
            ce = CalibrationEntry(
                calib_id=r.frame_id,
                path=r.path,
                kind="bias",
                instrument=str(meta.instrument),
                date_time_utc=_iso(meta.date_time_utc),
                geometry=g,
                readout=ro,
                spectro=None,
                sperange=r.sperange,
                slit_pos=meta.slit_pos,
                sha256=r.sha256,
                size_bytes=r.size_bytes,
            )
            pools.bias.append(ce)
            calib_index[ce.calib_id] = ce
        elif r.frame_class == FrameClass.FLAT:
            ce = CalibrationEntry(
                calib_id=r.frame_id,
                path=r.path,
                kind="flat",
                instrument=str(meta.instrument),
                date_time_utc=_iso(meta.date_time_utc),
                geometry=g,
                readout=ro,
                spectro=_mk_spec_model(meta),
                sperange=r.sperange,
                slit_pos=meta.slit_pos,
                sha256=r.sha256,
                size_bytes=r.size_bytes,
            )
            pools.flat.append(ce)
            calib_index[ce.calib_id] = ce
        elif r.frame_class == FrameClass.ARC:
            ce = CalibrationEntry(
                calib_id=r.frame_id,
                path=r.path,
                kind="arc",
                instrument=str(meta.instrument),
                date_time_utc=_iso(meta.date_time_utc),
                geometry=g,
                readout=ro,
                spectro=_mk_spec_model(meta),
                sperange=r.sperange,
                slit_pos=meta.slit_pos,
                sha256=r.sha256,
                size_bytes=r.size_bytes,
            )
            pools.arc.append(ce)
            calib_index[ce.calib_id] = ce

    # Stable ordering for deterministic JSON output (P0-B6).
    if include_frame_index:
        frame_index.sort(key=lambda e: (str(e.frame_id), str(e.path)))
    try:
        pools.bias.sort(key=lambda c: (str(c.instrument), str(c.date_time_utc), str(c.calib_id)))
        pools.flat.sort(key=lambda c: (str(c.instrument), str(c.date_time_utc), str(c.calib_id)))
        pools.arc.sort(key=lambda c: (str(c.instrument), str(c.date_time_utc), str(c.calib_id)))
    except Exception:
        pass

    # Build science sets: only long-slit science frames
    science_frames = [r for r in records if (r.frame_class == FrameClass.SCIENCE and r.is_longslit)]

    # Group by (object_norm, config_key)
    # NOTE: ConfigKey is a frozen dataclass and therefore hashable.
    groups: dict[tuple[str, ConfigKey], list[FrameRecord]] = {}
    for r in science_frames:
        objn = _norm_object(r.meta.object_name)
        ck = _config_key(r.meta)
        k = (objn, ck)
        groups.setdefault(k, []).append(r)

    # Stable ordering
    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: (kv[0][0], min(x.meta.date_time_utc for x in kv[1])),
    )

    science_sets: list[ScienceSet] = []
    config_lookup: dict[str, ConfigKey] = {}

    for si, ((_objn, _ck), frames) in enumerate(sorted_groups, start=1):
        frames_sorted = sorted(frames, key=lambda r: r.meta.date_time_utc)
        set_id = f"S{si:03d}"
        obj = frames_sorted[0].meta.object_name
        objn = _norm_object(obj)
        # Representative config key from first frame
        ck = _config_key(frames_sorted[0].meta)
        config_lookup[set_id] = ck
        # Timing
        start = frames_sorted[0].meta.date_time_utc
        end = frames_sorted[-1].meta.date_time_utc
        mid = start + (end - start) / 2
        # Soft-match helpers
        sper = None
        for r in frames_sorted:
            if r.sperange:
                sper = r.sperange
                break
        slitpos = None
        vals = [r.meta.slit_pos for r in frames_sorted if r.meta.slit_pos is not None]
        if vals:
            slitpos = float(sorted(vals)[len(vals) // 2])

        ss = ScienceSet(
            science_set_id=set_id,
            object=str(obj),
            object_norm=objn,
            frames=[r.frame_id for r in frames_sorted],
            config=ConfigKeyModel(
                instrument=ck.instrument,
                geometry=GeometryKeyModel(
                    naxis1=ck.geometry.naxis1,
                    naxis2=ck.geometry.naxis2,
                    bin_x=ck.geometry.bin_x,
                    bin_y=ck.geometry.bin_y,
                ),
                readout=ReadoutKeyModel(
                    node=ck.readout.node,
                    rate=ck.readout.rate,
                    gain=ck.readout.gain,
                ),
                spectro=SpectroKeyModel(
                    mode=ck.spectro.mode,
                    disperser=ck.spectro.disperser,
                    slit_width_arcsec=ck.spectro.slit_width_arcsec,
                    slit_width_key=ck.spectro.slit_width_key,
                ),
            ),
            start_utc=_iso(start),
            end_utc=_iso(end),
            mid_utc=_iso(mid),
            n_frames=len(frames_sorted),
            sperange=sper,
            slit_pos=slitpos,
        )
        science_sets.append(ss)

    # Convert pools to Candidate lists
    cand_bias = [
        Candidate(
            calib_id=c.calib_id,
            kind="bias",
            date_time_utc=datetime.fromisoformat(c.date_time_utc.replace("Z", "+00:00")),
            instrument=c.instrument,
            geometry=GeometryKey(c.geometry.naxis1, c.geometry.naxis2, c.geometry.bin_x, c.geometry.bin_y),
            readout=make_readout_key(c.readout.node, c.readout.rate, c.readout.gain),
            spectro=None,
            sperange=c.sperange,
            slit_pos=c.slit_pos,
        )
        for c in pools.bias
    ]
    cand_flat = [
        Candidate(
            calib_id=c.calib_id,
            kind="flat",
            date_time_utc=datetime.fromisoformat(c.date_time_utc.replace("Z", "+00:00")),
            instrument=c.instrument,
            geometry=GeometryKey(c.geometry.naxis1, c.geometry.naxis2, c.geometry.bin_x, c.geometry.bin_y),
            readout=make_readout_key(c.readout.node, c.readout.rate, c.readout.gain),
            spectro=(
                SpectroKey(
                    mode=c.spectro.mode,
                    disperser=c.spectro.disperser,
                    slit_width_key=c.spectro.slit_width_key,
                    slit_width_arcsec=c.spectro.slit_width_arcsec,
                )
                if c.spectro is not None
                else None
            ),
            sperange=c.sperange,
            slit_pos=c.slit_pos,
        )
        for c in pools.flat
    ]
    cand_arc = [
        Candidate(
            calib_id=c.calib_id,
            kind="arc",
            date_time_utc=datetime.fromisoformat(c.date_time_utc.replace("Z", "+00:00")),
            instrument=c.instrument,
            geometry=GeometryKey(c.geometry.naxis1, c.geometry.naxis2, c.geometry.bin_x, c.geometry.bin_y),
            readout=make_readout_key(c.readout.node, c.readout.rate, c.readout.gain),
            spectro=(
                SpectroKey(
                    mode=c.spectro.mode,
                    disperser=c.spectro.disperser,
                    slit_width_key=c.spectro.slit_width_key,
                    slit_width_arcsec=c.spectro.slit_width_arcsec,
                )
                if c.spectro is not None
                else None
            ),
            sperange=c.sperange,
            slit_pos=c.slit_pos,
        )
        for c in pools.arc
    ]

    matches: list[MatchEntry] = []
    warnings: list[ManifestWarning] = []

    for ss in science_sets:
        sk = config_lookup[ss.science_set_id]
        mid = datetime.fromisoformat(ss.mid_utc.replace("Z", "+00:00"))

        # bias
        sel_b, wb = select_best(
            kind="bias",
            science_key=sk,
            science_mid=mid,
            science_sperange=ss.sperange,
            science_slitpos=ss.slit_pos,
            pool=cand_bias,
        )
        # flat
        sel_f, wf = select_best(
            kind="flat",
            science_key=sk,
            science_mid=mid,
            science_sperange=ss.sperange,
            science_slitpos=ss.slit_pos,
            pool=cand_flat,
            allow_readout_diff=flat_allow_readout_diff,
        )

        flat_ids, wf_set = select_flat_set(
            science_key=sk,
            science_mid=mid,
            science_sperange=ss.sperange,
            science_slitpos=ss.slit_pos,
            pool=cand_flat,
            max_abs_dt_s=flat_time_window_s,
            allow_readout_diff=flat_allow_readout_diff,
            fallback_best_id=sel_f.selected_id,
        )
        if (not flat_ids) and sel_f.selected_id:
            flat_ids = [str(sel_f.selected_id)]
        wf = [*wf, *wf_set]

        # arc
        # P0-G: allow readout gain/rate mismatch for arcs (NODE stays strict),
        # defaulting to allowed for long-slit SCORPIO/SCORPIO-2.
        arc_allow = arc_allow_readout_diff
        if arc_allow is None:
            inst = str(sk.instrument or "").strip().upper()
            arc_allow = inst in {"SCORPIO1", "SCORPIO2", "SCORPIO", "SCORPIO-2"}

        sel_a, wa = select_best(
            kind="arc",
            science_key=sk,
            science_mid=mid,
            science_sperange=ss.sperange,
            science_slitpos=ss.slit_pos,
            pool=cand_arc,
            allow_readout_diff=bool(arc_allow),
        )

        def _build_match_reason(kind: str, sel, extra: dict[str, Any] | None) -> str:
            # Deterministic ranking used by dataset.match.select_best.
            # Keep this string stable for provenance/QC.
            parts: list[str] = []
            if extra and extra.get("readout_policy") == "prefer_same_readout_but_allow":
                parts.append("prefer_same_readout")
            parts += ["abs_dt_s", "sperange_mismatch", "slitpos_diff", "calib_id"]
            return f"ranked_by:{','.join(parts)}"


        def _build_qc_deltas(sel, extra: dict[str, Any] | None) -> dict[str, Any]:
            qc: dict[str, Any] = {}
            # Explicit QC-only deltas (redundant with top-level fields, but convenient).
            if sel.abs_dt_s is not None:
                qc["abs_dt_s"] = float(sel.abs_dt_s)
            if sel.sperange_mismatch is not None:
                qc["sperange_mismatch"] = bool(sel.sperange_mismatch)
            if sel.slitpos_diff is not None:
                qc["slitpos_diff"] = float(sel.slitpos_diff)
            if extra:
                # Readout mismatch allowance is QC-only for some kinds.
                if "selected_readout_match" in extra:
                    qc["selected_readout_match"] = bool(extra.get("selected_readout_match"))
                for k in ("flat_set_n", "flat_set_in_time_window", "flat_time_window_s", "flat_set_readout_unique"):
                    if k in extra:
                        qc[k] = extra.get(k)
            return qc


        def _is_suboptimal_match(kind: str, sel, extra: dict[str, Any] | None, warn_codes: set[str]) -> tuple[bool, str]:
            # Suboptimal = hard-compatible, but with QC-only degradations.
            reasons: list[str] = []
            if sel.sperange_mismatch:
                reasons.append("sperange_mismatch")
            if sel.slitpos_diff is not None and float(sel.slitpos_diff) != 0.0:
                reasons.append("slitpos_diff")
            if extra and extra.get("readout_policy") == "prefer_same_readout_but_allow":
                if extra.get("selected_readout_match") is False:
                    reasons.append("readout_mismatch_allowed")
            if kind == "flat" and ("FLAT_NO_IN_TIME_WINDOW" in warn_codes):
                reasons.append("no_flat_in_time_window")
            # Leave bias alone: if it matches, it is hard-compatible by definition.
            return (len(reasons) > 0, ",".join(reasons) if reasons else "")


        def _meta(sel, kind: str, *, extra: dict[str, Any] | None = None, warn_codes: set[str] | None = None) -> MatchSelectionMeta:
            # MatchSelectionMeta is intentionally permissive (extra=allow).
            warn_codes = warn_codes or set()
            reason = _build_match_reason(kind, sel, extra)
            qc_deltas = _build_qc_deltas(sel, extra)
            meta = MatchSelectionMeta(
                n_pool=sel.n_pool,
                n_hard_compatible=sel.n_hard_compatible,
                abs_dt_s=sel.abs_dt_s,
                sperange_mismatch=sel.sperange_mismatch,
                slitpos_diff=sel.slitpos_diff,
                tie_n=sel.tie_n,
                tie_break=sel.tie_break,
                match_reason=reason,
                qc_deltas=qc_deltas,
                **(extra or {}),
            )

            subopt, sub_reason = _is_suboptimal_match(kind, sel, extra, warn_codes)
            if subopt and sel.selected_id is not None:
                warnings.append(
                    ManifestWarning(
                        code="CALIB_SUBOPTIMAL_MATCH",
                        severity="WARN",
                        message=(
                            f"Suboptimal {kind} association for science_set={ss.science_set_id}; reason={sub_reason}"
                        ),
                        context={
                            "science_set_id": ss.science_set_id,
                            "kind": kind,
                            "calib_id": str(sel.selected_id),
                            "reason": sub_reason,
                            "qc_deltas": qc_deltas,
                            "match_reason": reason,
                        },
                    )
                )

            return meta

        # P0-F: Add transparent readout reasoning for flats when readout
        # (gain/rate) mismatches are allowed.
        flat_meta_extra: dict[str, Any] = {}
        if flat_allow_readout_diff:
            hard_flat = [
                c
                for c in cand_flat
                if hard_compatible(sk, c, "flat", allow_readout_diff=True)
            ]
            same_flat = [c for c in hard_flat if c.readout == sk.readout]
            sel_c = None
            if sel_f.selected_id:
                for c in cand_flat:
                    if c.calib_id == sel_f.selected_id:
                        sel_c = c
                        break

            if sel_c is not None:
                readout_match = bool(sel_c.readout == sk.readout)
                flat_meta_extra.update(
                    {
                        "readout_policy": "prefer_same_readout_but_allow",
                        "science_readout": {
                            "node": sk.readout.node,
                            "rate": sk.readout.rate,
                            "gain": sk.readout.gain,
                        },
                        "selected_readout": {
                            "node": sel_c.readout.node,
                            "rate": sel_c.readout.rate,
                            "gain": sel_c.readout.gain,
                        },
                        "selected_readout_match": readout_match,
                        "n_hard_flat": len(hard_flat),
                        "n_same_readout_flat": len(same_flat),
                    }
                )
                if not readout_match:
                    flat_meta_extra["selection_reason"] = (
                        "No same-readout flats in the hard-compatible pool; "
                        "allowed readout mismatch and selected best by |Δtime| (then id)."
                    )
                else:
                    flat_meta_extra["selection_reason"] = (
                        "Preferred same-readout flats (available) and selected best by |Δtime| (then id)."
                    )

            # Also record whether the *flat set* mixes readouts.
            if flat_ids:
                sel_set = [c for c in cand_flat if c.calib_id in set(flat_ids)]
                if sel_set:
                    ro_set = {(c.readout.node, c.readout.rate, c.readout.gain) for c in sel_set}
                    flat_meta_extra["flat_set_readout_unique"] = len(ro_set)

        # P0-B4: explicit flat-set association context (QC-only)
        flat_meta_extra["flat_set_n"] = int(len(flat_ids or []))
        if flat_time_window_s is not None:
            flat_meta_extra["flat_time_window_s"] = float(flat_time_window_s)
            flat_meta_extra["flat_set_in_time_window"] = ("FLAT_NO_IN_TIME_WINDOW" not in {str(w.get("code")) for w in (wf or [])})
        # P0-G: transparent readout reasoning for arcs when readout (gain/rate)
        # mismatches are allowed.
        arc_meta_extra: dict[str, Any] = {}
        if bool(arc_allow):
            hard_arc = [
                c
                for c in cand_arc
                if hard_compatible(sk, c, "arc", allow_readout_diff=True)
            ]
            same_arc = [c for c in hard_arc if c.readout == sk.readout]
            sel_c = None
            if sel_a.selected_id:
                for c in cand_arc:
                    if c.calib_id == sel_a.selected_id:
                        sel_c = c
                        break

            if sel_c is not None:
                readout_match = bool(sel_c.readout == sk.readout)
                arc_meta_extra.update(
                    {
                        "readout_policy": "prefer_same_readout_but_allow",
                        "science_readout": {
                            "node": sk.readout.node,
                            "rate": sk.readout.rate,
                            "gain": sk.readout.gain,
                        },
                        "selected_readout": {
                            "node": sel_c.readout.node,
                            "rate": sel_c.readout.rate,
                            "gain": sel_c.readout.gain,
                        },
                        "selected_readout_match": readout_match,
                        "n_hard_arc": len(hard_arc),
                        "n_same_readout_arc": len(same_arc),
                    }
                )
                if not readout_match:
                    arc_meta_extra["selection_reason"] = (
                        "No same-readout arcs in the hard-compatible pool; "
                        "allowed readout mismatch and selected best by |Δtime| (then id)."
                    )
                else:
                    arc_meta_extra["selection_reason"] = (
                        "Preferred same-readout arcs (available) and selected best by |Δtime| (then id)."
                    )

        bias_warn_codes = {str(w.get("code")) for w in (wb or []) if str(w.get("code"))}
        flat_warn_codes = {str(w.get("code")) for w in (wf or []) if str(w.get("code"))}
        arc_warn_codes = {str(w.get("code")) for w in (wa or []) if str(w.get("code"))}


        matches.append(
            MatchEntry(
                science_set_id=ss.science_set_id,
                bias_id=sel_b.selected_id,
                # P0-E: keep the *full* flat set (time-windowed) for MasterFlat.
                # flat_id remains as the "best" representative for backward
                # compatibility with older configs and QC reporters.
                flat_id=sel_f.selected_id,
                flat_ids=flat_ids,
                arc_id=sel_a.selected_id,
                bias_meta=_meta(sel_b, "bias", warn_codes=bias_warn_codes),
                # P0-F: attach transparent readout-selection provenance for flats
                # when gain/rate mismatches are allowed.
                flat_meta=_meta(sel_f, "flat", extra=flat_meta_extra or None, warn_codes=flat_warn_codes),
                arc_meta=_meta(sel_a, "arc", extra=arc_meta_extra or None, warn_codes=arc_warn_codes),
            )
        )

        for w in (wb + wf + wa):
            warnings.append(
                ManifestWarning(
                    code=str(w.get("code")),
                    severity=str(w.get("severity", "WARN")),
                    message=str(w.get("message")),
                    context={
                        **(w.get("context") or {}),
                        "science_set_id": ss.science_set_id,
                    },
                )
            )

    # Summary convenience
    summary = {
        "n_frames_total": len(records),
        "n_science_longslit": len(science_frames),
        "n_science_sets": len(science_sets),
        "n_bias": len(pools.bias),
        "n_flat": len(pools.flat),
        "n_arc": len(pools.arc),
        "n_warnings": len(scan_warnings) + len(warnings),
        "compact_config_keys": {
            s.science_set_id: s.config.as_compact_str() for s in science_sets
        },
    }

    man = DatasetManifest(
        pipeline_version=pipeline_version,
        generated_utc=DatasetManifest.now_utc_iso(),
        data_dir=data_dir,
        night_id=night_id,
        science_sets=science_sets,
        calibration_pools=pools,
        matches=matches,
        warnings=[*scan_warnings, *warnings],
        frames=frame_index if include_frame_index else None,
        summary=summary,
    )

    return man


def resolve_global_exclude(
    data_dir: Path,
    *,
    exclude_paths: Sequence[str] | None = None,
    manifest_name: str = "project_manifest.yaml",
) -> tuple[list[str], dict[str, Any]]:
    """Resolve absolute exclude paths for a dataset directory.

    Policy (P0-K):
    - If ``<data_dir>/<manifest_name>`` exists, apply its ``exclude`` section
      *before* scanning/matching.
    - Merge with optional ``exclude_paths`` provided by the caller.
    - Return a stable, JSON-serializable ``excluded_summary`` for provenance.

    This helper is intentionally *Astropy-free* so it can be unit tested in
    lightweight environments.
    """

    data_dir_p = Path(data_dir).expanduser().resolve()
    excluded_summary: dict[str, Any] = {}
    excluded_abs: list[str] = []
    missing_files: list[str] = []
    unmatched_globs: list[str] = []

    manifest_path = data_dir_p / str(manifest_name)
    if manifest_path.is_file():
        try:
            import yaml  # type: ignore

            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict) and "exclude_frames" in raw and "exclude" not in raw:
                raw = dict(raw)
                raw["exclude"] = raw.pop("exclude_frames")

            ex = raw.get("exclude") if isinstance(raw, dict) else None
            ex_files: list[str] = []
            ex_globs: list[str] = []
            if isinstance(ex, list):
                ex_files = [str(x) for x in ex if str(x).strip()]
            elif isinstance(ex, dict):
                f = ex.get("files", [])
                g = ex.get("globs", [])
                if isinstance(f, str):
                    f = [f]
                if isinstance(g, str):
                    g = [g]
                if isinstance(f, list):
                    ex_files = [str(x) for x in f if str(x).strip()]
                if isinstance(g, list):
                    ex_globs = [str(x) for x in g if str(x).strip()]
            elif isinstance(ex, str):
                ex_files = [ex]

            ex_set: set[str] = set()

            # Resolve explicit files.
            for s in ex_files:
                try:
                    pp = Path(str(s)).expanduser()
                    if not pp.is_absolute():
                        pp = (data_dir_p / pp).resolve()
                    else:
                        pp = pp.resolve()
                    if pp.exists() and pp.is_file():
                        ex_set.add(str(pp))
                    else:
                        missing_files.append(str(s))
                except Exception:
                    missing_files.append(str(s))

            # Expand globs relative to data_dir (and allow './' prefix).
            for pat in ex_globs:
                try:
                    base = data_dir_p
                    pat_s = str(pat).replace("\\", "/").strip()
                    if pat_s.startswith("./"):
                        pat_s = pat_s[2:]
                    matches = [p.resolve() for p in sorted(base.glob(pat_s)) if p.is_file()]
                    if not matches:
                        unmatched_globs.append(str(pat))
                    for pp in matches:
                        ex_set.add(str(pp))
                except Exception:
                    unmatched_globs.append(str(pat))

            excluded_abs = sorted(ex_set)
            excluded_summary = {
                "manifest_path": str(manifest_path.resolve()),
                "exclude_files": ex_files,
                "exclude_globs": ex_globs,
                "excluded_n": int(len(excluded_abs)),
                "excluded_paths": [
                    str(Path(p).relative_to(data_dir_p)) if p.startswith(str(data_dir_p)) else p
                    for p in excluded_abs
                ],
                "missing_files": missing_files,
                "unmatched_globs": unmatched_globs,
            }
        except Exception as e:
            excluded_summary = {"manifest_path": str(manifest_path.resolve()), "error": str(e)}

    # Merge with explicit exclude_paths argument (if provided).
    merged_exclude: list[str] = []
    seen: set[str] = set()

    for s in (excluded_abs or []):
        if s not in seen:
            seen.add(s)
            merged_exclude.append(s)

    if exclude_paths:
        for s in exclude_paths:
            try:
                pp = Path(str(s)).expanduser()
                if not pp.is_absolute():
                    pp = (data_dir_p / pp).resolve()
                else:
                    pp = pp.resolve()
                ps = str(pp)
                if ps not in seen:
                    seen.add(ps)
                    merged_exclude.append(ps)
            except Exception:
                continue

    return merged_exclude, excluded_summary


def build_dataset_manifest(
    data_dir: str | Path,
    *,
    out_path: str | Path | None = None,
    recursive: bool = True,
    max_files: int | None = None,
    include_hashes: bool = False,
    include_frame_index: bool = True,
    flat_time_window_s: float | None = 14400.0,
    flat_allow_readout_diff: bool = True,
    arc_allow_readout_diff: bool | None = None,
    night_id: str | None = None,
    exclude_paths: Sequence[str] | None = None,
    pipeline_version: str,
) -> DatasetManifest:
    """Scan a directory of FITS files and build ``dataset_manifest.json``."""

    data_dir_p = Path(data_dir).expanduser().resolve()

    # --- P0-K: apply global exclude from project_manifest.yaml before scanning/matching ---
    merged_exclude, excluded_summary = resolve_global_exclude(data_dir_p, exclude_paths=exclude_paths)

    recs, scan_warns = scan_fits_directory(
        data_dir_p,
        recursive=recursive,
        max_files=max_files,
        include_hashes=include_hashes,
        exclude_paths=merged_exclude if merged_exclude else None,
    )
    man = build_dataset_manifest_from_records(
        recs,
        pipeline_version=pipeline_version,
        data_dir=str(data_dir_p),
        night_id=night_id,
        include_frame_index=include_frame_index,
        flat_time_window_s=flat_time_window_s,
        flat_allow_readout_diff=flat_allow_readout_diff,
        arc_allow_readout_diff=arc_allow_readout_diff,
    )

    # Attach P0-K excluded summary for provenance/QA.
    try:
        if excluded_summary:
            man.excluded_summary = dict(excluded_summary)
            # Keep a numeric summary key for quick QA.
            if isinstance(man.summary, dict):
                man.summary['n_excluded'] = int(excluded_summary.get('excluded_n') or 0)
                man.summary['excluded_manifest'] = str(excluded_summary.get('manifest_path') or '')
    except Exception:
        pass
    # append scan warnings (open/contract errors)
    if scan_warns:
        man.warnings = [*scan_warns, *(man.warnings or [])]
        man.summary["n_warnings"] = len(man.warnings)

    if out_path is None:
        out_path = data_dir_p / "dataset_manifest.json"
    man.write_json(out_path)
    return man
