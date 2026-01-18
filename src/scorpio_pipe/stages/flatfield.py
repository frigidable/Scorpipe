from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from scorpio_pipe.app_paths import ensure_dir
from scorpio_pipe.calib.compat import ensure_compatible_calib
from scorpio_pipe.fits_utils import open_fits_smart
from scorpio_pipe import maskbits
from scorpio_pipe.io.mef import write_sci_var_mask
from scorpio_pipe.noise_model import estimate_variance_adu2, resolve_noise_params
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.units_model import ensure_electron_units
from scorpio_pipe.workspace_paths import stage_dir


def _read_sci_var_mask(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, fits.Header]:
    """Read SCI (+ optional VAR/MASK) from either a simple FITS or MEF."""

    with open_fits_smart(
        path,
        memmap="auto",
        ignore_missing_end=True,
        ignore_missing_simple=True,
        do_not_scale_image_data=False,
    ) as hdul:
        hdr = fits.Header(hdul[0].header)

        # Prefer explicit EXTNAME=SCI if present; otherwise fall back to primary.
        sci: np.ndarray | None = None
        var: np.ndarray | None = None
        mask: np.ndarray | None = None

        for h in hdul:
            extname = str(h.header.get("EXTNAME", "")).strip().upper()
            if extname == "SCI" and getattr(h, "data", None) is not None:
                sci = np.asarray(h.data)
            elif extname == "VAR" and getattr(h, "data", None) is not None:
                var = np.asarray(h.data)
            elif extname == "MASK" and getattr(h, "data", None) is not None:
                mask = np.asarray(h.data)

        if sci is None:
            data0 = hdul[0].data
            if data0 is None:
                raise ValueError(f"Empty FITS data: {path}")
            sci = np.asarray(data0)

    sci = np.asarray(sci, dtype=np.float32)
    if var is not None:
        var = np.asarray(var, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != np.uint16:
            mask = np.asarray(mask, dtype=np.uint16)
    return sci, var, mask, hdr


# Flatfield stage.
#
# Note:
# This module intentionally does *not* implement its own superflat builder.
# The canonical implementation lives in ``scorpio_pipe.stages.calib`` and is
# shared by both the dedicated `superflat` stage and `flatfield` (when it needs
# to ensure a superflat exists).


def apply_flat(
    data_path: Path,
    superflat_path: Path,
    out_path: Path,
    *,
    cfg: dict,
    do_bias_subtract: bool = True,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
    stage_flags: list[dict] | None = None,
) -> tuple[Path, "BiasSelection | None"]:
    """Apply flatfield correction to a single frame.

    - optionally subtract a *readout-aware* MasterBias (P0-C1)
    - divide by superflat

    Returns
    -------
    (out_path, bias_selection)
    """

    from scorpio_pipe.stages.calib import resolve_master_bias, stamp_bias_selection

    from scorpio_pipe.variance_model import propagate_divide

    data, var, mask, hdr = _read_sci_var_mask(data_path)
    # Flat can be a legacy PRIMARY-only "superflat" or a MEF MasterFlat
    # (SCI+VAR+MASK). We support both.
    superflat, sf_var, sf_mask, _sf_hdr = _read_sci_var_mask(superflat_path)

    # Calibration compatibility check (strict geometry key; readout differences allowed).
    ensure_compatible_calib(
        hdr,
        superflat_path,
        kind="superflat",
        strict=True,
        allow_readout_diff=True,
        stage_flags=stage_flags,
    )

    sel = None
    bvar_to_add = None
    if do_bias_subtract and not bool(hdr.get("BIASSUB", False)):
        sel = resolve_master_bias(cfg, hdr, sci_shape=data.shape)
        if not sel.sci_path.exists():
            raise RuntimeError(f"MasterBias not found: {sel.sci_path}")

        sb = fits.getdata(sel.sci_path).astype(np.float32)
        if sb.shape != data.shape:
            raise RuntimeError(
                f"MasterBias shape mismatch: bias={sb.shape} vs frame={data.shape} ({data_path})"
            )

        # Optional VAR/DQ layers.
        bvar = None
        bdq = None
        try:
            if sel.var_path and sel.var_path.exists():
                bvar = fits.getdata(sel.var_path).astype(np.float32)
        except Exception:
            bvar = None
        try:
            if sel.dq_path and sel.dq_path.exists():
                bdq = fits.getdata(sel.dq_path)
                bdq = np.asarray(bdq, dtype=np.uint16)
        except Exception:
            bdq = None

        data = (data - sb).astype(np.float32)
        hdr["BIASSUB"] = (True, "MasterBias subtracted")
        try:
            hdr = stamp_bias_selection(hdr, sel)
        except Exception:
            pass
        hdr["HISTORY"] = f"scorpio_pipe flatfield: master_bias subtracted (gid={getattr(sel,'gid','?')})"
        if getattr(sel, "degraded", False):
            hdr["HISTORY"] = f"scorpio_pipe flatfield: bias selection degraded ({getattr(sel,'reason','')})"

        # If bias VAR is present, subtraction adds variance (same units).
        if bvar is not None and bvar.shape == data.shape:
            if var is None:
                bvar_to_add = bvar.astype(np.float32)
            else:
                var = (var + bvar).astype(np.float32)

        # Combine DQ into the pipeline mask.
        if bdq is not None and bdq.shape == data.shape:
            if mask is None:
                mask = np.zeros(data.shape, dtype=np.uint16)
            mask = (mask.astype(np.uint16) | bdq.astype(np.uint16)).astype(np.uint16)

    # Avoid division by zero / invalid flat pixels.
    sf = superflat.astype(np.float32)
    bad_flat = (~np.isfinite(sf)) | (sf == 0)
    sf_safe = sf.copy()
    sf_safe[bad_flat] = np.nan

    corr = (data / sf_safe).astype(np.float32)
    if np.any(bad_flat):
        # Mark invalid flat pixels as BADPIX in the mask and zero-out NaNs.
        if mask is None:
            mask = np.zeros(data.shape, dtype=np.uint16)
        mask = mask.astype(np.uint16)
        mask[bad_flat] |= np.uint16(maskbits.BADPIX)
        corr = corr.copy()
        corr[bad_flat] = 0.0

    rn_hint_e = None
    try:
        if sel is not None and getattr(sel, 'rdnoise_e', None) is not None:
            rn_hint_e = float(getattr(sel, 'rdnoise_e'))
    except Exception:
        rn_hint_e = None

    # Variance: if missing, estimate from a simple CCD noise model.
    if var is None:
        var, npar = estimate_variance_adu2(
            data,
            hdr,
            cfg=cfg,
            gain_override=gain_override,
            rdnoise_override=rdnoise_override,
            bias_rn_est_e=rn_hint_e,
            instrument_hint=str(cfg.get("instrument_hint") or ""),
            require_gain=True,
        )
        hdr.add_history("scorpio_pipe flatfield: VAR estimated")
        # Keep resolved params in the header for downstream stages.
        if "GAIN" not in hdr:
            hdr["GAIN"] = float(npar.gain_e_per_adu)
        if "RDNOISE" not in hdr:
            hdr["RDNOISE"] = float(npar.rdnoise_e)
        if "NOISRC" not in hdr:
            hdr["NOISRC"] = str(npar.source)
    else:
        # Still resolve params for header consistency (do not override explicit cards).
        npar = resolve_noise_params(
            hdr,
            cfg=cfg,
            gain_override=gain_override,
            rdnoise_override=rdnoise_override,
            bias_rn_est_e=rn_hint_e,
            instrument_hint=str(cfg.get("instrument_hint") or ""),
        )
        if "GAIN" not in hdr:
            hdr["GAIN"] = float(npar.gain_e_per_adu)
        if "RDNOISE" not in hdr:
            hdr["RDNOISE"] = float(npar.rdnoise_e)
        if "NOISRC" not in hdr:
            hdr["NOISRC"] = str(npar.source)

    # If bias VAR was deferred until after VAR estimation, add it now.
    if bvar_to_add is not None and var is not None and bvar_to_add.shape == var.shape:
        var = (var + bvar_to_add).astype(np.float32)

    # Multiplicative correction: propagate division noise.
    # If flat VAR is present (MasterFlat), include it in the error budget:
    # Var(S/F) = Var(S)/F^2 + S^2*Var(F)/F^4.
    if sf_var is not None and sf_var.shape == var.shape:
        var_corr = propagate_divide(data, var, sf_safe, sf_var).astype(np.float32)
    else:
        var_corr = (var / (sf_safe**2)).astype(np.float32)
    if np.any(bad_flat):
        var_corr = var_corr.copy()
        var_corr[bad_flat] = 0.0

    # If flat carries a MASK plane (MasterFlat), propagate it.
    if sf_mask is not None and sf_mask.shape == data.shape:
        if mask is None:
            mask = np.zeros(data.shape, dtype=np.uint16)
        mask = (mask.astype(np.uint16) | sf_mask.astype(np.uint16)).astype(np.uint16)
    hdr["FLATCOR"] = (True, "Flat-fielding applied")
    hdr["HISTORY"] = "scorpio_pipe flatfield: divided by superflat"

    # Convert to internal unit standard: electrons.
    corr_e, var_corr_e, hdr_e, prov, _ = ensure_electron_units(
        corr,
        var_corr,
        hdr,
        cfg=cfg,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        bias_rn_est_e=rn_hint_e,
        instrument_hint=str(cfg.get("instrument_hint") or ""),
        require_gain=True,
    )
    corr = corr_e
    var_corr = var_corr_e
    hdr = hdr_e

    # Contract (P0-B): MEF products must always carry a MASK plane.
    # Upstream inputs may lack a DQ/mask, in which case we create an all-zero mask
    # for determinism and downstream compatibility.
    if mask is None:
        mask = np.zeros(corr.shape, dtype=np.uint16)
    else:
        if getattr(mask, "dtype", None) != np.uint16:
            mask = np.asarray(mask, dtype=np.uint16)

    ensure_dir(out_path.parent)
    write_sci_var_mask(
        out_path,
        corr,
        var=var_corr,
        mask=mask,
        header=hdr,
        primary_data=corr,
        overwrite=True,
    )
    return out_path, sel


def _resolve_path(p: str | Path, base: Path) -> Path:
    pp = Path(str(p)).expanduser()
    if not pp.is_absolute():
        pp = (base / pp).resolve()
    return pp


def run_flatfield(cfg: dict, *, out_dir: Path | None = None) -> Path:
    """Run the Flat-fielding stage.

    Layout
    ------
    - calib/superflat.fits  (created if absent)
    - 04_flat/<kind>/*_flat.fits
    - 04_flat/flatfield_done.json

    Notes
    -----
    - If cosmics stage was run with bias_subtract=True, then the cosmics-cleaned
      frames are already superbias-subtracted (BIASSUB=True). In that case, we
      avoid subtracting the superbias again.
    """

    work_dir = resolve_work_dir(cfg)
    # Canonical v5.40+ layout uses numbered stage directories.
    # Legacy fallback `work_dir/flatfield` is still *read* by some tools,
    # but the stage must write to the canonical location by default.
    out_dir = out_dir or stage_dir(work_dir, "flatfield")
    ensure_dir(out_dir)

    block = cfg.get("flatfield", {}) or {}
    enabled = bool(block.get("enabled", False))

    done_path = out_dir / "flatfield_done.json"

    # Stage-local QC flags (P2 contract); populated by compat checks, etc.
    stage_flags: list[dict] = []

    if not enabled:
        done_path.write_text(
            json.dumps({"enabled": False, "status": "skipped", "flags": []}, indent=2),
            encoding="utf-8",
        )
        # Also write canonical done.json for aggregation/QC (best-effort).
        try:
            from scorpio_pipe.io.done_json import write_done_json

            write_done_json(
                stage="flatfield",
                stage_dir=out_dir,
                status="skipped",
                flags=[],
                qc={"flags": [], "max_severity": "INFO"},
            )
        except Exception:
            pass
        return done_path.resolve()

    frames = cfg.get("frames", {}) or {}

    # --- P0-E: optional dataset_manifest-driven per-science-set MasterFlat ---
    use_manifest = bool(block.get("use_manifest", True))
    manifest = None
    manifest_path: Path | None = None
    if use_manifest:
        # Resolution order: explicit config key -> work_dir -> data_dir.
        cand: list[Path] = []
        mp = cfg.get("dataset_manifest_path") or cfg.get("manifest_path")
        if mp:
            cand.append(Path(str(mp)).expanduser())
        cand.append((work_dir / "dataset_manifest.json").resolve())
        try:
            cand.append((Path(str(cfg.get("data_dir") or "")) / "dataset_manifest.json").expanduser().resolve())
        except Exception:
            pass

        for cpath in cand:
            if cpath and cpath.exists():
                manifest_path = cpath
                break

        if manifest_path and manifest_path.exists():
            from scorpio_pipe.dataset.manifest import DatasetManifest

            try:
                manifest = DatasetManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse dataset_manifest.json: {manifest_path}"
                ) from e

    # --- P0-K defense: re-apply global exclude to manifest associations (stale dataset_manifest) ---
    # IMPORTANT: exclude source of truth is data_dir/project_manifest.yaml; config exclude_frames is merged in.
    from scorpio_pipe.exclude_policy import resolve_exclude_set

    _ex = resolve_exclude_set(cfg, data_dir=cfg.get("data_dir"))
    cfg_excluded_set: set[str] = set(_ex.excluded_abs)
    excluded_summary: dict[str, Any] = dict(_ex.summary) if isinstance(_ex.summary, dict) else {}

    # Resolve calibration masters robustly (canonical stage dirs + legacy fallbacks).
    # NOTE: do *not* hard-code work_dir/calibs or work_dir/calib here — GUI/CLI may
    # use the canonical NN_* stage layout.
    from scorpio_pipe.stages.calib import (
        build_superbias as _build_superbias,
        build_superflat as _build_superflat,
        _resolve_superbias_path as _resolve_superbias_path,
        _resolve_superflat_path as _resolve_superflat_path,
    )

    superbias_path = _resolve_superbias_path(cfg, work_dir)
    superflat_path = _resolve_superflat_path(cfg, work_dir)

    apply_to = list(block.get("apply_to") or ["obj", "sky", "sunsky"])  # + optional 'neon'

    # Global (single) superflat is only required when we are *not* using a
    # dataset_manifest (P0-E), or when we need to flatfield non-obj kinds.
    need_global_flat = (manifest is None) or any(
        (k != "obj") and isinstance(frames.get(k), list) and bool(frames.get(k)) for k in apply_to
    )

    flat_paths = [_resolve_path(p, work_dir) for p in (frames.get("flat") or [])]
    if need_global_flat and not flat_paths:
        raise ValueError("No flat frames selected (frames.flat is empty), but global superflat is required")

    # Ensure superbias exists (flatfield may be executed standalone).
    if not superbias_path.exists():
        bias_paths = frames.get("bias") or []
        if bias_paths:
            try:
                _build_superbias(cfg, out_path=superbias_path)
            except Exception as e:
                raise RuntimeError(
                    f"FlatFielding requires superbias, and auto-build failed. "
                    f"Tried to build at: {superbias_path}"
                ) from e
        else:
            raise RuntimeError(
                f"FlatFielding requires superbias, but it was not found at: {superbias_path}. "
                "Run the 'superbias' stage first or set calib.superbias_path in config."
            )

    # Build/refresh superflat only if needed and missing or explicitly requested.
    rebuild_superflat = bool(block.get("rebuild_superflat", False))
    if need_global_flat and (rebuild_superflat or (not superflat_path.exists())):
        ensure_dir(superflat_path.parent)
        # Use config-driven builder to avoid diverging behavior between different call paths.
        superflat_path = _build_superflat(cfg, out_path=superflat_path)

    # After builds, resolve again (builders may mirror into canonical locations).
    superbias_path = _resolve_superbias_path(cfg, work_dir)
    superflat_path = _resolve_superflat_path(cfg, work_dir)

    if need_global_flat and (not superflat_path.exists()):
        raise RuntimeError(
            f"FlatFielding requires superflat, but it was not found at: {superflat_path}. "
            "Run the 'superflat' stage first (or enable flatfield.rebuild_superflat) or set calib.superflat_path in config."
        )

    # --- P0-E: build per-science-set MasterFlats from dataset_manifest associations ---
    masterflat_by_set: dict[str, Path] = {}
    masterflat_info: dict[str, dict] = {}
    frame_set_by_path: dict[str, str] = {}

    if manifest is not None:
        from scorpio_pipe.stages.calib import build_masterflat_set

        rebuild_masterflat = bool(block.get("rebuild_masterflat", False))
        mf_dir = out_dir / "masterflats"
        ensure_dir(mf_dir)

        # Which science sets to process?
        ss_sel = block.get("science_sets")
        sel_ids: list[str]
        if isinstance(ss_sel, str) and ss_sel.strip():
            sel_ids = [s.strip() for s in ss_sel.split(",") if s.strip()]
        elif isinstance(ss_sel, list) and ss_sel:
            sel_ids = [str(s) for s in ss_sel if str(s).strip()]
        else:
            sel_ids = [str(s.science_set_id) for s in (manifest.science_sets or [])]

        # Lookups.
        match_by_set = {str(m.science_set_id): m for m in (manifest.matches or [])}
        if not manifest.frames:
            raise RuntimeError(
                "Manifest mode requires `frames` index in dataset_manifest.json (include_frame_index=True)."
            )
        frame_by_id = {str(f.frame_id): f for f in (manifest.frames or [])}
        pools = manifest.calibration_pools
        flat_items = (pools.flat if pools is not None else [])
        # NOTE: CalibrationEntry uses `calib_id` (not `frame_id`).
        flat_pool = {str(c.calib_id): c for c in (flat_items or [])}
        base_dir = Path(str(manifest.data_dir or cfg.get("data_dir") or work_dir)).expanduser().resolve()

        def _resolve_man_path(p: str) -> Path:
            pp = Path(str(p)).expanduser()
            if not pp.is_absolute():
                pp = (base_dir / pp).resolve()
            return pp

        def _is_excluded(p: Path) -> bool:
            if not cfg_excluded_set:
                return False
            try:
                return str(p.resolve()) in cfg_excluded_set
            except Exception:
                return str(p) in cfg_excluded_set

        excluded_hits: dict[str, list[str]] = {}

        # Build MasterFlats and a frame->set mapping for obj frames.
        ss_by_id = {str(s.science_set_id): s for s in (manifest.science_sets or [])}
        for sid in sorted(sel_ids):
            ss = ss_by_id.get(str(sid))
            if ss is None:
                raise RuntimeError(f"Science set id not found in manifest: {sid}")

            m = match_by_set.get(str(sid))
            if m is None:
                raise RuntimeError(f"No match entry for science set: {sid}")

            flat_ids = list(m.flat_ids or [])
            if not flat_ids and m.flat_id:
                flat_ids = [str(m.flat_id)]
            if not flat_ids:
                raise RuntimeError(f"No flats associated with science set: {sid}")
            # Resolve flat paths from calibration pool (preferred), else frame index.
            fpaths: list[Path] = []
            kept_ids: list[str] = []
            excluded_ids: list[str] = []
            for fid in flat_ids:
                item = flat_pool.get(str(fid))
                if item is not None:
                    pth = _resolve_man_path(item.path)
                else:
                    fe = frame_by_id.get(str(fid))
                    if fe is None:
                        raise RuntimeError(f"Flat id not found in manifest pools/index: {fid} (set {sid})")
                    pth = _resolve_man_path(fe.path)

                if _is_excluded(pth):
                    excluded_ids.append(str(fid))
                    continue
                kept_ids.append(str(fid))
                fpaths.append(pth)

            if excluded_ids:
                excluded_hits[str(sid)] = list(excluded_ids)
                # Warn once per science set (not per exposure).
                stage_flags.append({
                    'code': 'MANIFEST_EXCLUDE_APPLIED',
                    'severity': 'WARN',
                    'message': 'Removed excluded flat frames from manifest associations',
                    'context': {'science_set_id': str(sid), 'removed_flat_ids': list(excluded_ids)},
                })

            if not kept_ids:
                raise RuntimeError(f"No flats left for science set {sid} after applying exclude")

            flat_ids = kept_ids

            out_mf = mf_dir / f"masterflat_{sid}.fits"
            if rebuild_masterflat or (not out_mf.exists()):
                out_mf, stats = build_masterflat_set(
                    cfg,
                    flat_paths=fpaths,
                    out_path=out_mf,
                    set_id=str(sid),
                )
            else:
                stats = {"cached": True}

            masterflat_by_set[str(sid)] = out_mf
            masterflat_info[str(sid)] = {
                "path": str(out_mf),
                "flat_ids": flat_ids,
                "flat_paths": [str(p) for p in fpaths],
                "stats": stats,
            }

            # Map science frames to their set id.
            for fid in (ss.frames or []):
                fe = frame_by_id.get(str(fid))
                if fe is None:
                    continue
                fp = _resolve_man_path(fe.path)
                frame_set_by_path[str(fp.resolve())] = str(sid)

    cosmics_bias_sub = bool((cfg.get("cosmics", {}) or {}).get("bias_subtract", True))
    do_bias_sub_flat = bool(block.get("bias_subtract", True))

    gain_override = block.get("gain_e_per_adu")
    rdnoise_override = block.get("read_noise_e")

    outputs: list[str] = []

    bias_used: dict[str, int] = {}
    n_bias_degraded = 0
    bias_degraded_reasons: dict[str, int] = {}

    for kind in apply_to:
        # In manifest mode, "obj" frames come directly from the selected science
        # sets, not from config.frames.obj.
        if (manifest is not None) and (kind == "obj") and frame_set_by_path:
            kind_frames = [str(Path(p)) for p in sorted(frame_set_by_path.keys())]
        else:
            kind_frames = frames.get(kind) or []

        if not isinstance(kind_frames, list) or not kind_frames:
            continue

        kind_out = out_dir / kind
        ensure_dir(kind_out)

        # Prefer canonical 05_cosmics layout, with legacy fallback.
        kind_clean_dir: Path | None = None
        for root in (stage_dir(work_dir, "cosmics"), work_dir / "cosmics"):
            cand = root / kind / "clean"
            if cand.exists():
                kind_clean_dir = cand
                break
        if kind_clean_dir is None:
            kind_clean_dir = stage_dir(work_dir, "cosmics") / kind / "clean"

        for fp in kind_frames:
            src0 = _resolve_path(fp, work_dir)
            if not src0.exists():
                continue

            clean = kind_clean_dir / f"{src0.stem}_clean.fits"
            src = clean if clean.exists() else src0

            # If we use a cosmics-cleaned product AND cosmics already subtracted bias,
            # then avoid doing it again.
            do_bias_subtract = do_bias_sub_flat
            if clean.exists() and cosmics_bias_sub:
                do_bias_subtract = False
            # Choose the flat product for this frame.
            flat_for_frame = superflat_path
            if (manifest is not None) and (kind == "obj"):
                sid = frame_set_by_path.get(str(src0.resolve()))
                if not sid:
                    raise RuntimeError(
                        f"Manifest mode: frame not assigned to a science_set: {src0}"
                    )
                flat_for_frame = masterflat_by_set.get(str(sid))
                if flat_for_frame is None:
                    raise RuntimeError(
                        f"Manifest mode: no MasterFlat built for science_set={sid} (frame={src0.name})"
                    )

            dst = kind_out / f"{src0.stem}_flat.fits"
            out_f, sel = apply_flat(
                src,
                flat_for_frame,
                dst,
                cfg=cfg,
                do_bias_subtract=do_bias_subtract,
                gain_override=float(gain_override) if gain_override is not None else None,
                rdnoise_override=float(rdnoise_override) if rdnoise_override is not None else None,
                stage_flags=stage_flags,
            )
            outputs.append(str(out_f))
            if sel is not None:
                gid = str(getattr(sel, 'gid', ''))
                if gid:
                    bias_used[gid] = int(bias_used.get(gid, 0)) + 1
                if bool(getattr(sel, 'degraded', False)):
                    n_bias_degraded += 1
                    rr = str(getattr(sel, 'reason', '')) or 'UNKNOWN'
                    bias_degraded_reasons[rr] = int(bias_degraded_reasons.get(rr, 0)) + 1

    done = {
        "enabled": True,
        "status": "ok",
        "superbias": str(superbias_path),
        "bias_policy": str((cfg.get('calib', {}) or {}).get('bias_policy', 'degraded')),
        "bias_used": bias_used,
        "n_bias_degraded": int(n_bias_degraded),
        "bias_degraded_reasons": bias_degraded_reasons,
        "superflat": str(superflat_path) if need_global_flat else None,
        "apply_to": apply_to,
        "outputs": outputs,
        "n_outputs": len(outputs),
    }

    if manifest is not None:
        done["manifest"] = {
            "path": str(manifest_path) if manifest_path else None,
            "science_sets": sorted(masterflat_by_set.keys()),
            "masterflats": masterflat_info,
        }

    # Normalize QC flags and set status accordingly.
    try:
        from scorpio_pipe.qc.flags import coerce_flags, max_severity

        flags = coerce_flags(stage_flags)
        # De-duplicate by (code, message) to avoid per-exposure spam.
        seen = set()
        dedup: list[dict] = []
        for f in flags:
            key = (str(f.get("code")), str(f.get("message")))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(f)
        flags = dedup
        qc_max = max_severity(flags)
    except Exception:
        flags = list(stage_flags)
        qc_max = "INFO"

    status = "ok" if qc_max == "INFO" else "warn"
    done["status"] = status
    done["flags"] = flags
    done["qc_max_severity"] = qc_max

    done_path.write_text(
        json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Canonical done.json for aggregation/QC.
    try:
        from datetime import datetime, timezone

        from scorpio_pipe.io.done_json import write_done_json
        from scorpio_pipe.version import PIPELINE_VERSION

        created_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        write_done_json(
            stage="flatfield",
            stage_dir=out_dir,
            status=status,
            inputs={
                "superbias_path": str(superbias_path),
                "superflat_path": str(superflat_path) if need_global_flat else None,
                "dataset_manifest": str(manifest_path) if manifest_path else None,
            },
            params={
                "apply_to": apply_to,
                "use_manifest": bool(manifest is not None),
                "allow_readout_diff": True,
            },
            outputs={
                "outputs": outputs,
                "n_outputs": int(len(outputs)),
            },
            flags=flags,
            qc={"flags": flags, "max_severity": qc_max},
            extra={"version": PIPELINE_VERSION, "created_utc": created_utc, "excluded_summary": excluded_summary or None},
        )
    except Exception:
        pass

    return done_path.resolve()
