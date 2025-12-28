from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from scorpio_pipe.app_paths import ensure_dir
from scorpio_pipe.fits_utils import open_fits_smart
from scorpio_pipe.io.mef import write_sci_var_mask
from scorpio_pipe.noise_model import estimate_variance_adu2, resolve_noise_params
from scorpio_pipe.paths import resolve_work_dir


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
    superbias_path: Path,
    out_path: Path,
    *,
    do_bias_subtract: bool = True,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
) -> Path:
    """Apply flatfield correction to a single frame.

    - subtract superbias (if requested and not already marked as BIASSUB)
    - divide by superflat
    """

    data, var, mask, hdr = _read_sci_var_mask(data_path)
    superflat = fits.getdata(superflat_path).astype(np.float32)

    if do_bias_subtract and not bool(hdr.get("BIASSUB", False)):
        superbias = fits.getdata(superbias_path).astype(np.float32)
        if superbias.shape == data.shape:
            data = data - superbias.astype(np.float32)
            hdr["BIASSUB"] = (True, "Superbias subtracted")
            hdr["HISTORY"] = "scorpio_pipe flatfield: superbias subtracted"

    # Avoid division by zero
    sf = superflat.astype(np.float32)
    sf = np.where(np.isfinite(sf) & (sf != 0), sf, np.nan)

    corr = (data / sf).astype(np.float32)

    # Variance: if missing, estimate from a simple CCD noise model.
    if var is None:
        var, npar = estimate_variance_adu2(
            data,
            hdr,
            gain_override=gain_override,
            rdnoise_override=rdnoise_override,
        )
        hdr.setdefault("HISTORY", "scorpio_pipe flatfield: VAR estimated")
        # Keep resolved params in the header for downstream stages.
        hdr.setdefault("GAIN", float(npar.gain_e_per_adu))
        hdr.setdefault("RDNOISE", float(npar.rdnoise_e))
        hdr.setdefault("NOISRC", str(npar.source))
    else:
        # Still resolve params for header consistency (do not override explicit cards).
        npar = resolve_noise_params(
            hdr,
            gain_override=gain_override,
            rdnoise_override=rdnoise_override,
        )
        hdr.setdefault("GAIN", float(npar.gain_e_per_adu))
        hdr.setdefault("RDNOISE", float(npar.rdnoise_e))
        hdr.setdefault("NOISRC", str(npar.source))

    # Multiplicative correction: variance scales as 1/flat^2.
    var_corr = (var / (sf**2)).astype(np.float32)
    hdr["FLATCOR"] = (True, "Flat-fielding applied")
    hdr["HISTORY"] = "scorpio_pipe flatfield: divided by superflat"

    # Keep science units explicit.
    hdr.setdefault("BUNIT", "ADU")

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
    return out_path


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
    - flatfield/<kind>/*_flat.fits
    - flatfield/flatfield_done.json

    Notes
    -----
    - If cosmics stage was run with bias_subtract=True, then the cosmics-cleaned
      frames are already superbias-subtracted (BIASSUB=True). In that case, we
      avoid subtracting the superbias again.
    """

    work_dir = resolve_work_dir(cfg)
    out_dir = out_dir or (work_dir / "flatfield")
    ensure_dir(out_dir)

    block = cfg.get("flatfield", {}) or {}
    enabled = bool(block.get("enabled", False))

    done_path = out_dir / "flatfield_done.json"

    if not enabled:
        done_path.write_text(
            json.dumps({"enabled": False, "status": "skipped"}, indent=2),
            encoding="utf-8",
        )
        return done_path.resolve()

    frames = cfg.get("frames", {}) or {}

    superbias_path = _resolve_path(
        (cfg.get("calib", {}) or {}).get(
            "superbias_path", work_dir / "calib" / "superbias.fits"
        ),
        work_dir,
    )
    superflat_path = _resolve_path(
        (cfg.get("calib", {}) or {}).get(
            "superflat_path", work_dir / "calib" / "superflat.fits"
        ),
        work_dir,
    )

    flat_paths = [_resolve_path(p, work_dir) for p in (frames.get("flat") or [])]
    if not flat_paths:
        raise ValueError("No flat frames selected (frames.flat is empty)")

    # Always build / refresh superflat for the current object.
    # Canonical builder lives in stages.calib (single source of truth).
    from scorpio_pipe.stages.calib import (
        build_superflat as _build_superflat,
        _resolve_superbias_path as _resolve_superbias_path,
    )

    ensure_dir(superflat_path.parent)
    # Use config-driven builder to avoid diverging behavior between different call paths.
    superflat_path = _build_superflat(cfg, out_path=superflat_path)
    # Make sure we use the same superbias file as the superflat builder.
    superbias_path = _resolve_superbias_path(cfg, work_dir)

    apply_to = list(
        block.get("apply_to") or ["obj", "sky", "sunsky"]
    )  # + optional 'neon'

    cosmics_bias_sub = bool((cfg.get("cosmics", {}) or {}).get("bias_subtract", True))
    do_bias_sub_flat = bool(block.get("bias_subtract", True))

    gain_override = block.get("gain_e_per_adu")
    rdnoise_override = block.get("read_noise_e")

    outputs: list[str] = []

    for kind in apply_to:
        kind_frames = frames.get(kind) or []
        if not isinstance(kind_frames, list) or not kind_frames:
            continue

        kind_out = out_dir / kind
        ensure_dir(kind_out)

        kind_clean_dir = work_dir / "cosmics" / kind / "clean"

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

            dst = kind_out / f"{src0.stem}_flat.fits"
            apply_flat(
                src,
                superflat_path,
                superbias_path,
                dst,
                do_bias_subtract=do_bias_subtract,
                gain_override=float(gain_override) if gain_override is not None else None,
                rdnoise_override=float(rdnoise_override) if rdnoise_override is not None else None,
            )
            outputs.append(str(dst))

    done = {
        "enabled": True,
        "status": "ok",
        "superbias": str(superbias_path),
        "superflat": str(superflat_path),
        "apply_to": apply_to,
        "outputs": outputs,
        "n_outputs": len(outputs),
    }

    done_path.write_text(
        json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return done_path.resolve()
