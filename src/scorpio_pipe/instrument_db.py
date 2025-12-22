from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GrismSpec:
    id: str
    grooves_lmm: int | None = None
    range_A: tuple[float, float] | None = None
    dispersion_A_per_pix: float | None = None
    fwhm_A_at_slit_arcsec: dict[str, float] | None = None
    notes: str = ""
    quality: str = ""


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    display_name: str
    plate_scale_arcsec_per_pix: float | None = None
    fov_arcmin: tuple[float, float] | None = None
    slit_length_arcmin: float | None = None
    detector: dict[str, Any] | None = None
    grisms: dict[str, GrismSpec] | None = None


def _resource_path(*parts: str) -> Path:
    # resources live alongside this module inside the installed package
    here = Path(__file__).resolve().parent
    return here / "resources" / Path(*parts)


@lru_cache(maxsize=1)
def load_instrument_db() -> dict[str, InstrumentSpec]:
    p = _resource_path("instruments", "scorpio_instruments.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    inst_raw = raw.get("instruments", {}) if isinstance(raw, dict) else {}
    out: dict[str, InstrumentSpec] = {}

    for name, v in inst_raw.items():
        if not isinstance(v, dict):
            continue
        grisms: dict[str, GrismSpec] = {}
        for g in v.get("grisms", []) or []:
            if not isinstance(g, dict) or not g.get("id"):
                continue
            gid = str(g.get("id"))
            rng = g.get("range_A")
            rng_val = None
            quality = ""
            if isinstance(rng, dict):
                quality = str(rng.get("quality", "") or "")
                rv = rng.get("value")
                if isinstance(rv, (list, tuple)) and len(rv) == 2:
                    try:
                        rng_val = (float(rv[0]), float(rv[1]))
                    except Exception:
                        rng_val = None
            elif isinstance(rng, (list, tuple)) and len(rng) == 2:
                try:
                    rng_val = (float(rng[0]), float(rng[1]))
                except Exception:
                    rng_val = None

            disp = g.get("dispersion_A_per_pix")
            disp_val = None
            if isinstance(disp, dict):
                if not quality:
                    quality = str(disp.get("quality", "") or "")
                dv = disp.get("value")
                try:
                    disp_val = float(dv)
                except Exception:
                    disp_val = None
            else:
                try:
                    disp_val = float(disp)
                except Exception:
                    disp_val = None

            fwhm = g.get("fwhm_A_at_slit_arcsec")
            fwhm_val: dict[str, float] | None = None
            if isinstance(fwhm, dict):
                fwhm_val = {}
                for slit, sv in fwhm.items():
                    if isinstance(sv, dict):
                        if not quality:
                            quality = str(sv.get("quality", "") or "")
                        val = sv.get("value")
                        if isinstance(val, (list, tuple)) and len(val) == 2:
                            # keep mid-point as a single number for UI hints
                            try:
                                fwhm_val[str(slit)] = (float(val[0]) + float(val[1])) / 2.0
                            except Exception:
                                pass
                        else:
                            try:
                                fwhm_val[str(slit)] = float(val)
                            except Exception:
                                pass
                    else:
                        try:
                            fwhm_val[str(slit)] = float(sv)
                        except Exception:
                            pass
                if not fwhm_val:
                    fwhm_val = None

            grisms[gid] = GrismSpec(
                id=gid,
                grooves_lmm=(int(g.get("grooves_lmm")) if g.get("grooves_lmm") is not None else None),
                range_A=rng_val,
                dispersion_A_per_pix=disp_val,
                fwhm_A_at_slit_arcsec=fwhm_val,
                notes=str(g.get("notes", "") or ""),
                quality=quality,
            )

        out[name] = InstrumentSpec(
            name=str(name),
            display_name=str(v.get("display_name", name) or name),
            plate_scale_arcsec_per_pix=_to_float(v.get("plate_scale_arcsec_per_pix")),
            fov_arcmin=_to_pair(v.get("fov_arcmin")),
            slit_length_arcmin=_to_float(v.get("slit_length_arcmin")),
            detector=v.get("detector") if isinstance(v.get("detector"), dict) else None,
            grisms=grisms,
        )
    return out


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _to_pair(x: Any) -> tuple[float, float] | None:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        try:
            return float(x[0]), float(x[1])
        except Exception:
            return None
    return None


def guess_instrument_from_header(hdr: Any) -> str | None:
    """Best-effort instrument guess.

    Headers vary; we use common keywords and relaxed matching.
    """
    keys = ["INSTRUME", "INSTRUMENT", "TELESCOP", "OBSERVAT", "DETNAM"]
    vals: list[str] = []
    for k in keys:
        try:
            if k in hdr:
                vals.append(str(hdr.get(k) or ""))
        except Exception:
            continue
    s = " ".join(vals).upper()
    if "SCORPIO-2" in s or "SCORPIO2" in s:
        return "SCORPIO-2"
    if "SCORPIO" in s:
        return "SCORPIO"
    return None


def find_grism(instrument: str | None, grism_id: str | None) -> GrismSpec | None:
    if not instrument or not grism_id:
        return None
    db = load_instrument_db()
    inst = db.get(instrument)
    if inst is None or not inst.grisms:
        return None
    # exact match first
    if grism_id in inst.grisms:
        return inst.grisms[grism_id]
    # relaxed
    gnorm = "".join(ch for ch in str(grism_id).upper() if ch.isalnum() or ch in "@")
    for k, v in inst.grisms.items():
        knorm = "".join(ch for ch in str(k).upper() if ch.isalnum() or ch in "@")
        if knorm == gnorm:
            return v
    return None
