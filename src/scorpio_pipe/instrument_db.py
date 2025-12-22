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
    resolution_fwhm_A: float | None = None
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
    """Load instrument database shipped with the package.

    Supports both legacy formats:

    1) mapping:
       instruments:
         SCORPIO: { ... }
    2) list:
       instruments:
         - id: SCORPIO
           label: SCORPIO-1
           ...

    The list format is preferred as it preserves ordering and allows richer metadata.
    """
    p = _resource_path("instruments", "scorpio_instruments.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    inst_raw = raw.get("instruments") if isinstance(raw, dict) else None

    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(inst_raw, dict):
        for k, v in inst_raw.items():
            if isinstance(v, dict):
                entries.append((str(k), v))
    elif isinstance(inst_raw, list):
        for it in inst_raw:
            if not isinstance(it, dict):
                continue
            key = str(it.get("id") or it.get("name") or it.get("label") or "").strip()
            if not key:
                continue
            entries.append((key, it))

    out: dict[str, InstrumentSpec] = {}

    for key, v in entries:
        name = str(v.get("id") or key).strip()
        if not name:
            continue

        display = str(v.get("label") or v.get("display_name") or name)

        grisms: dict[str, GrismSpec] = {}
        for g in v.get("grisms", []) or []:
            if not isinstance(g, dict) or not g.get("id"):
                continue
            gid = str(g.get("id"))

            rng_val = None
            quality = str(g.get("quality", "") or "")
            rng = g.get("range_A")
            if isinstance(rng, dict):
                quality = quality or str(rng.get("quality", "") or "")
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

            disp_val = None
            disp = g.get("dispersion_A_per_pix")
            if isinstance(disp, dict):
                quality = quality or str(disp.get("quality", "") or "")
                try:
                    disp_val = float(disp.get("value"))
                except Exception:
                    disp_val = None
            else:
                try:
                    disp_val = float(disp)
                except Exception:
                    disp_val = None

            fwhm_val: dict[str, float] | None = None
            fwhm = g.get("fwhm_A_at_slit_arcsec")
            if isinstance(fwhm, dict):
                fwhm_val = {}
                for slit, sv in fwhm.items():
                    if isinstance(sv, dict):
                        quality = quality or str(sv.get("quality", "") or "")
                        val = sv.get("value")
                        if isinstance(val, (list, tuple)) and len(val) == 2:
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

            res_fwhm = None
            try:
                rv = g.get("resolution_fwhm_A")
                if isinstance(rv, (int, float)):
                    res_fwhm = float(rv)
            except Exception:
                res_fwhm = None
            if res_fwhm is None and fwhm_val:
                if "1.0" in fwhm_val:
                    res_fwhm = float(fwhm_val["1.0"])
                else:
                    try:
                        res_fwhm = float(sum(fwhm_val.values()) / max(1, len(fwhm_val)))
                    except Exception:
                        res_fwhm = None

            grisms[gid] = GrismSpec(
                id=gid,
                grooves_lmm=(int(g.get("grooves_lmm")) if g.get("grooves_lmm") is not None else None),
                range_A=rng_val,
                dispersion_A_per_pix=disp_val,
                resolution_fwhm_A=res_fwhm,
                fwhm_A_at_slit_arcsec=fwhm_val,
                notes=str(g.get("notes", "") or ""),
                quality=quality,
            )

        out[name] = InstrumentSpec(
            name=name,
            display_name=display,
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
    gnorm = "".join(ch for ch in str(grism_id).upper() if ch.isalnum())
    for k, v in inst.grisms.items():
        knorm = "".join(ch for ch in str(k).upper() if ch.isalnum())
        if knorm == gnorm:
            return v
    return None
