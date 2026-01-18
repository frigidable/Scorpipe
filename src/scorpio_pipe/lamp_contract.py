from __future__ import annotations

"""Lamp type + line-list contract (P0-M).

Motivation
----------
SCORPIO/SCORPIO-2 arc frames are often labeled simply as "NEON" in headers/logs,
while the physical calibration lamp used for wavelength calibration is typically
He+Ne+Ar (HeNeAr). This is an easy place to get a *quiet systematic* in
wavelength solutions.

Contract
--------
- Expose an explicit ``lamp_type``: ``HeNeAr`` | ``Ne`` | ``Unknown``.
- Prefer explicit operator/config override (``wavesol.lamp_type``).
- If header evidence is weak (e.g. only ``OBJECT=NEON``), do *not* treat it as
  a reliable lamp type; fall back to instrument defaults.
- Default for long-slit SCORPIO/SCORPIO-2: ``HeNeAr``.
- When a default is used, record provenance (``source='default'``).
- In strict mode, unknown lamp must be overridden by config.

This module is deliberately small and stable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LAMP_HENEAR = "HeNeAr"
LAMP_NE = "Ne"
LAMP_UNKNOWN = "Unknown"


@dataclass(frozen=True)
class LampResolution:
    lamp_raw: str
    lamp_type: str
    source: str  # config|header|default|none


def normalize_lamp_type(v: Any) -> str:
    s = str(v or "").strip()
    if not s:
        return LAMP_UNKNOWN
    u = s.upper().replace("-", "").replace("_", "").replace(" ", "")

    # HeNeAr family
    if u in {"HENEAR", "HE+NE+AR", "HENEARLAMP"}:
        return LAMP_HENEAR
    if "HE" in u and "NE" in u and "AR" in u:
        return LAMP_HENEAR

    # Neon
    if u in {"NE", "NEON", "NEONLAMP"}:
        return LAMP_NE

    # Explicit unknown tokens
    if u in {"UNKNOWN", "UNDEF", "NONE", "N/A", "NA"}:
        return LAMP_UNKNOWN

    # Fall back
    return LAMP_UNKNOWN


def infer_lamp_from_header(hdr: Mapping[str, Any] | None) -> tuple[str, str]:
    """Infer (lamp_raw, lamp_type) from a FITS header.

    We treat ``OBJECT`` as a weak hint: ``OBJECT=NEON`` is *not* considered
    sufficient evidence for ``lamp_type='Ne'`` (it is typically just an
    exposure label).
    """

    if hdr is None or not isinstance(hdr, Mapping):
        return "", LAMP_UNKNOWN

    raw = ""
    src = ""

    # Stronger / more explicit keys first.
    strong_keys = [
        "LAMP",
        "LAMPTYPE",
        "LAMPID",
        "LAMP_NAME",
        "ARC_LAMP",
        "ARC_LMP",
        "ARCNAME",
    ]

    for k in strong_keys:
        try:
            if k in hdr:
                vv = str(hdr.get(k, "") or "").strip()
                if vv:
                    raw = vv
                    src = k
                    break
        except Exception:
            continue

    # Weak fallback: OBJECT. Keep raw but do not trust "NEON" as lamp type.
    if not raw:
        try:
            vv = str(hdr.get("OBJECT", "") or "").strip()
        except Exception:
            vv = ""
        if vv:
            raw = vv
            src = "OBJECT"

    lt = normalize_lamp_type(raw)

    # Policy: OBJECT=NEON is only a label; treat it as unknown for typing.
    if src == "OBJECT" and lt == LAMP_NE:
        lt = LAMP_UNKNOWN

    return raw, lt


def default_lamp_for_setup(setup: Mapping[str, Any] | None) -> str:
    """Instrument/mode defaults.

    Minimal P0 rule: long-slit SCORPIO/SCORPIO-2 -> HeNeAr.
    """

    setup = setup or {}
    instr = str(setup.get("instrument") or "").strip().lower()
    mode = str(setup.get("mode") or "").strip().lower()

    if instr in {"scorpio", "scorpio-1", "scorpio1", "scorpio_1", "scorpio2", "scorpio-2", "scorpio_2"}:
        # Default applies to long-slit (or when mode is unknown).
        if mode in {"", "longslit", "long-slit", "ls", "long"}:
            return LAMP_HENEAR

    return LAMP_UNKNOWN


def resolve_lamp_type(
    cfg: dict[str, Any],
    *,
    hdr: Mapping[str, Any] | None = None,
    arc_path: str | Path | None = None,
    instrument_hint: str | None = None,
    strict: bool = False,
) -> LampResolution:
    """Resolve lamp type with explicit provenance.

    Order:
      1) config override (wavesol.lamp_type)
      2) header inference (explicit keys)
      3) instrument default (SCORPIO/SCORPIO-2 -> HeNeAr)
      4) Unknown (warn/strict)

    Parameters
    ----------
    cfg : dict
        Pipeline config.
    hdr : mapping, optional
        Header to inspect.
    arc_path : path-like, optional
        If ``hdr`` is not provided, try to read header from this FITS path.
    instrument_hint : str, optional
        Best-effort instrument name, used for defaults if cfg lacks setup.
    strict : bool
        If True, unknown lamp raises RuntimeError.
    """

    wcfg = cfg.get("wavesol", {}) if isinstance(cfg.get("wavesol"), dict) else {}

    # 1) Config override
    lamp_cfg = None
    if isinstance(wcfg, dict):
        lamp_cfg = wcfg.get("lamp_type") or wcfg.get("lamptype") or wcfg.get("lamp")
    if lamp_cfg:
        lt = normalize_lamp_type(lamp_cfg)
        if lt == LAMP_UNKNOWN and strict:
            raise RuntimeError(
                "wavesol.lamp_type is set but could not be parsed. Use 'HeNeAr' or 'Ne'."
            )
        return LampResolution(lamp_raw=str(lamp_cfg), lamp_type=lt, source="config")

    # 2) Header (possibly from FITS)
    hdr_use: Mapping[str, Any] | None = hdr
    if hdr_use is None and arc_path is not None:
        try:
            from astropy.io import fits  # type: ignore

            with fits.open(str(arc_path), memmap=False) as hdul:
                # Prefer SCI header if present; otherwise PRIMARY.
                try:
                    if "SCI" in hdul:
                        hdr_use = dict(hdul["SCI"].header)
                    else:
                        hdr_use = dict(hdul[0].header)
                except Exception:
                    hdr_use = dict(hdul[0].header)
        except Exception:
            hdr_use = None

    raw = ""
    lt = LAMP_UNKNOWN
    if hdr_use is not None:
        raw, lt = infer_lamp_from_header(hdr_use)
        if lt != LAMP_UNKNOWN:
            return LampResolution(lamp_raw=raw, lamp_type=lt, source="header")

    # 3) Default from setup
    setup = None
    fr = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    if isinstance(fr, dict) and isinstance(fr.get("__setup__"), dict):
        setup = fr.get("__setup__")

    if setup is None and instrument_hint:
        setup = {"instrument": instrument_hint}

    d = default_lamp_for_setup(setup)
    if d != LAMP_UNKNOWN:
        return LampResolution(lamp_raw=raw, lamp_type=d, source="default")

    # 4) Unknown
    if strict:
        raise RuntimeError(
            "Lamp type is unknown. Provide wavesol.lamp_type in config.yaml (e.g. 'HeNeAr' or 'Ne')."
        )

    return LampResolution(lamp_raw=raw, lamp_type=LAMP_UNKNOWN, source="none")


def choose_linelist_name(lamp_type: str) -> str:
    lt = normalize_lamp_type(lamp_type)
    if lt == LAMP_HENEAR:
        return "henear_lines.csv"
    # Historical default.
    return "neon_lines.csv"


def resolve_linelist_csv(cfg: dict[str, Any], *, lamp_type: str) -> str:
    """Resolve the line-list CSV path (string) for UI/provenance."""

    from scorpio_pipe.paths import resolve_work_dir
    from scorpio_pipe.refs.store import resolve_reference

    wcfg = cfg.get("wavesol", {}) if isinstance(cfg.get("wavesol"), dict) else {}

    csv_name = None
    if isinstance(wcfg, dict):
        # New preferred key(s).
        csv_name = wcfg.get("linelist_csv") or wcfg.get("line_list_csv")
        if not csv_name:
            # Back-compat key.
            csv_name = wcfg.get("neon_lines_csv")

    if not csv_name:
        csv_name = choose_linelist_name(lamp_type)

    rr = resolve_reference(
        str(csv_name),
        resources_dir=cfg.get("resources_dir"),
        work_dir=resolve_work_dir(cfg),
        config_dir=cfg.get("config_dir"),
        project_root=cfg.get("data_dir") or cfg.get("project_root"),
        allow_package=True,
    )
    return str(Path(rr.resolved_path).expanduser())


def resolve_linelist_csv_path(cfg: dict[str, Any], lamp_type: str) -> Path:
    """Path-typed helper used by stage code (compat)."""

    return Path(resolve_linelist_csv(cfg, lamp_type=lamp_type)).expanduser()
