from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
from astropy.io import fits

from scorpio_pipe.fits_utils import open_fits_smart

from scorpio_pipe.nightlog import find_nightlog, parse_nightlog
from scorpio_pipe.instrument_db import guess_instrument_from_header

EXPECTED_COLUMNS = [
    "path",
    "fid",
    "kind",
    "object",
    "object_norm",
    "date_obs",
    "exptime",
    "instrument",
    "mode",
    "disperser",
    "slit",
    "binning",
    "window",
    "shape",
]


# Stable display order for frame kinds (used by UI browsers)
KIND_ORDER = [
    'bias',
    'flat',
    'neon',
    'obj',
    'sky',
    'sunsky',
]

def _binning_from_header(hdr) -> str:
    """Return binning as 'Bx×By' when possible, else ''."""
    # common formats: CCDSUM='1 1', BINNING='1x1', BINX/BINY, CDELT1? (not reliable)
    ccdsum = _safe_get(hdr, "CCDSUM", "BINNING", default=None)
    if ccdsum is not None:
        s = str(ccdsum).replace("X", "x").strip()
        if "x" in s and any(ch.isdigit() for ch in s):
            # already 1x1-like
            parts = [p for p in s.replace(" ", "").split("x") if p]
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                return f"{int(parts[0])}x{int(parts[1])}"
        # '1 1'
        parts = [p for p in s.replace("x", " ").split() if p]
        if len(parts) == 2 and all(p.replace(".", "").isdigit() for p in parts):
            try:
                return f"{int(float(parts[0]))}x{int(float(parts[1]))}"
            except Exception:
                pass

    bx = _safe_get(hdr, "BINX", default=None)
    by = _safe_get(hdr, "BINY", default=None)
    try:
        if bx is not None and by is not None:
            return f"{int(bx)}x{int(by)}"
    except Exception:
        pass
    return ""


def _window_from_header(hdr) -> str:
    # CCDSEC/DATASEC are typical: '[x1:x2,y1:y2]'
    for k in ("CCDSEC", "DATASEC", "TRIMSEC"):
        v = _safe_get(hdr, k, default=None)
        if v:
            return str(v)
    return ""

def _norm_obj(s: str | None) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s.strip().upper() if ch.isalnum())

def _norm_obj_tokens(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"[^0-9A-Z]+", " ", s.strip().upper()).strip()


def _is_sunsky_name(s: str | None) -> bool:
    """Detect SUNSKY frames by object/name strings.

    SUNSKY frames often carry MODE=... identical to science frames,
    so we use the name in nightlog/FITS header.
    """
    if not s:
        return False
    # Normalize: keep alnum only
    norm = "".join(ch for ch in s.strip().upper() if ch.isalnum())
    # Common patterns: SUNSKY, SUN_SKY, SUN-SKY, SUN SKY
    return ("SUNSKY" in norm)



def _safe_get(header, *keys, default=None):
    for k in keys:
        if k in header:
            return header.get(k)
    return default


def iter_fits_files(root: Path) -> Iterable[Path]:
    # универсально: рекурсивно ищем fits/fts
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".fits", ".fit", ".fts"}:
            yield p


def classify_frame(header) -> str:
    """
    Простейшая классификация. Мы её будем усиливать по мере интеграции твоих правил.
    """
    obj = str(_safe_get(header, "OBJECT", "OBJNAME", default="") or "")
    imtyp = str(_safe_get(header, "IMAGETYP", "OBSTYPE", "IMTYPE", default="") or "")
    exptime = _safe_get(header, "EXPTIME", "EXPOSURE", default=None)

    obj_n = _norm_obj(obj)
    obj_tokens = _norm_obj_tokens(obj)
    im_n = _norm_obj(imtyp)

    # bias: exptime==0 или по типу
    if exptime is not None:
        try:
            if float(exptime) == 0.0:
                return "bias"
        except Exception:
            pass
    if "BIAS" in im_n:
        return "bias"

    # flat
    if "FLAT" in im_n or "FLAT" in obj_n:
        return "flat"

    # lamp/neon/arc
    if re.search(r"\b(NEON|LAMP|ARC|AR|HG|HE)\b", obj_tokens) or "ARC" in im_n:
        return "neon"

    # sky / sunsky
    # NOTE: some observers mark scattered-light frames as "sunsky"; keep it separate from normal sky.
    if "SUNSKY" in obj_n:
        return "sunsky"
    if "SKY" in obj_n:
        return "sky"

    # всё остальное — object
    return "obj"


@dataclass(frozen=True)
class InspectResult:
    table: pd.DataFrame
    n_found: int
    n_opened: int
    nightlog_path: str | None
    n_nightlog_rows: int
    open_errors: list[str]  # первые N ошибок открытия FITS

    @property
    def objects(self) -> list[str]:
        df = self.table
        if df.empty or "kind" not in df.columns:
            return []
        df = df[df["kind"] == "obj"]
        if df.empty:
            return []
        return sorted(df["object"].dropna().unique().tolist())

def _fid_from_path(p: Path) -> str:
    # s23841404.fits  -> s23841404
    # s23841404.fts   -> s23841404
    # s23841404.fits.gz -> s23841404
    name = p.name.lower()
    for suf in (".fits.gz", ".fit.gz", ".fts.gz", ".fits", ".fit", ".fts"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return p.stem.lower()

def _open_fits_header_safe(fp: Path) -> fits.Header:
    """Open a FITS header in the most resilient way.

    Важно: для `inspect` нам НЕ нужно читать данные (это медленно). Форму
    кадра берём из NAXIS* в заголовке.
    """
    try:
        with open_fits_smart(fp, prefer_memmap=True) as hdul:
            return hdul[0].header
    except Exception:
        # часто помогает для "подуставших" .fts
        with fits.open(fp, memmap=False, ignore_missing_end=True, ignore_missing_simple=True) as hdul:
            return hdul[0].header
def inspect_dataset(data_dir: Path, max_files: int | None = None) -> InspectResult:
    rows = []
    n_found = 0
    n_opened = 0
    open_errors: list[str] = []

    log_path = find_nightlog(data_dir)
    log_meta = parse_nightlog(log_path) if log_path else {}

    for i, fp in enumerate(iter_fits_files(data_dir)):
        if max_files is not None and i >= max_files:
            break

        n_found += 1

        try:
            hdr = _open_fits_header_safe(fp)
        except Exception as e:
            if len(open_errors) < 5:
                open_errors.append(f"{fp} -> {type(e).__name__}: {e}")
            continue

        n_opened += 1

        fid = _fid_from_path(fp)

        obj = str(_safe_get(hdr, "OBJECT", "OBJNAME", default="") or "")
        kind = classify_frame(hdr)

        mode = str(_safe_get(hdr, "MODE", "OBSMODE", "OBS_MODE", default="") or "")
        disperser = str(_safe_get(hdr, "GRISM", "GRATING", "DISPERSER", "ELEMENT", default="") or "")
        slit = str(_safe_get(hdr, "SLIT", "SLITWID", "SLITW", "SLIT_WIDTH", default="") or "")

        # shape без чтения data
        shape = ""
        try:
            nx = int(_safe_get(hdr, "NAXIS1", default=0) or 0)
            ny = int(_safe_get(hdr, "NAXIS2", default=0) or 0)
            if nx > 0 and ny > 0:
                shape = f"{ny}x{nx}"
        except Exception:
            shape = ""

        instrument = guess_instrument_from_header(hdr) or str(_safe_get(hdr, "INSTRUME", "INSTRUMENT", default="") or "")
        row = dict(
            path=str(fp),
            fid=fid,
            kind=kind,
            object=obj,
            object_norm=_norm_obj(obj),
            date_obs=_safe_get(hdr, "DATE-OBS", default=None),
            exptime=_safe_get(hdr, "EXPTIME", "EXPOSURE", default=None),
            instrument=str(instrument or ""),
            mode=mode,
            disperser=disperser,
            slit=slit,
            binning=_binning_from_header(hdr),
            window=_window_from_header(hdr),
            shape=shape,
        )

        # override из ночного лога
        if fid in log_meta:
            m = log_meta[fid]
            row["object"] = m.object
            row["object_norm"] = _norm_obj(m.object)
            row["kind"] = m.kind
            if m.exptime == m.exptime:  # NaN guard
                row["exptime"] = m.exptime
            if m.mode:
                row["mode"] = m.mode
            if m.disperser:
                row["disperser"] = m.disperser
            if m.slit:
                row["slit"] = m.slit


        # SUNSKY force: MODE may match science frames, so detect by name
        try:
            if _is_sunsky_name(str(row.get("object", ""))) or _is_sunsky_name(obj):
                row["kind"] = "sunsky"
        except Exception:
            pass

        rows.append(row)

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    return InspectResult(
        df,
        n_found=n_found,
        n_opened=n_opened,
        nightlog_path=str(log_path) if log_path else None,
        n_nightlog_rows=len(log_meta),
        open_errors=open_errors,
    )
