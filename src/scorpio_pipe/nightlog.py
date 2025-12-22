from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class NightLogRow:
    fid: str            # s23841309
    object: str         # ngc2784 / AGK+81d266 / etc
    kind: str           # bias/flat/neon/obj/sky...
    start: str          # HH:MM:SS
    exptime: float
    mode: str           # Image / Spectra / ImaSlit
    disperser: str      # VPHG1200@540 (может быть пусто)
    slit: str           # 1.00 / 0.46 / ...
    filt: str           # R/V/B... (может быть пусто)


def _to_float(x: str) -> float:
    try:
        return float(x.strip())
    except Exception:
        return float("nan")


def _norm_kind(k: str) -> str:
    k = (k or "").strip().lower()
    # типы из логов бывают "object", "obj", "science" и т.п.
    if k in {"object", "obj", "science"}:
        return "obj"
    return k


def parse_nightlog(path: Path) -> Dict[str, NightLogRow]:
    """
    Парсим строки формата:
    |s23841309|ngc2784|neon|02:31:40|40|Spectra|VPHG1200@540|1.00|...|R|...
    Мы берём ключевые колонки по фиксированным позициям (как в вашем логе),
    а фильтр пытаемся вытащить мягко.
    """
    meta: Dict[str, NightLogRow] = {}
    if not path.exists():
        return meta

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s.startswith("|s"):
            continue

        parts = [p.strip() for p in s.split("|")]
        # parts[0] == "" из-за ведущего "|"
        if len(parts) < 9:
            continue

        fid = parts[1].lower()
        obj = parts[2]
        kind = _norm_kind(parts[3])
        start = parts[4] if len(parts) > 4 else ""
        exptime = _to_float(parts[5]) if len(parts) > 5 else float("nan")
        mode = parts[6] if len(parts) > 6 else ""
        disperser = parts[7] if len(parts) > 7 else ""
        slit = parts[8] if len(parts) > 8 else ""

        # фильтр в логе может “гулять”, поэтому ищем самое похожее
        filt = ""
        for cand in reversed(parts):
            c = cand.strip().upper()
            if c in {"U", "B", "V", "R", "I"}:
                filt = c
                break

        meta[fid] = NightLogRow(
            fid=fid,
            object=obj,
            kind=kind,
            start=start,
            exptime=exptime,
            mode=mode,
            disperser=disperser,
            slit=slit,
            filt=filt,
        )

    return meta


def find_nightlog(data_dir: Path) -> Optional[Path]:
    """
    Ищем txt, где реально есть строки кадров вида "|s2384...|".
    Берём файл с максимальным числом таких строк.
    """
    candidates = list(data_dir.glob("s*.txt")) + list(data_dir.glob("*.txt"))
    best = None
    best_score = -1
    best_size = -1

    for p in candidates:
        try:
            # читаем только начало, чтобы не тянуть мегабайты
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                head = f.read(200_000)
        except Exception:
            continue

        score = sum(1 for line in head.splitlines() if line.lstrip().startswith("|s"))
        if score > best_score or (score == best_score and p.stat().st_size > best_size):
            best = p
            best_score = score
            best_size = p.stat().st_size

    return best

