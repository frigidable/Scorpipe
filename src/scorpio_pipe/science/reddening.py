from __future__ import annotations

import math
from typing import Any


def ccm89_k_lambda(name: str) -> float | None:
    key = str(name or '').strip().lower()
    mapping = {
        'halpha': 2.53,
        'hβ': 3.61,
        'hbeta': 3.61,
        'hb': 3.61,
    }
    return mapping.get(key)


def balmer_decrement_summary(lines_by_name: dict[str, dict[str, Any]], *, intrinsic_ratio: float = 2.86) -> dict[str, Any] | None:
    ha = lines_by_name.get('Halpha') or lines_by_name.get('Hα') or lines_by_name.get('Ha')
    hb = lines_by_name.get('Hbeta') or lines_by_name.get('Hβ') or lines_by_name.get('Hb')
    if not isinstance(ha, dict) or not isinstance(hb, dict):
        return None
    try:
        fha = float(ha.get('flux'))
        fhb = float(hb.get('flux'))
    except Exception:
        return None
    if not math.isfinite(fha) or not math.isfinite(fhb) or fha <= 0 or fhb <= 0:
        return None
    ratio = fha / fhb
    kha = ccm89_k_lambda('halpha')
    khb = ccm89_k_lambda('hbeta')
    if kha is None or khb is None or khb == kha:
        return None
    ebv = (2.5 / (khb - kha)) * math.log10(max(ratio / intrinsic_ratio, 1e-6))
    av = 3.1 * ebv
    return {
        'line_ratio_Ha_Hb': float(ratio),
        'intrinsic_ratio': float(intrinsic_ratio),
        'ebv_ccm89': float(ebv),
        'av_ccm89': float(av),
    }
