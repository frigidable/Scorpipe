from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.wavesol_paths import get_selected_disperser, slugify_disperser


def _norm_key(value: str | None) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


@dataclass(frozen=True)
class LongSetupPrior:
    setup: str
    anchor_file: str
    lambda_min_A: float
    lambda_max_A: float
    median_dispersion_A_per_px: float
    n_anchor_points: int
    line_table_count: int | None = None
    strongest_lines_A: tuple[float, ...] = ()

    @property
    def slug(self) -> str:
        return slugify_disperser(self.setup)

    @property
    def coverage_A(self) -> float:
        return float(max(0.0, self.lambda_max_A - self.lambda_min_A))

    @property
    def trimmed_margin_A(self) -> float:
        disp = float(max(0.05, self.median_dispersion_A_per_px))
        return float(max(8.0, min(25.0, 10.0 * disp)))

    @property
    def trimmed_lambda_min_A(self) -> float:
        return float(self.lambda_min_A + self.trimmed_margin_A)

    @property
    def trimmed_lambda_max_A(self) -> float:
        return float(self.lambda_max_A - self.trimmed_margin_A)

    @property
    def suggested_pair_tol_A(self) -> float:
        disp = float(max(0.05, self.median_dispersion_A_per_px))
        return float(min(4.25, max(1.25, 2.75 * disp)))

    @property
    def suggested_pair_top_lines(self) -> int:
        if self.line_table_count is not None and self.line_table_count > 0:
            return int(min(160, max(40, self.line_table_count)))
        guess = int(np.clip(round(self.coverage_A / max(25.0, 20.0 * self.median_dispersion_A_per_px)), 40, 140))
        return int(guess)

    @property
    def suggested_peak_target_min(self) -> int:
        return int(min(18, max(8, round(self.coverage_A / max(220.0, 180.0 * self.median_dispersion_A_per_px)))))

    @property
    def suggested_peak_target_max(self) -> int:
        return int(max(self.suggested_peak_target_min + 8, min(40, round(self.coverage_A / max(90.0, 75.0 * self.median_dispersion_A_per_px)))))

    @property
    def empirical_resolution_fwhm_A_1arcsec(self) -> float:
        # LONG anchor tables do not directly contain instrumental FWHM, but the setups imply a useful
        # empirical starting point.  A 6-pixel FWHM is conservative for SCORPIO long-slit arc profiles.
        return float(max(1.8, 6.0 * self.median_dispersion_A_per_px))

    def as_dict(self) -> dict[str, Any]:
        return {
            'setup': self.setup,
            'slug': self.slug,
            'anchor_file': self.anchor_file,
            'lambda_min_A': float(self.lambda_min_A),
            'lambda_max_A': float(self.lambda_max_A),
            'trimmed_lambda_min_A': float(self.trimmed_lambda_min_A),
            'trimmed_lambda_max_A': float(self.trimmed_lambda_max_A),
            'median_dispersion_A_per_px': float(self.median_dispersion_A_per_px),
            'n_anchor_points': int(self.n_anchor_points),
            'line_table_count': int(self.line_table_count) if self.line_table_count is not None else None,
            'strongest_lines_A': [float(v) for v in self.strongest_lines_A],
            'suggested_pair_tol_A': float(self.suggested_pair_tol_A),
            'suggested_pair_top_lines': int(self.suggested_pair_top_lines),
            'suggested_peak_target_min': int(self.suggested_peak_target_min),
            'suggested_peak_target_max': int(self.suggested_peak_target_max),
            'empirical_resolution_fwhm_A_1arcsec': float(self.empirical_resolution_fwhm_A_1arcsec),
            'source': 'LONG anchor/line tables',
        }


@lru_cache(maxsize=1)
def _load_payload() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / 'resources' / 'setup_priors' / 'long_setup_priors.json'
    return json.loads(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def _load_prior_map() -> dict[str, LongSetupPrior]:
    payload = _load_payload()
    out: dict[str, LongSetupPrior] = {}
    for row in list(payload.get('setups') or []):
        try:
            prior = LongSetupPrior(
                setup=str(row.get('setup') or ''),
                anchor_file=str(row.get('anchor_file') or ''),
                lambda_min_A=float(row['lambda_min_A']),
                lambda_max_A=float(row['lambda_max_A']),
                median_dispersion_A_per_px=float(row['median_dispersion_A_per_px']),
                n_anchor_points=int(row.get('n_anchor_points') or 0),
                line_table_count=int(row['line_table_count']) if row.get('line_table_count') is not None else None,
                strongest_lines_A=tuple(float(v) for v in list(row.get('strongest_lines_A') or [])),
            )
        except Exception:
            continue
        out[_norm_key(prior.setup)] = prior
        out[_norm_key(prior.slug)] = prior
    return out


def available_long_setup_priors() -> list[LongSetupPrior]:
    seen: set[str] = set()
    out: list[LongSetupPrior] = []
    for prior in _load_prior_map().values():
        if prior.setup in seen:
            continue
        seen.add(prior.setup)
        out.append(prior)
    out.sort(key=lambda p: p.setup)
    return out


def get_long_setup_prior(setup_or_cfg: str | dict | None) -> LongSetupPrior | None:
    if isinstance(setup_or_cfg, dict):
        key = _norm_key(get_selected_disperser(setup_or_cfg))
    else:
        key = _norm_key(str(setup_or_cfg or ''))
    if not key:
        return None
    return _load_prior_map().get(key)


def filter_lab_lines_for_setup(
    lam_lab: np.ndarray,
    w_lab: np.ndarray | None,
    *,
    setup_or_cfg: str | dict | None,
    margin_A: float | None = None,
    boost_strong_lines: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lam = np.asarray(lam_lab, dtype=float)
    if w_lab is None:
        wt = np.ones_like(lam, dtype=float)
    else:
        wt = np.asarray(w_lab, dtype=float)
        if wt.shape != lam.shape:
            wt = np.ones_like(lam, dtype=float)
    m = np.isfinite(lam) & np.isfinite(wt) & (wt > 0)
    lam = lam[m]
    wt = wt[m]

    prior = get_long_setup_prior(setup_or_cfg)
    if prior is None or lam.size == 0:
        return lam, wt, {'applied': False}

    pad = float(margin_A if margin_A is not None else max(10.0, 3.0 * prior.median_dispersion_A_per_px))
    lo = float(prior.lambda_min_A - pad)
    hi = float(prior.lambda_max_A + pad)
    keep = (lam >= lo) & (lam <= hi)
    lam2 = lam[keep]
    wt2 = wt[keep].astype(float, copy=True)

    boosted: list[float] = []
    if boost_strong_lines and lam2.size:
        rad = float(max(0.35, 0.75 * prior.median_dispersion_A_per_px))
        for center in prior.strongest_lines_A[:12]:
            near = np.abs(lam2 - float(center)) <= rad
            if np.any(near):
                wt2[near] *= 2.5
                boosted.append(float(center))

    meta = {
        'applied': True,
        'setup': prior.setup,
        'coverage_A': [float(prior.lambda_min_A), float(prior.lambda_max_A)],
        'trimmed_coverage_A': [float(prior.trimmed_lambda_min_A), float(prior.trimmed_lambda_max_A)],
        'median_dispersion_A_per_px': float(prior.median_dispersion_A_per_px),
        'pair_tol_A': float(prior.suggested_pair_tol_A),
        'input_n_lines': int(lam.size),
        'kept_n_lines': int(lam2.size),
        'boosted_lines_A': boosted,
        'source': 'LONG anchor/line tables',
    }
    return lam2, wt2, meta


__all__ = [
    'LongSetupPrior',
    'available_long_setup_priors',
    'get_long_setup_prior',
    'filter_lab_lines_for_setup',
]
