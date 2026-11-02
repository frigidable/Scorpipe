from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scorpio_pipe.calib.common import ensure_refctx
from scorpio_pipe.calib.fluxcal import apply_fluxcal_to_spec1d, write_fluxcal_products
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
from scorpio_pipe.io.done_json import write_done_json
from scorpio_pipe.pipeline.context import StageResult


def _single_science_airmass_model(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Use only a single source exposure with explicit header/derived airmass.

    No effective airmass is invented for a variable-airmass robust stack:
    weighted transmission is wavelength- and position-dependent.
    """
    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    sources = list((frames or {}).get("obj") or [])
    if len(sources) != 1:
        return None
    from astropy.io import fits
    from scorpio_pipe.instruments import parse_frame_meta
    from scorpio_pipe.calib.common import read_small_csv, reference_path

    path = Path(str(sources[0]))
    if not path.is_file():
        return None
    try:
        meta = parse_frame_meta(fits.getheader(path, 0), strict=False)
        x = float(meta.airmass)
        provenance = str(meta.provenance_class("airmass"))
    except (ValueError, OSError, TypeError, KeyError, AttributeError):
        return None
    if provenance not in {"header", "derived"} or not (1.0 <= x <= 5.0):
        return None
    extinct = reference_path(cfg, prefix="extinction_curve.")
    if extinct is None or not extinct.is_file():
        return None
    tbl = read_small_csv(extinct)
    if "wavelength_A" not in tbl or "k_lambda" not in tbl:
        raise ValueError("Extinction reference must have wavelength_A and k_lambda")
    return {
        "airmass": x,
        "provenance": provenance,
        "source_exposure": str(path),
        "extinction_reference": str(extinct),
        "wavelength_A": tbl["wavelength_A"],
        "k_lambda": tbl["k_lambda"],
    }


def run_fluxcal_apply(cfg: dict[str, Any], *, run_root: Path, stage_dir_path: Path) -> StageResult:
    stage_dir_path.mkdir(parents=True, exist_ok=True)
    spec_path = PRODUCT_CATALOG.best_path(cfg, 'spec1d_fits', run_root=run_root)
    sens_path = PRODUCT_CATALOG.best_path(cfg, 'sensfunc_table_fits', run_root=run_root)
    try:
        tel_path = PRODUCT_CATALOG.best_path(cfg, 'telluric_model_fits', run_root=run_root)
    except Exception:
        tel_path = None
    qc_path = PRODUCT_CATALOG.best_path(cfg, 'standard_qc_json', run_root=run_root)
    qc = json.loads(qc_path.read_text(encoding='utf-8')) if qc_path.exists() else {}
    refctx = ensure_refctx(cfg)
    sens_qc_path = PRODUCT_CATALOG.best_path(cfg, 'sensfunc_qc_json', run_root=run_root)
    sens_qc = json.loads(sens_qc_path.read_text(encoding='utf-8')) if sens_qc_path.exists() else {}
    # Apply atmosphere independently to the SCIENCE target only with
    # verified single-exposure airmass. Multiple-exposure stacks need
    # per-exposure transmission weighting, not a guessed mean airmass.
    science_extinction = (
        _single_science_airmass_model(cfg)
        if bool(sens_qc.get("extinction_applied")) else None
    )
    spec_payload = apply_fluxcal_to_spec1d(
        spec_path, sensfunc_path=sens_path, telluric_path=tel_path,
        standard_qc=qc, sensfunc_qc=sens_qc,
        science_extinction=science_extinction,
    )
    products = write_fluxcal_products(cfg, stage_dir_path, spec_payload=spec_payload, sensfunc_path=sens_path, telluric_path=tel_path, reference_context_id=str(refctx.get('context_id') or ''), science_spec_path=spec_path, standard_qc=qc)
    qcp = json.loads(Path(products['fluxcal_qc_json']).read_text(encoding='utf-8'))
    status = 'warn' if str(qcp.get('status') or '').upper() in {'RELATIVE', 'RELATIVE_TELLURIC'} else 'ok'
    write_done_json(stage='fluxcal_apply', stage_dir=stage_dir_path, status=status, outputs=products, metrics={'status': qcp.get('status')}, qc=qcp)
    return (StageResult.warn if status == 'warn' else StageResult.ok)(products=products, artifacts=list(products.values()), qc=qcp, messages=[f"fluxcal_apply {qcp.get('status')}"])
