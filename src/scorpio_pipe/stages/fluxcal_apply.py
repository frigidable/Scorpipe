from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scorpio_pipe.calib.common import ensure_refctx
from scorpio_pipe.calib.fluxcal import apply_fluxcal_to_spec1d, write_fluxcal_products
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
from scorpio_pipe.io.done_json import write_done_json
from scorpio_pipe.pipeline.context import StageResult


def _single_science_airmass_model(
    cfg: dict[str, Any], *, science_spec_path: str | Path, run_root: Path,
) -> dict[str, Any] | None:
    """Offer an atmospheric transfer only for one explicitly linked exposure.

    A configuration that merely lists one OBJ does not prove the reduction
    used it. Require Extract1D INPUT2D -> current Stack2D FITS -> completed
    Stack2D report with a single accepted input matching the raw OBJ stem.
    Multiple-source stacks cannot use an arithmetic mean airmass.
    """
    import numpy as np
    from astropy.io import fits
    from scorpio_pipe.instruments import parse_frame_meta
    from scorpio_pipe.calib.common import read_small_csv, reference_path
    from scorpio_pipe.paths import resolve_work_dir
    from scorpio_pipe.workspace_paths import stage_dir

    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    sources = list(frames.get("obj") or [])
    if len(sources) != 1 or not isinstance(sources[0], (str, Path)):
        return None
    raw = Path(str(sources[0])).expanduser()
    if not raw.is_file():
        return None
    wd = Path(resolve_work_dir(cfg))
    stack_path = stage_dir(wd, "stack2d") / "stack2d.fits"
    stack_report = stage_dir(wd, "stack2d") / "stack2d_done.json"
    if not stack_path.is_file() or not stack_report.is_file():
        return None
    try:
        with fits.open(science_spec_path, memmap=False) as hdul:
            input2d = str(hdul[0].header.get("INPUT2D") or "").strip()
        if not input2d:
            return None
        linked_2d = Path(input2d)
        if not linked_2d.is_absolute():
            linked_2d = wd / linked_2d
        if linked_2d.resolve() != stack_path.resolve():
            return None
        report = json.loads(stack_report.read_text(encoding="utf-8"))
        if not bool(report.get("ok")) or int(report.get("n_inputs", -1)) != 1:
            return None
        outpath = Path(str((report.get("outputs") or {}).get("stack2d_fits") or ""))
        if outpath.resolve() != stack_path.resolve():
            return None
        accepted = [
            rec for rec in (report.get("inputs") or [])
            if isinstance(rec, dict) and not bool(rec.get("rejected"))
        ]
        if len(accepted) != 1:
            return None
        processed = Path(str(accepted[0].get("path") or accepted[0].get("file") or ""))
        # Reduced products use the raw stem with an _skysub suffix. A stem
        # match is a useful admission guard, not a cryptographic RAW lineage.
        if processed.stem.casefold() != f"{raw.stem}_skysub".casefold():
            return None
        meta = parse_frame_meta(fits.getheader(raw, 0), strict=False)
        provenance = str(meta.provenance_class("airmass"))
        x = float(meta.airmass)
    except (ValueError, TypeError, OSError, KeyError, AttributeError, json.JSONDecodeError):
        return None
    if provenance not in {"header", "derived"} or not np.isfinite(x) or not (1.0 <= x <= 5.0):
        return None

    ext_path = reference_path(cfg, prefix="extinction_curve.")
    if ext_path is None or not ext_path.is_file():
        return None
    table = read_small_csv(ext_path)
    if "wavelength_A" not in table or "k_lambda" not in table:
        raise ValueError(
            "Science atmospheric reference needs wavelength_A and k_lambda"
        )
    return {
        "airmass": x,
        "provenance": provenance,
        "source_exposure": str(raw),
        "source_stack": str(stack_path),
        "lineage_qualification": "SINGLE_STACK_STEM_MATCH_ONLY",
        "extinction_reference": str(ext_path),
        "wavelength_ref": "air",
        "wavelength_A": table["wavelength_A"],
        "k_lambda": table["k_lambda"],
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
    # A corrected standard response does not correct the science target.
    # Apply the target's own atmospheric transfer only with a single,
    # auditable source, never with an invented robust-stack mean airmass.
    science_extinction = (
        _single_science_airmass_model(
            cfg, science_spec_path=spec_path, run_root=run_root,
        )
        if bool(sens_qc.get('extinction_applied')) else None
    )
    spec_payload = apply_fluxcal_to_spec1d(
        spec_path,
        sensfunc_path=sens_path,
        telluric_path=tel_path,
        standard_qc=qc,
        sensfunc_qc=sens_qc,
        science_extinction=science_extinction,
    )
    products = write_fluxcal_products(cfg, stage_dir_path, spec_payload=spec_payload, sensfunc_path=sens_path, telluric_path=tel_path, reference_context_id=str(refctx.get('context_id') or ''), science_spec_path=spec_path, standard_qc=qc)
    qcp = json.loads(Path(products['fluxcal_qc_json']).read_text(encoding='utf-8'))
    status = 'warn' if str(qcp.get('status') or '').upper() in {'RELATIVE', 'RELATIVE_TELLURIC', 'ABSOLUTE_APPROX'} else 'ok'
    write_done_json(stage='fluxcal_apply', stage_dir=stage_dir_path, status=status, outputs=products, metrics={'status': qcp.get('status')}, qc=qcp)
    return (StageResult.warn if status == 'warn' else StageResult.ok)(products=products, artifacts=list(products.values()), qc=qcp, messages=[f"fluxcal_apply {qcp.get('status')}"])
