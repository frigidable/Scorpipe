from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.airvac import air_to_vac, vac_to_air
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.refs.context import ensure_reference_context, load_reference_context


def slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(s or "").strip()).strip("_") or "default"


def normalize_name(s: str) -> str:
    return "".join(ch.lower() for ch in str(s or "") if ch.isalnum())


def safe_json(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [safe_json(v) for v in x]
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return str(x)


def ensure_refctx(cfg: dict[str, Any]) -> dict[str, Any]:
    ctx = load_reference_context(resolve_work_dir(cfg))
    if isinstance(ctx, dict) and ctx.get("context_id"):
        return ctx
    return ensure_reference_context(cfg)


def reference_entry(cfg: dict[str, Any], *, exact: str | None = None, prefix: str | None = None) -> dict[str, Any] | None:
    ctx = ensure_refctx(cfg)
    for ref in ctx.get("references", []) or []:
        name = str(ref.get("name") or "")
        if exact is not None and name == exact:
            return dict(ref)
        if prefix is not None and name.startswith(prefix):
            return dict(ref)
    return None


def reference_path(cfg: dict[str, Any], *, exact: str | None = None, prefix: str | None = None) -> Path | None:
    ref = reference_entry(cfg, exact=exact, prefix=prefix)
    if ref is None:
        return None
    p = Path(str(ref.get("resolved") or ""))
    if p.is_absolute():
        return p
    return (resolve_work_dir(cfg) / p).resolve()


def reference_metadata(cfg: dict[str, Any], *, exact: str | None = None, prefix: str | None = None) -> dict[str, Any]:
    ref = reference_entry(cfg, exact=exact, prefix=prefix)
    if ref is None:
        return {}
    meta = dict(ref.get("metadata") or {})
    meta.setdefault("name", ref.get("name"))
    meta.setdefault("requested", ref.get("requested"))
    meta.setdefault("resolved", ref.get("resolved"))
    meta.setdefault("version", ref.get("version"))
    meta.setdefault("sha256", ref.get("sha256"))
    return meta


def read_small_csv(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    rows = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(ln for ln in f if not ln.lstrip().startswith("#")):
            rows.append(row)
    if not rows:
        return {}
    header = [str(x).strip() for x in rows[0]]
    cols = {h: [] for h in header}
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        for h, v in zip(header, row):
            try:
                cols[h].append(float(v))
            except Exception:
                cols[h].append(v)
    out: dict[str, np.ndarray] = {}
    for k, vals in cols.items():
        try:
            out[k] = np.asarray(vals, dtype=float)
        except Exception:
            out[k] = np.asarray(vals, dtype=object)
    return out


def read_interval_csv(path: str | Path) -> list[tuple[float, float, str]]:
    tbl = read_small_csv(path)
    if not tbl:
        return []
    lo = tbl["lambda_min_A"] if "lambda_min_A" in tbl else tbl.get("lambda_min")
    hi = tbl["lambda_max_A"] if "lambda_max_A" in tbl else tbl.get("lambda_max")
    labels = tbl.get("label")
    out = []
    if lo is None or hi is None:
        return out
    n = min(len(lo), len(hi))
    for i in range(n):
        lab = ""
        if labels is not None and i < len(labels):
            lab = str(labels[i])
        out.append((float(lo[i]), float(hi[i]), lab))
    return out


def load_library_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return dict(obj) if isinstance(obj, dict) else {}


def find_standard_record(library: dict[str, Any], selected_name: str | None) -> tuple[str | None, dict[str, Any] | None]:
    stds = library.get("standards") if isinstance(library, dict) else None
    if not isinstance(stds, dict) or not stds:
        return None, None
    if selected_name:
        nsel = normalize_name(selected_name)
        for key, rec in stds.items():
            disp = str((rec or {}).get("display_name") or key)
            if normalize_name(disp) == nsel or normalize_name(str(key)) == nsel:
                return str(key), dict(rec)
    first_key = next(iter(stds.keys()))
    return str(first_key), dict(stds[first_key])


def air_to_vacuum(lam_air: np.ndarray) -> np.ndarray:
    """Compatibility entrypoint for the common standard-air kernel (Angstrom)."""
    return air_to_vac(lam_air)


def vacuum_to_air(lam_vac: np.ndarray) -> np.ndarray:
    """Compatibility entrypoint for the common standard-air kernel (Angstrom)."""
    return vac_to_air(lam_vac)


def convert_wavelength_convention(wave: np.ndarray, *, from_ref: str, to_ref: str) -> np.ndarray:
    """Convert only explicitly declared air/vacuum wavelengths, without metadata I/O."""
    a = str(from_ref).strip().lower()
    b = str(to_ref).strip().lower()
    if a in {"vac", "vacuum"}:
        a = "vacuum"
    if b in {"vac", "vacuum"}:
        b = "vacuum"
    if a not in {"air", "vacuum"} or b not in {"air", "vacuum"}:
        raise ValueError(f"Unknown wavelength convention: {from_ref!r} -> {to_ref!r}")
    if a == b:
        # No medium conversion, but do not discard imaginary parts or accept
        # physically undefined wavelengths. UV identity needs no air model.
        if np.iscomplexobj(wave):
            raise ValueError("Wavelengths must be real, finite, positive values")
        unchanged = np.asarray(wave, dtype=float)
        if np.any(~np.isfinite(unchanged)) or np.any(unchanged <= 0):
            raise ValueError("Wavelengths must be real, finite, positive values")
        return unchanged
    if a == "air" and b == "vacuum":
        return air_to_vacuum(wave)
    return vacuum_to_air(wave)


def read_spec1d(path: str | Path, *, mode: str = "trace") -> dict[str, Any]:
    p = Path(path)
    from astropy.io import fits
    with fits.open(p, memmap=False) as hdul:
        hdr0 = hdul[0].header.copy()
        extname = "SPEC_TRACE" if str(mode).lower() != "fixed" else "SPEC_FIXED"
        tab = hdul[extname].data
        if extname == "SPEC_TRACE":
            flux_col, var_col, mask_col, sky_col, npix_col = "FLUX_TRACE", "VAR_TRACE", "MASK_TRACE", "SKY_TRACE", "NPIX_TRACE"
        else:
            flux_col, var_col, mask_col, sky_col, npix_col = "FLUX_FIXED", "VAR_FIXED", "MASK_FIXED", "SKY_FIXED", "NPIX_FIXED"
        return {
            "header": hdr0,
            "wave": np.asarray(tab["LAMBDA"], dtype=float),
            "flux": np.asarray(tab[flux_col], dtype=float),
            "var": np.asarray(tab[var_col], dtype=float),
            "mask": np.asarray(tab[mask_col]),
            "sky": np.asarray(tab[sky_col], dtype=float) if sky_col in tab.names else np.zeros(len(tab), dtype=float),
            "npix": np.asarray(tab[npix_col], dtype=float) if npix_col in tab.names else np.zeros(len(tab), dtype=float),
            "waveref": str(hdr0.get("WAVEREF") or "air").strip().lower(),
            "bunit": str(hdr0.get("BUNIT") or hdr0.get("FLUXUNIT") or "ADU").strip(),
            "path": str(p),
        }


def write_bintable_fits(path: str | Path, *, primary_header: fits.Header | None = None, extname: str, columns: dict[str, Any]) -> Path:
    cols = []
    for name, arr in columns.items():
        aa = np.asarray(arr)
        if aa.dtype.kind in 'if':
            fmt = 'D' if aa.dtype.kind == 'f' and aa.dtype.itemsize > 4 else 'E'
            if aa.dtype.kind == 'i':
                fmt = 'J'
        else:
            aa = aa.astype(str)
            width = max(1, max(len(str(x)) for x in aa))
            fmt = f'{width}A'
        cols.append(fits.Column(name=str(name), format=fmt, array=aa))
    from astropy.io import fits
    hdul = fits.HDUList([fits.PrimaryHDU(header=(fits.Header() if primary_header is None else primary_header)), fits.BinTableHDU.from_columns(cols, name=str(extname))])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(path, overwrite=True)
    return path


def save_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(safe_json(payload), indent=2, ensure_ascii=False), encoding='utf-8')
    return p


def plot_lines_png(path: str | Path, series: list[tuple[np.ndarray, np.ndarray, str]], *, title: str, vspans: list[tuple[float, float]] | None = None) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    for x, y, label in series:
        ax.plot(np.asarray(x, dtype=float), np.asarray(y, dtype=float), lw=1.0, label=str(label))
    for a, b in vspans or []:
        ax.axvspan(float(a), float(b), alpha=0.12)
    ax.set_title(str(title))
    ax.set_xlabel('Wavelength [A]')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p
