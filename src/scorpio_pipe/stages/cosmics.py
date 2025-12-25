from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class CosmicsSummary:
    kind: str
    n_frames: int
    k: float
    replaced_pixels: int
    replaced_fraction: float
    per_frame_fraction: list[float]
    outputs: dict[str, str]


def _as_path(x: Any) -> Path:
    return x if isinstance(x, Path) else Path(str(x))


def _load_cfg_any(cfg: Any) -> dict[str, Any]:
    """Normalize config input (path/dict/RunContext) into a config dict."""
    from scorpio_pipe.config import load_config_any

    return load_config_any(cfg)


def _resolve_path(p: Path, *, data_dir: Path, work_dir: Path, base_dir: Path) -> Path:
    """Resolve a possibly-relative path.

    Preference order is chosen to match user expectation:
    1) data_dir (raw frames)
    2) work_dir (derived products)
    3) base_dir (config dir)
    """
    if p.is_absolute():
        return p
    for root in (data_dir, work_dir, base_dir):
        cand = (root / p).resolve()
        if cand.exists():
            return cand
    # last resort: resolve against work_dir even if it doesn't exist yet
    return (work_dir / p).resolve()


def _load_superbias(work_dir: Path) -> np.ndarray | None:
    # New layout (v5+): work_dir/calibs/*.fits, but keep legacy fallback too.
    for rel in (Path("calibs") / "superbias.fits", Path("calib") / "superbias.fits"):
        p = (work_dir / rel)
        if not p.is_file():
            continue
        try:
            return fits.getdata(p, memmap=False).astype(np.float32)
        except Exception:
            continue
    return None


def _robust_mad(x: np.ndarray, axis: int = 0) -> np.ndarray:
    med = np.median(x, axis=axis)
    mad = np.median(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)
    return mad


def _save_png(path: Path, arr: np.ndarray, title: str | None = None) -> None:
    # Optional visualization; avoid heavy dependencies.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 5), dpi=160)
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _stack_mad_clean(
    paths: list[Path],
    *,
    out_dir: Path,
    superbias: np.ndarray | None,
    k: float,
    bias_subtract: bool,
    save_png: bool,
    save_mask_fits: bool,
) -> CosmicsSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    if save_mask_fits:
        (out_dir / "masks_fits").mkdir(parents=True, exist_ok=True)

    datas: list[np.ndarray] = []
    headers: list[fits.Header] = []
    names: list[str] = []

    for p in paths:
        with fits.open(p) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()
        if bias_subtract and superbias is not None and superbias.shape == data.shape:
            data = data - superbias
            hdr["BIASSUB"] = (True, "Superbias subtracted")
            hdr["HISTORY"] = "scorpio_pipe cosmics: bias subtracted using superbias.fits"
        datas.append(data)
        headers.append(hdr)
        names.append(p.stem)

    stack = np.stack(datas, axis=0)  # (N, H, W)
    med = np.median(stack, axis=0)
    mad = _robust_mad(stack, axis=0)

    # Protect against zero MAD (flat pixels)
    eps = np.finfo(np.float32).eps
    mad = np.maximum(mad, eps)

    thr = (k * mad).astype(np.float32)
    mask = np.abs(stack - med[None, :, :]) > thr[None, :, :]

    # Replace cosmics by per-pixel median of the stack
    cleaned = np.where(mask, med[None, :, :], stack)

    replaced_pixels = int(mask.sum())
    replaced_fraction = float(replaced_pixels) / float(stack.size)
    per_frame_fraction = [float(mask[i].mean()) for i in range(mask.shape[0])]

    # Write per-frame cleaned files
    out_files: dict[str, str] = {}
    for i, (name, hdr) in enumerate(zip(names, headers)):
        out_f = out_dir / "clean" / f"{name}_clean.fits"
        h = hdr.copy()
        h["COSMCLEA"] = (True, "Cosmics cleaned")
        h["COSM_K"] = (float(k), "MAD threshold multiplier")
        h["COSM_MD"] = ("stack_mad", "Cosmics method")
        h["HISTORY"] = "scorpio_pipe cosmics: replaced cosmic pixels with stack median"
        fits.writeto(out_f, cleaned[i].astype(np.float32), header=h, overwrite=True)
        out_files[name] = str(out_f)

        if save_png:
            # quicklook mask
            mpath = out_dir / "masks" / f"{name}_mask.png"
            _save_png(mpath, mask[i].astype(np.uint8), title=f"Cosmic mask: {name}")

        if save_mask_fits:
            mf = out_dir / "masks_fits" / f"{name}_mask.fits"
            # uint16 mask: 1 = cosmic (first reserved bit)
            fits.writeto(mf, (mask[i].astype(np.uint16)) * 1, overwrite=True)

    # Reference products: sum excluding masked pixels + coverage map
    sum_excl = np.sum(stack * (~mask), axis=0)
    cov = np.sum(~mask, axis=0).astype(np.int16)

    sum_f = out_dir / "sum_excl_cosmics.fits"
    cov_f = out_dir / "coverage.fits"
    fits.writeto(sum_f, sum_excl.astype(np.float32), overwrite=True)
    fits.writeto(cov_f, cov, overwrite=True)

    if save_png:
        _save_png(out_dir / "sum_excl_cosmics.png", sum_excl, title="Sum (cosmics excluded)")
        _save_png(out_dir / "coverage.png", cov, title="Coverage (non-cosmic count)")

    outputs = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "sum_excl_fits": str(sum_f.resolve()),
        "coverage_fits": str(cov_f.resolve()),
    }
    if save_png:
        outputs.update(
            {
                "sum_excl_png": str((out_dir / "sum_excl_cosmics.png").resolve()),
                "coverage_png": str((out_dir / "coverage.png").resolve()),
                "masks_dir": str((out_dir / "masks").resolve()),
            }
        )
    if save_mask_fits:
        outputs["masks_fits_dir"] = str((out_dir / "masks_fits").resolve())

    return CosmicsSummary(
        kind="",
        n_frames=len(paths),
        k=float(k),
        replaced_pixels=replaced_pixels,
        replaced_fraction=replaced_fraction,
        per_frame_fraction=per_frame_fraction,
        outputs=outputs,
    )


def clean_cosmics(cfg: Any, *, out_dir: str | Path | None = None) -> Path:
    """Clean cosmics and write a report.

    Default method is the robust stack-based MAD detection inspired by the user's
    `SKY_MODEL Object Cosmic.py` workflow: build per-pixel median/MAD across a
    stack of exposures, mask outliers, and replace them by the median.

    Outputs:
      work_dir/cosmics/<kind>/clean/*.fits
      work_dir/cosmics/<kind>/summary.json
    """
    cfg = _load_cfg_any(cfg)
    base_dir = Path(str(cfg.get("config_dir", "."))).resolve()
    data_dir = Path(str(cfg.get("data_dir", "."))).expanduser().resolve()

    work_dir = _as_path(cfg.get("work_dir", "work"))
    if not work_dir.is_absolute():
        work_dir = (base_dir / work_dir).resolve()

    ccfg = cfg.get("cosmics", {}) or {}
    if not bool(ccfg.get("enabled", True)):
        out_root = Path(out_dir) if out_dir is not None else (work_dir / "cosmics")
        if not out_root.is_absolute():
            out_root = (work_dir / out_root).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"disabled": True}, f, indent=2, ensure_ascii=False)
        return out_path.resolve()

    method = str(ccfg.get("method", "stack_mad")).strip().lower()
    apply_to = ccfg.get("apply_to", ["obj"]) or ["obj"]
    if isinstance(apply_to, str):
        apply_to = [apply_to]

    # Threshold: prefer explicit k; keep backward compat with older sigma_clip
    k = ccfg.get("k", None)
    if k is None:
        k = ccfg.get("sigma_clip", 9.0)
    try:
        k = float(k)
    except Exception:
        k = 9.0

    bias_subtract = bool(ccfg.get("bias_subtract", True))
    save_png = bool(ccfg.get("save_png", True))
    save_mask_fits = bool(ccfg.get("save_mask_fits", True))

    superbias = _load_superbias(work_dir) if bias_subtract else None

    out_root = Path(out_dir) if out_dir is not None else (work_dir / "cosmics")
    if not out_root.is_absolute():
        out_root = (work_dir / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    frames = cfg.get("frames", {}) or {}

    all_summaries: list[dict[str, Any]] = []

    for kind in apply_to:
        kind = str(kind)
        rel_paths = frames.get(kind, []) or []
        if not isinstance(rel_paths, (list, tuple)):
            continue

        paths = [_resolve_path(_as_path(pp), data_dir=data_dir, work_dir=work_dir, base_dir=base_dir) for pp in rel_paths]
        paths = [p for p in paths if p.is_file()]

        if len(paths) == 0:
            continue

        kind_out = out_root / kind
        kind_out.mkdir(parents=True, exist_ok=True)

        summary: CosmicsSummary
        if method in ("stack_mad", "mad_stack", "stack") and len(paths) >= 2:
            # Ensure all frames have the same shape
            shapes = []
            for pth in paths:
                try:
                    with fits.open(pth) as hdul:
                        shapes.append(np.asarray(hdul[0].data).shape)
                except Exception:
                    shapes.append(None)
            shapes_ok = all(s == shapes[0] and s is not None for s in shapes)

            if shapes_ok:
                summary = _stack_mad_clean(
                    paths,
                    out_dir=kind_out,
                    superbias=superbias,
                    k=k,
                    bias_subtract=bias_subtract,
                    save_png=save_png,
                    save_mask_fits=save_mask_fits,
                )
                summary = CosmicsSummary(
                    kind=kind,
                    n_frames=summary.n_frames,
                    k=summary.k,
                    replaced_pixels=summary.replaced_pixels,
                    replaced_fraction=summary.replaced_fraction,
                    per_frame_fraction=summary.per_frame_fraction,
                    outputs=summary.outputs,
                )
            else:
                # Fallback: just copy files to make the stage non-fatal
                (kind_out / "clean").mkdir(parents=True, exist_ok=True)
                outputs = {"note": "shape mismatch: skipped stack MAD cleaning"}
                summary = CosmicsSummary(kind=kind, n_frames=len(paths), k=float(k), replaced_pixels=0, replaced_fraction=0.0,
                                         per_frame_fraction=[0.0]*len(paths), outputs=outputs)
        else:
            outputs = {"note": f"method={method} not supported in this build"}
            summary = CosmicsSummary(kind=kind, n_frames=len(paths), k=float(k), replaced_pixels=0, replaced_fraction=0.0,
                                     per_frame_fraction=[0.0]*len(paths), outputs=outputs)

        all_summaries.append(
            {
                "kind": summary.kind,
                "n_frames": summary.n_frames,
                "method": method,
                "k": summary.k,
                "replaced_pixels": summary.replaced_pixels,
                "replaced_fraction": summary.replaced_fraction,
                "per_frame_fraction": summary.per_frame_fraction,
                "outputs": summary.outputs,
            }
        )

        # write per-kind summary too
        with (kind_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(all_summaries[-1], f, indent=2, ensure_ascii=False)

    out = {
        "method": method,
        "k": float(k),
        "bias_subtract": bias_subtract,
        "save_png": save_png,
        "items": all_summaries,
    }

    out_path = out_root / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out_path.resolve()
