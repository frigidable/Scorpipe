#!/usr/bin/env python3
"""Read-only SCORPIO-1 fast-readout FLAT spatial variation QA.

For the audited N2146sat night only. Candidate side-band geometry, the
physical CCD crop, header GAIN units and bias correction are NOT certified.
Separates coherent illumination variations from high-frequency flat residuals.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

FLATS = ("s22800614.fts", "s22800615.fts", "s22800616.fts")
SHAPE = (1032, 2080)
ACTIVE = (slice(32, 1000), slice(32, 2048))
LEFT = (slice(32, 1000), slice(2, 14))
RIGHT = (slice(32, 1000), slice(2066, 2078))


def clipped_mean(data):
    v = np.asarray(data, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 100:
        raise ValueError("Insufficient candidate side-band samples")
    for _ in range(4):
        center = np.median(v)
        sigma = 1.4826 * np.median(np.abs(v-center))
        if sigma <= 0:
            sigma = float(np.std(v))
        if sigma <= 0:
            break
        good = np.abs(v-center) <= 4*sigma
        if good.sum() < 100:
            raise RuntimeError("Candidate side-band clipping removed all samples")
        if good.all():
            break
        v = v[good]
    return float(np.mean(v))


def robust_sigma(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(1.4826*np.median(np.abs(v-np.median(v))))


def spatial_pair(a, b, *, gain=3.5, readnoise_adu=2.1, tile=64):
    """Sparse-coverage-safe diagnostic; no interpretation as certified flat."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Flat dimensions mismatch")
    if gain <= 0 or readnoise_adu < 0 or tile <= 0:
        raise ValueError("Nonphysical diagnostic gain/noise/tile")
    h = (a.shape[0]//tile)*tile
    w = (a.shape[1]//tile)*tile
    if h < tile or w < tile:
        raise ValueError("Image too small for requested tiles")
    a = np.asarray(a[:h, :w], dtype=np.float64)
    b = np.asarray(b[:h, :w], dtype=np.float64)
    valid = (np.isfinite(a) & np.isfinite(b) & (a > 2500) & (b > 2500)
             & (a < 60000) & (b < 60000))
    if valid.sum() < 1000:
        raise RuntimeError("Insufficient valid bright flat pixels")
    scale = float(np.median(a[valid]/b[valid]))
    ratio = np.full(a.shape, np.nan, dtype=np.float64)
    ratio[valid] = a[valid]/(scale*b[valid])-1.0
    ny, nx = h//tile, w//tile
    tiles = ratio.reshape(ny, tile, nx, tile).transpose(
        0, 2, 1, 3
    ).reshape(ny, nx, tile*tile)
    flags = valid.reshape(ny, tile, nx, tile).transpose(
        0, 2, 1, 3
    ).reshape(ny, nx, tile*tile)
    counts = flags.sum(axis=-1)
    tile_fraction = counts.astype(float)/(tile*tile)
    enough = counts >= int(np.ceil(0.75*tile*tile))
    n_tiles = int(enough.size)
    n_used = int(enough.sum())
    # Allow edge/low-illumination tiles, but reject unusably narrow coverage.
    min_tiles = min(n_tiles, max(8, int(np.ceil(0.10*n_tiles))))
    if n_used < min_tiles:
        raise RuntimeError(
            f"Insufficient spatial coverage: {n_used}/{n_tiles} tiles "
            f"with >=75% valid bright pixels (require {min_tiles})"
        )

    tile_med = np.full((ny, nx), np.nan, dtype=np.float64)
    for iy, ix in zip(*np.nonzero(enough)):
        tile_med[iy, ix] = float(np.median(tiles[iy, ix, flags[iy, ix]]))

    # Use the locally estimated ratio only in sufficiently illuminated tiles.
    model_rel = np.repeat(np.repeat(tile_med, tile, axis=0), tile, axis=1)
    selected = valid & np.repeat(np.repeat(enough, tile, axis=0), tile, axis=1)
    local_scale = scale*(1.0+model_rel[selected])
    aa = a[selected]
    bb = b[selected]
    # Noise model is conditional on GAIN being e-/ADU; not instrument-certified.
    predicted = np.sqrt(
        (np.maximum(aa, 0.0)+local_scale**2*np.maximum(bb, 0.0))/gain
        +(1.0+local_scale**2)*readnoise_adu**2
    )
    z = (aa-local_scale*bb)/predicted

    # Missing tiles make naive row/column variance decomposition biased.
    # Calculate sequential explained-variance shares on only observed tiles:
    # dispersion-X first, slit-Y conditional on X, then interaction residual.
    iy, ix = np.nonzero(enough)
    v = tile_med[enough]
    grand = float(np.mean(v))
    counts_x = np.bincount(ix, minlength=nx)
    counts_y = np.bincount(iy, minlength=ny)
    pred_x = np.bincount(ix, weights=v, minlength=nx)[ix]/counts_x[ix]
    ss0 = float(np.sum((v-grand)**2))
    ssx = float(np.sum((v-pred_x)**2))
    x_effect = np.zeros(nx, dtype=np.float64)
    y_effect = np.zeros(ny, dtype=np.float64)
    for _ in range(100):
        x_next = np.bincount(
            ix, weights=v-grand-y_effect[iy], minlength=nx
        )/np.maximum(counts_x, 1)
        y_next = np.bincount(
            iy, weights=v-grand-x_next[ix], minlength=ny
        )/np.maximum(counts_y, 1)
        change = max(
            float(np.max(np.abs(x_next-x_effect))),
            float(np.max(np.abs(y_next-y_effect))),
        )
        x_effect, y_effect = x_next, y_next
        if change < 1e-12:
            break
    ssxy = float(np.sum((v-grand-x_effect[ix]-y_effect[iy])**2))
    if ss0 > 1e-24:
        shares = np.maximum(
            [ss0-ssx, ssx-ssxy, ssxy], 0.0
        )
        shares = shares/np.sum(shares)
        fractions = dict(
            dispersion_x=float(shares[0]),
            slit_y=float(shares[1]),
            interaction_xy=float(shares[2]),
        )
    else:
        fractions = dict(dispersion_x=0.0, slit_y=0.0, interaction_xy=0.0)

    return dict(
        tile_size=tile,
        n_pixels=int(valid.sum()),
        n_pixels_in_accepted_tiles=int(selected.sum()),
        n_tiles=n_used,
        n_tiles_total=n_tiles,
        n_tiles_excluded_low_coverage=n_tiles-n_used,
        fraction_tiles_accepted=n_used/n_tiles,
        fraction_pixels_bright_and_valid=float(valid.mean()),
        fraction_selected_among_valid=float(selected.sum()/valid.sum()),
        n_pixels_below_brightness_threshold=int(np.sum(
            np.isfinite(a) & np.isfinite(b)
            & ((a <= 2500) | (b <= 2500))
        )),
        n_pixels_above_saturation_threshold=int(np.sum(
            np.isfinite(a) & np.isfinite(b)
            & ((a >= 60000) | (b >= 60000))
        )),
        scalar_a_over_b=scale,
        tile_ratio_std_percent=100*float(np.std(v)),
        tile_ratio_p05_percent=100*float(np.percentile(v, 5)),
        tile_ratio_p95_percent=100*float(np.percentile(v, 95)),
        tile_ratio_peak_to_peak_percent=100*float(np.ptp(v)),
        highpass_residual_normalized_mad=robust_sigma(z),
        highpass_residual_fraction_gt_5sigma=float(np.mean(np.abs(z)>5)),
        median_expected_pixel_sigma_adu=float(np.median(predicted)),
        axis_fraction=fractions,
        axis_decomposition="sequential_OLS_X_first_Y_conditional_observed_tiles",
        tile_medians_percent=100*tile_med,
        tile_valid_fraction=tile_fraction,
        tile_accepted=enough,
    )

def selftest():
    rng = np.random.default_rng(1206)
    y,x=np.mgrid[:256,:256]
    base=12000+7*x+4*y
    b=base+rng.normal(0,np.sqrt(base/3.5+2.1**2),base.shape)
    shaped=1.004*(1+0.005*(x/255.0-0.5)+0.001*(y/255.0-0.5))*base
    a=shaped+rng.normal(0,np.sqrt(shaped/3.5+2.1**2),base.shape)
    a[30,30]=50000
    res=spatial_pair(a,b,gain=3.5,readnoise_adu=2.1,tile=64)
    assert res["n_tiles"]==16
    assert res["tile_ratio_std_percent"]>0.08
    assert res["axis_fraction"]["dispersion_x"]>0.7
    assert 0.7<res["highpass_residual_normalized_mad"]<1.3
    assert abs(sum(res["axis_fraction"].values())-1)<1e-8
    assert np.isfinite(res["tile_medians_percent"]).all()
    print("SELFTEST PASSED")


def run(csv_path):
    from astropy.io import fits
    with csv_path.open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    inv={Path(r["file"]).name.lower(): Path(r["file"]) for r in rows
         if r["role"].lower().strip()=="flat"}
    if not all(x in inv for x in FLATS):
        raise RuntimeError("Missing one of the three audited FLATs")
    corrected=[]
    meta=[]
    for name in FLATS:
        with fits.open(inv[name],memmap=False) as hdul:
            hdr=hdul[0].header
            image=hdul[0].data
            if image is None or image.shape!=SHAPE:
                raise RuntimeError("Unexpected geometry "+name)
            if (abs(float(hdr["RATE"])-225)>0.01
                    or abs(float(hdr["GAIN"])-3.5)>0.01
                    or str(hdr["NODE"]).strip().upper()!="A"):
                raise RuntimeError("Unexpected readout "+name)
            left=clipped_mean(image[LEFT]);right=clipped_mean(image[RIGHT])
            side=(left+right)/2
            data=np.array(image[ACTIVE],dtype=np.float32,copy=True)
            data-=np.float32(side)
        corrected.append(data)
        meta.append(dict(file=name,side_mean=side,left_minus_right=left-right))
    results=[]
    tile_rows=[]
    for i,j in ((0,1),(0,2),(1,2)):
        for tile in (32,64,128):
            res=spatial_pair(corrected[i],corrected[j],tile=tile)
            tile_map=res.pop("tile_medians_percent")
            fractions=res.pop("tile_valid_fraction")
            accepted=res.pop("tile_accepted")
            if tile==64:
                for y,x in np.ndindex(tile_map.shape):
                    is_accepted=bool(accepted[y,x])
                    tile_rows.append(dict(
                        flat_a=FLATS[i], flat_b=FLATS[j], tile_y=int(y),
                        tile_x=int(x),
                        valid_fraction=float(fractions[y,x]),
                        accepted_for_model=is_accepted,
                        relative_offset_percent=(
                            float(tile_map[y,x]) if is_accepted else ""
                        ),
                    ))
            res.update(flat_a=FLATS[i],flat_b=FLATS[j])
            results.append(res)
            print(f"{FLATS[i]} / {FLATS[j]} tile={tile}: "
                  f"smooth={res['tile_ratio_std_percent']:.4f}% "
                  f"highpass_zMAD={res['highpass_residual_normalized_mad']:.4f} "
                  f"coverage={res['n_tiles']}/{res['n_tiles_total']} tiles "
                  f"Xshare={100*res['axis_fraction']['dispersion_x']:.1f}%",flush=True)
    summary=dict(
        status="DIAGNOSTIC_ONLY_HOLD",
        candidate_sidebands_not_certified=True,
        gain_e_per_adu_assumption_unverified=True,
        physical_cause_of_differences_not_identified=True,
        low_coverage_tiles_excluded_not_silently_imputed=True,
        x_y_shares_are_order_dependent_for_sparse_coverage=True,
        flats=meta, comparisons=results,
    )
    jpath=csv_path.with_name("sc1_fast_flat_spatial_summary.json")
    cpath=csv_path.with_name("sc1_fast_flat_spatial_64px_tiles.csv")
    jpath.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    with cpath.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(tile_rows[0]))
        writer.writeheader();writer.writerows(tile_rows)
    print("OUTPUT:",jpath,cpath,sep="\n  ")
    print("READ-ONLY SPATIAL DIAGNOSTIC FINISHED; NO PRODUCTION CALIBRATION APPROVED")


if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("Usage: script.py --selftest | selected_frames.csv")
    if sys.argv[1]=="--selftest":
        selftest()
    else:
        run(Path(sys.argv[1]))
