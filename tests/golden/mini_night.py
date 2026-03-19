from __future__ import annotations

"""Golden-nights fixtures: a tiny deterministic synthetic "night".

P0-ARCH-06.2 requires at least one true end-to-end pipeline run that can be
used as a regression net in CI.

This fixture intentionally stays small (fast) while still exercising the
scientific chain through the engine:
  superbias → superflat → superneon → wavesolution → sky → linearize → stack2d → extract1d → qc_report → navigator
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MiniNight:
    work_dir: Path
    data_dir: Path
    resources_dir: Path
    cfg: dict[str, Any]
    cfg_path: Path


def _sc2_header(*, imagetyp: str, obj: str, ny: int, nx: int) -> "fits.Header":
    # Local import to keep tests import-time light.
    from astropy.io import fits

    h = fits.Header()
    h["DATE-OBS"] = "2025/16/12"
    h["TIME-OBS"] = "17:59:06.633"
    h["INSTRUME"] = "SCORPIO-2"
    h["OBJECT"] = str(obj)
    h["IMAGETYP"] = str(imagetyp)
    h["RATE"] = 185.0
    h["GAIN"] = 0.62
    h["NODE"] = "E"
    h["BINNING"] = "1x1"
    h["UT"] = "14:59:02.89"
    h["MODE"] = "Spectra"
    h["DISPERSE"] = "VPHG1200@540"
    h["SLITWID"] = 1.0
    h["SLITPOS"] = 514.2
    # Astropy will overwrite NAXIS1/2 from data, but keep it explicit for clarity.
    h["NAXIS"] = 2
    h["NAXIS1"] = int(nx)
    h["NAXIS2"] = int(ny)
    return h


def _write_fits(path: Path, data: np.ndarray, header: "fits.Header") -> Path:
    from astropy.io import fits

    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data.astype("float32", copy=False), header=header).writeto(
        path, overwrite=True
    )
    return path


def _gauss1d(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - float(mu)) / float(sig)) ** 2)


def make_mini_night(tmp_path: Path) -> MiniNight:
    """Create a deterministic tiny synthetic dataset + config."""

    from astropy.io import fits  # noqa: F401

    work_dir = tmp_path / "work"
    data_dir = tmp_path / "data"
    resources_dir = tmp_path / "resources"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    # Minimal resources required by reference_context + lamp contract.
    (resources_dir / "linelist.csv").write_text(
        "wavelength,intensity\n" + "\n".join([f"{5000+2*i},1" for i in range(0, 200, 20)]) + "\n",
        encoding="utf-8",
    )
    (resources_dir / "atlas.pdf").write_bytes(
        b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
    )

    rng = np.random.default_rng(7)
    ny, nx = 64, 128
    x = np.arange(nx, dtype=float)

    # --- bias frames ---
    bias_level = 1000.0
    rn = 3.0
    for i in range(3):
        h = _sc2_header(imagetyp="bias", obj=f"bias{i+1}", ny=ny, nx=nx)
        img = bias_level + rng.normal(0.0, rn, size=(ny, nx))
        _write_fits(data_dir / f"bias_{i+1:02d}.fits", img, h)

    # --- flat frames ---
    for i in range(3):
        h = _sc2_header(imagetyp="flat", obj=f"flat{i+1}", ny=ny, nx=nx)
        grad = 1.0 + 0.15 * (x / float(nx - 1))
        img = bias_level + 5000.0 * grad[None, :] + rng.normal(0.0, 10.0, size=(ny, nx))
        _write_fits(data_dir / f"flat_{i+1:02d}.fits", img, h)

    # --- neon (arc) frames ---
    line_x = np.array([14, 28, 44, 60, 78, 96, 112], dtype=float)
    for i in range(2):
        h = _sc2_header(imagetyp="neon", obj=f"neon{i+1}", ny=ny, nx=nx)
        spec = np.zeros(nx, dtype=float)
        for mu in line_x:
            spec += 700.0 * _gauss1d(x, mu, 1.2)
        img = bias_level + 50.0 + spec[None, :] + rng.normal(0.0, 2.0, size=(ny, nx))
        _write_fits(data_dir / f"neon_{i+1:02d}.fits", img, h)

    # --- object (science) frames ---
    # Background sky lines (same positions as arc, weaker) + faint object bump.
    sky_spec = np.zeros(nx, dtype=float)
    for mu in line_x:
        sky_spec += 80.0 * _gauss1d(x, mu, 1.4)
    obj_bump = 25.0 * _gauss1d(x, 64.0, 6.0)

    for i in range(2):
        h = _sc2_header(imagetyp="object", obj=f"obj{i+1}", ny=ny, nx=nx)
        img = bias_level + 120.0 + sky_spec[None, :] + rng.normal(0.0, 4.0, size=(ny, nx))
        img[28:36, :] += obj_bump[None, :]
        _write_fits(data_dir / f"obj_{i+1:02d}.fits", img, h)

    # Hand pairs for wavesolution: enforce a perfectly linear dispersion law.
    # λ = 5000 + 2*x (air).
    hand_pairs = tmp_path / "hand_pairs.txt"
    hp_lines = [f"{float(xx):.3f} {5000.0 + 2.0*float(xx):.6f}" for xx in line_x]
    hand_pairs.write_text("\n".join(hp_lines) + "\n", encoding="utf-8")

    bias_paths = sorted(str(p) for p in data_dir.glob("bias_*.fits"))
    flat_paths = sorted(str(p) for p in data_dir.glob("flat_*.fits"))
    neon_paths = sorted(str(p) for p in data_dir.glob("neon_*.fits"))
    obj_paths = sorted(str(p) for p in data_dir.glob("obj_*.fits"))

    cfg: dict[str, Any] = {
        "work_dir": str(work_dir),
        "data_dir": str(data_dir),
        "resources_dir": str(resources_dir),
        "project": {"wave_reference": "air"},
        "frames": {
            "bias": bias_paths,
            "flat": flat_paths,
            "neon": neon_paths,
            "obj": obj_paths,
            "__setup__": {
                "instrument": "SCORPIO2",
                "mode": "Spectra",
                "disperser": "VPHG1200@540",
                "slit": "1.0",
                "binning": "1x1",
                "shape": f"{ny}x{nx}",
            },
        },
        "calib": {
            "bias_combine": "median",
            "flat_combine": "mean",
            "flat_sigma_clip": 0.0,
        },
        "superneon": {
            "bias_sub": True,
            "max_abs_shift": 4,
        },
        "wavesol": {
            "hand_pairs_path": str(hand_pairs),
            "linelist_csv": "linelist.csv",
            "atlas_pdf": "atlas.pdf",
            "poly_deg_1d": 4,
            "poly_sigma_clip": 4.0,
            "poly_maxiter": 8,
            # keep 2D model modest for stability
            "poly_deg_x": 4,
            "poly_deg_y": 2,
        },
        "sky": {
            "failure_policy": "strict",
            "rng_seed": 7,
            "kelson_raw": {
                # Keep the flexure scan cheap and deterministic.
                "flexure_n_samples": 20000,
                "delta_score_warn": 0.0,
                "delta_uncertain_A": 1e9,
            },
            "roi": {
                "obj_y0": 28,
                "obj_y1": 35,
                "sky_bot_y0": 5,
                "sky_bot_y1": 20,
                "sky_top_y0": 44,
                "sky_top_y1": 59,
                "units": "px",
            },
        },
        "linearize": {
            "dlambda_A": 2.0,
            "per_exposure": True,
            "save_preview": True,
        },
        "stack2d": {
            "sigma_clip": 4.0,
            "maxiter": 2,
        },
        "extract1d": {
            "method": "boxcar",
            "aperture_half_width": 4,
        },
    }

    # Write a real YAML config to mimic normal CLI/GUI runs.
    cfg_path = work_dir / "config.yaml"
    try:
        import yaml

        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except Exception:
        # YAML is a runtime dependency, but keep the fixture ultra-defensive.
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    return MiniNight(work_dir=work_dir, data_dir=data_dir, resources_dir=resources_dir, cfg=cfg, cfg_path=cfg_path)
