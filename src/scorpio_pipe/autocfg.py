from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


def _norm_setup_str(s: str) -> str:
    return (s or "").strip()


def _norm(s: str) -> str:
    return "".join(ch for ch in s.strip().upper() if ch.isalnum())


@dataclass(frozen=True)
class AutoConfig:
    data_dir: str
    object_name: str
    frames: dict
    work_dir: str
    calib: dict | None = None

    def to_yaml_text(self) -> str:
        """Serialize config to YAML text.

        The GUI uses this to populate the in-app config editor.
        """
        setup = self.frames.get("__setup__", {}) if isinstance(self.frames, dict) else {}
        if not isinstance(setup, dict):
            setup = {}

        instrument_section = {
            "name": str(setup.get("instrument", "") or ""),
            "mode": str(setup.get("mode", "") or ""),
            "disperser": str(setup.get("disperser", "") or ""),
            "slit": str(setup.get("slit", "") or ""),
            "binning": str(setup.get("binning", "") or ""),
            "window": str(setup.get("window", "") or ""),
            "shape": str(setup.get("shape", "") or ""),
        }

        payload = dict(
            data_dir=self.data_dir,
            object=self.object_name,
            # Config is written into work_dir. Using "." keeps it self-contained and avoids
            # duplicated "work/.../work/..." when resolving relative paths.
            work_dir=self.work_dir,
            instrument=instrument_section,
            calib={
                **(self.calib or {}),
                # Bias is best combined with a median to suppress rare outliers.
                "bias_combine": "median",  # or: mean
                "bias_sigma_clip": 0.0,  # set to e.g. 5.0 to suppress cosmics
            },
            cosmics={
                "enabled": True,
                "method": "stack_mad",
                "k": 9.0,
                "bias_subtract": True,
                "save_png": True,
                "apply_to": ["obj", "sky"],
            },
            flatfield={
                # Optional step after cosmics: divide by object-matched flats.
                "enabled": False,
                "bias_subtract": True,
                "save_png": True,
                # Neon can be optionally included by enabling it in the GUI.
                "apply_to": ["obj", "sky", "sunsky"],
            },
            runtime={
                "n_jobs": 0,  # 0/None = auto
            },
            wavesol={
                # y-range for 1D profile extraction from superneon; if None, use central band.
                "profile_y": None,
                "y_half": 20,

                # alignment: how far frames can be shifted (pixels)
                "xshift_max_abs": 6,

                # peak detection: thresholds are based on robust sigma
                "peak_snr": 5.0,
                "peak_prom_snr": 4.0,
                "peak_distance": 3,

                # GUI: auto min amplitude = median(noise peaks) + k*MAD
                "gui_min_amp_sigma_k": 5.0,

                # 1D polynomial degree for lambda(x) from hand pairs
                "poly_deg_1d": 4,

                # Optional override for hand pairs file. If empty, the pipeline uses
                # wavesol/<disperser>/hand_pairs.txt
                "hand_pairs_path": "",

                # list of laboratory wavelengths
                "neon_lines_csv": "neon_lines.csv",

                # HeNeAr atlas (optional, used by GUI button "Атлас")
                "atlas_pdf": "HeNeAr_atlas.pdf",
            },
            frames=self.frames,
        )
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    def to_yaml(self, path: Path) -> None:
        """Write YAML to the provided path (CLI-friendly)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml_text(), encoding="utf-8")


def _pick_setup_for_object(
    df: pd.DataFrame,
    object_name: str,
    *,
    disperser: str | None = None,
    slit: str | None = None,
    binning: str | None = None,
) -> dict:
    obj_n = _norm(object_name)

    sci = df[(df["kind"] == "obj") & (df["object_norm"] == obj_n)].copy()
    if sci.empty:
        raise ValueError(f"No science frames found for object='{object_name}'")

    if disperser:
        disp = _norm_setup_str(disperser)
        if "disperser" in sci.columns:
            sci = sci[sci["disperser"].astype(str).map(_norm_setup_str) == disp]
        if sci.empty:
            raise ValueError(f"No science frames for object='{object_name}' with disperser='{disperser}'")

    if slit:
        sl = _norm_setup_str(slit)
        if "slit" in sci.columns:
            sci = sci[sci["slit"].astype(str).map(_norm_setup_str) == sl]
        if sci.empty:
            raise ValueError(f"No science frames for object='{object_name}' with slit='{slit}'")

    if binning:
        bn = _norm_setup_str(binning)
        if "binning" in sci.columns:
            sci = sci[sci["binning"].astype(str).map(_norm_setup_str) == bn]
        if sci.empty:
            raise ValueError(f"No science frames for object='{object_name}' with binning='{binning}'")

    # priority: Spectra (if present)
    if "mode" in sci.columns and (sci["mode"] == "Spectra").any():
        sci = sci[sci["mode"] == "Spectra"]

    # choose the most frequent configuration
    key_cols = ["instrument", "mode", "disperser", "slit", "binning", "window", "shape"]
    for c in key_cols:
        if c not in sci.columns:
            sci[c] = ""

    for c in key_cols:
        sci[c] = sci[c].astype(str).map(_norm_setup_str)

    g = sci.groupby(key_cols, dropna=False).size().sort_values(ascending=False)
    instrument, mode, disperser0, slit0, binning0, window0, shape0 = g.index[0]

    return {
        "instrument": instrument,
        "mode": mode,
        "disperser": disperser0,
        "slit": slit0,
        "binning": binning0,
        "window": window0,
        "shape": shape0,
    }


def _select_by_setup(df: pd.DataFrame, kind: str, setup: dict) -> pd.DataFrame:
    out = df[df["kind"] == kind].copy()
    for k in ("instrument", "mode", "disperser", "slit", "binning", "window", "shape"):
        v = _norm_setup_str(setup.get(k, ""))
        if v and k in out.columns:
            out = out[out[k].astype(str).map(_norm_setup_str) == v]
    return out


def build_autoconfig(
    df: pd.DataFrame,
    data_dir: Path,
    object_name: str,
    work_dir: Path,
    *,
    disperser: str | None = None,
    slit: str | None = None,
    binning: str | None = None,
) -> AutoConfig:
    setup = _pick_setup_for_object(df, object_name, disperser=disperser, slit=slit, binning=binning)

    frames: dict[str, list[str] | dict] = {}

    # --- BIAS: only same size as science frames ---
    setup_shape = setup.get("shape", "")
    bias_df = df[df["kind"] == "bias"].copy()
    if setup_shape:
        bias_df = bias_df[bias_df["shape"].astype(str) == str(setup_shape)]
    frames["bias"] = bias_df["path"].tolist()

    # flats/neon/sky/sunsky: strict by setup
    # Flats are often taken specifically for the target; prefer OBJECT-matched flats when available.
    obj_n = _norm(object_name)
    if "object_norm" in df.columns:
        flat_df_obj = df[(df["kind"] == "flat") & (df["object_norm"].astype(str) == obj_n)]
    else:
        flat_df_obj = df.iloc[0:0]
    if len(flat_df_obj) > 0:
        frames["flat"] = _select_by_setup(flat_df_obj, "flat", setup)["path"].tolist()
    else:
        frames["flat"] = _select_by_setup(df, "flat", setup)["path"].tolist()
    frames["neon"] = _select_by_setup(df, "neon", setup)["path"].tolist()
    frames["sky"] = _select_by_setup(df, "sky", setup)["path"].tolist()
    frames["sunsky"] = _select_by_setup(df, "sunsky", setup)["path"].tolist()

    # science frames for the selected object (also by setup)
    obj_df = df[(df["kind"] == "obj") & (df["object_norm"] == obj_n)].copy()
    if "mode" in obj_df.columns:
        obj_df = obj_df[obj_df["mode"].astype(str) == "Spectra"]
    for k in ("instrument", "mode", "disperser", "slit", "binning", "window", "shape"):
        v = setup.get(k, "")
        if v not in ("", None) and k in obj_df.columns:
            obj_df = obj_df[obj_df[k].astype(str) == str(v)]
    frames["obj"] = obj_df["path"].tolist()

    # attach setup as an internal field
    frames["__setup__"] = setup

    calib = {"superbias_path": "calib/superbias.fits", "superflat_path": "calib/superflat.fits"}
    return AutoConfig(
        data_dir=str(data_dir),
        object_name=object_name,
        frames=frames,
        work_dir=".",
        calib=calib,
    )
