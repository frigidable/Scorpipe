from __future__ import annotations

"""Pydantic schema for config.yaml (v4+).

The pipeline intentionally continues to operate on a plain dict/YAML config
for backward compatibility. This schema is used by validation/doctor to catch
common issues earlier and to warn about likely typos.

We allow extra keys, but we surface them as warnings via `find_unknown_keys`.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QCBlock(BaseModel):
    """QC configuration.

    thresholds: numeric values that override auto thresholds.
    auto: whether to compute sensible defaults when thresholds are absent.
    """

    model_config = ConfigDict(extra="allow")

    thresholds: Dict[str, float] = Field(default_factory=dict)
    auto: bool = True


class WavesolBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    disperser: Optional[str] = None
    slit: Optional[str] = None
    binning: Optional[str] = None

    neon_lines_csv: str = "neon_lines.csv"
    atlas_pdf: str = "HeNeAr_atlas.pdf"

    y_half: int = 20
    hand_pairs_path: Optional[str] = None

    qc: QCBlock = Field(default_factory=QCBlock)


class CalibBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    superbias_path: Optional[str] = None
    superflat_path: Optional[str] = None


class SuperneonBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    bias_sub: bool = True


class FramesBlock(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    bias: List[str] = Field(default_factory=list)
    flat: List[str] = Field(default_factory=list)
    neon: List[str] = Field(default_factory=list)
    obj: List[str] = Field(default_factory=list)
    sky: List[str] = Field(default_factory=list)

    # stored under frames.__setup__ in YAML
    setup: Dict[str, Any] = Field(default_factory=dict, alias="__setup__")

    @model_validator(mode="before")
    @classmethod
    def _coerce_lists(cls, data: Any) -> Any:
        # Allow a single string where a list is expected (common user typo).
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for k in ("bias", "flat", "neon", "obj", "sky"):
            v = out.get(k)
            if isinstance(v, str):
                out[k] = [v]
        return out


class ConfigSchema(BaseModel):
    """Schema for the *resolved* config dict (after load_config).

    We include a few computed fields (config_dir, project_root, ...), because
    load_config() injects them.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    work_dir: str
    data_dir: str

    frames: FramesBlock = Field(default_factory=FramesBlock)
    calib: CalibBlock = Field(default_factory=CalibBlock)
    wavesol: WavesolBlock = Field(default_factory=WavesolBlock)
    superneon: SuperneonBlock = Field(default_factory=SuperneonBlock)

    profiles: Optional[Dict[str, Any]] = None

    # computed / meta
    config_path: Optional[str] = None
    config_dir: Optional[str] = None
    project_root: Optional[str] = None
    work_dir_abs: Optional[str] = None
    setup: Optional[Dict[str, Any]] = None
    _profiles_applied: Optional[List[str]] = None


# ----------------------- typo/unknown key support -----------------------

_TOP_KEYS = {
    "work_dir",
    "data_dir",
    "frames",
    "calib",
    "wavesol",
    "superneon",
    "profiles",
    "config_path",
    "config_dir",
    "project_root",
    "work_dir_abs",
    "setup",
    "_profiles_applied",
}

_FRAMES_KEYS = {"bias", "flat", "neon", "obj", "sky", "__setup__"}

_WAVESOL_KEYS = {
    "disperser",
    "slit",
    "binning",
    "neon_lines_csv",
    "atlas_pdf",
    "y_half",
    "hand_pairs_path",
    "qc",
}

_CALIB_KEYS = {"superbias_path", "superflat_path"}

_SUPERNEON_KEYS = {"bias_sub"}

_QC_KEYS = {"thresholds", "auto"}


def find_unknown_keys(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return unknown keys grouped by section.

    Keys injected by load_config are considered known.
    """

    unknown: Dict[str, List[str]] = {}

    top_unknown = sorted(k for k in cfg.keys() if str(k) not in _TOP_KEYS)
    if top_unknown:
        unknown["top"] = top_unknown

    frames = cfg.get("frames")
    if isinstance(frames, dict):
        u = sorted(k for k in frames.keys() if str(k) not in _FRAMES_KEYS)
        if u:
            unknown["frames"] = u

    w = cfg.get("wavesol")
    if isinstance(w, dict):
        u = sorted(k for k in w.keys() if str(k) not in _WAVESOL_KEYS)
        if u:
            unknown["wavesol"] = u
        qc = w.get("qc")
        if isinstance(qc, dict):
            u2 = sorted(k for k in qc.keys() if str(k) not in _QC_KEYS)
            if u2:
                unknown["wavesol.qc"] = u2

    c = cfg.get("calib")
    if isinstance(c, dict):
        u = sorted(k for k in c.keys() if str(k) not in _CALIB_KEYS)
        if u:
            unknown["calib"] = u

    sn = cfg.get("superneon")
    if isinstance(sn, dict):
        u = sorted(k for k in sn.keys() if str(k) not in _SUPERNEON_KEYS)
        if u:
            unknown["superneon"] = u

    return unknown


def schema_validate(cfg: Dict[str, Any]) -> Tuple[Optional[ConfigSchema], str | None]:
    """Validate and return (model, error_text)."""
    try:
        m = ConfigSchema.model_validate(cfg)
        return m, None
    except Exception as e:
        # keep it simple and human-readable
        return None, str(e)
