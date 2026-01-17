from __future__ import annotations

"""Dataset manifest (P0-B1).

This module defines the on-disk artifact ``dataset_manifest.json``.

Design goals
------------
* Explicit and auditable calibration matching.
* Deterministic selection rules.
* Importable in lightweight environments: **no Astropy imports**.

Schema v2 (P0-E)
----------------
Adds list-valued association fields for calibrations that may be combined:

- ``MatchEntry.flat_ids``: list of flat frame IDs to combine into MasterFlat for
  the corresponding science_set.
- ``MatchEntry.arc_ids``: reserved for future (e.g. multi-arc solutions).

Backward compatibility
----------------------
Older manifests (schema_version=1) only contain singular ``flat_id``/``arc_id``.
The v2 loader auto-populates the list fields from the singular ones.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Severity = Literal["INFO", "WARN", "ERROR"]


class ReadoutKeyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str
    rate: float
    gain: float


class GeometryKeyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    naxis1: int
    naxis2: int
    bin_x: int
    bin_y: int


class SpectroKeyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "Spectra"
    disperser: str
    slit_width_arcsec: float
    slit_width_key: str


class ConfigKeyModel(BaseModel):
    """Full compatibility key used for science-set grouping."""

    model_config = ConfigDict(extra="forbid")

    instrument: str
    geometry: GeometryKeyModel
    readout: ReadoutKeyModel
    spectro: SpectroKeyModel

    def as_compact_str(self) -> str:
        g = self.geometry
        r = self.readout
        s = self.spectro
        return (
            f"{self.instrument}|{g.naxis1}x{g.naxis2}|bin={g.bin_x}x{g.bin_y}"
            f"|node={r.node}|rate={r.rate:g}|gain={r.gain:g}"
            f"|{s.mode}|{s.disperser}|slit={s.slit_width_key}"
        )


class FrameIndexEntry(BaseModel):
    """Optional full frame index (convenience for GUI/debugging)."""

    model_config = ConfigDict(extra="allow")

    frame_id: str
    path: str
    kind: str
    instrument: str
    date_time_utc: str
    object: str = ""
    mode: str = ""
    disperser: str = ""
    slit_width_key: str = ""
    slit_pos: float | None = None
    sperange: str | None = None
    geometry: GeometryKeyModel
    readout: ReadoutKeyModel
    spectro: SpectroKeyModel | None = None
    sha256: str | None = None
    size_bytes: int | None = None


class ScienceSet(BaseModel):
    model_config = ConfigDict(extra="allow")

    science_set_id: str
    object: str
    object_norm: str
    frames: List[str]
    config: ConfigKeyModel

    # convenience timing summary
    start_utc: str
    end_utc: str
    mid_utc: str
    n_frames: int

    # soft-match helpers (optional)
    sperange: str | None = None
    slit_pos: float | None = None


class CalibrationEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    calib_id: str
    path: str
    kind: Literal["bias", "flat", "arc"]
    instrument: str
    date_time_utc: str
    geometry: GeometryKeyModel
    readout: ReadoutKeyModel

    # flats/arcs only
    spectro: SpectroKeyModel | None = None

    sperange: str | None = None
    slit_pos: float | None = None
    sha256: str | None = None
    size_bytes: int | None = None


class CalibrationPools(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bias: List[CalibrationEntry] = Field(default_factory=list)
    flat: List[CalibrationEntry] = Field(default_factory=list)
    arc: List[CalibrationEntry] = Field(default_factory=list)


class MatchSelectionMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_pool: int = 0
    n_hard_compatible: int = 0
    abs_dt_s: float | None = None
    sperange_mismatch: bool | None = None
    slitpos_diff: float | None = None
    tie_n: int | None = None
    tie_break: str | None = None


class MatchEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    science_set_id: str
    bias_id: str | None = None

    # Flat: singular id is kept for backward compatibility;
    # flat_ids is authoritative for MasterFlat.
    flat_id: str | None = None
    flat_ids: List[str] | None = None

    # Arc: singular id is used today; arc_ids reserved.
    arc_id: str | None = None
    arc_ids: List[str] | None = None

    bias_meta: MatchSelectionMeta | None = None
    flat_meta: MatchSelectionMeta | None = None
    arc_meta: MatchSelectionMeta | None = None

    @model_validator(mode="after")
    def _coerce_list_fields(self) -> "MatchEntry":
        if self.flat_ids is None and self.flat_id is not None:
            self.flat_ids = [self.flat_id]
        if self.arc_ids is None and self.arc_id is not None:
            self.arc_ids = [self.arc_id]
        return self


class ManifestWarning(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    severity: Severity = "WARN"
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    """Top-level dataset manifest."""

    # NOTE: We intentionally avoid a field named "schema" because pydantic's
    # BaseModel already defines schema-related helpers. We keep the *JSON key*
    # "schema" via an alias.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_id: str = Field(default="scorpio-pipe.dataset-manifest.v2", alias="schema")
    schema_version: int = Field(default=2)
    pipeline_version: str
    generated_utc: str

    data_dir: str | None = None
    night_id: str | None = None

    science_sets: List[ScienceSet] = Field(default_factory=list)
    calibration_pools: CalibrationPools = Field(default_factory=CalibrationPools)
    matches: List[MatchEntry] = Field(default_factory=list)
    warnings: List[ManifestWarning] = Field(default_factory=list)

    # Summary of globally excluded frames (from project_manifest.yaml / CLI exclude).
    excluded_summary: Dict[str, Any] = Field(default_factory=dict)

    # Convenience/debugging extras
    frames: List[FrameIndexEntry] | None = None
    summary: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True, by_alias=True)

    def to_json_text(self, *, indent: int = 2) -> str:
        # Use Pydantic's JSON mode for consistent encoding.
        return self.model_dump_json(exclude_none=True, indent=indent, by_alias=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        p = Path(path)
        p.write_text(self.to_json_text(indent=indent), encoding="utf-8")
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetManifest":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
