"""Product catalog (single source of truth).

P0-ARCH-01.2
------------
This module provides a unified registry of *named products*.

- :class:`ProductSpec` describes what a product is and how to locate it.
- :class:`ProductCatalog` resolves paths, validates products, and supports
  downstream consumers (QC report, export package, boundary validation).

Design notes
-----------
We keep the data model intentionally minimal but extensible.
A product may map to a single path (e.g. manifest/manifest.json) or to a
collection of paths (e.g. per-exposure rectified frames).

The catalog supports a tiny "glob DSL" for default paths:

- ``"manifest/manifest.json"``: relative to work_dir
- ``"@manifest/manifest.json"``: relative to work_dir/manifest
- ``"@qc/qc_report.json"``: relative to work_dir/qc
- ``"@ui/navigator"``: relative to work_dir/ui
- ``"@stage(stack2d)/stacked2d.fits"``: relative to the numbered stage dir
- ``"@wavesol/lambda_map.fits"``: relative to the disperser-aware wavesol dir

Globs are supported ("*", "?", "[abc]", "**").

Validation
----------
Validation is intentionally staged:
- For FITS products with a declared contract kind, we call strict validators
  from :mod:`scorpio_pipe.contracts.validators`.
- For other file kinds, we only perform lightweight checks (existence, JSON
  parseability) to keep boundary enforcement predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.workspace_paths import stage_dir
from scorpio_pipe.wavesol_paths import wavesol_dir


@dataclass(frozen=True)
class ProductSpec:
    """Single product specification."""

    key: str
    kind: str
    default_paths: tuple[str, ...]
    required_headers_contract: str | None
    schema_version: int | str | None
    role: str

    # Optional convenience metadata (not part of the minimal required design)
    optional: bool = True
    description: str | None = None

    def is_collection(self) -> bool:
        """True if any default path uses globbing and may yield multiple files."""

        wild = ("*", "?", "[", "]")
        for p in self.default_paths:
            if "**" in p:
                return True
            if any(w in p for w in wild):
                return True
        return False


@dataclass(frozen=True)
class Product:
    """Resolved product record used by QC/UI."""

    key: str
    stage: str
    path: Path
    kind: str
    optional: bool = True
    description: str | None = None

    def exists(self) -> bool:
        return self.path.exists()

    def size(self) -> int | None:
        try:
            return int(self.path.stat().st_size) if self.path.exists() else None
        except Exception:
            return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "path": str(self.path),
            "kind": self.kind,
            "optional": self.optional,
            "exists": self.exists(),
            "description": self.description,
        }


def _stable_unique(seq: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for x in seq:
        s = str(x).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return tuple(out)


class ProductCatalog:
    def __init__(self, *, specs: Sequence[ProductSpec]) -> None:
        self._specs = tuple(specs)
        self._by_key: dict[str, ProductSpec] = {}
        for s in self._specs:
            k = str(s.key).strip().lower()
            if not k:
                raise ValueError("ProductSpec.key must be non-empty")
            if k in self._by_key:
                raise ValueError(f"Duplicate ProductSpec key: {k!r}")
            self._by_key[k] = s

    # --------------------- Public API: spec access ---------------------

    def keys(self) -> tuple[str, ...]:
        return tuple(s.key for s in self._specs)

    def get_spec(self, key: str) -> ProductSpec:
        k = str(key or "").strip().lower()
        if k not in self._by_key:
            raise KeyError(f"Unknown product key: {key!r}")
        return self._by_key[k]

    def specs(self) -> tuple[ProductSpec, ...]:
        return self._specs

    # --------------------- Path resolution helpers ---------------------

    def _base_and_pattern(
        self,
        cfg: Mapping[str, Any],
        pat: str,
        *,
        run_root: str | Path | None = None,
    ) -> tuple[Path, str]:
        # Engine-owned run root override (P0-ARCH-02.1)
        work_dir = Path(run_root).expanduser().resolve() if run_root is not None else resolve_work_dir(dict(cfg))
        p = str(pat).strip()

        if p.startswith("@manifest/"):
            return Path(work_dir) / "manifest", p[len("@manifest/") :]
        if p.startswith("@qc/"):
            return Path(work_dir) / "qc", p[len("@qc/") :]
        if p.startswith("@ui/"):
            return Path(work_dir) / "ui", p[len("@ui/") :]
        if p.startswith("@work/"):
            return Path(work_dir), p[len("@work/") :]
        if p.startswith("@wavesol/"):
            return wavesol_dir(dict(cfg), run_root=work_dir), p[len("@wavesol/") :]
        if p.startswith("@stage("):
            # @stage(stack2d)/stacked2d.fits
            j = p.find(")")
            if j < 0:
                return Path(work_dir), p
            st = p[len("@stage(") : j]
            rest = p[j + 1 :]
            if rest.startswith("/"):
                rest = rest[1:]
            return stage_dir(work_dir, st), rest

        # default: relative to work_dir
        return Path(work_dir), p

    def expected_paths(
        self,
        cfg: Mapping[str, Any],
        key: str,
        *,
        run_root: str | Path | None = None,
    ) -> list[Path]:
        """Return canonical expected paths (non-glob patterns only)."""

        spec = self.get_spec(key)
        out: list[Path] = []
        for pat in spec.default_paths:
            base, rel = self._base_and_pattern(cfg, pat, run_root=run_root)
            # For non-glob patterns, materialize an expected path.
            if any(w in rel for w in ("*", "?", "[", "]")) or "**" in rel:
                continue
            out.append((base / rel).resolve())
        return out

    def find_existing_paths(
        self,
        cfg: Mapping[str, Any],
        key: str,
        *,
        run_root: str | Path | None = None,
    ) -> list[Path]:
        """Resolve product paths that exist on disk.

        For non-glob patterns we return the path if it exists.
        For glob patterns we return matching files.
        """

        spec = self.get_spec(key)
        found: list[Path] = []
        for pat in spec.default_paths:
            base, rel = self._base_and_pattern(cfg, pat, run_root=run_root)
            try:
                if any(w in rel for w in ("*", "?", "[", "]")) or "**" in rel:
                    # glob patterns
                    for p in sorted(base.glob(rel)):
                        if p.is_file():
                            found.append(p.resolve())
                        elif p.is_dir() and spec.kind == "dir":
                            found.append(p.resolve())
                else:
                    p = (base / rel).resolve()
                    if p.exists():
                        found.append(p)
            except Exception:
                continue

        # De-duplicate while preserving order.
        seen: set[str] = set()
        uniq: list[Path] = []
        for p in found:
            s = str(p)
            if s in seen:
                continue
            seen.add(s)
            uniq.append(p)
        return uniq

    def best_path(
        self,
        cfg: Mapping[str, Any],
        key: str,
        *,
        run_root: str | Path | None = None,
    ) -> Path:
        """Choose a single representative path for UI/QC listings."""

        found = self.find_existing_paths(cfg, key, run_root=run_root)
        if found:
            # prefer existing
            return found[0]
        exp = self.expected_paths(cfg, key, run_root=run_root)
        if exp:
            return exp[0]
        # last resort: resolve the first pattern as best-effort
        spec = self.get_spec(key)
        base, rel = self._base_and_pattern(cfg, spec.default_paths[0], run_root=run_root)
        return (base / rel).resolve()

    # --------------------- Validation helpers ---------------------

    def _validate_one_path(self, *, stage: str, spec: ProductSpec, path: Path) -> None:
        # Directories are validated by existence only.
        if spec.kind == "dir":
            if not path.exists():
                return
            if not path.is_dir():
                raise RuntimeError(f"[{stage}] {spec.key}: expected directory, got file: {path}")
            return

        if not path.exists():
            return

        k = (spec.required_headers_contract or "").strip().lower()
        if not k or k in {"none", "skip"}:
            # Lightweight checks
            if spec.kind == "json":
                try:
                    import json

                    json.loads(path.read_text(encoding="utf-8", errors="replace"))
                except Exception as e:
                    raise RuntimeError(f"[{stage}] {spec.key}: invalid JSON: {e}")
            return

        # FITS contract validators
        from scorpio_pipe.contracts.validators import (
            validate_lambda_map_product,
            validate_mef2d_product,
            validate_calibrated_mef2d_product,
            validate_calibrated_spec1d_product,
            validate_calibration_systematics_product,
            validate_sensfunc_product,
            validate_spec1d_product,
            validate_telluric_model_product,
            validate_product,
        )

        if k == "mef":
            validate_mef2d_product(path, stage=stage)
            return
        if k == "spec1d":
            validate_spec1d_product(path, stage=stage)
            return
        if k == "lambda_map":
            validate_lambda_map_product(path, stage=stage)
            return
        if k == "mef_calibrated":
            validate_calibrated_mef2d_product(path, stage=stage)
            return
        if k == "spec1d_calibrated":
            validate_calibrated_spec1d_product(path, stage=stage)
            return
        if k == "sensfunc":
            validate_sensfunc_product(path, stage=stage)
            return
        if k == "telluric_model":
            validate_telluric_model_product(path, stage=stage)
            return
        if k == "calsys":
            validate_calibration_systematics_product(path, stage=stage)
            return
        # fallback
        validate_product(path, stage=stage)

    def validate_products(
        self,
        cfg: Mapping[str, Any],
        *,
        stage_key: str,
        product_keys: Sequence[str],
        run_root: str | Path | None = None,
    ) -> None:
        """Validate exactly the provided product keys.

        Missing products are skipped ("optional" semantics).
        Violations raise the underlying :class:`ProductContractError` when
        applicable, otherwise raise :class:`RuntimeError`.
        """

        stage = str(stage_key).strip() or "stage"

        for k in product_keys:
            if not str(k).strip():
                continue
            spec = self.get_spec(k)
            paths = self.find_existing_paths(cfg, spec.key, run_root=run_root)
            # If nothing exists, do not fail here (existence is tracked elsewhere).
            for p in paths:
                self._validate_one_path(stage=stage, spec=spec, path=p)

    # --------------------- Resolved product listing (QC/UI) ---------------------

    def list_products(
        self,
        cfg: Mapping[str, Any],
        *,
        include_collections: bool = False,
        run_root: str | Path | None = None,
    ) -> list[Product]:
        """Return stable products for QC/UI listings.

        By default, products that map to large file collections (glob patterns)
        are excluded from listings to avoid overwhelming the report/UI.
        """

        from scorpio_pipe.catalog.stages import CATALOG as STAGE_CATALOG

        # Build a stable owner map: product_key -> first task that declares it.
        owner: dict[str, str] = {}
        for t in STAGE_CATALOG.task_infos():
            for pk in (t.product_keys or ()):  # type: ignore
                if pk not in owner:
                    owner[pk] = t.key

        out: list[Product] = []
        for spec in self._specs:
            if (not include_collections) and spec.is_collection():
                continue
            st = owner.get(spec.key, spec.role)
            out.append(
                Product(
                    key=spec.key,
                    stage=str(st),
                    path=self.best_path(cfg, spec.key, run_root=run_root),
                    kind=str(spec.kind),
                    optional=bool(spec.optional),
                    description=spec.description,
                )
            )
        return out

    # --------------------- Export helpers ---------------------

    def iter_export_files(
        self,
        cfg: Mapping[str, Any],
        *,
        product_keys: Sequence[str],
        include_dirs: bool = True,
        run_root: str | Path | None = None,
    ) -> Iterator[tuple[Path, str]]:
        """Yield (path, arcname) for an export package."""

        work_dir = Path(run_root).expanduser().resolve() if run_root is not None else resolve_work_dir(dict(cfg))
        wd = Path(work_dir).expanduser().resolve()

        def _arc(p: Path) -> str:
            try:
                return str(p.resolve().relative_to(wd.resolve())).replace("\\", "/")
            except Exception:
                return p.name

        for k in product_keys:
            if not str(k).strip():
                continue
            spec = self.get_spec(k)
            paths = self.find_existing_paths(cfg, spec.key, run_root=run_root)
            for p in paths:
                if spec.kind == "dir":
                    if not include_dirs:
                        continue
                    if p.is_dir():
                        for f in sorted(p.rglob("*")):
                            if f.is_file():
                                yield (f, _arc(f))
                    continue
                if p.is_file():
                    yield (p, _arc(p))


# ----------------------------------------------------------------------------
# Canonical specs
# ----------------------------------------------------------------------------


def _specs() -> tuple[ProductSpec, ...]:
    # Compact helpers
    def S(
        key: str,
        kind: str,
        paths: Sequence[str],
        contract: str | None,
        schema_version: int | str | None,
        role: str,
        *,
        optional: bool = True,
        description: str | None = None,
    ) -> ProductSpec:
        return ProductSpec(
            key=str(key),
            kind=str(kind),
            default_paths=_stable_unique(paths),
            required_headers_contract=contract,
            schema_version=schema_version,
            role=str(role),
            optional=bool(optional),
            description=description,
        )

    # Schema version for FITS data model
    try:
        from scorpio_pipe.contracts.data_model import DATA_MODEL_VERSION

        dm = int(DATA_MODEL_VERSION)
    except Exception:
        dm = None

    return (
        # Manifest + QC (canonical)
        S(
            "manifest",
            "json",
            ["@manifest/manifest.json"],
            "none",
            1,
            "manifest",
            optional=False,
            description="Reproducibility manifest",
        ),
        S(
            "dependency_lock",
            "text",
            ["@manifest/requirements.lock"],
            "none",
            1,
            "manifest",
            optional=True,
            description="Dependency lock (requirements.lock)",
        ),
        S(
            "deps_snapshot",
            "json",
            ["@manifest/deps_snapshot.json"],
            "none",
            1,
            "manifest",
            optional=True,
            description="Full dependency snapshot (importlib.metadata)",
        ),
        S(
            "products_manifest",
            "json",
            ["@manifest/products_manifest.json"],
            "none",
            1,
            "manifest",
            optional=True,
            description="Products manifest (incl. per-exposure trees)",
        ),
        S(
            "timings",
            "text",
            ["@manifest/timings.jsonl", "@manifest/timings.json"],
            "none",
            1,
            "manifest",
            optional=True,
            description="Stage timings (JSONL append-only; legacy JSON fallback)",
        ),
        S(
            "qc_json",
            "json",
            [
                "@qc/qc_report.json",
                "@manifest/qc_report.json",
            ],
            "none",
            1,
            "qc",
            optional=True,
            description="QC summary (machine-readable)",
        ),
        S(
            "qc_html",
            "html",
            ["@work/index.html"],
            "none",
            None,
            "qc",
            optional=True,
            description="QC report (human-readable)",
        ),
        S(
            "linearize_qc",
            "json",
            [
                "@qc/linearize_qc.json",
                "@manifest/linearize_qc.json",
                "@stage(linearize)/linearize_qc.json",
            ],
            "none",
            1,
            "qc",
            optional=True,
            description="Linearize QC metrics (S/N, coverage, mask fractions)",
        ),

        # Workspace UI bundle
        S(
            "navigator_dir",
            "dir",
            ["@ui/navigator"],
            "none",
            None,
            "ui",
            optional=True,
            description="Navigator HTML bundle (ui/navigator/*)",
        ),

        # Calibrations
        S(
            "superbias_fits",
            "fits",
            ["@stage(superbias)/superbias.fits"],
            "none",
            dm,
            "calib",
            optional=True,
        ),
        S(
            "master_bias_fits",
            "fits",
            ["@stage(superbias)/master_bias.fits"],
            "none",
            dm,
            "calib",
            optional=True,
            description="Master bias (SCI, default group)",
        ),
        S(
            "master_bias_var_fits",
            "fits",
            ["@stage(superbias)/master_bias_var.fits"],
            "none",
            dm,
            "calib",
            optional=True,
            description="Master bias variance (VAR, default group)",
        ),
        S(
            "master_bias_dq_fits",
            "fits",
            ["@stage(superbias)/master_bias_dq.fits"],
            "none",
            dm,
            "calib",
            optional=True,
            description="Master bias data-quality mask (DQ, default group)",
        ),
        S(
            "master_bias_index",
            "json",
            ["@stage(superbias)/master_bias_index.json"],
            "none",
            1,
            "calib",
            optional=True,
            description="Master bias group index (geometry+readout)",
        ),
        S(
            "superflat_fits",
            "fits",
            ["@stage(superflat)/superflat.fits"],
            "none",
            dm,
            "calib",
            optional=True,
        ),

        # Flatfield stage
        S(
            "flatfield_done",
            "json",
            ["@stage(flatfield)/flatfield_done.json"],
            "none",
            1,
            "science",
            optional=True,
        ),
        # Collection: per-kind corrected frames (usually obj/*_flat.fits)
        S(
            "flatfield_flat_fits",
            "fits",
            ["@stage(flatfield)/*/*_flat.fits"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Per-frame flat-corrected products",
        ),

        # Cosmics stage
        S(
            "cosmics_summary",
            "json",
            ["@stage(cosmics)/summary.json"],
            "none",
            1,
            "science",
            optional=True,
        ),
        S(
            "cosmics_clean_fits",
            "fits",
            ["@stage(cosmics)/*/clean/*_clean.fits"],
            "none",
            None,
            "science",
            optional=True,
            description="Per-frame cosmics-cleaned images (primary image FITS)",
        ),
        S(
            "cosmics_mask_fits",
            "fits",
            ["@stage(cosmics)/*/masks_fits/*_mask.fits"],
            "none",
            None,
            "science",
            optional=True,
            description="Per-frame cosmics mask FITS (uint8 bitmask)",
        ),

        # SuperNeon + LineID + Wavesolution (disperser-aware)
        S(
            "superneon_fits",
            "fits",
            ["@wavesol/superneon.fits"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "superneon_png",
            "png",
            ["@wavesol/superneon.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "peaks_candidates_csv",
            "csv",
            ["@wavesol/peaks_candidates.csv"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "superneon_shifts_csv",
            "csv",
            ["@wavesol/superneon_shifts.csv"],
            "none",
            None,
            "calib",
            optional=True,
            description="Per-frame X shifts + basic QC for SuperNeon stacking",
        ),
        S(
            "superneon_shifts_json",
            "json",
            ["@wavesol/superneon_shifts.json"],
            "none",
            1,
            "calib",
            optional=True,
            description="Per-frame X shifts + QC (machine-readable)",
        ),
        S(
            "superneon_qc_json",
            "json",
            ["@wavesol/superneon_qc.json"],
            "none",
            1,
            "calib",
            optional=True,
            description="SuperNeon QC summary (SNR, shifts, saturation, clipping)",
        ),


S(
    "slit_sanity_json",
    "json",
    ["@qc/slit_sanity.json"],
    "none",
    1,
    "qc",
    optional=True,
    description="Slit metadata vs measured slit-width proxy audit",
),
S(
    "slit_profile_png",
    "png",
    ["@qc/slit_profile.png"],
    "none",
    None,
    "qc",
    optional=True,
),
S(
    "slit_consistency_md",
    "txt",
    ["@qc/slit_consistency.md"],
    "none",
    None,
    "qc",
    optional=True,
),
S(
    "geometry_model_json",
    "json",
    ["@work/07_geometry/geometry_model.json"],
    "none",
    1,
    "calib",
    optional=True,
),
S(
    "geometry_preview_fits",
    "fits",
    ["@work/07_geometry/geometry_preview.fits"],
    "none",
    None,
    "calib",
    optional=True,
),
S(
    "geometry_qc_json",
    "json",
    ["@work/07_geometry/geometry_qc.json"],
    "none",
    1,
    "calib",
    optional=True,
),
S(
    "repers_json",
    "json",
    ["@work/07_geometry/repers.json"],
    "none",
    1,
    "calib",
    optional=True,
),
        S(
            "lineid_template_csv",
            "csv",
            ["@wavesol/manual_pairs_template.csv"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "lineid_auto_csv",
            "csv",
            ["@wavesol/manual_pairs_auto.csv"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "lineid_report_txt",
            "txt",
            ["@wavesol/lineid_report.txt"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "hand_pairs_txt",
            "txt",
            ["@wavesol/hand_pairs.txt"],
            "none",
            None,
            "calib",
            optional=True,
            description="Manual line pairs (x_pix, lambda)",
        ),

        # Wavesolution key artifacts
        S(
            "wavesolution_1d_png",
            "png",
            ["@wavesol/wavesolution_1d.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "wavesolution_1d_json",
            "json",
            ["@wavesol/wavesolution_1d.json"],
            "none",
            1,
            "calib",
            optional=True,
        ),
        S(
            "wavesolution_2d_json",
            "json",
            ["@wavesol/wavesolution_2d.json"],
            "none",
            1,
            "calib",
            optional=True,
        ),
        S(
            "lambda_map",
            "fits",
            ["@wavesol/lambda_map.fits"],
            "lambda_map",
            None,
            "calib",
            optional=True,
        ),
        S(
            "rectification_model_json",
            "json",
            ["@wavesol/rectification_model.json"],
            "none",
            1,
            "calib",
            optional=True,
            description="Rectification model (lambda-map signature + VAR/MASK policies)",
        ),
        S(
            "wave_done_json",
            "json",
            ["@wavesol/wave_done.json"],
            "none",
            1,
            "calib",
            optional=True,
            description="Wavesolution done marker (P1-B contract)",
        ),
        S(
            "wavelength_matrix_png",
            "png",
            ["@wavesol/wavelength_matrix.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "residuals_2d_png",
            "png",
            ["@wavesol/residuals_2d.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "residuals_2d_audit_png",
            "png",
            ["@wavesol/residuals_2d_audit.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "control_points_2d_csv",
            "csv",
            ["@wavesol/control_points_2d.csv"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "residuals_1d_csv",
            "csv",
            ["@wavesol/residuals_1d.csv"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "residuals_2d_csv",
            "csv",
            ["@wavesol/residuals_2d.csv"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "residuals_vs_lambda_png",
            "png",
            ["@wavesol/residuals_vs_lambda.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "residuals_vs_y_png",
            "png",
            ["@wavesol/residuals_vs_y.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "residuals_hist_png",
            "png",
            ["@wavesol/residuals_hist.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "wavesolution_report_txt",
            "txt",
            ["@wavesol/wavesolution_report.txt"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "line_id_overview_png",
            "png",
            ["@wavesol/line_id_overview.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "geometry_overlay_png",
            "png",
            ["@wavesol/geometry_overlay.png"],
            "none",
            None,
            "calib",
            optional=True,
        ),
        S(
            "geometry_prior_used_json",
            "json",
            ["@wavesol/geometry_prior_used.json"],
            "none",
            1,
            "calib",
            optional=True,
        ),

        # Linearize (stage dir)
        S(
            "lin_preview_fits",
            "fits",
            ["@stage(linearize)/lin_preview.fits"],
            "mef",
            dm,
            "science",
            optional=True,
        ),
        S(
            "lin_preview_png",
            "png",
            ["@stage(linearize)/lin_preview.png"],
            "none",
            None,
            "science",
            optional=True,
        ),
        S(
            "linearize_done",
            "json",
            ["@stage(linearize)/linearize_done.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Linearization summary (per exposure)",
        ),
        S(
            "linearize_skysub_fits",
            "fits",
            ["@stage(linearize)/**/*_skysub.fits"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Per-exposure rectified sky-subtracted frames",
        ),
        S(
            "linearize_skymodel_fits",
            "fits",
            ["@stage(linearize)/**/*_skymodel.fits"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Per-exposure rectified sky model frames (optional)",
        ),

        # Sky (stage dir)
        S(
            "lsf_map_fits",
            "fits",
            ["lsf_model/lsf_map.fits"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Per-pixel instrumental sigma map in raw detector geometry",
        ),
        S(
            "lsf_model_json",
            "json",
            ["lsf_model/lsf_model.json"],
            "none",
            1,
            "science",
            optional=True,
            description="LSF model sidecar with line fits and 1D summary",
        ),
        S(
            "lsf_qc_json",
            "json",
            ["lsf_model/lsf_qc.json"],
            "none",
            1,
            "science",
            optional=True,
            description="LSF QC summary",
        ),

        S(
            "sky_done",
            "json",
            ["@stage(sky)/sky_done.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Sky stage summary (per exposure)",
        ),
        S(
            "qc_sky_json",
            "json",
            ["@stage(sky)/qc_sky.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Sky QC (residual metrics + diag paths)",
        ),
        S(
            "sky_preview_fits",
            "fits",
            ["@stage(sky)/preview.fits"],
            "mef",
            dm,
            "science",
            optional=True,
        ),
        S(
            "sky_preview_png",
            "png",
            ["@stage(sky)/preview.png"],
            "none",
            None,
            "science",
            optional=True,
        ),
        S(
            "sky_skysub_raw_fits",
            "fits",
            ["@stage(sky)/*_skysub_raw.fits"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Per-exposure sky-subtracted frames in RAW geometry",
        ),
        S(
            "sky_refine_done",
            "json",
            ["sky_refine/done.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Sky residual refine summary",
        ),
        S(
            "qc_sky_refine_json",
            "json",
            ["sky_refine/qc_sky_refine.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Sky refine QC and per-window diagnostics",
        ),
        S(
            "sky_refined_raw_fits",
            "fits",
            ["sky_refine/*_skysub_refined_raw.fits"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Per-exposure sky-subtracted frames after residual refine",
        ),
        S(
            "skycorr_raw_fits",
            "fits",
            ["sky_refine/*_skycorr_raw.fits"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Residual correction model applied by sky refine",
        ),
        S(
            "sky_residual_windows_csv",
            "csv",
            ["sky_refine/sky_residual_windows.csv"],
            "none",
            1,
            "science",
            optional=True,
            description="Aggregated problematic skyline windows",
        ),
        S(
            "sky_residual_windows_json",
            "json",
            ["sky_refine/sky_residual_windows.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Aggregated problematic skyline windows (JSON)",
        ),
        S(
            "sky_observer_pack_md",
            "md",
            ["sky_refine/sky_observer_pack.md"],
            "none",
            1,
            "science",
            optional=True,
            description="Human-readable sky refine observer summary",
        ),

        # Stack2D
        S(
            "stack2d_done",
            "json",
            ["@stage(stack2d)/stack2d_done.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Stacking summary",
        ),
        S(
            "stack_done",
            "json",
            ["@stage(stack2d)/stack_done.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Stacking summary (canonical)",
        ),
        S(
            "stacked2d_fits",
            "fits",
            ["@stage(stack2d)/stacked2d.fits"],
            "mef",
            dm,
            "science",
            optional=True,
        ),
        S(
            "stack2d_fits",
            "fits",
            ["@stage(stack2d)/stack2d.fits"],
            "mef",
            dm,
            "science",
            optional=True,
        ),
        S(
            "eta_lambda_fits",
            "fits",
            ["@stage(stack2d)/eta_lambda.fits"],
            "mef",
            dm,
            "science",
            optional=True,
        ),
        S(
            "coverage_png",
            "png",
            ["@stage(stack2d)/coverage.png"],
            "none",
            None,
            "science",
            optional=True,
        ),

        # Extract1D
        S(
            "extract_done",
            "json",
            ["@stage(extract1d)/extract_done.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Extraction summary (canonical)",
        ),
        S(
            "trace_json",
            "json",
            ["@stage(extract1d)/trace.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Trace model and aperture definition",
        ),
        S(
            "extract1d_done",
            "json",
            ["@stage(extract1d)/extract1d_done.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Extraction summary (legacy alias)",
        ),
        S(
            "spec1d_fits",
            "fits",
            ["@stage(extract1d)/spec1d.fits", "@stage(extract1d)/spec1d*.fits"],
            "spec1d",
            dm,
            "science",
            optional=True,
        ),
        S(
            "spec1d_png",
            "png",
            ["@stage(extract1d)/spec1d.png"],
            "none",
            None,
            "science",
            optional=True,
        ),
        S(
            "extraction_qc_json",
            "json",
            ["@stage(extract1d)/extraction_qc.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Extraction diagnostics and trust metrics",
        ),
        S(
            "extraction_diagnostics_png",
            "png",
            ["@stage(extract1d)/extraction_diagnostics.png"],
            "none",
            None,
            "science",
            optional=True,
            description="Extraction diagnostics quicklook",
        ),
        S(
            "publication_spectrum_fts",
            "fits",
            ["publication_ready/Spectrum.fts"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Linearized publication-ready 2D science frame",
        ),
        S(
            "publication_wl_fts",
            "fits",
            ["publication_ready/wl.fts"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Publication-ready per-pixel wavelength map",
        ),
        S(
            "publication_dismap_fts",
            "fits",
            ["publication_ready/dismap.fts"],
            "mef",
            dm,
            "science",
            optional=True,
            description="Publication-ready per-pixel instrumental sigma map",
        ),
        S(
            "publication_provenance_json",
            "json",
            ["publication_ready/publication_provenance.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Publication-ready export sidecar",
        ),
        S(
            "spec_kin_fits",
            "fits",
            ["kinematics_prep/spec_kin.fits"],
            "none",
            1,
            "science",
            optional=True,
            description="Kinematics-prep 1D spectrum export",
        ),
        S(
            "kinematics_masks_json",
            "json",
            ["kinematics_prep/masks.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Kinematics-prep mask intervals",
        ),
        S(
            "kinematics_readme_md",
            "md",
            ["kinematics_prep/README.md"],
            "none",
            1,
            "science",
            optional=True,
            description="Kinematics-prep human-readable guidance",
        ),

        # Calibration / flux domain (P0.6)
        S(
            "standard_prepare_plan_json",
            "json",
            ["@stage(standard_prepare)/standard_prepare_plan.json"],
            "none",
            1,
            "calibration",
            optional=True,
            description="Machine-readable standard selection / preparation summary",
        ),
        S(
            "standard_1d_fits",
            "fits",
            ["@stage(standard_prepare)/standard_1d.fits"],
            "spec1d",
            dm,
            "calibration",
            optional=True,
            description="Standard-star extracted 1D spectrum",
        ),
        S(
            "standard_trace_json",
            "json",
            ["@stage(standard_prepare)/standard_trace.json"],
            "none",
            1,
            "calibration",
            optional=True,
            description="Standard-star trace / aperture definition",
        ),
        S(
            "standard_qc_json",
            "json",
            ["@stage(standard_prepare)/standard_qc.json"],
            "none",
            1,
            "calibration",
            optional=True,
            description="Standard-star QC summary",
        ),
        S(
            "sensfunc_plan_json",
            "json",
            ["@stage(sensfunc)/sensfunc_plan.json"],
            "none",
            1,
            "calibration",
            optional=True,
            description="Placeholder stage plan for sensfunc",
        ),
        S(
            "sensfunc_table_fits",
            "fits",
            ["@stage(sensfunc)/sensfunc_table.fits"],
            "sensfunc",
            dm,
            "calibration",
            optional=True,
            description="Sensitivity / response table",
        ),
        S(
            "sensfunc_model_json",
            "json",
            ["@stage(sensfunc)/sensfunc_model.json"],
            "none",
            1,
            "calibration",
            optional=True,
            description="Sensitivity-function model metadata",
        ),
        S(
            "sensfunc_preview_png",
            "png",
            ["@stage(sensfunc)/sensfunc_preview.png"],
            "none",
            None,
            "calibration",
            optional=True,
            description="Sensitivity-function preview plot",
        ),
        S(
            "sensfunc_qc_json",
            "json",
            ["@stage(sensfunc)/sensfunc_qc.json"],
            "none",
            1,
            "calibration",
            optional=True,
            description="Sensitivity-function QC summary",
        ),
        S(
            "telluric_model_plan_json",
            "json",
            ["@stage(telluric_model)/telluric_model_plan.json"],
            "none",
            1,
            "calibration",
            optional=True,
            description="Placeholder stage plan for telluric_model",
        ),
        S(
            "telluric_model_fits",
            "fits",
            ["@stage(telluric_model)/telluric_model.fits"],
            "telluric_model",
            dm,
            "calibration",
            optional=True,
            description="Telluric transmission model",
        ),
        S(
            "telluric_preview_png",
            "png",
            ["@stage(telluric_model)/telluric_preview.png"],
            "none",
            None,
            "calibration",
            optional=True,
            description="Telluric model preview",
        ),
        S(
            "telluric_qc_json",
            "json",
            ["@stage(telluric_model)/telluric_qc.json"],
            "none",
            1,
            "calibration",
            optional=True,
            description="Telluric model QC",
        ),
        S(
            "fluxcal_apply_plan_json",
            "json",
            ["@stage(fluxcal_apply)/fluxcal_apply_plan.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Placeholder stage plan for fluxcal_apply",
        ),
        S(
            "fluxcal_spec1d_fits",
            "fits",
            ["@stage(fluxcal_apply)/fluxcal_spec1d.fits"],
            "spec1d_calibrated",
            dm,
            "science",
            optional=True,
            description="Flux-calibrated 1D science spectrum",
        ),
        S(
            "fluxcal_2d_fits",
            "fits",
            ["@stage(fluxcal_apply)/fluxcal_2d.fits"],
            "mef_calibrated",
            dm,
            "science",
            optional=True,
            description="Flux-calibrated 2D product",
        ),
        S(
            "fluxcal_preview_png",
            "png",
            ["@stage(fluxcal_apply)/fluxcal_preview.png"],
            "none",
            None,
            "science",
            optional=True,
            description="Flux-calibration preview",
        ),
        S(
            "fluxcal_qc_json",
            "json",
            ["@stage(fluxcal_apply)/fluxcal_qc.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Flux-calibration QC summary",
        ),
        S(
            "calibration_systematics_fits",
            "fits",
            ["@stage(fluxcal_apply)/calibration_systematics.fits"],
            "calsys",
            dm,
            "science",
            optional=True,
            description="Calibration-systematics companion product",
        ),
        S(
            "line_measure_plan_json",
            "json",
            ["@stage(line_measure)/line_measure_plan.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Placeholder stage plan for line_measure",
        ),
        S(
            "science_lines_csv",
            "csv",
            ["@stage(line_measure)/science_lines.csv"],
            "none",
            1,
            "science",
            optional=True,
            description="Measured science-line table (CSV)",
        ),
        S(
            "science_lines_json",
            "json",
            ["@stage(line_measure)/science_lines.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Measured science-line table (JSON)",
        ),
        S(
            "science_lines_preview_png",
            "png",
            ["@stage(line_measure)/science_lines_preview.png"],
            "none",
            None,
            "science",
            optional=True,
            description="Measured science-line preview",
        ),
        S(
            "science_report_plan_json",
            "json",
            ["@stage(science_report)/science_report_plan.json"],
            "none",
            1,
            "science",
            optional=True,
            description="Placeholder stage plan for science_report",
        ),
        S(
            "science_report_md",
            "md",
            ["@stage(science_report)/science_report.md"],
            "none",
            1,
            "science",
            optional=True,
            description="Science report (Markdown)",
        ),
        S(
            "science_report_html",
            "html",
            ["@stage(science_report)/science_report.html"],
            "none",
            1,
            "science",
            optional=True,
            description="Science report (HTML)",
        ),
        S(
            "observer_dashboard_json",
            "json",
            ["observer_dashboard/observer_dashboard.json"],
            "none",
            1,
            "report",
            optional=True,
            description="Observer dashboard machine-readable summary",
        ),
        S(
            "observer_dashboard_md",
            "md",
            ["observer_dashboard/observer_dashboard.md"],
            "none",
            1,
            "report",
            optional=True,
            description="Observer dashboard markdown summary",
        ),
        S(
            "observer_dashboard_html",
            "html",
            ["observer_dashboard/observer_dashboard.html"],
            "none",
            1,
            "report",
            optional=True,
            description="Observer dashboard HTML summary",
        ),
        S(
            "recovery_plan_json",
            "json",
            ["observer_dashboard/recovery_plan.json"],
            "none",
            1,
            "report",
            optional=True,
            description="Actionable recovery workflow summary",
        ),
    )


CATALOG = ProductCatalog(specs=_specs())

__all__ = [
    "ProductSpec",
    "Product",
    "ProductCatalog",
    "CATALOG",
]
