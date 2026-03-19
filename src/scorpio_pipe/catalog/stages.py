"""Stage/task catalog (single source of truth).

This module consolidates:
- GUI stage registry (stage numbering + directory slugs)
- Pipeline task specifications (validate globs + contract kind)
- Stage contracts (product keys + metrics keys)

It is intentionally *data-first*: the catalog is the only place where a new
stage/task must be added.

Migration notes
---------------
- :mod:`scorpio_pipe.stage_registry` and :mod:`scorpio_pipe.stage_contracts`
  are kept as *thin deprecated shims* that read from this catalog.
- :mod:`scorpio_pipe.pipeline.engine` derives CANONICAL_TASKS and STAGE_SPECS
  from this catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Sequence


# -----------------------------------------------------------------------------
# Public data shapes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiStageDef:
    """GUI-visible stage definition (directory numbering + slug)."""

    id: int
    key: str
    slug: str
    label: str
    ui_only: bool = False

    @property
    def dir_name(self) -> str:
        return f"{int(self.id):02d}_{self.slug}"

    @property
    def title(self) -> str:
        return f"{int(self.id):02d} {self.label}"


@dataclass(frozen=True)
class StageInfo:
    """Single pipeline task definition.

    Field names follow the minimal design requested in P0-ARCH-01.1.

    Notes
    -----
    - ``id`` is the *GUI stage id* when the task writes under a numbered stage
      directory. For tasks that write into special locations (manifest/qc/ui),
      ``id`` is ``None``.
    - ``done_dir_policy`` is a small string enum describing where ``done.json``
      should live for the task.
    """

    # Minimal requested fields
    id: int | None
    key: str
    label: str
    ui_only: bool
    aliases: tuple[str, ...]

    entrypoint: str | None
    done_dir_policy: str
    contract_kind: str
    validate_globs: tuple[str, ...]
    product_keys: tuple[str, ...]
    metrics_keys: tuple[str, ...]

    # Optional / compatibility fields (used by shims and UI)
    stage_key: str | None = None  # canonical GUI stage key for directory mapping
    stage_slug: str | None = None  # GUI stage directory slug
    inputs: tuple[str, ...] = ()
    notes: str | None = None


# -----------------------------------------------------------------------------
# Catalog implementation
# -----------------------------------------------------------------------------


class StageCatalog:
    def __init__(
        self,
        *,
        gui_stages: Sequence[GuiStageDef],
        tasks: Sequence[StageInfo],
    ) -> None:
        self._gui_stages = tuple(gui_stages)
        self._tasks = tuple(tasks)

        self._task_by_key: dict[str, StageInfo] = {}
        for t in self._tasks:
            k = str(t.key).strip().lower()
            if not k:
                raise ValueError("StageInfo.key must be non-empty")
            if k in self._task_by_key:
                raise ValueError(f"Duplicate task key in catalog: {k!r}")
            self._task_by_key[k] = t

            # P0-ARCH-02.2: entrypoints are the source of truth for task execution.
            if not bool(getattr(t, 'ui_only', False)):
                ep = str(getattr(t, 'entrypoint', None) or '').strip()
                if not ep:
                    raise ValueError(f"Task {t.key!r} must define a non-empty entrypoint")

        # Alias -> canonical task key
        self._task_alias_to_key: dict[str, str] = {}
        for t in self._tasks:
            self._register_task_alias(t.key, t.key)
            for a in t.aliases:
                self._register_task_alias(a, t.key)

        # Build a stage-key alias map (task/stage synonyms -> canonical stage key).
        stage_keys = {s.key for s in self._gui_stages}
        self._stage_aliases: dict[str, str] = {k: k for k in stage_keys}

        # Include task keys + task aliases for tasks that map to a GUI stage.
        for t in self._tasks:
            if not t.stage_key:
                continue
            sk = str(t.stage_key).strip().lower()
            if sk not in stage_keys:
                raise ValueError(
                    f"Task {t.key!r} refers to unknown stage_key {sk!r} (not in gui_stages)"
                )
            self._stage_aliases[str(t.key).strip().lower()] = sk
            for a in t.aliases:
                aa = str(a).strip().lower()
                if aa:
                    self._stage_aliases[aa] = sk

        # Validate GUI stage ids/keys are unique.
        ids = [int(s.id) for s in self._gui_stages]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate GUI stage ids in catalog")
        if len(set(stage_keys)) != len(self._gui_stages):
            raise ValueError("Duplicate GUI stage keys in catalog")

    def _register_task_alias(self, alias: str, canonical: str) -> None:
        a = str(alias or "").strip().lower()
        c = str(canonical or "").strip().lower()
        if not a:
            return
        if a in self._task_alias_to_key and self._task_alias_to_key[a] != c:
            raise ValueError(
                f"Alias collision in StageCatalog: {a!r} -> {self._task_alias_to_key[a]!r} vs {c!r}"
            )
        self._task_alias_to_key[a] = c

    # ------------------------- Public API: GUI stages -------------------------

    def gui_stages(self) -> tuple[GuiStageDef, ...]:
        return self._gui_stages

    def iter_gui_stages(self) -> Iterator[GuiStageDef]:
        yield from self._gui_stages

    def stage_key_aliases(self) -> dict[str, str]:
        """Return mapping: task/stage aliases -> canonical GUI stage key."""

        return dict(self._stage_aliases)

    def get_gui_stage(self, key: str) -> GuiStageDef:
        k = str(key or "").strip().lower()
        k = self._stage_aliases.get(k, k)
        for s in self._gui_stages:
            if s.key == k:
                return s
        raise KeyError(f"Unknown GUI stage key: {key!r} (resolved={k!r})")

    # ------------------------- Public API: pipeline tasks ---------------------

    def task_keys(self) -> tuple[str, ...]:
        """Canonical task keys (engine-level)."""

        return tuple(t.key for t in self._tasks if not t.ui_only)

    def task_infos(self) -> tuple[StageInfo, ...]:
        return tuple(t for t in self._tasks if not t.ui_only)

    def canonical_task_name(self, name: str) -> str:
        k = str(name or "").strip().lower()
        if not k:
            raise KeyError("task name is empty")
        return self._task_alias_to_key.get(k, k)

    def get_task(self, name: str) -> StageInfo:
        canon = self.canonical_task_name(name)
        if canon not in self._task_by_key:
            raise KeyError(f"Unknown task: {name!r} (canonical={canon!r})")
        return self._task_by_key[canon]

    def task_label(self, task_name: str) -> str:
        """Human label for logging (prefers GUI numbering when available)."""

        t = self.get_task(task_name)
        if t.stage_key:
            try:
                return self.get_gui_stage(t.stage_key).title
            except Exception:
                pass
        return str(t.label)


# -----------------------------------------------------------------------------
# Canonical catalog data
# -----------------------------------------------------------------------------


_GUI_STAGES: tuple[GuiStageDef, ...] = (
    GuiStageDef(1, "project", "project", "Project", ui_only=True),
    GuiStageDef(2, "setup", "setup", "Setup", ui_only=True),
    GuiStageDef(3, "biascorr", "bias", "Bias Correction"),
    GuiStageDef(4, "flatfield", "flat", "Flat-Fielding"),
    GuiStageDef(5, "cosmics", "cosmics", "Cosmics Cleaning"),
    GuiStageDef(6, "superneon", "superneon", "Superneon"),
    GuiStageDef(7, "arclineid", "arc_line_id", "Arc Line ID"),
    GuiStageDef(8, "wavesol", "wavesol", "Wavelength Solution"),
    GuiStageDef(9, "sky", "sky", "Sky Subtraction"),
    GuiStageDef(10, "linearize", "linearize", "Linearization"),
    GuiStageDef(11, "stack2d", "stack", "Frame Stacking"),
    GuiStageDef(12, "extract1d", "extract", "Object Extraction"),
)


def _u(xs: Iterable[str]) -> tuple[str, ...]:
    """Stable order unique."""

    out: list[str] = []
    seen: set[str] = set()
    for x in xs:
        s = str(x).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return tuple(out)


_TASKS: tuple[StageInfo, ...] = (
    # UI-only placeholders (kept for GUI list completeness)
    StageInfo(
        id=1,
        key="project",
        label="Project",
        ui_only=True,
        aliases=(),
        entrypoint=None,
        done_dir_policy="ui_only",
        contract_kind="none",
        validate_globs=(),
        product_keys=(),
        metrics_keys=(),
        stage_key="project",
        stage_slug="project",
    ),
    StageInfo(
        id=2,
        key="setup",
        label="Setup",
        ui_only=True,
        aliases=(),
        entrypoint=None,
        done_dir_policy="ui_only",
        contract_kind="none",
        validate_globs=(),
        product_keys=(),
        metrics_keys=(),
        stage_key="setup",
        stage_slug="setup",
    ),

    # Engine tasks (canonical)
    StageInfo(
        id=None,
        key="manifest",
        label="Manifest",
        ui_only=False,
        aliases=(),
        entrypoint="scorpio_pipe.pipeline.entry_manifest:run",
        done_dir_policy="manifest",
        contract_kind="none",
        validate_globs=("manifest/manifest.json", "manifest/requirements.lock", "manifest/deps_snapshot.json"),
        product_keys=_u((
            "manifest",
            "dependency_lock",
            "deps_snapshot",
            "products_manifest",
            "timings",
        )),
        metrics_keys=_u((
            "provenance",
        )),
        inputs=_u(("config.yaml", "raw frames referenced by config")),
    ),
    StageInfo(
        id=3,
        key="superbias",
        label="Superbias",
        ui_only=False,
        aliases=_u((
            # stage key + common synonyms
            "biascorr",
            "bias",
            # P0-C1 naming
            "bias_combine",
            "masterbias",
        )),
        entrypoint="scorpio_pipe.stages.entry_superbias:run",
        done_dir_policy="stage",
        contract_kind="calib_bias_triplet",
        validate_globs=(
            "superbias.fits",
            "master_bias.fits",
            "master_bias_var.fits",
            "master_bias_dq.fits",
            "master_bias_index.json",
        ),
        product_keys=_u((
            "master_bias_index",
            "master_bias_fits",
            "master_bias_var_fits",
            "master_bias_dq_fits",
            "superbias_fits",
        )),
        metrics_keys=_u((
            "superbias.n_inputs",
            "superbias.combine",
            "superbias.sigma_clip",
            "superbias.n_groups",
            "superbias.default_gid",
        )),
        stage_key="biascorr",
        stage_slug="bias",
        inputs=_u(("raw/bias/*.fits",)),
    ),
    StageInfo(
        id=4,
        key="superflat",
        label="Superflat",
        ui_only=False,
        aliases=_u((
            "flat",  # frequent shorthand
        )),
        entrypoint="scorpio_pipe.stages.entry_superflat:run",
        done_dir_policy="stage",
        contract_kind="calib_flat",
        validate_globs=("superflat.fits",),
        product_keys=_u((
            "superflat_fits",
        )),
        metrics_keys=(),
        stage_key="flatfield",
        stage_slug="flat",
        inputs=_u(("raw/flat/*.fits", "superbias")),
    ),
    StageInfo(
        id=4,
        key="flatfield",
        label="Flatfield",
        ui_only=False,
        aliases=_u((
            "flatfield",  # explicit
            "flatfielding",
        )),
        entrypoint="scorpio_pipe.stages.entry_flatfield:run",
        done_dir_policy="stage",
        contract_kind="mef",
        validate_globs=("*_flat.fits",),
        product_keys=_u((
            "flatfield_done",
            "flatfield_flat_fits",
        )),
        metrics_keys=(),
        stage_key="flatfield",
        stage_slug="flat",
        inputs=_u(("raw/obj/*.fits", "superflat", "superbias")),
    ),
    StageInfo(
        id=5,
        key="cosmics",
        label="Cosmics",
        ui_only=False,
        aliases=(),
        entrypoint="scorpio_pipe.stages.entry_cosmics:run",
        done_dir_policy="stage",
        contract_kind="mef",
        validate_globs=("cosmics_summary.json", "obj/clean/*_clean.fits", "obj/mask/*_mask.fits"),
        product_keys=_u((
            "cosmics_summary",
            "cosmics_clean_fits",
            "cosmics_mask_fits",
        )),
        metrics_keys=_u((
            "cosmics.replaced_fraction",
            "cosmics.kind",
        )),
        stage_key="cosmics",
        stage_slug="cosmics",
        inputs=_u(("raw/obj/*.fits", "superbias")),
    ),
    StageInfo(
        id=6,
        key="superneon",
        label="Superneon",
        ui_only=False,
        aliases=(),
        entrypoint="scorpio_pipe.stages.entry_superneon:run",
        done_dir_policy="stage",
        contract_kind="calib_neon",
        validate_globs=("superneon.fits", "peaks_candidates.csv", "profile_peaks.png"),
        product_keys=_u((
            "superneon_fits",
            "superneon_png",
            "peaks_candidates_csv",
        )),
        metrics_keys=_u((
            "superneon.n_peaks",
            "superneon.snr",
        )),
        stage_key="superneon",
        stage_slug="superneon",
        inputs=_u(("raw/neon*.fits", "superbias (optional)")),
    ),

StageInfo(
    id=None,
    key="slit_sanity",
    label="Slit sanity",
    ui_only=False,
    aliases=_u((
        "slitcheck",
        "slit_check",
    )),
    entrypoint="scorpio_pipe.stages.entry_slit_sanity:run",
    done_dir_policy="qc",
    contract_kind="none",
    validate_globs=("slit_sanity.json",),
    product_keys=_u((
        "slit_sanity_json",
        "slit_profile_png",
        "slit_consistency_md",
    )),
    metrics_keys=_u((
        "slit.median_width_px",
        "slit.expected_width_px",
    )),
    stage_key=None,
    stage_slug=None,
    inputs=_u(("raw/flat*.fits", "raw/obj*.fits", "metadata headers")),
),
StageInfo(
    id=None,
    key="geometry_prior",
    label="Geometry prior",
    ui_only=False,
    aliases=_u((
        "geometry",
        "geom",
    )),
    entrypoint="scorpio_pipe.stages.entry_geometry_prior:run",
    done_dir_policy="stage",
    contract_kind="none",
    validate_globs=("geometry_model.json",),
    product_keys=_u((
        "geometry_model_json",
        "geometry_preview_fits",
        "geometry_qc_json",
        "repers_json",
    )),
    metrics_keys=_u((
        "geometry.trust_score",
        "geometry.fit_rms_px",
    )),
    stage_key="wavesol",
    stage_slug="wavesol",
    inputs=_u(("superneon_fits",)),
),
    StageInfo(
        id=7,
        key="lineid_prepare",
        label="LineID prep",
        ui_only=False,
        aliases=_u((
            "arclineid",
            "arc_line_id",
            "lineid",
        )),
        entrypoint="scorpio_pipe.stages.entry_lineid_prepare:run",
        done_dir_policy="stage",
        contract_kind="none",
        validate_globs=(
            "lineid_template.csv",
            "lineid_auto.csv",
            "lineid_report.txt",
        ),
        product_keys=_u((
            "lineid_template_csv",
            "lineid_auto_csv",
            "lineid_report_txt",
        )),
        metrics_keys=(),
        stage_key="arclineid",
        stage_slug="arc_line_id",
        inputs=_u(("superneon_fits", "line atlas resources")),
    ),
    StageInfo(
        id=8,
        key="wavesolution",
        label="Wavesol",
        ui_only=False,
        aliases=_u((
            "wavesol",
            "wavesol2d",
            "wavesolution2d",
            "wavelength_solution",
            "wave",
        )),
        entrypoint="scorpio_pipe.stages.entry_wavesolution:run",
        done_dir_policy="stage",
        contract_kind="lambda_map",
        validate_globs=("lambda_map.fits",),
        product_keys=_u((
            "wavesolution_1d_png",
            "wavesolution_1d_json",
            "wavesolution_2d_json",
            "lambda_map",
            "rectification_model_json",
            "wave_done_json",
            "residuals_1d_csv",
            "residuals_2d_csv",
            "control_points_2d_csv",
            "residuals_2d_png",
            "residuals_2d_audit_png",
            "residuals_vs_lambda_png",
            "residuals_vs_y_png",
            "residuals_hist_png",
            "wavesolution_report_txt",
        )),
        metrics_keys=_u((
            "wavesol.rms1d_A",
            "wavesol.wrms1d_A",
            "wavesol.sigma1d_mad_A",
            "wavesol.rms2d_A",
            "wavesol.wrms2d_A",
            "wavesol.sigma2d_mad_A",
            "wavesol.n_pairs_used",
            "wavesol.n_lines_used",
        )),
        stage_key="wavesol",
        stage_slug="wavesol",
        inputs=_u((
            "peaks_candidates_csv",
            "hand_pairs.txt (from LineID GUI or library)",
        )),
        notes="Manual line rejection must recompute RMS on the *same* residual definition as QC.",
    ),
    StageInfo(
        id=None,
        key="lsf_model",
        label="LSF model",
        ui_only=False,
        aliases=_u((
            "lsf",
            "resolution_model",
        )),
        entrypoint="scorpio_pipe.stages.entry_lsf_model:run",
        done_dir_policy="stage",
        contract_kind="none",
        validate_globs=("lsf_map.fits", "lsf_model.json", "lsf_qc.json"),
        product_keys=_u((
            "lsf_map_fits",
            "lsf_model_json",
            "lsf_qc_json",
        )),
        metrics_keys=_u((
            "lsf.sigma_A_median",
            "lsf.coverage_frac",
        )),
        stage_key=None,
        stage_slug=None,
        inputs=_u(("wavesolution", "superneon", "lineid products")),
    ),

    StageInfo(
        id=9,
        key="sky",
        label="Sky subtraction",
        ui_only=False,
        aliases=_u((
            "sky_sub",
        )),
        entrypoint="scorpio_pipe.stages.entry_sky:run",
        done_dir_policy="stage",
        contract_kind="mef",
        validate_globs=("*_skysub.fits", "*_skysub_raw.fits", "*_skymodel.fits", "preview.fits"),
        product_keys=_u((
            "sky_done",
            "qc_sky_json",
            "sky_preview_fits",
            "sky_preview_png",
            "sky_skysub_raw_fits",
        )),
        metrics_keys=_u((
            "sky.rms_sky",
            "sky.flexure_shift_A",
        )),
        stage_key="sky",
        stage_slug="sky",
        inputs=_u((
            "05_cosmics/*_clean.fits (preferred) OR raw/obj/*.fits (RAW detector geometry)",
            "05_cosmics/*_mask.fits (optional)",
            "08_wavesol/lambda_map.fits (λ(x,y) in RAW geometry)",
            "ROI/geometry settings (cfg.sky.geometry.*)",
        )),
    ),
    StageInfo(
        id=None,
        key="sky_refine",
        label="Sky residual refine",
        ui_only=False,
        aliases=_u((
            "sky_residual_refine",
            "skyrefine",
        )),
        entrypoint="scorpio_pipe.stages.entry_sky_refine:run",
        done_dir_policy="task",
        contract_kind="mef",
        validate_globs=("*_skysub_refined_raw.fits", "*_skycorr_raw.fits", "qc_sky_refine.json"),
        product_keys=_u((
            "sky_refine_done",
            "qc_sky_refine_json",
            "sky_refined_raw_fits",
            "skycorr_raw_fits",
            "sky_residual_windows_csv",
            "sky_residual_windows_json",
            "sky_observer_pack_md",
        )),
        metrics_keys=_u((
            "sky_refine.sky_sigma_before",
            "sky_refine.sky_sigma_after",
            "sky_refine.sky_sigma_gain",
        )),
        stage_key=None,
        stage_slug=None,
        inputs=_u((
            "09_sky/*_skysub_raw.fits",
            "09_sky/*_skymodel_raw.fits",
            "08_wavesol/lambda_map.fits",
        )),
    ),
    StageInfo(
        id=10,
        key="linearize",
        label="Linearize",
        ui_only=False,
        aliases=_u(("lin",)),
        entrypoint="scorpio_pipe.stages.entry_linearize:run",
        done_dir_policy="stage",
        contract_kind="mef",
        validate_globs=("lambda_map.fits", "*_skysub.fits", "*_lin.fits", "linearize_qc.json"),
        product_keys=_u((
            "linearize_done",
            "lin_preview_fits",
            "lin_preview_png",
            "linearize_qc",
            "linearize_skysub_fits",
            "linearize_skymodel_fits",
        )),
        metrics_keys=(),
        stage_key="linearize",
        stage_slug="linearize",
        inputs=_u(("raw/obj/*.fits", "lambda_map.fits")),
    ),
    StageInfo(
        id=11,
        key="stack2d",
        label="Stack2D",
        ui_only=False,
        aliases=_u(("stack",)),
        entrypoint="scorpio_pipe.stages.entry_stack2d:run",
        done_dir_policy="stage",
        contract_kind="mef",
        validate_globs=("stack2d*.fits", "stacked2d*.fits"),
        product_keys=_u((
            "stack2d_done",
            "stack_done",
            "stacked2d_fits",
            "stack2d_fits",
            "eta_lambda_fits",
            "coverage_png",
        )),
        metrics_keys=(),
        stage_key="stack2d",
        stage_slug="stack",
        inputs=_u(("per-exp sky-subtracted frames (*_skysub.fits)",)),
    ),
    StageInfo(
        id=12,
        key="extract1d",
        label="Extract1D",
        ui_only=False,
        aliases=_u(("extract",)),
        entrypoint="scorpio_pipe.stages.entry_extract1d:run",
        done_dir_policy="stage",
        contract_kind="spec1d",
        validate_globs=("spec1d*.fits", "extraction_qc.json", "extraction_diagnostics.png"),
        product_keys=_u((
            "extract_done",
            "trace_json",
            "spec1d_fits",
            "spec1d_png",
            "extract1d_done",
            "extraction_qc_json",
            "extraction_diagnostics_png",
        )),
        metrics_keys=(),
        stage_key="extract1d",
        stage_slug="extract",
        inputs=_u((
            "11_stack/stack2d.fits (default) or 10_linearize/<stem>_skysub.fits (extract1d.input_mode=single_frame)",
        )),
    ),
    StageInfo(
        id=None,
        key="publication_ready",
        label="Publication ready",
        ui_only=False,
        aliases=_u(("publication", "pubready")),
        entrypoint="scorpio_pipe.stages.entry_publication_ready:run",
        done_dir_policy="stage",
        contract_kind="none",
        validate_globs=("Spectrum.fts", "wl.fts", "dismap.fts", "publication_provenance.json"),
        product_keys=_u((
            "publication_spectrum_fts",
            "publication_wl_fts",
            "publication_dismap_fts",
            "publication_provenance_json",
        )),
        metrics_keys=_u(("publication_ready.spectrum_shape_x",)),
        stage_key=None,
        stage_slug=None,
        inputs=_u(("stack2d", "wavesolution", "lsf_model")),
    ),
    StageInfo(
        id=None,
        key="kinematics_prep",
        label="Kinematics prep",
        ui_only=False,
        aliases=_u(("kinprep", "ppxf_prep")),
        entrypoint="scorpio_pipe.stages.entry_kinematics_prep:run",
        done_dir_policy="stage",
        contract_kind="none",
        validate_globs=("spec_kin.fits", "masks.json", "README.md"),
        product_keys=_u((
            "spec_kin_fits",
            "kinematics_masks_json",
            "kinematics_readme_md",
        )),
        metrics_keys=_u((
            "kinematics_prep.velscale_kms",
            "kinematics_prep.sigma_inst_A_median",
        )),
        stage_key=None,
        stage_slug=None,
        inputs=_u(("extract1d", "lsf_model", "sky_refine (optional)")),
    ),

    StageInfo(
        id=None,
        key="standard_prepare",
        label="Standard prepare",
        ui_only=False,
        aliases=_u(("standard", "std_prepare")),
        entrypoint="scorpio_pipe.stages.entry_standard_prepare:run",
        done_dir_policy="stage",
        contract_kind="standard_prepare",
        validate_globs=("standard_prepare_plan.json", "standard_1d.fits", "standard_trace.json", "standard_qc.json"),
        product_keys=_u((
            "standard_prepare_plan_json",
            "standard_1d_fits",
            "standard_trace_json",
            "standard_qc_json",
        )),
        metrics_keys=_u(("standard_prepare.status",)),
        inputs=_u(("linearize outputs", "STANDARD_FRAMES / TELLURIC_STANDARD_FRAMES")),
    ),
    StageInfo(
        id=None,
        key="sensfunc",
        label="Sensfunc",
        ui_only=False,
        aliases=_u(("response", "sensitivity")),
        entrypoint="scorpio_pipe.stages.entry_sensfunc:run",
        done_dir_policy="stage",
        contract_kind="sensfunc",
        validate_globs=("sensfunc_plan.json", "sensfunc_table.fits", "sensfunc_model.json", "sensfunc_preview.png", "sensfunc_qc.json"),
        product_keys=_u((
            "sensfunc_plan_json",
            "sensfunc_table_fits",
            "sensfunc_model_json",
            "sensfunc_preview_png",
            "sensfunc_qc_json",
        )),
        metrics_keys=_u(("sensfunc.status",)),
        inputs=_u(("standard_prepare",)),
    ),
    StageInfo(
        id=None,
        key="telluric_model",
        label="Telluric model",
        ui_only=False,
        aliases=_u(("telluric",)),
        entrypoint="scorpio_pipe.stages.entry_telluric_model:run",
        done_dir_policy="stage",
        contract_kind="telluric_model",
        validate_globs=("telluric_model_plan.json", "telluric_model.fits", "telluric_preview.png", "telluric_qc.json"),
        product_keys=_u((
            "telluric_model_plan_json",
            "telluric_model_fits",
            "telluric_preview_png",
            "telluric_qc_json",
        )),
        metrics_keys=_u(("telluric_model.status",)),
        inputs=_u(("standard_prepare",)),
    ),
    StageInfo(
        id=None,
        key="fluxcal_apply",
        label="Flux calibration",
        ui_only=False,
        aliases=_u(("fluxcal", "flux_calibration")),
        entrypoint="scorpio_pipe.stages.entry_fluxcal_apply:run",
        done_dir_policy="stage",
        contract_kind="fluxcal_apply",
        validate_globs=("fluxcal_apply_plan.json", "fluxcal_spec1d.fits", "fluxcal_2d.fits", "fluxcal_preview.png", "fluxcal_qc.json", "calibration_systematics.fits"),
        product_keys=_u((
            "fluxcal_apply_plan_json",
            "fluxcal_spec1d_fits",
            "fluxcal_2d_fits",
            "fluxcal_preview_png",
            "fluxcal_qc_json",
            "calibration_systematics_fits",
        )),
        metrics_keys=_u(("fluxcal_apply.status",)),
        inputs=_u(("extract1d", "sensfunc", "telluric_model")),
    ),
    StageInfo(
        id=None,
        key="line_measure",
        label="Line measurement",
        ui_only=False,
        aliases=_u(("science_lines",)),
        entrypoint="scorpio_pipe.stages.entry_line_measure:run",
        done_dir_policy="stage",
        contract_kind="line_measure",
        validate_globs=("line_measure_plan.json", "science_lines.csv", "science_lines.json", "science_lines_preview.png"),
        product_keys=_u((
            "line_measure_plan_json",
            "science_lines_csv",
            "science_lines_json",
            "science_lines_preview_png",
        )),
        metrics_keys=_u(("line_measure.status",)),
        inputs=_u(("fluxcal_apply",)),
    ),
    StageInfo(
        id=None,
        key="science_report",
        label="Science report",
        ui_only=False,
        aliases=_u(("report_science",)),
        entrypoint="scorpio_pipe.stages.entry_science_report:run",
        done_dir_policy="stage",
        contract_kind="science_report",
        validate_globs=("science_report_plan.json", "science_report.md", "science_report.html"),
        product_keys=_u((
            "science_report_plan_json",
            "science_report_md",
            "science_report_html",
        )),
        metrics_keys=_u(("science_report.status",)),
        inputs=_u(("line_measure",)),
    ),
    StageInfo(
        id=None,
        key="qc_report",
        label="QC report",
        ui_only=False,
        aliases=_u(("qc",)),
        entrypoint="scorpio_pipe.pipeline.entry_qc_report:run",
        done_dir_policy="qc",
        contract_kind="none",
        validate_globs=(
            "qc/qc.json",
            "qc/index.html",
            "qc/timings.json",
            "qc/metrics.json",
        ),
        product_keys=_u((
            "qc_json",
            "qc_html",
            "timings",
            "products_manifest",
        )),
        metrics_keys=(),
        inputs=_u(("products registry", "stage state", "timings")),
    ),
    StageInfo(
        id=None,
        key="observer_dashboard",
        label="Observer dashboard",
        ui_only=False,
        aliases=_u(("dashboard", "observer_qc", "cockpit")),
        entrypoint="scorpio_pipe.stages.entry_observer_dashboard:run",
        done_dir_policy="stage",
        contract_kind="none",
        validate_globs=("observer_dashboard.json", "observer_dashboard.md", "observer_dashboard.html", "recovery_plan.json"),
        product_keys=_u((
            "observer_dashboard_json",
            "observer_dashboard_md",
            "observer_dashboard_html",
            "recovery_plan_json",
        )),
        metrics_keys=(),
        inputs=_u(("qc_report", "geometry_prior", "wavesolution", "sky_refine", "extract1d")),
    ),
    StageInfo(
        id=None,
        key="navigator",
        label="Navigator",
        ui_only=False,
        aliases=_u(("nav",)),
        entrypoint="scorpio_pipe.pipeline.entry_navigator:run",
        done_dir_policy="ui",
        contract_kind="none",
        validate_globs=("ui/navigator/index.html", "ui/navigator/data.json"),
        product_keys=(),
        metrics_keys=(),
        inputs=(),
    ),
)


CATALOG = StageCatalog(gui_stages=_GUI_STAGES, tasks=_TASKS)


# Convenience re-exports (public API)

def iter_gui_stages() -> Iterator[GuiStageDef]:
    """Iterate GUI stages (incl. UI-only placeholders)."""

    yield from CATALOG.iter_gui_stages()

