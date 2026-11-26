"""Presentation-only taxonomy for the Scorpipe reduction pipeline.

This module defines navigation groups and descriptive help without changing
canonical task names, the execution graph, the QC gate, or numbered directories.
Labels come from the canonical stage/task catalog; they are not copied here.
"""

from __future__ import annotations

from dataclasses import dataclass

from scorpio_pipe.catalog.stages import CATALOG


@dataclass(frozen=True)
class PipelineSection:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class PipelinePresentation:
    key: str
    label: str
    section_key: str
    description: str


SECTIONS: tuple[PipelineSection, ...] = (
    PipelineSection("observations", "Observations & Setup", "Inspect observation frames, classify data, and configure the reduction."),
    PipelineSection("detector", "Detector Processing", "Build detector calibrations and reject non-astronomical artifacts."),
    PipelineSection("spectral", "Spectral Calibration", "Determine the wavelength solution and characterize spectral resolution."),
    PipelineSection("science", "Science Reduction", "Model sky background, resample, coadd, and extract science spectra."),
    PipelineSection("spectrophotometry", "Spectrophotometric Calibration", "Reduce standards, derive sensitivity, and calibrate fluxes."),
    PipelineSection("analysis", "Analysis & Quality Control", "Measure lines, inspect quality, and export science products."),
)

# Keys here are stable identifiers. This ordering is NOT an execution order.
GUI_STAGE_SECTIONS: dict[str, str] = {
    "project": "observations",
    "setup": "observations",
    "biascorr": "detector",
    "flatfield": "detector",
    "cosmics": "detector",
    "superneon": "spectral",
    "arclineid": "spectral",
    "wavesol": "spectral",
    "sky": "science",
    "linearize": "science",
    "stack2d": "science",
    "extract1d": "science",
}

# Canonical engine tasks are classified independently of their legacy GUI page.
TASK_SECTIONS: dict[str, str] = {
    "manifest": "observations",
    "superbias": "detector",
    "superflat": "detector",
    "flatfield": "detector",
    "cosmics": "detector",
    "superneon": "spectral",
    "slit_sanity": "spectral",
    "geometry_prior": "spectral",
    "lineid_prepare": "spectral",
    "wavesolution": "spectral",
    "lsf_model": "spectral",
    "sky": "science",
    "sky_refine": "science",
    "linearize": "science",
    "stack2d": "science",
    "extract1d": "science",
    "standard_prepare": "spectrophotometry",
    "sensfunc": "spectrophotometry",
    "telluric_model": "spectrophotometry",
    "fluxcal_apply": "spectrophotometry",
    "publication_ready": "analysis",
    "kinematics_prep": "analysis",
    "line_measure": "analysis",
    "science_report": "analysis",
    "qc_report": "analysis",
    "observer_dashboard": "analysis",
    "navigator": "analysis",
}

GUI_STAGE_HELP: dict[str, str] = {
    "project": "Inspect and organize observations for the current reduction workspace.",
    "setup": "Select input frames, instrumental configuration, and reduction parameters.",
    "biascorr": "Construct master bias data and subtract electronic bias from CCD frames.",
    "flatfield": "Correct relative detector response using compatible flat-field calibrations.",
    "cosmics": "Identify and mask or replace cosmic-ray affected detector samples.",
    "superneon": "Combine compatible arc-lamp exposures for wavelength calibration.",
    "arclineid": "Identify reference emission lines in the combined arc spectrum.",
    "wavesol": "Derive a two-dimensional wavelength mapping for the detector.",
    "sky": "Model and subtract night-sky emission on the original detector grid.",
    "linearize": "Resample spectra onto a defined wavelength grid, preserving flux conventions.",
    "stack2d": "Register and statistically combine calibrated two-dimensional exposures.",
    "extract1d": "Extract one-dimensional spectra from chosen spatial apertures.",
}

TASK_HELP: dict[str, str] = {
    "manifest": "Record observation inputs, configuration, and provenance.",
    "superbias": "Combine bias frames into a master bias and uncertainty products.",
    "superflat": "Combine lamp flat frames into a reference flat-field product.",
    "flatfield": "Apply compatible flat-field corrections to science frames.",
    "cosmics": "Detect cosmic rays and propagate data-quality information.",
    "superneon": "Combine arc exposures for reference line measurements.",
    "slit_sanity": "Evaluate observational consistency of the slit geometry.",
    "geometry_prior": "Derive geometric constraints used by wavelength calibration.",
    "lineid_prepare": "Prepare reference-line candidates and operator selections.",
    "wavesolution": "Fit the two-dimensional wavelength calibration.",
    "lsf_model": "Characterize instrumental line-spread properties.",
    "sky": "Model and subtract the atmospheric sky spectrum.",
    "sky_refine": "Evaluate and refine residual sky-background structure.",
    "linearize": "Resample science spectra to a specified wavelength grid.",
    "stack2d": "Coadd aligned two-dimensional science exposures.",
    "extract1d": "Extract one-dimensional spectra in selected spatial apertures.",
    "standard_prepare": "Prepare and extract an observed spectrophotometric standard.",
    "sensfunc": "Derive wavelength-dependent instrumental sensitivity.",
    "telluric_model": "Estimate wavelength-dependent atmospheric molecular transmission.",
    "fluxcal_apply": "Apply the selected flux-calibration model to science products.",
    "publication_ready": "Export science products without asserting publication qualification.",
    "kinematics_prep": "Prepare spectra and metadata for kinematic analysis.",
    "line_measure": "Measure emission-line properties from extracted spectra.",
    "science_report": "Generate a scientific measurements summary.",
    "qc_report": "Generate a quality-control report from reduction diagnostics.",
    "observer_dashboard": "Summarize reduction state and quality for the observer.",
    "navigator": "Collect pointers to reduction outputs and their state.",
}

_SECTION_BY_KEY = {section.key: section for section in SECTIONS}


def get_section(key: str) -> PipelineSection:
    """Resolve a UI-only grouping identifier (not a pipeline task)."""
    return _SECTION_BY_KEY[key]


def gui_stage_presentation(key: str) -> PipelinePresentation:
    """Presentation for a legacy GUI stage; task/stage aliases are accepted."""
    stage = CATALOG.get_gui_stage(key)
    return PipelinePresentation(
        key=stage.key,
        label=stage.label,
        section_key=GUI_STAGE_SECTIONS[stage.key],
        description=GUI_STAGE_HELP[stage.key],
    )


def task_presentation(key: str) -> PipelinePresentation:
    """Presentation for a canonical engine task; aliases are accepted."""
    task = CATALOG.get_task(key)
    return PipelinePresentation(
        key=task.key,
        label=task.label,
        section_key=TASK_SECTIONS[task.key],
        description=TASK_HELP[task.key],
    )


def gui_stages_in_section(section_key: str) -> tuple[PipelinePresentation, ...]:
    """Navigation group contents in legacy GUI order; never a run recipe."""
    get_section(section_key)
    return tuple(
        gui_stage_presentation(stage.key)
        for stage in CATALOG.gui_stages()
        if GUI_STAGE_SECTIONS[stage.key] == section_key
    )
