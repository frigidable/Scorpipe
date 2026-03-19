from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RecipeDef:
    key: str
    label: str
    tasks: tuple[str, ...]
    description: str


RECIPES: dict[str, RecipeDef] = {
    'quicklook_science': RecipeDef(
        key='quicklook_science',
        label='Quicklook science',
        tasks=('manifest', 'superbias', 'biascorr', 'wavesolution', 'lsf_model', 'sky', 'sky_refine', 'linearize', 'stack2d', 'extract1d', 'publication_ready', 'kinematics_prep', 'qc_report', 'observer_dashboard'),
        description='Fast observer quicklook from raw science frames to extracted spectra and QC.',
    ),
    'full_reduction': RecipeDef(
        key='full_reduction',
        label='Full reduction',
        tasks=('manifest', 'superbias', 'biascorr', 'superflat', 'flatfield', 'cosmics', 'superneon', 'slit_sanity', 'geometry_prior', 'lineid_prepare', 'arclineid', 'wavesolution', 'lsf_model', 'sky', 'sky_refine', 'linearize', 'stack2d', 'extract1d', 'publication_ready', 'kinematics_prep', 'qc_report', 'observer_dashboard'),
        description='Full reduction stack without flux calibration / science interpretation.',
    ),
    'standard_star_reduction': RecipeDef(
        key='standard_star_reduction',
        label='Standard-star reduction',
        tasks=('manifest', 'superbias', 'biascorr', 'superflat', 'flatfield', 'superneon', 'slit_sanity', 'geometry_prior', 'lineid_prepare', 'arclineid', 'wavesolution', 'standard_prepare'),
        description='Reduce and extract the standard star as a calibration object.',
    ),
    'relative_flux_calibration': RecipeDef(
        key='relative_flux_calibration',
        label='Relative flux calibration',
        tasks=('manifest', 'superbias', 'biascorr', 'superflat', 'flatfield', 'superneon', 'slit_sanity', 'geometry_prior', 'lineid_prepare', 'arclineid', 'wavesolution', 'standard_prepare', 'sensfunc', 'extract1d', 'fluxcal_apply'),
        description='Build sensfunc and apply relative flux calibration.',
    ),
    'full_science_calibration_report': RecipeDef(
        key='full_science_calibration_report',
        label='Full science + calibration + report',
        tasks=('manifest', 'superbias', 'biascorr', 'superflat', 'flatfield', 'cosmics', 'superneon', 'slit_sanity', 'geometry_prior', 'lineid_prepare', 'arclineid', 'wavesolution', 'lsf_model', 'sky', 'sky_refine', 'linearize', 'stack2d', 'extract1d', 'publication_ready', 'kinematics_prep', 'standard_prepare', 'sensfunc', 'telluric_model', 'fluxcal_apply', 'line_measure', 'science_report', 'qc_report', 'observer_dashboard'),
        description='Closed observer workflow from reduction to calibrated report and science lines.',
    ),
}


def get_recipe(key: str) -> RecipeDef:
    try:
        return RECIPES[str(key)]
    except KeyError as e:
        raise KeyError(f'Unknown recipe: {key}') from e


def recipe_tasks(key: str) -> tuple[str, ...]:
    return get_recipe(key).tasks


def iter_recipes() -> Iterable[RecipeDef]:
    return RECIPES.values()
