"""Audit only: SCORPIO-1 historic He-Ne-Ar arc line identity and blends.

NIST *Strong Lines* is an abbreviated handbook, not the complete ASD.
Data below are explicit documentary claims, not machine-generated
identifications or laboratory-strength predictions. This module never
rewrites manual LineID, Wave2D, the LSF or any downstream science file.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


FROZEN_HAND_PAIRS_SHA256 = (
    "2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623"
)
FROZEN_LINE_LIST_SHA256 = (
    "5d681399323d7f861120f48203e4278d09d8c45d40cba056976d7b40de1a869a"
)
SOURCES = {
    "He": "https://physics.nist.gov/PhysRefData/Handbook/Tables/heliumtable2.htm",
    "Ar": "https://physics.nist.gov/PhysRefData/Handbook/Tables/argontable2.htm",
    "Ne": "https://physics.nist.gov/PhysRefData/Handbook/Tables/neontable2.htm",
    "NIST_ASD": "https://physics.nist.gov/PhysRefData/ASD/lines_form.html",
    "SCORPIO1": "https://www.sao.ru/hq/lsfvo/devices/scorpio/BSAO_eng.pdf",
    "ADAPTER": "https://www.sao.ru/hq/lsfvo/adap/index-en.html",
    "NE_508": "https://www.mdpi.com/2218-2004/7/1/9",
    "AR_522": "https://www.mdpi.com/2218-2004/7/1/9",
}

# Original wavelength, species assignment, direct handbook line,
# classification. A direct NIST strong-line match does NOT validate
# the shape or isolation of the instrumental emission-line profile.
_REFS = (
    (3888.65, "He I", 3888.6489, "MULTIPLET"),
    (3948.98, "Ar I", 3948.979, "CATALOG_MATCH"),
    (4044.42, "Ar I", 4044.418, "NEIGHBOR_REVIEW"),
    (4181.88, "Ar I", 4181.884, "NEIGHBOR_REVIEW"),
    (4190.71, "Ar I", 4190.713, "UNRESOLVED_MULTIPLET"),
    (4259.36, "Ar I", 4259.362, "CATALOG_MATCH"),
    (4333.56, "Ar I", 4333.561, "NEIGHBOR_REVIEW"),
    (4345.17, "Ar I", 4345.168, "NEIGHBOR_REVIEW"),
    (4471.48, "He I", 4471.479, "MULTIPLET"),
    (4713.00, "He I", 4713.146, "EFFECTIVE_CENTROID_REVIEW"),
    (4863.96, "UNKNOWN", None, "LABORATORY_ID_UNKNOWN"),
    (4921.93, "He I", 4921.931, "CATALOG_MATCH"),
    (5015.68, "He I", 5015.678, "CATALOG_MATCH"),
    (5037.75, "Ne I", 5037.7512, "MIXED_SPECIES_PAIR"),
    (5047.74, "He I", 5047.74, "MIXED_SPECIES_PAIR"),
    (5080.38, "Ne I", 5080.38, "INDEPENDENT_NEI_REFERENCE"),
    (5221.80, "UNKNOWN", None, "LABORATORY_ID_UNKNOWN"),
    (5330.78, "Ne I", 5330.7775, "NEIGHBOR_REVIEW"),
    (5341.60, "Ne I candidates", None, "EFFECTIVE_CENTROID_REVIEW"),
)

_NEIGHBORS = {
    3888.65: ((3888.6046, "He I"),(3888.6456, "He I")),
    4044.42: ((4042.894, "Ar II"),),
    4181.88: ((4179.297, "Ar II"),),
    4190.71: ((4191.029, "Ar I"),),
    4333.56: ((4332.030, "Ar II"),(4335.338, "Ar I")),
    4345.17: ((4348.064, "Ar II"),),
    4471.48: ((4471.68, "He I"),),
    4713.00: ((4713.38, "He I"),),
    5221.80: ((5221.27, "Ar I PROVISIONAL: experimental Stark paper"),),
    5330.78: ((5326.3968, "Ne I, historic lab study"),),
    5341.60: ((5341.0938, "Ne I"),(5343.2834, "Ne I")),
}

# The 5221.27 Å Ar I candidate is NOT a NIST ASD laboratory validation
# of the old 5221.80 Å. Absence from the NIST Handbook cannot refute it.
# The source He-Ne-Ar lamp at SCORPIO-1 is grounded by the instrument
# paper; the two 2024 FITS headers only state IMAGETYP='neon'.


def infer_positive_blend_centroid_ratio(
    component_air_A: tuple[float, float],
    effective_air_A: float,
) -> dict[str, float | bool]:
    """Hypothetical narrow-line flux ratio needed for an effective centroid.

    Assumes exactly two positive component integrated fluxes, symmetric
    identical normalized profiles and no additional unlisted neighbors.
    A mathematically plausible ratio is NOT a measurement of lamp fluxes.
    """
    a, b = [float(q) for q in component_air_A]
    center = float(effective_air_A)
    if not np.isfinite([a,b,center]).all() or not a < b:
        raise ValueError("Expected two ordered, finite laboratory air wavelengths")
    if not a <= center <= b:
        raise ValueError("A positive-flux blend centroid must lie between its components")
    frac_red = (center-a)/(b-a)
    return {
        "fraction_red_component_of_total_flux":float(frac_red),
        "red_to_blue_integrated_flux_ratio":float(
            frac_red/(1-frac_red) if frac_red < 1 else np.inf
        ),
        "scientifically_measured_line_ratio":False,
    }


def gaussian_component_identifiability(
    first_air_A: float, second_air_A: float, *, fwhm_A: float,
) -> dict[str, float | bool]:
    """Analytic normalized-Gaussian profile overlap, not a significance test."""
    a,b,w=map(float,(first_air_A,second_air_A,fwhm_A))
    if not np.isfinite([a,b,w]).all() or not a<b or w<=0:
        raise ValueError("Expected distinct ordered wavelengths and positive FWHM")
    sigma=w/2.354820045
    correlation=float(np.exp(-0.25*((b-a)/sigma)**2))
    # Condition number for symmetric two-column normalized Gram
    # matrix (without continuum or additional pixels).
    condition=float((1+correlation)/(1-correlation))
    return {
        "component_spacing_A": b-a,
        "separation_in_fwhm": (b-a)/w,
        "gaussian_profile_inner_product": correlation,
        "two_component_gram_condition_number":condition,
        "ill_conditioned_gram_over_100":bool(condition>100.0),
        "line_strengths_inferred":False,
        "science_model_approved":False,
    }


def _parse_hand_pairs(data: bytes) -> list[tuple[float,float]]:
    try:
        string=data.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("Frozen manual LineID must be UTF-8") from exc
    pairs=[]
    for raw in string.splitlines():
        stripped=raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        columns=stripped.split()
        if len(columns)!=2:
            raise ValueError("Expected two-column immutable manual wavelength anchors")
        x,lam=map(float,columns)
        if not np.isfinite([x,lam]).all() or x<0 or lam<=0:
            raise ValueError("Nonfinite or unphysical manual LineID")
        pairs.append((x,lam))
    if len(pairs)!=19 or not all(
        a[0] < b[0] and a[1] < b[1]
        for a,b in zip(pairs,pairs[1:])
    ):
        raise ValueError("Expected 19 strictly ordered immutable anchors")
    return pairs


def review_sc1_original_lineid(
    hand_pairs_path: Path,
    original_moiseev_line_list_path: Path,
) -> dict[str, Any]:
    """SHA-provenance line identity review; errors block further diagnosis.

    No catalog auto-discovery, wavelength offset correction or line
    replacement happens inside this function.
    """
    hp=Path(hand_pairs_path)
    listing=Path(original_moiseev_line_list_path)
    source_bytes=hp.read_bytes()
    list_bytes=listing.read_bytes()
    if (hashlib.sha256(source_bytes).hexdigest()!=FROZEN_HAND_PAIRS_SHA256
        or hashlib.sha256(list_bytes).hexdigest()!=FROZEN_LINE_LIST_SHA256):
        raise RuntimeError("Source SHA mismatch: historical LineID audit is not reproducible")
    pairs=_parse_hand_pairs(source_bytes)
    obs=np.asarray([lam for _,lam in pairs],float)
    ref=np.asarray([row[0] for row in _REFS],float)
    if not np.array_equal(obs,ref):
        raise ValueError("LineID differs from reviewed 19 source wavelengths")
    listed=[]
    for raw in list_bytes.decode("utf-8-sig").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        listed.append(float(raw.split()[0]))
    if not all(np.any(np.isclose(listed,lam,rtol=0.0,atol=1e-8))
               for lam in obs):
        raise ValueError("An anchored line is absent from historic Moiseev list")
    output=[]
    for (x,lam),(_,ion,lab,status) in zip(pairs,_REFS):
        neighbors=[
            {"air_A":pos, "species_or_evidence":ionn, "offset_from_anchor_A":pos-lam}
            for pos,ionn in _NEIGHBORS.get(lam,())
        ]
        evidence=(
            "NIST Handbook He/Ar/Ne Strong Lines AIR"
            if lab is not None and status!="INDEPENDENT_NEI_REFERENCE"
            else (
                "Independent 2012 and 2019 Ne I sources"
                if status=="INDEPENDENT_NEI_REFERENCE"
                else "Historic SCORPIO line list; provisional candidate only"
            )
        )
        output.append({
            "x_manual_zero_based":x,
            "historic_air_A":lam,
            "species_candidate":ion,
            "published_transition_air_A":lab,
            "historic_minus_published_A":None if lab is None else lam-lab,
            "status":status,
            "neighbor_candidates":neighbors,
            "reference_quality":evidence,
            "isolation_verified_from_raw_arc":False,
            "lab_line_approved_for_new_wave2d":False,
        })
    red_ratio=infer_positive_blend_centroid_ratio(
        (5341.0938,5343.2834),5341.60
    )
    return {
        "status":"LAB_ATLAS_REVIEW_REQUIRED",
        "n_historic_anchors":len(output),
        "wavelength_convention":"air",
        "instrument":"SCORPIO-1 VPHG1200B",
        "calibration_source_documented":"He-Ne-Ar",
        "specific_2024_lamp_filling_confirmed_from_FITS":False,
        "hand_pairs_sha256":FROZEN_HAND_PAIRS_SHA256,
        "moiseev_line_list_sha256":FROZEN_LINE_LIST_SHA256,
        "entries":output,
        "red_5341_60_blend_hypothesis":{
            **red_ratio,
            "assumption":"exactly two symmetric Ne I components with positive flux"
        },
        "arity_and_line_id_risks":{
            "unresolved_atomic_identifications_air_A":[4863.96,5221.80],
            "independently_supported_nei_air_A":[5080.38],
            "mixed_species_pair_air_A":[5037.75,5047.74],
            "unresolved_sub_fwhm_nearby_lines_air_A":[4190.713,4191.029],
        },
        "full_ASD_transition_export_completed":False,
        "external_night_lineid_check_completed":False,
        "wave2d_calibration_approved":False,
        "lsf_for_stellar_science_approved":False,
        "auto_rewrite_hand_pairs":False,
        "science_hold":True,
        "sources":SOURCES,
    }
