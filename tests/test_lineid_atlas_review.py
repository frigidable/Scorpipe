"""Pure regression tests: original He-Ne-Ar 19-line audit remains science-HOLD."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scorpio_pipe import lineid_atlas_review as review


def _synthetic_original_sources(tmp_path: Path, monkeypatch):
    positions=np.arange(19,dtype=float)*90+350.
    historical=[r[0] for r in review._REFS]
    body=(
        "# frozen synthetic two-column hand pairs\n"
        + "".join(f"{x:.3f} {lam:.4f}\n" for x,lam in zip(positions,historical))
    ).encode("utf-8")
    original=(
        "# synthetic original reference list\n"
        + "".join(f"{lam:.4f} 3 0.5\n" for lam in historical)
    ).encode("utf-8")
    hp=tmp_path/"hand_pairs.txt"
    catalog=tmp_path/"lines_long(1).tab"
    hp.write_bytes(body)
    catalog.write_bytes(original)
    monkeypatch.setattr(
        review,"FROZEN_HAND_PAIRS_SHA256",hashlib.sha256(body).hexdigest()
    )
    monkeypatch.setattr(
        review,"FROZEN_LINE_LIST_SHA256",hashlib.sha256(original).hexdigest()
    )
    return hp,catalog


def test_red_multiplet_historic_effective_centroid_requires_positive_flux():
    ratio=review.infer_positive_blend_centroid_ratio(
        (5341.0938,5343.2834),5341.60
    )
    assert ratio["fraction_red_component_of_total_flux"]==pytest.approx(
        .5062/2.1896,abs=1e-9
    )
    assert ratio["red_to_blue_integrated_flux_ratio"]==pytest.approx(
        .5062/1.6834,abs=1e-9
    )
    assert ratio["scientifically_measured_line_ratio"] is False
    with pytest.raises(ValueError,match="between its components"):
        review.infer_positive_blend_centroid_ratio(
            (5341.0938,5343.2834),5345.1
        )


def test_ar_4191_unresolved_neighbors_ill_conditioned_at_observed_fwhm():
    q=review.gaussian_component_identifiability(
        4190.713,4191.029,fwhm_A=6.0
    )
    assert q["separation_in_fwhm"] < .06
    assert q["ill_conditioned_gram_over_100"] is True
    assert q["two_component_gram_condition_number"] > 100
    assert q["science_model_approved"] is False
    extremely_close=review.gaussian_component_identifiability(
        4190.713,4190.713+1e-9,fwhm_A=6.0
    )
    assert np.isfinite(extremely_close["two_component_gram_condition_number"])
    assert extremely_close["ill_conditioned_gram_over_100"] is True


def test_red_ne_doublet_resolved_better_than_ar_but_not_science_approved():
    red=review.gaussian_component_identifiability(
        5341.0938,5343.2834,fwhm_A=6.0
    )
    argon=review.gaussian_component_identifiability(
        4190.713,4191.029,fwhm_A=6.0
    )
    assert red["two_component_gram_condition_number"] < (
        argon["two_component_gram_condition_number"]
    )
    assert red["line_strengths_inferred"] is False


def test_canonical_original_source_hashes_cannot_be_63_digits():
    assert len(review.FROZEN_HAND_PAIRS_SHA256)==64
    assert len(review.FROZEN_LINE_LIST_SHA256)==64
    assert review.FROZEN_HAND_PAIRS_SHA256==(
        "2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623"
    )


def test_review_all_19_including_original_5050_region_mixed_species(
    tmp_path,monkeypatch
):
    hp,catalog=_synthetic_original_sources(tmp_path,monkeypatch)
    result=review.review_sc1_original_lineid(hp,catalog)
    assert result["status"]=="LAB_ATLAS_REVIEW_REQUIRED"
    assert result["n_historic_anchors"]==19
    assert result["calibration_source_documented"]=="He-Ne-Ar"
    assert result["specific_2024_lamp_filling_confirmed_from_FITS"] is False
    assert result["auto_rewrite_hand_pairs"] is False
    assert result["wave2d_calibration_approved"] is False
    assert result["science_hold"] is True
    entries={item["historic_air_A"]:item for item in result["entries"]}
    assert entries[5037.75]["species_candidate"]=="Ne I"
    assert entries[5047.74]["species_candidate"]=="He I"
    assert entries[5080.38]["status"]=="INDEPENDENT_NEI_REFERENCE"
    assert entries[4863.96]["status"]=="LABORATORY_ID_UNKNOWN"
    assert entries[5221.80]["status"]=="LABORATORY_ID_UNKNOWN"
    assert entries[5341.60]["published_transition_air_A"] is None
    assert result["full_ASD_transition_export_completed"] is False


def test_actual_source_manifest_prevents_edited_or_wrong_night_lineid(
    tmp_path
):
    hp=tmp_path/"hand_pairs.txt"
    catalog=tmp_path/"lines_long(1).tab"
    hp.write_text("400 3888.65\n",encoding="utf-8")
    catalog.write_text("3888.65 3 0.4\n",encoding="utf-8")
    with pytest.raises(RuntimeError,match="Source SHA mismatch"):
        review.review_sc1_original_lineid(hp,catalog)


def test_changed_original_line_not_accepted_even_if_hashes_updated(
    tmp_path,monkeypatch
):
    hp,catalog=_synthetic_original_sources(tmp_path,monkeypatch)
    data=hp.read_text(encoding="utf-8").replace("5341.6000","5341.7000")
    hp.write_text(data,encoding="utf-8")
    monkeypatch.setattr(
        review,"FROZEN_HAND_PAIRS_SHA256",hashlib.sha256(hp.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError,match="reviewed 19"):
        review.review_sc1_original_lineid(hp,catalog)


def test_bad_line_list_membership_rejected_even_with_matching_hash(
    tmp_path,monkeypatch
):
    hp,catalog=_synthetic_original_sources(tmp_path,monkeypatch)
    data=catalog.read_text(encoding="utf-8").replace("5221.8000","5221.9000")
    catalog.write_text(data,encoding="utf-8")
    monkeypatch.setattr(
        review,"FROZEN_LINE_LIST_SHA256",
        hashlib.sha256(catalog.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError,match="absent"):
        review.review_sc1_original_lineid(hp,catalog)


def test_reject_nonphysical_blend_and_wavelengths():
    with pytest.raises(ValueError,match="ordered"):
        review.infer_positive_blend_centroid_ratio((5343.28,5341.09),5341.60)
    with pytest.raises(ValueError,match="positive FWHM"):
        review.gaussian_component_identifiability(
            5341.09,5343.28,fwhm_A=-1.
        )


def test_production_wave2d_lsf_stages_remain_unmodified():
    from scorpio_pipe.stages import lsf_model,wavesolution
    for stage in (lsf_model,wavesolution):
        text=Path(stage.__file__).read_text(encoding="utf-8")
        assert "lineid_atlas_review" not in text
    assert review._REFS[-1][-1]=="EFFECTIVE_CENTROID_REVIEW"
