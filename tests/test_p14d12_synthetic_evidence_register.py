"""Synthetic-evidence register: real diagnostic reports and adversarial inputs."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scorpio_pipe.calib.sky_evidence_gate import assess_synthetic_sky_evidence


@pytest.fixture(scope="module")
def complete_reports(tmp_path_factory):
    # Each item comes from a DIFFERENT controlled synthetic science
    # scenario. No claim of common instrument lineage is made here.
    from scorpio_pipe.calib.sky_fit_audit import audit_frozen_kelson_products
    from scorpio_pipe.calib.sky_lsqr_derivative import assess_kelson_lsqr_directional_derivatives
    from scorpio_pipe.calib.sky_wls_conditioning import audit_frozen_wls_numerics
    from scorpio_pipe.calib.sky_mc_coverage import synthetic_grouped_sky_coverage
    from scorpio_pipe.calib.sky_mc_reclipping_coverage import synthetic_reclipped_sky_coverage
    from scorpio_pipe.calib.sky_conditioning_coverage_sweep import synthetic_wls_conditioning_coverage_sweep
    from tests.test_kelson_lsqr_derivative import _actual_fit_paths
    from tests.test_p14d7_kelson_clipping_sensitivity import _synthetic_sky, _call
    from tests.test_p14d10_frozen_sky_svd_conditioning import _well_conditioned_op

    f, m, s = _actual_fit_paths(tmp_path_factory.mktemp("evidence_synthetic_fits"))
    clipped_args = _synthetic_sky()
    adversarial_args = tuple(a.copy() for a in clipped_args)
    adversarial_args[0][3, 34] += 15.
    probe = np.zeros((1, *adversarial_args[0].shape))
    probe[0, 3, 34] = -1.

    return {
        "fidelity": audit_frozen_kelson_products(f, m, s),
        "lsqr": assess_kelson_lsqr_directional_derivatives(
            f, m, s, seed=20261104, n_probes=3,
            epsilon_sigma=.15, relative_tolerance=.05,
            replay_tolerance=.01,
        ),
        "svd": audit_frozen_wls_numerics(
            _well_conditioned_op(), n_probes=3,
        ),
        "frozen_mc": synthetic_grouped_sky_coverage(
            n_realizations=8000, seed=1881, group_ids=("A", "A"),
        ),
        "reclipped_mc": synthetic_reclipped_sky_coverage(
            n_realizations=220, seed=419, group_ids=("A", "A"),
        ),
        "stable_clipping": _call(*clipped_args),
        "changed_clipping": _call(
            *adversarial_args, n_probes=1,
            perturbation_sigma=15., probe_vectors=probe,
        ),
        "changed_reclipping": synthetic_reclipped_sky_coverage(
            n_realizations=300, seed=1904, group_ids=("A", "A"),
            clip_sigma=3.2, contamination_sigma=4.1,
        ),
        "condition_sweep": synthetic_wls_conditioning_coverage_sweep(
            n_realizations=3000, seed=113,
        ),
    }


def test_real_synthetic_suite_reports_can_be_registered_without_releasing_var(complete_reports):
    result = assess_synthetic_sky_evidence(complete_reports)
    assert result["status"] == "SYNTHETIC_CONTRACTS_SUPPORTED", result
    assert result["n_supported"] == result["n_required"] == 9
    assert result["all_synthetic_contracts_supported"] is True
    assert result["scope"] == "CROSS_SCENARIO_SYNTHETIC_REGRESSION_ONLY"
    assert result["observation_specific_qualification"] == "NOT_ASSESSED"
    assert result["science_uncertainty_qualified"] is False
    assert result["production_variance_release_allowed"] is False
    assert result["override_science_status_supported"] is False
    assert all(v["status"] == "SUPPORTED" for v in result["scenarios"].values())


def test_missing_scenarios_can_never_count_as_complete(complete_reports):
    result = assess_synthetic_sky_evidence(
        {"fidelity": complete_reports["fidelity"]}
    )
    assert result["status"] == "EVIDENCE_INCOMPLETE"
    assert result["n_supported"] == 1
    assert result["science_uncertainty_qualified"] is False
    assert result["production_variance_release_allowed"] is False


@pytest.mark.parametrize("bad", [
    "fake_good_mc_metric",
    "fake_good_lsqr_stop",
    "fake_good_svd_rank",
    "fake_nominal_reclipping",
    "fake_no_adversarial_selection",
    "fake_stable_conditioning",
    "invalid_science_qualification",
])
def test_report_claims_do_not_overrule_quantitative_or_provenance_checks(
    complete_reports, bad,
):
    reports = deepcopy(complete_reports)
    if bad == "fake_good_mc_metric":
        reports["frozen_mc"]["empirical_covariance"][0][0] *= 3.
    elif bad == "fake_good_lsqr_stop":
        reports["lsqr"]["all_lsqr_stops_nominal"] = False
    elif bad == "fake_good_svd_rank":
        reports["svd"]["effective_rank"] = 0
    elif bad == "fake_nominal_reclipping":
        reports["reclipped_mc"]["selection_change_fraction_any_exposure"] = .25
    elif bad == "fake_no_adversarial_selection":
        reports["changed_clipping"]["status"] = "FROZEN_STABLE"
    elif bad == "fake_stable_conditioning":
        reports["condition_sweep"]["trials"] = [
            x for x in reports["condition_sweep"]["trials"]
            if x["status"] == "NUMERIC_SAFE"
        ]
    elif bad == "invalid_science_qualification":
        reports["fidelity"]["science_uncertainty_qualified"] = True

    result = assess_synthetic_sky_evidence(reports)
    assert result["status"] == "EVIDENCE_INVALID", (bad, result)
    assert result["n_supported"] < 9
    assert result["science_uncertainty_qualified"] is False
    assert result["production_variance_release_allowed"] is False


def test_malformed_register_is_fail_closed_and_no_unrecognized_qa_track():
    with pytest.raises(ValueError, match="mapping"):
        assess_synthetic_sky_evidence([1, 2])
    with pytest.raises(ValueError, match="Unknown"):
        assess_synthetic_sky_evidence({"observational": {"status": "PASS"}})
    invalid = assess_synthetic_sky_evidence({"svd": [1, 2, 3]})
    assert invalid["status"] == "EVIDENCE_INVALID"
    assert invalid["scenarios"]["svd"]["status"] == "INVALID"
    assert invalid["production_variance_release_allowed"] is False
