"""Conservative two-ARC, feature-blocked diagnostic LSF validation.

Input measurements must come from separately inspected original ARC files.
A feature is one isolated laboratory line or a jointly fitted laboratory pair.
Both ARC repeats and all its spatial bins are held out together during LOFO.
No fitted coefficient is permission to alter science LSF products.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import re
import numpy as np


def _basis(wavelength: np.ndarray, y: np.ndarray) -> np.ndarray:
    u = (wavelength - 4600.0) / 1000.0
    v = (y - 515.5) / 515.5
    return np.column_stack([np.ones(u.size), u, v, v*v, u*v])


def _fit(wave: np.ndarray, y: np.ndarray, fwhm: np.ndarray,
         feature: np.ndarray, *, keep: np.ndarray | None = None) -> np.ndarray:
    if keep is None:
        keep = np.ones(fwhm.size, bool)
    ids, counts = np.unique(feature[keep], return_counts=True)
    if ids.size < 5:
        raise ValueError("At least five independent features required")
    if np.ptp(wave[keep]) < 200.0 or np.ptp(y[keep]) < 100.0:
        raise ValueError("Insufficient wavelength or slit spatial span")
    x = _basis(wave[keep], y[keep])
    if np.linalg.matrix_rank(x, tol=1e-9) != x.shape[1]:
        raise ValueError("Rank deficient LSF candidate basis")
    target = fwhm[keep]
    selected_features = feature[keep]
    by_id = dict(zip(ids, counts))
    base_w = np.asarray([1.0/by_id[g] for g in selected_features])
    weights = base_w.copy()
    beta = np.linalg.lstsq(
        x*np.sqrt(weights)[:,None], target*np.sqrt(weights), rcond=None
    )[0]
    for _ in range(20):
        residual = target-x@beta
        rms_by_group = np.array([
            np.sqrt(np.median(residual[selected_features == g]**2)) for g in ids
        ])
        scale = max(float(
            1.4826*np.median(np.abs(rms_by_group-np.median(rms_by_group)))
        ), 0.15)
        huber = np.minimum(1.0,1.5*scale/np.maximum(np.abs(residual),1e-12))
        weights = base_w*huber
        new_beta = np.linalg.lstsq(
            x*np.sqrt(weights)[:,None],target*np.sqrt(weights),rcond=None
        )[0]
        if float(np.max(np.abs(beta-new_beta))) < 1e-8:
            beta = new_beta
            break
        beta = new_beta
    return beta


def validate_lsf_multiline(
    records: Sequence[Mapping[str,Any]], *,
    arc_ids: tuple[str,str],
    source_arc_sha256: Mapping[str,str],
    wavelength_map_sha256: str,
    manual_lineid_sha256: str,
    min_matched_bins: int = 12,
    max_arc_median_difference_A: float = 0.20,
    max_arc_p90_difference_A: float = 0.40,
    max_lolo_median_abs_A: float = 0.50,
    max_lolo_p90_abs_A: float = 0.95,
) -> dict[str,Any]:
    """Two independent lamp repeats and leave-one-feature-out validation.

    The thresholds are engineering screening gates, NOT statistically
    calibrated scientific uncertainties. SHA values are mandatory declared
    provenance; the caller must first hash the actual source FITS bytes.
    """
    if len(arc_ids)!=2 or len(set(arc_ids))!=2 or any(not str(x).strip() for x in arc_ids):
        raise ValueError("Two different named original ARC exposures required")
    def valid_hash(value: object) -> bool:
        return isinstance(value,str) and re.fullmatch("[0-9a-fA-F]{64}",value) is not None
    if (set(source_arc_sha256)!=set(arc_ids)
        or any(not valid_hash(source_arc_sha256[a]) for a in arc_ids)
        or source_arc_sha256[arc_ids[0]].lower()==source_arc_sha256[arc_ids[1]].lower()
        or not valid_hash(wavelength_map_sha256)
        or not valid_hash(manual_lineid_sha256)):
        raise ValueError(
            "Two distinct original ARC content hashes and frozen map/LineID digests required"
        )
    for name,val in [
        ("min_matched_bins",min_matched_bins),
        ("max_arc_median_difference_A",max_arc_median_difference_A),
        ("max_arc_p90_difference_A",max_arc_p90_difference_A),
        ("max_lolo_median_abs_A",max_lolo_median_abs_A),
        ("max_lolo_p90_abs_A",max_lolo_p90_abs_A),
    ]:
        if not np.isfinite(val) or val<=0:
            raise ValueError("Invalid QC threshold: "+name)
    if not records:
        raise ValueError("Missing ARC measurements")
    groups:dict[str,dict[str,Any]]={}
    observed=set()
    used_members:dict[float,str]={}
    for n,rec in enumerate(records):
        try:
            group=str(rec["feature_id"]).strip()
            arc=str(rec["arc_id"]).strip()
            kind=str(rec["fit_kind"]).strip()
            wave=float(rec["wavelength_A"])
            y=float(rec["y_pix"])
            width=float(rec["fwhm_A"])
            members=tuple(sorted(float(x) for x in rec["lab_lines_A"]))
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError(f"Malformed measurement {n}") from exc
        if (not group or arc not in arc_ids or kind not in {"single","pair"}
                or len(members)!=(1 if kind=="single" else 2)
                or not np.isfinite(members).all() or any(x<=0 for x in members)
                or len(set(members))!=len(members)
                or abs(wave-float(np.mean(members)))>0.05
                or not np.isfinite([wave,y,width]).all() or wave<=0
                or not 0<=y<=10000 or not 0.25<=width<=80):
            raise ValueError(f"Invalid ARC line measurement {n}")
        key=(group,arc,y)
        if key in observed:
            raise ValueError("Duplicate feature+ARC+spatial sample")
        observed.add(key)
        for member in members:
            old=used_members.setdefault(member,group)
            if old!=group:
                raise ValueError("Laboratory line member double-counted in two features")
        ref=groups.setdefault(group,{
            "wave":wave,"kind":kind,"members":members,
            "arc":{a:{} for a in arc_ids}
        })
        if (ref["wave"]!=wave or ref["kind"]!=kind
                or ref["members"]!=members):
            raise ValueError("A feature cannot change laboratory centroid or family")
        ref["arc"][arc][y]=width

    gdata={}
    screening=[]
    for group,raw in sorted(groups.items(),key=lambda pair:pair[1]["wave"]):
        a,b=(raw["arc"][arc] for arc in arc_ids)
        yy=np.array(sorted(set(a).intersection(b)),float)
        values_a=np.array([a[v] for v in yy],float)
        values_b=np.array([b[v] for v in yy],float)
        delta=values_b-values_a
        enough=yy.size>=min_matched_bins
        med_diff=float(abs(np.median(delta))) if yy.size else None
        p90_diff=float(np.percentile(np.abs(delta),90)) if yy.size else None
        consistent=bool(
            enough and med_diff<=max_arc_median_difference_A
            and p90_diff<=max_arc_p90_difference_A
        )
        if consistent:
            # Repeats are averaged; they are NOT independent line features.
            gdata[group]=(yy,0.5*(values_a+values_b),raw["wave"])
        screening.append({
            "feature_id":group,"kind":raw["kind"],
            "laboratory_air_wavelengths_A":list(raw["members"]),
            "wavelength_A":float(raw["wave"]),
            "n_paired_bins":int(yy.size),
            "arc_median_abs_difference_A":med_diff,
            "arc_p90_abs_difference_A":p90_diff,
            "two_arc_consistent":consistent,
        })
    if len(gdata)<6:
        return {
            "status":"REJECT","reason":"insufficient_repeatable_features",
            "features":screening,"science_lsf_validated":False,
            "auto_replace_production_lsf":False,
        }
    fids=list(gdata)
    lam=np.concatenate([
        np.full(gdata[g][0].size,gdata[g][2]) for g in fids
    ])
    ys=np.concatenate([gdata[g][0] for g in fids])
    fs=np.concatenate([gdata[g][1] for g in fids])
    fid=np.concatenate([np.full(gdata[g][0].size,g) for g in fids])
    for item in screening:
        name=item["feature_id"]
        if name not in gdata:
            item.update({
                "lolo_median_abs_A":None,
                "lolo_p90_abs_A":None,
                "heldout_model_consistent":False,
                "accepted_for_candidate":False,
            })
            continue
        held=fid==name
        try:
            beta=_fit(lam,ys,fs,fid,keep=~held)
            err=fs[held]-_basis(lam[held],ys[held])@beta
            med_abs=float(np.median(np.abs(err)))
            p90_abs=float(np.percentile(np.abs(err),90))
            good=bool(
                med_abs<=max_lolo_median_abs_A
                and p90_abs<=max_lolo_p90_abs_A
            )
        except ValueError:
            med_abs=p90_abs=None
            good=False
        item.update({
            "lolo_median_abs_A":med_abs,
            "lolo_p90_abs_A":p90_abs,
            "heldout_model_consistent":good,
            "accepted_for_candidate":good,
        })
    accepted={
        row["feature_id"] for row in screening if row["accepted_for_candidate"]
    }
    if len(accepted)<5:
        return {
            "status":"REJECT","reason":"insufficient_crossvalidated_features",
            "features":screening,"science_lsf_validated":False,
            "auto_replace_production_lsf":False,
        }
    try:
        beta=_fit(lam,ys,fs,fid,keep=np.isin(fid,list(accepted)))
    except ValueError:
        return {
            "status":"REJECT","reason":"insufficient_model_rank_after_qc",
            "features":screening,"science_lsf_validated":False,
            "auto_replace_production_lsf":False,
        }
    lam_ok=np.sort(np.array([gdata[g][2] for g in accepted]))
    y_min=max(float(min(gdata[g][0])) for g in accepted)
    y_max=min(float(max(gdata[g][0])) for g in accepted)
    if y_max-y_min < 100.0:
        return {
            "status":"REJECT",
            "reason":"insufficient_common_spatial_coverage",
            "features":screening,
            "science_lsf_validated":False,
            "auto_replace_production_lsf":False,
        }
    accepted_cv=[row for row in screening if row["accepted_for_candidate"]]
    equal_feature_lolo_rms=float(np.sqrt(np.mean([
        row["lolo_median_abs_A"]**2 for row in accepted_cv
    ])))
    return {
        "status":"DIAGNOSTIC_CANDIDATE",
        "science_lsf_validated":False,
        "auto_replace_production_lsf":False,
        "arc_ids":list(arc_ids),
        "source_arc_sha256":dict(source_arc_sha256),
        "wavelength_map_sha256":wavelength_map_sha256,
        "manual_lineid_sha256":manual_lineid_sha256,
        "feature_blocked_lolo_rms_of_median_abs_A":equal_feature_lolo_rms,
        "n_independent_features":len(groups),
        "n_repeatable_features":len(gdata),
        "n_cv_accepted_features":len(accepted),
        "accepted_feature_ids":sorted(accepted),
        "wavelength_support_A":[float(lam_ok[0]),float(lam_ok[-1])],
        "common_spatial_support_pix":[y_min,y_max] if y_min<=y_max else None,
        "largest_internal_gap_A":float(np.max(np.diff(lam_ok))),
        "model":"robust_lambda_linear_y_quadratic_lambda_y_interaction",
        "coefficients":list(map(float,beta)),
        "features":screening,
        "uncertainty_status":"NOT_QUANTIFIED_FOR_SCIENCE",
        "interpretation":(
            "LOFO grouped both ARC repeats and all y bins by laboratory feature; "
            "pass is engineering-only; no LSF transfer to science objects"
        ),
    }


def predict_lsf_candidate(
    lambda_map_A: np.ndarray,
    y_pix: np.ndarray,
    report: Mapping[str,Any],
) -> tuple[np.ndarray,np.ndarray]:
    """Inspection-only candidate FWHM: unbracketed regions remain NaN."""
    if report.get("status")!="DIAGNOSTIC_CANDIDATE":
        raise ValueError("Candidate requires passing independent feature audit")
    wave=np.asarray(lambda_map_A,float)
    y=np.asarray(y_pix,float)
    if wave.shape!=y.shape or wave.ndim!=2:
        raise ValueError(
            "Wavelength map and y coordinate grids must be matching 2D"
        )
    lo,hi=report["wavelength_support_A"]
    spatial=report.get("common_spatial_support_pix")
    if spatial is None:
        raise ValueError("No common spatial support")
    good=(
        np.isfinite(wave)&np.isfinite(y)
        &(wave>=lo)&(wave<=hi)
        &(y>=spatial[0])&(y<=spatial[1])
    )
    values=np.full(wave.shape,np.nan)
    if np.any(good):
        values[good]=_basis(wave[good],y[good])@np.asarray(
            report["coefficients"],float
        )
    good &= np.isfinite(values)&(values>0)
    values[~good]=np.nan
    return values,good
