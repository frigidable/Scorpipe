"""Read-only diagnostics of historic red Ne ARC blend and physical profiles.

NIST Ne I air wavelengths are distinct laboratory lines; the legacy
5341.60 Å in Moiseev lines_long is not assumed to be an isolated Ne I
transition. All outputs are engineering-only, never science calibration.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.optimize import least_squares

NIST_NEI_RED_AIR_A = (5330.7775, 5341.0938, 5343.2834)
ADDITIONAL_NEI_5326_A = 5326.3968  # Ne I, interferometry JOSA 40 (1950)
HISTORIC_RED_PAIR_A = (5330.78, 5341.60)
NIST_REFERENCE = "https://physics.nist.gov/PhysRefData/Handbook/Tables/neontable2.htm"


def red_lineid_review() -> dict[str, Any]:
    """Flag atlas/legacy discrepancy without changing any fitted Wave2D map."""
    return {
        "status": "REVIEW_REQUIRED",
        "historic_moiseev_line_list_air_A": list(HISTORIC_RED_PAIR_A),
        "nist_nei_nearby_air_A": list(NIST_NEI_RED_AIR_A),
        "additional_nei_5326_air_A": ADDITIONAL_NEI_5326_A,
        "closest_nist_nei_to_historic_5341_60_A": 5341.0938,
        "difference_historic_minus_nearest_lab_A": 5341.60-5341.0938,
        "another_nei_component_in_5341_blend_A": 5343.2834,
        "interpretation": (
            "5341.60 may be a blend effective centroid rather than an "
            "individual Ne I laboratory line; source list intent unresolved"
        ),
        "atlas_reference": NIST_REFERENCE,
        "science_lsf_validated": False,
        "wave2d_calibration_approved": False,
        "auto_replace_production_lsf": False,
    }


def fit_original_arc_multiplet(
    wavelength_air_A: np.ndarray,
    profile: np.ndarray,
    line_centers_air_A: tuple[float, ...],
    *,
    fit_range_air_A: tuple[float, float] = (5320.0, 5352.0),
) -> dict[str, Any]:
    """Common-width Gaussians, free strengths, common shift, linear continuum.

    Compare different component hypotheses ONLY using the same full window.
    Fitted shift is a nuisance parameter, not a science flexure correction.
    """
    w=np.asarray(wavelength_air_A,float)
    f=np.asarray(profile,float)
    lines=np.asarray(line_centers_air_A,float)
    if (w.ndim!=1 or w.shape!=f.shape or w.size<80
        or not np.isfinite(w).all() or not np.isfinite(f).all()
        or not np.all(np.diff(w)>0)):
        raise ValueError("Multiplet requires finite 1D increasing air wavelength map")
    if (lines.ndim!=1 or not 2<=lines.size<=6
        or not np.isfinite(lines).all() or np.any(np.diff(lines)<=0)
        or lines[0]<=w[0] or lines[-1]>=w[-1]):
        raise ValueError("Two to six distinct ordered on-detector lines required")
    lo_A,hi_A=map(float,fit_range_air_A)
    if (not np.isfinite([lo_A,hi_A]).all() or not lo_A<hi_A
        or lo_A>=lines[0] or hi_A<=lines[-1]):
        raise ValueError("Fit interval must span every line and include sidebands")
    lo=int(np.searchsorted(w,lo_A))
    hi=int(np.searchsorted(w,hi_A))+1
    if lo<4 or hi>w.size-4 or hi-lo<25:
        return {"status":"REJECT","reason":"insufficient_detector_sidebands"}
    x=np.arange(lo,hi,dtype=float)
    y=f[lo:hi]
    centers=np.interp(lines,w,np.arange(w.size,dtype=float))
    t=x-float(np.mean(x))
    bg=float(np.median(np.r_[y[:5],y[-5:]]))
    d2=np.diff(y,n=2)
    rough=max(2.0,float(1.4826*np.median(abs(d2-np.median(d2)))/np.sqrt(6)))
    amps=[max(5.0,float(f[int(round(cx))]-bg)) for cx in centers]

    def model(q: np.ndarray) -> np.ndarray:
        strength=q[:len(lines)]
        delta,sigma,offset,slope=q[len(lines):]
        result=offset+slope*t
        for amp,c in zip(strength,centers):
            result=result+amp*np.exp(-0.5*((x-c-delta)/sigma)**2)
        return result

    try:
        fitted=least_squares(
            lambda q: model(q)-y,
            [*amps,0.,2.8,bg,0.],
            bounds=([0.]*len(lines)+[-3.5,1.,-np.inf,-np.inf],
                    [np.inf]*len(lines)+[3.5,5.5,np.inf,np.inf]),
            loss="soft_l1",f_scale=rough,max_nfev=300,
        )
    except (ValueError,RuntimeError,FloatingPointError):
        return {"status":"REJECT","reason":"optimizer_exception"}
    sigma=float(fitted.x[len(lines)+1])
    if (not fitted.success or not np.isfinite(fitted.x).all()
        or sigma<=1.02 or sigma>=5.45):
        return {"status":"REJECT","reason":"optimizer_failed_or_width_bounded"}
    residue=y-model(fitted.x)
    strength=np.asarray(fitted.x[:len(lines)],float)
    if max(strength)<=0:
        return {"status":"REJECT","reason":"no_detected_line"}
    dispersion=float(np.median(np.diff(w[lo:hi])))
    return {
        "status":"DIAGNOSTIC_ONLY",
        "line_centers_air_A":list(map(float,lines)),
        "fwhm_gaussian_equivalent_A":float(2.354820045*sigma*dispersion),
        "shared_shift_px":float(fitted.x[len(lines)]),
        "amplitudes":list(map(float,strength)),
        "normalized_rmse":float(np.sqrt(np.mean(residue**2))/max(strength)),
        "residual_wavelength_air_A":list(map(float,w[lo:hi])),
        "residual_normalized":list(map(float,residue/max(strength))),
        "n_fit_pixels":int(hi-lo),
        "continuum_degree":1,
        "uncertainty_status":"MODEL_COMPARISON_ONLY_CORRELATED_PIXELS",
        "science_lsf_validated":False,
        "wave2d_calibration_approved":False,
        "auto_replace_production_lsf":False,
    }


def trace_registered_median(
    arc_2d: np.ndarray,
    wavelength_map_air_A: np.ndarray,
    center_y: int,
    *,
    half_strip: int = 10,
) -> np.ndarray:
    """Diagnostic only; resampling itself modifies LSF, no science adoption."""
    arc=np.asarray(arc_2d,float)
    wave=np.asarray(wavelength_map_air_A,float)
    if (arc.ndim!=2 or arc.shape!=wave.shape
        or not np.isfinite(arc).all() or not np.isfinite(wave).all()):
        raise ValueError("Finite same-shape original ARC and wavelength map required")
    if (not isinstance(center_y,int) or not isinstance(half_strip,int)
        or half_strip<0 or center_y-half_strip<0
        or center_y+half_strip>=arc.shape[0]):
        raise ValueError("Invalid detector spatial strip")
    rows=range(center_y-half_strip,center_y+half_strip+1)
    if any(not np.all(np.diff(wave[y])>0) for y in rows):
        raise ValueError("Wavelengths must strictly increase along each ARC row")
    x=np.arange(arc.shape[1],dtype=float)
    target=wave[center_y]
    profiles=[]
    for y in rows:
        source_x=np.interp(target,wave[y],x,left=np.nan,right=np.nan)
        finite=np.isfinite(source_x)
        profile=np.full(target.shape,np.nan)
        profile[finite]=np.interp(source_x[finite],x,arc[y])
        profiles.append(profile)
    stack=np.asarray(profiles)
    supported=np.any(np.isfinite(stack),axis=0)
    output=np.full(stack.shape[1],np.nan)
    if np.any(supported):
        output[supported]=np.nanmedian(stack[:,supported],axis=0)
    return output


def residual_agreement(first: dict, second: dict) -> dict[str,Any]:
    """Raw-ARC residual-shape comparison, not a p-value or line detection."""
    if any(item.get("status")!="DIAGNOSTIC_ONLY" for item in (first,second)):
        raise ValueError("Two successful original ARC profile fits required")
    a=np.asarray(first["residual_wavelength_air_A"],float)
    b=np.asarray(second["residual_wavelength_air_A"],float)
    r=np.asarray(first["residual_normalized"],float)
    s=np.asarray(second["residual_normalized"],float)
    if (a.ndim!=1 or b.ndim!=1 or a.size<20 or b.size<20
        or a.shape!=r.shape or b.shape!=s.shape
        or not np.isfinite([*a,*b,*r,*s]).all()
        or np.any(np.diff(a)<=0) or np.any(np.diff(b)<=0)):
        raise ValueError("Invalid independent original ARC wavelength residuals")
    lo=max(a[0],b[0]);hi=min(a[-1],b[-1])
    grid=a[(a>=lo)&(a<=hi)]
    if grid.size<20:return {"status":"REJECT","reason":"insufficient_overlap"}
    ra=np.interp(grid,a,r);rb=np.interp(grid,b,s)
    if np.std(ra)<=1e-12 or np.std(rb)<=1e-12:
        return {"status":"REJECT","reason":"flat_residual"}
    return {
        "status":"DIAGNOSTIC_ONLY",
        "pearson_correlation":float(np.corrcoef(ra,rb)[0,1]),
        "n_overlap_wavelength_pixels":int(grid.size),
        "median_abs_residual_fraction":[float(np.median(abs(ra))),float(np.median(abs(rb)))],
        "sign_agreement_fraction":float(np.mean(np.sign(ra)==np.sign(rb))),
        "interpretation":"Shared model/blend artifacts are not independent significance",
        "science_lsf_validated":False,
        "wave2d_calibration_approved":False,
        "auto_replace_production_lsf":False,
    }
