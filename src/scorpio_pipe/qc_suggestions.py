"""QC-guided suggestions (P3 / BL-P3-UI-010).

The pipeline emits QC *flags* (minimal mandatory set in P1-G). This module maps
those flags to conservative "what to do next" suggestions.

Principles
----------
- No side effects: suggestions do NOT auto-run anything.
- Conservative: focus on safe verification steps and reversible changes.
- Stable: suggestions depend only on flag codes, not on file system state.

The UI can call :func:`recommend_actions` and display the returned list.
"""

from __future__ import annotations

from typing import Any, Iterable


# Flag-code -> list of recommended actions (ordered, short).
_RULES: dict[str, list[str]] = {
    # Sky ROI/windows issues
    "NO_SKY_WINDOWS": [
        "Review sky windows/ROI: ensure they cover real sky (not object/lines) and are wide enough.",
        "If the slit is filled by the object, mark it as slit-filled and use an appropriate mode (or provide separate sky frames).",
        "Re-run sky subtraction and inspect residuals in the sky windows.",
    ],
    # Flexure / wavelength solution uncertainty
    "FLEXURE_UNCERTAIN": [
        "Inspect line identification / wavelength solution quality (residuals, rejected lines).",
        "Consider disabling application of Δλ flexure correction for this run if the estimate is unstable.",
        "If available, use a better reference (stronger sky lines / arc / refined line list) and re-run.",
    ],
    # η(λ) / correlated noise diagnostics
    "ETA_ANOMALY": [
        "Check sky contamination in the sky windows (object light, gradients, scattered light).",
        "Inspect sigma_before_eta / sigma_after_eta: values far from 1 suggest a mismatch in VAR vs residuals.",
        "Consider adjusting the stacking strategy (rejection thresholds) or revisiting per-frame VAR / gain settings.",
    ],
    "ETA_BAD_CALIBRATION": [
        "Check sky contamination in the sky windows (object light, gradients, scattered light).",
        "Inspect sigma_before_eta / sigma_after_eta: values far from 1 suggest a mismatch in VAR vs residuals.",
        "Consider adjusting the stacking strategy (rejection thresholds) or revisiting per-frame VAR / gain settings.",
    ],
    # Optional P3 sky scaling
    "SKY_SCALE_NO_GAIN": [
        "Sky-frame scaling did not improve residuals. Consider disabling sky scaling for this run.",
        "If you keep sky scaling: refine sky windows or line groups used for scaling and re-run.",
    ],
}


def extract_flag_codes(flags: Iterable[Any] | None) -> list[str]:
    """Extract normalized flag codes from a list of flag dicts (or strings)."""

    out: list[str] = []
    if not flags:
        return out
    for it in flags:
        if isinstance(it, str):
            code = it
        elif isinstance(it, dict):
            code = it.get("code")
        else:
            code = None
        c = str(code or "").strip().upper()
        if c:
            out.append(c)
    return out


def recommend_actions(flag_codes: Iterable[str] | None) -> list[str]:
    """Return recommended actions for the provided QC flag codes."""

    if not flag_codes:
        return []

    seen: set[str] = set()
    actions: list[str] = []

    for raw in flag_codes:
        c = str(raw or "").strip().upper()
        if not c:
            continue
        for a in _RULES.get(c, []):
            if a not in seen:
                seen.add(a)
                actions.append(a)

    return actions
