from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.benchmark.registry_tools import DEFAULT_CASE_METRICS, upsert_registry_entry
from scorpio_pipe.benchmark.long_params import discover_long_setups
from scorpio_pipe.calib.common import save_json
from scorpio_pipe.nightlog import parse_nightlog


def parse_header_text(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    out: dict[str, Any] = {"_path": str(p)}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, rest = line.split("=", 1)
        key = key.strip()
        val = rest.split("/", 1)[0].strip()
        if not key:
            continue
        if val.startswith("'") and "'" in val[1:]:
            out[key] = val.strip().strip("'").strip()
            continue
        try:
            if re.match(r"^[-+]?\d+$", val):
                out[key] = int(val)
            else:
                out[key] = float(val)
        except Exception:
            out[key] = val.strip()
    return out


def parse_bta_nightlog_table(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s.startswith("|s"):
            continue
        parts = [x.strip() for x in s.split("|")]
        if len(parts) < 18:
            continue
        row = {
            "fid": parts[1].lower(),
            "object": parts[2],
            "kind": parts[3].lower(),
            "start": parts[4],
            "exptime": _to_float(parts[5]),
            "mode": parts[6],
            "disperser": parts[7],
            "slit": _to_float(parts[8]),
            "filters": parts[10] if len(parts) > 10 else "",
            "seeing": _to_float(parts[14]) if len(parts) > 14 else None,
            "zenith_distance_deg": _to_float(parts[15]) if len(parts) > 15 else None,
            "azimuth_deg": _to_float(parts[16]) if len(parts) > 16 else None,
            "p2_deg": _to_float(parts[17]) if len(parts) > 17 else None,
            "ptb_deg": _to_float(parts[18]) if len(parts) > 18 else None,
            "comment": parts[19] if len(parts) > 19 else "",
        }
        rows.append(row)
    return rows


def _to_float(x: Any) -> float | None:
    try:
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _looks_like_science_object(name: str) -> bool:
    s = str(name or '').strip().lower()
    if not s:
        return False
    return s not in {'sunsky', 'sun_sky', 'sun-sky', 'sky', 'bias', 'flat', 'neon'}


def _group_spectral_sequences(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("mode") or "").strip().lower() != "spectra":
            continue
        obj = str(row.get("object") or "").strip()
        disp = str(row.get("disperser") or "").strip()
        slit = row.get("slit")
        key = (obj, disp, f"{float(slit):.2f}" if slit is not None else "")
        groups[key].append(dict(row))

    out: list[dict[str, Any]] = []
    for (obj, disp, slit), items in groups.items():
        kinds = defaultdict(int)
        obj_exptimes: list[float] = []
        seeing: list[float] = []
        p2: list[float] = []
        z: list[float] = []
        files: list[str] = []
        for row in items:
            kinds[str(row.get("kind") or "").lower()] += 1
            if str(row.get("kind") or "").lower() == "obj" and row.get("exptime") is not None:
                obj_exptimes.append(float(row["exptime"]))
            if row.get("seeing") is not None:
                seeing.append(float(row["seeing"]))
            if row.get("p2_deg") is not None:
                p2.append(float(row["p2_deg"]))
            if row.get("zenith_distance_deg") is not None:
                z.append(float(row["zenith_distance_deg"]))
            files.append(str(row.get("fid") or ""))
        out.append(
            {
                "object": obj,
                "disperser": disp,
                "slit": _to_float(slit),
                "n_rows": int(len(items)),
                "counts": {k: int(v) for k, v in sorted(kinds.items())},
                "obj_exptime_total_s": float(sum(obj_exptimes)) if obj_exptimes else 0.0,
                "obj_exptime_median_s": _median(obj_exptimes),
                "seeing_median": _median(seeing),
                "p2_span_deg": _span(p2),
                "zenith_span_deg": _span(z),
                "files": files,
            }
        )
    out.sort(key=lambda r: (-int((r.get("counts") or {}).get("obj", 0)), str(r.get("object") or "")))
    return out


def _median(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    if n % 2:
        return float(vals[n // 2])
    return float((vals[n // 2 - 1] + vals[n // 2]) / 2.0)


def _span(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return float(max(vals) - min(vals))


def _case_payload_from_candidate(candidate: dict[str, Any], *, slot: str) -> dict[str, Any]:
    return {
        "case_id": slot,
        "title": str(candidate.get("title") or slot.replace("_", " ").title()),
        "category": str(candidate.get("category") or "reference"),
        "source_type": "bootstrap_from_nightlog",
        "description": str(candidate.get("description") or ""),
        "observer_notes": str(candidate.get("observer_notes") or ""),
        "scorpipe_bundle": None,
        "long_bundle": None,
        "metrics": json.loads(json.dumps(DEFAULT_CASE_METRICS)),
        "bootstrap_evidence": dict(candidate.get("evidence") or {}),
    }


def suggest_registry_candidates(rows: list[dict[str, Any]], *, header_texts: list[dict[str, Any]] | None = None, long_setups: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    groups = _group_spectral_sequences(rows)
    suggestions: list[dict[str, Any]] = []

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in groups:
        by_object[str(g.get("object") or "")].append(g)

    # reference candidates
    refs = [g for g in groups if _looks_like_science_object(str(g.get("object") or "")) and int((g.get("counts") or {}).get("obj", 0)) >= 2 and int((g.get("counts") or {}).get("neon", 0)) >= 1 and int((g.get("counts") or {}).get("flat", 0)) >= 1]
    refs.sort(key=lambda g: (-(g.get("obj_exptime_total_s") or 0.0), abs((g.get("slit") or 1.0) - 1.0)))
    if refs:
        suggestions.append(
            {
                "slot": "reference_candidate",
                "title": f"Reference candidate: {refs[0].get('object')} {refs[0].get('disperser')}",
                "category": "reference",
                "confidence": "high",
                "description": "Spectral sequence with object, neon and flat frames present in the same setup.",
                "observer_notes": "Bootstrap suggestion from night log only; attach real Scorpipe/LONG bundles before certification.",
                "evidence": dict(refs[0]),
            }
        )

    # repeatability
    rep = [g for g in groups if _looks_like_science_object(str(g.get("object") or "")) and int((g.get("counts") or {}).get("obj", 0)) >= 3]
    rep.sort(key=lambda g: (-(g.get("counts") or {}).get("obj", 0), -(g.get("obj_exptime_total_s") or 0.0)))
    if rep:
        suggestions.append(
            {
                "slot": "repeatability_multi_exposure",
                "title": f"Repeatability: {rep[0].get('object')} {rep[0].get('disperser')}",
                "category": "repeatability",
                "confidence": "high",
                "description": "Multiple science exposures exist in the same spectral setup.",
                "observer_notes": "Good candidate for Scorpipe vs LONG repeatability regression checks.",
                "evidence": dict(rep[0]),
            }
        )

    # flexure-sensitive
    flex = [g for g in groups if _looks_like_science_object(str(g.get("object") or "")) and (g.get("p2_span_deg") or 0.0) >= 3.0 and int((g.get("counts") or {}).get("obj", 0)) >= 2]
    flex.sort(key=lambda g: (-(g.get("p2_span_deg") or 0.0), -(g.get("obj_exptime_total_s") or 0.0)))
    if flex:
        suggestions.append(
            {
                "slot": "flexure_sensitive_setup",
                "title": f"Flexure-sensitive setup: {flex[0].get('object')} {flex[0].get('disperser')}",
                "category": "sky",
                "confidence": "medium",
                "description": "The same spectral setup spans noticeable P2 changes across repeated exposures.",
                "observer_notes": "Use for flexure / skyline residual stress testing once bundles are exported.",
                "evidence": dict(flex[0]),
            }
        )

    # mixed slit metadata
    for obj, obj_groups in by_object.items():
        slits = sorted({f"{float(g.get('slit')):.2f}" for g in obj_groups if g.get("slit") is not None})
        if len(slits) > 1:
            suggestions.append(
                {
                    "slot": "mixed_slit_metadata",
                    "title": f"Mixed slit metadata: {obj}",
                    "category": "calibration",
                    "confidence": "medium",
                    "description": "The same object appears in spectral setups with multiple slit widths in the same night log.",
                    "observer_notes": "Good slit-sanity / calibration-association stress case candidate.",
                    "evidence": {
                        "object": obj,
                        "slits": slits,
                        "groups": obj_groups,
                    },
                }
            )
            break

    # skyline-rich / red setups
    if long_setups and isinstance(long_setups.get("setups"), list):
        setup_map = {str(r.get("setup") or ""): r for r in long_setups.get("setups") or []}
    else:
        setup_map = {}
    redish: list[dict[str, Any]] = []
    for g in groups:
        setup = setup_map.get(str(g.get("disperser") or ""), {})
        lam_max = setup.get("lambda_max_A")
        if (lam_max is not None and float(lam_max) >= 7000.0) or any(tok in str(g.get("disperser") or "") for tok in ["735", "860", "940"]):
            redish.append(g)
    redish.sort(key=lambda g: (-(g.get("obj_exptime_total_s") or 0.0), -(g.get("slit") or 0.0)))
    if redish:
        suggestions.append(
            {
                "slot": "skyline_rich_residuals",
                "title": f"Skyline-rich candidate: {redish[0].get('object')} {redish[0].get('disperser')}",
                "category": "sky",
                "confidence": "medium",
                "description": "Setup extends into redder wavelengths where skyline residual control is usually more demanding.",
                "observer_notes": "Bootstrap heuristic from LONG setup coverage and night log.",
                "evidence": dict(redish[0]),
            }
        )

    # extended object sky stress
    ext = [g for g in groups if _looks_like_science_object(str(g.get("object") or "")) and int((g.get("counts") or {}).get("obj", 0)) >= 2 and (g.get("slit") or 0.0) >= 1.0 and re.search(r"^(ngc|ugc|m33|ic)", str(g.get("object") or ""), re.IGNORECASE)]
    ext.sort(key=lambda g: (-(g.get("slit") or 0.0), -(g.get("obj_exptime_total_s") or 0.0)))
    if ext:
        suggestions.append(
            {
                "slot": "extended_object_sky_stress",
                "title": f"Extended-object sky stress: {ext[0].get('object')} {ext[0].get('disperser')}",
                "category": "sky",
                "confidence": "medium",
                "description": "Large/extended target with repeated spectral exposures and a relatively wide slit.",
                "observer_notes": "Useful for sky-window placement and object-protection validation.",
                "evidence": dict(ext[0]),
            }
        )

    # weak-object candidate heuristic (wide slit, many exposures, modest seeing often means faint object work)
    weak = [g for g in groups if _looks_like_science_object(str(g.get("object") or "")) and int((g.get("counts") or {}).get("obj", 0)) >= 4 and (g.get("obj_exptime_median_s") or 0.0) >= 300.0]
    weak.sort(key=lambda g: (-(g.get("counts") or {}).get("obj", 0), -(g.get("obj_exptime_total_s") or 0.0)))
    if weak:
        suggestions.append(
            {
                "slot": "weak_object_extraction",
                "title": f"Weak-object extraction candidate: {weak[0].get('object')} {weak[0].get('disperser')}",
                "category": "extraction",
                "confidence": "low",
                "description": "Repeated long spectral exposures make this a plausible extraction-stress case.",
                "observer_notes": "Night-log-only guess; verify from reduced spectra before counting this as a real benchmark case.",
                "evidence": dict(weak[0]),
            }
        )

    if header_texts:
        for hdr in header_texts:
            fname = str(hdr.get("FILE") or "")
            obj = str(hdr.get("OBJECT") or "").strip()
            for cand in suggestions:
                ev = cand.setdefault("evidence", {})
                if obj and obj == str(ev.get("object") or ""):
                    ev.setdefault("header_preview", {})
                    ev["header_preview"].update(
                        {
                            "FILE": fname,
                            "INSTRUME": hdr.get("INSTRUME"),
                            "DISPERSE": hdr.get("DISPERSE"),
                            "SLITWID": hdr.get("SLITWID"),
                            "BINNING": hdr.get("BINNING"),
                            "SEEING": hdr.get("SEEING"),
                            "SPERANGE": hdr.get("SPERANGE"),
                        }
                    )
    return suggestions


def build_night_bootstrap_pack(
    *,
    nightlog_path: str | Path,
    long_root: str | Path | None = None,
    header_text_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    nightlog_p = Path(nightlog_path).expanduser().resolve()
    rows = parse_bta_nightlog_table(nightlog_p)
    simple_rows = parse_nightlog(nightlog_p)
    header_texts = [parse_header_text(p) for p in (header_text_paths or [])]
    long_setups = discover_long_setups(long_root) if long_root else None
    suggestions = suggest_registry_candidates(rows, header_texts=header_texts, long_setups=long_setups)
    return {
        "schema": "scorpio_pipe.long_parity_night_bootstrap",
        "schema_version": 1,
        "nightlog_path": str(nightlog_p),
        "n_rows": int(len(rows)),
        "n_simple_rows": int(len(simple_rows)),
        "long_setup_summary": long_setups,
        "header_texts": header_texts,
        "candidate_suggestions": suggestions,
    }


def write_night_bootstrap_pack(path_or_dir: str | Path, payload: dict[str, Any]) -> dict[str, Path]:
    dest = Path(path_or_dir).expanduser().resolve()
    if dest.suffix.lower() == ".json":
        out_json = dest
        out_dir = dest.parent
    else:
        out_dir = dest
        out_json = out_dir / "night_bootstrap.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = save_json(out_json, payload)
    out_md = out_dir / "night_bootstrap.md"
    out_md.write_text(render_night_bootstrap_markdown(payload), encoding="utf-8")
    return {"json": out_json, "markdown": out_md}


def render_night_bootstrap_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LONG parity night bootstrap",
        "",
        f"- Night log: `{payload.get('nightlog_path')}`",
        f"- Parsed rows: {int(payload.get('n_rows') or 0)}",
        "",
        "## Suggested benchmark candidates",
        "",
    ]
    for cand in list(payload.get("candidate_suggestions") or []):
        lines.append(f"### {cand.get('slot')}")
        lines.append("")
        lines.append(f"- Title: {cand.get('title')}")
        lines.append(f"- Category: {cand.get('category')}")
        lines.append(f"- Confidence: {cand.get('confidence')}")
        lines.append(f"- Why: {cand.get('description')}")
        ev = cand.get("evidence") or {}
        if ev.get("object"):
            lines.append(f"- Object: {ev.get('object')}")
        if ev.get("disperser"):
            lines.append(f"- Disperser: {ev.get('disperser')}")
        if ev.get("slit") is not None:
            lines.append(f"- Slit: {ev.get('slit')}")
        counts = ev.get("counts") or {}
        if counts:
            lines.append("- Counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        files = ev.get("files") or []
        if files:
            lines.append("- Files: " + ", ".join(str(x) for x in files[:12]))
        lines.append("")
    if not list(payload.get("candidate_suggestions") or []):
        lines.append("No candidate slots were inferred automatically.")
        lines.append("")
    lines.append("## Honesty note")
    lines.append("")
    lines.append("These are bootstrap suggestions from the night log and optional header text only. They do not count as populated real benchmark cases until Scorpipe and LONG observer bundles are attached.")
    lines.append("")
    return "\n".join(lines)


def write_case_drafts_from_bootstrap(
    *,
    payload: dict[str, Any],
    registry_path: str | Path,
    cases_dir: str | Path,
) -> list[Path]:
    registry_p = Path(registry_path).expanduser().resolve()
    cases_dir_p = Path(cases_dir).expanduser().resolve()
    cases_dir_p.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cand in list(payload.get("candidate_suggestions") or []):
        slot = str(cand.get("slot") or "").strip()
        if not slot:
            continue
        manifest_p = cases_dir_p / f"{slot}.case.json"
        row = _case_payload_from_candidate(cand, slot=slot)
        save_json(manifest_p, row)
        upsert_registry_entry(
            registry_path=registry_p,
            case_id=slot,
            category=str(cand.get("category") or "reference"),
            manifest_path=manifest_p,
            status="bootstrap-only",
        )
        written.append(manifest_p)
    return written
