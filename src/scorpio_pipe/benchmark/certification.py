from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scorpio_pipe.benchmark.long_compare import (
    load_case_manifest,
    load_long_bundle,
    run_case_comparison,
    write_case_report,
)
from scorpio_pipe.calib.common import save_json


DEFAULT_REGISTRY = Path("benchmarks/long_parity/cases/registry.json")


def _counts_as_real_slot(status: str) -> bool:
    s = str(status or '').strip().lower()
    if s == 'fixture':
        return False
    if s.startswith('pending'):
        return False
    if s in {'bootstrap-only', 'draft', 'candidate', 'metadata-only'}:
        return False
    return True


def load_registry(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Registry must be a JSON object: {p}")
    data["_path"] = str(p)
    return data


def _resolve_manifest(registry_path: Path, entry: Mapping[str, Any]) -> Path | None:
    raw = entry.get("manifest")
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return (registry_path.parent / p).resolve()


def _parse_known_limitations(path: str | Path | None) -> list[str]:
    if not path:
        return []
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return [f"Known-limitations file was requested but is missing: {p}"]
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(v).strip() for v in data if str(v).strip()]
            if isinstance(data, dict):
                items = data.get("known_limitations")
                if isinstance(items, list):
                    return [str(v).strip() for v in items if str(v).strip()]
        except Exception:
            pass
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("- ", "* ")):
            s = s[2:].strip()
        if s:
            out.append(s)
    return out


def _load_golden_compare(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"status": "MISSING", "path": str(p), "regressions": ["missing-file"]}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "UNREADABLE", "path": str(p), "regressions": [str(e)]}
    out = dict(data) if isinstance(data, dict) else {"raw": data}
    out["path"] = str(p)
    return out


def build_release_certification(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    reports_root: str | Path | None = None,
    golden_compare: str | Path | None = None,
    known_limitations_path: str | Path | None = None,
    observer_signoff: str | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    registry_p = Path(str(registry["_path"]))
    entries = list(registry.get("entries") or [])
    minimum_real_cases_required = int(registry.get("minimum_real_cases_required") or 0)
    reports_root_p = Path(reports_root).expanduser().resolve() if reports_root else None
    if reports_root_p is not None:
        reports_root_p.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, Any]] = []
    pending_private = 0
    populated_real_cases = 0
    passed_real_cases = 0
    failed_real_cases = 0

    for raw_entry in entries:
        entry = dict(raw_entry or {})
        case_id = str(entry.get("case_id") or "unknown_case")
        category = str(entry.get("category") or "unknown")
        declared_status = str(entry.get("status") or "unknown")
        manifest = _resolve_manifest(registry_p, entry)
        is_real_slot = _counts_as_real_slot(declared_status)

        row: dict[str, Any] = {
            "case_id": case_id,
            "category": category,
            "registry_status": declared_status,
            "manifest_path": str(manifest) if manifest is not None else None,
            "status": declared_status,
            "summary": None,
            "report_paths": None,
        }

        if declared_status.startswith("pending"):
            pending_private += 1

        if manifest is not None and manifest.exists():
            if is_real_slot:
                populated_real_cases += 1
            case = load_case_manifest(manifest)
            sp_path = None
            lg_path = None
            if case.scorpipe_bundle:
                p = Path(case.scorpipe_bundle)
                sp_path = p if p.is_absolute() else (manifest.parent / p).resolve()
            if case.long_bundle:
                p = Path(case.long_bundle)
                lg_path = p if p.is_absolute() else (manifest.parent / p).resolve()
            if sp_path is None or not sp_path.exists() or lg_path is None or not lg_path.exists():
                row["status"] = "MISSING_BUNDLE"
                row["summary"] = {
                    "status": "MISSING_BUNDLE",
                    "scorpipe_bundle": str(sp_path) if sp_path else None,
                    "long_bundle": str(lg_path) if lg_path else None,
                }
                if is_real_slot:
                    failed_real_cases += 1
            else:
                payload = run_case_comparison(
                    case,
                    scorpipe_bundle=load_long_bundle(sp_path),
                    long_bundle=load_long_bundle(lg_path),
                )
                row["summary"] = dict(payload.get("summary") or {})
                row["status"] = str((payload.get("summary") or {}).get("status") or "UNKNOWN")
                if reports_root_p is not None:
                    row["report_paths"] = {
                        k: str(v) for k, v in write_case_report(reports_root_p / case.case_id, payload).items()
                    }
                if is_real_slot:
                    if row["status"] == "PASS":
                        passed_real_cases += 1
                    else:
                        failed_real_cases += 1
        elif is_real_slot and declared_status not in {"pending-private-data", "fixture", "bootstrap-only", "draft", "candidate", "metadata-only"}:
            failed_real_cases += 1

        case_rows.append(row)

    golden = _load_golden_compare(golden_compare)
    golden_regressions = list((golden or {}).get("regressions") or [])
    golden_missing = list((golden or {}).get("missing_in_current") or [])
    golden_ok = bool(golden) and not golden_regressions and not golden_missing and str((golden or {}).get("status") or "").upper() not in {"MISSING", "UNREADABLE", "FAIL"}

    coverage_ok = populated_real_cases >= minimum_real_cases_required
    no_case_failures_ok = failed_real_cases == 0
    signoff_ok = bool(str(observer_signoff or "").strip())
    observatory_ready = bool(coverage_ok and no_case_failures_ok and golden_ok and signoff_ok)

    blocking_requirements: list[str] = []
    if not coverage_ok:
        blocking_requirements.append(
            f"benchmark coverage {populated_real_cases}/{minimum_real_cases_required} real cases"
        )
    if not no_case_failures_ok:
        blocking_requirements.append(f"{failed_real_cases} failing or incomplete populated benchmark case(s)")
    if not golden_ok:
        blocking_requirements.append("golden weekly/manual evidence is missing or not clean")
    if not signoff_ok:
        blocking_requirements.append("observer sign-off missing")

    known_limitations = _parse_known_limitations(known_limitations_path)
    if pending_private:
        known_limitations.append(
            f"Registry still contains {pending_private} pending private/off-repo benchmark slot(s); scientific LONG-level certification is incomplete."
        )
    if not coverage_ok:
        known_limitations.append(
            f"Only {populated_real_cases} real case(s) are populated; at least {minimum_real_cases_required} are required for observatory-ready wording."
        )
    if failed_real_cases:
        known_limitations.append(f"{failed_real_cases} populated real benchmark case(s) currently fail or are missing bundles.")
    if not golden_ok:
        if golden is None:
            known_limitations.append("Golden weekly/manual time-budget evidence was not attached to this certification package.")
        else:
            known_limitations.append("Golden weekly/manual gate is not clean; inspect regressions/missing timings before calling the release observatory-ready.")
    if not signoff_ok:
        known_limitations.append("Observer sign-off is missing.")

    summary = {
        "status": "PASS" if observatory_ready else ("INCOMPLETE" if no_case_failures_ok else "FAIL"),
        "observatory_ready": observatory_ready,
        "allow_long_level_wording": observatory_ready,
        "coverage_ok": coverage_ok,
        "no_case_failures_ok": no_case_failures_ok,
        "golden_ok": golden_ok,
        "observer_signoff_ok": signoff_ok,
        "total_registry_entries": int(len(entries)),
        "minimum_real_cases_required": int(minimum_real_cases_required),
        "populated_real_cases": int(populated_real_cases),
        "passed_real_cases": int(passed_real_cases),
        "failed_real_cases": int(failed_real_cases),
        "pending_private_cases": int(pending_private),
        "blocking_requirements": list(blocking_requirements),
    }

    return {
        "schema": "scorpio_pipe.long_parity_certification",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_p),
        "summary": summary,
        "cases": case_rows,
        "golden": golden,
        "known_limitations": known_limitations,
        "observer_signoff": {"name": str(observer_signoff or "").strip() or None, "provided": signoff_ok},
        "wording_rule": (
            "LONG-level / observatory-ready wording is allowed only when observatory_ready=true. "
            "Otherwise the release must be described as partial, scaffold, or not yet certified."
        ),
    }


def render_certification_markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("summary") or {})
    golden = dict(payload.get("golden") or {}) if isinstance(payload.get("golden"), Mapping) else {}
    lines: list[str] = []
    lines.append("# LONG parity release certification")
    lines.append("")
    lines.append(f"- Status: **{summary.get('status') or 'UNKNOWN'}**")
    lines.append(f"- Observatory-ready: **{'yes' if summary.get('observatory_ready') else 'no'}**")
    lines.append(f"- LONG-level wording allowed: **{'yes' if summary.get('allow_long_level_wording') else 'no'}**")
    lines.append(f"- Populated real cases: {summary.get('populated_real_cases', 0)} / {summary.get('minimum_real_cases_required', 0)} required")
    lines.append(f"- Real cases passed: {summary.get('passed_real_cases', 0)}")
    lines.append(f"- Real cases failed/missing bundles: {summary.get('failed_real_cases', 0)}")
    lines.append(f"- Pending private slots: {summary.get('pending_private_cases', 0)}")
    lines.append(f"- Golden gate clean: {'yes' if summary.get('golden_ok') else 'no'}")
    signoff = dict(payload.get("observer_signoff") or {})
    lines.append(f"- Observer sign-off: {signoff.get('name') or 'missing'}")
    blockers = list(summary.get('blocking_requirements') or [])
    lines.append("")
    lines.append("## Blocking requirements")
    lines.append("")
    if blockers:
        for item in blockers:
            lines.append(f"- {item}")
    else:
        lines.append("- None. Certification package meets all release gates.")
    lines.append("")
    lines.append("## Case summary")
    lines.append("")
    lines.append("| Case | Category | Registry status | Result |")
    lines.append("|---|---|---|---|")
    for row in payload.get("cases") or []:
        row = dict(row or {})
        lines.append(
            f"| {row.get('case_id')} | {row.get('category') or ''} | {row.get('registry_status') or ''} | {row.get('status') or ''} |"
        )
    lines.append("")
    lines.append("## Golden weekly/manual evidence")
    lines.append("")
    if golden:
        lines.append(f"- Source: `{golden.get('path') or 'n/a'}`")
        regressions = list(golden.get("regressions") or [])
        missing = list(golden.get("missing_in_current") or [])
        lines.append(f"- Regressions: {len(regressions)}")
        lines.append(f"- Missing timings: {len(missing)}")
    else:
        lines.append("- No golden comparison JSON was attached.")
    lines.append("")
    lines.append("## Known limitations")
    lines.append("")
    for item in payload.get("known_limitations") or []:
        lines.append(f"- {item}")
    if not list(payload.get("known_limitations") or []):
        lines.append("- None declared.")
    lines.append("")
    lines.append("## Wording rule")
    lines.append("")
    lines.append(str(payload.get("wording_rule") or ""))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_certification_html(payload: Mapping[str, Any]) -> str:
    md = render_certification_markdown(payload)
    return "<html><body><pre>" + html.escape(md) + "</pre></body></html>\n"


def write_release_certification(out_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Path]:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_p = save_json(out / "long_parity_certification.json", dict(payload))
    md_p = out / "long_parity_certification.md"
    html_p = out / "long_parity_certification.html"
    md_p.write_text(render_certification_markdown(payload), encoding="utf-8")
    html_p.write_text(render_certification_html(payload), encoding="utf-8")
    return {"json": json_p, "markdown": md_p, "html": html_p}


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m scorpio_pipe.benchmark.certification")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to long-parity registry JSON")
    ap.add_argument("--reports-root", default=None, help="Optional directory where per-case reports will be written")
    ap.add_argument("--golden-compare", default=None, help="Path to golden_time_budget_compare.json")
    ap.add_argument("--known-limitations", default=None, help="Optional markdown/json list of known limitations")
    ap.add_argument("--observer-signoff", default=None, help="Observer sign-off name/identifier")
    ap.add_argument("--output", required=True, help="Directory where certification JSON/markdown/html will be written")
    ap.add_argument("--soft-fail", action="store_true", help="Always exit 0 even when the package is incomplete/failing")
    ap.add_argument("--fail-on-failures", action="store_true", help="Exit 2 when benchmark cases or golden evidence fail")
    ap.add_argument("--fail-on-not-ready", action="store_true", help="Exit 3 unless observatory_ready=true")
    return ap


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    reports_root = ns.reports_root or (Path(ns.output).expanduser().resolve() / "cases")
    payload = build_release_certification(
        registry_path=ns.registry,
        reports_root=reports_root,
        golden_compare=ns.golden_compare,
        known_limitations_path=ns.known_limitations,
        observer_signoff=ns.observer_signoff,
    )
    write_release_certification(ns.output, payload)
    summary = dict(payload.get("summary") or {})
    if ns.fail_on_failures and (not summary.get("no_case_failures_ok", False) or not summary.get("golden_ok", False)):
        return 2
    if ns.fail_on_not_ready and not summary.get("observatory_ready", False):
        return 3
    if ns.soft_fail:
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
