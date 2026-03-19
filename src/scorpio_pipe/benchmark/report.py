from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scorpio_pipe.calib.common import save_json


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    case = dict(payload.get("case") or {})
    summary = dict(payload.get("summary") or {})
    metrics = dict(payload.get("metrics") or {})

    lines: list[str] = []
    lines.append(f"# LONG parity report — {case.get('title') or case.get('case_id') or 'case'}")
    lines.append("")
    lines.append(f"- Case ID: `{case.get('case_id')}`")
    lines.append(f"- Category: `{case.get('category') or 'unknown'}`")
    lines.append(f"- Source type: `{case.get('source_type') or 'unknown'}`")
    lines.append(f"- Status: **{summary.get('status') or 'UNKNOWN'}**")
    lines.append(f"- Metrics passed: {summary.get('passed_metrics', 0)} / {summary.get('total_metrics', 0)}")
    if case.get("description"):
        lines.append(f"- Description: {case.get('description')}")
    if case.get("observer_notes"):
        lines.append(f"- Observer notes: {case.get('observer_notes')}")
    lines.append("")
    lines.append("| Metric | Current | LONG | Status | Rule | Notes |")
    lines.append("|---|---:|---:|---|---|---|")
    for name, row in metrics.items():
        d = dict(row or {})
        lines.append(
            f"| {d.get('label') or name} | {_fmt_value(d.get('current'))} | {_fmt_value(d.get('reference'))} | {d.get('status')} | {d.get('prefer')} | {d.get('reason') or ''} |"
        )
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_report_files(out_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_json(out_dir / "long_parity_report.json", dict(payload))
    md_path = out_dir / "long_parity_report.md"
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
