"""Static HTML navigator export.

P1-G / BL-P1-UI-040
--------------------
This module generates a tiny, filesystem-only "navigator" page for a run:

    <run_root>/ui/navigator/index.html
    <run_root>/ui/navigator/data.json

The navigator is designed to be portable (no backend). It uses ONLY:
 - run.json (run passport)
 - stage done markers (e.g. sky_done.json)
 - file existence checks

It does not parse FITS content, and it does not depend on pipeline internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.run_passport import read_run_passport
from scorpio_pipe.stage_registry import REGISTRY
from scorpio_pipe.workspace_paths import stage_dir


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rel(run_root: Path, p: str | Path | None) -> str | None:
    if not p:
        return None
    try:
        pp = Path(str(p))
        return str(pp.relative_to(run_root))
    except Exception:
        return str(p)


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_done(st_dir: Path, candidates: Iterable[str]) -> Path | None:
    for name in candidates:
        p = st_dir / name
        if p.is_file():
            return p
    return None


def _status_from_done(done: dict[str, Any] | None) -> str:
    if not done:
        return "NOT_RUN"
    st = (done.get("status") or "").lower()
    if st == "ok":
        return "DONE_OK"
    if st == "warn":
        return "DONE_WARN"
    if st in {"fail", "error"}:
        return "DONE_FAIL"
    # Fallback: use qc max severity if present
    sev = str(done.get("qc", {}).get("max_severity", "") or "").upper()
    if sev == "ERROR":
        return "DONE_FAIL"
    if sev == "WARN":
        return "DONE_WARN"
    if sev == "OK":
        return "DONE_OK"
    return "DONE_OK"


def _collect_products(run_root: Path, done: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return products list compatible with navigator JSON contract.

    We prefer explicit product lists from done.json (when present). If absent,
    we try to construct a minimal list from common keys in done['outputs'].
    """

    prods: list[dict[str, Any]] = []
    if not done:
        return prods

    # Preferred form: done['products'] is a list of {kind,label,path,...}
    if isinstance(done.get("products"), list):
        for it in done["products"]:
            if not isinstance(it, dict):
                continue
            p = it.get("path")
            rp = _rel(run_root, p)
            if not rp:
                continue
            if (run_root / rp).exists():
                prods.append(
                    {
                        "label": str(it.get("label") or Path(rp).name),
                        "relpath": rp,
                        "kind": str(it.get("kind") or "file"),
                    }
                )

    # Fallback: stage outputs
    out = done.get("outputs")
    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, str) and v:
                rp = _rel(run_root, v)
                if rp and (run_root / rp).exists():
                    prods.append({"label": k, "relpath": rp, "kind": "file"})

        # Per-exposure: keep it light (first 6 entries)
        per = out.get("per_exposure") if isinstance(out.get("per_exposure"), list) else []
        for row in per[:6]:
            if not isinstance(row, dict):
                continue
            stem = row.get("stem") or "exposure"
            o2 = row.get("outputs") if isinstance(row.get("outputs"), dict) else {}
            for kk, vv in o2.items():
                if not isinstance(vv, str) or not vv:
                    continue
                rp = _rel(run_root, vv)
                if rp and (run_root / rp).exists():
                    prods.append({"label": f"{stem}: {kk}", "relpath": rp, "kind": "file"})

    # Deduplicate by relpath
    seen = set()
    uniq: list[dict[str, Any]] = []
    for it in prods:
        rp = it.get("relpath")
        if not rp or rp in seen:
            continue
        seen.add(rp)
        uniq.append(it)
    return uniq


def _stage_flags(done: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not done:
        return []
    flags: list[dict[str, Any]] = []
    if isinstance(done.get("qc", {}).get("flags"), list):
        flags += [f for f in done["qc"]["flags"] if isinstance(f, dict)]
    if isinstance(done.get("flags"), list):
        flags += [f for f in done["flags"] if isinstance(f, dict)]
    return flags


def _summarize_qc(flags: list[dict[str, Any]]) -> dict[str, Any]:
    sev_rank = {"OK": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
    max_sev = "OK"
    n_warn = 0
    n_error = 0
    for f in flags:
        s = str(f.get("severity") or "OK").upper()
        if sev_rank.get(s, 0) > sev_rank.get(max_sev, 0):
            max_sev = s
        if s == "WARN":
            n_warn += 1
        elif s == "ERROR":
            n_error += 1
    top = []
    for f in flags:
        if len(top) >= 6:
            break
        top.append(
            {
                "code": f.get("code"),
                "severity": f.get("severity"),
                "stage": f.get("stage"),
                "message": f.get("message"),
                "hint": f.get("hint"),
            }
        )
    return {"max_severity": max_sev, "n_warn": n_warn, "n_error": n_error, "flags": top}


@dataclass
class _StageSpec:
    key: str
    label: str
    done_candidates: tuple[str, ...]


_STAGES: list[_StageSpec] = [
    _StageSpec("wavesol", "Wavelength Solution", ("wavesol_done.json", "wavesolution_done.json", "done.json")),
    _StageSpec("sky", "Sky Subtraction", ("sky_done.json", "done.json")),
    _StageSpec("linearize", "Linearization", ("linearize_done.json", "done.json")),
    _StageSpec("stack", "Frame Stacking", ("stack_done.json", "stack2d_done.json", "done.json")),
    _StageSpec("extract", "Object Extraction", ("extract_done.json", "extract1d_done.json", "done.json")),
]


def build_navigator(run_root: str | Path, *, overwrite: bool = True) -> Path:
    """Build/refresh the static navigator.

    Returns
    -------
    Path
        Path to ``ui/navigator/index.html``.
    """

    rr = Path(run_root).expanduser().resolve()
    nav_dir = rr / "ui" / "navigator"
    nav_dir.mkdir(parents=True, exist_ok=True)

    passport = read_run_passport(rr) or {}
    pipeline_version = str(passport.get("pipeline_version") or "")

    # QC summary
    qc_json = rr / "qc" / "qc_report.json"
    qc_doc = _read_json(qc_json) if qc_json.is_file() else None
    qc_summary = _summarize_qc(qc_doc.get("flags", []) if isinstance(qc_doc, dict) else [])
    qc_summary["report"] = _rel(rr, qc_json) if qc_json.is_file() else None

    stages: list[dict[str, Any]] = []
    for spec in _STAGES:
        # Stage dir follows registry numbering; stage_dir() resolves aliases.
        try:
            st_dir = stage_dir(rr, spec.key)
        except Exception:
            st_dir = rr / REGISTRY.dir_name(spec.key)

        done_path = _find_done(st_dir, spec.done_candidates) if st_dir.is_dir() else None
        done = _read_json(done_path) if done_path else None
        flags = _stage_flags(done)
        stage_entry: dict[str, Any] = {
            "key": spec.key,
            "label": spec.label,
            "dir": _rel(rr, st_dir) if st_dir.exists() else None,
            "done": _rel(rr, done_path) if done_path else None,
            "status": _status_from_done(done) if done else "NOT_RUN",
            "qc": _summarize_qc(flags),
            "params": done.get("params") if isinstance(done, dict) else None,
            "products": _collect_products(rr, done),
        }

        # Compare cache: show latest diff index if present
        cmp_root = rr / "ui" / "compare_cache" / spec.key
        if cmp_root.is_dir():
            try:
                stamps = sorted([p for p in cmp_root.iterdir() if p.is_dir()])
                if stamps:
                    latest = stamps[-1]
                    diff_index = latest / "diff" / "index.html"
                    if diff_index.is_file():
                        stage_entry["compare"] = {
                            "stamp": latest.name,
                            "diff_index": _rel(rr, diff_index),
                        }
            except Exception:
                pass

        stages.append(stage_entry)

    data = {
        "schema": 1,
        "generated_at": _iso_now(),
        "run": passport,
        "pipeline_version": pipeline_version,
        "qc_summary": qc_summary,
        "stages": stages,
    }

    data_path = nav_dir / "data.json"
    if overwrite or not data_path.exists():
        data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    html_path = nav_dir / "index.html"
    if overwrite or not html_path.exists():
        html_path.write_text(_NAV_HTML, encoding="utf-8")
    return html_path


_NAV_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>Scorpipe Navigator</title>
  <style>
    :root{--fg:#111;--muted:#666;--b:#e5e7eb;--card:#fff;--bg:#fafafa}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;background:var(--bg);color:var(--fg)}
    h1{margin:0 0 4px 0;font-size:20px}
    h2{margin:16px 0 8px 0;font-size:16px}
    .meta{color:var(--muted);font-size:13px;margin-bottom:12px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
    .card{border:1px solid var(--b);border-radius:14px;padding:14px;background:var(--card)}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .pill{display:inline-block;border:1px solid var(--b);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--muted)}
    .pill.ok{color:#065f46;border-color:#a7f3d0;background:#ecfdf5}
    .pill.warn{color:#92400e;border-color:#fde68a;background:#fffbeb}
    .pill.err{color:#991b1b;border-color:#fecaca;background:#fef2f2}
    ul{margin:8px 0 0 18px}
    a{color:#0b57d0;text-decoration:none}
    a:hover{text-decoration:underline}
    .muted{color:var(--muted);font-size:13px}
    .small{font-size:12px}
    code{background:#f3f4f6;padding:1px 6px;border-radius:8px}
  </style>
</head>
<body>
  <h1>Scorpipe Navigator</h1>
  <div class=\"meta\" id=\"meta\">Loading…</div>

  <div class=\"card\" id=\"runCard\" style=\"margin-bottom:14px\"></div>

  <h2>Stages</h2>
  <div class=\"grid\" id=\"stages\"></div>

  <h2>QC Summary</h2>
  <div class=\"card\" id=\"qc\"></div>

  <script>
    function pillForStatus(status){
      const s = (status||'NOT_RUN');
      if (s==='DONE_OK') return `<span class='pill ok'>DONE_OK</span>`;
      if (s==='DONE_WARN') return `<span class='pill warn'>DONE_WARN</span>`;
      if (s==='DONE_FAIL') return `<span class='pill err'>DONE_FAIL</span>`;
      return `<span class='pill'>NOT_RUN</span>`;
    }
    function pillForSev(sev){
      const s = (sev||'OK').toUpperCase();
      if (s==='OK') return `<span class='pill ok'>OK</span>`;
      if (s==='WARN') return `<span class='pill warn'>WARN</span>`;
      if (s==='ERROR') return `<span class='pill err'>ERROR</span>`;
      return `<span class='pill'>${s}</span>`;
    }
    function link(label, rel){
      if (!rel) return `<span class='muted'>${label}: not found</span>`;
      return `<a href='../../${rel}'>${label}</a>`;
    }
    async function main(){
      const r = await fetch('data.json', {cache:'no-store'});
      const d = await r.json();
      const run = d.run || {};
      const meta = document.getElementById('meta');
      const night = run.night_date || '';
      const obj = run.object || run.object_key || '';
      const disp = run.disperser || run.disperser_key || '';
      const id = (run.run_id !== undefined ? run.run_id : run.run_id_str) || '';
      const created = run.created_at || '';
      const pv = d.pipeline_version || run.pipeline_version || '';
      meta.innerHTML = `Night <code>${night}</code> · Object <code>${obj}</code> · Disperser <code>${disp}</code> · Run <code>${id}</code>`;

      const runCard = document.getElementById('runCard');
      runCard.innerHTML = `
        <div class='row'>
          <span class='pill'>pipeline ${pv}</span>
          <span class='pill'>created ${created}</span>
          <span class='pill'>generated ${d.generated_at||''}</span>
        </div>
        <div style='margin-top:8px' class='row small'>
          ${link('run.json', run.run_json_path || 'run.json')}
          ${link('config.yaml', 'config.yaml')}
          ${link('ui/session.json', 'ui/session.json')}
          ${link('ui/history/', 'ui/history/')}
        </div>
      `;

      const stages = document.getElementById('stages');
      for (const st of (d.stages||[])){
        const div = document.createElement('div');
        div.className = 'card';
        const q = st.qc || {};
        const prods = (st.products||[]).slice(0, 10);
        const prodList = prods.length ? ('<ul>' + prods.map(p => `<li>${link(p.label, p.relpath)}</li>`).join('') + '</ul>') : `<div class='muted'>No products referenced in done.json</div>`;
        const compare = st.compare && st.compare.diff_index ? `<div style='margin-top:8px'>${link('Compare A/B (diff)', st.compare.diff_index)} <span class='muted small'>(stamp ${st.compare.stamp})</span></div>` : '';
        div.innerHTML = `
          <div class='row'>
            <div style='flex:1'><strong>${st.label}</strong><div class='muted small'>${st.dir ? st.dir : ''}</div></div>
            ${pillForStatus(st.status)}
            ${pillForSev(q.max_severity)}
          </div>
          <div style='margin-top:8px' class='row small'>
            ${link('done.json', st.done)}
          </div>
          <div style='margin-top:8px'>${prodList}</div>
          ${compare}
        `;
        stages.appendChild(div);
      }

      const qc = document.getElementById('qc');
      const qs = d.qc_summary || {};
      const flags = qs.flags || [];
      const fl = flags.length ? ('<ul>' + flags.map(f => {
        const s = (f.severity||'').toUpperCase();
        const cls = s==='ERROR' ? 'err' : (s==='WARN' ? 'warn' : 'ok');
        const hint = f.hint ? ` <span class='muted small'>— ${f.hint}</span>` : '';
        return `<li><span class='pill ${cls}'>${s}</span> <code>${f.code||''}</code> <span class='muted small'>(${f.stage||''})</span> — ${f.message||''}${hint}</li>`;
      }).join('') + '</ul>') : `<div class='muted'>No QC flags.</div>`;
      qc.innerHTML = `
        <div class='row'>
          ${pillForSev(qs.max_severity)}
          <span class='pill'>WARN ${qs.n_warn||0}</span>
          <span class='pill'>ERROR ${qs.n_error||0}</span>
          <span style='flex:1'></span>
          ${link('qc_report.json', qs.report)}
        </div>
        <div style='margin-top:8px'>${fl}</div>
      `;
    }
    main().catch(err => {
      document.getElementById('meta').textContent = 'Failed to load data.json: ' + err;
    });
  </script>
</body>
</html>
"""
