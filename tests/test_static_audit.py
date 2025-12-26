from __future__ import annotations

from pathlib import Path


def _project_root() -> Path:
    # tests/ -> project root
    return Path(__file__).resolve().parents[1]


def test_resolve_work_dir_is_imported_where_used() -> None:
    """Prevent regressions of `NameError: resolve_work_dir is not defined`.

    We do a cheap static audit: if a module calls `resolve_work_dir(` it must
    either import it from `scorpio_pipe.paths` (preferred) / `scorpio_pipe.wavesol_paths`
    (legacy), or use a qualified reference like `paths.resolve_work_dir(...)`.
    """
    root = _project_root() / "src" / "scorpio_pipe"

    offenders: list[str] = []

    for p in root.rglob("*.py"):
        if p.name == "paths.py":
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "resolve_work_dir" not in txt:
            continue
        if "resolve_work_dir(" not in txt and "resolve_work_dir (" not in txt:
            continue

        ok = False
        if "from scorpio_pipe.paths import resolve_work_dir" in txt:
            ok = True
        if "from scorpio_pipe.wavesol_paths import resolve_work_dir" in txt:
            ok = True
        if "paths.resolve_work_dir(" in txt or "paths.resolve_work_dir (" in txt:
            ok = True
        if "scorpio_pipe.paths.resolve_work_dir(" in txt:
            ok = True

        if not ok:
            offenders.append(str(p.relative_to(_project_root())))

    assert not offenders, "Missing resolve_work_dir import/qualification in:\n" + "\n".join(offenders)


def test_qc_report_does_not_double_prefix_v() -> None:
    """QC HTML should not print 'vv5.xx' in titles/badges."""
    p = _project_root() / "src" / "scorpio_pipe" / "qc_report.py"
    txt = p.read_text(encoding="utf-8", errors="ignore")

    assert "QC — v{_html_escape(PIPELINE_VERSION)}" not in txt
    assert "pill'>v{_html_escape(PIPELINE_VERSION)}" not in txt
