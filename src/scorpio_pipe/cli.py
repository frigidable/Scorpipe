from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from scorpio_pipe.autocfg import build_autoconfig
from scorpio_pipe.log import setup_logging
from scorpio_pipe.validation import validate_config
from scorpio_pipe.version import PIPELINE_VERSION, __version__


log = logging.getLogger("scorpio")
_console = Console()


def print(*objects: Any, markup: bool | None = None, **kwargs: Any) -> None:
    """CLI-safe Rich printer.

    The module historically called ``rich.print(..., markup=False)`` as if it
    were ``Console.print``. That raises ``TypeError`` at runtime because
    ``rich.print`` does not accept a ``markup`` keyword. Wrapping a dedicated
    ``Console`` here preserves existing call sites and makes literal/markup
    output explicit.
    """

    if markup is None:
        _console.print(*objects, **kwargs)
    else:
        _console.print(*objects, markup=markup, **kwargs)


def _esc(s: object) -> str:
    """Escape dynamic strings for Rich markup contexts.

    Many pipeline messages legitimately contain square brackets (e.g. frames[obj]),
    which Rich would otherwise interpret as markup tags.
    """

    if s is None:
        return ""
    return escape(str(s))


def _comma_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _print_preflight_report(run_root: Path) -> str | None:
    run_root = Path(run_root).expanduser().resolve()
    report_p = run_root / "manifest" / "preflight_report.json"
    rep = _read_json_file(report_p)
    if not rep:
        print(f"[red]Preflight report not found:[/red] {_esc(report_p)}")
        return None

    status = str(rep.get("status") or "").upper() or "UNKNOWN"
    summ = rep.get("summary") if isinstance(rep.get("summary"), dict) else {}
    print(f"\n[bold]Preflight:[/bold] {status}  (run_root={_esc(run_root)})")
    print(f"Report: {_esc(report_p)}")
    if summ:
        n_total = summ.get("n_total_fits")
        n_q = summ.get("n_quarantine")
        n_r = summ.get("n_reasons")
        print(f"FITS: {n_total} | quarantine: {n_q} | reasons: {n_r}", markup=False)

    reasons = rep.get("reasons") if isinstance(rep.get("reasons"), list) else []
    if reasons:
        t = Table(title="Preflight reasons")
        t.add_column("Severity")
        t.add_column("Code")
        t.add_column("N", justify="right")
        t.add_column("Sample paths", overflow="fold")
        for r in reasons[:50]:
            if not isinstance(r, dict):
                continue
            sev = str(r.get("severity") or "")
            code = str(r.get("code") or "")
            n = str(r.get("n") or "")
            sample_paths = r.get("sample_paths") or []
            if isinstance(sample_paths, list):
                sample = "; ".join(str(x) for x in sample_paths[:5])
            else:
                sample = str(sample_paths)
            t.add_row(sev, code, n, sample)
        print(t)

    return status


def cmd_version(_args: argparse.Namespace) -> int:
    print(
        f"[bold]scorpio-pipe[/bold] pipeline {PIPELINE_VERSION} (package {__version__})"
    )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    # Lazy import: lets users run `scorpio-pipe version/validate` in minimal
    # environments (e.g. without Astropy).
    from scorpio_pipe.inspect import inspect_dataset

    res = inspect_dataset(data_dir, max_files=args.max_files)
    df = res.table

    print(f"[bold]Nightlog:[/bold] {res.nightlog_path} (rows: {res.n_nightlog_rows})")
    print(f"[bold]FITS found/opened:[/bold] {res.n_found} / {res.n_opened}")
    if res.n_opened == 0:
        print("\n[red]No FITS files could be opened by astropy.[/red]")
        if res.open_errors:
            print("[yellow]First errors:[/yellow]")
            for s in res.open_errors[:10]:
                print(f" - {s}", markup=False)
        return 2

    print("\n[bold]Objects found (science frames):[/bold]")
    for o in res.objects:
        print(f" - {o}")

    print("\n[bold]Counts by kind:[/bold]")
    if df.empty:
        print("(empty)")
    else:
        print(df["kind"].value_counts(dropna=False))

    # setup matrix
    if not df.empty:
        print("\n[bold]Setup matrix (mode / disperser / slit / kind):[/bold]")
        cols = ["mode", "disperser", "slit", "kind"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        m = df.groupby(cols).size().sort_values(ascending=False)
        print(m.head(30))

    return 0


def _infer_night_dir(res) -> str | None:
    """Infer night dir name dd_mm_yyyy from nightlog filename or DATE-OBS."""
    try:
        p = getattr(res, "nightlog_path", None)
        if p:
            import re

            stem = Path(str(p)).stem.lower()
            m = re.search(r"s(\d{2})(\d{2})(\d{2})", stem)
            if m:
                yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                yyyy = (2000 + yy) if yy < 80 else (1900 + yy)
                return f"{dd:02d}_{mm:02d}_{yyyy:04d}"
    except Exception:
        pass

    try:
        if (
            hasattr(res, "table")
            and res.table is not None
            and (not res.table.empty)
            and ("date_obs" in res.table.columns)
        ):
            import re

            vals = res.table["date_obs"].dropna().astype(str).tolist()
            for v in vals[:200]:
                m = re.search(r"(\d{4})[-./](\d{2})[-./](\d{2})", v)
                if m:
                    yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    return f"{dd:02d}_{mm:02d}_{yyyy:04d}"
    except Exception:
        pass

    return None



def cmd_preflight(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    if args.work_dir:
        run_root = Path(args.work_dir)
    else:
        # Best-effort default: work/<dd_mm_yyyy> inferred from nightlog name.
        from scorpio_pipe.inspect import inspect_dataset

        res = inspect_dataset(data_dir, max_files=getattr(args, "max_files", None))
        night_dir = _infer_night_dir(res)
        run_root = Path("work") / (night_dir or "run1")

    # Minimal config for preflight.
    cfg: dict[str, Any] = {"data_dir": str(data_dir)}
    from scorpio_pipe.preflight import ensure_preflight

    out = ensure_preflight(
        cfg,
        run_root,
        recursive=(not bool(getattr(args, "no_recursive", False))),
        force=bool(getattr(args, "force", False)),
    )

    status = _print_preflight_report(Path(run_root))
    if (status or str(out.get("status") or "")).upper() == "ERROR":
        return 2
    return 0

def cmd_run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    from scorpio_pipe.inspect import inspect_dataset

    res = inspect_dataset(data_dir)
    df = res.table
    if df.empty:
        print("[red]Inspect produced empty table: cannot autoconfig.[/red]")
        return 2

    # Base directory: either user-provided, or inferred night folder, or work/run1 fallback.
    if args.work_dir:
        base_dir = Path(args.work_dir)
    else:
        night_dir = _infer_night_dir(res)
        base_dir = Path("work") / (night_dir or "run1")

    
    # P0-A3: preflight gate (CLI cannot bypass).
    from scorpio_pipe.preflight import ensure_preflight

    pf = ensure_preflight({"data_dir": str(data_dir)}, base_dir, recursive=True, force=False)
    pf_status = str(pf.get("status") or "").upper()
    if pf_status == "ERROR":
        _print_preflight_report(Path(base_dir))
        return 2
    if pf_status == "WARN":
        # Keep output compact; full details are in the JSON report.
        report_p = Path(base_dir).expanduser() / "manifest" / "preflight_report.json"
        print(f"[yellow]Preflight WARN[/yellow] (see {_esc(report_p)})")

    cfg_model = build_autoconfig(
        df,
        data_dir,
        args.object,
        base_dir,
        disperser=args.disperser,
        slit=args.slit,
        binning=args.binning,
    )

    # Smart run dir selection for night folders dd_mm_yyyy
    work_dir = base_dir
    try:
        import re
        from scorpio_pipe.workdir import RunSignature, pick_smart_run_dir

        setup = (
            cfg_model.frames.get("__setup__", {})
            if isinstance(cfg_model.frames, dict)
            else {}
        )
        if not isinstance(setup, dict):
            setup = {}

        if re.match(r"^\d{2}_\d{2}_\d{4}$", work_dir.name):
            sig = RunSignature(
                args.object,
                str(setup.get("disperser", "") or ""),
                str(setup.get("slit", "") or ""),
                str(setup.get("binning", "") or ""),
            )
            work_dir = pick_smart_run_dir(work_dir, sig, prefer_flat=True)
    except Exception:
        work_dir = base_dir

    # ------------------------------------------------------------------
    # P0-ARCH-03.1: single-writer guarantee.
    # Acquire a workspace lock before touching config/stage outputs.
    # ------------------------------------------------------------------
    from scorpio_pipe.adapters.workspace_fs import FSWorkspace
    from scorpio_pipe.ports.workspace import WorkspaceLockedError

    ws = FSWorkspace(work_dir)
    lock_override = bool(getattr(args, "override_lock", False))

    try:
        with ws.lock(override=lock_override):
            work_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = work_dir / "config.yaml"
            cfg_model.to_yaml(cfg_path)
            print(f"[green]Wrote autoconfig:[/green] {_esc(cfg_path)}")

            rep = validate_config(str(cfg_path), strict_paths=False)
            if rep.warnings:
                print("\n[yellow]Config warnings:[/yellow]")
                for w in rep.warnings[:20]:
                    print(f" - [{w.code}] {w.message}", markup=False)
                    if w.hint:
                        print(f"   hint: {w.hint}", markup=False)

            if not rep.ok:
                print("\n[red]Config errors:[/red]")
                for e in rep.errors:
                    print(f" - [{e.code}] {e.message}", markup=False)
                    if e.hint:
                        print(f"   hint: {e.hint}", markup=False)
                return 2

            if not args.execute:
                print(
                    "\n[bold]Next:[/bold] run UI or doit workflow, or run this command with --execute"
                )
                return 0

            tasks = _comma_list(args.tasks) or [
                "manifest",
                "superbias",
                "cosmics",
                "superneon",
                "qc_report",
                "observer_dashboard",
            ]
            # P0-ARCH-12: UI/CLI depend on StageCatalog only via pipeline public API.
            from scorpio_pipe.pipeline.public_api import STAGE_CATALOG, validate_and_canonicalize_tasks

            canon_tasks, bad = validate_and_canonicalize_tasks(tasks)

            if bad:
                avail = ', '.join(STAGE_CATALOG.task_keys())
                print(f"[red]Unknown task(s):[/red] {', '.join(bad)}", markup=False)
                print(f"[yellow]Available tasks:[/yellow] {avail}", markup=False)
                return 2

            # Run canonical tasks to avoid duplicate work when aliases were supplied.
            tasks = canon_tasks

            print(f"\n[bold]Executing tasks:[/bold] {', '.join(tasks)}")
            # Lazy import: UI runner pulls heavier optional deps.
            from scorpio_pipe.pipeline.public_api import run_sequence

            out = run_sequence(
                cfg_path,
                tasks,
                resume=bool(args.resume),
                force=bool(args.force),
                qc_override=bool(getattr(args, "override_qc_gate", False)),
            )
            print("\n[green]Done.[/green]")
            for k, v in out.items():
                print(f" - {k}: {v}", markup=False)

            # --------------------------------------------------------------
            # P0-ARCH-03.3: retention policy / cleanup (optional).
            # --------------------------------------------------------------
            retain = str(getattr(args, "retain", "all") or "all")
            if bool(getattr(args, "cleanup", False)):
                retain = "canonical"
            retain = retain.strip().lower()
            if retain not in {"all", "canonical", "none"}:
                retain = "all"

            if retain != "all":
                rep = ws.cleanup(retain)
                deleted_n = len(rep.get("deleted_files") or [])
                deleted_mb = float(rep.get("deleted_bytes") or 0) / (1024.0 * 1024.0)
                rp = rep.get("report_path")
                print(f"\n[bold]Retention:[/bold] {retain}")
                print(f"Deleted: {deleted_n} files ({deleted_mb:.2f} MiB)", markup=False)
                if rp:
                    print(f"Report: {_esc(rp)}", markup=False)

            return 0
    except WorkspaceLockedError as e:  # type: ignore
        # Rich-safe printing (no markup injection).
        print(f"[red]{_esc(str(e))}[/red]", markup=False)
        return 2


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Apply workspace retention/cleanup policy to an existing run_root."""

    from scorpio_pipe.adapters.workspace_fs import FSWorkspace
    from scorpio_pipe.ports.workspace import WorkspaceLockedError

    # Resolve run_root from --work-dir or --config.
    if getattr(args, "work_dir", None):
        run_root = Path(args.work_dir)
    else:
        from scorpio_pipe.config import load_config_any
        from scorpio_pipe.paths import resolve_work_dir

        cfg = load_config_any(Path(args.config).expanduser())
        run_root = Path(resolve_work_dir(cfg))

    ws = FSWorkspace(run_root)
    policy = str(getattr(args, "retain", "canonical") or "canonical")
    dry_run = bool(getattr(args, "dry_run", False))
    override_lock = bool(getattr(args, "override_lock", False))

    try:
        with ws.lock(override=override_lock):
            rep = ws.cleanup(policy, dry_run=dry_run)

        deleted_n = len(rep.get("deleted_files") or [])
        deleted_mb = float(rep.get("deleted_bytes") or 0) / (1024.0 * 1024.0)
        rp = rep.get("report_path")
        if dry_run:
            print(f"[yellow]DRY-RUN[/yellow] cleanup policy={_esc(policy)}")
        else:
            print(f"[green]Cleanup complete[/green] policy={_esc(policy)}")
        print(f"Deleted: {deleted_n} files ({deleted_mb:.2f} MiB)", markup=False)
        if rp:
            print(f"Report: {_esc(rp)}", markup=False)
        return 0

    except WorkspaceLockedError as e:  # type: ignore
        print(f"[red]{_esc(str(e))}[/red]", markup=False)
        return 2

def cmd_validate(args: argparse.Namespace) -> int:
    rep = validate_config(Path(args.config).expanduser(), strict_paths=args.strict)
    if rep.ok:
        print("[green]OK[/green]")
    else:
        print("[red]ERRORS[/red]")
    if rep.warnings:
        print("\n[yellow]Warnings:[/yellow]")
        for w in rep.warnings:
            # Use markup=False so bracketed codes/messages render literally.
            print(f" - [{w.code}] {w.message}", markup=False)
            if w.hint:
                print(f"   hint: {w.hint}", markup=False)
    if rep.errors:
        print("\n[red]Errors:[/red]")
        for e in rep.errors:
            print(f" - [{e.code}] {e.message}", markup=False)
            if e.hint:
                print(f"   hint: {e.hint}", markup=False)
    return 0 if rep.ok else 2


def cmd_doctor(args: argparse.Namespace) -> int:
    from scorpio_pipe.doctor import run_doctor

    print(
        f"[bold]scorpio-pipe[/bold] pipeline {PIPELINE_VERSION} (package {__version__})"
    )
    rep = run_doctor(
        config_path=getattr(args, "config", None), fix=bool(getattr(args, "fix", False))
    )

    gui = rep.get("gui", {})
    if gui.get("ok"):
        print("[green]GUI deps:[/green] OK")
    else:
        miss = ", ".join(gui.get("missing", []) or [])
        hint = gui.get("hint", "") or ""
        print(f"[yellow]GUI deps:[/yellow] missing {_esc(miss)} ({_esc(hint)})")

    for r in rep.get("resources", []) or []:
        if r.get("found"):
            print(
                f"[green]Resource:[/green] {r.get('name')} → {r.get('path')} ({r.get('source')})"
            )
        else:
            print(f"[red]Resource missing:[/red] {r.get('name')}")

    cfg = rep.get("config")
    if cfg:
        sch = cfg.get("schema") or {}
        val = cfg.get("validate") or {}
        print(f"\n[bold]Config:[/bold] {_esc(cfg.get('path'))}")
        print(
            f"Schema: {'OK' if sch.get('ok') else 'FAIL'} | Validate: {'OK' if val.get('ok') else 'FAIL'}"
        )
        if sch.get("warnings"):
            print("\n[yellow]Schema warnings:[/yellow]")
            for w in sch.get("warnings", [])[:15]:
                print(
                    f" - [{w.get('code')}] {w.get('message')}",
                    markup=False,
                )
        if sch.get("errors"):
            print("\n[red]Schema errors:[/red]")
            for e in sch.get("errors", [])[:15]:
                print(
                    f" - [{e.get('code')}] {e.get('message')}",
                    markup=False,
                )
        if val.get("warnings"):
            print("\n[yellow]Validation warnings:[/yellow]")
            for w in val.get("warnings", [])[:15]:
                print(
                    f" - [{w.get('code')}] {w.get('message')}",
                    markup=False,
                )
        if val.get("errors"):
            print("\n[red]Validation errors:[/red]")
            for e in val.get("errors", [])[:15]:
                print(
                    f" - [{e.get('code')}] {e.get('message')}",
                    markup=False,
                )

    if rep.get("fixes"):
        print("\n[bold]Autofixes:[/bold]")
        for f in rep.get("fixes", [])[:20]:
            print(f" - {f.get('action')}: {f.get('path') or ''}")

    return 0


def cmd_products(args: argparse.Namespace) -> int:
    """List expected pipeline products for a given config.yaml."""
    from scorpio_pipe.config import load_config
    from scorpio_pipe.products import list_products
    from scorpio_pipe.paths import resolve_work_dir

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    work_dir = resolve_work_dir(cfg)

    prods = list_products(cfg)

    t = Table(title=f"Pipeline products (work_dir={work_dir})")
    t.add_column("Stage")
    t.add_column("Key")
    t.add_column("Exists")
    t.add_column("Kind")
    t.add_column("Path", overflow="fold")
    t.add_column("Size")

    for p in prods:
        ex = p.exists()
        size = p.size()
        if size is None:
            ss = ""
        elif size < 1024 * 1024:
            ss = f"{size / 1024:.1f} KB"
        else:
            ss = f"{size / 1024 / 1024:.2f} MB"
        t.add_row(
            p.stage,
            p.key,
            "[green]yes[/green]"
            if ex
            else ("[red]no[/red]" if not p.optional else "[yellow]no[/yellow]"),
            p.kind,
            str(p.path),
            ss,
        )

    print(t)
    return 0



def cmd_export_package(args: argparse.Namespace) -> int:
    """Export a compact 'science package' ZIP: spec1d + qc + navigator + key PNG."""
    from scorpio_pipe.export_package import export_science_package
    from scorpio_pipe.config import load_config
    from scorpio_pipe.paths import resolve_work_dir

    work_dir: Path | None = None
    if getattr(args, "work_dir", None):
        work_dir = Path(args.work_dir).expanduser().resolve()
    elif getattr(args, "config", None):
        cfg = load_config(Path(args.config).expanduser().resolve())
        work_dir = resolve_work_dir(cfg)

    if work_dir is None:
        raise SystemExit("Need --work-dir or --config")

    out = Path(args.out).expanduser().resolve() if getattr(args, "out", None) else None
    zip_path = export_science_package(
        work_dir,
        out_zip=out,
        include_html=not bool(getattr(args, "no_html", False)),
        include_png=not bool(getattr(args, "no_png", False)),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    print(str(zip_path))
    return 0


def cmd_dataset_manifest(args: argparse.Namespace) -> int:
    """Build dataset_manifest.json for a raw-night directory."""

    from scorpio_pipe.dataset import build_dataset_manifest

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else (data_dir / "dataset_manifest.json")

    man = build_dataset_manifest(
        data_dir,
        out_path=out_path,
        recursive=not bool(getattr(args, "no_recursive", False)),
        max_files=getattr(args, "max_files", None),
        include_hashes=bool(getattr(args, "hash", False)),
        include_frame_index=not bool(getattr(args, "no_frames", False)),
        night_id=getattr(args, "night_id", None),
        pipeline_version=PIPELINE_VERSION,
    )

    # Validate on-disk contract (P0-B4)
    try:
        from scorpio_pipe.calib.manifest_schema import validate_dataset_manifest_file

        validate_dataset_manifest_file(out_path, require_v3=True)
    except Exception as e:
        print(f"[red]Manifest validation failed:[/red] {e}")
        return 2

    # Exit code: 0 OK, 1 warnings only, 2 errors.
    # Convenience: in --strict mode warnings also become a failure.
    severities = [w.severity for w in (man.warnings or [])]
    if any(s == "ERROR" for s in severities):
        print(f"[red]Wrote:[/red] {_esc(out_path)} (with ERRORs)")
        return 2
    if any(s == "WARN" for s in severities):
        if bool(getattr(args, "strict", False)):
            print(f"[red]Wrote:[/red] {_esc(out_path)} (warnings; strict mode)")
            return 2
        print(f"[yellow]Wrote:[/yellow] {_esc(out_path)} (with warnings)")
        return 1

    print(f"[green]Wrote:[/green] {_esc(out_path)}")
    return 0



def cmd_long_compare(args: argparse.Namespace) -> int:
    from scorpio_pipe.benchmark.long_compare import load_case_manifest, load_long_bundle, run_case_comparison, write_case_report

    case = load_case_manifest(Path(args.case).expanduser().resolve())
    sp_path = Path(args.scorpipe_bundle).expanduser().resolve() if getattr(args, "scorpipe_bundle", None) else None
    lg_path = Path(args.long_bundle).expanduser().resolve() if getattr(args, "long_bundle", None) else None
    if sp_path is None and case.scorpipe_bundle:
        sp_path = (case.manifest_path.parent / case.scorpipe_bundle).resolve()
    if lg_path is None and case.long_bundle:
        lg_path = (case.manifest_path.parent / case.long_bundle).resolve()
    if sp_path is None or not sp_path.exists():
        print(f"[red]Missing Scorpipe bundle:[/red] {_esc(sp_path)}")
        return 2
    if lg_path is None or not lg_path.exists():
        print(f"[red]Missing LONG bundle:[/red] {_esc(lg_path)}")
        return 2

    payload = run_case_comparison(case, scorpipe_bundle=load_long_bundle(sp_path), long_bundle=load_long_bundle(lg_path))
    out = write_case_report(Path(args.output).expanduser().resolve(), payload)
    status = str((payload.get('summary') or {}).get('status') or 'UNKNOWN')
    print(f"[bold]LONG parity[/bold] {status} for {_esc(case.case_id)}")
    print(f"JSON: {_esc(out['json'])}")
    print(f"Markdown: {_esc(out['markdown'])}")
    failed = bool((payload.get('summary') or {}).get('failed_metrics'))
    if failed and bool(getattr(args, 'fail_on_regression', False)) and not bool(getattr(args, 'soft_fail', False)):
        return 2
    return 0



def cmd_observer_dashboard(args: argparse.Namespace) -> int:
    from scorpio_pipe.config import load_config_any
    from scorpio_pipe.observer_dashboard import write_observer_dashboard

    cfg = load_config_any(args.config)
    run_root = Path(cfg.get("work_dir") or ".").expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve() if args.output else (run_root / "observer_dashboard")
    paths = write_observer_dashboard(cfg, out_dir=out_dir, run_root=run_root)
    print(f"[green]Observer dashboard written:[/green] {_esc(out_dir)}")
    for k, v in paths.items():
        print(f" - {k}: {v}", markup=False)
    return 0


def cmd_export_observer_bundle(args: argparse.Namespace) -> int:
    from scorpio_pipe.config import load_config_any
    from scorpio_pipe.benchmark.long_compare import build_scorpipe_observer_bundle, write_observer_bundle

    cfg = load_config_any(args.config)
    run_root = Path(cfg.get("work_dir") or ".").expanduser().resolve()
    bundle = build_scorpipe_observer_bundle(cfg, run_root=run_root)
    out = write_observer_bundle(Path(args.output).expanduser().resolve(), bundle)
    print(f"[green]Observer bundle written:[/green] {_esc(out)}")
    return 0


def cmd_certify_long_parity(args: argparse.Namespace) -> int:
    from scorpio_pipe.benchmark.certification import build_release_certification, write_release_certification

    reports_root = Path(args.reports_root).expanduser().resolve() if getattr(args, "reports_root", None) else (Path(args.output).expanduser().resolve() / "cases")
    payload = build_release_certification(
        registry_path=args.registry,
        reports_root=reports_root,
        golden_compare=getattr(args, "golden_compare", None),
        known_limitations_path=getattr(args, "known_limitations", None),
        observer_signoff=getattr(args, "observer_signoff", None),
    )
    paths = write_release_certification(Path(args.output).expanduser().resolve(), payload)
    s = dict(payload.get("summary") or {})
    print(f"[bold]Certification[/bold] {s.get('status') or 'UNKNOWN'}")
    print(f" - observatory_ready: {bool(s.get('observatory_ready'))}", markup=False)
    print(f" - populated_real_cases: {int(s.get('populated_real_cases', 0))}/{int(s.get('minimum_real_cases_required', 0))}", markup=False)
    print(f" - golden_ok: {bool(s.get('golden_ok'))}", markup=False)
    print(f" - observer_signoff_ok: {bool(s.get('observer_signoff_ok'))}", markup=False)
    for item in list(s.get('blocking_requirements') or []):
        print(f" - blocker: {item}", markup=False)
    for k, v in paths.items():
        print(f" - {k}: {v}", markup=False)
    if getattr(args, "fail_on_failures", False) and (not s.get("no_case_failures_ok", False) or not s.get("golden_ok", False)):
        return 2
    if getattr(args, "fail_on_not_ready", False) and not s.get("observatory_ready", False):
        return 3
    return 0


def cmd_prepare_long_case(args: argparse.Namespace) -> int:
    from scorpio_pipe.benchmark.registry_tools import build_case_manifest, export_bundle_from_config, upsert_registry_entry, write_case_manifest

    registry_path = Path(args.registry).expanduser().resolve()
    manifest_path = Path(args.output_manifest).expanduser().resolve() if getattr(args, 'output_manifest', None) else (registry_path.parent / f"{args.case_id}.case.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if getattr(args, 'config', None):
        scorpipe_bundle_path = Path(args.scorpipe_output).expanduser().resolve() if getattr(args, 'scorpipe_output', None) else (manifest_path.parent / f"{args.case_id}.scorpipe_bundle.json")
        scorpipe_bundle_path = export_bundle_from_config(args.config, scorpipe_bundle_path)
    else:
        scorpipe_bundle_path = Path(args.scorpipe_bundle).expanduser().resolve()

    long_bundle_path = Path(args.long_bundle).expanduser().resolve()
    payload = build_case_manifest(
        case_id=args.case_id,
        title=args.title,
        category=args.category,
        manifest_path=manifest_path,
        scorpipe_bundle=scorpipe_bundle_path,
        long_bundle=long_bundle_path,
        source_type=args.source_type,
        description=getattr(args, 'description', '') or '',
        observer_notes=getattr(args, 'observer_notes', '') or '',
    )
    write_case_manifest(manifest_path, payload)
    if not getattr(args, 'no_registry_update', False):
        upsert_registry_entry(
            registry_path=registry_path,
            case_id=args.case_id,
            category=args.category,
            manifest_path=manifest_path,
            status=args.status,
        )
    print(f"[green]Prepared LONG parity case:[/green] {_esc(args.case_id)}")
    print(f" - manifest: {manifest_path}", markup=False)
    print(f" - scorpipe_bundle: {scorpipe_bundle_path}", markup=False)
    print(f" - long_bundle: {long_bundle_path}", markup=False)
    if getattr(args, 'no_registry_update', False):
        print(f" - registry_update: skipped ({registry_path})", markup=False)
    else:
        print(f" - registry: {registry_path}", markup=False)
    return 0



def cmd_inspect_long_params(args: argparse.Namespace) -> int:
    from scorpio_pipe.benchmark.long_params import discover_long_setups, render_long_setup_summary_markdown

    payload = discover_long_setups(args.long_root)
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "long_setup_summary.json"
    md_path = out_dir / "long_setup_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    md_path.write_text(render_long_setup_summary_markdown(payload), encoding='utf-8')
    print(f"[green]LONG setup summary written:[/green] {_esc(out_dir)}")
    print(f" - json: {json_path}", markup=False)
    print(f" - markdown: {md_path}", markup=False)
    return 0


def cmd_bootstrap_night_cases(args: argparse.Namespace) -> int:
    from scorpio_pipe.benchmark.night_bootstrap import build_night_bootstrap_pack, write_case_drafts_from_bootstrap, write_night_bootstrap_pack

    header_texts = list(getattr(args, 'header_text', None) or [])
    payload = build_night_bootstrap_pack(
        nightlog_path=args.nightlog,
        long_root=getattr(args, 'long_root', None),
        header_text_paths=header_texts,
    )
    out_dir = Path(args.output).expanduser().resolve()
    paths = write_night_bootstrap_pack(out_dir, payload)
    print(f"[green]Night bootstrap pack written:[/green] {_esc(out_dir)}")
    for k, v in paths.items():
        print(f" - {k}: {v}", markup=False)
    if getattr(args, 'write_case_drafts', False):
        registry = getattr(args, 'registry', None) or 'benchmarks/long_parity/cases/registry.json'
        cases_dir = out_dir / 'case_drafts'
        drafts = write_case_drafts_from_bootstrap(payload=payload, registry_path=registry, cases_dir=cases_dir)
        print(f" - draft_cases: {len(drafts)}", markup=False)
        print(f" - registry: {Path(registry).expanduser().resolve()}", markup=False)
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scorpio-pipe")
    p.add_argument(
        "--log-level",
        default=None,
        help="CRITICAL|ERROR|WARNING|INFO|DEBUG (or env SCORPIO_LOG_LEVEL)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="Print version").set_defaults(func=cmd_version)

    p_ins = sub.add_parser("inspect", help="Scan data dir and list objects/frames")
    p_ins.add_argument("--data-dir", required=True)
    p_ins.add_argument(
        "--max-files", type=int, default=None, help="Limit FITS scan (debug)"
    )
    p_ins.set_defaults(func=cmd_inspect)

    p_pf = sub.add_parser("preflight", help="Run preflight scan + quarantine")
    p_pf.add_argument("--data-dir", required=True, help="Raw night directory with FITS files")
    p_pf.add_argument(
        "--work-dir",
        default=None,
        help="Run root where manifest/ and quarantine/ will be written (default: infer work/<dd_mm_yyyy> or work/run1)",
    )
    p_pf.add_argument(
        "--max-files", type=int, default=None, help="Limit nightlog inference scan (debug)"
    )
    p_pf.add_argument("--no-recursive", action="store_true", help="Do not scan subdirectories")
    p_pf.add_argument("--force", action="store_true", help="Force re-run even if JSON already exists")
    p_pf.set_defaults(func=cmd_preflight)

    p_dm = sub.add_parser(
        "dataset-manifest",
        help="Build dataset_manifest.json for a raw-night directory",
    )
    p_dm.add_argument("--data-dir", required=True, help="Raw night directory with FITS files")
    p_dm.add_argument(
        "--out",
        default=None,
        help="Output manifest path (default: <data-dir>/dataset_manifest.json)",
    )
    p_dm.add_argument(
        "--max-files", type=int, default=None, help="Limit FITS scan (debug)"
    )
    p_dm.add_argument(
        "--no-recursive", action="store_true", help="Do not scan subdirectories"
    )
    p_dm.add_argument(
        "--hash",
        action="store_true",
        help="Compute SHA256 and file sizes for frames (slower)",
    )
    p_dm.add_argument(
        "--no-frames",
        action="store_true",
        help="Do not include full frame index in manifest (smaller JSON)",
    )
    p_dm.add_argument(
        "--night-id",
        default=None,
        help="Override inferred night id (dd_mm_yyyy)",
    )
    p_dm.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if warnings are present (errors always fail)",
    )
    p_dm.set_defaults(func=cmd_dataset_manifest)

    p_val = sub.add_parser("validate", help="Validate config.yaml")
    p_val.add_argument("--config", required=True)
    p_val.add_argument(
        "--strict", action="store_true", help="Treat missing files as errors"
    )
    p_val.set_defaults(func=cmd_validate)

    p_doc = sub.add_parser("doctor", help="Environment/resource diagnostics")
    p_doc.add_argument(
        "--config", default=None, help="Optional path to config.yaml to validate"
    )
    p_doc.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe autofixes (mkdir, materialize resources, write patched config)",
    )
    p_doc.set_defaults(func=cmd_doctor)

    p_prod = sub.add_parser(
        "products", help="List expected pipeline products for a config"
    )
    p_prod.add_argument("--config", required=True)
    p_prod.set_defaults(func=cmd_products)


    p_exp = sub.add_parser(
        "export-package", help="Export a compact result ZIP: spec1d + qc + navigator"
    )
    g = p_exp.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--work-dir",
        default=None,
        help="Run work_dir (workplace/<date>/<obj>_...>)",
    )
    g.add_argument("--config", default=None, help="Config path (work_dir resolved from config)")
    p_exp.add_argument(
        "--out",
        default=None,
        help="Output zip path (default: <work_dir>/science_package.zip)",
    )
    p_exp.add_argument("--overwrite", action="store_true", help="Overwrite existing zip")
    p_exp.add_argument("--no-html", action="store_true", help="Do not include navigator HTML")
    p_exp.add_argument("--no-png", action="store_true", help="Do not include PNG quicklooks")
    p_exp.set_defaults(func=cmd_export_package)

    p_cln = sub.add_parser("cleanup", help="Apply retention policy to a run_root (collapse to publish package)")
    g = p_cln.add_mutually_exclusive_group(required=True)
    g.add_argument("--work-dir", default=None, help="Run root directory to clean up")
    g.add_argument("--config", default=None, help="Config path (run_root resolved from it)")
    p_cln.add_argument(
        "--retain",
        choices=["all", "canonical", "none"],
        default="canonical",
        help="Retention policy: all keeps everything; canonical keeps publish products + qc + provenance; none aggressively prunes intermediates.",
    )
    p_cln.add_argument("--dry-run", action="store_true", help="Do not delete; only report what would be removed")
    p_cln.add_argument(
        "--override-lock",
        action="store_true",
        help="Override existing run.lock (UNSAFE; may corrupt the workspace)",
    )
    p_cln.set_defaults(func=cmd_cleanup)

    p_cmp = sub.add_parser(
        "long-compare", help="Run a LONG parity comparison from a case manifest"
    )
    p_cmp.add_argument("--case", required=True, help="Path to a benchmark case manifest JSON")
    p_cmp.add_argument("--scorpipe-bundle", default=None, help="Override Scorpipe observer bundle JSON")
    p_cmp.add_argument("--long-bundle", default=None, help="Override LONG observer bundle JSON")
    p_cmp.add_argument("--output", required=True, help="Directory where JSON/markdown reports will be written")
    p_cmp.add_argument("--soft-fail", action="store_true", help="Always exit 0 even when some metrics fail")
    p_cmp.add_argument("--fail-on-regression", action="store_true", help="Exit with code 2 when any metric fails")
    p_cmp.set_defaults(func=cmd_long_compare)


    p_bundle = sub.add_parser("export-observer-bundle", help="Export a machine-readable Scorpipe observer bundle")
    p_bundle.add_argument("--config", required=True)
    p_bundle.add_argument("--output", required=True, help="Output JSON file or directory")
    p_bundle.set_defaults(func=cmd_export_observer_bundle)

    p_cert = sub.add_parser("certify-long-parity", help="Build a release-certification package from the LONG registry")
    p_cert.add_argument("--registry", default="benchmarks/long_parity/cases/registry.json")
    p_cert.add_argument("--reports-root", default=None, help="Optional directory for per-case reports (default: <output>/cases)")
    p_cert.add_argument("--golden-compare", default=None, help="Path to golden_time_budget_compare.json")
    p_cert.add_argument("--known-limitations", default=None, help="Optional markdown/json list of known limitations")
    p_cert.add_argument("--observer-signoff", default=None, help="Observer sign-off name/identifier")
    p_cert.add_argument("--output", required=True, help="Directory where certification files will be written")
    p_cert.add_argument("--fail-on-failures", action="store_true", help="Exit with code 2 when cases or golden evidence fail")
    p_cert.add_argument("--fail-on-not-ready", action="store_true", help="Exit with code 3 unless observatory_ready=true")
    p_cert.set_defaults(func=cmd_certify_long_parity)

    p_prep = sub.add_parser("prepare-long-case", help="Export/update one real LONG parity case manifest and upsert the registry")
    g_case = p_prep.add_mutually_exclusive_group(required=True)
    g_case.add_argument("--config", default=None, help="Config path for a completed Scorpipe run; observer bundle will be exported automatically")
    g_case.add_argument("--scorpipe-bundle", default=None, help="Existing Scorpipe observer bundle JSON")
    p_prep.add_argument("--long-bundle", required=True, help="LONG observer bundle JSON for the same case")
    p_prep.add_argument("--registry", default="benchmarks/long_parity/cases/registry.json")
    p_prep.add_argument("--case-id", required=True)
    p_prep.add_argument("--title", required=True)
    p_prep.add_argument("--category", required=True, help="reference|wavesolution|sky|extraction|geometry|calibration|repeatability")
    p_prep.add_argument("--source-type", default="real_private")
    p_prep.add_argument("--description", default="")
    p_prep.add_argument("--observer-notes", default="")
    p_prep.add_argument("--status", default="ready", help="Registry status to write, usually ready")
    p_prep.add_argument("--output-manifest", default=None, help="Override output case manifest path")
    p_prep.add_argument("--scorpipe-output", default=None, help="Override where the exported Scorpipe bundle JSON is written when --config is used")
    p_prep.add_argument("--no-registry-update", action="store_true", help="Write the case manifest but do not modify the registry")
    p_prep.set_defaults(func=cmd_prepare_long_case)

    p_lset = sub.add_parser("inspect-long-params", help="Mine setup parameters already present in the LONG tree")
    p_lset.add_argument("--long-root", required=True, help="Path to extracted LONG directory")
    p_lset.add_argument("--output", required=True, help="Output directory for JSON/markdown summaries")
    p_lset.set_defaults(func=cmd_inspect_long_params)

    p_boot = sub.add_parser("bootstrap-night-cases", help="Infer candidate LONG-parity benchmark cases from a BTA night log")
    p_boot.add_argument("--nightlog", required=True, help="Path to a BTA/SCORPIO night log txt")
    p_boot.add_argument("--long-root", default=None, help="Optional extracted LONG directory to enrich setup coverage")
    p_boot.add_argument("--header-text", action="append", default=[], help="Optional FITS-header text sidecar(s) for exact setup metadata")
    p_boot.add_argument("--output", required=True, help="Output directory for bootstrap JSON/markdown")
    p_boot.add_argument("--write-case-drafts", action="store_true", help="Also write bootstrap-only case manifests and upsert the registry")
    p_boot.add_argument("--registry", default="benchmarks/long_parity/cases/registry.json")
    p_boot.set_defaults(func=cmd_bootstrap_night_cases)

    p_dash = sub.add_parser("observer-dashboard", help="Build the observer QC cockpit / recovery dashboard")
    p_dash.add_argument("--config", required=True)
    p_dash.add_argument("--output", default=None, help="Output directory (default: <work_dir>/observer_dashboard)")
    p_dash.set_defaults(func=cmd_observer_dashboard)


    p_run = sub.add_parser(
        "run", help="Auto-config + (optional) execute pipeline tasks"
    )
    p_run.add_argument("--data-dir", required=True)
    p_run.add_argument("--object", required=True)
    p_run.add_argument(
        "--work-dir",
        default=None,
        help="Output directory. If omitted, inferred as work/<dd_mm_yyyy> from nightlog, else work/run1.",
    )
    p_run.add_argument("--disperser", default=None)
    p_run.add_argument("--slit", default=None)
    p_run.add_argument("--binning", default=None)
    p_run.add_argument(
        "--execute", action="store_true", help="Actually run non-interactive tasks"
    )
    p_run.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip tasks when their products already exist (default)",
    )
    p_run.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Always run tasks even if products exist",
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        help="Never skip tasks (stronger than --no-resume)",
    )
    p_run.add_argument(
        "--override-qc-gate",
        dest="override_qc_gate",
        action="store_true",
        help="Bypass QC gate for upstream ERROR (FATAL always blocks)",
    )
    p_run.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated tasks. Default: manifest,superbias,cosmics,superneon,qc_report,observer_dashboard",
    )
    p_run.add_argument(
        "--override-lock",
        action="store_true",
        help="Override existing run.lock (UNSAFE; may corrupt the workspace)",
    )
    p_run.add_argument(
        "--retain",
        choices=["all", "canonical", "none"],
        default="all",
        help="Retention policy after --execute: all keeps everything; canonical keeps publish products + qc + provenance; none aggressively prunes intermediates.",
    )
    p_run.add_argument(
        "--cleanup",
        action="store_true",
        help="Shortcut for --retain canonical (publish package cleanup).",
    )
    p_run.set_defaults(func=cmd_run)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    fn = getattr(args, "func", None)
    if fn is None:
        parser.print_help()
        sys.exit(2)

    rc = int(fn(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
