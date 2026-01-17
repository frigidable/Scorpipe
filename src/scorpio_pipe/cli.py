from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich import print
from rich.markup import escape
from rich.table import Table

from scorpio_pipe.autocfg import build_autoconfig
from scorpio_pipe.log import setup_logging
from scorpio_pipe.validation import validate_config
from scorpio_pipe.version import PIPELINE_VERSION, __version__


log = logging.getLogger("scorpio")


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
    ]
    print(f"\n[bold]Executing tasks:[/bold] {', '.join(tasks)}")
    # Lazy import: UI runner pulls heavier optional deps.
    from scorpio_pipe.ui.pipeline_runner import run_sequence

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
    return 0


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
        help="Comma-separated tasks. Default: manifest,superbias,cosmics,superneon,qc_report",
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
