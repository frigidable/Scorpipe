from __future__ import annotations

import argparse
from pathlib import Path

import logging

from rich import print

from scorpio_pipe.inspect import inspect_dataset
from scorpio_pipe.autocfg import build_autoconfig
from scorpio_pipe.log import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(prog="scorpio-pipe")
    p.add_argument(
        "--log-level",
        default=None,
        help="CRITICAL|ERROR|WARNING|INFO|DEBUG (or env SCORPIO_LOG_LEVEL)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="Scan data dir and list objects/frames")
    p_ins.add_argument("--data-dir", required=True)
    p_ins.add_argument("--max-files", type=int, default=None, help="Limit FITS scan (debug)")

    p_run = sub.add_parser("run", help="Auto-config + run DAG")
    p_run.add_argument("--data-dir", required=True)
    p_run.add_argument("--object", required=True)
    p_run.add_argument("--work-dir", default="work/run1")

    args = p.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger("scorpio")

    data_dir = Path(args.data_dir)
    res = inspect_dataset(data_dir, max_files=getattr(args, "max_files", None))
    df = res.table

    if args.cmd == "inspect":
        print(f"[bold]Nightlog:[/bold] {res.nightlog_path} (rows: {res.n_nightlog_rows})")
        print(f"[bold]FITS found/opened:[/bold] {res.n_found} / {res.n_opened}")
        if res.n_opened == 0:
            print("\n[red]No FITS files could be opened by astropy.[/red]")
            if res.open_errors:
                print("[yellow]First errors:[/yellow]")
                for s in res.open_errors:
                    print(f" - {s}")
            return

        log.info("Inspect finished: opened=%d / found=%d", res.n_opened, res.n_found)

        print("\n[bold]Objects found (science frames):[/bold]")
        for o in res.objects:
            print(f" - {o}")

        print("\n[bold]Counts by kind:[/bold]")
        if res.table.empty:
            print("(empty)")
        else:
            print(df["kind"].value_counts(dropna=False))
        return

    # setup-матрица: что есть в ночи по режиму/решётке/щели
    if not df.empty:
        print("\n[bold]Setup matrix (mode / disperser / slit / kind):[/bold]")
        cols = ["mode", "disperser", "slit", "kind"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        m = df.groupby(cols).size().sort_values(ascending=False)
        # печатаем первые 30 строк, чтобы не захламлять консоль
        print(m.head(30))


    if args.cmd == "run":
        work_dir = Path(args.work_dir)
        cfg = build_autoconfig(df, data_dir, args.object, work_dir)
        cfg_path = work_dir / "config.yaml"
        cfg.to_yaml(cfg_path)
        print(f"[green]Wrote autoconfig:[/green] {cfg_path}")

        print(
            "[yellow]Next:[/yellow] add workflow/dodo.py and stage wrappers, then run doit."
        )
        return

if __name__ == "__main__":
    main()
